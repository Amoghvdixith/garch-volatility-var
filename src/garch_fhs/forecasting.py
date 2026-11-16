"""Strictly out-of-sample rolling one-day-ahead VaR/ES forecasts.

Timing for forecast index ``t`` (zero-based position in the return series):

* training window = positions ``t-window .. t-1`` (``window`` observations);
* every model is refitted on exactly that window at every origin;
* ``sigma_t`` and all scenarios use the training window only;
* the realised ``return_t`` is attached *after* the forecast is produced and
  is used only for evaluation.

:func:`forecast_from_window` receives nothing but the training returns, so
the forecast for ``t`` cannot depend on ``return_t`` or later data by
construction; the leakage tests in ``tests/test_forecasting.py`` verify this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import MODEL_GARCH, MODEL_GJR, MODEL_HS
from .filtered_historical import fhs_var_es, standardized_residuals
from .historical import historical_simulation
from .metrics import quantile_loss
from .volatility import FitResult, fit_volatility

VOL_MODEL_FOR = {MODEL_GARCH: "GARCH", MODEL_GJR: "GJR"}
OK_STATUSES = ("ok", "ok_retry")
# Persistence within this distance of 1 is reported as "at the stationarity
# boundary": arch constrains persistence <= 1, so estimates pile up there
# (near-integrated, IGARCH-like fits) rather than exceeding it materially.
BOUNDARY_TOL = 1e-4

FitFn = Callable[[np.ndarray, str], FitResult]


@dataclass
class OriginForecast:
    """Forecast of one model at one origin, for all confidence levels."""

    model: str
    status: str
    sigma: float = float("nan")
    var: dict[float, float] = field(default_factory=dict)
    es: dict[float, float] = field(default_factory=dict)
    n_scenarios: int = 0
    fit: FitResult | None = None
    z: np.ndarray | None = None
    n_dropped: int = 0
    error: str = ""


def forecast_from_window(
    train_returns: np.ndarray,
    models: Iterable[str],
    confidences: Iterable[float],
    min_residuals: int,
    fit_fn: FitFn = fit_volatility,
) -> dict[str, OriginForecast]:
    """Produce HS / FHS forecasts from a training window only."""
    train = np.asarray(train_returns, dtype=float).copy()
    confidences = list(confidences)
    out: dict[str, OriginForecast] = {}
    for model in models:
        if model == MODEL_HS:
            fc = OriginForecast(model=model, status="ok", n_scenarios=train.size)
            for c in confidences:
                est = historical_simulation(train, c)
                fc.var[c], fc.es[c] = est.var, est.es
            out[model] = fc
            continue
        if model not in VOL_MODEL_FOR:
            raise ValueError(f"unknown model {model!r}")
        fit = fit_fn(train, VOL_MODEL_FOR[model])
        fc = OriginForecast(model=model, status="fit_failed", fit=fit, error=fit.error)
        if fit.ok:
            sr = standardized_residuals(fit.resid, fit.cond_vol)
            fc.z, fc.n_dropped, fc.sigma = sr.z, sr.n_dropped, fit.sigma_forecast
            if sr.z.size < min_residuals:
                fc.status = "insufficient_residuals"
                fc.error = f"{sr.z.size} valid residuals < minimum {min_residuals}"
            else:
                fc.status = fit.status
                fc.n_scenarios = int(sr.z.size)
                for c in confidences:
                    est = fhs_var_es(fit.sigma_forecast, sr.z, c, min_residuals)
                    fc.var[c], fc.es[c] = est.var, est.es
        out[model] = fc
    return out


@dataclass
class RollingOutput:
    """Rolling forecast results."""

    forecasts: pd.DataFrame
    fits: pd.DataFrame
    residual_windows: list[dict]
    n_origins: int


def _fit_record(date, model: str, fc: OriginForecast, train_obs: int) -> dict:
    fit = fc.fit
    p = fit.params if fit else {}
    pers = fit.persistence if fit and p else float("nan")
    return {
        "forecast_date": date,
        "model": model,
        "status": fc.status,
        "attempts": fit.attempts if fit else 0,
        "converged": bool(fit.converged) if fit else False,
        "omega": p.get("omega", np.nan),
        "alpha": p.get("alpha", np.nan),
        "gamma": p.get("gamma", np.nan),
        "beta": p.get("beta", np.nan),
        "persistence": pers,
        "persistence_below_one": bool(pers < 1.0) if np.isfinite(pers) else False,
        "at_stationarity_boundary": bool(pers >= 1.0 - BOUNDARY_TOL) if np.isfinite(pers) else False,
        "loglik": fit.loglik if fit else np.nan,
        "sigma_forecast_pct": fc.sigma,
        "train_obs": train_obs,
        "n_residuals_valid": int(fc.z.size) if fc.z is not None else 0,
        "n_residuals_dropped": fc.n_dropped,
        "warnings": " | ".join(fit.warnings) if fit else "",
        "error": fc.error,
    }


def run_rolling_forecasts(
    data: pd.DataFrame,
    window: int,
    confidences: Iterable[float],
    models: Iterable[str],
    min_residuals: int,
    max_forecasts: int | None = None,
    fit_fn: FitFn = fit_volatility,
    progress: Callable[[int, int], None] | None = None,
) -> RollingOutput:
    """Run the rolling one-day-ahead forecast loop over ``data``.

    ``data`` is date-indexed with ``return_pct`` and optionally ``regime``,
    ``stress_flag`` and ``true_vol_pct`` (synthetic only). The first forecast
    date is position ``window``. ``max_forecasts`` truncates the loop (used
    only for labelled smoke tests).
    """
    confidences = list(confidences)
    models = list(models)
    returns = data["return_pct"].to_numpy(dtype=float)
    dates = data.index
    n = returns.size
    if n <= window:
        raise ValueError(f"need more than window={window} observations, got {n}")
    last = n if max_forecasts is None else min(n, window + int(max_forecasts))
    regime = data["regime"].to_numpy() if "regime" in data else np.full(n, "normal", dtype=object)
    stress = data["stress_flag"].to_numpy() if "stress_flag" in data else (regime != "normal").astype(int)
    true_vol = data["true_vol_pct"].to_numpy(dtype=float) if "true_vol_pct" in data else np.full(n, np.nan)

    rows: list[dict] = []
    fit_rows: list[dict] = []
    first_z: dict[str, dict] = {}
    last_z: dict[str, dict] = {}
    total = last - window
    for k, t in enumerate(range(window, last), start=1):
        train = returns[t - window : t]
        fcs = forecast_from_window(train, models, confidences, min_residuals, fit_fn)
        date, train_start, train_end = dates[t], dates[t - window], dates[t - 1]
        r_t = returns[t]  # realised value, attached only for evaluation
        loss_t = -r_t
        for model, fc in fcs.items():
            if model != MODEL_HS:
                fit_rows.append(_fit_record(date, model, fc, window))
                if fc.status in OK_STATUSES:
                    rec = {"model": model, "train_start": train_start, "train_end": train_end, "z": fc.z}
                    first_z.setdefault(model, rec)
                    last_z[model] = rec
            ok = fc.status in OK_STATUSES
            for c in confidences:
                var = fc.var.get(c, np.nan)
                es = fc.es.get(c, np.nan)
                rows.append(
                    {
                        "forecast_date": date,
                        "train_start": train_start,
                        "train_end": train_end,
                        "model": model,
                        "confidence": c,
                        "return_pct": r_t,
                        "loss_pct": loss_t,
                        "vol_forecast_pct": fc.sigma if model != MODEL_HS else np.nan,
                        "true_vol_pct": true_vol[t],
                        "var_pct": var,
                        "es_pct": es,
                        "exception": int(loss_t > var) if ok else np.nan,
                        "quantile_loss": quantile_loss(loss_t, var, c) if ok else np.nan,
                        "stress_flag": int(stress[t]),
                        "regime": regime[t],
                        "status": fc.status,
                        "n_scenarios": fc.n_scenarios,
                    }
                )
        if progress is not None:
            progress(k, total)

    windows = []
    for label, store in (("first", first_z), ("last", last_z)):
        for model, rec in store.items():
            windows.append({"window": label, **rec})
    return RollingOutput(
        forecasts=pd.DataFrame(rows),
        fits=pd.DataFrame(fit_rows),
        residual_windows=windows,
        n_origins=total,
    )


def common_sample(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Restrict to dates where every model has a valid forecast.

    Returns the filtered frame and a summary of excluded dates and per-model
    failure counts. Model-specific diagnostics are kept elsewhere unchanged.
    """
    valid = forecasts["status"].isin(OK_STATUSES)
    by_date = valid.groupby(forecasts["forecast_date"]).all()
    keep_dates = by_date[by_date].index
    excluded = by_date[~by_date].index
    failed = forecasts.loc[~valid].drop_duplicates(["forecast_date", "model"])
    info = {
        "n_dates_total": int(by_date.size),
        "n_dates_common": int(keep_dates.size),
        "n_dates_excluded": int(excluded.size),
        "excluded_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in excluded],
        "invalid_forecasts_by_model": {str(k): int(v) for k, v in failed.groupby("model").size().items()},
        "invalid_forecasts_by_status": {str(k): int(v) for k, v in failed.groupby("status").size().items()},
    }
    return forecasts[forecasts["forecast_date"].isin(keep_dates)].copy(), info
