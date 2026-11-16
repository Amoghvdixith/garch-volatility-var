"""Forecast-evaluation metrics, regime classification and summary tables.

Conventions: ``loss = -return`` (percentage points); a VaR exception is the
strict event ``loss > VaR``.

Quantile (pinball / tick) loss for the ``c``-quantile of the loss::

    QL = (c - 1[loss < VaR]) * (loss - VaR)
       = c * (loss - VaR)          if loss >= VaR
       = (1 - c) * (VaR - loss)    if loss <  VaR

It is non-negative, zero only when ``loss == VaR``, and its expectation is
minimised by the true conditional ``c``-quantile of the loss.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SHORT_SAMPLE_THRESHOLD = 250  # observations; regime samples below this are flagged


def quantile_loss(loss, var, confidence: float):
    """Pinball loss of a loss-quantile forecast (scalar or array)."""
    u = np.asarray(loss, dtype=float) - np.asarray(var, dtype=float)
    out = u * (confidence - (u < 0.0))
    return float(out) if out.ndim == 0 else out


def max_consecutive(exceptions: Iterable[int]) -> int:
    """Longest run of consecutive 1s in an exception indicator sequence."""
    best = run = 0
    for e in exceptions:
        run = run + 1 if int(e) == 1 else 0
        best = max(best, run)
    return best


def rolling_exception_rate(exceptions: pd.Series, window: int) -> pd.Series:
    """Trailing exception rate: value at date ``t`` averages ``t-window+1 .. t``.

    The first ``window - 1`` values are NaN (no partial windows).
    """
    return exceptions.astype(float).rolling(window, min_periods=window).mean()


def classify_regimes(dates: pd.DatetimeIndex, regimes: list[dict]) -> np.ndarray:
    """Label each date with the regime whose inclusive [start_date, end_date] contains it.

    Dates outside every regime are labelled ``'normal'``. Overlapping regimes
    are rejected.
    """
    dates = pd.DatetimeIndex(dates)
    labels = np.full(dates.size, "normal", dtype=object)
    for r in regimes:
        start, end = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
        if end < start:
            raise ValueError(f"regime {r['label']} ends before it starts")
        mask = (dates >= start) & (dates <= end)
        if np.any(labels[mask] != "normal"):
            raise ValueError(f"regime {r['label']} overlaps another regime")
        labels[mask] = r["label"]
    return labels


def volatility_rmse(forecast_vol, true_vol) -> float:
    """Root mean squared error in volatility units (percentage points)."""
    f, t = np.asarray(forecast_vol, float), np.asarray(true_vol, float)
    return float(np.sqrt(np.mean((f - t) ** 2)))


def qlike(forecast_var, true_var) -> float:
    """Normalised QLIKE variance loss ``mean(h/f - log(h/f) - 1)``, ``h`` = true variance.

    Equals zero for a perfect forecast and is always non-negative.
    """
    f, h = np.asarray(forecast_var, float), np.asarray(true_var, float)
    ratio = h / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def summarize_group(g: pd.DataFrame, confidence: float) -> dict:
    """Coverage and tail-severity statistics for one model/confidence sample."""
    n = int(len(g))
    exc = g["exception"].astype(int).to_numpy()
    x = int(exc.sum())
    p = 1.0 - confidence
    on_exc = g.loc[exc == 1]
    return {
        "n": n,
        "exceptions": x,
        "expected_exceptions": n * p,
        "exception_rate": x / n if n else np.nan,
        "nominal_rate": p,
        "rate_difference": (x / n - p) if n else np.nan,
        "mean_var_pct": float(g["var_pct"].mean()),
        "mean_es_pct": float(g["es_pct"].mean()),
        "mean_loss_on_exceptions_pct": float(on_exc["loss_pct"].mean()) if x else np.nan,
        "mean_es_on_exceptions_pct": float(on_exc["es_pct"].mean()) if x else np.nan,
        "mean_excess_loss_on_exceptions_pct": float((on_exc["loss_pct"] - on_exc["var_pct"]).mean()) if x else np.nan,
        "mean_quantile_loss": float(g["quantile_loss"].mean()),
        "max_consecutive_exceptions": max_consecutive(exc),
        "es_below_var_count": int((g["es_pct"] < g["var_pct"] - 1e-12).sum()),
    }


def summary_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summary by model and confidence (pass the common-sample frame)."""
    rows = []
    for (model, c), g in forecasts.groupby(["model", "confidence"], sort=False):
        rows.append({"model": model, "confidence": c, **summarize_group(g.sort_values("forecast_date"), c)})
    return pd.DataFrame(rows)


def stress_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summary by model, confidence and regime, plus an 'all_stress' pool.

    Runs inside a regime are evaluated on that regime's dates only, so
    ``max_consecutive_exceptions`` never spans two regimes.
    """
    from .backtesting import kupiec_test

    frames = [forecasts]
    stressed = forecasts[forecasts["regime"] != "normal"]
    if not stressed.empty and stressed["regime"].nunique() > 1:
        frames.append(stressed.assign(regime="all_stress"))
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (model, c, regime), g in data.groupby(["model", "confidence", "regime"], sort=False):
        g = g.sort_values("forecast_date")
        s = summarize_group(g, c)
        k = kupiec_test(g["exception"].astype(int).to_numpy(), 1.0 - c)
        notes = []
        if s["n"] < SHORT_SAMPLE_THRESHOLD:
            notes.append(f"short sample (<{SHORT_SAMPLE_THRESHOLD} obs)")
        if s["expected_exceptions"] < 5:
            notes.append("expected exceptions < 5: coverage test has very low power")
        if regime == "all_stress":
            notes.append("pooled non-contiguous stress regimes")
        rows.append(
            {
                "model": model,
                "confidence": c,
                "regime": regime,
                "start_date": g["forecast_date"].min(),
                "end_date": g["forecast_date"].max(),
                **s,
                "kupiec_statistic": k["statistic"],
                "kupiec_p_value": k["p_value"],
                "short_sample": s["n"] < SHORT_SAMPLE_THRESHOLD,
                "notes": "; ".join(notes),
            }
        )
    order = {"normal": 0, "all_stress": 99}
    out = pd.DataFrame(rows)
    out["_o"] = out["regime"].map(lambda r: order.get(r, 1))
    return out.sort_values(["confidence", "_o", "regime", "model"], kind="stable").drop(columns="_o").reset_index(drop=True)


def volatility_accuracy_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Forecast-vs-true volatility accuracy by model and regime (synthetic data only)."""
    vol = forecasts.dropna(subset=["vol_forecast_pct", "true_vol_pct"])
    vol = vol.drop_duplicates(["forecast_date", "model"])
    rows = []
    for regime_label, sub in [("all", vol)] + list(vol.groupby("regime")):
        for model, g in sub.groupby("model", sort=False):
            rows.append(
                {
                    "model": model,
                    "regime": regime_label,
                    "n": int(len(g)),
                    "rmse_vol_pct": volatility_rmse(g["vol_forecast_pct"], g["true_vol_pct"]),
                    "mae_vol_pct": float(np.mean(np.abs(g["vol_forecast_pct"] - g["true_vol_pct"]))),
                    "mean_forecast_vol_pct": float(g["vol_forecast_pct"].mean()),
                    "mean_true_vol_pct": float(g["true_vol_pct"].mean()),
                    "qlike_true_variance": qlike(g["vol_forecast_pct"] ** 2, g["true_vol_pct"] ** 2),
                }
            )
    return pd.DataFrame(rows)
