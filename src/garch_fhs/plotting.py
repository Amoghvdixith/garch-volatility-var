"""Matplotlib charts built from saved outputs (headless: Agg backend).

Every function saves one PNG and closes its figure. Models keep the same
colour in every chart. Losses (positive = loss) and returns are labelled
explicitly on the axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnostics import autocorrelations  # noqa: E402

MODEL_COLORS = {"HS": "#6b6b6b", "GARCH-FHS": "#1f77b4", "GJR-FHS": "#2ca02c"}
TRUE_COLOR = "#000000"
EXCEPTION_COLOR = "#d62728"
STRESS_SHADE = "#f4a582"
DPI = 130


def _tag(synthetic: bool) -> str:
    return "SYNTHETIC DATA - " if synthetic else ""


def regime_spans(dates: pd.Series | pd.DatetimeIndex, regimes: pd.Series | np.ndarray) -> list[tuple]:
    """Contiguous (label, start, end) spans for every non-'normal' regime label."""
    df = pd.DataFrame({"date": pd.to_datetime(np.asarray(dates)), "regime": np.asarray(regimes)})
    df = df.sort_values("date")
    spans = []
    block = (df["regime"] != df["regime"].shift()).cumsum()
    for _, g in df.groupby(block):
        label = g["regime"].iloc[0]
        if label != "normal":
            spans.append((label, g["date"].iloc[0], g["date"].iloc[-1]))
    return spans


def _shade(ax, spans, label_once: bool = True) -> None:
    for i, (lab, start, end) in enumerate(spans):
        ax.axvspan(start, end, color=STRESS_SHADE, alpha=0.35, lw=0,
                   label="predefined stress regime" if (label_once and i == 0) else None)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_returns_and_volatility(data: pd.DataFrame, spans, path: Path, synthetic: bool, first_forecast=None) -> None:
    """Daily returns and (if known) true conditional volatility, with stress shading."""
    has_true = "true_vol_pct" in data and data["true_vol_pct"].notna().any()
    fig, axes = plt.subplots(2 if has_true else 1, 1, figsize=(12, 6.5 if has_true else 3.8), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(data.index, data["return_pct"], lw=0.5, color="#333333", label="daily log return")
    _shade(ax, spans)
    ax.set_ylabel("Return (% log, pp)")
    ax.axhline(0, color="grey", lw=0.5)
    if first_forecast is not None:
        ax.axvline(first_forecast, color="purple", ls="--", lw=1, label="first out-of-sample forecast date")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{_tag(synthetic)}Daily returns" + (" and true conditional volatility" if has_true else ""))
    if has_true:
        ax2 = axes[1, 0]
        ax2.plot(data.index, data["true_vol_pct"], lw=0.8, color=TRUE_COLOR, label="true conditional volatility sigma_t")
        _shade(ax2, spans)
        if first_forecast is not None:
            ax2.axvline(first_forecast, color="purple", ls="--", lw=1)
        ax2.set_ylabel("Volatility (pp)")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("Date (synthetic business-day index)" if synthetic else "Date")
    _save(fig, path)


def plot_volatility_forecasts(fc: pd.DataFrame, spans, path: Path, synthetic: bool) -> None:
    """One-step volatility forecasts of the FHS filters versus true volatility if known."""
    v = fc.drop_duplicates(["forecast_date", "model"]).dropna(subset=["vol_forecast_pct"])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for model, g in v.groupby("model", sort=False):
        ax.plot(g["forecast_date"], g["vol_forecast_pct"], lw=0.8, color=MODEL_COLORS[model], label=f"{model} forecast sigma_t")
    truth = v.drop_duplicates("forecast_date")
    if truth["true_vol_pct"].notna().any():
        ax.plot(truth["forecast_date"], truth["true_vol_pct"], lw=0.8, color=TRUE_COLOR, ls="--", label="true sigma_t")
    _shade(ax, spans)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Volatility (pp)")
    ax.set_xlabel("Forecast date")
    ax.set_title(f"{_tag(synthetic)}One-day-ahead volatility forecasts (out-of-sample)")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, path)


def plot_var_comparison(fc: pd.DataFrame, confidence: float, spans, path: Path, synthetic: bool) -> None:
    """Small multiples: realised loss, VaR forecast and exceptions for each model."""
    sub = fc[fc["confidence"] == confidence]
    models = list(dict.fromkeys(sub["model"]))
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 2.8 * len(models)), sharex=True, sharey=True, squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        g = sub[sub["model"] == model].sort_values("forecast_date")
        ax.plot(g["forecast_date"], g["loss_pct"], lw=0.4, color="#9e9e9e", label="realised loss (-return)")
        ax.plot(g["forecast_date"], g["var_pct"], lw=1.0, color=MODEL_COLORS[model], label=f"{model} VaR {confidence:.0%}")
        exc = g[g["exception"] == 1]
        ax.scatter(exc["forecast_date"], exc["loss_pct"], s=12, color=EXCEPTION_COLOR, zorder=3,
                   label=f"exceptions (loss > VaR): {len(exc)} of {len(g)}")
        _shade(ax, spans)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Loss (pp)")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
    axes[0, 0].set_title(f"{_tag(synthetic)}Realised losses and {confidence:.0%} one-day VaR forecasts (common sample)")
    axes[-1, 0].set_xlabel("Forecast date")
    _save(fig, path)


def plot_var_es(fc: pd.DataFrame, confidence: float, spans, path: Path, synthetic: bool) -> None:
    """Small multiples of VaR and ES forecasts at one confidence level."""
    sub = fc[fc["confidence"] == confidence]
    models = list(dict.fromkeys(sub["model"]))
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 2.8 * len(models)), sharex=True, sharey=True, squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        g = sub[sub["model"] == model].sort_values("forecast_date")
        ax.plot(g["forecast_date"], g["loss_pct"], lw=0.4, color="#c7c7c7", label="realised loss")
        ax.plot(g["forecast_date"], g["var_pct"], lw=1.0, color=MODEL_COLORS[model], label=f"VaR {confidence:.0%}")
        ax.plot(g["forecast_date"], g["es_pct"], lw=1.0, color=MODEL_COLORS[model], ls=":", label=f"ES {confidence:.0%}")
        _shade(ax, spans)
        ax.set_ylabel("Loss (pp)")
        ax.set_title(model, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right", ncol=3)
    axes[0, 0].figure.suptitle(f"{_tag(synthetic)}VaR versus ES forecasts at {confidence:.0%} confidence (one day, loss units)", y=1.0)
    axes[-1, 0].set_xlabel("Forecast date")
    _save(fig, path)


def plot_rolling_exceptions(fc: pd.DataFrame, window: int, spans, path: Path, synthetic: bool) -> None:
    """Trailing exception rates per model with nominal-rate reference lines."""
    confs = sorted(fc["confidence"].unique())
    fig, axes = plt.subplots(len(confs), 1, figsize=(12, 3.4 * len(confs)), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], confs):
        sub = fc[fc["confidence"] == c]
        for model, g in sub.groupby("model", sort=False):
            g = g.sort_values("forecast_date").set_index("forecast_date")
            rate = g["exception"].astype(float).rolling(window, min_periods=window).mean()
            ax.plot(rate.index, 100 * rate, color=MODEL_COLORS[model], lw=1.0, label=model)
        ax.axhline(100 * (1 - c), color="black", ls="--", lw=1, label=f"nominal {100 * (1 - c):.0f}%")
        _shade(ax, spans)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Exception rate (%)")
        ax.set_title(f"{c:.0%} VaR: trailing {window}-forecast exception rate", fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
    fig.suptitle(f"{_tag(synthetic)}Rolling exception rates (value at t uses forecasts t-{window - 1}..t)", y=1.0)
    axes[-1, 0].set_xlabel("Forecast date")
    _save(fig, path)


def plot_stress_detail(fc: pd.DataFrame, span: tuple, confidence: float, pad_days: int, path: Path, synthetic: bool) -> None:
    """Close-up of one predefined stress regime: losses, VaR per model, exceptions."""
    label, start, end = span
    lo, hi = start - pd.Timedelta(days=pad_days), end + pd.Timedelta(days=pad_days)
    sub = fc[(fc["confidence"] == confidence) & (fc["forecast_date"] >= lo) & (fc["forecast_date"] <= hi)]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    first = sub.drop_duplicates("forecast_date").sort_values("forecast_date")
    ax.bar(first["forecast_date"], first["loss_pct"], width=1.0, color="#bdbdbd", label="realised loss (-return)")
    for model, g in sub.groupby("model", sort=False):
        g = g.sort_values("forecast_date")
        ax.plot(g["forecast_date"], g["var_pct"], color=MODEL_COLORS[model], lw=1.4, label=f"{model} VaR")
        exc = g[g["exception"] == 1]
        ax.scatter(exc["forecast_date"], exc["loss_pct"], marker="x", s=30, color=MODEL_COLORS[model], zorder=3,
                   label=f"{model} exceptions in view: {len(exc)}")
    ax.axvspan(start, end, color=STRESS_SHADE, alpha=0.3, lw=0, label=f"predefined regime '{label}'")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Loss (pp)")
    ax.set_xlabel("Forecast date")
    ax.set_title(f"{_tag(synthetic)}Stress-period close-up: {label}, {confidence:.0%} VaR (+/- {pad_days} calendar days)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    _save(fig, path)


def plot_residual_diagnostics(z_windows: pd.DataFrame, nlags: int, path: Path, synthetic: bool) -> None:
    """ACF of z and z^2 for the first and last estimation windows of each filter."""
    keys = list(z_windows.groupby(["model", "window"], sort=False).groups)
    models = list(dict.fromkeys(k[0] for k in keys))
    windows = ["first", "last"]
    fig, axes = plt.subplots(len(models), 4, figsize=(15, 3.0 * len(models)), sharey=True, squeeze=False)
    lags = np.arange(1, nlags + 1)
    for i, model in enumerate(models):
        for j, (win, transform) in enumerate([(w, t) for w in windows for t in ("z", "z^2")]):
            ax = axes[i, j]
            g = z_windows[(z_windows["model"] == model) & (z_windows["window"] == win)]
            if g.empty:
                ax.set_visible(False)
                continue
            z = g["z"].to_numpy()
            series = z if transform == "z" else z**2
            r = autocorrelations(series, nlags)
            band = 1.96 / np.sqrt(z.size)
            ax.bar(lags, r, color=MODEL_COLORS[model], width=0.7)
            ax.axhline(band, color="black", ls="--", lw=0.8)
            ax.axhline(-band, color="black", ls="--", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(
                f"{model} {win} window: ACF of {transform}\n{g['train_start'].iloc[0]} to {g['train_end'].iloc[0]}, n={z.size}",
                fontsize=8,
            )
            ax.set_xlabel("Lag")
            if j == 0:
                ax.set_ylabel("Autocorrelation")
    fig.suptitle(f"{_tag(synthetic)}In-sample standardised-residual autocorrelations (dashed: +/-1.96/sqrt(n))", y=1.01)
    fig.tight_layout()
    _save(fig, path)
