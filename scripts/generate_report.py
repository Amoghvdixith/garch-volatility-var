"""Build figures and report.md from the saved outputs of run_backtest.py.

Usage (from the project root):

    python scripts/generate_report.py --results-dir results/synthetic_baseline

Nothing is re-estimated: every number and chart comes from the CSV/JSON files
already in the results directory (plus the input data file named in
run_metadata.json for the returns chart). The report states facts only;
interpretation lives in docs/results.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from garch_fhs.data import load_returns_csv, resolve  # noqa: E402
from garch_fhs import plotting  # noqa: E402

REQUIRED = [
    "run_metadata.json",
    "forecasts.csv",
    "summary.csv",
    "stress_summary.csv",
    "coverage_tests.csv",
    "residual_diagnostics.csv",
    "fit_diagnostics.csv",
    "standardized_residual_windows.csv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate charts and report.md from saved backtest outputs.")
    p.add_argument("--results-dir", required=True, help="Results folder, e.g. results/synthetic_baseline")
    p.add_argument("--stress-pad-days", type=int, default=60, help="Calendar days shown either side of the stress close-up.")
    return p.parse_args(argv)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def pct(v: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(v) else f"{100 * v:.{digits}f}%"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rdir = resolve(PROJECT_ROOT, args.results_dir)
    missing = [f for f in REQUIRED if not (rdir / f).is_file()]
    if missing:
        raise SystemExit(f"Missing outputs in {args.results_dir}: {missing}. Run scripts/run_backtest.py first.")
    meta = json.loads((rdir / "run_metadata.json").read_text(encoding="utf-8"))
    parse = ["forecast_date", "train_start", "train_end"]
    fc = pd.read_csv(rdir / "forecasts.csv", parse_dates=parse)
    summary = pd.read_csv(rdir / "summary.csv")
    stress = pd.read_csv(rdir / "stress_summary.csv")
    cov = pd.read_csv(rdir / "coverage_tests.csv")
    cov["note"] = cov["note"].fillna("")
    stress["notes"] = stress["notes"].fillna("")
    rdiag = pd.read_csv(rdir / "residual_diagnostics.csv")
    fits = pd.read_csv(rdir / "fit_diagnostics.csv", parse_dates=["forecast_date"])
    zwin = pd.read_csv(rdir / "standardized_residual_windows.csv")
    vacc_path = rdir / "volatility_accuracy.csv"
    vacc = pd.read_csv(vacc_path) if vacc_path.is_file() else None

    synthetic = bool(meta.get("synthetic_data"))
    partial = meta["run_type"] != "full"
    excluded = set(meta["common_sample"]["excluded_dates"])
    common = fc[~fc["forecast_date"].dt.strftime("%Y-%m-%d").isin(excluded)]

    data_path = resolve(PROJECT_ROOT, meta["dataset"]["path"])
    data = load_returns_csv(data_path).frame
    if "regime" not in data:
        from garch_fhs.metrics import classify_regimes

        data["regime"] = classify_regimes(data.index, meta["config"].get("stress_regimes", []))
    spans_all = plotting.regime_spans(data.index, data["regime"])
    lo, hi = common["forecast_date"].min(), common["forecast_date"].max()
    spans_fc = [s for s in spans_all if s[2] >= lo and s[1] <= hi]

    figs = rdir / "figures"
    figs.mkdir(exist_ok=True)
    window_roll = int(meta["config"]["backtest"].get("rolling_exception_window", 250))
    has_truth = synthetic and "true_vol_pct" in data
    first_fig = "returns_and_true_volatility.png" if has_truth else "returns_overview.png"
    plotting.plot_returns_and_volatility(data, spans_all, figs / first_fig, synthetic, first_forecast=lo)
    plotting.plot_volatility_forecasts(common, spans_fc, figs / "volatility_forecasts.png", synthetic)
    plotting.plot_var_comparison(common, 0.95, spans_fc, figs / "var_95_comparison.png", synthetic)
    plotting.plot_var_comparison(common, 0.99, spans_fc, figs / "var_99_comparison.png", synthetic)
    plotting.plot_var_es(common, 0.99, spans_fc, figs / "var_es_comparison.png", synthetic)
    plotting.plot_rolling_exceptions(common, window_roll, spans_fc, figs / "rolling_exception_rates.png", synthetic)
    if spans_fc:
        # Predefined rule: close-up of the first configured stress regime inside the forecast period.
        plotting.plot_stress_detail(common, spans_fc[0], 0.99, args.stress_pad_days, figs / "stress_period_detail.png", synthetic)
    plotting.plot_residual_diagnostics(zwin, 20, figs / "residual_diagnostics.png", synthetic)

    # ---------------- report ----------------
    L: list[str] = []
    title = meta["experiment"].replace("_", " ")
    L.append(f"# Backtest report: {title}")
    L.append("")
    if synthetic:
        L.append("> **SYNTHETIC DATA.** All observations are simulated from a GJR-GARCH Student-t process with "
                 "predefined stress regimes. Nothing here is market history.")
        L.append("")
    if partial:
        L.append(f"> **PARTIAL SMOKE TEST.** {meta['run_type_note']} These numbers are not the baseline result.")
        L.append("")
    L.append("This report is generated by `scripts/generate_report.py` from the saved files in this folder. "
             "It records observed values only; interpretation and caveats are discussed in `docs/results.md`.")
    L.append("")
    L.append("## Run")
    L.append("")
    ds = meta["dataset"]
    L.append(f"- Run type: `{meta['run_type']}`; invocation: `{meta['invocation']}`")
    L.append(f"- Data: `{ds['path']}` ({ds['rows_after_preparation']} rows, {ds['first_date']} to {ds['last_date']}), "
             f"SHA-256 `{ds['sha256'][:16]}...`")
    L.append(f"- Rolling window: {meta['window']} observations, refit at every origin; minimum residual pool "
             f"{meta['min_residuals']}")
    L.append(f"- Forecast origins: {meta['forecast_origins']} ({meta['first_forecast_date']} to {meta['last_forecast_date']})")
    cs = meta["common_sample"]
    L.append(f"- Common valid sample: {cs['n_dates_common']} dates; excluded {cs['n_dates_excluded']} "
             f"(invalid forecasts by model: {cs['invalid_forecasts_by_model'] or 'none'})")
    for model, fsum in meta["fit_summary"].items():
        L.append(f"- {model} fits: {fsum['fits']}, status {fsum['status_counts']}, retries used {fsum['retries_used']}, "
                 f"fits with warnings {fsum['fits_with_warnings']}, persistence range "
                 f"[{fsum['persistence_min']:.4f}, {fsum['persistence_max']:.4f}], median "
                 f"{fsum.get('persistence_median', float('nan')):.4f}; fits at the stationarity boundary "
                 f"(persistence >= 0.9999): {fsum.get('fits_at_stationarity_boundary_persistence_ge_0.9999', 'n/a')}")
    L.append(f"- Python {meta['python_version']}, {meta['platform']}; runtime {meta['runtime_seconds']} s; "
             f"finished {meta['finished_utc']}")
    L.append("")

    L.append("## Full common-sample results")
    L.append("")
    tab = summary[["model", "confidence", "n", "exceptions", "expected_exceptions", "exception_rate",
                   "mean_var_pct", "mean_es_pct", "mean_excess_loss_on_exceptions_pct", "mean_quantile_loss",
                   "max_consecutive_exceptions"]].copy()
    tab.columns = ["model", "c", "n", "exceptions", "expected", "exc. rate", "mean VaR (pp)", "mean ES (pp)",
                   "mean excess loss on exc. (pp)", "mean quantile loss", "max run"]
    tab["c"] = tab["c"].map("{:.2f}".format)
    L.append(md_table(tab))
    L.append("")
    L.append(f"ES below VaR (should be 0): {int(summary['es_below_var_count'].sum())} forecasts.")
    L.append("")

    L.append("### Coverage tests (full common sample)")
    L.append("")
    full = cov[cov["sample"] == "full_common"][["model", "confidence", "test", "statistic", "df", "p_value", "status", "note"]]
    full = full.assign(confidence=full["confidence"].map("{:.2f}".format))
    L.append(md_table(full))
    L.append("")
    L.append("Asymptotic chi-square p-values; they ignore parameter-estimation error from the rolling windows. "
             "A non-rejection does not show a model is correct.")
    L.append("")

    L.append("### Observed coverage, stated factually")
    L.append("")
    for r in summary.itertuples(index=False):
        k = cov[(cov["sample"] == "full_common") & (cov["model"] == r.model) & (cov["confidence"] == r.confidence)]
        kp = k.loc[k["test"] == "kupiec_uc", "p_value"].iloc[0]
        cc = k.loc[k["test"] == "christoffersen_cc"]
        ccp = cc["p_value"].iloc[0] if cc["status"].iloc[0] == "defined" else float("nan")
        L.append(f"- {r.model} at {r.confidence:.0%}: {r.exceptions} exceptions vs {r.expected_exceptions:.1f} expected "
                 f"(rate {pct(r.exception_rate)} vs nominal {pct(r.nominal_rate)}); Kupiec p = "
                 f"{'n/a' if pd.isna(kp) else f'{kp:.3f}'}, conditional-coverage p = {'undefined' if pd.isna(ccp) else f'{ccp:.3f}'}.")
    L.append("")

    L.append("## Stress and non-stress regimes")
    L.append("")
    st = stress[["model", "confidence", "regime", "n", "exceptions", "expected_exceptions", "exception_rate",
                 "mean_var_pct", "mean_es_pct", "mean_quantile_loss", "kupiec_p_value", "notes"]].copy()
    st.columns = ["model", "c", "regime", "n", "exceptions", "expected", "exc. rate", "mean VaR", "mean ES",
                  "mean QL", "Kupiec p", "notes"]
    st["c"] = st["c"].map("{:.2f}".format)
    L.append(md_table(st))
    L.append("")
    L.append("Stress samples are short; p-values there have very low power and are shown for completeness only.")
    L.append("")

    if vacc is not None:
        L.append("## Volatility forecasts versus true synthetic volatility")
        L.append("")
        L.append("Available only because the synthetic data-generating volatility is known. Not available for market data.")
        L.append("")
        L.append(md_table(vacc))
        L.append("")

    L.append("## In-sample residual diagnostics (first and last estimation windows)")
    L.append("")
    rd = rdiag.copy()
    rd["lag"] = rd["lag"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    L.append(md_table(rd[["model", "window", "train_start", "train_end", "diagnostic", "lag", "statistic", "p_value", "n"]]))
    L.append("")
    L.append("These describe in-sample fit of two windows only; they are not out-of-sample accuracy measures and "
             "do not prove that standardised shocks are independent.")
    L.append("")

    L.append("## Figures")
    L.append("")
    fig_list = [
        (first_fig, "Returns" + (" and true conditional volatility" if has_truth else "") + " with predefined stress shading"),
        ("volatility_forecasts.png", "One-step volatility forecasts" + (" versus true volatility" if has_truth else "")),
        ("var_95_comparison.png", "Realised losses, 95% VaR and exceptions"),
        ("var_99_comparison.png", "Realised losses, 99% VaR and exceptions"),
        ("var_es_comparison.png", "VaR and ES at 99% confidence"),
        ("rolling_exception_rates.png", f"Trailing {window_roll}-forecast exception rates"),
        ("stress_period_detail.png", "Close-up of the first predefined stress regime (99% VaR)"),
        ("residual_diagnostics.png", "Standardised residual and squared-residual autocorrelations"),
    ]
    L.append("Synthetic-data note: the volatility-accuracy table above is the only place where the true "
             "data-generating volatility is used; all VaR/ES forecasts use returns only."
             if vacc is not None else "")
    L.append("")
    for fname, caption in fig_list:
        if (figs / fname).is_file():
            L.append(f"### {caption}")
            L.append("")
            L.append(f"![{caption}](figures/{fname})")
            L.append("")

    L.append("## Limitations")
    L.append("")
    L.append("- One asset, one-day horizon, zero conditional mean, Gaussian QML estimation.")
    L.append("- VaR coverage tests are not ES backtests; ES is compared descriptively only.")
    L.append("- 99% VaR has few expected exceptions, so coverage tests have limited power.")
    if synthetic:
        L.append("- The synthetic process is GJR-GARCH-type, which may favour the related filters; "
                 "synthetic results are not evidence about real markets.")
    L.append("")
    (rdir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.results_dir}/report.md and {len(list(figs.glob('*.png')))} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
