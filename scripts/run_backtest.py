"""Run the rolling out-of-sample HS / GARCH-FHS / GJR-FHS backtest.

Usage (from the project root):

    python scripts/run_backtest.py --config configs/synthetic_baseline.json

Smoke test (explicitly labelled partial run, written to a separate folder):

    python scripts/run_backtest.py --config configs/synthetic_baseline.json --max-forecasts 60

A run with ``--max-forecasts`` is always labelled ``smoke_test_partial`` in
``run_metadata.json`` and, unless ``--output-dir`` is given, is written to
``results/<experiment>_smoke`` so it can never overwrite the baseline.
Figures and report.md are produced afterwards by ``generate_report.py`` from
the saved outputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from garch_fhs import __version__  # noqa: E402
from garch_fhs.backtesting import coverage_tests  # noqa: E402
from garch_fhs.data import load_config, load_returns_csv, resolve, sha256_file  # noqa: E402
from garch_fhs.diagnostics import residual_diagnostics  # noqa: E402
from garch_fhs.forecasting import common_sample, run_rolling_forecasts  # noqa: E402
from garch_fhs.metrics import classify_regimes, stress_table, summary_table, volatility_accuracy_table  # noqa: E402

DEPENDENCIES = ["numpy", "pandas", "scipy", "matplotlib", "arch", "statsmodels", "pytest"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling one-day-ahead VaR/ES backtest of HS, GARCH-FHS and GJR-FHS.")
    p.add_argument("--config", required=True, help="Experiment JSON config, e.g. configs/synthetic_baseline.json")
    p.add_argument(
        "--max-forecasts",
        type=int,
        default=None,
        help="Limit the number of forecast origins (SMOKE TEST ONLY; result is labelled partial).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory (relative to project root). Default: config output_dir, "
        "or results/<experiment>_smoke when --max-forecasts is set.",
    )
    p.add_argument("--progress-every", type=int, default=100, help="Print progress every N forecast origins.")
    return p.parse_args(argv)


def _versions() -> dict[str, str]:
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _fmt_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def load_experiment_data(config: dict) -> tuple[pd.DataFrame, dict]:
    """Load returns, attach regimes and return (frame, dataset info)."""
    data_cfg = config["data"]
    path = resolve(PROJECT_ROOT, data_cfg["path"])
    if not path.is_file():
        hint = (
            "run scripts/generate_synthetic_data.py first"
            if data_cfg.get("source") == "synthetic"
            else "run scripts/download_market_data.py first or supply your own CSV"
        )
        raise SystemExit(f"Input data not found: {data_cfg['path']} ({hint}).")
    loaded = load_returns_csv(path)
    df = loaded.frame
    info = {
        "source": data_cfg.get("source"),
        "path": Path(data_cfg["path"]).as_posix(),
        "sha256": sha256_file(path),
        "schema": loaded.schema,
        "rows_after_preparation": int(len(df)),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "input_was_resorted": loaded.was_sorted,
    }
    meta_path = data_cfg.get("metadata_path")
    if meta_path and resolve(PROJECT_ROOT, meta_path).is_file():
        meta = json.loads(resolve(PROJECT_ROOT, meta_path).read_text(encoding="utf-8"))
        recorded = meta.get("sha256")
        info["metadata_path"] = Path(meta_path).as_posix()
        info["checksum_matches_metadata"] = (recorded == info["sha256"]) if recorded else None
        if recorded and recorded != info["sha256"]:
            raise SystemExit("Data checksum does not match its metadata file; regenerate or re-download the data.")
    if "regime" not in df.columns:
        regimes = config.get("stress_regimes", [])
        df["regime"] = classify_regimes(df.index, regimes)
        df["stress_flag"] = (df["regime"] != "normal").astype(int)
        info["regime_definition"] = regimes
    else:
        info["regime_definition"] = "regime column supplied by the dataset (see data metadata)"
    return df, info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(resolve(PROJECT_ROOT, args.config))
    bt = config["backtest"]
    max_fc = args.max_forecasts if args.max_forecasts is not None else bt.get("max_forecasts")
    partial = max_fc is not None
    if args.output_dir:
        out_rel = Path(args.output_dir)
    elif partial:
        out_rel = Path("results") / f"{config['experiment_name']}_smoke"
    else:
        out_rel = Path(config["output_dir"])
    out_dir = resolve(PROJECT_ROOT, out_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, data_info = load_experiment_data(config)
    window = int(bt["window"])
    confs = [float(c) for c in bt["confidence_levels"]]
    models = list(bt["models"])
    if int(bt.get("refit_every", 1)) != 1:
        raise SystemExit("Only daily refitting (refit_every = 1) is implemented as the reference design.")
    n_origins = len(data) - window if not partial else min(len(data) - window, max_fc)
    run_type = "smoke_test_partial" if partial else "full"
    print(f"[{run_type}] {config['experiment_name']}: {len(data)} observations, window {window}, "
          f"{n_origins} forecast origins, models {models}, confidence {confs}")

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    def progress(k: int, total: int) -> None:
        if k % args.progress_every == 0 or k == total:
            el = time.perf_counter() - t0
            eta = el / k * (total - k)
            print(f"  origin {k}/{total}  elapsed {el:6.1f}s  eta {eta:6.1f}s", flush=True)

    out = run_rolling_forecasts(data, window, confs, models, int(bt["min_residuals"]), max_fc, progress=progress)
    runtime = time.perf_counter() - t0

    fc = out.forecasts
    common, common_info = common_sample(fc)
    summary = summary_table(common)
    stress = stress_table(common)

    cov_rows = []
    for (model, c), g in common.groupby(["model", "confidence"], sort=False):
        g = g.sort_values("forecast_date")
        samples = [("full_common", g)] + [
            (lab, g[g["regime"] == lab]) for lab in g["regime"].unique() if lab != "normal"
        ]
        for name, s in samples:
            for res in coverage_tests(s["exception"].astype(int).to_numpy(), 1.0 - c):
                note = res["note"]
                if name != "full_common":
                    note = "; ".join(x for x in [note, "stress-regime subsample: few observations, interpret with caution"] if x)
                cov_rows.append({"model": model, "confidence": c, "sample": name,
                                 "start_date": s["forecast_date"].min(), "end_date": s["forecast_date"].max(),
                                 **{**res, "note": note}})
    coverage = pd.DataFrame(cov_rows)

    lags = bt.get("residual_ljung_box_lags", [10, 20])
    diag_rows, z_rows = [], []
    for w in out.residual_windows:
        for row in residual_diagnostics(w["z"], lags):
            diag_rows.append({"model": w["model"], "window": w["window"], "train_start": w["train_start"],
                              "train_end": w["train_end"], **row})
        for i, z in enumerate(w["z"]):
            z_rows.append({"model": w["model"], "window": w["window"], "train_start": w["train_start"],
                           "train_end": w["train_end"], "obs": i, "z": z})
    resid_diag = pd.DataFrame(diag_rows)
    z_windows = pd.DataFrame(z_rows)

    date_cols = ["forecast_date", "train_start", "train_end", "start_date", "end_date"]
    _fmt_dates(fc, date_cols).to_csv(out_dir / "forecasts.csv", index=False, lineterminator="\n")
    _fmt_dates(out.fits, date_cols).to_csv(out_dir / "fit_diagnostics.csv", index=False, lineterminator="\n")
    summary.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    _fmt_dates(stress, date_cols).to_csv(out_dir / "stress_summary.csv", index=False, lineterminator="\n")
    _fmt_dates(coverage, date_cols).to_csv(out_dir / "coverage_tests.csv", index=False, lineterminator="\n")
    _fmt_dates(resid_diag, date_cols).to_csv(out_dir / "residual_diagnostics.csv", index=False, lineterminator="\n")
    _fmt_dates(z_windows, date_cols).to_csv(
        out_dir / "standardized_residual_windows.csv", index=False, float_format="%.10f", lineterminator="\n"
    )
    has_truth = fc["true_vol_pct"].notna().any()
    if has_truth:
        volatility_accuracy_table(common).to_csv(out_dir / "volatility_accuracy.csv", index=False, lineterminator="\n")

    fits = out.fits
    fit_summary = {}
    for model, g in fits.groupby("model"):
        fit_summary[model] = {
            "fits": int(len(g)),
            "status_counts": {str(k): int(v) for k, v in g["status"].value_counts().items()},
            "retries_used": int((g["attempts"] > 1).sum()),
            "fits_with_warnings": int((g["warnings"].fillna("") != "").sum()),
            "persistence_min": float(g["persistence"].min()),
            "persistence_max": float(g["persistence"].max()),
            "fits_with_persistence_at_or_above_one": int((~g["persistence_below_one"]).sum()),
            "fits_at_stationarity_boundary_persistence_ge_0.9999": int(g["at_stationarity_boundary"].sum()),
            "persistence_median": float(g["persistence"].median()),
        }
    finished = datetime.now(timezone.utc)
    run_meta = {
        "experiment": config["experiment_name"],
        "run_type": run_type,
        "run_type_note": "Full rolling backtest over all available forecast origins."
        if not partial
        else f"PARTIAL SMOKE TEST limited to {max_fc} forecast origins. Not the baseline result.",
        "synthetic_data": data_info["source"] == "synthetic",
        "dataset": data_info,
        "config_file": Path(args.config).as_posix(),
        "config": config,
        "seed": config.get("synthetic", {}).get("seed"),
        "units": {
            "returns": "percentage-point log returns (1.0 = 1%)",
            "loss": "loss_pct = -return_pct (negative log return in percentage points; not exact monetary loss)",
            "volatility": "conditional standard deviation in percentage points",
            "monetary_mapping": "exact monetary loss = notional * (1 - exp(return_pct / 100)); not used for the tables",
        },
        "conventions": {
            "empirical_distribution": "equal mass 1/n on each scenario",
            "var": "inverse empirical CDF of loss at c: L_(ceil(n c)) in ascending order",
            "es": "average of worst (1-c) probability mass with fractional boundary weight; ES >= VaR",
            "exception": "strict: realised loss > VaR",
            "quantile_loss": "(c - 1[loss < VaR]) * (loss - VaR)",
            "mean_model": "zero conditional mean for all methods",
            "volatility_estimation": "arch package, Gaussian QML, mean='zero', p=1, q=1, o=0 (GARCH) or 1 (GJR), power=2, rescale=False",
            "fhs": "z_i = eps_i / sigma_i over the training window (no recentring); scenarios sigma_t * z_i; no resampling",
            "refit": "every forecast origin; training window = previous `window` observations",
            "rolling_exception_window": bt.get("rolling_exception_window"),
        },
        "window": window,
        "min_residuals": int(bt["min_residuals"]),
        "forecast_origins": out.n_origins,
        "first_forecast_date": fc["forecast_date"].min().strftime("%Y-%m-%d"),
        "last_forecast_date": fc["forecast_date"].max().strftime("%Y-%m-%d"),
        "common_sample": common_info,
        "fit_summary": fit_summary,
        "dependency_versions": _versions(),
        "garch_fhs_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(runtime, 1),
        "invocation": "python scripts/run_backtest.py --config " + Path(args.config).as_posix()
        + (f" --max-forecasts {args.max_forecasts}" if args.max_forecasts is not None else "")
        + (f" --output-dir {Path(args.output_dir).as_posix()}" if args.output_dir else ""),
        "output_dir": out_rel.as_posix(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")

    print(f"Finished in {runtime:.1f}s. Common-sample dates: {common_info['n_dates_common']} "
          f"(excluded {common_info['n_dates_excluded']}). Outputs in {out_rel.as_posix()}/")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["model", "confidence", "n", "exceptions", "expected_exceptions", "exception_rate",
                       "mean_var_pct", "mean_es_pct", "mean_quantile_loss"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
