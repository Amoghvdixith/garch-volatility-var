"""Generate the synthetic GJR-GARCH Student-t return series and its metadata.

Usage (from the project root):

    python scripts/generate_synthetic_data.py --config configs/synthetic_baseline.json

Writes ``data.path`` (CSV) and ``data.metadata_path`` (JSON) from the config.
The output is deterministic for a given configuration and seed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from garch_fhs import __version__  # noqa: E402
from garch_fhs.data import load_config, resolve, sha256_file  # noqa: E402
from garch_fhs.synthetic import (  # noqa: E402
    SyntheticConfig,
    generate,
    persistence,
    standardized_t_scale,
    unconditional_variance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the reproducible synthetic GJR-GARCH(1,1,1) Student-t return series."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment JSON config with a 'synthetic' section "
        "(e.g. configs/synthetic_baseline.json). Relative paths are resolved from the project root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    if "synthetic" not in config:
        raise SystemExit(f"config {args.config} has no 'synthetic' section")

    cfg = SyntheticConfig.from_dict(config["synthetic"])
    df = generate(cfg)

    csv_path = resolve(PROJECT_ROOT, config["data"]["path"])
    meta_path = resolve(PROJECT_ROOT, config["data"]["metadata_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.assign(date=df["date"].dt.strftime("%Y-%m-%d"))
    # Fixed float format and '\n' line endings make the checksum platform-independent.
    out.to_csv(csv_path, index=False, float_format="%.10f", lineterminator="\n")

    pers = persistence(cfg.alpha, cfg.gamma, cfg.beta)
    stress_pers_var = unconditional_variance(
        cfg.omega * cfg.stress_omega_multiplier, cfg.alpha, cfg.gamma, cfg.beta
    )
    metadata = {
        "dataset": "synthetic GJR-GARCH(1,1,1) with standardised Student-t innovations",
        "synthetic_data_notice": "SYNTHETIC DATA. These are simulated observations, not market history.",
        "file": config["data"]["path"],
        "sha256": sha256_file(csv_path),
        "rows": int(len(df)),
        "first_date": out["date"].iloc[0],
        "last_date": out["date"].iloc[-1],
        "date_index": "Synthetic Monday-Friday business-day index from pandas.bdate_range. "
        "It is NOT an exchange trading calendar and contains no holidays.",
        "columns": {
            "date": "synthetic business date (YYYY-MM-DD)",
            "return_pct": "simulated log return in percentage points (1.0 = 1%)",
            "true_vol_pct": "true conditional volatility sigma_t of return_pct, percentage points; "
            "known at t-1 in the data-generating process",
            "stress_flag": "1 inside a predefined stress regime, else 0",
            "regime": "'normal' or the stress-regime label",
        },
        "parameters": cfg.to_dict(),
        "process": {
            "returns": "r_t = mu + eps_t, eps_t = sigma_t * z_t",
            "variance": "sigma_t^2 = omega_t + (alpha + gamma*1[eps_{t-1}<0]) * eps_{t-1}^2 + beta*sigma_{t-1}^2",
            "innovations": "z_t = T_t * sqrt((nu-2)/nu), T_t ~ Student-t(nu) via numpy Generator.standard_t",
            "unit_variance_scale": standardized_t_scale(cfg.nu),
            "rng": "numpy.random.default_rng(seed); all burn-in + retained shocks drawn in one call",
            "initialisation": "sigma_0^2 at the first burn-in observation equals the baseline "
            "unconditional variance omega/(1 - alpha - gamma/2 - beta); burn-in observations are discarded",
            "stress_definition": "for retained zero-based indices within each regime (inclusive), "
            "omega_t = omega * stress_omega_multiplier; burn-in is never stressed",
            "stress_note": "Stress regimes are fixed before any forecasting and intentionally depart "
            "from a single constant-parameter GJR process.",
        },
        "admissibility": {
            "persistence_alpha_plus_half_gamma_plus_beta": pers,
            "finite_variance_condition": "alpha + gamma/2 + beta < 1 (symmetric innovations)",
            "condition_satisfied": bool(pers < 1.0),
            "baseline_unconditional_variance": unconditional_variance(cfg.omega, cfg.alpha, cfg.gamma, cfg.beta),
            "stress_omega_unconditional_variance_if_permanent": stress_pers_var,
        },
        "sample_statistics_retained": {
            "return_mean": float(df["return_pct"].mean()),
            "return_std": float(df["return_pct"].std(ddof=1)),
            "true_vol_min": float(df["true_vol_pct"].min()),
            "true_vol_max": float(df["true_vol_pct"].max()),
            "stress_observations": int(df["stress_flag"].sum()),
        },
        "generator": {
            "package": "garch_fhs",
            "package_version": __version__,
            "script": "scripts/generate_synthetic_data.py",
            "invocation": "python scripts/generate_synthetic_data.py --config "
            + Path(args.config).as_posix(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} synthetic observations to {config['data']['path']}")
    print(f"Wrote metadata to {config['data']['metadata_path']} (sha256 {metadata['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
