"""OPTIONAL: download daily SPY prices with yfinance for the market example.

Requires the optional extra:  python -m pip install -e ".[dev,market]"

Usage (from the project root):

    python scripts/download_market_data.py --config configs/market_example.json

Price convention: yfinance is called with ``auto_adjust=False`` and the
``Adj Close`` field is used (split- and dividend-adjusted close as supplied by
Yahoo Finance). The end date in the config is **exclusive**, matching
yfinance's ``end`` argument.

Missing-value policy: rows whose adjusted close is missing are dropped and
counted in the metadata; no interpolation is performed.

Data-use note: yfinance is an unofficial interface to Yahoo Finance. Yahoo's
terms govern use of the data; this project does not assume redistribution is
permitted, so the downloaded CSV is a local artefact and is not a required
publication file. Users recreate it by running this script.

The synthetic baseline does not use this script. It was run once for the
local SPY example (see docs/market_results.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from garch_fhs.data import extract_adjusted_close, load_config, resolve, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily adjusted prices with yfinance (optional market example).")
    p.add_argument("--config", required=True, help="Market config, e.g. configs/market_example.json")
    p.add_argument("--force", action="store_true", help="Overwrite an existing downloaded CSV.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(resolve(PROJECT_ROOT, args.config))
    m = config["market"]
    out_csv = resolve(PROJECT_ROOT, config["data"]["path"])
    out_meta = resolve(PROJECT_ROOT, config["data"]["metadata_path"])
    if out_csv.exists() and not args.force:
        raise SystemExit(f"{config['data']['path']} already exists; use --force to overwrite.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit('yfinance is not installed. Run: python -m pip install -e ".[dev,market]"') from exc

    downloaded_at = datetime.now(timezone.utc)
    raw = yf.download(
        m["symbol"],
        start=m["start_date"],
        end=m["end_date_exclusive"],
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise SystemExit("Download returned no rows (network unavailable, symbol invalid, or provider change).")
    adj = extract_adjusted_close(raw, m["symbol"], m.get("price_field", "Adj Close"))
    n_raw = int(adj.size)
    adj = adj.dropna()
    n_missing = n_raw - int(adj.size)
    df = adj.reset_index()
    df.columns = ["date", "adjusted_close"]
    dates = pd.to_datetime(df["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    df["date"] = dates.dt.strftime("%Y-%m-%d")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f", lineterminator="\n")

    meta = {
        "source": m.get("provider", "yfinance"),
        "symbol": m["symbol"],
        "price_field": m.get("price_field", "Adj Close"),
        "adjustment": "Yahoo Finance adjusted close (splits and dividends), downloaded with auto_adjust=False",
        "requested_start": m["start_date"],
        "requested_end_exclusive": m["end_date_exclusive"],
        "first_date": df["date"].iloc[0],
        "last_date": df["date"].iloc[-1],
        "rows": int(len(df)),
        "rows_dropped_missing_adjusted_close": n_missing,
        "missing_value_policy": "rows with missing adjusted close dropped; no interpolation",
        "returns_definition": "r_t = 100 * log(P_t / P_{t-1}) computed by the backtest loader",
        "downloaded_utc": downloaded_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "file": Path(config["data"]["path"]).as_posix(),
        "sha256": sha256_file(out_csv),
        "redistribution": "Not assumed to be permitted. Keep this file local unless the provider's terms allow sharing.",
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(df)} rows ({meta['first_date']} to {meta['last_date']}) to {config['data']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
