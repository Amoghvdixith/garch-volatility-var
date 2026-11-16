# Optional market data (not included)

**Status: downloaded and run locally on 2026-11-15** (3,774 rows,
2010-01-04 to 2024-12-31, `yfinance` 1.8.0, SHA-256 `db49baad…be427c7`).
`spy_daily.csv` and `spy_daily_metadata.json` in this folder are **local
files only**. Leave them out of any published copy, and recreate them with
the commands below. Results are summarised in
[`docs/market_results.md`](../../docs/market_results.md).

## Recreate

```bash
python -m pip install -e ".[dev,market]"
python scripts/download_market_data.py --config configs/market_example.json
python scripts/run_backtest.py --config configs/market_example.json
python scripts/generate_report.py --results-dir results/market_example
```

This creates `spy_daily.csv` (`date`, `adjusted_close`) and
`spy_daily_metadata.json` in this folder.

## Specification (`configs/market_example.json`)

- **Source**: Yahoo Finance via the unofficial `yfinance` package. No API key
  is needed.
- **Symbol**: SPY.
- **Date range**: start 2010-01-01; the end date 2025-01-01 is **exclusive**.
- **Price convention**: `auto_adjust=False`, using the `Adj Close` field
  (Yahoo's split- and dividend-adjusted close). The script fails rather than
  silently using the unadjusted `Close`.
- **Returns**: `r_t = 100·log(P_t/P_{t−1})` on adjusted closes.
- **Missing values**: rows with a missing adjusted close are dropped and
  counted in the metadata, with no interpolation. Dates are the exchange
  trading days returned by the provider.
- **Recorded in the metadata**: download time (UTC), actual date coverage,
  row count, yfinance version and checksum.
- **Predefined evaluation intervals**, fixed in the config before any results
  were seen: `covid_2020` = 2020-02-20 to 2020-04-30, and `year_2022` =
  calendar 2022.
- **Volatility accuracy**: true volatility is unobserved for market data, so
  no volatility RMSE or QLIKE is produced.

## Redistribution

Do **not** assume the downloaded prices may be redistributed. Yahoo Finance's
terms govern their use, and yfinance is not affiliated with Yahoo. Keep the
downloaded files local and leave them out of any published copy of this
project. Others can recreate the files with the commands above. Because the
provider may revise adjusted prices, a later download may not match an
earlier one exactly; the checksum in the metadata identifies the version
used.

## Your own CSV

You can place a CSV here instead, with `date` + `adjusted_close` or
`date` + `return_pct` (percentage-point log returns), and point
`data.path` in a copy of the config at it.
