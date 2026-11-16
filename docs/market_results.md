# Results: optional SPY market example

> **Market data, one asset, one run.** These results use SPY adjusted closes
> downloaded from Yahoo Finance through the unofficial `yfinance` package.
> The downloaded prices are a local file and are **not** part of the
> project's distributable content (see
> [`data/market/README.md`](../data/market/README.md)).

Source files: `results/market_example/`, with the factual report in
`results/market_example/report.md`. Both are produced locally by the commands
in the README and depend on a downloaded file that is not redistributed.

## 1. Data and executed configuration

- Source: Yahoo Finance via `yfinance` 1.8.0, `auto_adjust=False`, field
  `Adj Close`. Downloaded 2026-11-15 (UTC).
- Requested range: 2010-01-01 up to, but not including, 2025-01-01.
  Received 3,774 trading days (2010-01-04 to 2024-12-31), with 0 rows
  dropped for a missing adjusted close.
- File SHA-256: `db49baad…be427c7` (full value in
  `data/market/spy_daily_metadata.json`). Yahoo may revise adjusted prices,
  so a later download may give a different checksum and slightly different
  numbers.
- Returns: 3,773 daily log returns in percentage points (2010-01-05 to
  2024-12-31). Sanity checks before the run: the largest absolute returns
  were all in March 2020 (largest −11.6% on 2020-03-16), and the longest gap
  between trading dates was 5 calendar days.
- Rolling window 1,000; GARCH and GJR refitted at every one of **2,773**
  forecast origins (2013-12-24 to 2024-12-31). Same code, conventions and
  settings as the synthetic baseline. Runtime 62 s.
- Evaluation regimes, fixed in `configs/market_example.json` before the run:
  `covid_2020` (2020-02-20 to 2020-04-30, 50 forecast days) and `year_2022`
  (calendar 2022, 251 forecast days).

## 2. Failures, exclusions and fit behaviour

- All 5,546 fits converged on the first attempt, with no retries, no
  warnings and no failed forecasts. Common sample: 2,773 dates, 0 excluded.
- GARCH persistence ranged from 0.872 to 0.995 (median 0.954), with no fits
  at the stationarity boundary.
- GJR persistence ranged from 0.912 to 1.000 (median 0.968). **331 GJR fits
  were at the boundary** (persistence ≥ 0.9999), all for forecast dates from
  2020-06-12 to 2021-11-26, that is, for training windows that contained
  the March 2020 crash.

## 3. Full common-sample results (n = 2,773)

| model | c | exceptions (expected) | rate | Kupiec p | Christoffersen IND p | CC p | mean VaR | mean ES | mean QL |
|---|---|---|---|---|---|---|---|---|---|
| HS | 95% | 142 (138.7) | 5.12% | 0.771 | <0.001 | <0.001 | 1.640 | 2.630 | 0.1377 |
| GARCH-FHS | 95% | 145 (138.7) | 5.23% | 0.583 | 0.116 | 0.249 | 1.557 | 2.288 | 0.1153 |
| GJR-FHS | 95% | 145 (138.7) | 5.23% | 0.583 | 0.875 | 0.847 | 1.582 | 2.278 | 0.1126 |
| HS | 99% | 38 (27.7) | 1.37% | 0.063 | 0.002 | 0.001 | 3.015 | 4.432 | 0.0476 |
| GARCH-FHS | 99% | 31 (27.7) | 1.12% | 0.540 | 0.004 | 0.014 | 2.754 | 3.464 | 0.0351 |
| GJR-FHS | 99% | 29 (27.7) | 1.05% | 0.810 | 0.311 | 0.581 | 2.626 | 3.502 | 0.0345 |

Units are percentage-point losses. ES ≥ VaR held for every forecast.

Observations:

- **Unconditional coverage.** No model is rejected by the Kupiec test at
  the 5% level. HS at 99% is closest (38 vs 27.7 expected, p = 0.063).
- **Clustering.** HS is rejected by the independence test at both levels:
  at 95% it had 22 back-to-back exception pairs, against 12 for GARCH-FHS
  and 8 for GJR-FHS. GARCH-FHS is also rejected for independence at 99%
  (p = 0.004), driven by 3 consecutive pairs (2015-08-21, 2015-08-24 and
  2018-02-05). With only 31 exceptions this rests on very few events.
  GJR-FHS is not rejected by any test.
- **Quantile loss.** Relative to HS, it is about 16% (GARCH) and 18% (GJR)
  lower at 95%, and about 26% and 27% lower at 99%. GJR-FHS has the lowest
  quantile loss at both levels, by a small margin over GARCH-FHS.
- Both FHS models also have lower average VaR and ES than HS.
- **Descriptive ES comparison.** At 99%, mean realised loss on exception
  days versus mean ES on those days was 4.01 vs 3.98 (HS), 2.88 vs 2.84
  (GARCH) and 2.76 vs 2.71 (GJR). This is not a formal ES backtest.

## 4. Predefined stress regimes

| regime (n) | c | HS exc. | GARCH-FHS exc. | GJR-FHS exc. | expected |
|---|---|---|---|---|---|
| normal (2,472) | 95% | 103 | 118 | 120 | 123.6 |
| covid_2020 (50) | 95% | 18 | 7 | 6 | 2.5 |
| year_2022 (251) | 95% | 21 | 20 | 19 | 12.6 |
| normal (2,472) | 99% | 22 | 28 | 25 | 24.7 |
| covid_2020 (50) | 99% | 11 | 1 | 1 | 0.5 |
| year_2022 (251) | 99% | 5 | 2 | 3 | 2.5 |

- **COVID-2020.** Mean 95% HS VaR in this window was 1.51 pp, compared with
  5.42 (GARCH) and 5.71 (GJR). HS hardly moved and was breached on 18 of 50
  days. The FHS models scaled up quickly but still had more exceptions than
  expected at 95% (6–7 vs 2.5). All 7 GARCH-FHS exceptions and 5 of the 6
  GJR-FHS exceptions fell between 2020-02-21 and 2020-03-16, the first weeks
  of the crash (see `results/market_example/figures/stress_period_detail.png`).
- **2022.** All three models had more exceptions than nominal at 95%
  (19–21 vs 12.6; Kupiec p = 0.025 HS, 0.046 GARCH, 0.081 GJR). The models
  were much closer to one another than in 2020.
- In the normal regime, HS is on the conservative side at 95% (103 vs 123.6
  expected, Kupiec p = 0.051), partly because the 2020 losses sit in its
  window for the following four years.
- The COVID window has 50 observations and 0.5–2.5 expected exceptions, so
  its tests have very little power. They are descriptive only.

## 5. Residual diagnostics (in-sample, first and last windows)

Standardised residuals have variance ≈ 1.0, negative skewness (−0.39 to
−0.46) and excess kurtosis 0.87–1.07. Ljung–Box tests on `z` do not reject
at lags 10 or 20. For `z²`, GARCH in the first window (2010-01 to 2013-12)
rejects at lag 10 (p = 0.014), suggesting some volatility dynamics left
unfiltered in that window; GJR's corresponding p-value is 0.099. The last
window shows no rejections. These are two in-sample windows only.

## 6. Hypothesis assessment (SPY, 2013-12 to 2024-12)

- **FHS versus HS: supported.** All three pass unconditional coverage over
  the full sample, so the difference is not the exception count. It is that
  HS exceptions cluster, HS fails badly in the COVID window, and HS has
  higher quantile loss.
- **GJR versus GARCH filtering: weakly supported.** GJR-FHS has slightly
  lower quantile loss and no test rejections, while GARCH-FHS fails the 99%
  independence test. The differences are small and depend on a handful of
  exception days, so they are not decisive.
- **Stress performance: mixed.** FHS was far better than HS in the COVID
  crash but still under-covered at its onset. In 2022 no model held 95%
  coverage, and the filters gave little advantage.

## 7. Evidence limits

- One asset (SPY), one sample period, one download. No other assets, periods
  or data sources were tested, so these results do not establish general
  superiority.
- The two evaluation regimes were chosen by name before the run but with
  prior knowledge that they were turbulent periods.
- Many tests are reported across models, levels and regimes with no
  multiple-testing adjustment; individual p-values near 0.05 should be read
  cautiously.
- Asymptotic χ² p-values that ignore parameter-estimation error; few
  exceptions at 99%.
- True volatility is unobserved, so there is no volatility-accuracy table.
- No formal ES backtest. Zero conditional mean; Gaussian QML estimation.
- Yahoo adjusted prices can be revised, and `yfinance` is unofficial.
