# Results

## Execution status

| experiment | folder | status |
|---|---|---|
| Synthetic baseline | `synthetic_baseline/` | **Full run completed**: 2,500 forecast origins, 0 fit failures, 0 excluded dates (2026-11-15, Python 3.14.4, Windows 11) |
| Optional SPY market example | `market_example/` | **Full run completed locally**: 2,773 forecast origins, 0 fit failures, 0 excluded dates (2026-11-15). Derived from downloaded Yahoo Finance prices, so it is a **local folder and not part of the distributable project**. Recreate it with the commands in `data/market/README.md`; the summary is in `docs/market_results.md`. |

A labelled smoke test (`--max-forecasts 60`) was run before the baseline and
then deleted. It is described in `docs/validation.md`.

## Files in `synthetic_baseline/`

| file | produced by | content |
|---|---|---|
| `forecasts.csv` | `run_backtest.py` | one row per forecast date × model × confidence: training dates, realised return and loss, σ forecast, true σ (synthetic only), VaR, ES, exception flag, quantile loss, regime, status, number of scenarios |
| `fit_diagnostics.csv` | `run_backtest.py` | one row per fit (date × GARCH/GJR): parameters, convergence, attempts, warnings, log likelihood, persistence, boundary flag, σ forecast, residual counts, training size |
| `summary.csv` | `run_backtest.py` | common-sample metrics by model and confidence |
| `stress_summary.csv` | `run_backtest.py` | metrics by model, confidence and regime (normal, each stress regime, pooled stress), with Kupiec p-values and short-sample notes |
| `coverage_tests.csv` | `run_backtest.py` | Kupiec, Christoffersen IND and CC for the full common sample and each stress regime: statistic, df, p-value, n, status and notes |
| `residual_diagnostics.csv` | `run_backtest.py` | moments and Ljung–Box tests (lags 10 and 20) on z and z² for the first and last estimation windows |
| `standardized_residual_windows.csv` | `run_backtest.py` | the z values for those two windows per model (used for the residual chart). An extra file, kept small. |
| `volatility_accuracy.csv` | `run_backtest.py` | σ RMSE, MAE and QLIKE against the true synthetic σ, by model and regime. An extra file, synthetic only. |
| `run_metadata.json` | `run_backtest.py` | dataset checksum, full configuration, units, conventions, dependency versions, Python version, platform, UTC timestamps, runtime, fit summary, common-sample exclusions, invocation. Paths are relative. |
| `test_results.xml` | `scripts/run_tests.py` | JUnit XML of the final test run (hostname removed) |
| `report.md` | `generate_report.py` | factual report built from the files above |
| `figures/*.png` | `generate_report.py` | 8 charts built from the saved outputs |

`market_example/` has the same files except `volatility_accuracy.csv`
(true volatility is unobserved) and `test_results.xml` (the test suite does
not depend on market data).

Fitted model objects are not saved. The fit diagnostics and forecasts are
enough to reproduce every table.

## Reproduce

From the project root, with the virtual environment active:

```bash
python scripts/generate_synthetic_data.py --config configs/synthetic_baseline.json
python scripts/run_backtest.py --config configs/synthetic_baseline.json
python scripts/generate_report.py --results-dir results/synthetic_baseline
python scripts/run_tests.py --results-dir results/synthetic_baseline
```

The backtest takes about 1 minute. Numbers should match to floating-point
precision with the pinned versions in `requirements-lock.txt`; other
`arch`/`scipy` versions may produce small optimiser differences. Timestamps
and runtimes will differ.
