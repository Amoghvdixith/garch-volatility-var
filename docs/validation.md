# Validation

## Test run of record

Command: `python scripts/run_tests.py --results-dir results/synthetic_baseline`.
It is equivalent to `python -m pytest -q --junitxml=...`, then removes the
machine hostname from the XML.

Result saved in
[`results/synthetic_baseline/test_results.xml`](../results/synthetic_baseline/test_results.xml):
**121 tests, 121 passed, 0 failed, 0 errors, 0 skipped** (Python 3.14.4,
Windows 11, pytest 9.2.0, 2026-11-15 UTC).

| test file | tests | what it covers |
|---|---|---|
| `test_data.py` | 19 | log-return units, positive and finite prices, sorting, duplicate rejection, ambiguous and missing CSV schemas, paths with spaces, yfinance column layouts |
| `test_synthetic.py` | 22 | seed reproducibility, lengths, positive variance, exact stress indices, ω multiplier, hand-worked recursion, unit-variance t scaling, parameter validation, config matching the specification |
| `test_volatility.py` | 12 | hand-worked GARCH and GJR recursions (positive and negative shocks), `arch` configuration and parameter mapping, `arch` in-sample variances and one-step forecast against the documented recursion, failure handling |
| `test_historical.py` | 19 | hand-worked VaR/ES, fractional ES boundary, whole-number tails robust to float error, ties, comparison with an independent formula, ES ≥ VaR, no clipping, sign and units, monetary mapping |
| `test_filtered_historical.py` | 8 | residual exclusion, no recentring, σ·z scaling without an extra ×100, hand-worked FHS VaR/ES, determinism, minimum residual pool |
| `test_forecasting.py` | 10 | training window ends before the forecast date, identical windows for all models, **no-look-ahead (future and same-day perturbation)**, HS matches the training window, strict exceptions, failed-fit recording, common-sample exclusion, insufficient residuals, regime alignment |
| `test_backtesting.py` | 14 | Kupiec known case, zero and all exceptions, nominal rate gives zero, Christoffersen hand calculation, CC = UC + IND decomposition, zero-cell handling, degenerate tables reported as undefined, p-values in [0, 1], low-power note |
| `test_metrics.py` | 17 | quantile-loss examples, consecutive-exception runs, rolling-window alignment, stress classification with inclusive bounds and overlap rejection, RMSE and QLIKE, hand-worked summary statistics, stress-table flags |

## Independent numerical checks

The expected values are written out by hand or from textbook formulas, not
taken from the code. Examples:

- **VaR and ES** on losses 1 to 10:
  - c = 0.75 gives VaR 8 and ES = (10 + 9 + 0.5·8)/2.5 = 9.2.
  - n = 250 at c = 0.99 gives VaR 248 and ES = (250 + 249 + 0.5·248)/2.5.
  - n = 1000 at 95% and 99% gives 950/975.5 and 990/995.5.
- **ES cross-check**: compared with `VaR + E[(L−VaR)⁺]/(1−c)` to a relative
  tolerance of 1e-12, across 3 sample sizes and 3 confidence levels.
- **Recursions**:
  - GARCH: ε = [2, −1] gives σ² = [1, 1.3, 1.24].
  - GJR: gives [1, 1.1, 1.13], and the asymmetry difference equals γ·ε².
  - Synthetic simulator: z = [1, −1, 2] gives σ² = [1, 0.7, 0.66].
- **arch consistency**: in-sample `arch` variances satisfy the documented
  recursion to rtol 1e-10. `arch`'s one-step forecast equals the manual
  recursion to rtol 1e-10.
- **Coverage tests**:
  - Kupiec with n = 250, x = 5, p = 0.01 is written out with `math.log`.
  - Zero exceptions gives −2n·log(1−p); all exceptions gives −2n·log p.
  - Christoffersen on a 10-day sequence with counts (3, 3, 2, 1) is
    evaluated by hand.

Tolerances: `pytest.approx` defaults (relative 1e-6) unless stated
otherwise. Two checks use large fixed-seed samples, so they are
deterministic rather than flaky:

- Unit variance of the t shocks: 400k draws, tolerance 0.03, which is about
  5× the sampling standard error.
- Loose bounds on GJR parameters estimated from 1,000 synthetic observations.

## Leakage tests

`tests/test_forecasting.py` runs the real rolling loop with `arch` fits:

1. `train_end < forecast_date` for every row. Each position-based window is
   exactly `t−window … t−1`.
2. Multiplying all returns from position `t+2` onward by −5 leaves the
   forecasts for positions `t` and `t+1` bit-for-bit unchanged, while later
   forecasts do change.
3. Replacing the realised return on the forecast date by −25 leaves that
   day's VaR, ES and σ forecasts unchanged. Only the realised loss and the
   exception flag change.
4. A spy on the fitting function confirms it only ever receives the
   training-window array.

## End-to-end runs

- **Smoke test** (explicitly partial): `run_backtest.py --max-forecasts 60`
  produced 60 origins with 0 fit failures in 1.7 s. It was written to
  `results/synthetic_baseline_smoke/` and labelled `smoke_test_partial`.
  This folder was deleted after the full run, because it is superseded and
  not part of the deliverable. It is recreated by the same command.
- **Full baseline**: 2,500 origins × 2 refitted models, 0 failures, 52 s.
- **Determinism**: regenerating the synthetic data reproduced the same
  SHA-256. `run_backtest.py` checks the data checksum against
  `metadata.json` and refuses to run on a mismatch.

## Output checks performed

- ES ≥ VaR for all 15,000 baseline forecasts (`es_below_var_count = 0`).
- Every baseline forecast row satisfies `train_end < forecast_date`, and each
  model/confidence pair has 2,500 rows.
- All 8 figures were opened and visually inspected for axes, units, labels,
  the synthetic tag and date alignment.
- Relative links in the markdown files resolve to existing files.
- No absolute user paths appear in the saved results or metadata.

## What validation does not establish

- It does not show that any model is correct for real market data.
- Passing coverage tests is not proof of correct calibration, and the tests
  have low power at 99% and inside stress regimes.
- The `arch` optimiser is trusted to find a good optimum. It is checked for
  convergence and recursion consistency, not for global optimality.
- Results come from a single simulated path, not a distribution over many
  paths.
- There is no formal ES backtest.
- The optional market-data download script is unit-tested only for its
  column-extraction logic. With the user's approval it was run once on
  2026-11-15 (SPY, 3,774 rows). Before the backtest, the downloaded
  returns were sanity-checked: the largest moves fell in March 2020, the
  longest gap between dates was 5 calendar days, and 0 rows were missing.
  The SPY outputs were checked the same way as the synthetic ones (ES ≥ VaR,
  `train_end` < `forecast_date` on every row, 0 exclusions, figures
  inspected). Downloaded data cannot be re-verified against a fixed
  checksum, because the provider may revise adjusted prices.
