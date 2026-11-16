# Results: synthetic baseline

> **All results below come from SYNTHETIC data.** The optional SPY
> market example is reported separately in
> [`market_results.md`](market_results.md).

Source files: [`results/synthetic_baseline/`](../results/synthetic_baseline/).
The auto-generated factual report is
[`results/synthetic_baseline/report.md`](../results/synthetic_baseline/report.md).

## 1. Executed configuration

- Data: `data/synthetic/returns.csv`, 3,500 synthetic observations,
  SHA-256 `4df1b68d…845a6f`, generated with `configs/synthetic_baseline.json`
  (seed 42).
- Rolling window: 1,000 observations. GARCH and GJR were refitted at every
  one of the **2,500** forecast origins (2003-11-03 to 2013-05-31).
- Confidence levels: 95% and 99%. Models: HS, GARCH(1,1)-FHS and
  GJR-GARCH(1,1,1)-FHS.
- Run type: `full` (the complete baseline, not a subset). Runtime was 52 s on
  Python 3.14.4, Windows 11.

## 2. Failures, exclusions and fit behaviour

- All 5,000 volatility fits (2,500 × 2 models) converged on the first
  attempt. No retries were needed, no warnings were raised and no forecasts
  failed.
- Common sample: 2,500 dates, with **0 excluded**.
- **Stationarity-boundary fits.** 737 GARCH fits and 642 GJR fits had
  estimated persistence ≥ 0.9999, which is at the constraint `arch` imposes.
  Every one of them came from a training window that contained a stress
  regime:
  - GJR: all 642 had `stress_1` in the window.
  - GARCH: 729 had `stress_1` and 8 had `stress_2` in the window.
  - No window without a stress regime produced a boundary fit.

  This matches the known effect of an unmodelled level shift in `ω`: a
  constant-parameter GARCH explains the shift as near-integrated persistence.
  The true persistence is 0.97.
- Median estimated persistence: 0.992 (GARCH) and 0.990 (GJR).

## 3. Full common-sample VaR/ES results (n = 2,500)

| model | c | exceptions (expected) | rate | Kupiec p | Christoffersen IND p | CC p | mean VaR | mean ES | mean QL |
|---|---|---|---|---|---|---|---|---|---|
| HS | 95% | 154 (125) | 6.16% | 0.010 | 0.001 | 0.0002 | 1.622 | 2.495 | 0.1325 |
| GARCH-FHS | 95% | 137 (125) | 5.48% | 0.278 | 0.842 | 0.541 | 1.601 | 2.152 | 0.1082 |
| GJR-FHS | 95% | 138 (125) | 5.52% | 0.240 | 0.520 | 0.406 | 1.610 | 2.157 | 0.1081 |
| HS | 99% | 40 (25) | 1.60% | 0.006 | 0.165 | 0.008 | 2.945 | 4.133 | 0.0475 |
| GARCH-FHS | 99% | 28 (25) | 1.12% | 0.554 | 0.426 | 0.610 | 2.455 | 3.159 | 0.0331 |
| GJR-FHS | 99% | 28 (25) | 1.12% | 0.554 | 0.426 | 0.610 | 2.450 | 3.165 | 0.0330 |

VaR, ES and QL are in percentage-point loss units. ES ≥ VaR held for all
15,000 forecasts.

Observations:

- **HS** is rejected at the 5% level by the unconditional-coverage and
  conditional-coverage tests at both confidence levels. At 95% it is also
  rejected by the independence test: its exceptions cluster (20 back-to-back
  pairs, compared with 7 and 6 for the FHS models).
- **Both FHS models** are not rejected by any of the three tests at either
  level. This does not show they are correct. At 99% the tests have limited
  power, with 25 expected exceptions.
- Both FHS variants have **lower average VaR and fewer exceptions** than HS,
  and lower quantile loss: about 18% lower at 95% and 30% lower at 99%. Their
  VaR is not uniformly higher; it is higher when volatility is high and lower
  when it is low.
- **GARCH-FHS and GJR-FHS are practically indistinguishable** on VaR coverage
  and quantile loss in this sample. Their 99% exception sequences have
  identical counts and identical test statistics.
- **Descriptive ES comparison.** On exception days, the mean realised loss
  exceeds the mean ES forecast slightly for every model (99%: HS 3.64 vs
  3.61, GARCH 3.13 vs 3.00, GJR 3.18 vs 3.08). This is descriptive only, not
  a formal ES backtest.

## 4. Stress and non-stress regimes (common sample)

| regime (n) | c | HS exc. | GARCH-FHS exc. | GJR-FHS exc. | expected |
|---|---|---|---|---|---|
| normal (2,300) | 95% | 117 | 115 | 116 | 115 |
| stress_1 (100) | 95% | 16 | 9 | 9 | 5 |
| stress_2 (100) | 95% | 21 | 13 | 13 | 5 |
| normal (2,300) | 99% | 24 | 20 | 20 | 23 |
| stress_1 (100) | 99% | 8 | 4 | 4 | 1 |
| stress_2 (100) | 99% | 8 | 4 | 4 | 1 |

Mean VaR inside stress regimes (95%):

| regime | HS | GARCH-FHS | GJR-FHS |
|---|---|---|---|
| stress_1 | 1.23 | 2.40 | 2.36 |
| stress_2 | 1.72 | 2.74 | 2.82 |

- Inside the stress regimes, HS VaR barely moves, because only a small part
  of its 1,000-day window is stressed. It has roughly 3–8 times the expected
  number of exceptions.
- The FHS models adapt but **still have more exceptions than nominal** inside
  stress (9–13 vs 5 at 95%; 4 vs 1 at 99%). Within-regime Kupiec p-values
  range from 0.002 to 0.097. The volatility filter reacts with a lag after
  the onset of an abrupt `ω` shift, as the stress close-up chart shows.
- These are 100-observation samples with 1–5 expected exceptions. The
  within-stress p-values have very low power and are descriptive at best.
- **After stress, HS becomes very conservative.** While the stress
  observations stay in its 1,000-day window, HS VaR stays high. In 2011-01 to
  2012-08 (435 forecasts, 21.75 expected at 95%), HS had 1 exception versus 21
  for each FHS model. The overall HS coverage statistics mix
  under-coverage during stress with over-coverage afterwards. This is why the
  overall exception count alone is a poor ranking criterion.
- In the normal regime, all three methods are close to nominal.

## 5. Volatility forecasts versus true synthetic volatility

| model | regime | RMSE σ (pp) | mean forecast σ | mean true σ | QLIKE |
|---|---|---|---|---|---|
| GARCH-FHS | all | 0.144 | 0.975 | 0.966 | 0.0277 |
| GJR-FHS | all | 0.124 | 0.979 | 0.966 | 0.0216 |
| GARCH-FHS | normal | 0.114 | 0.923 | 0.888 | 0.0167 |
| GJR-FHS | normal | 0.091 | 0.927 | 0.888 | 0.0094 |
| GARCH-FHS | stress_1 | 0.326 | 1.473 | 1.755 | 0.162 |
| GJR-FHS | stress_1 | 0.331 | 1.452 | 1.755 | 0.162 |
| GARCH-FHS | stress_2 | 0.331 | 1.665 | 1.977 | 0.147 |
| GJR-FHS | stress_2 | 0.286 | 1.720 | 1.977 | 0.162 |

- GJR has lower σ RMSE and QLIKE overall and in the normal regime. That is
  consistent with the data being generated by a GJR process.
- Inside stress, both filters **under-forecast** volatility on average (by
  about 0.3 pp) and neither is uniformly better. GARCH has slightly lower RMSE
  in `stress_1`. GJR has lower RMSE but higher QLIKE in `stress_2`.
- GJR's better volatility accuracy did **not** produce a visible VaR
  coverage difference against GARCH-FHS in this sample. The empirical
  residual distribution partly absorbs misspecification of the filter.

## 6. Residual diagnostics (in-sample, first and last windows only)

The standardised residuals have variance ≈ 1.0 and positive excess kurtosis
(0.9–1.3). With Ljung–Box at lags 10 and 20, no test on `z` rejects at 5%.
On `z²`, the smallest p-value is 0.052 (GJR, first window, lag 20). These
describe two in-sample windows only and are not evidence of i.i.d. shocks or
of out-of-sample accuracy.

## 7. Hypothesis assessment

Working hypothesis: *volatility-filtered HS gives better-calibrated one-day
VaR than plain HS when volatility changes, and GJR filtering helps further
because of asymmetric shocks.*

- **FHS versus HS: supported on this synthetic data.** HS fails coverage and
  independence tests and has higher quantile loss. Both FHS variants pass the
  full-sample tests, have lower quantile loss, and have fewer stress-regime
  exceptions with lower average capital.
- **GJR versus GARCH filtering: not supported for VaR/ES, mixed for
  volatility.** The VaR backtests are essentially identical. GJR tracks the
  true volatility better outside stress, but not consistently inside it.
- **Stress performance: mixed.** FHS improves greatly on HS, but still
  under-covers at the onset of the abrupt regime shifts.

## 8. Evidence limits

- **Synthetic, not market, evidence.** The data come from a GJR-GARCH process
  with Student-t shocks. That design may favour the related filters, so
  synthetic success says nothing about superiority on real markets. The
  single-asset SPY example in [`market_results.md`](market_results.md) is
  a separate, limited check.
- One simulated path (seed 42). Results will vary across seeds, and no
  Monte Carlo study over many paths was run.
- One asset, one-day horizon, zero mean, Gaussian QML. Rolling-window
  dependence means the χ² p-values are approximate.
- 99% tests and all stress-regime tests have low power.
- ES is compared descriptively only, with no formal ES backtest.
