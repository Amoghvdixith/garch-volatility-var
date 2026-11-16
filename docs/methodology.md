# Methodology

This document specifies exactly what the code computes. Code references are
given as `module.function`.

## 1. Units and loss convention

- Returns are **percentage-point log returns**: `r_t = 100 * log(P_t / P_{t-1})`.
  A 1% log return is stored as `1.0`.
- Position: long, fixed notional, one asset, one-day horizon.
- Primary loss: `loss_pct = -return_pct`. This is the **negative log return in
  percentage points**, not an exact monetary loss.
- Exact monetary loss for notional `N`: `N * (1 - exp(r/100))`
  (`historical.monetary_losses`). Because this map is monotone, it can be
  applied to every scenario return before taking VaR/ES. It is provided as a
  utility; the reported tables use `loss_pct`. `N * (-r/100)` is only a
  linear approximation and is never labelled as exact.
- Confidence `c ∈ {0.95, 0.99}`, tail probability `p = 1 - c`.

## 2. Volatility models

Zero conditional mean for every method (`r_t = ε_t`), a deliberate
simplification: daily mean returns are small relative to volatility and
estimating them adds noise.

GARCH(1,1):

```
σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²
```

GJR-GARCH(1,1,1):

```
σ_t² = ω + (α + γ · 1[ε_{t-1} < 0]) ε_{t-1}² + β σ_{t-1}²
```

`γ` (the `o=1` term in `arch`) adds extra variance after negative shocks.

### Estimation (third-party: `arch`)

Parameters are estimated by the `arch` package (Sheppard) with:

```python
arch_model(y, mean="zero", vol="garch", p=1, o=0 or 1, q=1, power=2.0,
           dist="normal", rescale=False)
```

- `dist="normal"` gives **Gaussian quasi-maximum likelihood (QML)**
  estimates. QML is consistent for the variance parameters under correct
  variance specification even when shocks are not Gaussian (Bollerslev &
  Wooldridge, 1992). It does **not** make the FHS tails Gaussian, because FHS
  uses the empirical standardised residuals (section 4).
- `rescale=False`: inputs are already in percentage points (daily variance of
  order 1), which is the scale `arch` would otherwise rescale towards; fixing
  it keeps parameters in documented units. A test asserts `res.scale == 1`.
- `arch` names map as `omega→ω`, `alpha[1]→α`, `gamma[1]→γ`, `beta[1]→β`
  (`volatility.ARCH_PARAM_MAP`, tested).
- `arch` initialises the in-sample recursion with its own backcast of the
  first residuals. Tests verify that every later in-sample variance follows
  the recursion above and that `arch`'s one-step forecast equals
  `ω + (α + γ 1[ε_T<0]) ε_T² + β σ_T²`.
- `arch` imposes `α + γ/2 + β ≤ 1` during optimisation. Estimates at that
  bound are reported as "at the stationarity boundary" (persistence ≥ 0.9999)
  rather than hidden.

Recorded per fit (`fit_diagnostics.csv`): parameters, convergence flag,
attempts, warnings, log likelihood, persistence (`α+β` or `α+γ/2+β`, the latter
assuming symmetric shocks), boundary flag, one-step σ forecast, residual counts.

### Failure policy

`volatility.fit_volatility` makes one attempt with `arch` defaults. If it
raises, fails to converge, or gives a non-positive/non-finite forecast, it
retries **once on the same window** with documented starting values
(`ω = 0.05·var(y)`, `α = 0.05` or `0.03`, `γ = 0.06`, `β = 0.90`) and
`maxiter = 1000`. If that fails, the forecast is marked `fit_failed` and left
empty. No other model is substituted.

## 3. Strict out-of-sample timing

For forecast position `t` (`forecasting.run_rolling_forecasts`):

1. The training window is positions `t-1000 … t-1`, the same for every model.
2. GARCH and GJR are refitted on that window at **every** origin.
3. `σ_t` is the one-step forecast from information through `t-1`.
4. Scenarios use only the training-window residuals or returns.
5. The realised `r_t` is attached afterwards, for evaluation only.

`forecasting.forecast_from_window` receives only the training array, so
`r_t` and later data cannot enter the forecast by construction. Tests confirm
that perturbing future returns, or the realised `r_t`, leaves the forecast
for `t` unchanged.

Cross-model tables use the **common sample**: dates on which every model has
a valid forecast. Excluded dates and per-model failure counts are written to
`run_metadata.json`.

## 4. Historical simulation and FHS

**Plain HS** (`historical.historical_simulation`): the scenario losses are the
1,000 training-window losses `-r_i`.

**FHS** (`filtered_historical.py`):

1. `z_i = ε_i / σ_i` for every training observation, where `σ_i` is the fitted
   in-sample conditional volatility. Only non-finite values or `σ_i ≤ 0` are
   dropped and counted. With `arch`'s backcast, `σ_i` is defined from the
   first observation, so in the baseline no residuals were dropped.
2. `z_i` is **not** recentred or rescaled to unit variance.
3. Scenario returns are `σ_t · z_i` (no extra factor of 100). Losses are
   `-σ_t · z_i`.
4. VaR and ES use the same empirical convention as HS (section 5).
5. A minimum of 750 valid residuals is required, otherwise the status is
   `insufficient_residuals`.

All `n` residual scenarios are used with equal weight. This is the exact,
deterministic one-step FHS distribution, so no Monte Carlo resampling is
added (resampling would only add simulation noise).

Things to keep separate: conditional *variance* `σ_t²` versus *volatility*
`σ_t`; standardised residuals `z` versus raw returns `r`; FHS (empirical `z`)
versus parametric Gaussian VaR (`σ_t · Φ⁻¹(c)`), which is **not** used here.

## 5. Empirical VaR and ES convention

Every scenario loss has mass `1/n` (`historical.py`).

- **VaR**: the inverse empirical CDF at `c`. Sort losses ascending,
  `L_(1) ≤ … ≤ L_(n)`, and take `VaR_c = L_(k)` with `k = ⌈n c⌉`.
- **ES**: the average over the worst probability mass `p`. With `m = n p`,
  the `⌊m⌋` largest losses get full weight and the next one gets weight
  `m − ⌊m⌋`:

```
ES_c = ( Σ_{j=1}^{⌊m⌋} L_[j] + (m − ⌊m⌋) L_[⌊m⌋+1] ) / m     (L_[j] descending)
```

This equals `VaR_c + E[(L − VaR_c)⁺]/p` under the empirical measure
(Acerbi & Tasche, 2002; Rockafellar & Uryasev, 2002), so `ES ≥ VaR` always
holds. For `n = 1000`: at 95%, `k = 950` and the ES tail is 50 losses; at
99%, `k = 990` and the tail is 10 losses. A tolerance of `1e-9` makes
`1000 × 0.95` count as an integer. Results are never clipped to be positive.

**Exceptions** are strict: `loss_t > VaR_t`. A tie (`loss = VaR`) is not an
exception. Ties have probability zero for continuous returns but can occur
with discretised prices. With a finite window, VaR is an order statistic, so
it moves in jumps. This is clearly visible for HS.

## 6. Evaluation metrics (`metrics.py`)

- Counts: `n`, exceptions `x`, expected `n p`, observed rate `x/n`, nominal
  rate `p`, and the difference.
- Mean VaR, mean ES, mean loss and mean ES on exception days, and mean excess
  loss `loss − VaR` on exception days.
- **Quantile (pinball) loss** of the loss quantile:
  `QL = (c − 1[loss < VaR]) (loss − VaR)`. Lower is better. It is a
  consistent scoring function for the `c`-quantile.
- Longest run of consecutive exceptions.
- Rolling exception rate: a trailing 250-forecast mean, labelled at the window
  end, with no partial windows.
- Synthetic only: σ RMSE and MAE versus the true `σ_t`, and normalised QLIKE
  `mean(h/f − log(h/f) − 1)` with `h` = true and `f` = forecast variance.
  This is impossible for market data, where `σ_t` is unobserved.

## 7. Coverage tests (`backtesting.py`)

Log-likelihoods use `scipy.special.xlogy` (so `0·log 0 = 0`). p-values come
from `scipy.stats.chi2.sf`.

- **Kupiec (1995) unconditional coverage**:
  `LR_uc = −2[(n−x)log(1−p) + x log p − (n−x)log(1−π̂) − x log π̂]`, χ²(1),
  `π̂ = x/n`. Defined for `x = 0` and `x = n`.
- **Christoffersen (1998) independence**: first-order Markov alternative using
  the transition counts `n_ij` over the `n−1` consecutive pairs.
  `LR_ind = −2[L(π̂₁) − L(π̂₀₁, π̂₁₁)]`, χ²(1).
- **Conditional coverage**: `LR_cc = −2[L(p) − L(π̂₀₁, π̂₁₁)]`, χ²(2), on the
  same `n−1` transitions. It therefore equals `LR_uc` (on those transitions)
  plus `LR_ind` exactly, which is tested.
- **Undefined cases**: if a row of the transition table has no observations
  (no exceptions, an exception only on the final day, or no non-exception
  days), the independence alternative is not identified. Both Christoffersen
  statistics are then reported as `undefined`, with NaN values and a reason;
  no p-value is invented.

Caveats:

- A non-rejection does not prove a model is correct.
- With 25 expected 99% exceptions in 2,500 days (and 1 per 100-day stress
  regime), the tests have limited power.
- Rolling estimation, overlapping windows and parameter uncertainty mean the
  asymptotic χ² reference distributions are approximations.
- A model should not be ranked by the fewest exceptions: an over-conservative
  VaR is also miscalibrated, and quantile loss penalises it.

**ES is not formally backtested.** VaR coverage tests are not ES tests. ES is
compared descriptively: mean ES, and mean loss versus mean ES on exception
days.

## 8. Synthetic data design (`synthetic.py`)

GJR-GARCH(1,1,1) with standardised Student-t shocks
`z_t = T_t √((ν−2)/ν)`, where `T_t ~ t(ν)`, so `Var(z_t) = 1`:

| parameter | value |
|---|---|
| μ | 0 |
| ω | 0.02 |
| α | 0.05 |
| γ | 0.08 |
| β | 0.88 |
| ν | 8 |
| burn-in | 1,000 |
| retained | 3,500 |
| seed | 42 (`numpy.random.default_rng`) |

- Admissibility: `ω > 0`, `α, β ≥ 0`, `α + γ ≥ 0`, `ν > 2`.
- Finite-variance condition for symmetric shocks: `α + γ/2 + β = 0.97 < 1`.
- Long-run variance: `ω / 0.03 ≈ 0.667` (σ ≈ 0.82 pp).
- Initialisation: `σ₀²` equals that long-run variance at the first burn-in
  observation. The burn-in is then discarded.
- Dates: a synthetic Monday-to-Friday index starting 2000-01-03
  (`pandas.bdate_range`). It is **not** an exchange calendar and has no
  holidays.

## 9. Stress regimes (fixed before any forecasting)

| data | regime | definition |
|---|---|---|
| Synthetic | `stress_1` | retained indices 1,400–1,499 inclusive (2005-05-16 to 2005-09-30) |
| Synthetic | `stress_2` | retained indices 2,500–2,599 inclusive (2009-08-03 to 2009-12-18) |
| Market (optional) | `covid_2020` | 2020-02-20 to 2020-04-30 |
| Market (optional) | `year_2022` | 2022-01-01 to 2022-12-31 |

In the synthetic stress regimes, `ω_t = 9ω`. This temporary regime shift
deliberately departs from a single constant-parameter GJR process, so no
fitted model is exactly correctly specified across the sample.

The stress close-up chart uses a rule fixed in advance: the first configured
regime inside the forecast period.

## References

- Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487–1503.
- Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews*, 11(2), 143–172.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Hull, J., & White, A. (1998). Incorporating volatility updating into the historical simulation method for value-at-risk. *Journal of Risk*, 1(1), 5–19.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Rockafellar, R. T., & Uryasev, S. (2002). Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
- Sheppard, K. *arch* Python package documentation: https://arch.readthedocs.io/
