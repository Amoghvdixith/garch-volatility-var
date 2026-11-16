# Learning guide

A plain-language companion to the project. The formal details are in
[`methodology.md`](methodology.md).

## Volatility clustering

Calm days tend to follow calm days, and turbulent days follow turbulent days.
Big moves cluster together even though their *direction* is hard to predict.
The synthetic series shows this clearly (see
`results/synthetic_baseline/figures/returns_and_true_volatility.png`).

## Conditional variance versus volatility

- **Conditional variance** `σ_t²` is the expected squared return for tomorrow,
  given everything known today.
- **Volatility** `σ_t` is its square root, in the same units as returns. With
  percentage-point returns, `σ_t = 1.2` means a typical daily move of about
  1.2%.
- GARCH models write the recursion for the variance. VaR scales with the
  volatility.

## GARCH versus GJR-GARCH

- **GARCH(1,1)**: tomorrow's variance = a constant + a fraction of today's
  squared shock + a fraction of today's variance. A +3% day and a −3% day
  raise variance equally.
- **GJR-GARCH**: adds an extra term (γ) only after **negative** shocks. This
  captures the "leverage effect" seen in equity markets, where falls raise
  volatility more than rises do.
- **Persistence** (α + β, or α + γ/2 + β): how slowly shocks fade. Values
  near 1 mean very long memory.

## VaR versus ES

- **VaR at 99%**: a loss threshold that should be exceeded on only about 1%
  of days. It says nothing about *how bad* those days are.
- **ES at 99%**: the average loss over the worst 1% of outcomes. It looks
  into the tail, so it is always at least as large as VaR.

## Historical simulation versus filtered historical simulation

- **HS**: "tomorrow looks like a random day from the last 1,000 days." It is
  simple, but it treats a calm day from three years ago as just as relevant as
  yesterday. It reacts slowly when volatility jumps, and it stays high long
  after a crisis has passed. Both effects appear in this project's results.
- **FHS** (Barone-Adesi et al.; Hull & White):
  1. Divide each past return by the volatility of its day. This gives the
     standardised shock `z`.
  2. Multiply every `z` by **tomorrow's** forecast volatility.
  3. Read VaR and ES off these rescaled scenarios.

  The shape of the tails comes from history, and the scale comes from today's
  conditions.

## Why standardised residuals

Raw returns from calm and turbulent periods are not comparable: a 2% move is
extreme on a calm day and ordinary in a crisis. Dividing by `σ_i` puts every
day on the same footing ("how many standard deviations was this?"). Those
standardised shocks are closer to identically distributed, so reusing them
for tomorrow is more defensible.

## Why forecast alignment matters

A VaR forecast for day `t` must use only information available at the end of
day `t−1`. If day `t`'s own return leaks into the model, through fitting,
residual quantiles or scaling, the backtest will look better than anything
achievable in practice. This project refits every model using only the
previous 1,000 days, and tests that changing day `t` or later never changes
the forecast for day `t`.

## Why few exceptions do not mean a better model

A model that sets VaR at 50% would almost never be breached, but it would be
useless and would waste capital. The goal is the **right** exception rate
(about 5% or about 1%), with exceptions that do not cluster. In this
project's synthetic results:

- HS had *no* advantage from being conservative after the stress periods. It
  went 435 days with 1 exception where about 22 were expected.
- It then breached far too often when the next stress began.

The quantile (pinball) loss rewards being close to the true quantile, not
just being high.

## Limits of short stress periods

Each synthetic stress regime has 100 days. At 99%, about 1 exception is
expected in 100 days, so observing 0, 1, 2 or 4 is weak evidence either way.
Statistical tests on such small samples have very little power. Use them as
descriptions, not verdicts.

## Interview questions

1. Why does FHS use standardised residuals instead of raw returns?
2. What is the difference between conditional and unconditional coverage?
3. Why can a VaR model with too *few* exceptions be rejected by the Kupiec
   test?
4. What does Christoffersen's independence test detect, and when is it
   undefined?
5. Why is Gaussian QML estimation acceptable even if the shocks have fat
   tails? Does it make FHS VaR Gaussian?
6. How does GJR-GARCH capture the leverage effect? What is its persistence
   condition?
7. Why is ES considered harder to backtest than VaR?
8. What goes wrong if you fit a GARCH model on the full sample and call the
   fitted values "out-of-sample"?
9. Why do structural breaks in volatility push GARCH persistence estimates
   towards 1? (Hint: see the boundary-fit counts in `docs/results.md`.)
10. Why is a synthetic GJR data-generating process a biased testbed for
    comparing GARCH and GJR filters?

## Self-check exercises

1. Losses are {1, 2, …, 20} and c = 0.95. Compute VaR and ES under this
   project's convention.
   *Answer: VaR = 19 (k = ⌈19⌉), ES = 20 (tail mass = 1 observation).*
2. Repeat for c = 0.90 and c = 0.875.
   *Answer: at 0.90, VaR = 18 and ES = 19.5. At 0.875, k = ⌈17.5⌉ = 18 so
   VaR = 18, and ES = (20 + 19 + 0.5·18)/2.5 = 19.2.*
3. GJR with ω = 0.02, α = 0.05, γ = 0.08, β = 0.88, σ_{t−1}² = 1 and
   ε_{t−1} = −2. What is σ_t²? And if ε_{t−1} = +2?
   *Answer: 0.02 + 0.13·4 + 0.88 = 1.42; and 0.02 + 0.05·4 + 0.88 = 1.10.*
4. Run the backtest with a different seed. Copy `configs/synthetic_baseline.json`
   to a new config and change `seed`, `data.path`, `data.metadata_path` and
   `output_dir`, so the baseline files are not overwritten. Then run the
   generate, backtest and report scripts. Do the main conclusions hold?
5. Re-run with a 500-day window in a new config. Also lower `min_residuals`
   to at most 500, otherwise every FHS forecast is marked
   `insufficient_residuals`. How do HS and FHS stress exceptions change?
