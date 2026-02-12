# Allocation Probability Test

This project compares allocation policies under uncertain demand.

## What is included

- `probability_test.py`: 2-store simulation with **normal** demand.
- `negative_binomial_100_stores.py`: 100-store simulation with **negative binomial** demand.
- `streamlit_app.py`: interactive app with 3 tabs:
  - `README`
  - `Normal (2-store)`
  - `Negative Binomial (100 stores)`

## Policies compared

- **Mean-share**: allocate proportional to expected means.
- **Volatility-aware**: optimize allocation using model-based marginal expected sell.

## Demand assumptions

### Normal (2-store)

- Store demand sampled from normal distribution.
- Rounded to non-negative integers.

### Negative Binomial (100 stores)

- Mean generation uses Pareto:
  - `mean_i = mean_min * (1 + Pareto(alpha))`, clipped to `[mean_min, mean_max]`
  - smaller `alpha` gives a heavier tail (more extreme high-mean stores)
- Planning: `VMR_i = 1 + 0.1 * mean_i` (no error).
- Realized demand: `coeff_i = max(0, 0.1 + error_abs * u_i)`, where `u_i ~ Uniform(-1, 1)`, and `VMR_i = 1 + coeff_i * mean_i`.
- If `VMR = 1`, demand uses Poisson limit.
- Allocation constraint: each store gets at least `1` unit.
- Optional mean-share control: `mean_share_max_wos` (soft cap on `alloc/mean` in mean-share allocation).
- Mean-share WOS mode: choose `after` (`(alloc+1)/mean`) or `before` (`alloc/mean`) when selecting the next unit.

## Statistical significance of sold gain

Both tabs report significance for sold gain using a **paired daily test**:

- Daily difference: `d_t = sold_vol_t - sold_mean_t`.
- Reported values:
  - mean daily gain
  - 95% confidence interval
  - two-sided p-value (normal approximation)
  - whether significant at 5%

Interpretation:

- If 95% CI excludes `0`, sold gain is statistically significant at ~5%.

## Run

```bash
.venv/bin/streamlit run streamlit_app.py
```

## CLI scripts

Normal model:

```bash
.venv/bin/python probability_test.py
```

Negative binomial model:

```bash
.venv/bin/python negative_binomial_100_stores.py
```

Note: for the negative binomial model, total allocation must be at least number of stores due to the minimum `1` unit/store rule.
