# Allocation Probability Test

Compares inventory allocation policies under uncertain demand using a 100-store Negative Binomial simulation.

## Policies compared

| Policy | How it allocates |
|---|---|
| **Mean-share** | Proportional to forecast means via WOS-balancing. Each unit goes to the store with the lowest alloc/mean ratio. |
| **Volatility-aware** | Greedy by marginal expected sales. Each unit goes to the store where P(demand >= alloc+1) is highest, using exact NB tail probabilities. |

## Demand model

**Store means** are drawn from a Pareto distribution:

    mean_i = mean_min * (1 + Pareto(alpha)),  clipped to [mean_min, mean_max]

Smaller alpha = heavier tail = more extreme high-mean stores.

**Target store mean** (optional): when set, all means are rescaled so their average equals this value while preserving the Pareto shape.

**Variance-to-mean ratio (VMR):**

    VMR_i = 1 + coeff_i * mean_i

- Planning uses a fixed coeff = 0.1 for all stores.
- Realized coeff per store: `max(0, 0.1 + error_abs * u_i)` where `u_i ~ U(-1, 1)`.
- When VMR = 1, demand falls back to Poisson.

**Realized demand bias** (optional): `realized_mean_i = mean_i * bias_multiplier`
- 1.0 = no bias, 1.1 = +10% demand, 0.8 = -20% demand.

**Constraints:** each store receives at least 1 unit.

## Mean-share controls

- **WOS mode**: `before` uses `alloc/mean`, `after` uses `(alloc+1)/mean` when picking the next store.
- **Max WOS cap** (CLI only): soft cap on `alloc/mean` per store.

## Visualizations

- **Allocation CDF by Policy** — cumulative distribution of units allocated per store, one line per policy (Mean-share vs Volatility-aware).
- **Store Mean CDF** — cumulative share of stores with mean demand <= x.

## Statistical significance

Sold gain significance is tested via a paired daily test:

- `d_t = sold_vol_t - sold_mean_t` for each simulated day
- Reports: mean daily gain, 95% CI, two-sided p-value (normal approx)
- Significant at 5% if the CI excludes 0

## Run

```bash
# Web UI
.venv/bin/streamlit run streamlit_app.py

# CLI
.venv/bin/python negative_binomial_100_stores.py --stores 100 --pareto-alpha 1.15 --total-factor 2.1 --target-store-mean 5.0
```
