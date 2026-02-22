# Allocation Probability Simulation

Compares two inventory allocation policies across stores:
1. **Mean-share** — allocates proportional to forecast mean (WOS-balancing)
2. **Volatility-aware** — greedy allocation maximizing marginal expected sales using model-based tail probabilities

## Files

| File | Purpose |
|---|---|
| `negative_binomial_100_stores.py` | Core 100-store NB simulation (CLI + library) |
| `streamlit_app.py` | Interactive web UI (Streamlit) |

## Key Functions (`negative_binomial_100_stores.py`)

- `generate_store_means()` — Pareto-distributed store means, clipped to [mean_min, mean_max], optional `target_store_mean` rescaling
- `build_vmr()` — VMR = 1 + coeff * mean (NB dispersion parameter r = mean/(VMR-1) = 1/coeff)
- `sample_nb_demands()` — NB samples parameterized via mean and VMR
- `proportional_integer_allocation()` — WOS-balancing allocator with optional max-WOS cap, "before"/"after" mode
- `optimize_volatility_aware()` — greedy by P(D_i >= alloc_i + 1) from exact NB tail probabilities
- `evaluate()` — computes avg sold/lost/leftover, fill rate, stockout probability
- `summarize_by_mean_bins()` — groups stores into mean bins for comparison

## Demand Model

- Store means: Pareto(alpha) scaled by mean_min, clipped to [mean_min, mean_max], optionally rescaled to a target average
- VMR: `1 + coeff * mean` where planning coeff = 0.1
- Realized coeff: `max(0, 0.1 + coef_error_abs * u)`, u ~ U(-1,1)
- Realized means: `planned_mean * realized_demand_bias`
- NB parameterization: r = mean/(VMR-1), p = r/(r+mean)

## Running

```bash
# CLI
.venv/bin/python negative_binomial_100_stores.py --stores 100 --pareto-alpha 1.15 --total-factor 2.1

# Web UI
.venv/bin/streamlit run streamlit_app.py
```

## Python Environment

Always use `.venv/` in the project directory. Never use `--break-system-packages`.
