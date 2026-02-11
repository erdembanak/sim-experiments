#!/usr/bin/env python3
"""100-store allocation simulation under Negative Binomial demand.

Demand generation:
- mean_i ~ Uniform(0, 5)
- VMR_i = max(1, 0.9 * mean_i^0.9)

Policies compared:
1) Mean-share allocation (proportional to means)
2) Volatility-aware allocation (greedy by empirical marginal sell probability)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Metrics:
    avg_demand: float
    avg_sold: float
    avg_lost: float
    avg_leftover: float
    fill_rate: float
    stockout_any: float


@dataclass(frozen=True)
class Significance:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


def proportional_integer_allocation(total_units: int, weights: np.ndarray) -> np.ndarray:
    weight_sum = float(weights.sum())
    if total_units <= 0 or weight_sum <= 0:
        return np.zeros_like(weights, dtype=int)

    raw = total_units * (weights / weight_sum)
    alloc = np.floor(raw).astype(int)
    remainder = total_units - int(alloc.sum())

    if remainder > 0:
        frac = raw - alloc
        idx = np.argsort(-frac)[:remainder]
        alloc[idx] += 1

    return alloc


def generate_store_params(
    n_stores: int,
    mean_low: float,
    mean_high: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.uniform(mean_low, mean_high, size=n_stores)
    vmr = np.maximum(1.0, 0.9 * np.power(means, 0.9))
    return means, vmr


def sample_nb_demands(
    means: np.ndarray,
    vmr: np.ndarray,
    trials: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_stores = means.size
    demands = np.zeros((trials, n_stores), dtype=np.int32)

    for i, (mu, vmr_i) in enumerate(zip(means, vmr)):
        if mu <= 0:
            continue
        if vmr_i <= 1.0 + 1e-12:
            # Poisson limit case when variance equals mean.
            demands[:, i] = rng.poisson(mu, size=trials)
            continue

        # For NB(n, p): mean = n(1-p)/p and VMR = 1 + mean/n.
        n_param = mu / (vmr_i - 1.0)
        p_param = n_param / (n_param + mu)
        demands[:, i] = rng.negative_binomial(n=n_param, p=p_param, size=trials)

    return demands


def optimize_volatility_aware(
    opt_demands: np.ndarray,
    total_units: int,
) -> np.ndarray:
    n_trials, n_stores = opt_demands.shape
    alloc = np.zeros(n_stores, dtype=np.int32)

    # tails[i, k] estimates P(D_i >= k+1), for k = 0..total_units-1
    tails = np.zeros((n_stores, total_units), dtype=np.float64)

    for i in range(n_stores):
        clipped = np.clip(opt_demands[:, i], 0, total_units)
        counts = np.bincount(clipped, minlength=total_units + 1)
        cc = np.cumsum(counts[::-1])[::-1]
        tails[i, :] = cc[1:] / n_trials

    for _ in range(total_units):
        marginal = np.where(alloc < total_units, tails[np.arange(n_stores), alloc], -1.0)
        best_store = int(np.argmax(marginal))
        alloc[best_store] += 1

    return alloc


def evaluate(demands: np.ndarray, alloc: np.ndarray) -> Metrics:
    sold_by_store = np.minimum(demands, alloc)
    sold_per_day = sold_by_store.sum(axis=1)
    demand_per_day = demands.sum(axis=1)

    total_units = int(alloc.sum())
    lost_per_day = demand_per_day - sold_per_day
    left_per_day = total_units - sold_per_day

    stockout_any = np.mean((demands > alloc).any(axis=1))
    fill_rate = sold_per_day.sum() / max(1.0, demand_per_day.sum())

    return Metrics(
        avg_demand=float(demand_per_day.mean()),
        avg_sold=float(sold_per_day.mean()),
        avg_lost=float(lost_per_day.mean()),
        avg_leftover=float(left_per_day.mean()),
        fill_rate=float(fill_rate),
        stockout_any=float(stockout_any),
    )


def paired_significance(sold_a: np.ndarray, sold_b: np.ndarray) -> Significance:
    diff = sold_b - sold_a
    n = diff.size
    if n < 2:
        return Significance(0.0, 0.0, 0.0, 1.0, False)

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0.0
    if se == 0:
        p_value = 1.0
    else:
        z = mean_diff / se
        p_value = math.erfc(abs(z) / math.sqrt(2.0))

    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se
    significant = not (ci_low <= 0 <= ci_high)
    return Significance(mean_diff, ci_low, ci_high, p_value, significant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="100-store Negative Binomial allocation comparison",
    )
    parser.add_argument("--stores", type=int, default=100)
    parser.add_argument("--mean-low", type=float, default=0.0)
    parser.add_argument("--mean-high", type=float, default=5.0)
    parser.add_argument(
        "--total-factor",
        type=float,
        default=0.9,
        help="Total inventory = round(total mean demand * total-factor)",
    )
    parser.add_argument("--eval-trials", type=int, default=120_000)
    parser.add_argument("--opt-trials", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    means, vmr = generate_store_params(
        n_stores=args.stores,
        mean_low=args.mean_low,
        mean_high=args.mean_high,
        seed=args.seed,
    )

    total_mean = float(means.sum())
    total_units = int(round(total_mean * args.total_factor))

    alloc_mean = proportional_integer_allocation(total_units, means)

    opt_demands = sample_nb_demands(
        means=means,
        vmr=vmr,
        trials=args.opt_trials,
        seed=args.seed + 100,
    )
    alloc_vol = optimize_volatility_aware(opt_demands, total_units)

    eval_demands = sample_nb_demands(
        means=means,
        vmr=vmr,
        trials=args.eval_trials,
        seed=args.seed + 200,
    )

    metrics_mean = evaluate(eval_demands, alloc_mean)
    metrics_vol = evaluate(eval_demands, alloc_vol)
    sold_mean_daily = np.minimum(eval_demands, alloc_mean).sum(axis=1)
    sold_vol_daily = np.minimum(eval_demands, alloc_vol).sum(axis=1)
    sig = paired_significance(sold_mean_daily, sold_vol_daily)

    sold_gain = metrics_vol.avg_sold - metrics_mean.avg_sold
    lost_reduction = metrics_mean.avg_lost - metrics_vol.avg_lost
    fill_gain_pp = 100.0 * (metrics_vol.fill_rate - metrics_mean.fill_rate)

    print("Negative Binomial Allocation Test (100 Stores)")
    print("=" * 88)
    print(f"Stores:                   {args.stores}")
    print(f"Mean range:               [{args.mean_low}, {args.mean_high}]")
    print("VMR rule:                 max(1, 0.9 * mean^0.9)")
    print(f"Total expected demand:    {total_mean:.2f}")
    print(f"Total allocation budget:  {total_units}")
    print(f"Evaluation trials:        {args.eval_trials:,}")
    print()

    print(
        f"{'Policy':<28}{'Avg Sold':>12}{'Avg Lost':>12}"
        f"{'Avg Left':>12}{'Fill Rate':>12}{'Stockout Any':>14}"
    )
    print("-" * 88)
    print(
        f"{'Mean-share':<28}{metrics_mean.avg_sold:>12.2f}{metrics_mean.avg_lost:>12.2f}"
        f"{metrics_mean.avg_leftover:>12.2f}{100*metrics_mean.fill_rate:>11.2f}%"
        f"{100*metrics_mean.stockout_any:>13.2f}%"
    )
    print(
        f"{'Volatility-aware':<28}{metrics_vol.avg_sold:>12.2f}{metrics_vol.avg_lost:>12.2f}"
        f"{metrics_vol.avg_leftover:>12.2f}{100*metrics_vol.fill_rate:>11.2f}%"
        f"{100*metrics_vol.stockout_any:>13.2f}%"
    )

    print("\nComparison")
    print("-" * 88)
    print(f"Sold gain (vol - mean):   {sold_gain:.2f}")
    print(f"Lost reduction:           {lost_reduction:.2f}")
    print(f"Fill-rate gain:           {fill_gain_pp:.2f} pp")
    print("\nSignificance (paired by day)")
    print("-" * 88)
    print(f"Mean sold gain:           {sig.mean_diff:.6f}")
    print(f"95% CI:                   [{sig.ci_low:.6f}, {sig.ci_high:.6f}]")
    print(f"p-value:                  {sig.p_value:.10f}")
    print(f"Significant at 5%:        {sig.significant}")

    top_diff_idx = np.argsort(np.abs(alloc_vol - alloc_mean))[::-1][:10]
    print("\nTop 10 stores by allocation difference (vol - mean)")
    print("-" * 88)
    print(f"{'Store':>8}{'Mean':>10}{'VMR':>10}{'MeanAlloc':>12}{'VolAlloc':>10}{'Diff':>8}")
    for idx in top_diff_idx:
        diff = int(alloc_vol[idx] - alloc_mean[idx])
        print(
            f"{idx:>8}{means[idx]:>10.3f}{vmr[idx]:>10.3f}{int(alloc_mean[idx]):>12}"
            f"{int(alloc_vol[idx]):>10}{diff:>8}"
        )


if __name__ == "__main__":
    main()
