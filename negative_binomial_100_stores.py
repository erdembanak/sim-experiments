#!/usr/bin/env python3
"""100-store allocation simulation under Negative Binomial demand.

Demand generation:
- store means from a Pareto profile
- VMR_i = 1 + coeff * mean_i

Policies compared:
1) Mean-share allocation (proportional to means)
2) Volatility-aware allocation (greedy by model-based marginal expected sales)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

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


def proportional_integer_allocation(
    total_units: int,
    weights: np.ndarray,
    min_per_store: int = 1,
    max_wos: float | None = None,
    share_wos_mode: str = "before",
) -> np.ndarray:
    n_stores = int(weights.size)
    base_required = n_stores * min_per_store
    if total_units < base_required:
        raise ValueError(
            f"Total allocation ({total_units}) must be >= stores*min_per_store ({base_required})."
        )

    if total_units <= 0:
        return np.zeros_like(weights, dtype=int)

    if base_required == total_units:
        return np.full(n_stores, min_per_store, dtype=int)

    alloc = np.full(n_stores, min_per_store, dtype=int)
    remaining = total_units - base_required
    eps = 1e-12
    if share_wos_mode not in {"before", "after"}:
        raise ValueError("share_wos_mode must be 'before' or 'after'.")
    cap_units: np.ndarray | None = None
    if max_wos is not None and max_wos > 0:
        cap_units = np.maximum(min_per_store, np.ceil(weights * max_wos).astype(int))

    # WOS-balancing rule: add each next unit to the store that would still
    # have the lowest weeks-of-supply after receiving one more unit.
    # WOS proxy here is alloc / forecast_mean.
    for _ in range(remaining):
        if share_wos_mode == "after":
            ratio = np.where(weights > eps, (alloc + 1) / weights, np.inf)
        else:
            ratio = np.where(weights > eps, alloc / weights, np.inf)
        candidates = np.arange(n_stores)
        if cap_units is not None:
            under_cap = np.where(alloc < cap_units)[0]
            if under_cap.size > 0:
                candidates = under_cap
        k = int(candidates[np.argmin(ratio[candidates])])
        if not np.isfinite(ratio[k]):
            k = int(np.argmin(alloc))
        alloc[k] += 1

    return alloc


def generate_store_means(
    n_stores: int,
    seed: int,
    pareto_alpha: float = 1.5,
    mean_min: float = 0.5,
    mean_max: float = 50.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if pareto_alpha <= 0:
        raise ValueError("pareto_alpha must be > 0.")
    if mean_min <= 0:
        raise ValueError("mean_min must be > 0.")
    if mean_max <= mean_min:
        raise ValueError("mean_max must be > mean_min.")

    means = mean_min * (1.0 + rng.pareto(pareto_alpha, size=n_stores))
    return np.clip(means, mean_min, mean_max)


def build_vmr(means: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return 1.0 + coeff * means


def sample_realized_coeff(
    n_stores: int,
    seed: int,
    coef_error_abs: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    error = rng.uniform(-1.0, 1.0, size=n_stores)
    return np.maximum(0.0, 0.1 + error * coef_error_abs)


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


def _tail_probs_poisson(mu: float, max_k: int) -> np.ndarray:
    tail = np.zeros(max_k, dtype=float)
    pmf = math.exp(-mu)
    cdf = pmf
    for k in range(max_k):
        tail[k] = max(0.0, 1.0 - cdf)
        # next pmf for k+1
        pmf = pmf * mu / (k + 1)
        cdf += pmf
    return np.clip(tail, 0.0, 1.0)


def _tail_probs_nb(mu: float, vmr: float, max_k: int) -> np.ndarray:
    tail = np.zeros(max_k, dtype=float)
    if mu <= 0:
        return tail
    if vmr <= 1.0 + 1e-12:
        return _tail_probs_poisson(mu, max_k)

    # For NB(r, p): mean = r(1-p)/p and VMR = 1 + mean/r.
    r = mu / (vmr - 1.0)
    p = r / (r + mu)
    q = 1.0 - p

    pmf = p**r  # P(X=0)
    cdf = pmf
    for k in range(max_k):
        tail[k] = max(0.0, 1.0 - cdf)
        # recurrence: P(X=k+1) = P(X=k) * ((k+r)/(k+1)) * q
        pmf = pmf * ((k + r) / (k + 1)) * q
        cdf += pmf

    return np.clip(tail, 0.0, 1.0)


def optimize_volatility_aware(
    means: np.ndarray,
    vmr: np.ndarray,
    total_units: int,
    min_per_store: int = 1,
) -> np.ndarray:
    n_stores = int(means.size)
    base_required = n_stores * min_per_store
    if total_units < base_required:
        raise ValueError(
            f"Total allocation ({total_units}) must be >= stores*min_per_store ({base_required})."
        )

    alloc = np.full(n_stores, min_per_store, dtype=np.int32)
    remaining = total_units - base_required
    if remaining == 0:
        return alloc

    # tails[i, k] is exact (model-based) P(D_i >= k+1), for k = 0..total_units-1.
    tails = np.zeros((n_stores, total_units), dtype=np.float64)
    for i in range(n_stores):
        tails[i, :] = _tail_probs_nb(float(means[i]), float(vmr[i]), total_units)

    for _ in range(remaining):
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


def summarize_by_mean_bins(
    means: np.ndarray,
    realized_demand: np.ndarray,
    alloc_mean: np.ndarray,
    alloc_vol: np.ndarray,
    sold_mean_store: np.ndarray,
    sold_vol_store: np.ndarray,
) -> List[dict]:
    edges = np.array([0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, np.inf])
    labels = ["[0,1)", "[1,5)", "[5,10)", "[10,20)", "[20,30)", "[30,40)", "[40,50)", "[50,+)"]
    idx = np.digitize(means, edges, right=False) - 1

    rows: List[dict] = []
    for i, label in enumerate(labels):
        mask = idx == i
        if not np.any(mask):
            continue

        sold_mean_total = float(sold_mean_store[mask].sum())
        sold_vol_total = float(sold_vol_store[mask].sum())
        sold_gain = sold_vol_total - sold_mean_total
        sold_gain_pct = 100.0 * sold_gain / sold_mean_total if sold_mean_total > 0 else 0.0

        rows.append(
            {
                "bin": label,
                "stores": int(mask.sum()),
                "avg_mean": float(means[mask].mean()),
                "avg_realized_demand": float(realized_demand[mask].mean()),
                "alloc_mean_total": int(alloc_mean[mask].sum()),
                "alloc_vol_total": int(alloc_vol[mask].sum()),
                "sold_mean_total": sold_mean_total,
                "sold_vol_total": sold_vol_total,
                "sold_gain": sold_gain,
                "sold_gain_pct": sold_gain_pct,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="100-store Negative Binomial allocation comparison",
    )
    parser.add_argument("--stores", type=int, default=100)
    parser.add_argument(
        "--pareto-alpha",
        type=float,
        default=1.5,
        help="Pareto alpha shape for store mean generation (smaller = heavier tail).",
    )
    parser.add_argument("--mean-min", type=float, default=0.5)
    parser.add_argument("--mean-max", type=float, default=50.0)
    parser.add_argument(
        "--total-factor",
        type=float,
        default=0.9,
        help="Total inventory = round(total mean demand * total-factor)",
    )
    parser.add_argument("--eval-trials", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mean-share-max-wos",
        type=float,
        default=0.0,
        help="Soft WOS cap for mean-share allocator (<=0 disables).",
    )
    parser.add_argument(
        "--share-wos-mode",
        type=str,
        default="before",
        choices=["before", "after"],
        help="Use WOS before or after hypothetical +1 when choosing next unit.",
    )
    parser.add_argument(
        "--coef-error-abs",
        type=float,
        default=0.0,
        help=(
            "Amplitude e for realized coeff: coeff=max(0, 0.1 + e*u), "
            "u~Uniform(-1,1)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.coef_error_abs < 0:
        raise ValueError("--coef-error-abs must be >= 0.")

    means = generate_store_means(
        n_stores=args.stores,
        seed=args.seed,
        pareto_alpha=args.pareto_alpha,
        mean_min=args.mean_min,
        mean_max=args.mean_max,
    )
    coeff_plan = np.full(args.stores, 0.1, dtype=float)
    coeff_realized = sample_realized_coeff(
        n_stores=args.stores,
        seed=args.seed + 1,
        coef_error_abs=args.coef_error_abs,
    )
    vmr_plan = build_vmr(means, coeff_plan)
    vmr_realized = build_vmr(means, coeff_realized)

    total_mean = float(means.sum())
    total_units = int(round(total_mean * args.total_factor))

    alloc_mean = proportional_integer_allocation(
        total_units,
        means,
        min_per_store=1,
        max_wos=(args.mean_share_max_wos if args.mean_share_max_wos > 0 else None),
        share_wos_mode=args.share_wos_mode,
    )

    alloc_vol = optimize_volatility_aware(
        means=means,
        vmr=vmr_plan,
        total_units=total_units,
        min_per_store=1,
    )

    eval_demands = sample_nb_demands(
        means=means,
        vmr=vmr_realized,
        trials=args.eval_trials,
        seed=args.seed + 200,
    )

    metrics_mean = evaluate(eval_demands, alloc_mean)
    metrics_vol = evaluate(eval_demands, alloc_vol)
    sold_mean_daily = np.minimum(eval_demands, alloc_mean).sum(axis=1)
    sold_vol_daily = np.minimum(eval_demands, alloc_vol).sum(axis=1)
    realized_demand_store = eval_demands.mean(axis=0)
    sold_mean_store = np.minimum(eval_demands, alloc_mean).mean(axis=0)
    sold_vol_store = np.minimum(eval_demands, alloc_vol).mean(axis=0)
    sig = paired_significance(sold_mean_daily, sold_vol_daily)

    sold_gain = metrics_vol.avg_sold - metrics_mean.avg_sold
    sold_gain_pct = 100.0 * sold_gain / metrics_mean.avg_sold if metrics_mean.avg_sold > 0 else 0.0
    lost_reduction = metrics_mean.avg_lost - metrics_vol.avg_lost
    fill_gain_pp = 100.0 * (metrics_vol.fill_rate - metrics_mean.fill_rate)

    print("Negative Binomial Allocation Test (100 Stores)")
    print("=" * 88)
    print(f"Stores:                   {args.stores}")
    print(
        "Mean generation:          "
        f"Pareto(alpha={args.pareto_alpha}) clipped to [{args.mean_min}, {args.mean_max}]"
    )
    print("Planning VMR rule:        1 + 0.1 * mean")
    print("Realized VMR rule:        1 + coeff * mean")
    print(
        "Realized coeff rule:      "
        f"coeff = max(0, 0.1 + {args.coef_error_abs}*u), u~U(-1,1)"
    )
    print(
        "Realized coeff range:     "
        f"[{coeff_realized.min():.3f}, {coeff_realized.max():.3f}]"
    )
    print(f"Total expected demand:    {total_mean:.2f}")
    print(f"Total allocation budget:  {total_units}")
    print("Minimum allocation/store: 1")
    if args.mean_share_max_wos > 0:
        print(f"Mean-share max WOS:       {args.mean_share_max_wos} (soft cap)")
    else:
        print("Mean-share max WOS:       disabled")
    print(f"Share WOS mode:           {args.share_wos_mode}")
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
    print(f"Sold gain (%):            {sold_gain_pct:.2f}%")
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
    print(f"{'Store':>8}{'Mean':>10}{'RealDem':>10}{'Coeff':>10}{'VMR':>10}{'MeanAlloc':>10}{'VolAlloc':>10}{'Diff':>8}")
    for idx in top_diff_idx:
        diff = int(alloc_vol[idx] - alloc_mean[idx])
        print(
            f"{idx:>8}{means[idx]:>10.3f}{realized_demand_store[idx]:>10.3f}{coeff_realized[idx]:>10.3f}{vmr_realized[idx]:>10.3f}"
            f"{int(alloc_mean[idx]):>10}"
            f"{int(alloc_vol[idx]):>10}{diff:>8}"
        )

    bin_rows = summarize_by_mean_bins(
        means=means,
        realized_demand=realized_demand_store,
        alloc_mean=alloc_mean,
        alloc_vol=alloc_vol,
        sold_mean_store=sold_mean_store,
        sold_vol_store=sold_vol_store,
    )
    print("\nSummary by mean bins")
    print("-" * 88)
    print(
        f"{'Bin':>8}{'Stores':>8}{'AvgMean':>10}{'AvgDem':>10}"
        f"{'AllocM':>10}{'AllocV':>10}{'SoldGain':>10}{'Gain%':>9}"
    )
    for row in bin_rows:
        print(
            f"{row['bin']:>8}{row['stores']:>8}{row['avg_mean']:>10.2f}{row['avg_realized_demand']:>10.2f}"
            f"{row['alloc_mean_total']:>10}{row['alloc_vol_total']:>10}{row['sold_gain']:>10.2f}{row['sold_gain_pct']:>8.2f}%"
        )


if __name__ == "__main__":
    main()
