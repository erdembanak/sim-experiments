#!/usr/bin/env python3
"""2-store probability test with normal demand and CV (std/mean) controls.

Policy A: allocate by mean demand only.
Policy B: allocate using a volatility-aware optimization that searches for the
allocation with the highest expected sold units under sampled demand paths.

Demands are sampled from normal distributions and rounded to non-negative ints.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class NormalDemand:
    mean: float
    cv: float

    @property
    def std(self) -> float:
        return self.mean * self.cv


@dataclass(frozen=True)
class Metrics:
    avg_demand: float
    avg_sold: float
    avg_lost_sales: float
    avg_leftover: float
    fill_rate: float
    stockout_prob_any: float
    stockout_prob_store_1: float
    stockout_prob_store_2: float


@dataclass(frozen=True)
class ScenarioResult:
    realized_mean_1: float
    realized_mean_2: float
    realized_std_1: float
    realized_std_2: float
    alloc_mean: Tuple[int, int]
    alloc_volatility: Tuple[int, int]
    metrics_mean: Metrics
    metrics_volatility: Metrics


def parse_float_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def sample_nonnegative_normal_int(model: NormalDemand, rng: random.Random) -> int:
    # Integerized demand with floor at 0 to avoid impossible negative sales.
    return max(0, int(round(rng.gauss(model.mean, model.std))))


def generate_demands(
    model_1: NormalDemand,
    model_2: NormalDemand,
    trials: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    demand_1 = [sample_nonnegative_normal_int(model_1, rng) for _ in range(trials)]
    demand_2 = [sample_nonnegative_normal_int(model_2, rng) for _ in range(trials)]
    return demand_1, demand_2


def sample_mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[int], mean_value: float) -> float:
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def evaluate_allocation(
    demand_1: Sequence[int],
    demand_2: Sequence[int],
    alloc_1: int,
    alloc_2: int,
) -> Metrics:
    total_demand = 0.0
    total_sold = 0.0
    total_lost = 0.0
    total_leftover = 0.0

    stockout_any = 0
    stockout_1 = 0
    stockout_2 = 0

    for d_1, d_2 in zip(demand_1, demand_2):
        sold_1 = min(d_1, alloc_1)
        sold_2 = min(d_2, alloc_2)

        lost_1 = d_1 - sold_1
        lost_2 = d_2 - sold_2

        left_1 = alloc_1 - sold_1
        left_2 = alloc_2 - sold_2

        total_demand += d_1 + d_2
        total_sold += sold_1 + sold_2
        total_lost += lost_1 + lost_2
        total_leftover += left_1 + left_2

        is_stockout_1 = d_1 > alloc_1
        is_stockout_2 = d_2 > alloc_2

        if is_stockout_1:
            stockout_1 += 1
        if is_stockout_2:
            stockout_2 += 1
        if is_stockout_1 or is_stockout_2:
            stockout_any += 1

    trials = len(demand_1)
    return Metrics(
        avg_demand=total_demand / trials,
        avg_sold=total_sold / trials,
        avg_lost_sales=total_lost / trials,
        avg_leftover=total_leftover / trials,
        fill_rate=total_sold / total_demand if total_demand else 0.0,
        stockout_prob_any=stockout_any / trials,
        stockout_prob_store_1=stockout_1 / trials,
        stockout_prob_store_2=stockout_2 / trials,
    )


def allocate_by_mean(total_units: int, mean_1: float, mean_2: float) -> Tuple[int, int]:
    share_1 = mean_1 / (mean_1 + mean_2)
    alloc_1 = round(total_units * share_1)
    return alloc_1, total_units - alloc_1


def allocate_volatility_aware(
    total_units: int,
    demand_1_for_optimization: Sequence[int],
    demand_2_for_optimization: Sequence[int],
) -> Tuple[int, int]:
    best_alloc_1 = 0
    best_total_sold = -1

    for alloc_1 in range(total_units + 1):
        alloc_2 = total_units - alloc_1
        sold_sum = 0

        for d_1, d_2 in zip(demand_1_for_optimization, demand_2_for_optimization):
            sold_sum += min(d_1, alloc_1) + min(d_2, alloc_2)

        if sold_sum > best_total_sold:
            best_total_sold = sold_sum
            best_alloc_1 = alloc_1

    return best_alloc_1, total_units - best_alloc_1


def run_scenario(
    model_1: NormalDemand,
    model_2: NormalDemand,
    total_units: int,
    eval_trials: int,
    optimization_trials: int,
    seed: int,
) -> ScenarioResult:
    alloc_mean = allocate_by_mean(total_units, model_1.mean, model_2.mean)

    demand_1_opt, demand_2_opt = generate_demands(
        model_1,
        model_2,
        optimization_trials,
        seed + 101,
    )
    alloc_vol = allocate_volatility_aware(total_units, demand_1_opt, demand_2_opt)

    demand_1_eval, demand_2_eval = generate_demands(model_1, model_2, eval_trials, seed + 202)

    metrics_mean = evaluate_allocation(demand_1_eval, demand_2_eval, alloc_mean[0], alloc_mean[1])
    metrics_vol = evaluate_allocation(demand_1_eval, demand_2_eval, alloc_vol[0], alloc_vol[1])

    realized_mean_1 = sample_mean(demand_1_eval)
    realized_mean_2 = sample_mean(demand_2_eval)
    realized_std_1 = sample_std(demand_1_eval, realized_mean_1)
    realized_std_2 = sample_std(demand_2_eval, realized_mean_2)

    return ScenarioResult(
        realized_mean_1=realized_mean_1,
        realized_mean_2=realized_mean_2,
        realized_std_1=realized_std_1,
        realized_std_2=realized_std_2,
        alloc_mean=alloc_mean,
        alloc_volatility=alloc_vol,
        metrics_mean=metrics_mean,
        metrics_volatility=metrics_vol,
    )


def print_policy_row(name: str, alloc: Tuple[int, int], metrics: Metrics) -> None:
    print(
        f"{name:<34}"
        f"{alloc[0]:>9}"
        f"{alloc[1]:>9}"
        f"{metrics.avg_sold:>12.2f}"
        f"{metrics.avg_lost_sales:>12.2f}"
        f"{metrics.avg_leftover:>12.2f}"
        f"{100 * metrics.fill_rate:>11.2f}%"
    )


def print_scenario_report(
    model_1: NormalDemand,
    model_2: NormalDemand,
    total_units: int,
    eval_trials: int,
    result: ScenarioResult,
) -> None:
    sold_gain = result.metrics_volatility.avg_sold - result.metrics_mean.avg_sold
    lost_reduction = result.metrics_mean.avg_lost_sales - result.metrics_volatility.avg_lost_sales
    fill_gain_pp = 100 * (result.metrics_volatility.fill_rate - result.metrics_mean.fill_rate)

    print("2-Store Normal Demand Test")
    print("=" * 104)
    print(
        f"Input model S1: mean={model_1.mean:.2f}, std={model_1.std:.2f}, cv={model_1.cv:.2f}"
    )
    print(
        f"Input model S2: mean={model_2.mean:.2f}, std={model_2.std:.2f}, cv={model_2.cv:.2f}"
    )
    print(
        f"Realized eval S1: mean={result.realized_mean_1:.2f}, std={result.realized_std_1:.2f}"
    )
    print(
        f"Realized eval S2: mean={result.realized_mean_2:.2f}, std={result.realized_std_2:.2f}"
    )
    print(f"Total allocation budget: {total_units} units")
    print(f"Evaluation days:         {eval_trials:,}")
    print()

    print(
        f"{'Policy':<34}{'Alloc S1':>9}{'Alloc S2':>9}{'Avg Sold':>12}"
        f"{'Avg Lost':>12}{'Avg Left':>12}{'Fill Rate':>11}"
    )
    print("-" * 104)
    print_policy_row("Mean-demand split", result.alloc_mean, result.metrics_mean)
    print_policy_row("Volatility-aware optimization", result.alloc_volatility, result.metrics_volatility)

    print("\nConclusion")
    print("-" * 104)
    if sold_gain > 0:
        print(
            "Volatility-aware policy performs better: "
            f"+{sold_gain:.2f} sold/day, "
            f"-{lost_reduction:.2f} lost/day, "
            f"+{fill_gain_pp:.2f} pp fill rate."
        )
    elif sold_gain < 0:
        print("Mean-demand split performs better in this setup.")
    else:
        print("Both policies are effectively equal in this setup.")


def run_cv2_sweep(
    mean_1: float,
    mean_2: float,
    cv_1: float,
    cv_2_values: Sequence[float],
    total_units: int,
    eval_trials: int,
    optimization_trials: int,
    seed: int,
) -> None:
    print("\nCV Sweep (varying Store 2 CV)")
    print("=" * 104)
    print(
        f"Store 1 fixed: mean={mean_1:.2f}, cv={cv_1:.2f} | "
        f"Store 2 mean fixed: {mean_2:.2f}"
    )
    print(
        f"{'S2 CV':>8}{'Mean Alloc':>13}{'Vol Alloc':>13}{'Sold Gain':>12}{'Fill Gain':>12}"
    )
    print("-" * 104)

    for idx, cv_2 in enumerate(cv_2_values):
        model_1 = NormalDemand(mean=mean_1, cv=cv_1)
        model_2 = NormalDemand(mean=mean_2, cv=cv_2)

        result = run_scenario(
            model_1,
            model_2,
            total_units,
            eval_trials,
            optimization_trials,
            seed + 1000 * idx,
        )

        sold_gain = result.metrics_volatility.avg_sold - result.metrics_mean.avg_sold
        fill_gain_pp = 100 * (result.metrics_volatility.fill_rate - result.metrics_mean.fill_rate)

        mean_alloc_label = f"{result.alloc_mean[0]}/{result.alloc_mean[1]}"
        vol_alloc_label = f"{result.alloc_volatility[0]}/{result.alloc_volatility[1]}"

        print(
            f"{cv_2:>8.2f}"
            f"{mean_alloc_label:>13}"
            f"{vol_alloc_label:>13}"
            f"{sold_gain:>12.2f}"
            f"{fill_gain_pp:>11.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mean-only allocation vs volatility-aware allocation under normal demand.",
    )
    parser.add_argument("--mean1", type=float, default=40.0, help="Store 1 demand mean")
    parser.add_argument("--mean2", type=float, default=60.0, help="Store 2 demand mean")
    parser.add_argument("--cv1", type=float, default=0.35, help="Store 1 coefficient of variation")
    parser.add_argument("--cv2", type=float, default=0.70, help="Store 2 coefficient of variation")
    parser.add_argument("--total", type=int, default=119, help="Total units to allocate")
    parser.add_argument("--trials", type=int, default=20_000, help="Evaluation simulation days")
    parser.add_argument(
        "--opt-trials",
        type=int,
        default=40_000,
        help="Simulation days used by the volatility-aware optimizer",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--sweep-cv2",
        type=str,
        default="0.10,0.30,0.50,0.70,0.90,1.10",
        help="Comma-separated Store 2 CV values for sweep",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Disable CV sweep table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mean1 <= 0 or args.mean2 <= 0:
        raise ValueError("Means must be > 0.")
    if args.cv1 < 0 or args.cv2 < 0:
        raise ValueError("CV values must be >= 0.")
    if args.total <= 0:
        raise ValueError("Total allocation must be > 0.")
    if args.trials <= 0 or args.opt_trials <= 0:
        raise ValueError("trials and opt-trials must be > 0.")

    model_1 = NormalDemand(mean=args.mean1, cv=args.cv1)
    model_2 = NormalDemand(mean=args.mean2, cv=args.cv2)

    result = run_scenario(
        model_1,
        model_2,
        args.total,
        args.trials,
        args.opt_trials,
        args.seed,
    )

    print_scenario_report(model_1, model_2, args.total, args.trials, result)

    if not args.no_sweep:
        cv_2_values = parse_float_list(args.sweep_cv2)
        run_cv2_sweep(
            args.mean1,
            args.mean2,
            args.cv1,
            cv_2_values,
            args.total,
            max(50_000, args.trials // 2),
            args.opt_trials,
            args.seed,
        )


if __name__ == "__main__":
    main()
