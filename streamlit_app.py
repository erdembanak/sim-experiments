#!/usr/bin/env python3
"""Interactive Streamlit app for allocation simulations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st

from negative_binomial_100_stores import (
    evaluate as evaluate_nb,
    generate_store_params,
    optimize_volatility_aware,
    proportional_integer_allocation,
    sample_nb_demands,
)
from probability_test import NormalDemand, generate_demands, parse_float_list, run_scenario


st.set_page_config(page_title="Allocation Probability Test", layout="wide")

st.title("Allocation Probability Test")
st.caption(
    "Tab 1: 2-store normal demand. "
    "Tab 2: 100-store Negative Binomial demand with VMR rule."
)


def paired_significance_from_daily_sold(
    sold_a: np.ndarray,
    sold_b: np.ndarray,
) -> dict:
    diff = sold_b - sold_a
    n = diff.size
    if n < 2:
        return {
            "mean_diff": float(diff.mean()) if n else 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0.0
    if se == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = mean_diff / se
        p_value = math.erfc(abs(z) / math.sqrt(2.0))

    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se
    significant = not (ci_low <= 0 <= ci_high)

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": significant,
    }


def format_metrics(metrics) -> dict:
    return {
        "Avg Sold": round(metrics.avg_sold, 2),
        "Avg Lost": round(metrics.avg_lost_sales, 2),
        "Avg Left": round(metrics.avg_leftover, 2),
        "Fill Rate %": round(100 * metrics.fill_rate, 2),
        "Stockout Any %": round(100 * metrics.stockout_prob_any, 2),
        "Stockout S1 %": round(100 * metrics.stockout_prob_store_1, 2),
        "Stockout S2 %": round(100 * metrics.stockout_prob_store_2, 2),
    }


def build_sweep_rows(
    mean_1_value: float,
    mean_2_value: float,
    cv_1_value: float,
    cv2_values: List[float],
    total_units_value: int,
    eval_trials_value: int,
    opt_trials_value: int,
    seed_value: int,
) -> List[dict]:
    rows: List[dict] = []

    for idx, cv2_value in enumerate(cv2_values):
        model_1 = NormalDemand(mean=mean_1_value, cv=cv_1_value)
        model_2 = NormalDemand(mean=mean_2_value, cv=cv2_value)
        result = run_scenario(
            model_1=model_1,
            model_2=model_2,
            total_units=total_units_value,
            eval_trials=eval_trials_value,
            optimization_trials=opt_trials_value,
            seed=seed_value + 1000 * idx,
        )

        rows.append(
            {
                "S2 CV": round(cv2_value, 3),
                "Mean Alloc": f"{result.alloc_mean[0]}/{result.alloc_mean[1]}",
                "Vol Alloc": f"{result.alloc_volatility[0]}/{result.alloc_volatility[1]}",
                "Sold Gain": round(
                    result.metrics_volatility.avg_sold - result.metrics_mean.avg_sold,
                    3,
                ),
                "Fill Gain pp": round(
                    100 * (result.metrics_volatility.fill_rate - result.metrics_mean.fill_rate),
                    3,
                ),
            }
        )

    return rows


def run_normal_tab() -> None:
    st.subheader("2-Store Normal Demand")

    with st.form("normal_form"):
        c1, c2, c3, c4 = st.columns(4)
        mean_1 = c1.number_input("Store 1 mean", min_value=0.1, value=70.0, step=1.0)
        mean_2 = c2.number_input("Store 2 mean", min_value=0.1, value=65.0, step=1.0)
        cv_1 = c3.number_input("Store 1 CV", min_value=0.0, value=0.35, step=0.05, format="%.2f")
        cv_2 = c4.number_input("Store 2 CV", min_value=0.0, value=0.70, step=0.05, format="%.2f")

        d1, d2, d3, d4 = st.columns(4)
        total_units = d1.number_input("Total allocation", min_value=1, value=119, step=1)
        eval_trials = d2.number_input("Evaluation trials", min_value=1000, value=120_000, step=1000)
        opt_trials = d3.number_input("Optimization trials", min_value=1000, value=30_000, step=1000)
        seed = d4.number_input("Seed", min_value=0, value=7, step=1)

        sweep_cv2_text = st.text_input(
            "Store 2 CV list (comma separated)",
            value="0.10,0.30,0.50,0.70,0.90,1.10",
        )
        run_sweep = st.checkbox("Run CV sweep", value=True)

        run_clicked = st.form_submit_button("Run Normal Simulation", type="primary")

    if not run_clicked:
        st.info("Set parameters and click 'Run Normal Simulation'.")
        return

    model_1 = NormalDemand(mean=float(mean_1), cv=float(cv_1))
    model_2 = NormalDemand(mean=float(mean_2), cv=float(cv_2))

    result = run_scenario(
        model_1=model_1,
        model_2=model_2,
        total_units=int(total_units),
        eval_trials=int(eval_trials),
        optimization_trials=int(opt_trials),
        seed=int(seed),
    )

    sold_gain = result.metrics_volatility.avg_sold - result.metrics_mean.avg_sold
    lost_reduction = result.metrics_mean.avg_lost_sales - result.metrics_volatility.avg_lost_sales
    fill_gain_pp = 100 * (result.metrics_volatility.fill_rate - result.metrics_mean.fill_rate)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sold Gain (Vol - Mean)", f"{sold_gain:.2f}")
    col_b.metric("Lost Reduction", f"{lost_reduction:.2f}")
    col_c.metric("Fill Gain (pp)", f"{fill_gain_pp:.2f}")

    demand_1_eval, demand_2_eval = generate_demands(
        model_1,
        model_2,
        int(eval_trials),
        int(seed) + 202,
    )
    sold_mean_daily = np.minimum(demand_1_eval, result.alloc_mean[0]) + np.minimum(
        demand_2_eval, result.alloc_mean[1]
    )
    sold_vol_daily = np.minimum(demand_1_eval, result.alloc_volatility[0]) + np.minimum(
        demand_2_eval, result.alloc_volatility[1]
    )
    sig = paired_significance_from_daily_sold(sold_mean_daily, sold_vol_daily)

    st.markdown("**Sold Gain Significance (paired by day)**")
    st.write(
        {
            "Mean daily sold gain": round(sig["mean_diff"], 4),
            "95% CI": [round(sig["ci_low"], 4), round(sig["ci_high"], 4)],
            "p-value": round(sig["p_value"], 8),
            "Significant at 5%": bool(sig["significant"]),
        }
    )

    st.write(
        {
            "Store 1": {
                "Input Mean": model_1.mean,
                "Input Std": model_1.std,
                "Input CV": model_1.cv,
                "Realized Mean": round(result.realized_mean_1, 2),
                "Realized Std": round(result.realized_std_1, 2),
            },
            "Store 2": {
                "Input Mean": model_2.mean,
                "Input Std": model_2.std,
                "Input CV": model_2.cv,
                "Realized Mean": round(result.realized_mean_2, 2),
                "Realized Std": round(result.realized_std_2, 2),
            },
        }
    )

    comparison_rows = [
        {
            "Policy": "Mean-demand split",
            "Alloc S1": result.alloc_mean[0],
            "Alloc S2": result.alloc_mean[1],
            **format_metrics(result.metrics_mean),
        },
        {
            "Policy": "Volatility-aware optimization",
            "Alloc S1": result.alloc_volatility[0],
            "Alloc S2": result.alloc_volatility[1],
            **format_metrics(result.metrics_volatility),
        },
    ]
    st.dataframe(comparison_rows, use_container_width=True)

    if run_sweep:
        st.markdown("**Store 2 CV Sweep**")
        try:
            cv2_values = parse_float_list(sweep_cv2_text)
            if not cv2_values:
                raise ValueError("No CV values parsed.")

            sweep_rows = build_sweep_rows(
                mean_1_value=float(mean_1),
                mean_2_value=float(mean_2),
                cv_1_value=float(cv_1),
                cv2_values=cv2_values,
                total_units_value=int(total_units),
                eval_trials_value=max(50_000, int(eval_trials) // 2),
                opt_trials_value=int(opt_trials),
                seed_value=int(seed),
            )
            st.dataframe(sweep_rows, use_container_width=True)
            st.line_chart(sweep_rows, x="S2 CV", y=["Sold Gain", "Fill Gain pp"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run sweep: {exc}")


def run_nb_tab() -> None:
    st.subheader("100-Store Negative Binomial")
    st.caption("VMR is fixed to: max(1, 0.9 * mean^0.9)")

    with st.form("nb_form"):
        c1, c2, c3, c4 = st.columns(4)
        stores = c1.number_input("Stores", min_value=1, value=100, step=1)
        mean_low = c2.number_input("Mean low", min_value=0.0, value=0.0, step=0.1)
        mean_high = c3.number_input("Mean high", min_value=0.1, value=5.0, step=0.1)
        total_factor = c4.number_input("Total factor", min_value=0.1, value=0.9, step=0.05)

        d1, d2, d3 = st.columns(3)
        eval_trials = d1.number_input("Evaluation trials", min_value=1000, value=120_000, step=1000, key="nb_eval")
        opt_trials = d2.number_input("Optimization trials", min_value=1000, value=40_000, step=1000, key="nb_opt")
        seed = d3.number_input("Seed", min_value=0, value=7, step=1, key="nb_seed")

        run_clicked = st.form_submit_button("Run NB Simulation", type="primary")

    if not run_clicked:
        st.info("Set parameters and click 'Run NB Simulation'.")
        return

    if mean_high <= mean_low:
        st.error("`mean_high` must be greater than `mean_low`.")
        return

    means, vmr = generate_store_params(
        n_stores=int(stores),
        mean_low=float(mean_low),
        mean_high=float(mean_high),
        seed=int(seed),
    )

    total_mean = float(means.sum())
    total_units = int(round(total_mean * float(total_factor)))

    alloc_mean = proportional_integer_allocation(total_units, means)

    opt_demands = sample_nb_demands(
        means=means,
        vmr=vmr,
        trials=int(opt_trials),
        seed=int(seed) + 100,
    )
    alloc_vol = optimize_volatility_aware(opt_demands, total_units)

    eval_demands = sample_nb_demands(
        means=means,
        vmr=vmr,
        trials=int(eval_trials),
        seed=int(seed) + 200,
    )

    metrics_mean = evaluate_nb(eval_demands, alloc_mean)
    metrics_vol = evaluate_nb(eval_demands, alloc_vol)

    sold_gain = metrics_vol.avg_sold - metrics_mean.avg_sold
    lost_reduction = metrics_mean.avg_lost - metrics_vol.avg_lost
    fill_gain_pp = 100.0 * (metrics_vol.fill_rate - metrics_mean.fill_rate)

    k1, k2, k3 = st.columns(3)
    k1.metric("Sold Gain (Vol - Mean)", f"{sold_gain:.3f}")
    k2.metric("Lost Reduction", f"{lost_reduction:.3f}")
    k3.metric("Fill Gain (pp)", f"{fill_gain_pp:.3f}")

    sold_mean_daily = np.minimum(eval_demands, alloc_mean).sum(axis=1)
    sold_vol_daily = np.minimum(eval_demands, alloc_vol).sum(axis=1)
    sig = paired_significance_from_daily_sold(sold_mean_daily, sold_vol_daily)

    st.markdown("**Sold Gain Significance (paired by day)**")
    st.write(
        {
            "Mean daily sold gain": round(sig["mean_diff"], 6),
            "95% CI": [round(sig["ci_low"], 6), round(sig["ci_high"], 6)],
            "p-value": round(sig["p_value"], 10),
            "Significant at 5%": bool(sig["significant"]),
        }
    )

    st.write(
        {
            "Stores": int(stores),
            "Mean Range": [float(mean_low), float(mean_high)],
            "VMR Rule": "max(1, 0.9 * mean^0.9)",
            "Total Expected Demand": round(total_mean, 3),
            "Total Allocation": total_units,
        }
    )

    policy_rows = [
        {
            "Policy": "Mean-share",
            "Avg Sold": round(metrics_mean.avg_sold, 3),
            "Avg Lost": round(metrics_mean.avg_lost, 3),
            "Avg Left": round(metrics_mean.avg_leftover, 3),
            "Fill Rate %": round(100 * metrics_mean.fill_rate, 3),
            "Stockout Any %": round(100 * metrics_mean.stockout_any, 3),
        },
        {
            "Policy": "Volatility-aware",
            "Avg Sold": round(metrics_vol.avg_sold, 3),
            "Avg Lost": round(metrics_vol.avg_lost, 3),
            "Avg Left": round(metrics_vol.avg_leftover, 3),
            "Fill Rate %": round(100 * metrics_vol.fill_rate, 3),
            "Stockout Any %": round(100 * metrics_vol.stockout_any, 3),
        },
    ]
    st.dataframe(policy_rows, use_container_width=True)

    diffs = alloc_vol - alloc_mean
    idx_sorted = np.argsort(np.abs(diffs))[::-1][:10]
    top_rows = [
        {
            "Store": int(i),
            "Mean": round(float(means[i]), 4),
            "VMR": round(float(vmr[i]), 4),
            "Mean Alloc": int(alloc_mean[i]),
            "Vol Alloc": int(alloc_vol[i]),
            "Diff": int(diffs[i]),
        }
        for i in idx_sorted
    ]
    st.markdown("**Top 10 stores by allocation difference**")
    st.dataframe(top_rows, use_container_width=True)


tab_readme, tab_normal, tab_nb = st.tabs(
    ["README", "Normal (2-store)", "Negative Binomial (100 stores)"]
)

with tab_readme:
    readme_path = Path(__file__).with_name("README.md")
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("README.md not found.")

with tab_normal:
    run_normal_tab()

with tab_nb:
    run_nb_tab()
