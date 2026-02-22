#!/usr/bin/env python3
"""Interactive Streamlit app for allocation simulations."""

from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from negative_binomial_100_stores import (
    build_vmr,
    generate_store_means,
    optimize_volatility_aware,
    proportional_integer_allocation,
    summarize_by_mean_bins,
)


st.set_page_config(page_title="Allocation Probability Test", layout="wide")

st.title("Allocation Probability Test")
st.caption("100-store Negative Binomial demand with VMR rule. Evaluation: exact (analytical).")


# ── Analytical expected sales ─────────────────────────────────────────────────

def _expected_sold_nb(mu: float, vmr: float, alloc: int) -> float:
    """Exact E[min(D, alloc)] for a single store with NB(mu, vmr).

    Uses: E[min(D, a)] = sum_{k=0}^{a-1} P(D > k).
    """
    if alloc <= 0 or mu <= 0:
        return 0.0

    if vmr <= 1.0 + 1e-12:
        # Poisson case
        pmf = math.exp(-mu)
        cdf = pmf
        total = 0.0
        for k in range(alloc):
            total += max(0.0, 1.0 - cdf)
            pmf = pmf * mu / (k + 1)
            cdf += pmf
        return total

    r = mu / (vmr - 1.0)
    p = r / (r + mu)
    q = 1.0 - p

    pmf = p ** r  # P(X=0)
    cdf = pmf
    total = 0.0
    for k in range(alloc):
        total += max(0.0, 1.0 - cdf)
        pmf = pmf * q * (r + k) / (k + 1)
        cdf += pmf
    return total


def expected_sold_per_store(
    means: np.ndarray,
    vmr: np.ndarray,
    alloc: np.ndarray,
) -> np.ndarray:
    """Return array of E[sold_i] for each store."""
    return np.array([
        _expected_sold_nb(float(means[i]), float(vmr[i]), int(alloc[i]))
        for i in range(len(means))
    ])


# ── Main tab ──────────────────────────────────────────────────────────────────

def run_nb_tab() -> None:
    st.subheader("100-Store Negative Binomial")
    st.caption(
        "Planning uses VMR = 1 + coeff \u00d7 mean. Realized can shift means via a bias multiplier and uses "
        "coeff = max(0, coeff + error_abs*u), u~U(-1,1)."
    )

    c1, c2, c3 = st.columns(3)
    stores = c1.number_input("Stores", min_value=1, value=100, step=1)
    total_factor = c2.number_input("Total factor", min_value=0.1, value=2.1, step=0.05)
    share_wos_mode = c3.selectbox(
        "Share WOS mode",
        options=["after", "before"],
        index=1,
        help="after: choose by (alloc+1)/mean, before: choose by alloc/mean",
    )

    st.markdown("**Store Distribution Parameters**")
    s1, s2, s3, s4 = st.columns(4)
    pareto_alpha = s1.number_input(
        "Pareto alpha",
        min_value=0.1,
        value=1.15,
        step=0.1,
        help=(
            "Shape parameter for the Pareto distribution of store means. "
            "Formula: mean_i = mean_min * (1 + Pareto(alpha)), clipped to [min, max]. "
            "Lower alpha \u2192 heavier tail \u2192 few stores dominate demand. "
            "Higher alpha \u2192 lighter tail \u2192 stores more similar. "
            "Typical range: 0.5 (very skewed) to 3.0 (mild skew)."
        ),
    )
    mean_min = s2.number_input("Mean min", min_value=0.1, value=0.5, step=0.1)
    mean_max = s3.number_input("Mean max", min_value=0.2, value=50.0, step=0.5)
    target_store_mean = s4.number_input(
        "Target store mean",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.1f",
        help=(
            "Target average demand per store. 0 = disabled (use raw Pareto). "
            "When set, all means are rescaled so their average equals this value "
            "while preserving the Pareto shape."
        ),
    )

    st.markdown("**Demand Distribution Parameters**")
    d1, d2, d3 = st.columns(3)
    vmr_coeff = d1.number_input(
        "VMR coefficient",
        min_value=0.0,
        value=0.1,
        step=0.01,
        format="%.3f",
        help="Planning VMR = 1 + coeff \u00d7 mean. Default 0.1.",
    )
    coef_error_abs = d2.number_input(
        "Coeff error abs (u~U(-1,1))",
        min_value=0.0,
        value=0.0,
        step=0.05,
        format="%.3f",
    )
    realized_demand_bias = d3.number_input(
        "D / F (demand / forecast)",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.05,
        format="%.3f",
        help=(
            "Realized demand relative to forecast. "
            "1.0 = perfect forecast, 1.2 = +20% demand, 0.8 = -20% demand."
        ),
    )

    run_clicked = st.button("Run", type="primary", key="nb_run")

    if not run_clicked:
        st.info("Set parameters and click 'Run'.")
        return

    if mean_max <= mean_min:
        st.error("`mean_max` must be greater than `mean_min`.")
        return

    means = generate_store_means(
        n_stores=int(stores),
        seed=7,
        pareto_alpha=float(pareto_alpha),
        mean_min=float(mean_min),
        mean_max=float(mean_max),
        target_store_mean=(float(target_store_mean) if float(target_store_mean) > 0 else None),
    )
    coeff_plan = np.full(int(stores), float(vmr_coeff), dtype=float)
    rng_coeff = np.random.default_rng(8)
    coeff_error = rng_coeff.uniform(-1.0, 1.0, size=int(stores))
    coeff_realized = np.maximum(0.0, float(vmr_coeff) + coeff_error * float(coef_error_abs))
    realized_means = means * float(realized_demand_bias)
    vmr_plan = build_vmr(means, coeff_plan)
    vmr_realized = build_vmr(realized_means, coeff_realized)

    total_mean = float(means.sum())
    total_realized_mean = float(realized_means.sum())
    total_units = int(round(total_mean * float(total_factor)))
    if total_units < int(stores):
        st.error(
            "Total allocation is below the minimum required to give each store at least 1 unit. "
            f"Required: {int(stores)}, got: {total_units}. Increase total factor or mean range."
        )
        return

    alloc_mean = proportional_integer_allocation(
        total_units,
        means,
        min_per_store=1,
        share_wos_mode=str(share_wos_mode),
    )

    alloc_vol = optimize_volatility_aware(
        means=means,
        vmr=vmr_plan,
        total_units=total_units,
        min_per_store=1,
    )

    # ── Analytical evaluation ─────────────────────────────────────────────
    sold_mean_store = expected_sold_per_store(realized_means, vmr_realized, alloc_mean)
    sold_vol_store = expected_sold_per_store(realized_means, vmr_realized, alloc_vol)

    total_sold_mean = float(sold_mean_store.sum())
    total_sold_vol = float(sold_vol_store.sum())
    total_lost_mean = total_realized_mean - total_sold_mean
    total_lost_vol = total_realized_mean - total_sold_vol
    total_left_mean = total_units - total_sold_mean
    total_left_vol = total_units - total_sold_vol
    fill_rate_mean = total_sold_mean / total_realized_mean if total_realized_mean > 0 else 1.0
    fill_rate_vol = total_sold_vol / total_realized_mean if total_realized_mean > 0 else 1.0

    sold_gain = total_sold_vol - total_sold_mean
    sold_gain_pct = 100.0 * sold_gain / total_sold_mean if total_sold_mean > 0 else 0.0
    lost_reduction = total_lost_mean - total_lost_vol
    fill_gain_pp = 100.0 * (fill_rate_vol - fill_rate_mean)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sold Gain (Vol - Mean)", f"{sold_gain:.3f}")
    k2.metric("Sold Gain %", f"{sold_gain_pct:.3f}%")
    k3.metric("Lost Reduction", f"{lost_reduction:.3f}")
    k4.metric("Fill Gain (pp)", f"{fill_gain_pp:.3f}")

    lost_rate_mean = 100.0 * (1.0 - fill_rate_mean)
    lost_rate_vol = 100.0 * (1.0 - fill_rate_vol)
    policy_rows = [
        {
            "Policy": "Mean-share",
            "E[Sold]": round(total_sold_mean, 3),
            "E[Lost]": round(total_lost_mean, 3),
            "Lost Rate %": round(lost_rate_mean, 3),
        },
        {
            "Policy": "Volatility-aware",
            "E[Sold]": round(total_sold_vol, 3),
            "E[Lost]": round(total_lost_vol, 3),
            "Lost Rate %": round(lost_rate_vol, 3),
        },
    ]
    st.dataframe(policy_rows, use_container_width=True)

    # ── Delta chart ───────────────────────────────────────────────────────
    order = np.argsort(means)
    delta = (alloc_vol - alloc_mean)[order].astype(float)
    vol_alloc_sorted = alloc_vol[order].astype(float)
    mean_alloc_sorted = alloc_mean[order].astype(float)
    sign = np.where(delta > 0, "positive", np.where(delta < 0, "negative", "zero"))
    sorted_means = means[order].astype(float)
    delta_df = pd.DataFrame({
        "store_index": np.arange(len(delta)),
        "delta": delta,
        "sign": sign,
        "vol_alloc": vol_alloc_sorted,
        "mean_alloc": mean_alloc_sorted,
        "fc_mean": sorted_means,
    })
    sign_scale = alt.Scale(
        domain=["positive", "negative", "zero"],
        range=["#59a14f", "#e45756", "#999999"],
    )
    delta_bars = (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X("store_index:O", title="Store index (sorted by forecast mean)"),
            y=alt.Y("delta:Q", title="\u0394 units (Vol \u2212 Mean-share)"),
            color=alt.Color("sign:N", scale=sign_scale, legend=None),
            tooltip=[
                alt.Tooltip("store_index:O", title="Store"),
                alt.Tooltip("fc_mean:Q", title="Fc Mean", format=".2f"),
                alt.Tooltip("mean_alloc:Q", title="Mean-share", format=".0f"),
                alt.Tooltip("vol_alloc:Q", title="Vol-aware", format=".0f"),
                alt.Tooltip("delta:Q", title="\u0394", format=".0f"),
            ],
        )
    )
    delta_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="white", strokeWidth=1)
        .encode(y="y:Q")
    )
    delta_chart = (
        (delta_bars + delta_rule)
        .properties(height=300)
    )
    st.markdown("**Allocation Delta (Vol \u2212 Mean-share)**")
    st.caption("Per-store difference, sorted by forecast mean (ascending). Green = vol-aware gives more, red = fewer.")
    st.altair_chart(delta_chart, use_container_width=True)

    # ── Lost Sales Delta chart ─────────────────────────────────────────────
    lost_mean_store = realized_means[order] - sold_mean_store[order]
    lost_vol_store = realized_means[order] - sold_vol_store[order]
    lost_delta = (lost_mean_store - lost_vol_store).astype(float)
    lost_sign = np.where(lost_delta > 0.001, "positive", np.where(lost_delta < -0.001, "negative", "zero"))
    lost_delta_df = pd.DataFrame({
        "store_index": np.arange(len(lost_delta)),
        "lost_delta": lost_delta,
        "sign": lost_sign,
        "lost_baseline": np.round(lost_mean_store, 3),
        "lost_vol": np.round(lost_vol_store, 3),
        "fc_mean": sorted_means,
    })
    lost_delta_bars = (
        alt.Chart(lost_delta_df)
        .mark_bar()
        .encode(
            x=alt.X("store_index:O", title="Store index (sorted by forecast mean)"),
            y=alt.Y("lost_delta:Q", title="\u0394 E[Lost] (reduction = positive)"),
            color=alt.Color("sign:N", scale=sign_scale, legend=None),
            tooltip=[
                alt.Tooltip("store_index:O", title="Store"),
                alt.Tooltip("fc_mean:Q", title="Fc Mean", format=".2f"),
                alt.Tooltip("lost_baseline:Q", title="Lost (Mean-share)", format=".3f"),
                alt.Tooltip("lost_vol:Q", title="Lost (Vol-aware)", format=".3f"),
                alt.Tooltip("lost_delta:Q", title="Reduction", format=".3f"),
            ],
        )
    )
    lost_delta_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="white", strokeWidth=1)
        .encode(y="y:Q")
    )
    lost_delta_chart = (
        (lost_delta_bars + lost_delta_rule)
        .properties(height=300)
    )
    st.markdown("**Lost Sales Delta (Mean-share \u2212 Vol-aware)**")
    st.caption("Per-store reduction in expected lost sales. Green = vol-aware loses less, red = vol-aware loses more.")
    st.altair_chart(lost_delta_chart, use_container_width=True)

    bin_rows = summarize_by_mean_bins(
        means=means,
        realized_demand=realized_means,
        alloc_mean=alloc_mean,
        alloc_vol=alloc_vol,
        sold_mean_store=sold_mean_store,
        sold_vol_store=sold_vol_store,
    )
    bin_rows_fmt = []
    for row in bin_rows:
        demand_total = row["avg_realized_demand"] * row["stores"]
        lost_mean = demand_total - row["sold_mean_total"]
        lost_vol = demand_total - row["sold_vol_total"]
        lost_reduction = lost_mean - lost_vol
        lost_reduction_pct = 100.0 * lost_reduction / lost_mean if lost_mean > 0 else 0.0
        lost_rate_mean = 100.0 * lost_mean / demand_total if demand_total > 0 else 0.0
        lost_rate_vol = 100.0 * lost_vol / demand_total if demand_total > 0 else 0.0
        bin_rows_fmt.append({
            "Bin": row["bin"],
            "Stores": row["stores"],
            "Avg Mean": round(row["avg_mean"], 4),
            "E[Lost] Mean": round(lost_mean, 4),
            "E[Lost] Vol": round(lost_vol, 4),
            "Lost Rate Mean %": round(lost_rate_mean, 2),
            "Lost Rate Vol %": round(lost_rate_vol, 2),
            "Rate Diff (pp)": round(lost_rate_mean - lost_rate_vol, 2),
        })
    st.markdown("**Summary by mean bins**")
    st.dataframe(bin_rows_fmt, use_container_width=True)

    # ── Store Demand Profile chart ───────────────────────────────────────
    order_profile = np.argsort(means)
    sorted_means_profile = means[order_profile].astype(float)
    n_profile = len(sorted_means_profile)
    cum_pct = np.cumsum(sorted_means_profile) / sorted_means_profile.sum() * 100.0

    color_cat = np.where(sorted_means_profile <= 2, "\u2264 2",
                np.where(sorted_means_profile <= 5, "2\u20135", "> 5"))
    profile_df = pd.DataFrame({
        "rank": np.arange(n_profile),
        "mean": sorted_means_profile,
        "cumulative_demand_pct": cum_pct,
        "mean_band": color_cat,
    })

    color_scale = alt.Scale(
        domain=["\u2264 2", "2\u20135", "> 5"],
        range=["#2ec4b6", "#f5a623", "#e45756"],
    )

    profile_bars = (
        alt.Chart(profile_df)
        .mark_bar()
        .encode(
            x=alt.X("rank:Q", title="Store rank (sorted by mean \u2192)"),
            y=alt.Y("mean:Q", title="Store mean"),
            color=alt.Color("mean_band:N", scale=color_scale,
                            legend=alt.Legend(title="Mean band")),
            tooltip=[
                alt.Tooltip("rank:Q", title="Rank"),
                alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                alt.Tooltip("cumulative_demand_pct:Q", title="Cum. Demand %", format=".1f"),
            ],
        )
    )

    profile_line = (
        alt.Chart(profile_df)
        .mark_line(color="#f5a623", strokeWidth=2.5)
        .encode(
            x=alt.X("rank:Q"),
            y=alt.Y("cumulative_demand_pct:Q", title="Cumulative demand %",
                     scale=alt.Scale(domain=[0, 100])),
        )
    )

    ref_data = pd.DataFrame({"pct": [50, 80]})
    profile_refs = (
        alt.Chart(ref_data)
        .mark_rule(strokeDash=[4, 4], color="#bbbbbb")
        .encode(y=alt.Y("pct:Q", scale=alt.Scale(domain=[0, 100])))
    )
    profile_ref_labels = (
        alt.Chart(ref_data)
        .mark_text(align="left", dx=4, fontSize=10, color="#bbbbbb")
        .encode(
            y=alt.Y("pct:Q", scale=alt.Scale(domain=[0, 100])),
            x=alt.value(0),
            text=alt.Text("pct:Q", format=".0f"),
        )
    )

    profile_chart = (
        alt.layer(profile_bars, profile_line + profile_refs + profile_ref_labels)
        .resolve_scale(y="independent")
        .properties(
            title=f"Store Demand Profile \u2014 {n_profile} stores",
            height=350,
        )
    )
    st.markdown("**Store Demand Profile**")
    st.caption("Bars: store mean by rank. Line: cumulative demand %. Teal \u2264 2, amber 2\u20135, coral > 5.")
    st.altair_chart(profile_chart, use_container_width=True)

    st.write(
        {
            "Stores": int(stores),
            "Mean Generation": "Pareto",
            "Pareto alpha": float(pareto_alpha),
            "Mean range clip": [float(mean_min), float(mean_max)],
            "Planning VMR": f"1 + {float(vmr_coeff):.3f} * mean",
            "D / F (demand / forecast)": round(float(realized_demand_bias), 6),
            "Total Planned Demand": round(total_mean, 3),
            "Total Allocation": total_units,
            "Minimum Allocation/Store": 1,
            "Share WOS Mode": str(share_wos_mode),
            "Evaluation": "Exact (analytical)",
        }
    )


tab_readme, tab_nb = st.tabs(["README", "Negative Binomial (100 stores)"])

with tab_readme:
    readme_path = Path(__file__).with_name("README.md")
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("README.md not found.")

with tab_nb:
    run_nb_tab()
