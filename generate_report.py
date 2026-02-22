#!/usr/bin/env python3
"""Generate a self-contained React + Recharts HTML report.

Loads React, ReactDOM, Recharts and Babel from CDN so the output is a single
HTML file that can be opened directly in any browser — no build step needed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from pathlib import Path

from negative_binomial_100_stores import (
    build_vmr,
    generate_store_means,
    optimize_volatility_aware,
    proportional_integer_allocation,
    sample_realized_coeff,
)

# ── Fixed parameters ──────────────────────────────────────────────────────────
STORES = 100
PARETO_ALPHA = 1.15
MEAN_MIN = 0.5
MEAN_MAX = 50.0
TARGET_STORE_MEAN = 2.0
SEED = 7

TOTAL_FACTORS = [0.7, 1.0, 1.5, 2.0]
BIAS_RATES = [0.8, 1.0, 1.2]


# ── Analytical expected sales ─────────────────────────────────────────────────

def _expected_sold_nb(mu: float, vmr: float, alloc: int) -> float:
    if alloc <= 0 or mu <= 0:
        return 0.0
    if vmr <= 1.0 + 1e-12:
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
    pmf = p ** r
    cdf = pmf
    total = 0.0
    for k in range(alloc):
        total += max(0.0, 1.0 - cdf)
        pmf = pmf * q * (r + k) / (k + 1)
        cdf += pmf
    return total


@dataclass(frozen=True)
class AnalyticalMetrics:
    expected_sold: float
    expected_lost: float
    expected_leftover: float
    fill_rate: float


def evaluate_analytical(
    means: np.ndarray, vmr: np.ndarray, alloc: np.ndarray,
) -> AnalyticalMetrics:
    total_demand = float(means.sum())
    total_stock = int(alloc.sum())
    expected_sold = sum(
        _expected_sold_nb(float(means[i]), float(vmr[i]), int(alloc[i]))
        for i in range(len(means))
    )
    return AnalyticalMetrics(
        expected_sold=round(expected_sold, 4),
        expected_lost=round(total_demand - expected_sold, 4),
        expected_leftover=round(total_stock - expected_sold, 4),
        fill_rate=expected_sold / total_demand if total_demand > 0 else 1.0,
    )


# ── Scenario runner ───────────────────────────────────────────────────────────
def _expected_sold_per_store(
    means: np.ndarray, vmr: np.ndarray, alloc: np.ndarray,
) -> np.ndarray:
    return np.array([
        _expected_sold_nb(float(means[i]), float(vmr[i]), int(alloc[i]))
        for i in range(len(means))
    ])


def run_scenario(
    means: np.ndarray, vmr_plan: np.ndarray, total_factor: float, bias: float,
):
    realized_means = means * bias
    coeff_realized = sample_realized_coeff(STORES, seed=SEED + 1, coef_error_abs=0.0)
    vmr_realized = build_vmr(realized_means, coeff_realized)
    total_units = int(round(means.sum() * total_factor))

    allocations = {
        "WOS-before": proportional_integer_allocation(
            total_units, means, min_per_store=1, share_wos_mode="before",
        ),
        "WOS-after": proportional_integer_allocation(
            total_units, means, min_per_store=1, share_wos_mode="after",
        ),
        "Volatility-aware": optimize_volatility_aware(
            means=means, vmr=vmr_plan, total_units=total_units, min_per_store=1,
        ),
    }
    metrics = {
        name: evaluate_analytical(realized_means, vmr_realized, alloc)
        for name, alloc in allocations.items()
    }
    sold_per_store = {
        name: _expected_sold_per_store(realized_means, vmr_realized, alloc)
        for name, alloc in allocations.items()
    }
    return allocations, metrics, total_units, realized_means, vmr_realized, sold_per_store


# ── Build JSON data for React app ────────────────────────────────────────────
def build_report_data(means: np.ndarray, vmr_plan: np.ndarray) -> dict:
    order = np.argsort(means)
    sorted_means = means[order]

    scenarios = []
    for total_factor in TOTAL_FACTORS:
        for bias in BIAS_RATES:
            print(f"  Running: buy_qty={total_factor}x  D/F={bias:.1f} …")
            allocations, metrics, total_units, realized_means, vmr_realized, sold_per_store = run_scenario(
                means, vmr_plan, total_factor, bias,
            )

            stores = []
            for idx, orig_i in enumerate(order):
                rm = float(realized_means[orig_i])
                sold_wb = float(sold_per_store["WOS-before"][orig_i])
                sold_wa = float(sold_per_store["WOS-after"][orig_i])
                sold_v = float(sold_per_store["Volatility-aware"][orig_i])
                stores.append({
                    "idx": idx,
                    "mean": round(float(sorted_means[idx]), 3),
                    "wosBefore": int(allocations["WOS-before"][orig_i]),
                    "wosAfter": int(allocations["WOS-after"][orig_i]),
                    "vol": int(allocations["Volatility-aware"][orig_i]),
                    "lostWosBefore": round(rm - sold_wb, 4),
                    "lostWosAfter": round(rm - sold_wa, 4),
                    "lostVol": round(rm - sold_v, 4),
                })

            metrics_dict = {}
            for method_name, m in metrics.items():
                lost_rate = 100.0 * (1.0 - m.fill_rate)
                metrics_dict[method_name] = {
                    "expectedSold": m.expected_sold,
                    "expectedLost": m.expected_lost,
                    "expectedLeftover": m.expected_leftover,
                    "fillRate": round(100.0 * m.fill_rate, 2),
                    "lostRate": round(lost_rate, 2),
                }

            scenarios.append({
                "totalFactor": total_factor,
                "bias": bias,
                "biasLabel": f"{bias:.1f}",
                "totalUnits": total_units,
                "plannedDemand": round(float(means.sum()), 1),
                "realizedDemand": round(float(means.sum() * bias), 1),
                "stores": stores,
                "metrics": metrics_dict,
            })

    # ── Demand profile data (once, independent of scenarios) ────────────
    profile_data = []
    cum_sum = 0.0
    total_mean = float(sorted_means.sum())
    for idx, m in enumerate(sorted_means):
        cum_sum += float(m)
        profile_data.append({
            "rank": idx,
            "mean": round(float(m), 3),
            "cumDemandPct": round(cum_sum / total_mean * 100.0, 2),
        })

    return {
        "params": {
            "stores": STORES,
            "paretoAlpha": PARETO_ALPHA,
            "meanMin": MEAN_MIN,
            "meanMax": MEAN_MAX,
            "targetStoreMean": TARGET_STORE_MEAN,
        },
        "totalFactors": TOTAL_FACTORS,
        "biasRates": BIAS_RATES,
        "demandProfile": profile_data,
        "scenarios": scenarios,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Allocation Method Comparison Report</title>
<script src="https://cdn.jsdelivr.net/npm/react@18/umd/react.development.js" crossorigin></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.development.js" crossorigin></script>
<script src="https://cdn.jsdelivr.net/npm/prop-types@15/prop-types.min.js" crossorigin></script>
<script src="https://cdn.jsdelivr.net/npm/recharts@2/umd/Recharts.js" crossorigin></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7/babel.min.js" crossorigin></script>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; background: #0f0f1a; color: #e8e8f0;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #root { max-width: 960px; margin: 0 auto; padding: 24px 20px 60px; }
</style>
</head>
<body>
<div id="root"></div>

<script>
  window.__REPORT_DATA__ = __JSON_DATA__;
</script>

<script type="text/babel">
const {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer, Cell,
  LineChart, Line,
  ComposedChart,
} = Recharts;
const { useState, useMemo } = React;

const DATA = window.__REPORT_DATA__;
const BLUE = "#4c78a8";
const RED = "#e45756";
const ORANGE = "#f58518";
const GREEN = "#59a14f";
const METHODS = ["WOS-before", "WOS-after", "Volatility-aware"];
const METHOD_COLORS = { "WOS-before": RED, "WOS-after": ORANGE, "Volatility-aware": BLUE };

/* ── Helpers ──────────────────────────────────────────────────────────────── */
function enrichStores(stores) {
  return stores.map(s => ({
    ...s,
    diffBefore: s.vol - s.wosBefore,
    diffAfter: s.vol - s.wosAfter,
    lostDeltaBefore: s.lostWosBefore - s.lostVol,
    lostDeltaAfter: s.lostWosAfter - s.lostVol,
  }));
}

/* ── Pill button ─────────────────────────────────────────────────────────── */
function Pill({ active, onClick, children, color }) {
  const bg = active ? (color || BLUE) : "#1a1a30";
  const fg = active ? "#fff" : "#7a7a9a";
  return (
    <button onClick={onClick} style={{
      padding: "7px 16px", borderRadius: 8, border: "none", cursor: "pointer",
      fontSize: 13, fontWeight: 600,
      background: bg, color: fg, transition: "all 0.15s",
      boxShadow: active ? `0 2px 12px ${bg}44` : "none",
    }}>
      {children}
    </button>
  );
}

/* ── Tooltips ────────────────────────────────────────────────────────────── */
const ttStyle = { background: "#1a1a2e", border: "1px solid #444", borderRadius: 6,
                  padding: "8px 12px", fontSize: 12, color: "#e0e0e0" };

function DeltaTooltip({ active, payload, baseline }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const diff = baseline === "before" ? d.diffBefore : d.diffAfter;
  const base = baseline === "before" ? d.wosBefore : d.wosAfter;
  return (
    <div style={ttStyle}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Store {d.idx} · Mean: {d.mean.toFixed(2)}</div>
      <div>Baseline: {base}  →  Vol: {d.vol}  →
        <span style={{ color: diff > 0 ? GREEN : diff < 0 ? RED : "#aaa", fontWeight: 700 }}>
          {" "}Δ {diff > 0 ? "+" : ""}{diff}
        </span>
      </div>
    </div>
  );
}

function CDFTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={ttStyle}>
      <div style={{ fontWeight: 600, marginBottom: 2 }}>Allocation: {d.allocation} units</div>
      {METHODS.map(m => d[m] != null && (
        <div key={m} style={{ color: METHOD_COLORS[m] }}>{m}: {d[m]} stores</div>
      ))}
    </div>
  );
}

/* ── Chart: Delta ────────────────────────────────────────────────────────── */
function DeltaChart({ stores, baseline, title }) {
  const data = stores.map(s => ({
    ...s,
    diff: baseline === "before" ? s.diffBefore : s.diffAfter,
  }));
  return (
    <div>
      {title && <div style={{ fontSize: 13, fontWeight: 600, color: "#bbc", marginBottom: 4, textAlign: "center" }}>{title}</div>}
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="idx" tick={{ fontSize: 9, fill: "#999" }}
                 label={{ value: "Store (sorted by mean →)", position: "insideBottom",
                          offset: -10, style: { fill: "#bbb", fontSize: 12 } }} />
          <YAxis tick={{ fill: "#999", fontSize: 11 }}
                 label={{ value: "Δ Units (Vol − Baseline)", angle: -90,
                          position: "insideLeft", offset: 10,
                          style: { fill: "#bbb", fontSize: 12 } }} />
          <Tooltip content={<DeltaTooltip baseline={baseline} />} />
          <ReferenceLine y={0} stroke="#888" strokeWidth={1.5} />
          <Bar dataKey="diff" radius={[2, 2, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.diff > 0 ? GREEN : d.diff < 0 ? RED : "#555"} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Chart: Lost Sales Delta ─────────────────────────────────────────────── */
function LostDeltaTooltip({ active, payload, baseline }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const delta = baseline === "before" ? d.lostDeltaBefore : d.lostDeltaAfter;
  const lostBase = baseline === "before" ? d.lostWosBefore : d.lostWosAfter;
  return (
    <div style={ttStyle}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Store {d.idx} · Mean: {d.mean.toFixed(2)}</div>
      <div>Lost baseline: {lostBase.toFixed(3)} · Lost vol: {d.lostVol.toFixed(3)}</div>
      <div>
        <span style={{ color: delta > 0.001 ? GREEN : delta < -0.001 ? RED : "#aaa", fontWeight: 700 }}>
          Reduction: {delta > 0 ? "+" : ""}{delta.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function LostSalesDeltaChart({ stores, baseline, title }) {
  const data = stores.map(s => ({
    ...s,
    diff: baseline === "before" ? s.lostDeltaBefore : s.lostDeltaAfter,
  }));
  return (
    <div>
      {title && <div style={{ fontSize: 13, fontWeight: 600, color: "#bbc", marginBottom: 4, textAlign: "center" }}>{title}</div>}
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="idx" tick={{ fontSize: 9, fill: "#999" }}
                 label={{ value: "Store (sorted by mean →)", position: "insideBottom",
                          offset: -10, style: { fill: "#bbb", fontSize: 12 } }} />
          <YAxis tick={{ fill: "#999", fontSize: 11 }}
                 label={{ value: "Δ E[Lost] (reduction = positive)", angle: -90,
                          position: "insideLeft", offset: 10,
                          style: { fill: "#bbb", fontSize: 12 } }} />
          <Tooltip content={<LostDeltaTooltip baseline={baseline} />} />
          <ReferenceLine y={0} stroke="#888" strokeWidth={1.5} />
          <Bar dataKey="diff" radius={[2, 2, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.diff > 0.001 ? GREEN : d.diff < -0.001 ? RED : "#555"} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Chart: Allocation CDF ───────────────────────────────────────────────── */
function CDFChart({ stores }) {
  const cdfData = useMemo(() => {
    const allVals = new Set();
    METHODS.forEach(m => {
      const key = m === "WOS-before" ? "wosBefore" : m === "WOS-after" ? "wosAfter" : "vol";
      stores.forEach(s => allVals.add(s[key]));
    });
    const sortedVals = [...allVals].sort((a, b) => a - b);
    return sortedVals.map(v => {
      const row = { allocation: v };
      METHODS.forEach(m => {
        const key = m === "WOS-before" ? "wosBefore" : m === "WOS-after" ? "wosAfter" : "vol";
        row[m] = stores.filter(s => s[key] <= v).length;
      });
      return row;
    });
  }, [stores]);

  return (
    <ResponsiveContainer width="100%" height={440}>
      <LineChart data={cdfData} margin={{ top: 10, right: 20, bottom: 60, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="allocation" type="number" tick={{ fill: "#999", fontSize: 11 }}
               label={{ value: "Allocation (units per store)", position: "insideBottom",
                        offset: -4, style: { fill: "#bbb", fontSize: 13 } }} />
        <YAxis tick={{ fill: "#999", fontSize: 11 }}
               label={{ value: "Stores (cumulative)", angle: -90,
                        position: "insideLeft", offset: 10,
                        style: { fill: "#bbb", fontSize: 13 } }} />
        <Tooltip content={<CDFTooltip />} />
        {METHODS.map(m => (
          <Line key={m} type="stepAfter" dataKey={m} stroke={METHOD_COLORS[m]}
                strokeWidth={2} dot={false} name={m} />
        ))}
        <Legend wrapperStyle={{ color: "#bbb", fontSize: 12, paddingTop: 16, textAlign: "center" }} align="center" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ── Chart: Demand Profile ───────────────────────────────────────────────── */
const TEAL = "#2ec4b6";
const AMBER = "#f5a623";
const CORAL = "#e45756";

function ProfileTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={ttStyle}>
      <div style={{ fontWeight: 600, marginBottom: 2 }}>Rank {d.rank}</div>
      <div>Mean: {d.mean.toFixed(2)}</div>
      <div>Cum. Demand: {d.cumDemandPct.toFixed(1)}%</div>
    </div>
  );
}

function barColor(mean) {
  if (mean <= 2) return TEAL;
  if (mean <= 5) return AMBER;
  return CORAL;
}

function DemandProfileChart({ data, nStores }) {
  return (
    <div style={{ background: "#12122266", borderRadius: 12, padding: "16px 8px",
                  border: "1px solid #1e1e36", marginBottom: 20 }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: "#ccd", marginBottom: 4, textAlign: "center" }}>
        Store Demand Profile — {nStores} stores
      </div>
      <div style={{ fontSize: 11, color: "#888", textAlign: "center", marginBottom: 8 }}>
        <span style={{ color: TEAL, fontSize: 14 }}>■</span> mean ≤ 2 &nbsp;
        <span style={{ color: AMBER, fontSize: 14 }}>■</span> mean 2–5 &nbsp;
        <span style={{ color: CORAL, fontSize: 14 }}>■</span> mean &gt; 5 &nbsp;
        <span style={{ color: AMBER }}>—</span> cumulative demand %
      </div>
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={data} margin={{ top: 10, right: 50, bottom: 40, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="rank" tick={{ fill: "#999", fontSize: 10 }}
                 label={{ value: "Store rank (sorted by mean →)", position: "insideBottom",
                          offset: -10, style: { fill: "#bbb", fontSize: 12 } }} />
          <YAxis yAxisId="left" tick={{ fill: "#999", fontSize: 11 }}
                 label={{ value: "Store mean", angle: -90, position: "insideLeft",
                          offset: 10, style: { fill: "#bbb", fontSize: 12 } }} />
          <YAxis yAxisId="right" orientation="right" domain={[0, 100]}
                 tick={{ fill: "#999", fontSize: 11 }}
                 label={{ value: "Cumulative demand %", angle: 90, position: "insideRight",
                          offset: 10, style: { fill: "#bbb", fontSize: 12 } }} />
          <Tooltip content={<ProfileTooltip />} />
          <ReferenceLine yAxisId="right" y={50} stroke="#888" strokeDasharray="4 4"
                         label={{ value: "50%", position: "right", fill: "#888", fontSize: 10 }} />
          <ReferenceLine yAxisId="right" y={80} stroke="#888" strokeDasharray="4 4"
                         label={{ value: "80%", position: "right", fill: "#888", fontSize: 10 }} />
          <Bar yAxisId="left" dataKey="mean" radius={[2, 2, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={barColor(d.mean)} fillOpacity={0.85} />
            ))}
          </Bar>
          <Line yAxisId="right" type="monotone" dataKey="cumDemandPct"
                stroke={AMBER} strokeWidth={2.5} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Metrics table ───────────────────────────────────────────────────────── */
function MetricsTable({ metrics }) {
  const cellStyle = { padding: "6px 12px", borderBottom: "1px solid #2a2a44", textAlign: "right", fontSize: 13 };
  const hdrStyle = { ...cellStyle, fontWeight: 600,
                     background: "#16162a", borderBottom: "2px solid #333" };
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, color: "#888", marginTop: 16, marginBottom: 4 }}>Lost Sales Rate %</div>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            {METHODS.map(m => <th key={m} style={{ ...hdrStyle, color: METHOD_COLORS[m] }}>{m}</th>)}
        </tr>
      </thead>
      <tbody>
        <tr>
          {METHODS.map(m => (
            <td key={m} style={cellStyle}>{metrics[m].lostRate.toFixed(2)}%</td>
          ))}
        </tr>
      </tbody>
    </table>
    </div>
  );
}

/* ── Summary table ───────────────────────────────────────────────────────── */
function SummaryTable({ scenarios }) {
  const cellStyle = { padding: "5px 10px", borderBottom: "1px solid #2a2a44", textAlign: "right", fontSize: 12 };
  const hdrStyle = { ...cellStyle, fontWeight: 600, color: "#aab",
                     background: "#16162a", borderBottom: "2px solid #333" };
  return (
    <div style={{ marginTop: 32 }}>
      <h2 style={{ fontSize: 18, color: "#ccd", marginBottom: 12 }}>Lost Sales Rate by Scenario</h2>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ ...hdrStyle, textAlign: "left" }}>Buy Qty</th>
            <th style={hdrStyle}>D / F</th>
            <th style={hdrStyle}>Total Stock</th>
            {METHODS.map(m => <th key={m} style={{ ...hdrStyle, color: METHOD_COLORS[m] }}>{m}</th>)}
          </tr>
        </thead>
        <tbody>
          {scenarios.map((s, i) => {
            const rates = METHODS.map(m => s.metrics[m].lostRate);
            const bestRate = Math.min(...rates);
            return (
              <tr key={i}>
                <td style={{ ...cellStyle, textAlign: "left" }}>{s.totalFactor}x</td>
                <td style={{ ...cellStyle, textAlign: "center" }}>{s.biasLabel}</td>
                <td style={cellStyle}>{s.totalUnits}</td>
                {METHODS.map(m => {
                  const rate = s.metrics[m].lostRate;
                  const isWinner = rate <= bestRate + 1e-9;
                  return (
                    <td key={m} style={{ ...cellStyle,
                      ...(isWinner ? { color: GREEN, fontWeight: 700 } : {})
                    }}>{rate.toFixed(2)}%</td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Summary stat cards ──────────────────────────────────────────────────── */
function StatCards({ stores, baseline }) {
  const more = stores.filter(s => (baseline === "before" ? s.diffBefore : s.diffAfter) > 0).length;
  const less = stores.filter(s => (baseline === "before" ? s.diffBefore : s.diffAfter) < 0).length;
  const same = stores.length - more - less;
  const cards = [
    { label: "Vol > Baseline", value: more, color: GREEN },
    { label: "Vol < Baseline", value: less, color: RED },
    { label: "Unchanged", value: same, color: "#888" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 16 }}>
      {cards.map((c, i) => (
        <div key={i} style={{ background: "#16162a", border: "1px solid #2a2a44",
                              borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
          <div style={{ fontSize: 24, fontWeight: 700, color: c.color }}>{c.value}</div>
          <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>{c.label}</div>
        </div>
      ))}
    </div>
  );
}

/* ── README toggle ───────────────────────────────────────────────────────── */
const sectionStyle = { marginBottom: 12 };
const headingStyle = { fontSize: 14, fontWeight: 700, color: "#ccd", margin: "14px 0 6px" };
const paraStyle = { fontSize: 13, color: "#aab", lineHeight: 1.7, margin: "4px 0" };
const codeStyle = { background: "#1a1a30", padding: "2px 6px", borderRadius: 4, fontSize: 12, color: "#dda" };
const tblStyle = { borderCollapse: "collapse", width: "100%", marginBottom: 8 };
const thStyle = { padding: "6px 10px", textAlign: "left", borderBottom: "2px solid #333",
                  fontSize: 13, fontWeight: 600, color: "#aab", background: "#16162a" };
const tdStyle = { padding: "6px 10px", textAlign: "left", borderBottom: "1px solid #2a2a44",
                  fontSize: 13, color: "#bbc" };

function ReadmeToggle() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginBottom: 20 }}>
      <button onClick={() => setOpen(!open)} style={{
        background: "#16162a", border: "1px solid #2a2a44", borderRadius: 8,
        padding: "8px 16px", cursor: "pointer", fontSize: 13, fontWeight: 600,
        color: "#aab", width: "100%", textAlign: "left", transition: "all 0.15s",
      }}>
        {open ? "▾" : "▸"} How it works — Methods & Demand Model
      </button>
      {open && (
        <div style={{ background: "#16162a", border: "1px solid #2a2a44", borderTop: "none",
                      borderRadius: "0 0 8px 8px", padding: "16px 20px" }}>
          <div style={sectionStyle}>
            <div style={headingStyle}>Policies compared</div>
            <table style={tblStyle}>
              <thead><tr><th style={thStyle}>Policy</th><th style={thStyle}>How it allocates</th></tr></thead>
              <tbody>
                <tr>
                  <td style={{ ...tdStyle, fontWeight: 600, color: ORANGE }}>WOS-before / WOS-after</td>
                  <td style={tdStyle}>Proportional to forecast means via WOS-balancing. Each unit goes to the store with the lowest <span style={codeStyle}>alloc/mean</span> (before) or <span style={codeStyle}>(alloc+1)/mean</span> (after) ratio.</td>
                </tr>
                <tr>
                  <td style={{ ...tdStyle, fontWeight: 600, color: BLUE }}>Volatility-aware</td>
                  <td style={tdStyle}>Greedy by marginal expected sales. Each unit goes to the store where <span style={codeStyle}>P(demand &ge; alloc+1)</span> is highest, using exact NB tail probabilities.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={sectionStyle}>
            <div style={headingStyle}>Demand model</div>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Store means</b> drawn from a Pareto distribution:
              <span style={codeStyle}> mean_i = mean_min * (1 + Pareto(alpha))</span>, clipped to [mean_min, mean_max].
              Smaller alpha = heavier tail = more extreme high-mean stores.
            </p>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Target store mean:</b> when set, all means are rescaled so their average equals this value while preserving the Pareto shape.
            </p>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Variance-to-mean ratio (VMR):</b>
              <span style={codeStyle}> VMR_i = 1 + coeff_i * mean_i</span>.
              Planning uses fixed coeff = 0.1. Realized coeff:
              <span style={codeStyle}> max(0, 0.1 + error_abs * u_i)</span> where u_i ~ U(-1,1).
              When VMR = 1, demand is Poisson.
            </p>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>D/F (demand bias):</b>
              <span style={codeStyle}> realized_mean = planned_mean * bias</span>.
              0.8 = demand 20% below forecast, 1.0 = perfect, 1.2 = 20% above.
            </p>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Constraint:</b> each store receives at least 1 unit.
            </p>
          </div>

          <div style={sectionStyle}>
            <div style={headingStyle}>Charts</div>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Delta chart:</b> per-store allocation difference (Vol-aware minus baseline), sorted by forecast mean ascending.
              Green = vol-aware allocates more, red = fewer. Shows how vol-aware redistributes stock from predictable high-mean stores toward volatile ones.
            </p>
            <p style={paraStyle}>
              <b style={{ color: "#ccd" }}>Allocation CDF:</b> cumulative distribution of per-store allocation sizes.
              Steeper curve = more equal distribution. Compares all three methods.
            </p>
          </div>

          <div style={sectionStyle}>
            <div style={headingStyle}>Evaluation</div>
            <p style={paraStyle}>
              All metrics are <b style={{ color: "#ccd" }}>exact analytical</b> — no Monte Carlo.
              E[Sold] = <span style={codeStyle}>sum_i E[min(D_i, alloc_i)]</span> computed via NB PMF summation.
              Lost Rate = proportion of expected demand not fulfilled.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── App ─────────────────────────────────────────────────────────────────── */
function App() {
  const [factorIdx, setFactorIdx] = useState(2);  // default 1.5x
  const [biasIdx, setBiasIdx] = useState(1);       // default 1.0
  const [baseline, setBaseline] = useState("before");

  const scenario = DATA.scenarios[factorIdx * DATA.biasRates.length + biasIdx];
  const stores = useMemo(() => enrichStores(scenario.stores), [scenario]);

  const params = DATA.params;
  const baselineLabel = baseline === "before" ? "WOS-before" : "WOS-after";

  return (
    <div>
      {/* Header */}
      <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0, color: "#f0f0ff",
                   borderBottom: "2px solid " + BLUE, paddingBottom: 8, marginBottom: 16 }}>
        Allocation Method Comparison Report
      </h1>

      {/* Parameter banner */}
      <div style={{ background: "#16162a", border: "1px solid #2a2a44", borderRadius: 8,
                    padding: "10px 16px", marginBottom: 20, fontSize: 13, color: "#99a",
                    display: "flex", flexWrap: "wrap", gap: "4px 20px" }}>
        <span><b style={{ color: "#bbc" }}>Stores:</b> {params.stores}</span>
        <span><b style={{ color: "#bbc" }}>Pareto α:</b> {params.paretoAlpha}</span>
        <span><b style={{ color: "#bbc" }}>Mean range:</b> [{params.meanMin}, {params.meanMax}]</span>
        <span><b style={{ color: "#bbc" }}>Target mean:</b> {params.targetStoreMean}</span>
        <span><b style={{ color: "#bbc" }}>Evaluation:</b> exact (analytical)</span>
      </div>

      {/* Method legend */}
      <div style={{ fontSize: 13, color: "#99a", marginBottom: 20, display: "flex", gap: 16 }}>
        <span><b>Methods:</b></span>
        {METHODS.map(m => (
          <span key={m}><span style={{ color: METHOD_COLORS[m], fontSize: 16 }}>■</span> {m}</span>
        ))}
      </div>

      {/* README toggle */}
      <ReadmeToggle />

      {/* Store Demand Profile (same across all scenarios) */}
      <DemandProfileChart data={DATA.demandProfile} nStores={params.stores} />

      {/* Scenario selectors */}
      <div style={{ display: "flex", gap: 24, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "#888" }}>Buy Qty:</span>
          {DATA.totalFactors.map((f, i) => (
            <Pill key={f} active={i === factorIdx} onClick={() => setFactorIdx(i)}>{f}x</Pill>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "#888" }}>D/F:</span>
          {DATA.biasRates.map((b, i) => (
            <Pill key={b} active={i === biasIdx} onClick={() => setBiasIdx(i)}>{b.toFixed(1)}</Pill>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "#888" }}>Delta vs:</span>
          {["before", "after"].map(b => (
            <Pill key={b} active={baseline === b} onClick={() => setBaseline(b)} color={ORANGE}>
              WOS-{b}
            </Pill>
          ))}
        </div>
      </div>

      {/* Scenario info */}
      <div style={{ fontSize: 13, color: "#888", marginBottom: 16 }}>
        Total stock: <b style={{ color: "#ccd" }}>{scenario.totalUnits}</b> ·
        Planned demand: {scenario.plannedDemand} ·
        Realized demand: {scenario.realizedDemand}
      </div>

      {/* Metrics table */}
      <MetricsTable metrics={scenario.metrics} />

      {/* Allocation Delta Chart */}
      <div style={{ background: "#12122266", borderRadius: 12, padding: "16px 8px",
                    border: "1px solid #1e1e36", marginBottom: 20, marginTop: 20 }}>
        <DeltaChart stores={stores} baseline={baseline}
          title={`Allocation Delta (Vol − ${baselineLabel}) — Buy Qty ${scenario.totalFactor}x | D/F ${scenario.biasLabel}`} />
      </div>

      {/* Stat cards */}
      <StatCards stores={stores} baseline={baseline} />

      {/* Lost Sales Delta Chart */}
      <div style={{ background: "#12122266", borderRadius: 12, padding: "16px 8px",
                    border: "1px solid #1e1e36", marginBottom: 20, marginTop: 20 }}>
        <LostSalesDeltaChart stores={stores} baseline={baseline}
          title={`Lost Sales Delta (Baseline − Vol) — Buy Qty ${scenario.totalFactor}x | D/F ${scenario.biasLabel}`} />
      </div>

      {/* CDF Chart */}
      <div style={{ background: "#12122266", borderRadius: 12, padding: "16px 8px",
                    border: "1px solid #1e1e36", marginBottom: 20 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "#ccd", marginBottom: 4, textAlign: "center" }}>
          Allocation CDF — Buy Qty {scenario.totalFactor}x | D/F {scenario.biasLabel}
        </div>
        <CDFChart stores={stores} />
      </div>

      {/* Summary table — always at bottom */}
      <SummaryTable scenarios={DATA.scenarios} />

      <div style={{ marginTop: 32, fontSize: 12, color: "#666", textAlign: "center", lineHeight: 1.8 }}>
        <div>Interactive version: <a href="https://sim-experiments-8ketm5e4zs8wzwvgck5fyv.streamlit.app/"
          style={{ color: BLUE }} target="_blank" rel="noopener noreferrer">Streamlit App</a></div>
        <div style={{ fontSize: 11, color: "#444" }}>Generated with Python · React · Recharts</div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    means = generate_store_means(
        n_stores=STORES,
        seed=SEED,
        pareto_alpha=PARETO_ALPHA,
        mean_min=MEAN_MIN,
        mean_max=MEAN_MAX,
        target_store_mean=TARGET_STORE_MEAN,
    )
    coeff_plan = np.full(STORES, 0.1)
    vmr_plan = build_vmr(means, coeff_plan)

    data = build_report_data(means, vmr_plan)
    json_data = json.dumps(data, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__JSON_DATA__", json_data)

    out = Path(__file__).with_name("report.html")
    out.write_text(html, encoding="utf-8")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
