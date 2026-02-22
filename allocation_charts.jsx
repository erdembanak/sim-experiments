import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ReferenceLine, LineChart, Line, ResponsiveContainer, Cell, Label } from "recharts";

const RAW_DATA = {"means":[0.463,0.464,0.466,0.468,0.469,0.47,0.471,0.485,0.488,0.491,0.498,0.501,0.507,0.514,0.522,0.522,0.523,0.541,0.551,0.558,0.566,0.569,0.572,0.594,0.598,0.599,0.604,0.623,0.624,0.627,0.642,0.671,0.692,0.696,0.706,0.724,0.727,0.737,0.754,0.755,0.759,0.799,0.81,0.826,0.83,0.85,0.851,0.866,0.872,0.884,0.894,0.901,0.957,0.971,0.982,0.985,0.992,0.993,1.002,1.006,1.01,1.039,1.076,1.111,1.122,1.158,1.161,1.165,1.17,1.182,1.249,1.285,1.333,1.333,1.433,1.458,1.506,1.55,1.56,1.808,1.836,2.018,2.104,2.347,2.368,2.73,2.783,3.106,3.236,4.028,4.608,4.799,4.992,5.294,5.993,6.669,7.087,8.723,12.479,46.011],"wos_before":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,6,6,7,7,7,8,9,10,12,16,59],"wos_after":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,5,5,6,7,7,8,8,9,11,11,14,20,76],"vol":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,6,7,7,7,7,8,9,10,12,16,58]};

// Build per-store data sorted by mean
const storeData = RAW_DATA.means.map((m, i) => ({
  idx: i,
  mean: m,
  wosBefore: RAW_DATA.wos_before[i],
  wosAfter: RAW_DATA.wos_after[i],
  vol: RAW_DATA.vol[i],
  diffVolBefore: RAW_DATA.vol[i] - RAW_DATA.wos_before[i],
  diffVolAfter: RAW_DATA.vol[i] - RAW_DATA.wos_after[i],
}));

// Scatter data: WOS-before vs Vol
const scatterBefore = storeData.map(d => ({ x: d.wosBefore, y: d.vol, mean: d.mean }));
const scatterAfter = storeData.map(d => ({ x: d.wosAfter, y: d.vol, mean: d.mean }));

// QQ data
function qqData(a, b) {
  const sa = [...a].sort((x,y) => x - y);
  const sb = [...b].sort((x,y) => x - y);
  return sa.map((v, i) => ({ baseline: v, vol: sb[i], idx: i }));
}
const qqBefore = qqData(RAW_DATA.wos_before, RAW_DATA.vol);
const qqAfter = qqData(RAW_DATA.wos_after, RAW_DATA.vol);

const TABS = [
  { key: "delta", label: "① Delta Chart" },
  { key: "scatter", label: "② Paired Scatter" },
  { key: "butterfly", label: "③ Butterfly Bars" },
  { key: "qq", label: "④ QQ Plot" },
];

const BLUE = "#4c78a8";
const RED = "#e45756";
const ORANGE = "#f58518";
const GREEN = "#59a14f";

const CustomTooltipDelta = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #444", borderRadius: 6, padding: "8px 12px", fontSize: 12, color: "#e0e0e0" }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Store {d.idx} · Mean: {d.mean.toFixed(2)}</div>
      <div>WOS-before: {d.wosBefore} → Vol: {d.vol} → <span style={{ color: d.diffVolBefore > 0 ? GREEN : d.diffVolBefore < 0 ? RED : "#aaa", fontWeight: 700 }}>Δ {d.diffVolBefore > 0 ? "+" : ""}{d.diffVolBefore}</span></div>
    </div>
  );
};

const CustomTooltipScatter = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #444", borderRadius: 6, padding: "8px 12px", fontSize: 12, color: "#e0e0e0" }}>
      <div>Mean: {d.mean.toFixed(2)}</div>
      <div>Baseline alloc: {d.x} · Vol alloc: {d.y}</div>
    </div>
  );
};

const CustomTooltipQQ = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #444", borderRadius: 6, padding: "8px 12px", fontSize: 12, color: "#e0e0e0" }}>
      <div>Quantile #{d.idx + 1}</div>
      <div>Baseline: {d.baseline} · Vol-aware: {d.vol}</div>
    </div>
  );
};

function DeltaChart({ baseline }) {
  const data = storeData.map(d => ({
    ...d,
    diff: baseline === "before" ? d.diffVolBefore : d.diffVolAfter,
  }));
  return (
    <ResponsiveContainer width="100%" height={420}>
      <BarChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="idx" tick={{ fontSize: 9, fill: "#999" }} label={{ value: "Store (sorted by mean →)", position: "insideBottom", offset: -10, style: { fill: "#bbb", fontSize: 13 } }} />
        <YAxis tick={{ fill: "#999", fontSize: 11 }} label={{ value: "Δ Units (Vol − Baseline)", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#bbb", fontSize: 13 } }} />
        <Tooltip content={<CustomTooltipDelta />} />
        <ReferenceLine y={0} stroke="#888" strokeWidth={1.5} />
        <Bar dataKey="diff" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.diff > 0 ? GREEN : d.diff < 0 ? RED : "#555"} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function PairedScatter({ baseline }) {
  const data = baseline === "before" ? scatterBefore : scatterAfter;
  const maxVal = Math.max(...data.map(d => Math.max(d.x, d.y))) + 2;
  return (
    <ResponsiveContainer width="100%" height={440}>
      <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis type="number" dataKey="x" domain={[0, maxVal]} tick={{ fill: "#999", fontSize: 11 }} label={{ value: `${baseline === "before" ? "WOS-before" : "WOS-after"} allocation`, position: "insideBottom", offset: -10, style: { fill: "#bbb", fontSize: 13 } }} />
        <YAxis type="number" dataKey="y" domain={[0, maxVal]} tick={{ fill: "#999", fontSize: 11 }} label={{ value: "Volatility-aware allocation", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#bbb", fontSize: 13 } }} />
        <Tooltip content={<CustomTooltipScatter />} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: maxVal, y: maxVal }]} stroke="#f5c542" strokeDasharray="6 3" strokeWidth={1.5} />
        <Scatter data={data} fill={BLUE} fillOpacity={0.7} r={5} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function ButterflyChart({ baseline }) {
  // Only show stores where there's a difference, plus a few neighbours, sampled for readability
  const pick = storeData.filter((_, i) => i % 3 === 0 || i >= 85);
  const data = pick.map(d => ({
    label: `#${d.idx} (μ=${d.mean.toFixed(1)})`,
    baselineVal: -(baseline === "before" ? d.wosBefore : d.wosAfter),
    volVal: d.vol,
    mean: d.mean,
  })).reverse();

  return (
    <ResponsiveContainer width="100%" height={Math.max(500, data.length * 18)}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, bottom: 20, left: 100 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis type="number" tick={{ fill: "#999", fontSize: 11 }}>
          <Label value="← Baseline | Volatility-aware →" position="insideBottom" offset={-5} style={{ fill: "#bbb", fontSize: 12 }} />
        </XAxis>
        <YAxis type="category" dataKey="label" tick={{ fill: "#ccc", fontSize: 9 }} width={90} />
        <Tooltip formatter={(v) => Math.abs(v)} />
        <ReferenceLine x={0} stroke="#888" strokeWidth={1.5} />
        <Bar dataKey="baselineVal" fill={ORANGE} fillOpacity={0.8} name="Baseline" radius={[4, 0, 0, 4]} />
        <Bar dataKey="volVal" fill={BLUE} fillOpacity={0.8} name="Vol-aware" radius={[0, 4, 4, 0]} />
        <Legend />
      </BarChart>
    </ResponsiveContainer>
  );
}

function QQPlot({ baseline }) {
  const data = baseline === "before" ? qqBefore : qqAfter;
  const maxVal = Math.max(...data.map(d => Math.max(d.baseline, d.vol))) + 2;
  return (
    <ResponsiveContainer width="100%" height={440}>
      <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis type="number" dataKey="baseline" domain={[0, maxVal]} tick={{ fill: "#999", fontSize: 11 }} label={{ value: `${baseline === "before" ? "WOS-before" : "WOS-after"} quantiles`, position: "insideBottom", offset: -10, style: { fill: "#bbb", fontSize: 13 } }} />
        <YAxis type="number" dataKey="vol" domain={[0, maxVal]} tick={{ fill: "#999", fontSize: 11 }} label={{ value: "Volatility-aware quantiles", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#bbb", fontSize: 13 } }} />
        <Tooltip content={<CustomTooltipQQ />} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: maxVal, y: maxVal }]} stroke="#f5c542" strokeDasharray="6 3" strokeWidth={1.5} />
        <Scatter data={data} fill="#e07bef" fillOpacity={0.75} r={5} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

const descriptions = {
  delta: "Each bar = Vol-aware allocation minus baseline allocation for that store. Green bars: volatility-aware gives MORE units. Red bars: gives FEWER. Stores sorted by forecast mean (left = small, right = large).",
  scatter: "Each dot = one store. X = baseline allocation, Y = vol-aware allocation. Dots above the yellow 45° line → vol-aware gives more to that store. Dots cluster along the line with deviations in mid-range stores.",
  butterfly: "Baseline extends left (orange), vol-aware extends right (blue). Sampled stores sorted by mean. Direct side-by-side comparison of allocation magnitude per store.",
  qq: "Quantile-quantile plot of the two allocation distributions. Points on the yellow line = identical distributions. Deviations show where the distributions diverge — typically in the tails.",
};

export default function App() {
  const [tab, setTab] = useState("delta");
  const [baseline, setBaseline] = useState("before");

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", color: "#e8e8f0", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", padding: "24px 20px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0, letterSpacing: "-0.5px", color: "#f0f0ff" }}>
            Allocation Comparison · 4 Views
          </h1>
          <p style={{ fontSize: 13, color: "#888", margin: "6px 0 0 0" }}>
            Buy Qty 1.5× · D/F 1.0 · 100 stores · 300 total units · Target mean 2.0
          </p>
        </div>

        {/* Baseline toggle */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "#888", marginRight: 4 }}>Compare vs:</span>
          {["before", "after"].map(b => (
            <button
              key={b}
              onClick={() => setBaseline(b)}
              style={{
                padding: "5px 14px", borderRadius: 6, border: "none", cursor: "pointer",
                fontSize: 12, fontWeight: 600, fontFamily: "inherit",
                background: baseline === b ? ORANGE : "#222238",
                color: baseline === b ? "#000" : "#aaa",
                transition: "all 0.15s",
              }}
            >
              WOS-{b}
            </button>
          ))}
        </div>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
          {TABS.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                padding: "8px 16px", borderRadius: 8, border: "none", cursor: "pointer",
                fontSize: 13, fontWeight: 600, fontFamily: "inherit",
                background: tab === t.key ? BLUE : "#1a1a30",
                color: tab === t.key ? "#fff" : "#7a7a9a",
                transition: "all 0.15s",
                boxShadow: tab === t.key ? `0 2px 12px ${BLUE}44` : "none",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Description */}
        <div style={{ background: "#16162a", border: "1px solid #2a2a44", borderRadius: 8, padding: "12px 16px", marginBottom: 20, fontSize: 12, color: "#aab", lineHeight: 1.6 }}>
          {descriptions[tab]}
        </div>

        {/* Chart */}
        <div style={{ background: "#12122266", borderRadius: 12, padding: "16px 8px", border: "1px solid #1e1e36" }}>
          {tab === "delta" && <DeltaChart baseline={baseline} />}
          {tab === "scatter" && <PairedScatter baseline={baseline} />}
          {tab === "butterfly" && <ButterflyChart baseline={baseline} />}
          {tab === "qq" && <QQPlot baseline={baseline} />}
        </div>

        {/* Summary stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 20 }}>
          {[
            { label: "Stores where Vol > Baseline", value: storeData.filter(d => (baseline === "before" ? d.diffVolBefore : d.diffVolAfter) > 0).length },
            { label: "Stores where Vol < Baseline", value: storeData.filter(d => (baseline === "before" ? d.diffVolBefore : d.diffVolAfter) < 0).length },
            { label: "Stores unchanged", value: storeData.filter(d => (baseline === "before" ? d.diffVolBefore : d.diffVolAfter) === 0).length },
          ].map((s, i) => (
            <div key={i} style={{ background: "#16162a", border: "1px solid #2a2a44", borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: i === 0 ? GREEN : i === 1 ? RED : "#888" }}>{s.value}</div>
              <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
