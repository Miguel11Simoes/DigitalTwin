import React, { useEffect, useMemo, useState, useCallback } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Decimation
} from "chart.js";
import Twin3D from "./Twin3D";
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Decimation);
const API_BASE = "http://127.0.0.1:8000";
/* ---------- CONSTS FORA DO COMPONENTE ---------- */
const AVG_KEYS = ["10s", "1min", "1h", "24h", "all"];
const ML_KEYS = [
  "anomaly_score",
  "failure_probability",
  "health_index",
  "rul_minutes",
  "model_confidence",
];
// Map from base failure mode → recommended operation
export const MODE_TO_ACTION = {
  // Fluid/Hydraulic
  cavitation: "Check suction pressure, reduce pump speed",
  air_entrapment: "Bleed/vent air; check suction leaks",
  vapor_lock: "Cool/pressurize suction; purge vapor",
  low_flow: "Open valves / clean strainers",
  high_flow: "Throttle discharge valve / reduce speed",
  high_pressure: "Reduce throttling / remove restrictions",
  low_pressure: "Increase NPSH / open suction valve",
  npsh_insufficient: "Raise tank level / reduce suction losses",
  recirculation: "Open discharge / move away from low-flow",
  surge: "Retune control loop; avoid surge region",
  impeller_wear: "Inspect & replace impeller",
  fluid_contaminated: "Replace/flush fluid; clean system",
  fluid_mismatch: "Load correct fluid grade/spec",
  gas_overheat: "Improve cooling / reduce compression ratio",
  // Mechanical
  bearing_wear: "Lubricate or replace bearings",
  bearing_unlubricated: "Re-lubricate bearings; check regime",
  misalignment: "Realign motor and pump shaft",
  unbalance: "Balance rotating components",
  looseness: "Tighten fasteners; check fits",
  shaft_bent: "Inspect shaft; straighten/replace",
  seal_leak: "Inspect and replace mechanical seal",
  seal_dry: "Restore seal flush; verify lubrication",
  structural_fault: "Inspect supports/baseplate; stiffen structure",
  // Electrical / Motor
  motor_overload: "Reduce load, check electrical supply",
  electrical_fault: "Inspect motor windings and connections",
  single_phasing: "Restore missing phase; check protection",
  vfd_issue: "Check VFD params/filters; EMC/harmonics",
  insulation_breakdown: "Test insulation; dry/repair motor",
  bearing_fluting: "Install shaft grounding; fix VFD common-mode",
  // Lube / Ambient
  oil_degraded: "Change oil and check contamination",
  oil_wrong: "Fill correct oil grade/viscosity",
  overtemperature: "Reduce load; improve cooling",
  under_temperature: "Warm-up process; reduce viscosity",
  ambient_hot: "Improve ventilation/ambient cooling",
  ambient_cold: "Preheat fluid; insulate lines",
};
const BASE_KEYS = Object.keys(MODE_TO_ACTION);
function extractBaseKeysFromString(modeString) {
  if (!modeString) return [];
  const m = String(modeString).toLowerCase();
  const hits = BASE_KEYS.filter(b => m.includes(b));
  if (!hits.length && m !== "normal") return [m.split("_")[0]];
  return Array.from(new Set(hits));
}
function buildActionForMode(modeString) {
  const bases = extractBaseKeysFromString(modeString);
  if (!bases.length) return null;
  const actions = bases.map(b => MODE_TO_ACTION[b]).filter(Boolean);
  if (!actions.length) return null;
  return Array.from(new Set(actions)).join(" + ");
}

// === Prioridade de segurança/urgência para ordenar os modos detectados ===
const SAFETY_PRIORITY = [
  // hidráulica (risco de dano rápido)
  "cavitation", "npsh_insufficient", "surge", "low_pressure",
  "seal_leak", "vapor_lock", "high_pressure",
  // mecânica crítica
  "bearing_unlubricated", "bearing_wear", "shaft_bent", "looseness",
  "misalignment", "unbalance", "structural_fault", "bearing_fluting",
  // elétrica
  "single_phasing", "motor_overload", "electrical_fault", "insulation_breakdown", "vfd_issue",
  // processo/fluido
  "impeller_wear", "recirculation", "low_flow", "high_flow",
  "fluid_mismatch", "fluid_contaminated", "gas_overheat",
  // ambiente/lube
  "overtemperature", "oil_wrong", "oil_degraded", "under_temperature", "ambient_hot", "ambient_cold"
];

function orderByPriority(modes) {
  const rank = new Map(SAFETY_PRIORITY.map((m, i) => [m, i]));
  return [...new Set(modes)].sort((a, b) => (rank.get(a) ?? 9e9) - (rank.get(b) ?? 9e9));
}
function estimateSeverity(mode, L, A) {
  const val = (k, def = null) => (L?.[k] ?? A?.[k] ?? def);
  const us = val("ultrasonic_noise", 0);
  const ov = val("overall_vibration", 0);
  const vx = val("vibration_x", 0), vy = val("vibration_y", 0), vz = val("vibration_z", 0);
  const f  = val("flow", null);
  const p  = val("pressure", null);
  const sp = val("suction_pressure", null);
  const dp = val("delta_p", null);
  const dpres = val("discharge_pressure", null);
  const cur= val("motor_current", 0);
  const den= val("density", null);
  const vis= val("viscosity", null);
  const ft = val("ferrous_particles", 0);
  const t  = val("temperature", null);
  const freq = val("frequency", 50);
  const iA = val("current_A", 0), iB = val("current_B", 0), iC = val("current_C", 0);

  let s = 0; // 0..3 ~ early..severe
  switch (mode) {
    case "cavitation":
      if (us > 75) s++;
      if (ov > 0.05 || vz > 0.05) s++;
      if ((f != null && f < 6) || (sp != null && sp < 0.2)) s++;
      break;
    case "npsh_insufficient":
      if (us > 72) s++;
      if (f != null && f < 8) s++;
      if (sp != null && sp < 0.25) s++;
      break;
    case "surge":
      if ((p != null && p > 6.0) || (dp != null && dp > 9.0)) s++;
      if (ov > 0.05) s++;
      if (ov > 0.06) s++;
      break;
    case "low_pressure":
      if (p != null && p < 0.9) s++;
      if (sp != null && sp < 0.2) s++;
      if (f != null && f < 7) s++;
      break;
    case "seal_leak":
      if (us > 65) s++;
      if (f != null && f < 10) s++;
      if (p != null && p < 2.5) s++;
      break;
    case "vapor_lock":
      if (f != null && f < 4.5) s++;
      if ((p != null && p < 0.8) || (sp != null && sp < 0.2)) s++;
      if (cur < 6.5) s++;
      break;
    case "high_pressure":
      if ((dpres != null && dpres > 9.5) || (dp != null && dp > 9.0)) s++;
      if (p != null && p > 6.0) s++;
      if (ov > 0.05) s++;
      break;
    case "impeller_wear":
      if (f != null && f < 7.5) s++;
      if (dp != null && dp < 4.0) s++;
      if (ov > 0.035) s++;
      break;
    case "recirculation":
      if (ov > 0.05) s++;
      if (f != null && f < 8) s++;
      if (p != null && p < 2.5) s++;
      break;
    case "low_flow":
      if (f != null && f < 6) s++;
      if (dp != null && dp > 8.0) s++;
      break;
    case "high_flow":
      if (f != null && f > 24) s++;
      if ((p != null && p < 2.8) || (dp != null && dp < 3.8)) s++;
      break;
    case "fluid_contaminated":
      if (vis != null && vis > 2.2) s++;
      if (ft > 20) s++;
      if (ov > 0.04) s++;
      break;
    case "fluid_mismatch":
      if (vis != null && vis > 2.5) s++;
      if (den != null && den < 980) s++;
      if (f != null && f < 9) s++;
      if (cur > 11) s++;
      break;
    case "gas_overheat":
      if (den != null && den < 975) s++;
      if (t != null && t > 95) s++;
      break;

    // mecânica
    case "bearing_unlubricated":
      if (t != null && t > 95) s++;
      if (ov > 0.05) s++;
      if (ft > 30) s++;
      break;
    case "bearing_wear":
      if ((vx > 0.025 && vy > 0.025) || ov > 0.05) s++;
      if (ft > 25) s++;
      break;
    case "shaft_bent":
      if (vz > 0.035) s++;
      if (ov > 0.055) s++;
      if (cur > 12) s++;
      break;
    case "looseness":
      if (ov > 0.06) s += 2;
      else if (ov > 0.055) s++;
      break;
    case "misalignment":
      if (vx > 0.03) s++;
      if (ov > 0.05) s++;
      if (cur > 12.5) s++;
      break;
    case "unbalance":
      if (vy > 0.03) s++;
      if (ov > 0.05) s++;
      break;
    case "structural_fault":
      if (ov > 0.065) s += 2;
      else if (ov > 0.06) s++;
      break;
    case "bearing_fluting":
      if (ov > 0.055) s++;
      if (ft > 28) s++;
      break;

    // elétrica
    case "single_phasing":
      if (iA > 15.5 && iB < 4.5 && iC > 15.5) s += 2;
      else s++;
      if (ov > 0.045) s++;
      break;
    case "motor_overload":
      if (cur > 15) s++;
      if (t != null && t > 90) s++;
      break;
    case "electrical_fault":
      if (cur > 13) s++;
      if (ov > 0.05) s++;
      if (t != null && t > 85) s++;
      break;
    case "insulation_breakdown":
      if (cur > 13) s++;
      if (t != null && t > 95) s++;
      break;
    case "vfd_issue":
      if (Math.abs(freq - 50) > 0.4) s++;
      if (ov > 0.05) s++;
      break;

    // lub/ambiente
    case "overtemperature":
      if (t != null && t > 100) s += 2;
      else if (t != null && t > 97) s++;
      break;
    case "oil_wrong":
      if (vis != null && vis > 1.8) s++;
      if (t != null && t > 85) s++;
      break;
    case "oil_degraded":
      if (vis != null && vis < 0.65) s++;
      if (t != null && t > 90) s++;
      if (ov > 0.04) s++;
      break;
    case "under_temperature":
      if (t != null && t < 45) s++;
      if (vis != null && vis > 1.6) s++;
      if (f != null && f < 10) s++;
      break;
    case "ambient_hot":
      if (t != null && t > 95) s++;
      break;
    case "ambient_cold":
      if (t != null && t < 45) s++;
      break;
    default:
      s = 0;
  }

  if (s >= 3) return "severe";
  if (s === 2) return "moderate";
  return "early";
}


/* Orbitron */
function useOrbitronFont() {
  useEffect(() => {
    const id = "orbitron-font";
    if (!document.getElementById(id)) {
      const link = document.createElement("link");
      link.id = id;
      link.rel = "stylesheet";
      link.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;900&display=swap";
      document.head.appendChild(link);
    }
  }, []);
}
/* ---------- helpers: média/score ---------- */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function scoreBand(value, lo, hi, bestLo, bestHi) {
  if (value >= bestLo && value <= bestHi) return 1;
  if (value < bestLo) {
    if (bestLo === lo) return 0;
    const t = (value - lo) / (bestLo - lo);
    return clamp(t, 0, 1);
  }
  if (bestHi === hi) return 0;
  const t = (hi - value) / (hi - bestHi);
  return clamp(t, 0, 1);
}
function scoreLowerBetter(value, lo, hi, bestHi) {
  if (value <= bestHi) return 1;
  const t = (hi - value) / (hi - bestHi);
  return clamp(t, 0, 1);
}

// keep trendSlope above for simple trend calculations on arrays of values.
/* ============================== COMPONENTE ============================== */
export default function App() {
  useOrbitronFont();
  /* Histórico e leituras */
  const [history, setHistory] = useState([]);
  const [range, setRange] = useState("all");
  const [last, setLast] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [ops, setOps] = useState([]);
  const [currentMode, setCurrentMode] = useState("normal");
  // --- required by the JSX button / suggestedOperation display ---
  const [suggestedOperation, setSuggestedOperation] = useState(null);
  // apply button status: 'idle' | 'pending' | 'applied'
  const [applyStatus, setApplyStatus] = useState("idle");
  const applyTimerRef = React.useRef(null);
  const postOperation = useCallback(async (actionText, modeHint) => {
    try {
      const payload = {
        action: actionText,
        mode: modeHint || currentMode || "normal",
        ts: Date.now(),
      };
      const res = await fetch(`${API_BASE}/operations/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json().catch(() => null);
      if (!res.ok) {
        console.error("postOperation failed", res.status, json);
        return json;
      }
      // refresh operations list
      try {
        const r2 = await fetch(`${API_BASE}/operations?range=history`);
        if (r2.ok) setOps(await r2.json());
      } catch (e) { console.warn("refresh ops failed", e); }
      // update UI if backend confirms correction
      if (json && json.corrected) {
        setCurrentMode(json.new_mode || "normal");
      }
      return json;
    } catch (e) {
      console.error("postOperation error", e);
      return null;
    }
  }, [currentMode, setOps, setCurrentMode]);
  const [avgSets, setAvgSets] = useState({ "10s": [], "1min": [], "1h": [], "24h": [], "all": [] });
  // waveform/FFT
  const [fs, setFs] = useState(10000);
  const [duration, setDuration] = useState(2);
  const [wave, setWave] = useState(null);

  // --- Latch/hysteresis ---
const MIN_SUGGEST_MS = 10000; // 10s fixo visível
const MIN_ALERT_MS   = 6000;  // 6s cada alerta visível

const suggestionExpiryRef = React.useRef(0);
const lastSuggestedRef    = React.useRef(null);
const alertExpiryRef      = React.useRef({}); // { [msg]: expiryTs }


  /* ---------- Fetchers ---------- */
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/history?range=${range}`);
        setHistory(await res.json());
      } catch (e) { console.error(e); }
    };
    fetchHistory();
    const id = setInterval(fetchHistory, 5000);
    return () => clearInterval(id);
  }, [range]);
  // AJUSTE 1: deps vazias; AVG_KEYS está fora e não muda
  useEffect(() => {
    const fetchAllAverages = async () => {
      try {
        const resps = await Promise.all(AVG_KEYS.map(rk => fetch(`${API_BASE}/history?range=${rk}`)));
        const datas = await Promise.all(resps.map(r => r.json()));
        setAvgSets({ "10s": datas[0], "1min": datas[1], "1h": datas[2], "24h": datas[3], "all": datas[4] });
      } catch (e) { console.error(e); }
    };
    fetchAllAverages();
    const id = setInterval(fetchAllAverages, 5000);
    return () => clearInterval(id);
  }, []);
  // Alimentar histórico no backend
  useEffect(() => {
    const fetchSensor = async () => {
      try {
        const res = await fetch(`${API_BASE}/sensors`);
        setLast(await res.json());
      } catch (e) { console.error(e); }
    };
    fetchSensor();
    const id = setInterval(fetchSensor, 500);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await fetch(`${API_BASE}/alerts`);
        setAlerts(await res.json());
      } catch (e) { console.error(e); }
    };
    fetchAlerts();
    const id = setInterval(fetchAlerts, 5000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const fetchOps = async () => {
      try {
        const res = await fetch(`${API_BASE}/operations?range=history`);
        setOps(await res.json());
      } catch (e) { console.error(e); }
    };
    fetchOps();
    const id = setInterval(fetchOps, 15000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
  let stop = false;
  const poll = async () => {
    try {
      const res = await fetch(`${API_BASE}/mode`);
      if (!stop && res.ok) {
        const data = await res.json();
        setCurrentMode(data?.mode || "normal");
      }
    } catch (e) {
      console.error(e);
    }
  };
  poll();
  const id = setInterval(poll, 2000);
  return () => { stop = true; clearInterval(id); };
}, []);

  /* ---------- Eixos/labels ---------- */
  const axisMap = useMemo(() => ({
    temperature: "y", pressure: "y", flow: "y", motor_current: "y",
    vibration_x: "yVib", vibration_y: "yVib", vibration_z: "yVib", overall_vibration: "yVib",
    density: "yProc", viscosity: "yProc", ultrasonic_noise: "yProc", ferrous_particles: "yProc",
    rpm: "yProc",
    bearing_temp_DE: "yProc", bearing_temp_NDE: "yProc", casing_temp: "yProc",
    suction_pressure: "yProc", discharge_pressure: "yProc", delta_p: "yProc",
    gas_volume_fraction: "yProc",
    current_A: "yProc", current_B: "yProc", current_C: "yProc", power_factor: "yProc",
    frequency: "yProc", torque_est: "yProc",
    oil_temp: "yProc", oil_water_ppm: "yProc", particle_count: "yProc", oil_TAN: "yProc",
    seal_temp: "yProc", seal_flush_pressure: "yProc", leakage_rate: "yProc",
    shaft_displacement: "yProc", noise_dBA: "yProc",
    anomaly_score: "yProc", failure_probability: "yProc", health_index: "yProc", rul_minutes: "yProc", model_confidence: "yProc"
  }), []);
  const prettyLabel = useMemo(() => ({
    temperature: "Temperature (°C)",
    pressure: "Pressure (bar)",
    flow: "Flow (m³/h)",
    motor_current: "Motor current (A)",
    vibration_x: "Vib X (g)",
    vibration_y: "Vib Y (g)",
    vibration_z: "Vib Z (g)",
    overall_vibration: "Max Vib (g RMS)",
    density: "Density (kg/m³)",
    viscosity: "Viscosity (cP)",
    ultrasonic_noise: "Ultrasonic (dB)",
    ferrous_particles: "Ferrous particles (count/ml)",
    rpm: "RPM",
    bearing_temp_DE: "Bearing temp DE (°C)",
    bearing_temp_NDE: "Bearing temp NDE (°C)",
    casing_temp: "Casing temp (°C)",
    suction_pressure: "Suction pressure (bar)",
    discharge_pressure: "Discharge pressure (bar)",
    delta_p: "ΔP (bar)",
    gas_volume_fraction: "Gas volume fraction (-)",
    current_A: "Phase A current (A)",
    current_B: "Phase B current (A)",
    current_C: "Phase C current (A)",
    power_factor: "Power factor (-)",
    frequency: "Frequency (Hz)",
    torque_est: "Torque est (Nm)",
    oil_temp: "Oil temp (°C)",
    oil_water_ppm: "Oil water (ppm)",
    particle_count: "Particle count (ISO idx)",
    oil_TAN: "Oil TAN (mgKOH/g)",
    seal_temp: "Seal temp (°C)",
    seal_flush_pressure: "Seal flush pressure (bar)",
    leakage_rate: "Leakage rate (ml/min)",
    shaft_displacement: "Shaft displacement (µm)",
    noise_dBA: "Noise (dBA)",
    anomaly_score: "Anomaly score (0-1)",
    failure_probability: "Failure probability (0-1)",
    health_index: "Health index (0-100)",
    rul_minutes: "RUL (min)",
    model_confidence: "Model confidence (0-1)"
  }), []);
  const metricKeys = useMemo(() => ([
    "temperature","pressure","flow","motor_current",
    "vibration_x","vibration_y","vibration_z","overall_vibration",
    "density","viscosity","ultrasonic_noise","ferrous_particles","rpm",
    "bearing_temp_DE","bearing_temp_NDE","casing_temp",
    "suction_pressure","discharge_pressure","delta_p",
    "gas_volume_fraction",
    "current_A","current_B","current_C","power_factor","frequency","torque_est",
    "oil_temp","oil_water_ppm","particle_count","oil_TAN",
    "seal_temp","seal_flush_pressure","leakage_rate",
    "shaft_displacement","noise_dBA",
  ]), []);
  const chartData = useMemo(() => {
    const colors = i => `hsl(${(i * 37) % 360}, 80%, 45%)`;
    return {
      labels: history.map(r => new Date(r.timestamp).toLocaleTimeString()),
      datasets: metricKeys
        .filter(k => history.some(h => typeof h[k] === "number"))
        .map((k, i) => {
          const isVib = k === "vibration_x" || k === "vibration_y" || k === "vibration_z" || k === "overall_vibration";
          const visibleByDefault = (k === "temperature") || isVib;
          return {
            label: prettyLabel[k] || k,
            data: history.map(r => (typeof r[k] === "number" ? r[k] : null)),
            borderColor: colors(i),
            borderWidth: 2,
            fill: false,
            yAxisID: axisMap[k] || "y",
            tension: 0.25,
            pointRadius: 0,
            hidden: !visibleByDefault,
          };
        }),
    };
  }, [history, axisMap, metricKeys, prettyLabel]);
  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Sensor History" },
      decimation: { enabled: true, algorithm: "largest-triangle-three-buckets" }
    },
    scales: {
      y: { position: "left", title: { display: true, text: "Process" } },
      yVib: { position: "right", title: { display: true, text: "Vibration (g)" }, grid: { drawOnChartArea: false } },
      yProc: { position: "right", title: { display: true, text: "Other metrics" }, grid: { drawOnChartArea: false }, offset: true },
    },
  };
  /* ---------- Médias ---------- */
  const decimals = useMemo(() => ({
    temperature:2, pressure:2, flow:2, motor_current:2,
    density:1, viscosity:2, ultrasonic_noise:2, ferrous_particles:0,
    vibration_x:4, vibration_y:4, vibration_z:4, overall_vibration:4,
    rpm:0,
    bearing_temp_DE:2, bearing_temp_NDE:2, casing_temp:2,
    suction_pressure:2, discharge_pressure:2, delta_p:2,
    gas_volume_fraction:3,
    current_A:2, current_B:2, current_C:2, power_factor:3,
    frequency:2, torque_est:1,
    oil_temp:2, oil_water_ppm:0, particle_count:1, oil_TAN:2,
    seal_temp:2, seal_flush_pressure:2, leakage_rate:1,
    shaft_displacement:1, noise_dBA:1,
    anomaly_score:3, failure_probability:3, health_index:1, rul_minutes:0, model_confidence:3
  }), []);
  const meanFor = (arr, key) => {
    const v = arr.map(r => r[key]).filter(x => typeof x === "number");
    if (!v.length) return null;
    return v.reduce((a,b)=>a+b,0)/v.length;
  };
  const avgRows = useMemo(() => {
    return [
      "temperature","pressure",
      "vibration_x","vibration_y","vibration_z","overall_vibration",
      "flow","density","viscosity","ultrasonic_noise","ferrous_particles","motor_current","rpm",
      "bearing_temp_DE","bearing_temp_NDE","casing_temp",
      "suction_pressure","discharge_pressure","delta_p",
      "gas_volume_fraction",
      "current_A","current_B","current_C","power_factor","frequency","torque_est",
      "oil_temp","oil_water_ppm","particle_count","oil_TAN",
      "seal_temp","seal_flush_pressure","leakage_rate",
      "shaft_displacement","noise_dBA",
    ].map(k => {
      const vals = {};
      for (const rk of AVG_KEYS) {
        const m = meanFor(avgSets[rk] || [], k);
        vals[rk] = m == null ? "-" : Number(m.toFixed(decimals[k] ?? 2));
      }
      return { key: k, label: prettyLabel[k] || k, values: vals };
    });
  }, [avgSets, decimals, prettyLabel]);
  const mlRows = useMemo(() => {
    const labels = {
      anomaly_score: "Anomaly score (0–1)",
      failure_probability: "Failure probability (0–1)",
      health_index: "Health index (0–100)",
      rul_minutes: "RUL (min)",
      model_confidence: "Model confidence (0–1)",
    };
    return ML_KEYS.map((k) => {
      const vals = {};
      for (const rk of AVG_KEYS) {
        const m = meanFor(avgSets[rk] || [], k);
        const dp = decimals[k] ?? (k === "rul_minutes" ? 0 : 3);
        vals[rk] = m == null ? "-" : Number(m.toFixed(dp));
      }
      return { key: k, label: labels[k], values: vals };
    });
  }, [avgSets, decimals]);
  /* ---------- Optimization ---------- */
  const minuteAvg = useMemo(() => {
    const src = avgSets["1min"] || [];
    const out = {};
    [
      "temperature","pressure","flow","overall_vibration","ultrasonic_noise","motor_current",
      "density","viscosity","ferrous_particles","rpm",
      "bearing_temp_DE","bearing_temp_NDE","casing_temp",
      "suction_pressure","discharge_pressure","delta_p",
      "gas_volume_fraction",
      "current_A","current_B","current_C","power_factor","frequency","torque_est",
      "oil_temp","oil_water_ppm","particle_count","oil_TAN",
      "seal_temp","seal_flush_pressure","leakage_rate",
      "shaft_displacement","noise_dBA",
      "anomaly_score","failure_probability","health_index","rul_minutes","model_confidence"
    ].forEach(k => { out[k] = meanFor(src, k); });
    return out;
  }, [avgSets]);
  const optimizationPct = useMemo(() => {
    const s = minuteAvg && minuteAvg.temperature != null ? minuteAvg : last || {};
    if (!s) return null;
    const tempScore = scoreBand(s.temperature ?? 0, 40, 110, 60, 85);
    const pressScore = scoreBand(s.pressure ?? 0, 0.3, 6.5, 1.5, 4.8);
    const flowScore  = scoreBand(s.flow ?? 0, 2, 30, 8, 16);
    const vibScore   = scoreLowerBetter(s.overall_vibration ?? 0, 0, 0.1, 0.02);
    const usndScore  = scoreLowerBetter(s.ultrasonic_noise ?? 0, 20, 100, 60);
    const ampsScore  = scoreBand(s.motor_current ?? 0, 2, 40, 6, 12);
    const mlPenalty =
      (1 - (Math.max(0, Math.min(100, s.health_index ?? 100)) / 100)) * 0.10 +
      (Math.max(0, Math.min(1, s.failure_probability ?? 0)) * 0.10);
    const w = { temp:0.18, press:0.14, flow:0.18, vib:0.20, usnd:0.18, amps:0.12 };
    let score =
      tempScore*w.temp + pressScore*w.press + flowScore*w.flow +
      vibScore*w.vib + usndScore*w.usnd + ampsScore*w.amps;
    score = score * (1 - mlPenalty);
    return clamp(Math.round(score * 1000) / 10, 0, 100);
  }, [minuteAvg, last]);
  
// ---------- Helpers de severidade (coloca perto dos outros helpers) ----------
const sevFromUp = (v, tEarly, tMod, tSev) => {
  if (!Number.isFinite(v)) return null;
  if (v >= tSev) return "severe";
  if (v >= tMod) return "moderate";
  if (v >= tEarly) return "early";
  return null;
};

const sevFromDown = (v, tEarly, tMod, tSev) => {
  // para métricas "quanto mais baixo pior" (ex.: flow, suction_pressure)
  if (!Number.isFinite(v)) return null;
  if (v <= tSev) return "severe";
  if (v <= tMod) return "moderate";
  if (v <= tEarly) return "early";
  return null;
};

// devolve a “mais grave” entre duas severidades
const maxSev = (a, b) => {
  const rank = { early: 1, moderate: 2, severe: 3, null: 0, undefined: 0 };
  return (rank[a] >= rank[b]) ? a : b;
};

const sevRank = (s) => (s === "severe" ? 3 : s === "moderate" ? 2 : s === "early" ? 1 : 0);

// ======================= DETEÇÃO COM SEVERIDADE =======================
const detectModes = useCallback((L, A) => {
  // usar último sample (L) e média dos últimos 10s (A) como fallback
  const pick = (k, def = 0) => (Number.isFinite(L?.[k]) ? L[k] : (Number.isFinite(A?.[k]) ? A[k] : def));

  const us   = pick("ultrasonic_noise", 0);
  const ov   = pick("overall_vibration", 0);
  const vx   = pick("vibration_x", 0);
  const vy   = pick("vibration_y", 0);
  const vz   = pick("vibration_z", 0);
  const p    = pick("pressure", 0);
  const f    = pick("flow", 999);
  const cur  = pick("motor_current", 0);
  const den  = pick("density", 0);
  const vis  = pick("viscosity", 0);
  const dp   = pick("delta_p", 0);
  const sp   = pick("suction_pressure", 0);
  const dpres= pick("discharge_pressure", 0);
  const ft   = pick("ferrous_particles", 0);
  const t    = pick("temperature", 0);
  const leak = pick("leakage_rate", 0);
  const freq = pick("frequency", 50);
  const ca   = pick("current_A", 0), cb = pick("current_B", 0), cc = pick("current_C", 0);

  const modes = [];

  // ---------- Fluido / Hidráulica ----------
  // cavitation
  if ((us > 65 && (vz > 0.04 || ov > 0.04)) && (f < 10 || sp < 0.3 || p < 1.2)) {
    const s1 = sevFromUp(us, 60, 70, 80);
    const s2 = sevFromUp(Math.max(vz, ov), 0.035, 0.045, 0.06);
    const s3 = maxSev(sevFromDown(f, 10, 8, 6), sevFromDown(sp, 0.4, 0.25, 0.15));
    const sev = [s1, s2, s3].reduce(maxSev, null) || "early";
    modes.push(`cavitation_${sev}`);
  }
  // air_entrapment
  if (us > 60 && f < 8 && ov > 0.03) {
    const sev = [sevFromUp(us, 60, 68, 75), sevFromDown(f, 8, 6.5, 5.5)].reduce(maxSev, null) || "early";
    modes.push(`air_entrapment_${sev}`);
  }
  // vapor_lock
  if (f < 5 && (p < 0.8 || sp < 0.2) && cur < 7) {
    const sev = [sevFromDown(f, 5.5, 4.5, 3.5), sevFromDown(p, 1.0, 0.8, 0.6)].reduce(maxSev, null) || "moderate";
    modes.push(`vapor_lock_${sev}`);
  }
  // low_flow
  if (f < 7) {
    const sev = sevFromDown(f, 7, 5.5, 4.5) || "early";
    modes.push(`low_flow_${sev}`);
  }
  // high_flow
  if (f > 20 && (p < 3.0 || dp < 4.0)) {
    const sev = sevFromUp(f, 20, 25, 28) || "early";
    modes.push(`high_flow_${sev}`);
  }
  // high_pressure
  if (dpres > 9.0 || dp > 8.0 || p > 5.5) {
    const sev = [sevFromUp(dpres, 9.0, 10.0, 11.0), sevFromUp(dp, 8.0, 9.0, 10.0), sevFromUp(p, 5.5, 6.0, 6.3)]
      .reduce(maxSev, null) || "early";
    modes.push(`high_pressure_${sev}`);
  }
  // low_pressure
  if (p < 1.0 || sp < 0.2) {
    const sev = [sevFromDown(p, 1.0, 0.8, 0.6), sevFromDown(sp, 0.2, 0.15, 0.1)].reduce(maxSev, null) || "early";
    modes.push(`low_pressure_${sev}`);
  }
  // npsh_insufficient
  if (us > 65 && f < 10) {
    const sev = [sevFromUp(us, 65, 72, 80), sevFromDown(sp, 0.35, 0.25, 0.15)].reduce(maxSev, null) || "early";
    modes.push(`npsh_insufficient_${sev}`);
  }
  // recirculation
  if (ov > 0.045 && f < 10 && p < 3.0) {
    const sev = [sevFromUp(ov, 0.04, 0.055, 0.065), sevFromDown(f, 10, 8, 6)].reduce(maxSev, null) || "early";
    modes.push(`recirculation_${sev}`);
  }
  // surge
  if ((p > 5.8 || dp > 8.5) && ov > 0.04) {
    const sev = [sevFromUp(p, 5.8, 6.2, 6.6), sevFromUp(ov, 0.04, 0.055, 0.07)].reduce(maxSev, null) || "early";
    modes.push(`surge_${sev}`);
  }
  // impeller_wear (melhorado)
  if ((f < 9 || dp < 4.0) && ov > 0.02) {
    const sev = [sevFromDown(f, 9, 7, 5.5), sevFromUp(ov, 0.03, 0.045, 0.06)].reduce(maxSev, null) || "early";
    modes.push(`impeller_wear_${sev}`);
  }
  // fluid_contaminated
  if (vis > 2.0 && den > 1005 && (ov > 0.03 || ft > 20)) {
    const sev = [sevFromUp(vis, 2.0, 2.4, 3.0), sevFromUp(ft, 20, 35, 55)].reduce(maxSev, null) || "early";
    modes.push(`fluid_contaminated_${sev}`);
  }
  // fluid_mismatch
  if (vis > 2.5 && den < 980 && f < 10 && cur > 10) {
    const sev = [sevFromUp(vis, 2.5, 3.0, 3.6), sevFromDown(den, 985, 980, 975)].reduce(maxSev, null) || "moderate";
    modes.push(`fluid_mismatch_${sev}`);
  }
  // gas_overheat
  if (t > 95 && us > 55) {
    const sev = [sevFromUp(t, 95, 100, 105), sevFromUp(us, 55, 65, 75)].reduce(maxSev, null) || "early";
    modes.push(`gas_overheat_${sev}`);
  }

  // ---------- Mecânica ----------
  // bearing_wear
  if ((vx > 0.02 && vy > 0.02 && ov > 0.04) || ft > 30) {
    const sev = [sevFromUp(ov, 0.04, 0.055, 0.07), sevFromUp(ft, 30, 45, 65)].reduce(maxSev, null) || "moderate";
    modes.push(`bearing_wear_${sev}`);
  }
  // bearing_unlubricated
  if ((t > 90 && ov > 0.045) || ft > 35) {
    const sev = [sevFromUp(t, 90, 95, 100), sevFromUp(ov, 0.045, 0.06, 0.075)].reduce(maxSev, null) || "moderate";
    modes.push(`bearing_unlubricated_${sev}`);
  }
  // misalignment
  if (vx > 0.025 && ov > 0.045 && cur > 12) {
    const sev = [sevFromUp(vx, 0.025, 0.035, 0.05), sevFromUp(ov, 0.045, 0.06, 0.075)].reduce(maxSev, null) || "moderate";
    modes.push(`misalignment_${sev}`);
  }
  // unbalance
  if (vy > 0.025 && ov > 0.045 && vx < vy) {
    const sev = [sevFromUp(vy, 0.025, 0.035, 0.05), sevFromUp(ov, 0.045, 0.06, 0.075)].reduce(maxSev, null) || "moderate";
    modes.push(`unbalance_${sev}`);
  }
  // looseness
  if (ov > 0.055) {
    const sev = sevFromUp(ov, 0.055, 0.07, 0.09) || "early";
    modes.push(`looseness_${sev}`);
  }
  // shaft_bent
  if (vz > 0.03 && ov > 0.05 && cur > 10) {
    const sev = [sevFromUp(vz, 0.03, 0.045, 0.06), sevFromUp(ov, 0.05, 0.065, 0.085)].reduce(maxSev, null) || "moderate";
    modes.push(`shaft_bent_${sev}`);
  }
  // seal_leak
  if (us > 60 && p < 3.0 && f < 12) {
    const sev = [sevFromUp(us, 60, 70, 80), sevFromUp(leak, 10, 25, 50)].reduce(maxSev, null) || "early";
    modes.push(`seal_leak_${sev}`);
  }
  // seal_dry
  if (t > 95 && us > 60) {
    const sev = [sevFromUp(t, 95, 100, 105), sevFromUp(us, 60, 70, 80)].reduce(maxSev, null) || "early";
    modes.push(`seal_dry_${sev}`);
  }
  // structural_fault
  if (ov > 0.06) {
    const sev = sevFromUp(ov, 0.06, 0.08, 0.10) || "early";
    modes.push(`structural_fault_${sev}`);
  }

  // ---------- Elétrica / Motor ----------
  if (cur > 14 && t > 85) {
    const sev = [sevFromUp(cur, 14, 16, 18), sevFromUp(t, 85, 95, 105)].reduce(maxSev, null) || "moderate";
    modes.push(`motor_overload_${sev}`);
  }
  if (cur > 12 && ov > 0.04 && t > 80) {
    const sev = [sevFromUp(cur, 12, 14, 16), sevFromUp(ov, 0.04, 0.055, 0.07)].reduce(maxSev, null) || "moderate";
    modes.push(`electrical_fault_${sev}`);
  }
  if (ca > 14 && cb < 5 && cc > 14) {
    const sev = [sevFromUp(ca, 14, 16, 18), sevFromDown(cb, 5, 4, 3)].reduce(maxSev, null) || "moderate";
    modes.push(`single_phasing_${sev}`);
  }
  if (ov > 0.04 && cur > 11 && Math.abs(freq - 50) > 0.2) {
    const sev = [sevFromUp(ov, 0.04, 0.055, 0.07), sevFromUp(Math.abs(freq - 50), 0.2, 0.35, 0.5)]
      .reduce(maxSev, null) || "early";
    modes.push(`vfd_issue_${sev}`);
  }
  if (cur > 12 && t > 90) {
    const sev = [sevFromUp(cur, 12, 14, 16), sevFromUp(t, 90, 98, 105)].reduce(maxSev, null) || "moderate";
    modes.push(`insulation_breakdown_${sev}`);
  }
  if (ov > 0.048 && ft > 25) {
    const sev = [sevFromUp(ov, 0.048, 0.06, 0.075), sevFromUp(ft, 25, 40, 60)].reduce(maxSev, null) || "moderate";
    modes.push(`bearing_fluting_${sev}`);
  }

  // ---------- Lubrificante / Ambiente ----------
  if (vis < 0.7 && t > 85 && ov > 0.035) {
    const sev = [sevFromDown(vis, 0.7, 0.6, 0.5), sevFromUp(t, 85, 95, 105)].reduce(maxSev, null) || "early";
    modes.push(`oil_degraded_${sev}`);
  }
  if (vis > 1.6 && t > 80) {
    const sev = [sevFromUp(vis, 1.6, 2.0, 2.6), sevFromUp(t, 80, 90, 100)].reduce(maxSev, null) || "early";
    modes.push(`oil_wrong_${sev}`);
  }
  if (t > 95 && vis < 0.9) {
    const sev = [sevFromUp(t, 95, 100, 105), sevFromDown(vis, 0.9, 0.75, 0.6)].reduce(maxSev, null) || "moderate";
    modes.push(`overtemperature_${sev}`);
  }
  if (t < 50 && vis > 1.3 && f < 12) {
    const sev = [sevFromDown(t, 50, 45, 40), sevFromUp(vis, 1.3, 1.7, 2.2)].reduce(maxSev, null) || "early";
    modes.push(`under_temperature_${sev}`);
  }

  // deduplicar mantendo a primeira ocorrência
  const seen = new Set();
  const uniq = [];
  for (const m of modes) {
    if (!seen.has(m)) { seen.add(m); uniq.push(m); }
  }
  return uniq;
}, []);
const [cachedAlerts, setCachedAlerts] = useState([]);
const [cachedRecs, setCachedRecs] = useState([]);

// --- helpers: média de keys sobre um slice de histórico (últimos N s)
const recentAverage = (hist, seconds = 10) => {
  const now = Date.now();
  const slice = (hist || []).filter(r => {
    const t = new Date(r.timestamp).getTime();
    return Number.isFinite(t) && (now - t) <= seconds * 1000;
  });
  const out = {};
  [
    "temperature","pressure","flow","overall_vibration","ultrasonic_noise","motor_current",
    "density","viscosity","ferrous_particles","rpm",
    "bearing_temp_DE","bearing_temp_NDE","casing_temp",
    "suction_pressure","discharge_pressure","delta_p",
    "gas_volume_fraction",
    "current_A","current_B","current_C","power_factor","frequency","torque_est",
    "oil_temp","oil_water_ppm","particle_count","oil_TAN",
    "seal_temp","seal_flush_pressure","leakage_rate",
    "shaft_displacement","noise_dBA",
    "anomaly_score","failure_probability","health_index","rul_minutes","model_confidence"
  ].forEach(k => {
    const vals = slice.map(s => (typeof s[k] === "number" ? s[k] : null)).filter(Number.isFinite);
    out[k] = vals.length ? (vals.reduce((a,b)=>a+b,0) / vals.length) : null;
  });
  return { avg: out, hist: slice };
};


// --- compute alerts & recs every 10s using the last 10s of data
useEffect(() => {
  let stopped = false;

  const computeAndSet = () => {
    const { avg: A} = recentAverage(history, 10);

    // (Recomendação para reduzir jitter): usa A como fonte primária e só usa L se precisares mesmo
    const L = last || {};

    // 1) detectar modos e ordenar
    const modes = detectModes(L, A);
    const ordered = orderByPriority(modes);
    const modeHint = ordered.length ? ordered[0] : null;
    const severity = modeHint ? estimateSeverity(modeHint, L, A) : null;

    // 2) texto de ação
    const actionText = ordered.length ? buildActionForMode(ordered.join(" ")) : null;

    // ---------- LATCH para Suggested Operation ----------
    const now = Date.now();
    if (actionText) {
      const next = { action: actionText, mode: modeHint, severity };
      const prev = lastSuggestedRef.current;

      // só “troca” se for diferente ou mais grave
      const changed =
        !prev ||
        prev.action !== next.action ||
        prev.mode !== next.mode ||
        sevRank(next.severity) > sevRank(prev.severity);

      if (changed) {
        setSuggestedOperation(next);
        lastSuggestedRef.current = next;
        suggestionExpiryRef.current = now + MIN_SUGGEST_MS; // mantém pelo menos 10s
      } else {
        // sem mudanças — mantém enquanto durar o latch
        if (now > suggestionExpiryRef.current) {
          // expirou o latch mas ainda há uma sugestão igual → renova mais um ciclo mínimo
          suggestionExpiryRef.current = now + MIN_SUGGEST_MS;
        }
      }
    } else {
      // não há sugestão nova → só limpa se o latch tiver expirado
      if (now > suggestionExpiryRef.current) {
        setSuggestedOperation(null);
        lastSuggestedRef.current = null;
      }
    }

    // 3) Real-time alerts (o teu cálculo pode ficar como estava)
    const outAlerts = [];

    // --- exemplos: mantém os teus checks aqui ---
    if (Number.isFinite(L.failure_probability) && L.failure_probability >= 0.80) {
      outAlerts.push("⚠️ Failure probability ≥80% (current).");
    } else if (Number.isFinite(A.failure_probability) && A.failure_probability >= 0.80) {
      outAlerts.push("⚠️ Failure probability ≥80% (last 10s avg).");
    }
    if (Number.isFinite(L.health_index) && L.health_index <= 55) {
      outAlerts.push(`Asset health low (HI ≈ ${Math.round(L.health_index)}).`);
    }
    if (Number.isFinite(L.rul_minutes) && L.rul_minutes > 0 && L.rul_minutes <= 240) {
      outAlerts.push(`Low RUL (≈ ${Math.round(L.rul_minutes)} min).`);
    }

    const vib = Number.isFinite(L.overall_vibration) ? L.overall_vibration : A.overall_vibration;
    if (Number.isFinite(vib) && vib > 0.06) outAlerts.push("High vibration — inspect bearings/structure.");

    const usn = Number.isFinite(L.ultrasonic_noise) ? L.ultrasonic_noise : A.ultrasonic_noise;
    if (Number.isFinite(usn) && usn > 70) outAlerts.push("High ultrasonic noise — cavitation/leak likely.");

    const leak = Number.isFinite(L.leakage_rate) ? L.leakage_rate : A.leakage_rate;
    if (Number.isFinite(leak) && leak > 25) outAlerts.push("Seal leak suspected — leakage rising.");

    const den = Number.isFinite(L.density) ? L.density : A.density;
    const t   = Number.isFinite(L.temperature) ? L.temperature : A.temperature;
    if (Number.isFinite(den) && den < 980 && Number.isFinite(t) && t > 90) {
      outAlerts.push("Gas overheating pattern — density down, temperature high.");
    }

    // mensagens orientadas a modo (top-3)
    const msgMap = {
      cavitation: "Cavitation pattern detected.",
      npsh_insufficient: "Insufficient NPSH — cavitation risk.",
      recirculation: "Low-flow recirculation detected.",
      surge: "Pressure surge oscillations detected.",
      impeller_wear: "Impeller wear indicators detected.",
      bearing_wear: "Bearing wear indicators detected.",
      bearing_unlubricated: "Possible lubrication starvation.",
      misalignment: "Coupling misalignment indicators.",
      unbalance: "Rotor unbalance indicators.",
      motor_overload: "Motor overload condition.",
      electrical_fault: "Electrical fault indicators.",
      single_phasing: "Single-phasing suspected.",
      insulation_breakdown: "Insulation breakdown risk.",
      seal_leak: "Seal leak indicators.",
      seal_dry: "Seal running dry.",
    };
    ordered.slice(0, 3).forEach(m => {
      const base = m.replace(/_(early|moderate|severe)$/, "");
      const sev = m.match(/(early|moderate|severe)$/)?.[1] || "moderate";
      outAlerts.push(`${msgMap[base] || `Mode: ${base}`} [${sev}]`);
    });

    // 4) Recomendações rápidas (como tinhas)
    const recs = [];
    if ((usn ?? 0) > 70 || (A.pressure ?? 99) < 1.2) {
      recs.push("Mitigate cavitation: reduce RPM slightly or increase inlet head.");
      if ((A.flow ?? 999) < 8) recs.push("Open suction valve gradually to raise NPSH.");
    }
    if ((vib ?? 0) > 0.04) recs.push("Reduce RPM ~5% and schedule bearing inspection.");
    if (ordered.some(m => m.startsWith("misalignment_"))) recs.push("Realign coupling and recheck soft-foot.");
    if (ordered.some(m => m.startsWith("unbalance_")))   recs.push("Balance rotor at next stop.");
    if (ordered.some(m => m.startsWith("seal_leak_")))   recs.push("Inspect/replace mechanical seal; verify flush.");
    if (!recs.length) recs.push("System optimal — no action required.");

    // ---------- LATCH para Alerts ----------
    // mantém qualquer alerta pelo menos MIN_ALERT_MS desde a última vez que foi visto
    const nextSet = new Set();
    // renova expiração para alerts presentes agora
    outAlerts.forEach(a => {
      alertExpiryRef.current[a] = Math.max(alertExpiryRef.current[a] || 0, now + MIN_ALERT_MS);
      nextSet.add(a);
    });
    // mantém os que ainda não expiraram
    Object.entries(alertExpiryRef.current).forEach(([msg, exp]) => {
      if (exp > now) nextSet.add(msg);
      else if (!outAlerts.includes(msg)) delete alertExpiryRef.current[msg]; // limpa expirados
    });

    if (!stopped) {
      setCachedAlerts(Array.from(nextSet));
      setCachedRecs(Array.from(new Set(recs)));
    }
  };

  computeAndSet();
  const id = setInterval(computeAndSet, 10_000); // 10s estável
  return () => { stopped = true; clearInterval(id); };
}, [history, last, detectModes]); // <-- dependências mínimas


// cleanup apply timer on unmount
useEffect(() => {
  return () => {
    if (applyTimerRef.current) clearTimeout(applyTimerRef.current);
  };
}, []);

// substitute memoized values with cached states
const smartAlerts = cachedAlerts;
const recommendations = cachedRecs;


  /* ---------- UI ---------- */
  const buttonStyle = {
    fontFamily: "'Orbitron', system-ui",
    fontWeight: 700,
    padding: "8px 14px",
    margin: "4px",
    border: "2px solid #fff",
    borderRadius: "8px",
    background: "black",
    color: "#fff",
    cursor: "pointer"
  };
  const captureWave = async () => {
    try {
      const res = await fetch(`${API_BASE}/vibration/waveform?fs=${fs}&duration=${duration}`);
      const data = await res.json();
      if (!data.features && Array.isArray(data.samples) && data.samples.length) {
        const x = data.samples;
        const n = x.length;
        const mean = x.reduce((a,b)=>a+b,0)/n;
        const rms  = Math.sqrt(x.reduce((a,b)=>a+b*b,0)/n);
        const peak = x.reduce((m,v)=>Math.max(m, Math.abs(v)), 0);
        const cf   = peak / (rms || 1e-12);
        const m2 = x.reduce((a,b)=>a+(b-mean)**2,0)/n;
        const m4 = x.reduce((a,b)=>a+(b-mean)**4,0)/n;
        const kurtosisExcess = m2>0 ? (m4/(m2*m2)) - 3 : 0;
        data.features = {
          rms_g: rms,
          pp_g: peak * 2,
          crest_factor: cf,
          kurtosis_excess: kurtosisExcess,
          envelope_rms_g: rms,
          velocity_rms_ms: 0
        };
      }
      setWave(data);
    } catch (e) {
      console.error(e);
    }
  };
  const waveChartData = useMemo(() => {
    if (!wave) return null;
    const t = Array.from({length: wave.n}, (_,i) => (i / wave.fs).toFixed(5));
    return {
      labels: t,
      datasets: [{
        label: `Waveform — window ${(wave?.window_id ?? "").toString().slice(0,8)}…`,
        data: wave.samples,
        borderColor: "#1f77b4",
        borderWidth: 1,
        pointRadius: 0,
        tension: 0,
        yAxisID: "y"
      }]
    };
  }, [wave]);
  const fftChartData = useMemo(() => {
    if (!wave) return null;
    return {
      labels: wave.freqs,
      datasets: [{
        label: "FFT (|g| vs Hz)",
        data: wave.fft_mag,
        borderColor: "#d62728",
        borderWidth: 1,
        pointRadius: 0,
        tension: 0,
        yAxisID: "y"
      }]
    };
  }, [wave]);
  const simpleCard = (title, val) => (
    <div style={{ background:"#fff", borderRadius:12, padding:"14px 16px", border:"1px solid #e5e8eb" }}>
      <div style={{ opacity:0.7 }}>{title}</div>
      <div style={{ fontSize:"2rem", fontWeight:900 }}>{val}</div>
    </div>
  );



  /* ---------- Render ---------- */
  return (
    <div style={{ background: "#C4C6CB", minHeight: "100vh", fontFamily: "'Orbitron', system-ui", fontWeight: 700 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "18px 24px" }}>
        <h1 style={{ margin: 0 }}>Digital Twin</h1>
        <div>DEVICE #1</div>
      </header>
      {/* 3D + badge de Optimization */}
      <section style={{ height: 500, padding: "0 16px 24px" }}>
        <div style={{ position:"relative", width:"100%", height:"100%" }}>
          <Twin3D vibration={last?.overall_vibration} />
          <div style={{
            position:"absolute", bottom: 8, left: 16,
            background:"rgba(0,0,0,0.80)", color:"#fff",
            padding:"10px 14px", borderRadius:10, border:"1px solid #fff",
            display:"flex", alignItems:"baseline", gap:8
          }}>
            <span style={{ opacity:0.8, fontSize:"0.95rem" }}>Optimization</span>
            <span style={{ fontSize:"2rem", lineHeight:1, fontWeight:900 }}>
              {optimizationPct != null ? `${optimizationPct.toFixed(1)}%` : "–%"}
            </span>
          </div>
        </div>
      </section>
      {/* Chart */}
      <section style={{ padding: "0 16px 24px" }}>
        <div style={{ height: 520, background: "#fff", borderRadius: 12, padding: 12 }}>
          <Line data={chartData} options={chartOptions} />
          <div style={{ textAlign: "center", marginTop: 8 }}>
            <button style={buttonStyle} onClick={()=>setRange("10s")}>Last 10s</button>
            <button style={buttonStyle} onClick={()=>setRange("1min")}>Last minute</button>
            <button style={buttonStyle} onClick={()=>setRange("1h")}>Last hour</button>
            <button style={buttonStyle} onClick={()=>setRange("24h")}>Last 24h</button>
            <button style={buttonStyle} onClick={()=>setRange("all")}>Full history</button>
          </div>
        </div>
      </section>
      {/* Alerts */}
      <section style={{ padding: 16 }}>
        <h2 style={{ marginTop: 0 }}>Real-time alerts</h2>
        {alerts.map((a,i)=>
          <div key={`srv-${i}`} style={{ background:"#fff", padding:10, marginBottom:6, borderRadius:10 }}>{a}</div>
        )}
        {smartAlerts.map((a,i)=>
          <div key={`ml-${i}`} style={{ background:"#fff", padding:10, marginBottom:6, borderRadius:10 }}>{a}</div>
        )}
      </section>
      {/* One-click corrective action */}


{/* One-click corrective action */}
{suggestedOperation && (
  <section style={{ padding: "0 16px 24px" }}>
    <div style={{
      background:"#111417",
      color:"#fff",
      borderRadius:12,
      padding:16,
      border:"1px solid #1e2327",
      display:"flex",
      alignItems:"center",
      justifyContent:"space-between",
      gap:12
    }}>
      <div>
        <div style={{ opacity:0.8, marginBottom:6 }}>Suggested corrective action</div>
        <div style={{ fontSize:"1.05rem" }}>
          {suggestedOperation.action}
          {suggestedOperation.severity && (
            <span style={{
              marginLeft: 8,
              padding: "2px 8px",
              borderRadius: 8,
              border: "1px solid #444",
              fontSize: 12,
              opacity: 0.9
            }}>
              {suggestedOperation.severity === "severe" ? "SEVERE" :
               suggestedOperation.severity === "moderate" ? "MODERATE" : "EARLY"}
            </span>
          )}
        </div>
      </div>

      <button
        style={{
          fontFamily: "'Orbitron', system-ui",
          fontWeight: 700,
          padding: "10px 16px",
          border: "2px solid #fff",
          borderRadius: "10px",
          background: "black",
          color: "#fff",
          cursor: "pointer",
          whiteSpace: "nowrap"
        }}
        onClick={async () => {
          if (!suggestedOperation || applyStatus === "pending") return;
          try {
            setApplyStatus("pending");
            const res = await postOperation(suggestedOperation.action, suggestedOperation.mode);
            if (res && res.ok !== false) {
              setApplyStatus("applied");

              // manter visível por 10s
              const keepOp = suggestedOperation;
              setSuggestedOperation(keepOp);

              if (applyTimerRef.current) clearTimeout(applyTimerRef.current);
              applyTimerRef.current = setTimeout(() => {
                setApplyStatus("idle");
                setSuggestedOperation(null);
              }, 10000);
            } else {
              setApplyStatus("idle");
            }
          } catch (e) {
            console.error(e);
            setApplyStatus("idle");
          }
        }}
        title="Send this operation to the backend and log it"
      >
        {applyStatus === "pending" ? "Applying..." : applyStatus === "applied" ? "Applied" : "Apply fix"}
      </button>
    </div>
  </section>
)}  {/* <-- fecha o bloco condicional aqui, exatamente com )} */}

{/* ---- ML diagnostics (tabela preta) ---- */}
<section style={{ padding: "0 16px 24px" }}>
  {/* … resto da tua secção ML … */}
</section>


{/* ---- ML diagnostics (tabela preta) ---- */}

      {/* ---- ML diagnostics (tabela preta) ---- */}
      <section style={{ padding: "0 16px 24px" }}>
        <div style={{
          background:"#0f1113",
          color:"#fff",
          borderRadius:12,
          padding:16,
          border:"1px solid #1e2327"
        }}>
          <h2 style={{ marginTop: 0 }}>ML diagnostics</h2>
          <div style={{ overflowX:"auto" }}>
            <table style={{ width:"100%", borderCollapse:"collapse" }}>
              <thead>
                <tr style={{ background:"#161a1d" }}>
                  <th style={{ textAlign:"left",  padding:"10px 8px" }}>Metric</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>10s</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>1 min</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>1 h</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>24 h</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>Full</th>
                </tr>
              </thead>
              <tbody>
                {mlRows.map(({ key, label, values }, i) => (
                  <tr key={key} style={{ background: i % 2 ? "#0b0d0f" : "#111417" }}>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327" }}>{label}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>{values["10s"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>{values["1min"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>{values["1h"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>{values["24h"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>{values["all"]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      {/* Recomendações */}
      <section style={{ padding: "0 16px 24px" }}>
        <div style={{
          background:"#e6eaee",
          color:"#111",
          borderRadius:12,
          padding:"16px 18px",
          border: "1px solid #d2d7dc"
        }}>
          <h2 style={{ marginTop: 0 }}>Data-driven recommendations</h2>
          <ul style={{ margin:0, paddingLeft: "1.2rem" }}>
            {recommendations.map((r,i)=><li key={i} style={{ marginBottom:6 }}>{r}</li>)}
          </ul>
        </div>
      </section>
      {/* Sensor averages */}
      <section style={{ padding: "0 16px 24px" }}>
        <div style={{
          background:"#d9dde2",
          color:"#111",
          borderRadius:12,
          padding:16,
          border:"1px solid #c8cdd2"
        }}>
          <h2 style={{ marginTop: 0 }}>Sensor averages</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width:"100%", borderCollapse:"collapse" }}>
              <thead>
                <tr style={{ background:"#cfd4d9" }}>
                  <th style={{ textAlign:"left", padding:"10px 8px" }}>Sensor</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>10s</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>1 min</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>1 h</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>24 h</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>Full</th>
                </tr>
              </thead>
              <tbody>
                {avgRows.map(({ key, label, values }, idx) => (
                  <tr key={key} style={{ background: idx % 2 ? "#eef1f4" : "#ffffff" }}>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9" }}>{label}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9", textAlign:"right" }}>{values["10s"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9", textAlign:"right" }}>{values["1min"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9", textAlign:"right" }}>{values["1h"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9", textAlign:"right" }}>{values["24h"]}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #cfd4d9", textAlign:"right" }}>{values["all"]}</td>
                  </tr>
                ))}
                {!avgRows.length && (
                  <tr><td colSpan={6} style={{ padding:10, textAlign:"center" }}>No data.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      {/* Operations */}
      <section style={{ padding: "0 16px 36px" }}>
        <div style={{ background:"#0f1113", color:"#fff", borderRadius:12, padding:16 }}>
          <h2 style={{ marginTop: 0 }}>Recent Operations / Maintenance (history)</h2>
          <div style={{ overflowX:"auto" }}>
            <table style={{ width:"100%", borderCollapse:"collapse" }}>
              <thead>
                <tr style={{ background:"#161a1d" }}>
                  <th style={{ textAlign:"left", padding:"10px 8px" }}>Timestamp</th>
                  <th style={{ textAlign:"left", padding:"10px 8px" }}>Action</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>% Before</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>% After</th>
                  <th style={{ textAlign:"right", padding:"10px 8px" }}>Δ Opt</th>
                </tr>
              </thead>
              <tbody>
                {ops.length ? ops.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? "#0b0d0f" : "#111417" }}>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327" }}>{new Date(r.timestamp).toLocaleString()}</td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327" }}>{r.action}</td>
                    {/* AJUSTE 2: células null-safe */}
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>
                      {r.before == null ? "N/A" : `${r.before.toFixed(1)}%`}
                    </td>
                    <td style={{ padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right" }}>
                      {r.after == null ? "N/A" : `${r.after.toFixed(1)}%`}
                    </td>
                    <td style={{
                      padding:"10px 8px", borderTop:"1px solid #1e2327", textAlign:"right",
                      color: r.delta == null ? "#fff" : (r.delta >= 0 ? "#4ade80" : "#f87171")
                    }}>
                      {r.delta == null ? "N/A" : `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(1)}%`}
                    </td>
                  </tr>
                )) : (
                  <tr><td colSpan={5} style={{ padding:10, textAlign:"center" }}>No operations logged.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      {/* Waveform + FFT */}
      <section style={{ padding: "0 16px 36px" }}>
        <h2 style={{ marginTop: 0 }}>Vibration waveform</h2>
        <div style={{ display:"flex", gap:12, alignItems:"center", marginBottom:8 }}>
          <div>fs (Hz):</div>
          <input style={{ padding:6, borderRadius:8, border:"1px solid #cfd4d9", width:120 }}
                 value={fs} onChange={e=>setFs(Number(e.target.value)||10000)} />
          <div>duration (s):</div>
          <input style={{ padding:6, borderRadius:8, border:"1px solid #cfd4d9", width:80 }}
                 value={duration} onChange={e=>setDuration(Number(e.target.value)||1)} />
          <button style={buttonStyle} onClick={captureWave}>Capture</button>
        </div>
        {wave && (
          <>
            <div style={{ background:"#fff", borderRadius:12, padding:12, marginBottom:12 }}>
              <Line data={waveChartData} options={{
                responsive:true, maintainAspectRatio:false, height:180,
                plugins:{ legend:{display:false}, title:{display:true, text: waveChartData.datasets[0].label } },
                scales:{ y:{ title:{display:true,text:"g"} }, x:{ title:{display:true,text:"Time (s)"} } }
              }} height={180}/>
            </div>
            <div style={{ background:"#fff", borderRadius:12, padding:12 }}>
              <Line data={fftChartData} options={{
                responsive:true, maintainAspectRatio:false, height:220,
                plugins:{ legend:{display:false}, title:{display:true, text:"FFT (|g| vs Hz)"} },
                scales:{ y:{ title:{display:true,text:"|g|"} }, x:{ title:{display:true,text:"Hz"} } }
              }} height={220}/>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(5, 1fr)", gap:12, marginTop:12 }}>
              {simpleCard("RMS (g)", wave?.features?.rms_g != null ? wave.features.rms_g.toFixed(6) : "–")}
              {simpleCard("P-P (g)", wave?.features?.pp_g != null ? wave.features.pp_g.toFixed(6) : "–")}
              {simpleCard("Crest factor", wave?.features?.crest_factor != null ? wave.features.crest_factor.toFixed(3) : "–")}
              {simpleCard("Kurtosis (excess)", wave?.features?.kurtosis_excess != null ? wave.features.kurtosis_excess.toFixed(3) : "–")}
              {simpleCard("Envelope RMS (g)", wave?.features?.envelope_rms_g != null ? wave.features.envelope_rms_g.toFixed(6) : "–")}
              {simpleCard("Velocity RMS (m/s)", wave?.features?.velocity_rms_ms != null ? wave.features.velocity_rms_ms.toFixed(6) : "–")}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

