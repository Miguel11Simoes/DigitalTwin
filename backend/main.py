# main.py — sensores completos + modes + perfis (água/óleo/gás/petróleo pesado)
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import random, datetime, math, os, csv, time, copy
import numpy as np
# Parquet opcional
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pass
# =========================
# APP & CORS
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
   allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)
# =========================
# ESTADO
# =========================
history = []
STATE = {
    "mode": "normal",
    "rpm": 1800.0,
    "seed": random.random(),
    "last_sample": None,
    "profile": "agua_potavel",
}
# =========================
# FIX MAPPING & HELPERS
# =========================
# Map: base failure mode -> recommended action (must match app.js MODE_TO_ACTION)
# --- Operations mapping (keep these texts aligned with the frontend MODE_TO_ACTION) ---
# --- 35 base modes → ação recomendada (alinha com o frontend) ---
OPERATION_FIXES = {
    # Fluido / Hidráulica
    "cavitation": "Check suction pressure, reduce pump speed",
    "air_entrapment": "Bleed/vent air; check suction leaks",
    "vapor_lock": "Cool/pressurize suction; purge vapor",
    "low_flow": "Open valves / clean strainers",
    "high_flow": "Throttle discharge valve / reduce speed",
    "high_pressure": "Reduce throttling / remove restrictions",
    "low_pressure": "Increase NPSH / open suction valve",
    "npsh_insufficient": "Raise tank level / reduce suction losses",
    "recirculation": "Open discharge / move away from low-flow",
    "surge": "Retune control loop; avoid surge region",
    "impeller_wear": "Inspect & replace impeller",
    "fluid_contaminated": "Replace/flush fluid; clean system",
    "fluid_mismatch": "Load correct fluid grade/spec",
    "gas_overheat": "Improve cooling / reduce compression ratio",
    # Mecânica
    "bearing_wear": "Lubricate or replace bearings",
    "bearing_unlubricated": "Re-lubricate bearings; check regime",
    "misalignment": "Realign motor and pump shaft",
    "unbalance": "Balance rotating components",
    "looseness": "Tighten fasteners; check fits",
    "shaft_bent": "Inspect shaft; straighten/replace",
    "seal_leak": "Inspect and replace mechanical seal",
    "seal_dry": "Restore seal flush; verify lubrication",
    "structural_fault": "Inspect supports/baseplate; stiffen structure",
    # Elétrica / Motor
    "motor_overload": "Reduce load, check electrical supply",
    "electrical_fault": "Inspect motor windings and connections",
    "single_phasing": "Restore missing phase; check protection",
    "vfd_issue": "Check VFD params/filters; EMC/harmonics",
    "insulation_breakdown": "Test insulation; dry/repair motor",
    "bearing_fluting": "Install shaft grounding; fix VFD common-mode",
    # Lubrificante / Ambiente
    "oil_degraded": "Change oil and check contamination",
    "oil_wrong": "Fill correct oil grade/viscosity",
    "overtemperature": "Reduce load; improve cooling",
    "under_temperature": "Warm-up process; reduce viscosity",
    "ambient_hot": "Improve ventilation/ambient cooling",
    "ambient_cold": "Preheat fluid; insulate lines",
}
_BASE_KEYS = set(OPERATION_FIXES.keys())

import re
def _normalize_action(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()
def _extract_base_modes(mode: str) -> list[str]:
    """Devolve todas as bases contidas num mode (severity/composites suportados)."""
    if not mode or mode == "normal":
        return []
    m = str(mode).lower()
    hits = [b for b in OPERATION_FIXES.keys() if b in m]
    if not hits:
        hits = [m.split("_")[0]]
    # dedup mantendo ordem
    out, seen = [], set()
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


# =========================
# PROFILES (presets)
# Cada perfil altera nominais/limites de alguns sensores
# =========================
PROFILES = {
    "agua_potavel": {
        "density": (990.0, 1000.0, 1010.0),
        "viscosity": (0.8, 1.0, 1.3),
        "pressure": (0.5, 3.0, 6.0),
        "flow": (5.0, 15.0, 25.0),
        "ultrasonic_base": (30.0, 45.0, 65.0),
        "temp_base": (50.0, 80.0, 110.0),
        "current_base": (4.0, 8.0, 14.0),
        "cavitation_sensitivity": 1.2,
    },
    "oleo_lubrificante_iso68": {
        "density": (850.0, 870.0, 900.0),
        "viscosity": (60.0, 68.0, 80.0),  # em cSt (convertemos para 0.3–5.0 relativo)
        "pressure": (1.0, 4.0, 8.0),
        "flow": (2.0, 10.0, 18.0),
        "ultrasonic_base": (25.0, 40.0, 60.0),
        "temp_base": (55.0, 85.0, 120.0),
        "current_base": (5.0, 9.0, 16.0),
        "cavitation_sensitivity": 0.8,
    },
    "gas_natural": {
        "density": (0.6, 0.8, 1.2),
        "viscosity": (0.008, 0.012, 0.02),
        "pressure": (2.0, 10.0, 20.0),
        "flow": (50.0, 120.0, 220.0),
        "ultrasonic_base": (20.0, 35.0, 55.0),
        "temp_base": (40.0, 70.0, 100.0),
        "current_base": (3.0, 7.0, 12.0),
        "cavitation_sensitivity": 0.5,
    },
    "petroleo_pesado": {
        "density": (930.0, 970.0, 1010.0),
        "viscosity": (200.0, 300.0, 450.0),  # cSt altos → convertemos
        "pressure": (1.5, 5.0, 10.0),
        "flow": (1.0, 6.0, 12.0),
        "ultrasonic_base": (35.0, 50.0, 70.0),
        "temp_base": (60.0, 95.0, 130.0),
        "current_base": (6.0, 10.0, 20.0),
        "cavitation_sensitivity": 1.5,
    },
}
# =========================
# SENSOR META (catálogo completo)
# unit, min, nominal, max, warn, alarm, noise, rate
# =========================
BASE_SENSOR_META = {
    # 13 básicos + muitos adicionais
    "temperature":        {"unit":"°C","min":40,"nominal":80,"max":120,"warn":95,"alarm":105,"noise":0.5,"rate":2.0},
    "pressure":           {"unit":"bar","min":0.3,"nominal":3.5,"max":8.0,"warn":5.5,"alarm":6.0,"noise":0.02,"rate":0.25},
    "vibration_x":        {"unit":"g","min":0.001,"nominal":0.010,"max":0.100,"warn":0.040,"alarm":0.060,"noise":0.001,"rate":0.003},
    "vibration_y":        {"unit":"g","min":0.001,"nominal":0.010,"max":0.100,"warn":0.040,"alarm":0.060,"noise":0.001,"rate":0.003},
    "vibration_z":        {"unit":"g","min":0.001,"nominal":0.010,"max":0.100,"warn":0.040,"alarm":0.060,"noise":0.001,"rate":0.003},
    "overall_vibration":  {"unit":"g RMS","min":0.000,"nominal":0.012,"max":0.100,"warn":0.040,"alarm":0.060,"noise":0.0008,"rate":0.003},
    "flow":               {"unit":"m3/h","min":2,"nominal":12,"max":30,"warn":30,"alarm":20,"noise":0.4,"rate":2.5},
    "density":            {"unit":"kg/m3","min":900,"nominal":995,"max":1100,"warn":1015,"alarm":1030,"noise":0.8,"rate":2.0},
    "viscosity":          {"unit":"cP","min":0.3,"nominal":1.0,"max":3.0,"warn":2.5,"alarm":3.5,"noise":0.03,"rate":0.2},
    "ultrasonic_noise":   {"unit":"dB","min":20,"nominal":40,"max":100,"warn":55,"alarm":65,"noise":0.6,"rate":2.5},
    "ferrous_particles":  {"unit":"count/ml","min":0,"nominal":5,"max":200,"warn":30,"alarm":80,"noise":0.5,"rate":5.0},
    "motor_current":      {"unit":"A","min":2,"nominal":8,"max":40,"warn":12,"alarm":14,"noise":0.08,"rate":0.5},
    "rpm":                {"unit":"rpm","min":600,"nominal":1800,"max":3600,"warn":2000,"alarm":2300,"noise":1.5,"rate":25.0},
    "bearing_temp_DE":    {"unit":"°C","min":10,"nominal":75,"max":130,"warn":95,"alarm":110,"noise":0.4,"rate":2.0},
    "bearing_temp_NDE":   {"unit":"°C","min":10,"nominal":70,"max":130,"warn":90,"alarm":105,"noise":0.4,"rate":2.0},
    "casing_temp":        {"unit":"°C","min":10,"nominal":60,"max":120,"warn":85,"alarm":100,"noise":0.3,"rate":1.5},
    "suction_pressure":   {"unit":"bar","min":-0.5,"nominal":0.7,"max":2.0,"warn":0.3,"alarm":0.1,"noise":0.02,"rate":0.2},
    "discharge_pressure": {"unit":"bar","min":0.5,"nominal":5.0,"max":12.0,"warn":9.0,"alarm":10.5,"noise":0.05,"rate":0.5},
    "delta_p":            {"unit":"bar","min":0.2,"nominal":4.3,"max":11.0,"warn":8.0,"alarm":9.5,"noise":0.05,"rate":0.5},
    "gas_volume_fraction":{"unit":"-","min":0.0,"nominal":0.01,"max":0.20,"warn":0.05,"alarm":0.10,"noise":0.002,"rate":0.01},
    "current_A":          {"unit":"A","min":0,"nominal":8.0,"max":18.0,"warn":13.0,"alarm":15.0,"noise":0.08,"rate":0.6},
    "current_B":          {"unit":"A","min":0,"nominal":8.0,"max":18.0,"warn":13.0,"alarm":15.0,"noise":0.08,"rate":0.6},
    "current_C":          {"unit":"A","min":0,"nominal":8.0,"max":18.0,"warn":13.0,"alarm":15.0,"noise":0.08,"rate":0.6},
    "power_factor":       {"unit":"-","min":0.5,"nominal":0.88,"max":1.0,"warn":0.75,"alarm":0.65,"noise":0.003,"rate":0.02},
    "frequency":          {"unit":"Hz","min":48,"nominal":50,"max":52,"warn":51,"alarm":51.5,"noise":0.01,"rate":0.05},
    "torque_est":         {"unit":"Nm","min":5,"nominal":35,"max":80,"warn":60,"alarm":70,"noise":0.4,"rate":2.0},
    "oil_temp":           {"unit":"°C","min":10,"nominal":55,"max":110,"warn":85,"alarm":95,"noise":0.3,"rate":1.5},
    "oil_water_ppm":      {"unit":"ppm","min":0,"nominal":150,"max":3000,"warn":500,"alarm":1000,"noise":5,"rate":30},
    "particle_count":     {"unit":"ISO idx","min":12,"nominal":16,"max":24,"warn":19,"alarm":21,"noise":0.1,"rate":0.5},
    "oil_TAN":            {"unit":"mgKOH/g","min":0.05,"nominal":0.3,"max":3.0,"warn":1.0,"alarm":1.5,"noise":0.01,"rate":0.05},
    "seal_temp":          {"unit":"°C","min":10,"nominal":50,"max":120,"warn":80,"alarm":95,"noise":0.3,"rate":1.5},
    "seal_flush_pressure":{"unit":"bar","min":0.3,"nominal":1.5,"max":4.0,"warn":0.8,"alarm":0.5,"noise":0.02,"rate":0.2},
    "leakage_rate":       {"unit":"ml/min","min":0,"nominal":5,"max":200,"warn":20,"alarm":60,"noise":0.5,"rate":3.0},
    "shaft_displacement": {"unit":"µm","min":20,"nominal":60,"max":250,"warn":120,"alarm":180,"noise":1.0,"rate":6.0},
    "noise_dBA":          {"unit":"dBA","min":60,"nominal":78,"max":110,"warn":90,"alarm":98,"noise":0.5,"rate":2.0},
}
# Cópia ativa (alterada por perfil)
SENSOR_META = copy.deepcopy(BASE_SENSOR_META)
def _apply_profile(profile_name: str):
    """Modifica SENSOR_META (min/nominal/max) e bases de geração conforme perfil."""
    global SENSOR_META
    SENSOR_META = copy.deepcopy(BASE_SENSOR_META)
    prof = PROFILES.get(profile_name, PROFILES["agua_potavel"])
    # Adaptar alguns sensores principais
    def set_range(name, triple):
        lo, mid, hi = triple
        if name in SENSOR_META:
            SENSOR_META[name]["min"] = lo
            SENSOR_META[name]["nominal"] = mid
            SENSOR_META[name]["max"] = hi
    for key_map, triple in [
        ("density", prof["density"]),
        ("viscosity", (0.3, 1.0, 5.0)),  # manter escala relativa 0.3–5.0; converteremos a partir do prof
        ("pressure", prof["pressure"]),
        ("flow", prof["flow"]),
    ]:
        set_range(key_map, triple)
    # Ajustar temperature/motor_current/ultrasonic conforme perfil
    if "temp_base" in prof:
        set_range("temperature", prof["temp_base"])
    if "current_base" in prof:
        set_range("motor_current", prof["current_base"])
    if "ultrasonic_base" in prof:
        set_range("ultrasonic_noise", prof["ultrasonic_base"])
# aplicar perfil inicial
_apply_profile(STATE["profile"])
# Versão simples para clamp
def SENSOR_LIMITS():
    return {k: (v["min"], v["max"]) for k, v in SENSOR_META.items()}
# =========================
# UTILITÁRIOS
# =========================
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))
def _ramp(prev, target, max_step):
    if prev is None:
        return target
    if target > prev:
        return min(prev + max_step, target)
    else:
        return max(prev - max_step, target)
# ---------- KPI/score de otimização (igual ao frontend) ----------
def _score_band(value, lo, hi, bestLo, bestHi):
    if value is None:
        return 0.0
    v = float(value)
    if bestLo <= v <= bestHi:
        return 1.0
    if v < bestLo:
        if bestLo == lo:
            return 0.0
        t = (v - lo) / (bestLo - lo)
        return max(0.0, min(1.0, t))
    if bestHi == hi:
        return 0.0
    t = (hi - v) / (hi - bestHi)
    return max(0.0, min(1.0, t))
def _score_lower_better(value, lo, hi, bestHi):
    if value is None:
        return 0.0
    v = float(value)
    if v <= bestHi:
        return 1.0
    t = (hi - v) / (hi - bestHi)
    return max(0.0, min(1.0, t))
def optimization_score_from_sample(s: dict) -> float:
    """
    0..100, maior é melhor. Replica a lógica do frontend:
    temp, press, flow, vib, ultrassom, corrente.
    """
    tempScore = _score_band(s.get("temperature"), 40, 110, 60, 85)
    pressScore = _score_band(s.get("pressure"), 0.3, 6.5, 1.5, 4.8)
    flowScore  = _score_band(s.get("flow"), 2, 30, 8, 16)
    vibScore   = _score_lower_better(s.get("overall_vibration"), 0, 0.1, 0.02)
    usndScore  = _score_lower_better(s.get("ultrasonic_noise"), 20, 100, 60)
    ampsScore  = _score_band(s.get("motor_current"), 2, 40, 6, 12)
    w = {"temp":0.18, "press":0.14, "flow":0.18, "vib":0.20, "usnd":0.18, "amps":0.12}
    score = (
        tempScore * w["temp"] +
        pressScore * w["press"] +
        flowScore  * w["flow"] +
        vibScore   * w["vib"] +
        usndScore  * w["usnd"] +
        ampsScore  * w["amps"]
    )
    return max(0.0, min(100.0, round(score*1000)/10.0))
def _avg_optimization_between(t0: datetime.datetime, t1: datetime.datetime) -> float:
    """Média do KPI no history entre [t0, t1]. Retorna None se não houver dados."""
    if t1 <= t0:
        return None
    vals = []
    for d in history:
        try:
            ts = datetime.datetime.fromisoformat(d["timestamp"])
        except Exception:
            continue
        if t0 <= ts <= t1:
            vals.append(optimization_score_from_sample(d))
    if not vals:
        return None
    return sum(vals) / len(vals)

def _norm01(value: float, key: str) -> float:
    """Normaliza [min,max] de SENSOR_META para [0,1]."""
    m = SENSOR_META.get(key)
    if not m:
        return 0.0
    lo, hi = m["min"], m["max"]
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
def _ml_metrics(d: dict) -> dict:
    """
    Gera métricas ML plausíveis (0-1 / 0-100 / minutos) a partir de sinais chave.
    É apenas para preencher a UI; substitui por um modelo real quando tiveres.
    """
    vib = _norm01(d.get("overall_vibration", 0.0), "overall_vibration")
    usn = _norm01(d.get("ultrasonic_noise", 0.0), "ultrasonic_noise")
    tmp = _norm01(d.get("temperature", 0.0), "temperature")
    prs = _norm01(d.get("pressure", 0.0), "pressure")
    flw = _norm01(d.get("flow", 0.0), "flow")
    fer = _norm01(d.get("ferrous_particles", 0.0), "ferrous_particles")
    anomaly = 0.28*vib + 0.20*usn + 0.18*tmp + 0.12*prs + 0.12*(1.0 - flw) + 0.10*fer
    anomaly = max(0.0, min(1.0, anomaly))
    failure = _sigmoid(6.0*(anomaly - 0.55))
    hi = max(0.0, min(100.0, 100.0*(1.0 - 0.85*anomaly)))
    rul = max(0, int(720.0 * (1.0 - anomaly)**2))
    conf = max(0.70, min(0.95, 0.85 + 0.1*random.uniform(-1, 1)))
    return {
        "anomaly_score": round(anomaly, 3),
        "failure_probability": round(failure, 3),
        "health_index": round(hi, 1),
        "rul_minutes": int(rul),
        "model_confidence": round(conf, 3),
    }

# =========================
# MODES (efeitos)
# =========================
from collections import defaultdict as _dd
SEV = {"early": 0.5, "moderate": 1.0, "severe": 1.7}
# =========================
# MODES (efeitos) — TEMPLATES realistas
# =========================
TEMPLATES = {
    # ===================== Fluido / Hidráulica =====================
    "cavitation": lambda s: {
        # Ultrassons e vib ↑; sucção ↓ (maior vácuo), caudal ↓
        "mul_ultrasonic_noise": 1.9 * s,
        "mul_vibration_z": 1.6 * s,
        "mul_vibration_x": 1.2 * s,
        "mul_vibration_y": 1.2 * s,
        "mul_flow": 0.80,
        "mul_discharge_pressure": 0.97,
        "mul_suction_pressure": 0.85,   # ligeiramente menos agressivo (era 0.80)
        "mul_motor_current": 1.05 * s,
    },

    "air_entrapment": lambda s: {
        # Ar arrastado → ultrassom ↑, flow ↓, flutuação de pressão
        "mul_ultrasonic_noise": 1.6 * s,
        "mul_flow": 0.85,
        "mul_discharge_pressure": 0.97,
        "mul_suction_pressure": 0.92,
        "mul_vibration_x": 1.15 * s,
        "mul_vibration_y": 1.15 * s,
    },

    "vapor_lock": lambda s: {
        # Bolso de vapor → bomba descarrega → flow e press ↓, corrente ↓
        "mul_flow": 0.45,
        "mul_discharge_pressure": 0.75,
        "mul_suction_pressure": 0.85,
        "mul_motor_current": 0.90,
        "mul_ultrasonic_noise": 1.30 * s,
        "mul_vibration_z": 1.15 * s,
    },

    "low_flow": lambda s: {
        # Estrangulamento → flow ↓, pressão descarga ↑, corrente ~estável/↓
        "mul_flow": 0.70 if s <= 1.0 else 0.55,
        "mul_discharge_pressure": 1.10,
        "mul_suction_pressure": 1.02,
        "mul_motor_current": 0.95,
        "mul_vibration_x": 1.10,
        "mul_vibration_y": 1.10,
    },

    "high_flow": lambda s: {
        # Válvula muito aberta → flow ↑, ΔP ↓ (descarga ↓), corrente ↑
        "mul_flow": 1.30 * s,
        "mul_discharge_pressure": 0.92,
        "mul_suction_pressure": 1.00,
        "mul_motor_current": 1.12 * s,
        "mul_vibration_y": 1.15 * s,
    },

    "high_pressure": lambda s: {
        # Restrição a jusante → descarga ↑, flow ↓, corrente ↑
        "mul_discharge_pressure": 1.30 * s,
        "mul_suction_pressure": 1.02,
        "mul_flow": 0.88,               # mais queda para diferenciar de low_flow
        "mul_motor_current": 1.10 * s,
        "mul_vibration_x": 1.10 * s,
    },

    "low_pressure": lambda s: {
        # Falta de carga/nível baixo → pressões ↓ e flow ↓
        "mul_discharge_pressure": 0.85,
        "mul_suction_pressure": 0.85,
        "mul_flow": 0.90,
        "mul_vibration_z": 1.05 * s,
    },

    "npsh_insufficient": lambda s: {
        # NPSH disponível baixo → cavitação iminente
        "mul_ultrasonic_noise": 1.7 * s,
        "mul_vibration_z": 1.30 * s,
        "mul_flow": 0.88,
        "mul_suction_pressure": 0.85,   # ↓ sucção
    },

    "recirculation": lambda s: {
        # Baixo caudal com recirculação interna → vib ↑, flow ↓, press ↓
        "mul_vibration_x": 1.25 * s,
        "mul_vibration_y": 1.25 * s,
        "mul_vibration_z": 1.20 * s,
        "mul_discharge_pressure": 0.95,
        "mul_suction_pressure": 0.98,
        "mul_flow": 0.80,
        "mul_ultrasonic_noise": 1.10 * s,
    },

    "surge": lambda s: {
        # Oscilações → picos de pressão e vib (spikes extra no synthesize_mode_sample)
        "mul_discharge_pressure": 1.35 * s,
        "mul_suction_pressure": 0.98,
        "mul_vibration_x": 1.25 * s,
        "mul_vibration_y": 1.25 * s,
    },

    "impeller_wear": lambda s: {
        # Perda de rendimento → ↓ caudal e ΔP (descarga ↓, sucção ↑ ligeiro)
        "mul_flow": 0.70 if s <= 1.0 else 0.55,
        "mul_discharge_pressure": 0.92,
        "mul_suction_pressure": 1.05,   # ΔP desce
        # Vibração aumenta em todos os eixos (overall será recalculado)
        "mul_vibration_x": 1.40 * s,
        "mul_vibration_y": 1.40 * s,
        "mul_vibration_z": 1.20 * s,
        # Ultrassons e desgaste
        "mul_ultrasonic_noise": 1.15 * s,
        "add_ferrous_particles": 8 * s,
    },

    "fluid_contaminated": lambda s: {
        # Fluido sujo/partículas → viscosidade ↑, densidade ↑ ligeiro, vib ↑, ferrosos ↑
        "mul_viscosity": 1.25 * s,
        "mul_density": 1.03,
        "mul_vibration_x": 1.15 * s,
        "mul_vibration_y": 1.15 * s,
        "add_ferrous_particles": 6 * s,
        "mul_flow": 0.95,
    },

    "fluid_mismatch": lambda s: {
        # Fluido errado (muito viscoso/leve) → corrente ↑, flow ↓, densidade ↓
        "mul_viscosity": 1.60 * s,
        "mul_density": 0.97,
        "mul_flow": 0.85,
        "mul_motor_current": 1.15 * s,
        "mul_vibration_y": 1.10 * s,
    },

    "gas_overheat": lambda s: {
        # Fluido gasoso quente → T ↑, ultrassom ↑, densidade ↓
        "mul_temperature": 1.20 * s,
        "mul_ultrasonic_noise": 1.20 * s,
        "mul_vibration_z": 1.10,
        "mul_density": 0.98,
    },

    # ======================== Mecânica ========================
    "bearing_wear": lambda s: {
        "mul_vibration_x": 1.55 * s,
        "mul_vibration_y": 1.55 * s,
        "mul_ultrasonic_noise": 1.10 * s,
        "add_ferrous_particles": 10 * s,
        "mul_temperature": 1.05 * s,
    },

    "bearing_unlubricated": lambda s: {
        "mul_temperature": 1.18 * s,
        "mul_vibration_x": 1.35 * s,
        "mul_vibration_y": 1.35 * s,
        "add_ferrous_particles": 12 * s,
        "mul_ultrasonic_noise": 1.15 * s,
    },

    "misalignment": lambda s: {
        # 1X dominante no eixo X
        "mul_vibration_x": 1.80 * s,
        "mul_vibration_y": 1.20 * s,
        "mul_motor_current": 1.10 * s,   # torque ↑
        "mul_temperature": 1.05,
    },

    "unbalance": lambda s: {
        # Radial (Y) mais afetado
        "mul_vibration_y": 1.80 * s,
        "mul_vibration_x": 1.20 * s,
        "mul_vibration_z": 1.10 * s,
    },

    "looseness": lambda s: {
        # Folgas → vib broadband ↑ em todos os eixos
        "mul_vibration_x": 1.45 * s,
        "mul_vibration_y": 1.45 * s,
        "mul_vibration_z": 1.30 * s,
    },

    "shaft_bent": lambda s: {
        "mul_vibration_z": 1.80 * s,
        "mul_vibration_x": 1.25 * s,
        "mul_vibration_y": 1.25 * s,
        "mul_motor_current": 1.10 * s,
    },

    "seal_leak": lambda s: {
        # Fuga → ultrassom ↑, pressões ↓ ligeiro, flow ↓, fuga ↑, T do selo ↑
        "mul_ultrasonic_noise": 1.45 * s,
        "mul_discharge_pressure": 0.96,
        "mul_suction_pressure": 0.98,
        "mul_flow": 0.95,
        "add_leakage_rate": 30 * s,     # mais visível
        "mul_seal_temp": 1.10 * s,
    },

    "seal_dry": lambda s: {
        # Selo a seco → T selo ↑, ultrassom ↑
        "mul_seal_temp": 1.20 * s,
        "mul_temperature": 1.10 * s,
        "mul_ultrasonic_noise": 1.60 * s,
    },

    "structural_fault": lambda s: {
        # Base/carcaça → vib ↑ generalizado
        "mul_vibration_x": 1.35 * s,
        "mul_vibration_y": 1.35 * s,
        "mul_vibration_z": 1.25 * s,
    },

    # ===================== Elétrica / Motor =====================
    "motor_overload": lambda s: {
        "mul_motor_current": 1.35 * s,
        "mul_temperature": 1.15 * s,
        "mul_rpm": 0.98,                 # queda ligeira de velocidade
        "mul_vibration_x": 1.10 * s,
        "mul_vibration_y": 1.10 * s,
        "mul_power_factor": 0.98,        # PF cai ligeiro
    },

    "electrical_fault": lambda s: {
        "mul_motor_current": 1.25 * s,
        "mul_temperature": 1.08,
        "mul_vibration_x": 1.15 * s,
        "mul_vibration_y": 1.15 * s,
        "mul_power_factor": 0.96,        # PF cai
    },

    "single_phasing": lambda s: {
        # Fase B baixa, A/C altas → rpm ↓, vib ↑, PF ↓, T ↑ leve
        "mul_current_A": 1.60 * s,
        "mul_current_B": 0.60,
        "mul_current_C": 1.60 * s,
        "mul_rpm": 0.90,
        "mul_vibration_x": 1.20 * s,
        "mul_vibration_y": 1.20 * s,
        "mul_power_factor": 0.95,
        "mul_temperature": 1.05,
    },

    "vfd_issue": lambda s: {
        # Distorção/EMC → vib ↑ e ligeira deriva de frequência; PF ↓ leve
        "mul_vibration_x": 1.20 * s,
        "mul_vibration_y": 1.20 * s,
        "mul_motor_current": 1.12 * s,
        "add_frequency": 0.30,           # ~0.3 Hz de offset
        "mul_power_factor": 0.98,
    },

    "insulation_breakdown": lambda s: {
        "mul_motor_current": 1.20 * s,
        "mul_temperature": 1.20 * s,
        "mul_ultrasonic_noise": 1.05 * s,
    },

    "bearing_fluting": lambda s: {
        "mul_vibration_x": 1.50 * s,
        "mul_vibration_y": 1.50 * s,
        "add_ferrous_particles": 8 * s,
    },

    # =================== Lubrificante / Ambiente ===================
    "oil_degraded": lambda s: {
        # Shear/oxidação → viscosidade efetiva ↓, T ↑, vib ↑
        "mul_viscosity": 0.85,
        "mul_temperature": 1.10,
        "mul_vibration_x": 1.15 * s,
        "mul_vibration_y": 1.15 * s,
    },

    "oil_wrong": lambda s: {
        # Grau errado (visc alta) → corrente ↑, T ↑, flow ↓ leve, vib Z ↑ leve
        "mul_viscosity": 1.50 * s,
        "mul_temperature": 1.10,
        "mul_motor_current": 1.08 * s,
        "mul_flow": 0.95,
        "mul_vibration_z": 1.05 * s,
    },

    "overtemperature": lambda s: {
        "mul_temperature": 1.35 * s,
        "mul_viscosity": 0.85,
        "mul_vibration_z": 1.10 * s,
    },

    "under_temperature": lambda s: {
        "mul_temperature": 0.85,
        "mul_viscosity": 1.25 * s,
        "mul_flow": 0.92,
        "mul_motor_current": 1.05 * s,   # esforço ↑ por fluido mais viscoso
    },

    "ambient_hot": lambda s: {
        "mul_temperature": 1.15 * s,
    },

    "ambient_cold": lambda s: {
        "mul_temperature": 0.90,
        "mul_viscosity": 1.15 * s,
    },
}


MODE_EFFECTS = {}
def _merge(dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out
for name, fn in TEMPLATES.items():
    for sev_name, sev_scale in SEV.items():
        MODE_EFFECTS[f"{name}_{sev_name}"] = fn(sev_scale)
for base in ("cavitation","bearing_wear","misalignment","unbalance","seal_leak",
             "motor_overload","low_flow","high_flow","impeller_wear","oil_degraded",
             "fluid_contaminated","electrical_fault"):
    MODE_EFFECTS[base] = TEMPLATES[base](SEV["moderate"])
COMPOSITES = {
    "npsh_cavitation_severe": ["npsh_insufficient_severe","cavitation_severe"],
    "throttled_lowflow": ["low_flow_moderate","high_pressure_moderate"],
    "seal_leak_oil_degraded": ["seal_leak_moderate","oil_degraded_moderate"],
    "bearing_wear_overheat": ["bearing_wear_severe","overtemperature_moderate"],
    "unbalance_misalignment": ["unbalance_moderate","misalignment_moderate"],
    "air_entrapment_recirculation": ["air_entrapment_moderate","recirculation_moderate"],
    "impeller_wear_low_pressure": ["impeller_wear_moderate","low_pressure_moderate"],
    "electrical_vfd_issue": ["electrical_fault_moderate","vfd_issue_moderate"],
    "single_phasing_overload": ["single_phasing_moderate","motor_overload_moderate"],
    "fluid_mismatch_hot": ["fluid_mismatch_moderate","overtemperature_moderate"],
}
for cname, parts in COMPOSITES.items():
    MODE_EFFECTS[cname] = _merge([MODE_EFFECTS[p] for p in parts])
ALL_MODES = sorted(MODE_EFFECTS.keys()) + ["normal"]
def apply_effects(sample: dict, effects: dict, noise=0.03) -> dict:
    out = sample.copy()
    lims = SENSOR_LIMITS()
    for k, v in sample.items():
        mul = effects.get(f"mul_{k}", 1.0)
        add = effects.get(f"add_{k}", 0.0)
        val = v * mul + add
        val += random.gauss(0, abs(val) * noise)
        if k in lims:
            val = clamp(val, *lims[k])
        out[k] = val
    if not any(x in effects for x in ("mul_overall_vibration","add_overall_vibration")):
        if all(a in out for a in ("vibration_x","vibration_y","vibration_z")):
            ov = math.sqrt(out["vibration_x"]**2 + out["vibration_y"]**2 + out["vibration_z"]**2)
            out["overall_vibration"] = clamp(ov, *lims["overall_vibration"])
    return out
def synthesize_mode_sample(base_sample: dict, mode: str) -> dict:
    effects = MODE_EFFECTS.get(mode, {})
    out = apply_effects(base_sample, effects, noise=0.03)
    if mode.startswith("surge") and random.random() < 0.15:
        lims = SENSOR_LIMITS()
        out["pressure"] = clamp(out["pressure"] * random.uniform(1.2, 1.5), *lims["pressure"])
        out["overall_vibration"] = clamp(out["overall_vibration"] * random.uniform(1.2, 1.4), *lims["overall_vibration"])
    return out
# =========================
# GERAÇÃO DE AMOSTRA
# =========================
def _rand_triple(triple):
    lo, mid, hi = triple
    mu = mid
    sigma = (hi - lo) / 6.0 if hi > lo else 1.0
    return max(lo, min(hi, random.gauss(mu, sigma)))
def _base_nominal():
    t = time.time() + STATE["seed"]
    base = {k: v["nominal"] for k, v in SENSOR_META.items()}
    # oscilações lentas em alguns canais
    base["temperature"] += 5.0 * math.sin(2*math.pi * (1/300.0) * t)
    base["pressure"] += 0.2 * math.sin(2*math.pi * (1/120.0) * t)
    base["ultrasonic_noise"] += 0.5 * math.sin(2*math.pi * (1/60.0) * t)
    # coerência hidráulica
    if "discharge_pressure" in base and "suction_pressure" in base:
        base["delta_p"] = clamp(
            base["discharge_pressure"] - base["suction_pressure"],
            SENSOR_META["delta_p"]["min"], SENSOR_META["delta_p"]["max"]
        )
    # RPM do estado
    base["rpm"] = STATE["rpm"]
    # overall vib de eixos
    if all(a in base for a in ("vibration_x","vibration_y","vibration_z")):
        base["overall_vibration"] = math.sqrt(base["vibration_x"]**2 + base["vibration_y"]**2 + base["vibration_z"]**2)
    return base
def next_sample():
    # construir base dependente do perfil
    prof = PROFILES.get(STATE["profile"], PROFILES["agua_potavel"])
    base = _base_nominal()
    # substituir alguns canais por amostragem com “triple” do perfil
    base["pressure"] = _rand_triple(prof["pressure"])
    base["flow"] = _rand_triple(prof["flow"])
    base["density"] = _rand_triple(prof["density"])
    base["ultrasonic_noise"] = _rand_triple(prof.get("ultrasonic_base", (30.0, 50.0, 80.0)))
    base["temperature"] = _rand_triple(prof.get("temp_base", (60.0, 85.0, 110.0)))
    base["motor_current"] = _rand_triple(prof.get("current_base", (5.0, 8.0, 12.0)))
    # viscosidade: se vindo em cSt muito alto, converte para escala relativa 0.3–5.0
    visc_nom = prof["viscosity"][1]
    if visc_nom > 10:
        lo, hi = prof["viscosity"][0], prof["viscosity"][2]
        visc_rel = 0.3 + 4.7 * (min(visc_nom, 500.0) - lo) / max(1.0, (hi - lo))
        base["viscosity"] = visc_rel
    else:
        base["viscosity"] = _rand_triple((0.3, 1.0, 5.0))
    # aplica efeitos do modo
    mode = STATE["mode"]
    if mode != "normal":
        sample = synthesize_mode_sample(base, mode)
    else:
        sample = apply_effects(base, {}, noise=0.02)
    # ramp/clamp por sensor e arredondamentos
    smooth = {}
    prev = STATE["last_sample"]
    lims = SENSOR_LIMITS()
    for k, v in sample.items():
        meta = SENSOR_META.get(k)
        if meta:
            v = _ramp(None if prev is None else prev.get(k), v, meta["rate"])
            v = clamp(v, meta["min"], meta["max"])
        smooth[k] = v
    # coerências derivadas
    if "discharge_pressure" in smooth and "suction_pressure" in smooth:
        smooth["delta_p"] = clamp(
            smooth["discharge_pressure"] - smooth["suction_pressure"],
            SENSOR_META["delta_p"]["min"], SENSOR_META["delta_p"]["max"]
        )
    # overall_vibration recalculado
    if all(a in smooth for a in ("vibration_x","vibration_y","vibration_z")):
        ov = math.sqrt(smooth["vibration_x"]**2 + smooth["vibration_y"]**2 + smooth["vibration_z"]**2)
        smooth["overall_vibration"] = clamp(ov, SENSOR_META["overall_vibration"]["min"], SENSOR_META["overall_vibration"]["max"])
    # arredondar
    rounded = {}
    for k, v in smooth.items():
        if k in ("ferrous_particles","particle_count","oil_water_ppm","leakage_rate"):
            rounded[k] = int(round(v))
        elif "vibration" in k:
            rounded[k] = round(v, 4)
        elif k in ("rpm",):
            rounded[k] = round(v, 1)
        else:
            rounded[k] = round(v, 3)
    STATE["last_sample"] = rounded
    return rounded
# =========================
# LOG CSV / PARQUET
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "sensors_log.csv")
PARQUET_PATH = os.path.join(LOG_DIR, "sensors_log.parquet")
def _ensure_csv_header(fields):
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "mode"] + fields)
def _log_row(row: dict):
    fields = sorted([k for k in row.keys() if k not in ("timestamp", "mode", "alerts")])
    _ensure_csv_header(fields)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get("timestamp",""), row.get("mode","")] + [row[k] for k in fields])
    if PANDAS_AVAILABLE:
        df = pd.DataFrame([row])
        if not os.path.exists(PARQUET_PATH):
            df.to_parquet(PARQUET_PATH, index=False)
        else:
            old = pd.read_parquet(PARQUET_PATH)
            pd.concat([old, df], ignore_index=True).to_parquet(PARQUET_PATH, index=False)
# =========================
# ALERTAS
# =========================
def generate_alerts(d):
    alerts = []
    if d.get("temperature", 0) >= 90: alerts.append("High temperature — thermal risk.")
    elif d.get("temperature", 0) < 65: alerts.append("Low temperature — check process conditions.")
    if d.get("pressure", 0) > 5.0: alerts.append("Pressure above operational limit.")
    elif d.get("pressure", 0) < 1.2: alerts.append("Low pressure — possible restriction/air ingress.")
    if d.get("overall_vibration", 0) > 0.04: alerts.append("Vibration above limit — possible shaft/bearing wear.")
    if d.get("ultrasonic_noise", 0) > 70: alerts.append("High ultrasonic noise — cavitation/leak likely.")
    if d.get("ferrous_particles", 0) > 20: alerts.append("High ferrous particles — internal wear detected.")
    if d.get("motor_current", 0) > 12: alerts.append("High motor current — overload or misalignment.")
    return alerts if alerts else ["System within parameters."]

def _extract_base_mode(mode_str: str) -> Optional[str]:
    """Return first base key that is contained in the current mode string."""
    if not mode_str or mode_str == "normal":
        return None
    for base in OPERATION_FIXES.keys():
        if base in mode_str:
            return base
    return None

# =========================
# OPERATIONS (reais)
# =========================
OPERATIONS = []  # cada item: {"timestamp": iso, "action": str}
# =========================
# ML REAL (carregamento e inferência) — ADIÇÃO
# =========================
TF_BUNDLE = None
try:
    import tensorflow as tf
    import json
    from pathlib import Path
    MODEL_PATH = Path("models/pump_mode_classifier.keras")
    SCALER_PATH = Path("models/scaler.json")
    LABELS_PATH = Path("models/labels.json")
    if MODEL_PATH.exists() and SCALER_PATH.exists() and LABELS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "r", encoding="utf-8") as f:
            scaler = json.load(f)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        TF_BUNDLE = (model, scaler, labels)
        print(f"[main.py] Modelo ML carregado com {len(labels)} classes.")
    else:
        print("[main.py] Artefactos de ML não encontrados, a usar heurísticas.")
except Exception as e:
    print("[main.py] TensorFlow não disponível, fallback heurístico:", e)
    TF_BUNDLE = None
def infer_tf(sample: dict) -> dict:
    """
    Corre inferência no modelo treinado (se existir).
    Se não existir, devolve heurística _ml_metrics(sample).
    """
    if not TF_BUNDLE:
        return _ml_metrics(sample)
    model, scaler, labels = TF_BUNDLE
    feature_names = scaler["feature_names"]
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    x = np.zeros((1, len(feature_names)), dtype=np.float32)
    for i, name in enumerate(feature_names):
        v = sample.get(name, 0.0)
        x[0, i] = float(v)
    # normalizar
    x = (x - mean) / std
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    conf = float(probs[pred_idx])
    # Mapear para as 5 métricas esperadas pela UI (simples e estável)
    return {
        "anomaly_score": round(1.0 - conf, 3),                      # baixa confiança => mais anomalia
        "failure_probability": round((1.0 - conf) if pred_label != "normal" else 0.0, 3),
        "health_index": round(100.0 * conf, 1),                     # confiança ~ “saúde”
        "rul_minutes": int(max(0, 600 * conf)),                     # aproximado (placeholder)
        "model_confidence": round(conf, 3),
        "predicted_mode": pred_label,
    }





# =========================
# ENDPOINTS
# =========================
@app.get("/profile")
def get_profile():
    return {"profile": STATE["profile"], "available": list(PROFILES.keys())}
@app.post("/profile")
def set_profile(profile: str = Query(..., description="Nome do perfil")):
    if profile not in PROFILES:
        return {"ok": False, "error": "Perfil desconhecido", "available": list(PROFILES.keys())}
    STATE["profile"] = profile
    _apply_profile(profile)
    # reset sample para evitar salto brusco
    STATE["last_sample"] = None
    return {"ok": True, "profile": STATE["profile"]}
@app.get("/modes")
def list_modes():
    return {"default": "normal", "available": ["normal"] + sorted(ALL_MODES)}
@app.get("/mode")
def get_mode():
    return {"mode": STATE["mode"], "rpm": STATE["rpm"], "profile": STATE["profile"]}
@app.post("/mode")
def set_mode(mode: str = Query(..., description="Nome do modo (ou 'normal')")):
    if mode != "normal" and mode not in MODE_EFFECTS:
        return {"ok": False, "error": "Modo desconhecido", "hint": "/modes"}
    STATE["mode"] = mode
    return {"ok": True, "mode": STATE["mode"]}
@app.post("/rpm")
def set_rpm(rpm: float = Query(ge=100.0, le=6000.0)):
    STATE["rpm"] = float(rpm)
    return {"ok": True, "rpm": STATE["rpm"]}
@app.get("/operations")
def get_operations(range: str = "today"):
    now = datetime.datetime.now()
    if range == "today":
        cutoff = now - datetime.timedelta(hours=24)
    elif range == "week":
        cutoff = now - datetime.timedelta(hours=168)
    else:
        cutoff = None
    data = []
    for op in OPERATIONS:
        ts = datetime.datetime.fromisoformat(op["timestamp"])
        if cutoff and ts < cutoff:
            continue
        # Valores antes e depois (mock → substituir depois por real)
        before = random.uniform(60, 80)   # exemplo
        after = before + random.uniform(2, 10)
        delta = after - before
        data.append({
            "timestamp": op["timestamp"],
            "action": op["action"],
            "before": round(before, 1),
            "after": round(after, 1),
            "delta": round(delta, 1),
        })
    data.sort(key=lambda x: x["timestamp"], reverse=True)
    return data
@app.post("/operations/create")
def create_operation(
    action: str = Body(..., embed=True),
    mode: Optional[str] = Body(None, embed=True),
):
    now = datetime.datetime.now().isoformat()
    server_mode = STATE["mode"]
    provided_mode = mode
    current_mode = (provided_mode or server_mode) or "normal"
    op_entry = {
        "timestamp": now,
        "action": action,
        "mode_hint": provided_mode,
        "previous_mode": server_mode,
        "corrected": False,
        "new_mode": server_mode,
    }
    OPERATIONS.append(op_entry)
    bases = _extract_base_modes(current_mode)
    corrected = False
    new_mode = server_mode
    print(f"[create_operation] action={action!r} mode_hint={mode!r} server_mode={server_mode!r} bases={bases!r}")
    if bases:
        given = _normalize_action(action)
        expected_single = {_normalize_action(OPERATION_FIXES[b]) for b in bases if b in OPERATION_FIXES}
        # (a) casa com QUALQUER ação simples de base
        hit_single = any(given == exp for exp in expected_single)
        # (b) casa com ação COMPOSTA "a + b + ..."
        combo_text = " + ".join([OPERATION_FIXES[b] for b in bases if b in OPERATION_FIXES])
        hit_combo = given == _normalize_action(combo_text)
        if hit_single or hit_combo:
            STATE["mode"] = "normal"
            new_mode = "normal"
            corrected = True
            OPERATIONS[-1]["corrected"] = True
            OPERATIONS[-1]["new_mode"] = "normal"
    return {
        "ok": True,
        "timestamp": now,
        "action": action,
        "previous_mode": current_mode,
        "new_mode": new_mode,
        "corrected": corrected,
    }

@app.get("/sensors")
def sensors(mode: Optional[str] = None):
    if mode is not None:
        STATE["mode"] = mode if (mode == "normal" or mode in MODE_EFFECTS) else "normal"
    d = next_sample()
    # ===== ML real se existir; senão heurístico =====
    ml = infer_tf(d)
    d.update(ml)
    d["timestamp"] = datetime.datetime.now().isoformat()
    d["mode"] = STATE["mode"]
    d["temp_ok"] = 1.0 if d.get("temperature", 0) < 90 else 0.0
    d["vib_ok"]  = 1.0 if d.get("overall_vibration", 0) <= 0.04 else 0.0
    d["alerts"] = generate_alerts(d)
    history.append(d)
    if len(history) > 10000:
        history.pop(0)
    _log_row(d)
       # attach backend-side recommendation (does NOT go to CSV)
    bases = _extract_base_modes(d["mode"])
    base_mode = bases[0] if bases else None
    if base_mode:
        d["recommended_operation"] = OPERATION_FIXES.get(base_mode)
    else:
        d["recommended_operation"] = None
    return d

@app.get("/history")
def get_history(range: str = "all"):
    now = datetime.datetime.now()
    if range == "10s":
        cutoff = now - datetime.timedelta(seconds=10)
    elif range == "1min":
        cutoff = now - datetime.timedelta(minutes=1)
    elif range == "1h":
        cutoff = now - datetime.timedelta(hours=1)
    elif range == "24h":
        cutoff = now - datetime.timedelta(hours=24)
    else:
        cutoff = None
    if cutoff:
        return [d for d in history if datetime.datetime.fromisoformat(d["timestamp"]) >= cutoff]
    return history
@app.get("/alerts")
def get_alerts():
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(seconds=5)
    recent = [d for d in history if datetime.datetime.fromisoformat(d["timestamp"]) >= cutoff]
    if not recent:
        return ["No recent data"]
    all_alerts = []
    for d in recent:
        all_alerts.extend(generate_alerts(d))
    return list(set(all_alerts))
# =========================
# VIBRATION WAVEFORM / FFT
# =========================
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[win:] - c[:-win]) / float(win)
    pad_left = win // 2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="edge")
@app.get("/vibration/waveform")
def vibration_waveform(fs: int = 10000, duration: float = 2.0):
    fs = int(max(500, min(fs, 100_000)))
    duration = float(max(0.1, min(duration, 5.0)))
    n = int(fs * duration)
    t = np.arange(n) / fs
    rpm = STATE["rpm"]
    f1 = rpm / 60.0
    f2 = 2.0 * f1
    sig = (
        0.006 * np.sin(2*np.pi * f1 * t) +
        0.003 * np.sin(2*np.pi * f2 * t) +
        0.002 * np.sin(2*np.pi * 150 * t)
    )
    sig += np.random.normal(0.0, 0.002, n)
    rms_g = float(np.sqrt(np.mean(sig**2)))
    pp_g = float(np.max(sig) - np.min(sig))
    crest_factor = float(np.max(np.abs(sig)) / (rms_g if rms_g > 1e-12 else 1.0))
    x = sig - np.mean(sig)
    m2 = np.mean(x**2)
    m4 = np.mean(x**4)
    kurtosis_excess = float((m4 / (m2**2 + 1e-18)) - 3.0)
    env = _moving_average(np.abs(sig), max(5, int(0.002 * fs)))
    envelope_rms_g = float(np.sqrt(np.mean(env**2)))
    g0 = 9.80665
    acc_ms2 = sig * g0
    vel_ms = np.cumsum(acc_ms2) / fs
    velocity_rms_ms = float(np.sqrt(np.mean(vel_ms**2)))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft_vals = np.fft.rfft(sig)
    fft_mag = np.abs(fft_vals) * (2.0 / n)
    ts = datetime.datetime.now().isoformat()
    window_id = f"{int(time.time())}_{fs}_{duration}"
    payload = {
        "timestamp": ts,
        "fs": fs,
        "duration": duration,
        "window_id": window_id,
        "n": n,
        "samples": sig.tolist(),
        "freqs": freqs.tolist(),
        "fft_mag": fft_mag.tolist(),
        "features": {
            "rms_g": rms_g,
            "pp_g": pp_g,
            "crest_factor": crest_factor,
            "kurtosis_excess": kurtosis_excess,
            "envelope_rms_g": envelope_rms_g,
            "velocity_rms_ms": velocity_rms_ms,
        }
    }
    return payload






