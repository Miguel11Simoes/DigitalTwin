#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_realistic_dataset.py
Gera dataset sintético de ALTA QUALIDADE para manutenção preditiva industrial.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

N_ASSETS = 8
SAMPLES_PER_ASSET = 5000

SENSOR_NAMES = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "ultrasonic_noise", "motor_current", "pressure", "flow", "temperature",
    "bearing_temp_DE", "bearing_temp_NDE", "casing_temp"
]

REGIMES = {
    "normal": {"prob": 0.65, "rpm": 1500, "flow": 100, "load": 75},
    "high_flow": {"prob": 0.12, "rpm": 1750, "flow": 130, "load": 90},
    "low_flow": {"prob": 0.10, "rpm": 1200, "flow": 60, "load": 50},
    "start": {"prob": 0.05, "rpm": 800, "flow": 30, "load": 40},
    "stop": {"prob": 0.05, "rpm": 300, "flow": 10, "load": 20},
    "overload": {"prob": 0.03, "rpm": 1800, "flow": 140, "load": 110}
}

MODE_POOL = [
    {"name": "normal_operation", "prob": 0.65},
    {"name": "bearing_wear_moderate", "prob": 0.10},
    {"name": "bearing_wear_severe", "prob": 0.03},
    {"name": "cavitation_moderate", "prob": 0.08},
    {"name": "cavitation_severe", "prob": 0.02},
    {"name": "impeller_wear_moderate", "prob": 0.06},
    {"name": "impeller_wear_severe", "prob": 0.02},
    {"name": "electrical_fault_moderate", "prob": 0.03},
    {"name": "electrical_fault_severe", "prob": 0.01}
]

SEVERITY_POOL = ["normal", "early", "moderate", "severe", "failure"]

def generate_sensor_baseline():
    return {
        "overall_vibration": 2.5, "vibration_x": 1.8, "vibration_y": 1.6, "vibration_z": 1.4,
        "ultrasonic_noise": 45.0, "motor_current": 28.5, "pressure": 4.2, "flow": 100.0,
        "temperature": 65.0, "bearing_temp_DE": 58.0, "bearing_temp_NDE": 56.0, "casing_temp": 62.0
    }

def apply_regime_effect(baseline, regime_name):
    regime = REGIMES[regime_name]
    rpm_f = regime["rpm"] / 1500.0
    flow_f = regime["flow"] / 100.0
    load_f = regime["load"] / 75.0
    modified = baseline.copy()
    for v in ["overall_vibration", "vibration_x", "vibration_y", "vibration_z"]:
        modified[v] *= (0.7 * rpm_f + 0.3 * load_f)
    modified["motor_current"] *= load_f
    modified["flow"] *= flow_f
    modified["pressure"] *= (0.8 * flow_f + 0.2 * rpm_f)
    for t in ["temperature", "bearing_temp_DE", "bearing_temp_NDE", "casing_temp"]:
        modified[t] += (load_f - 1.0) * 15.0
    modified["ultrasonic_noise"] += (rpm_f - 1.0) * 10.0
    return modified

def apply_fault_signature(sensors, mode_name, severity_idx):
    sf = 1.0 + severity_idx * 0.4
    if "bearing_wear" in mode_name:
        sensors["overall_vibration"] *= sf * 1.5
        sensors["bearing_temp_DE"] += severity_idx * 12.0
        sensors["bearing_temp_NDE"] += severity_idx * 10.0
        sensors["ultrasonic_noise"] += severity_idx * 8.0
    elif "cavitation" in mode_name:
        sensors["overall_vibration"] *= sf * 1.3
        sensors["ultrasonic_noise"] += severity_idx * 15.0
        sensors["pressure"] -= severity_idx * 0.3
        sensors["flow"] -= severity_idx * 8.0
    elif "impeller_wear" in mode_name:
        sensors["pressure"] -= severity_idx * 0.4
        sensors["flow"] -= severity_idx * 12.0
        sensors["motor_current"] += severity_idx * 3.0
        sensors["vibration_x"] *= sf * 1.2
    elif "electrical_fault" in mode_name:
        sensors["motor_current"] += severity_idx * 5.0 + np.random.uniform(-3, 3)
        sensors["temperature"] += severity_idx * 8.0
        sensors["casing_temp"] += severity_idx * 6.0
    return sensors

def add_noise(sensors, level=0.03):
    for k in sensors:
        sensors[k] += np.random.normal(0, abs(sensors[k]) * level)
    return sensors

def add_transient(sensors, regime, is_transient):
    if not is_transient:
        return sensors
    if regime == "start":
        sensors["motor_current"] *= 1.8
        sensors["overall_vibration"] *= 1.5
        sensors["vibration_x"] *= 1.6
        sensors["pressure"] *= 0.6
    elif regime == "stop":
        sensors["motor_current"] *= 0.3
        sensors["pressure"] *= 0.4
        sensors["flow"] *= 0.2
    return sensors

print("[INFO] Gerando dataset sintético REALISTA de alta qualidade...")
rows = []
assets = [f"PUMP_{i:02d}" for i in range(1, N_ASSETS + 1)]
mode_names = [m["name"] for m in MODE_POOL]
mode_probs = [m["prob"] for m in MODE_POOL]
regime_names = list(REGIMES.keys())
regime_probs = [REGIMES[r]["prob"] for r in regime_names]

for ai, asset in enumerate(assets):
    print(f"  {asset}...")
    # Variar health inicial (alguns ativos começam mais degradados)
    initial_health = np.random.choice([95, 85, 70, 55, 40], p=[0.5, 0.25, 0.15, 0.07, 0.03])
    baseline = generate_sensor_baseline()
    for k in baseline:
        baseline[k] *= np.random.uniform(0.95, 1.05)
    start_time = datetime(2025, 1, 1) + timedelta(days=ai * 30)
    
    for i in range(SAMPLES_PER_ASSET):
        ts = start_time + timedelta(minutes=i * 5)
        progress = i / SAMPLES_PER_ASSET
        # Degradação mais agressiva
        degradation = (progress ** 1.2) * (1.5 - initial_health / 100.0)
        health = max(0, initial_health - degradation * 80)
        rul = max(0, (health / 100.0) ** 1.5 * 8000)
        
        if health >= 85:
            sev = "normal"
        elif health >= 70:
            sev = "early"
        elif health >= 50:
            sev = "moderate"
        elif health >= 25:
            sev = "severe"
        else:
            sev = "failure"
        
        sev_idx = SEVERITY_POOL.index(sev)
        
        if sev in ["severe", "failure"]:
            mode_cand = [m for m in mode_names if "severe" in m or m == "normal_operation"]
            mode = np.random.choice(mode_cand)
        else:
            mode = np.random.choice(mode_names, p=mode_probs)
        
        regime = np.random.choice(regime_names, p=regime_probs)
        is_trans = (regime in ["start", "stop"]) and (np.random.rand() < 0.3)
        
        sensors = baseline.copy()
        sensors = apply_regime_effect(sensors, regime)
        sensors = apply_fault_signature(sensors, mode, sev_idx)
        sensors = add_transient(sensors, regime, is_trans)
        sensors = add_noise(sensors, np.random.uniform(0.01, 0.05))
        
        row = {"timestamp": ts, "asset_id": asset, "mode": mode, "severity": sev,
               "health_index": health, "rul_minutes": rul, "regime": regime, "is_transient": is_trans}
        row.update(sensors)
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\n[STATS] Total: {len(df)} | Ativos: {df['asset_id'].nunique()}")
print("Severity:")
for s in SEVERITY_POOL:
    c = (df["severity"] == s).sum()
    print(f"  {s:12s}: {c:5d} ({c/len(df)*100:5.2f}%)")

out_path = Path(__file__).parent / "logs" / "sensors_log.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n[OK] Dataset salvo: {out_path}")
