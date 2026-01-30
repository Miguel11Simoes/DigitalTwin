#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_causal_dataset.py
Dataset sintético com ASSINATURAS CAUSAIS para ML funcionar.

CHAVE: Cada modo tem padrões espectrais ÚNICOS que o modelo pode aprender.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

np.random.seed(42)

N_ASSETS = 20
SAMPLES_PER_ASSET = 6000

# Assinaturas espectrais DISTINTAS por modo
MODES = {
    "normal_operation": {
        "vib_base": 0.5, "vib_noise": 0.08,
        "freqs": [50.0], "amps": [1.0],
        "pressure_off": 0.0, "current_off": 0.0, "temp_off": 0.0,
    },
    "bearing_wear": {
        "vib_base": 1.8, "vib_noise": 0.25,
        "freqs": [50.0, 120.0, 240.0], "amps": [1.0, 0.9, 0.5],
        "pressure_off": -0.08, "current_off": 0.12, "temp_off": 6.0,
    },
    "cavitation": {
        "vib_base": 2.2, "vib_noise": 0.5,
        "freqs": [50.0, 500.0, 900.0], "amps": [1.0, 1.3, 0.7],
        "pressure_off": -0.2, "current_off": 0.18, "temp_off": 4.0,
    },
    "imbalance": {
        "vib_base": 2.0, "vib_noise": 0.18,
        "freqs": [50.0, 100.0], "amps": [2.2, 0.4],
        "pressure_off": 0.0, "current_off": 0.08, "temp_off": 3.0,
    },
    "misalignment": {
        "vib_base": 1.7, "vib_noise": 0.22,
        "freqs": [50.0, 100.0, 150.0], "amps": [1.0, 1.8, 0.9],
        "pressure_off": -0.05, "current_off": 0.15, "temp_off": 5.0,
    },
}

SEVERITIES = {
    "normal":   {"mult": 1.0, "health": (85, 100), "rul": (700, 1000)},
    "early":    {"mult": 1.4, "health": (60, 85),  "rul": (400, 700)},
    "moderate": {"mult": 2.0, "health": (35, 60),  "rul": (150, 400)},
    "severe":   {"mult": 2.8, "health": (10, 35),  "rul": (30, 150)},
    "failure":  {"mult": 4.0, "health": (0, 10),   "rul": (0, 30)},
}


def gen_vibration(n, mode_cfg, sev_mult, asset_scale):
    t = np.arange(n) / 100.0
    sig = np.zeros(n, dtype=np.float32)
    for f, a in zip(mode_cfg["freqs"], mode_cfg["amps"]):
        sig += a * mode_cfg["vib_base"] * sev_mult * asset_scale * np.sin(2*np.pi*f*t + np.random.uniform(0, 2*np.pi))
    sig += np.random.normal(0, mode_cfg["vib_noise"] * sev_mult * asset_scale, n).astype(np.float32)
    return sig


def gen_asset(asset_id, n_samples, asset_scale, asset_offsets):
    rows = []
    ts = datetime(2025, 1, 1)
    seg_size = n_samples // 21  # 5 modes * ~4 severities avg
    
    for mode_name, mode_cfg in MODES.items():
        # normal_operation só tem normal/early severity
        sevs = ["normal", "early"] if mode_name == "normal_operation" else list(SEVERITIES.keys())
        
        for sev_name in sevs:
            sev_cfg = SEVERITIES[sev_name]
            sev_mult = sev_cfg["mult"]
            h_lo, h_hi = sev_cfg["health"]
            r_lo, r_hi = sev_cfg["rul"]
            
            vib = gen_vibration(seg_size, mode_cfg, sev_mult, asset_scale)
            vib_x = vib * (1.0 + 0.08*np.random.randn(seg_size)).astype(np.float32)
            vib_y = vib * (0.85 + 0.08*np.random.randn(seg_size)).astype(np.float32)
            vib_z = vib * (0.65 + 0.08*np.random.randn(seg_size)).astype(np.float32)
            overall = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2) / 1.73
            
            ultra_base = 2.5 if mode_name == "cavitation" else 0.6
            ultra = (ultra_base * sev_mult * asset_scale + np.random.normal(0, 0.08*sev_mult, seg_size)).astype(np.float32)
            
            pres = (5.0 + asset_offsets["p"] + mode_cfg["pressure_off"]*sev_mult + np.random.normal(0, 0.04, seg_size)).astype(np.float32)
            curr = (10.0 + asset_offsets["c"] + mode_cfg["current_off"]*sev_mult + np.random.normal(0, 0.08, seg_size)).astype(np.float32)
            temp = (40.0 + asset_offsets["t"] + mode_cfg["temp_off"]*sev_mult + np.random.normal(0, 0.15, seg_size)).astype(np.float32)
            flow = (100.0 + asset_offsets["f"] + np.random.normal(0, 0.8, seg_size)).astype(np.float32)
            
            health = np.linspace(h_hi, h_lo, seg_size).astype(np.float32)
            rul = np.linspace(r_hi, r_lo, seg_size).astype(np.float32)
            
            for i in range(seg_size):
                rows.append({
                    "timestamp": ts, "asset_id": asset_id,
                    "mode": mode_name, "severity": sev_name,
                    "health_index": float(health[i]), "rul_minutes": float(rul[i]),
                    "overall_vibration": float(overall[i]),
                    "vibration_x": float(vib_x[i]), "vibration_y": float(vib_y[i]), "vibration_z": float(vib_z[i]),
                    "ultrasonic_noise": float(ultra[i]),
                    "pressure": float(pres[i]), "motor_current": float(curr[i]),
                    "temperature": float(temp[i]), "flow": float(flow[i]),
                })
                ts += timedelta(seconds=0.1)
    
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("GERADOR DE DATASET CAUSAL")
    print("=" * 60)
    
    dfs = []
    for i in range(N_ASSETS):
        aid = f"PUMP_{i+1:03d}"
        scale = np.random.uniform(0.85, 1.15)
        offsets = {"p": np.random.uniform(-0.4, 0.4), "c": np.random.uniform(-0.4, 0.4),
                   "t": np.random.uniform(-2, 2), "f": np.random.uniform(-4, 4)}
        print(f"  [{i+1}/{N_ASSETS}] {aid} (scale={scale:.2f})")
        dfs.append(gen_asset(aid, SAMPLES_PER_ASSET, scale, offsets))
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n[STATS] Shape: {df.shape}")
    print(f"[STATS] Assets: {df['asset_id'].nunique()}")
    print(f"[STATS] Modes:\n{df['mode'].value_counts()}")
    print(f"[STATS] Severities:\n{df['severity'].value_counts()}")
    
    print("\n[VERIFY] Vibration by mode:")
    for m in MODES:
        d = df[df["mode"]==m]["overall_vibration"]
        print(f"  {m}: mean={d.mean():.3f}, std={d.std():.3f}")
    
    print("\n[VERIFY] Vibration by severity:")
    for s in SEVERITIES:
        d = df[df["severity"]==s]["overall_vibration"]
        if len(d) > 0:
            print(f"  {s}: mean={d.mean():.3f}, std={d.std():.3f}")
    
    out_dir = Path(__file__).parent / "logs"
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "sensors_log.csv"
    if csv_path.exists():
        bak = out_dir / f"sensors_log_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path.rename(bak)
        print(f"\n[BACKUP] {bak}")
    
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved: {csv_path} ({csv_path.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
