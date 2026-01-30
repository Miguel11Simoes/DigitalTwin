#!/usr/bin/env python3
"""
Dataset Generator v3 - Temporal Sequences for CNN Windowing
Gera dados com degradação progressiva por asset (necessário para CNN).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
np.random.seed(SEED)

SENSOR_NAMES = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "motor_current", "pressure", "flow", "temperature"
]

MODES = ["normal_operation", "bearing_wear", "cavitation", "misalignment", "imbalance"]
SEVERITIES = ["normal", "early", "moderate", "severe", "failure"]

def generate_degradation_curve(n_samples: int, rng) -> np.ndarray:
    """Gera curva de degradação progressiva (realista)."""
    # Degradação exponencial suave
    t = np.linspace(0, 1, n_samples)
    base = t ** 1.5  # Curva convexa - degradação acelera no fim
    noise = rng.normal(0, 0.02, n_samples)
    return np.clip(base + noise, 0, 1)


def degradation_to_severity(d: float) -> str:
    """Converte degradação (0-1) para severity com ranges não-overlapping."""
    if d < 0.12:
        return "normal"
    elif d < 0.30:
        return "early"
    elif d < 0.55:
        return "moderate"
    elif d < 0.80:
        return "severe"
    else:
        return "failure"


def generate_dataset(n_assets=8, samples_per_asset=10000):
    """Gera dataset com sequência temporal realista por asset."""
    rng = np.random.default_rng(SEED)
    rows = []
    
    for asset_idx in range(n_assets):
        asset_id = f"pump_{asset_idx+1:03d}"
        print(f"[INFO] Gerando asset {asset_id}...")
        
        # Cada asset tem uma curva de degradação progressiva
        degradation = generate_degradation_curve(samples_per_asset, rng)
        
        # Cada asset tem um modo de falha predominante (mais realista)
        primary_mode = rng.choice(["bearing_wear", "cavitation", "misalignment", "imbalance"])
        
        for i in range(samples_per_asset):
            timestamp = datetime(2025, 1, 1) + timedelta(days=asset_idx*30, minutes=i*5)
            d = degradation[i]
            
            # Severity baseado na degradação
            severity = degradation_to_severity(d)
            
            # Mode: normal até 10% degradação, depois modo de falha predominante
            if d < 0.10:
                mode = "normal_operation"
            else:
                # 70% primary mode, 30% random
                mode = primary_mode if rng.random() < 0.70 else rng.choice(MODES)
            
            # Health e RUL derivados da degradação
            health_index = max(0, 100 * (1.0 - d))
            rul_minutes = max(0, (samples_per_asset - i) * 5 * (1 - d*0.5))  # RUL diminui mais rápido com degradação
            
            # ===== SENSOR VALUES =====
            noise = rng.normal(0, 0.02)
            
            # Mode signatures
            if mode == "normal_operation":
                vib_mult, vib_x_ratio, vib_y_ratio, vib_z_ratio = 1.0, 0.35, 0.35, 0.30
                curr_offset, press_offset, flow_offset, temp_offset = 0.0, 0.0, 0.0, 0.0
            elif mode == "bearing_wear":
                vib_mult, vib_x_ratio, vib_y_ratio, vib_z_ratio = 1.5, 0.60, 0.25, 0.15
                curr_offset, press_offset, flow_offset, temp_offset = 0.5, -0.1, -3.0, 8.0
            elif mode == "cavitation":
                vib_mult, vib_x_ratio, vib_y_ratio, vib_z_ratio = 1.3, 0.25, 0.55, 0.20
                curr_offset, press_offset, flow_offset, temp_offset = 0.3, -0.4, -8.0, 3.0
            elif mode == "misalignment":
                vib_mult, vib_x_ratio, vib_y_ratio, vib_z_ratio = 1.4, 0.45, 0.45, 0.10
                curr_offset, press_offset, flow_offset, temp_offset = 1.0, -0.2, -5.0, 5.0
            else:  # imbalance
                vib_mult, vib_x_ratio, vib_y_ratio, vib_z_ratio = 1.35, 0.20, 0.20, 0.60
                curr_offset, press_offset, flow_offset, temp_offset = 0.4, -0.15, -4.0, 4.0
            
            # Vibração base (função da degradação)
            base_vib = 0.3 + 2.0 * d
            overall_vibration = base_vib * vib_mult * (1 + noise)
            vibration_x = overall_vibration * vib_x_ratio * (1 + rng.normal(0, 0.03))
            vibration_y = overall_vibration * vib_y_ratio * (1 + rng.normal(0, 0.03))
            vibration_z = overall_vibration * vib_z_ratio * (1 + rng.normal(0, 0.03))
            
            motor_current = 15.0 + 5.0 * d + curr_offset + rng.normal(0, 0.1)
            pressure = 2.5 - 1.0 * d + press_offset + rng.normal(0, 0.02)
            flow = 100.0 - 30.0 * d + flow_offset + rng.normal(0, 0.5)
            temperature = 45.0 + 20.0 * d + temp_offset + rng.normal(0, 0.2)
            
            row = {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "mode": mode,
                "severity": severity,
                "health_index": health_index,
                "rul_minutes": rul_minutes,
                "overall_vibration": overall_vibration,
                "vibration_x": vibration_x,
                "vibration_y": vibration_y,
                "vibration_z": vibration_z,
                "motor_current": motor_current,
                "pressure": pressure,
                "flow": flow,
                "temperature": temperature,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    # Manter ordem temporal por asset
    df = df.sort_values(['asset_id', 'timestamp']).reset_index(drop=True)
    return df


def main():
    print("=" * 70)
    print("GENERATING TEMPORAL DATASET v3 (for CNN windowing)")
    print("=" * 70)
    
    df = generate_dataset(n_assets=8, samples_per_asset=10000)
    
    # Salvar
    out_path = Path("logs/sensors_log_v2.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n[OK] Dataset gerado: {out_path}")
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Assets: {df['asset_id'].nunique()}")
    
    print("\n[INFO] Severity distribution:")
    print(df["severity"].value_counts(normalize=True).sort_index())
    
    print("\n[INFO] Mode distribution:")
    print(df["mode"].value_counts(normalize=True).sort_index())
    
    # Verificar separabilidade
    print("\n[VERIFY] Vibration by severity:")
    for sev in SEVERITIES:
        d = df[df["severity"] == sev]["overall_vibration"]
        if len(d) > 0:
            print(f"  {sev:10s}: min={d.min():.3f}, max={d.max():.3f}, mean={d.mean():.3f}")
    
    print("\n[VERIFY] Temporal sequence (first asset):")
    first_asset = df[df['asset_id'] == 'pump_001'].head(10)
    print(first_asset[['timestamp', 'severity', 'health_index']].to_string())


if __name__ == "__main__":
    main()
