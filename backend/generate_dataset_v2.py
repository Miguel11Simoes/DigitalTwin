#!/usr/bin/env python3
"""
Dataset Generator v2 - Improved Severity Separability
A severity é derivada diretamente da degradação, não do mode.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# Sensores
SENSOR_NAMES = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "motor_current", "pressure", "flow", "temperature"
]

# Modos de operação simplificados
MODES = [
    "normal_operation",
    "bearing_wear",
    "cavitation", 
    "misalignment",
    "imbalance"
]

def generate_dataset(n_assets=8, samples_per_asset=10000):
    """Gera dataset com severity claramente separável."""
    rng = np.random.default_rng(SEED)
    rows = []
    
    for asset_idx in range(n_assets):
        asset_id = f"pump_{asset_idx+1:03d}"
        print(f"[INFO] Gerando asset {asset_id}...")
        
        # Gerar sequência de degradação
        for i in range(samples_per_asset):
            # Tempo
            timestamp = datetime(2025, 1, 1) + timedelta(days=asset_idx*30, minutes=i*5)
            
            # ===== SEVERITY BASEADO EM THRESHOLDS CLAROS =====
            # Escolher aleatoriamente a severity (balanceado)
            severity_choice = rng.choice(["normal", "early", "moderate", "severe", "failure"], 
                                         p=[0.2, 0.2, 0.2, 0.2, 0.2])
            
            # Definir ranges DISTINTOS para cada severity
            if severity_choice == "normal":
                degradation = rng.uniform(0.00, 0.10)  # 0-10%
                health_index = rng.uniform(90, 100)
                rul_minutes = rng.uniform(8000, 10000)
            elif severity_choice == "early":
                degradation = rng.uniform(0.12, 0.25)  # 12-25% (gap de 2%)
                health_index = rng.uniform(75, 88)     # gap de 2%
                rul_minutes = rng.uniform(5000, 7500)
            elif severity_choice == "moderate":
                degradation = rng.uniform(0.27, 0.50)  # 27-50% (gap de 2%)
                health_index = rng.uniform(50, 73)     # gap de 2%
                rul_minutes = rng.uniform(2500, 4500)
            elif severity_choice == "severe":
                degradation = rng.uniform(0.52, 0.75)  # 52-75% (gap de 2%)
                health_index = rng.uniform(25, 48)     # gap de 2%
                rul_minutes = rng.uniform(500, 2000)
            else:  # failure
                degradation = rng.uniform(0.77, 1.00)  # 77-100% (gap de 2%)
                health_index = rng.uniform(0, 23)      # gap de 2%
                rul_minutes = rng.uniform(0, 300)
            
            # ===== MODE =====
            # Mode é escolhido aleatoriamente (balanceado)
            mode = rng.choice(MODES)
            
            # ===== SENSOR VALUES - DEPENDENTES DE DEGRADAÇÃO E MODE =====
            # Ruído base pequeno
            noise = rng.normal(0, 0.02)
            
            # ===== MODE SIGNATURES - padrões DISTINTOS por modo =====
            # Cada mode tem uma "assinatura" única nos sensores
            
            if mode == "normal_operation":
                # Operação normal: valores base, baixa vibração
                vib_mult = 1.0
                vib_x_ratio = 0.35  # distribuição uniforme
                vib_y_ratio = 0.35
                vib_z_ratio = 0.30
                curr_offset = 0.0
                press_offset = 0.0
                flow_offset = 0.0
                temp_offset = 0.0
                
            elif mode == "bearing_wear":
                # Bearing wear: alta vibração em X, temperatura elevada
                vib_mult = 1.5
                vib_x_ratio = 0.60  # predominantemente X
                vib_y_ratio = 0.25
                vib_z_ratio = 0.15
                curr_offset = 0.5
                press_offset = -0.1
                flow_offset = -3.0
                temp_offset = 8.0  # temperatura mais alta
                
            elif mode == "cavitation":
                # Cavitation: pressão baixa, flow irregular, vibração Y
                vib_mult = 1.3
                vib_x_ratio = 0.25
                vib_y_ratio = 0.55  # predominantemente Y
                vib_z_ratio = 0.20
                curr_offset = 0.3
                press_offset = -0.4  # pressão muito baixa
                flow_offset = -8.0   # flow reduzido
                temp_offset = 3.0
                
            elif mode == "misalignment":
                # Misalignment: vibração alta em X e Y, corrente alta
                vib_mult = 1.4
                vib_x_ratio = 0.45
                vib_y_ratio = 0.45  # X e Y similares
                vib_z_ratio = 0.10
                curr_offset = 1.0  # corrente elevada
                press_offset = -0.2
                flow_offset = -5.0
                temp_offset = 5.0
                
            else:  # imbalance
                # Imbalance: vibração alta em Z, flow inconsistente
                vib_mult = 1.35
                vib_x_ratio = 0.20
                vib_y_ratio = 0.20
                vib_z_ratio = 0.60  # predominantemente Z
                curr_offset = 0.4
                press_offset = -0.15
                flow_offset = -4.0
                temp_offset = 4.0
            
            # Vibração base (função da degradação)
            base_vib = 0.3 + 2.0 * degradation  # 0.3 -> 2.3
            overall_vibration = base_vib * vib_mult * (1 + noise)
            
            # Componentes X, Y, Z com assinaturas distintas
            vibration_x = overall_vibration * vib_x_ratio * (1 + rng.normal(0, 0.03))
            vibration_y = overall_vibration * vib_y_ratio * (1 + rng.normal(0, 0.03))
            vibration_z = overall_vibration * vib_z_ratio * (1 + rng.normal(0, 0.03))
            
            # Motor current (função de degradação + mode)
            motor_current = 15.0 + 5.0 * degradation + curr_offset + rng.normal(0, 0.1)
            
            # Pressure (função de degradação + mode)
            pressure = 2.5 - 1.0 * degradation + press_offset + rng.normal(0, 0.02)
            
            # Flow (função de degradação + mode)
            flow = 100.0 - 30.0 * degradation + flow_offset + rng.normal(0, 0.5)
            
            # Temperature (função de degradação + mode)
            temperature = 45.0 + 20.0 * degradation + temp_offset + rng.normal(0, 0.2)
            
            row = {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "mode": mode,
                "severity": severity_choice,
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
    
    # Sort by asset_id and timestamp (preserva sequência temporal para CNN windowing)
    df = df.sort_values(['asset_id', 'timestamp']).reset_index(drop=True)
    
    return df


def main():
    print("=" * 70)
    print("GENERATING IMPROVED DATASET v2")
    print("=" * 70)
    
    df = generate_dataset(n_assets=8, samples_per_asset=10000)
    
    # Salvar
    out_path = Path("logs/sensors_log_v2.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n[OK] Dataset gerado: {out_path}")
    print(f"[INFO] Total samples: {len(df)}")
    
    print("\n[INFO] Severity distribution:")
    print(df["severity"].value_counts(normalize=True).sort_index())
    
    print("\n[INFO] Mode distribution:")
    print(df["mode"].value_counts(normalize=True).sort_index())
    
    # Verificar separabilidade
    print("\n[VERIFY] Vibration by severity (should NOT overlap):")
    for sev in ["normal", "early", "moderate", "severe", "failure"]:
        d = df[df["severity"] == sev]["overall_vibration"]
        print(f"  {sev:10s}: min={d.min():.3f}, max={d.max():.3f}, mean={d.mean():.3f}")
    
    print("\n[VERIFY] Health by severity:")
    for sev in ["normal", "early", "moderate", "severe", "failure"]:
        d = df[df["severity"] == sev]["health_index"]
        print(f"  {sev:10s}: min={d.min():.1f}, max={d.max():.1f}")


if __name__ == "__main__":
    main()
