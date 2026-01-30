#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de dataset sintético REALISTA para manutenção preditiva de bombas.
Simula:
- 10 ativos (bombas) com histórico de 6 meses cada
- Regimes operacionais realistas (normal, alta carga, baixa carga, start, stop)
- Degradação progressiva (drift) por ativo
- Eventos de falha CMMS (ground truth consistente)
- Ruído 1-5% nos sensores
- Transientes (start/stop com picos)
- Classes raras (severe=7%, failure=2%)
- Sensores: vibração (x,y,z,overall), corrente motor, pressão, flow, temperatura
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
np.random.seed(SEED)

N_ASSETS = 10
DAYS_PER_ASSET = 180  # 6 meses
SAMPLES_PER_DAY = 288  # 1 sample/5min
TOTAL_SAMPLES = N_ASSETS * DAYS_PER_ASSET * SAMPLES_PER_DAY

# Sensores (valores base)
SENSOR_NAMES = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "motor_current", "pressure", "flow", "temperature",
    "bearing_temp_DE", "bearing_temp_NDE", "ultrasonic_noise", "casing_temp"
]

# Regimes operacionais
REGIMES = {
    "normal": {"prob": 0.70, "load": 1.0, "rpm": 1.0},
    "high_load": {"prob": 0.12, "load": 1.3, "rpm": 1.1},
    "low_load": {"prob": 0.10, "load": 0.6, "rpm": 0.8},
    "start": {"prob": 0.04, "load": 1.5, "rpm": 1.4},  # transiente
    "stop": {"prob": 0.04, "load": 0.3, "rpm": 0.5},   # transiente
}

# Modos de falha (mode) - distribuição industrial realista
MODES = {
    "normal_operation": 0.70,
    "bearing_wear_moderate": 0.08,
    "bearing_wear_severe": 0.03,
    "cavitation_moderate": 0.05,
    "cavitation_severe": 0.02,
    "impeller_wear_moderate": 0.04,
    "impeller_wear_severe": 0.015,
    "misalignment_moderate": 0.03,
    "looseness_moderate": 0.025,
    "electrical_fault_moderate": 0.02,
    "fluid_contaminated_moderate": 0.015,
}

# Severity - alinhado com mode
SEVERITY_MAP = {
    "normal_operation": "normal",
    "bearing_wear_moderate": "moderate",
    "bearing_wear_severe": "severe",
    "cavitation_moderate": "early",
    "cavitation_severe": "severe",
    "impeller_wear_moderate": "moderate",
    "impeller_wear_severe": "failure",
    "misalignment_moderate": "early",
    "looseness_moderate": "early",
    "electrical_fault_moderate": "moderate",
    "fluid_contaminated_moderate": "moderate",
}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def generate_regime_sequence(n_samples: int, seed: int) -> list:
    """Gera sequência de regimes com transições realistas (não aleatório puro)."""
    rng = np.random.default_rng(seed)
    regimes = []
    current_regime = "normal"
    regime_duration = 0
    min_duration = {"normal": 50, "high_load": 20, "low_load": 20, "start": 3, "stop": 3}
    
    for _ in range(n_samples):
        if regime_duration < min_duration[current_regime]:
            regimes.append(current_regime)
            regime_duration += 1
        else:
            # Trocar regime com probabilidade
            if rng.random() < 0.05:  # 5% chance de trocar
                current_regime = rng.choice(list(REGIMES.keys()), p=[r["prob"] for r in REGIMES.values()])
                regime_duration = 0
            regimes.append(current_regime)
            regime_duration += 1
    
    return regimes


def generate_degradation_curve(n_samples: int, failure_at: int, seed: int) -> np.ndarray:
    """
    Gera curva de degradação realista (exponencial nos últimos 20% da vida).
    failure_at: sample index onde ocorre falha (ou -1 se não falhar).
    """
    rng = np.random.default_rng(seed)
    if failure_at < 0:
        # Sem falha: degradação lenta linear
        return np.linspace(0, 0.15, n_samples) + rng.normal(0, 0.01, n_samples)
    else:
        # Com falha: exponencial nos últimos 20%
        health = np.ones(n_samples)
        for i in range(n_samples):
            progress = i / failure_at if failure_at > 0 else 0
            if progress < 0.8:
                health[i] = 1.0 - 0.1 * progress  # degradação lenta
            else:
                # degradação exponencial
                health[i] = 0.92 * np.exp(-5 * (progress - 0.8))
        return 1.0 - health + rng.normal(0, 0.01, n_samples)


def assign_mode_sequence(degradation: np.ndarray, seed: int) -> list:
    """Atribui mode baseado na degradação (ground truth consistente)."""
    rng = np.random.default_rng(seed)
    modes = []
    for d in degradation:
        if d < 0.1:
            mode = "normal_operation"
        elif d < 0.2:
            mode = rng.choice(["normal_operation", "cavitation_moderate", "misalignment_moderate", "looseness_moderate"], p=[0.7, 0.15, 0.1, 0.05])
        elif d < 0.4:
            mode = rng.choice(["bearing_wear_moderate", "cavitation_moderate", "impeller_wear_moderate", "electrical_fault_moderate", "fluid_contaminated_moderate"], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        elif d < 0.7:
            mode = rng.choice(["bearing_wear_severe", "cavitation_severe", "impeller_wear_moderate"], p=[0.4, 0.35, 0.25])
        else:
            mode = rng.choice(["bearing_wear_severe", "cavitation_severe", "impeller_wear_severe"], p=[0.4, 0.3, 0.3])
        modes.append(mode)
    return modes


def generate_sensor_values(
    regime: str,
    mode: str,
    degradation: float,
    base_noise: float,
    sensor_idx: int,
    rng: np.random.Generator
) -> float:
    """
    Gera valor de sensor baseado em regime, mode e degradação.
    Inclui ruído 1-5% e comportamento físico realista.
    """
    # Base do sensor (valores industriais típicos)
    base_values = {
        "overall_vibration": 2.5,  # mm/s
        "vibration_x": 1.5,
        "vibration_y": 1.3,
        "vibration_z": 1.1,
        "motor_current": 25.0,  # A
        "pressure": 5.5,  # bar
        "flow": 120.0,  # m3/h
        "temperature": 45.0,  # °C
        "bearing_temp_DE": 55.0,
        "bearing_temp_NDE": 52.0,
        "ultrasonic_noise": 30.0,  # dB
        "casing_temp": 48.0,
    }
    
    sensor_name = SENSOR_NAMES[sensor_idx]
    base = base_values[sensor_name]
    
    # Ajuste por regime
    regime_data = REGIMES[regime]
    load_factor = regime_data["load"]
    rpm_factor = regime_data["rpm"]
    
    # Vibração aumenta com load e degradação
    if "vibration" in sensor_name:
        value = base * (0.5 + 0.5 * load_factor) * (1.0 + degradation * 2.0)
        # Modos de falha específicos
        if "bearing_wear" in mode:
            value *= (1.5 + degradation)
        elif "cavitation" in mode:
            value *= (1.2 + degradation * 0.5)
        elif "misalignment" in mode:
            value *= (1.3 + degradation * 0.7)
    
    # Corrente aumenta com load
    elif "current" in sensor_name:
        value = base * load_factor * (1.0 + degradation * 0.3)
        if "electrical_fault" in mode:
            value *= (1.4 + degradation)
    
    # Pressão e flow dependem do regime
    elif sensor_name in ["pressure", "flow"]:
        value = base * load_factor * (1.0 - degradation * 0.2)
        if "cavitation" in mode:
            value *= (0.8 - degradation * 0.3)
        elif "impeller_wear" in mode:
            value *= (0.85 - degradation * 0.2)
    
    # Temperatura aumenta com degradação e load
    elif "temp" in sensor_name:
        value = base + 10 * load_factor + 15 * degradation
        if "bearing_wear" in mode:
            value += 10 * (1 + degradation)
    
    # Ultrasonic noise (cavitation)
    elif "ultrasonic" in sensor_name:
        value = base * (1.0 + degradation * 0.5)
        if "cavitation" in mode:
            value *= (2.0 + degradation)
    
    else:
        value = base * (1.0 + degradation * 0.5)
    
    # Ruído 1-5%
    noise = rng.normal(0, base_noise * value)
    return max(0, value + noise)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("[INFO] Gerando dataset sintético REALISTA para manutenção preditiva...")
    
    rng = np.random.default_rng(SEED)
    rows = []
    
    for asset_idx in range(N_ASSETS):
        asset_id = f"PUMP_{asset_idx+1:03d}"
        n_samples = DAYS_PER_ASSET * SAMPLES_PER_DAY
        
        # Decidir se este ativo vai falhar (20% dos ativos têm evento de falha CMMS)
        will_fail = rng.random() < 0.2
        failure_at = int(n_samples * rng.uniform(0.7, 0.95)) if will_fail else -1
        
        print(f"[ASSET] {asset_id}: {'FAILURE' if will_fail else 'NO FAILURE'} (samples={n_samples})")
        
        # Gerar sequências
        regimes = generate_regime_sequence(n_samples, SEED + asset_idx)
        degradation = generate_degradation_curve(n_samples, failure_at, SEED + asset_idx + 1000)
        modes = assign_mode_sequence(degradation, SEED + asset_idx + 2000)
        
        # Timestamp inicial
        start_time = datetime(2025, 1, 1) + timedelta(days=asset_idx * 10)
        
        for i in range(n_samples):
            timestamp = start_time + timedelta(minutes=i * 5)
            regime = regimes[i]
            mode = modes[i]
            severity = SEVERITY_MAP[mode]
            degrade = degradation[i]
            
            # Health e RUL
            health_index = max(0, 100 * (1.0 - degrade))
            if failure_at > 0:
                rul_minutes = max(0, (failure_at - i) * 5)
            else:
                rul_minutes = 10000  # sem falha prevista
            
            # Ruído base (1-5%)
            base_noise = rng.uniform(0.01, 0.05)
            
            # Gerar sensores
            sensor_values = {}
            for s_idx, s_name in enumerate(SENSOR_NAMES):
                sensor_values[s_name] = generate_sensor_values(
                    regime, mode, degrade, base_noise, s_idx, rng
                )
            
            row = {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "mode": mode,
                "severity": severity,
                "health_index": health_index,
                "rul_minutes": rul_minutes,
                "regime": regime,
                **sensor_values
            }
            rows.append(row)
    
    # Criar DataFrame
    df = pd.DataFrame(rows)
    
    # Shuffle para simular coleta de dados não sequencial
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Salvar
    out_path = Path("logs/sensors_log.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"\n[OK] Dataset gerado: {out_path}")
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Assets: {df['asset_id'].nunique()}")
    print(f"[INFO] Modes: {df['mode'].nunique()}")
    print(f"[INFO] Severity distribution:")
    print(df["severity"].value_counts(normalize=True).sort_index())
    print(f"[INFO] Mode distribution (top 10):")
    print(df["mode"].value_counts(normalize=True).head(10))


if __name__ == "__main__":
    main()
