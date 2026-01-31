#!/usr/bin/env python3
"""
generate_dataset_v3_pump.py
===========================
Dataset Generator v3 - Industrial Pump Dataset com Schema Completo

Este gerador cria um dataset realista para bombas industriais seguindo
as melhores práticas de manutenção preditiva:

SINAIS INCLUÍDOS:
- Processo/Hidráulica: suction_pressure, discharge_pressure, flow, valve_position, pump_speed_rpm
- Elétrica: motor_current, voltage_rms, power_kw, power_factor, motor_temperature
- Fluído: fluid_temperature, density
- Contexto: run_state, operating_mode, ambient_temperature
- Vibração: overall_vibration, vibration_x, vibration_y, vibration_z (+ raw se disponível)

FEATURES DERIVADAS (calculadas automaticamente):
- delta_p = discharge_pressure - suction_pressure
- head = delta_p / (rho * g)
- hydraulic_power = delta_p * flow
- efficiency_est = hydraulic_power / power_kw
- specific_energy = power_kw / flow

LABELS (baseados em eventos de manutenção):
- severity: normal, early, moderate, severe, failure
- mode: normal_operation, bearing_wear, cavitation, misalignment, imbalance, seal_leak
- rul_minutes: tempo até próxima falha (baseado em eventos)
- health_index: 0-100% saúde do equipamento

UNIDADES (todas convertidas para SI):
- Pressão: Pa
- Caudal: m³/s
- Potência: kW
- Temperatura: °C
- Densidade: kg/m³
- Velocidade: rpm
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

SEED = 42
np.random.seed(SEED)

# ============== CONFIGURAÇÃO DO SCHEMA ==============

# Unidades SI para todas as colunas
UNITS = {
    # Processo/Hidráulica
    "suction_pressure": "Pa",
    "discharge_pressure": "Pa",
    "delta_p": "Pa",
    "flow": "m3/s",
    "flow_setpoint": "m3/s",
    "pressure_setpoint": "Pa",
    "valve_position": "%",
    "pump_speed_rpm": "rpm",
    "head": "m",
    
    # Elétrica
    "motor_current": "A",
    "voltage_rms": "V",
    "power_kw": "kW",
    "power_factor": "dimensionless",
    "vfd_frequency_hz": "Hz",
    "motor_temperature": "°C",
    
    # Fluído
    "fluid_temperature": "°C",
    "density": "kg/m3",
    "viscosity": "Pa.s",
    
    # Contexto
    "run_state": "bool",
    "operating_mode": "categorical",
    "ambient_temperature": "°C",
    
    # Vibração
    "overall_vibration": "mm/s",
    "vibration_x": "mm/s",
    "vibration_y": "mm/s",
    "vibration_z": "mm/s",
    "vibration_envelope_rms": "g",
    "vibration_kurtosis": "dimensionless",
    
    # Derivadas
    "hydraulic_power": "kW",
    "efficiency_est": "%",
    "specific_energy": "kJ/m3",
}

# Modos de falha com assinaturas características
FAILURE_MODES = {
    "normal_operation": {
        "description": "Operação normal sem anomalias",
        "vibration_signature": {"x": 0.33, "y": 0.33, "z": 0.34},
        "vibration_mult": 1.0,
        "current_offset": 0.0,
        "efficiency_loss": 0.0,
        "temperature_offset": 0.0,
        "pressure_loss": 0.0,
        "flow_loss": 0.0,
    },
    "bearing_wear": {
        "description": "Desgaste de rolamentos - vibração alta em X, temperatura elevada",
        "vibration_signature": {"x": 0.60, "y": 0.25, "z": 0.15},
        "vibration_mult": 1.8,
        "current_offset": 0.5,
        "efficiency_loss": 0.08,
        "temperature_offset": 12.0,
        "pressure_loss": 0.02,
        "flow_loss": 0.05,
    },
    "cavitation": {
        "description": "Cavitação - vibração Y alta, pressão baixa, flow irregular",
        "vibration_signature": {"x": 0.25, "y": 0.55, "z": 0.20},
        "vibration_mult": 1.5,
        "current_offset": 0.3,
        "efficiency_loss": 0.15,
        "temperature_offset": 5.0,
        "pressure_loss": 0.10,
        "flow_loss": 0.12,
    },
    "misalignment": {
        "description": "Desalinhamento - vibração X+Y alta, corrente elevada",
        "vibration_signature": {"x": 0.45, "y": 0.45, "z": 0.10},
        "vibration_mult": 1.6,
        "current_offset": 1.2,
        "efficiency_loss": 0.10,
        "temperature_offset": 8.0,
        "pressure_loss": 0.05,
        "flow_loss": 0.08,
    },
    "imbalance": {
        "description": "Desequilíbrio - vibração Z dominante",
        "vibration_signature": {"x": 0.20, "y": 0.20, "z": 0.60},
        "vibration_mult": 1.4,
        "current_offset": 0.4,
        "efficiency_loss": 0.05,
        "temperature_offset": 4.0,
        "pressure_loss": 0.03,
        "flow_loss": 0.04,
    },
    "seal_leak": {
        "description": "Vazamento de selo - perda de pressão e flow",
        "vibration_signature": {"x": 0.35, "y": 0.35, "z": 0.30},
        "vibration_mult": 1.2,
        "current_offset": -0.2,
        "efficiency_loss": 0.20,
        "temperature_offset": 3.0,
        "pressure_loss": 0.15,
        "flow_loss": 0.18,
    },
}

# Constantes físicas
RHO_WATER = 998.0  # kg/m³ a 20°C
G = 9.81  # m/s²

# Parâmetros nominais da bomba
PUMP_NOMINAL = {
    "suction_pressure_pa": 101325.0,  # 1 atm
    "discharge_pressure_pa": 500000.0,  # 5 bar
    "flow_m3s": 0.03,  # 30 L/s = 108 m³/h
    "power_kw": 15.0,
    "speed_rpm": 1450.0,
    "current_a": 25.0,
    "voltage_v": 400.0,
    "efficiency": 0.75,
    "temperature_c": 45.0,
}


def generate_maintenance_events(
    n_assets: int,
    start_date: datetime,
    end_date: datetime,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Gera eventos de manutenção/falha realistas.
    Estes eventos são a "ground truth" para labels de RUL e mode.
    """
    rng = np.random.default_rng(seed)
    events = []
    
    modes = list(FAILURE_MODES.keys())
    modes.remove("normal_operation")  # Falhas apenas
    
    for asset_idx in range(n_assets):
        asset_id = f"pump_{asset_idx+1:03d}"
        
        # Cada bomba tem 2-5 eventos no período
        n_events = rng.integers(2, 6)
        
        # Gerar timestamps de eventos espaçados
        total_days = (end_date - start_date).days
        event_days = sorted(rng.choice(range(30, total_days - 5), size=n_events, replace=False))
        
        for i, day in enumerate(event_days):
            event_time = start_date + timedelta(days=int(day), hours=int(rng.integers(6, 18)))
            
            # Tipo de evento
            event_type = rng.choice(modes)
            
            events.append({
                "asset_id": asset_id,
                "event_time": event_time,
                "event_type": event_type,
                "confirmed": 1,
                "severity": rng.choice(["moderate", "severe", "failure"], p=[0.3, 0.4, 0.3]),
                "downtime_hours": float(rng.uniform(2, 48)),
                "notes": f"Evento {i+1} de {n_events} para {asset_id}",
            })
    
    return pd.DataFrame(events)


def calculate_rul_from_events(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    max_rul_days: float = 7.0
) -> pd.DataFrame:
    """
    Calcula RUL real baseado em eventos de manutenção.
    
    RUL = tempo até próximo evento de falha (minutos)
    Clipped a max_rul_days para não saturar.
    """
    df = df.copy()
    max_rul_minutes = max_rul_days * 24 * 60
    df["rul_minutes"] = max_rul_minutes  # Default max
    
    for asset_id in df["asset_id"].unique():
        asset_events = events_df[events_df["asset_id"] == asset_id].sort_values("event_time")
        asset_mask = df["asset_id"] == asset_id
        asset_df = df[asset_mask].copy()
        
        for _, event in asset_events.iterrows():
            event_time = pd.to_datetime(event["event_time"])
            
            # Timestamps antes do evento
            before_mask = asset_df["timestamp"] < event_time
            if before_mask.any():
                time_to_event = (event_time - asset_df.loc[before_mask, "timestamp"]).dt.total_seconds() / 60
                
                # Só atualiza se for menor que o RUL atual
                current_rul = df.loc[asset_mask & before_mask, "rul_minutes"].values
                new_rul = np.minimum(current_rul, time_to_event.values)
                df.loc[asset_mask & before_mask, "rul_minutes"] = new_rul
    
    # Clip a max
    df["rul_minutes"] = df["rul_minutes"].clip(upper=max_rul_minutes)
    
    return df


def rul_to_severity(rul_minutes: float) -> str:
    """
    Converte RUL em severity baseado em thresholds industriais.
    
    Thresholds:
    - normal: RUL > 7 dias
    - early: 7 dias >= RUL > 2 dias
    - moderate: 2 dias >= RUL > 6h
    - severe: 6h >= RUL > 1h
    - failure: RUL <= 1h
    """
    if rul_minutes > 7 * 24 * 60:  # > 7 dias
        return "normal"
    elif rul_minutes > 2 * 24 * 60:  # > 2 dias
        return "early"
    elif rul_minutes > 6 * 60:  # > 6 horas
        return "moderate"
    elif rul_minutes > 60:  # > 1 hora
        return "severe"
    else:
        return "failure"


def rul_to_health(rul_minutes: float, rul_max_minutes: float = 30 * 24 * 60) -> float:
    """
    Converte RUL em health_index (0-100).
    health = 100 * clip(rul / rul_max, 0, 1)
    """
    return 100.0 * min(max(rul_minutes / rul_max_minutes, 0.0), 1.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features derivadas obrigatórias para bombas.
    
    CRÍTICO: Todas as colunas devem estar em unidades SI!
    """
    df = df.copy()
    
    # Delta P (Pa)
    df["delta_p"] = df["discharge_pressure"] - df["suction_pressure"]
    
    # Head (m) = delta_p / (rho * g)
    df["head"] = df["delta_p"] / (df["density"] * G)
    
    # Potência hidráulica (kW) = delta_p * flow / 1000
    df["hydraulic_power"] = (df["delta_p"] * df["flow"]) / 1000.0
    
    # Eficiência estimada (%)
    df["efficiency_est"] = (df["hydraulic_power"] / df["power_kw"].clip(lower=0.1)) * 100.0
    df["efficiency_est"] = df["efficiency_est"].clip(0, 100)
    
    # Energia específica (kJ/m³)
    df["specific_energy"] = (df["power_kw"] / df["flow"].clip(lower=1e-6)) * 1000.0
    
    # Ratios úteis
    df["flow_per_rpm"] = df["flow"] / df["pump_speed_rpm"].clip(lower=1)
    df["power_per_rpm"] = df["power_kw"] / df["pump_speed_rpm"].clip(lower=1)
    df["current_per_power"] = df["motor_current"] / df["power_kw"].clip(lower=0.1)
    
    return df


def generate_sensor_values(
    mode: str,
    degradation: float,
    rng: np.random.Generator,
    run_state: int = 1
) -> Dict[str, float]:
    """
    Gera valores de sensores para um modo de falha e nível de degradação.
    
    Args:
        mode: Modo de falha (normal_operation, bearing_wear, etc.)
        degradation: Nível de degradação 0-1 (0=novo, 1=falha total)
        rng: Gerador de números aleatórios
        run_state: 1=a correr, 0=parado
    
    Returns:
        Dicionário com todos os valores de sensores
    """
    if run_state == 0:
        # Bomba parada - valores mínimos
        return {
            "suction_pressure": PUMP_NOMINAL["suction_pressure_pa"] + rng.normal(0, 1000),
            "discharge_pressure": PUMP_NOMINAL["suction_pressure_pa"] + rng.normal(0, 1000),
            "flow": 0.0,
            "flow_setpoint": 0.0,
            "pressure_setpoint": 0.0,
            "valve_position": 0.0,
            "pump_speed_rpm": 0.0,
            "motor_current": rng.uniform(0.5, 1.5),
            "voltage_rms": PUMP_NOMINAL["voltage_v"] + rng.normal(0, 5),
            "power_kw": rng.uniform(0.1, 0.3),
            "power_factor": 0.1,
            "vfd_frequency_hz": 0.0,
            "motor_temperature": 25.0 + rng.normal(0, 2),
            "fluid_temperature": 20.0 + rng.normal(0, 2),
            "density": RHO_WATER,
            "ambient_temperature": 22.0 + rng.normal(0, 3),
            "overall_vibration": rng.uniform(0.01, 0.05),
            "vibration_x": rng.uniform(0.01, 0.02),
            "vibration_y": rng.uniform(0.01, 0.02),
            "vibration_z": rng.uniform(0.01, 0.02),
        }
    
    # Obter assinatura do modo
    sig = FAILURE_MODES[mode]
    
    # Ruído base
    noise = rng.normal(0, 0.02)
    
    # === PROCESSO/HIDRÁULICA ===
    
    # Pressão de sucção (ligeira variação)
    suction_p = PUMP_NOMINAL["suction_pressure_pa"] * (1 + rng.normal(0, 0.02))
    
    # Pressão de descarga (afetada por degradação e modo)
    pressure_loss_factor = 1 - sig["pressure_loss"] * degradation
    discharge_p = PUMP_NOMINAL["discharge_pressure_pa"] * pressure_loss_factor * (1 + rng.normal(0, 0.03))
    
    # Caudal (afetado por degradação e modo)
    flow_loss_factor = 1 - sig["flow_loss"] * degradation
    flow = PUMP_NOMINAL["flow_m3s"] * flow_loss_factor * (1 + rng.normal(0, 0.02))
    
    # Setpoints (geralmente constantes com pequena variação)
    flow_setpoint = PUMP_NOMINAL["flow_m3s"] * (1 + rng.normal(0, 0.01))
    pressure_setpoint = PUMP_NOMINAL["discharge_pressure_pa"] * (1 + rng.normal(0, 0.01))
    
    # Válvula (ajusta para compensar perdas)
    valve_position = 80.0 + 15.0 * degradation + rng.normal(0, 2)
    valve_position = np.clip(valve_position, 0, 100)
    
    # Velocidade (pode variar com VFD)
    speed_variation = rng.uniform(-50, 50)
    pump_speed = PUMP_NOMINAL["speed_rpm"] + speed_variation
    
    # === ELÉTRICA ===
    
    # Corrente (aumenta com degradação em alguns modos)
    current_base = PUMP_NOMINAL["current_a"]
    current = current_base * (1 + 0.1 * degradation) + sig["current_offset"] * degradation
    current += rng.normal(0, 0.3)
    
    # Tensão (relativamente estável)
    voltage = PUMP_NOMINAL["voltage_v"] + rng.normal(0, 5)
    
    # Potência (aumenta com degradação devido a perdas)
    efficiency_loss = sig["efficiency_loss"] * degradation
    power = PUMP_NOMINAL["power_kw"] * (1 + efficiency_loss * 0.5) + rng.normal(0, 0.2)
    
    # Fator de potência (diminui com problemas)
    pf = 0.85 - 0.05 * degradation + rng.normal(0, 0.02)
    pf = np.clip(pf, 0.5, 0.98)
    
    # Frequência VFD
    vfd_freq = pump_speed / PUMP_NOMINAL["speed_rpm"] * 50.0
    
    # Temperatura motor
    temp_motor = PUMP_NOMINAL["temperature_c"] + sig["temperature_offset"] * degradation
    temp_motor += rng.normal(0, 1)
    
    # === FLUÍDO ===
    
    # Temperatura do fluido
    temp_fluid = 20.0 + 5.0 * degradation + rng.normal(0, 1)
    
    # Densidade (varia com temperatura)
    density = RHO_WATER - 0.5 * (temp_fluid - 20.0)
    
    # === CONTEXTO ===
    
    ambient_temp = 22.0 + rng.normal(0, 3)
    
    # === VIBRAÇÃO ===
    
    vib_sig = sig["vibration_signature"]
    vib_mult = sig["vibration_mult"]
    
    # Vibração base aumenta com degradação
    base_vib = 0.5 + 3.0 * degradation  # mm/s
    overall_vib = base_vib * vib_mult * (1 + noise)
    
    vib_x = overall_vib * vib_sig["x"] * (1 + rng.normal(0, 0.05))
    vib_y = overall_vib * vib_sig["y"] * (1 + rng.normal(0, 0.05))
    vib_z = overall_vib * vib_sig["z"] * (1 + rng.normal(0, 0.05))
    
    return {
        # Processo/Hidráulica
        "suction_pressure": suction_p,
        "discharge_pressure": discharge_p,
        "flow": flow,
        "flow_setpoint": flow_setpoint,
        "pressure_setpoint": pressure_setpoint,
        "valve_position": valve_position,
        "pump_speed_rpm": pump_speed,
        
        # Elétrica
        "motor_current": current,
        "voltage_rms": voltage,
        "power_kw": power,
        "power_factor": pf,
        "vfd_frequency_hz": vfd_freq,
        "motor_temperature": temp_motor,
        
        # Fluído
        "fluid_temperature": temp_fluid,
        "density": density,
        
        # Contexto
        "ambient_temperature": ambient_temp,
        
        # Vibração
        "overall_vibration": overall_vib,
        "vibration_x": vib_x,
        "vibration_y": vib_y,
        "vibration_z": vib_z,
    }


def generate_pump_timeseries(
    n_assets: int = 8,
    samples_per_asset: int = 10000,
    sample_rate_hz: float = 1.0,
    start_date: datetime = datetime(2025, 1, 1),
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera dataset completo de séries temporais para bombas industriais.
    
    Usa abordagem direta: primeiro escolhe severity/mode/rul,
    depois gera sensores com padrões consistentes.
    
    Returns:
        (timeseries_df, events_df): DataFrames com dados e eventos
    """
    rng = np.random.default_rng(seed)
    
    # Lista de modos (inclui normal_operation)
    modes = list(FAILURE_MODES.keys())
    n_modes = len(modes)
    
    # Severities com probabilidades (mais healthy que failure)
    severities = ["normal", "early", "moderate", "severe", "failure"]
    sev_probs = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    # Ranges de degradação por severity (não sobrepostos)
    SEVERITY_RANGES = {
        "normal":   {"deg": (0.00, 0.10), "health": (90, 100), "rul": (8000, 10080)},
        "early":    {"deg": (0.12, 0.25), "health": (75, 88),  "rul": (5000, 7500)},
        "moderate": {"deg": (0.27, 0.50), "health": (50, 73),  "rul": (2500, 4500)},
        "severe":   {"deg": (0.52, 0.75), "health": (25, 48),  "rul": (500, 2000)},
        "failure":  {"deg": (0.77, 1.00), "health": (0, 23),   "rul": (0, 300)},
    }
    
    # Probabilidades de modo por severity
    # normal_operation é o primeiro modo na lista
    normal_idx = modes.index("normal_operation")
    
    def get_mode_probs(severity):
        """Retorna probabilidades de modo baseado em severity."""
        probs = [1.0 / n_modes] * n_modes  # Base uniforme
        
        if severity == "normal":
            # Normal operation mais comum em equipamento saudável
            probs = [1.0 / (n_modes * 2)] * n_modes
            probs[normal_idx] = 0.7
            total = sum(probs)
            probs = [p / total for p in probs]
        elif severity == "failure":
            # Failure modes mais comuns em equipamento com falha
            probs = [0.15 / (n_modes - 1)] * n_modes
            probs[normal_idx] = 0.05
            total = sum(probs)
            probs = [p / total for p in probs]
        
        return probs
    
    rows = []
    
    for asset_idx in range(n_assets):
        asset_id = f"pump_{asset_idx+1:03d}"
        print(f"[INFO] Gerando asset {asset_id}...")
        
        for i in range(samples_per_asset):
            # Timestamp
            timestamp = start_date + timedelta(days=asset_idx * 30, minutes=i * 5)
            
            # Escolher severity
            severity = rng.choice(severities, p=sev_probs)
            ranges = SEVERITY_RANGES[severity]
            
            # Degradação, health, RUL baseados em ranges
            degradation = rng.uniform(*ranges["deg"])
            health_index = rng.uniform(*ranges["health"])
            rul_minutes = rng.uniform(*ranges["rul"])
            
            # Escolher mode baseado em severity
            mode_probs = get_mode_probs(severity)
            mode = rng.choice(modes, p=mode_probs)
            
            # Run state (mais paradas em failure)
            if severity == "failure":
                run_state = 1 if rng.random() > 0.2 else 0
            else:
                run_state = 1 if rng.random() > 0.03 else 0
            
            # Gerar valores de sensores
            sensors = generate_sensor_values(mode, degradation, rng, run_state)
            
            # Operating mode
            if run_state == 0:
                operating_mode = "stopped"
            elif rng.random() < 0.9:
                operating_mode = "auto"
            else:
                operating_mode = "manual"
            
            row = {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "run_state": run_state,
                "operating_mode": operating_mode,
                "mode": mode,
                "severity": severity,
                "rul_minutes": rul_minutes,
                "health_index": health_index,
                **sensors
            }
            rows.append(row)
    
    # Criar DataFrame
    df = pd.DataFrame(rows)
    
    # Ordenar por asset e timestamp
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    
    # Adicionar features derivadas
    print("[INFO] Calculando features derivadas...")
    df = add_derived_features(df)
    
    # Criar events_df vazio (não usado nesta versão)
    events_df = pd.DataFrame(columns=["asset_id", "event_time", "event_type", "severity"])
    
    return df, events_df


def save_dataset(df: pd.DataFrame, events_df: pd.DataFrame, base_path: Path):
    """Salva dataset em múltiplos formatos."""
    
    # Criar diretórios
    raw_dir = base_path / "raw"
    processed_dir = base_path / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Checar se parquet está disponível
    try:
        import pyarrow
        HAS_PARQUET = True
    except ImportError:
        HAS_PARQUET = False
        print("[WARN] pyarrow not available, skipping parquet export")
    
    # Salvar raw events
    if HAS_PARQUET:
        print(f"[SAVE] Raw events: {raw_dir / 'maintenance_events.parquet'}")
        events_df.to_parquet(raw_dir / "maintenance_events.parquet", index=False)
    else:
        print(f"[SAVE] Raw events: {raw_dir / 'maintenance_events.csv'}")
        events_df.to_csv(raw_dir / "maintenance_events.csv", index=False)
    
    # Salvar processed (CSV sempre + parquet se disponível)
    csv_path = processed_dir / "pump_timeseries_v3.csv"
    
    print(f"[SAVE] Processed CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    if HAS_PARQUET:
        parquet_path = processed_dir / "pump_timeseries_v3.parquet"
        print(f"[SAVE] Processed Parquet: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
    
    # Também salvar na pasta datasets/ para compatibilidade
    compat_path = base_path.parent / "datasets" / "pump_timeseries_v3.csv"
    df.to_csv(compat_path, index=False)
    print(f"[SAVE] Compat CSV: {compat_path}")
    
    # Salvar metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_assets": df["asset_id"].nunique(),
        "n_events": len(events_df),
        "columns": list(df.columns),
        "units": UNITS,
        "severity_distribution": df["severity"].value_counts().to_dict(),
        "mode_distribution": df["mode"].value_counts().to_dict(),
        "rul_stats": {
            "min": float(df["rul_minutes"].min()),
            "max": float(df["rul_minutes"].max()),
            "mean": float(df["rul_minutes"].mean()),
        },
        "health_stats": {
            "min": float(df["health_index"].min()),
            "max": float(df["health_index"].max()),
            "mean": float(df["health_index"].mean()),
        },
        "failure_modes": list(FAILURE_MODES.keys()),
        "seed": SEED,
    }
    
    metadata_path = processed_dir / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[SAVE] Metadata: {metadata_path}")


def main():
    print("=" * 70)
    print("GENERATING INDUSTRIAL PUMP DATASET v3")
    print("=" * 70)
    
    # Parâmetros
    n_assets = 8
    samples_per_asset = 10000
    
    # Gerar
    df, events_df = generate_pump_timeseries(
        n_assets=n_assets,
        samples_per_asset=samples_per_asset,
        sample_rate_hz=1.0,
        seed=SEED
    )
    
    print(f"\n[INFO] Total samples: {len(df)}")
    print(f"[INFO] Assets: {df['asset_id'].nunique()}")
    print(f"[INFO] Events: {len(events_df)}")
    
    # Estatísticas
    print("\n[STATS] Severity distribution:")
    print(df["severity"].value_counts(normalize=True).sort_index())
    
    print("\n[STATS] Mode distribution:")
    print(df["mode"].value_counts(normalize=True).sort_index())
    
    print("\n[STATS] RUL range:")
    print(f"  Min: {df['rul_minutes'].min():.0f} min")
    print(f"  Max: {df['rul_minutes'].max():.0f} min")
    print(f"  Mean: {df['rul_minutes'].mean():.0f} min")
    
    print("\n[STATS] Health range:")
    print(f"  Min: {df['health_index'].min():.1f}%")
    print(f"  Max: {df['health_index'].max():.1f}%")
    
    # Verificar features derivadas
    print("\n[VERIFY] Derived features sample:")
    sample = df[df["run_state"] == 1].iloc[0]
    print(f"  delta_p: {sample['delta_p']:.0f} Pa")
    print(f"  head: {sample['head']:.2f} m")
    print(f"  hydraulic_power: {sample['hydraulic_power']:.2f} kW")
    print(f"  efficiency_est: {sample['efficiency_est']:.1f}%")
    
    # Salvar
    base_path = Path(__file__).parent.parent / "data"
    save_dataset(df, events_df, base_path)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Dataset v3 generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
