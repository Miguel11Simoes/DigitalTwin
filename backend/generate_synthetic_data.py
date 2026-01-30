"""
Gerador de dados sintéticos com assinaturas espectrais realistas para predictive maintenance.

Cada modo de falha tem frequências características:
- normal_operation: baseline limpo
- bearing_wear: energia em altas frequências + sidebands
- cavitation: ruído broadband + picos irregulares
- imbalance: 1× RPM dominante
- misalignment: 2× RPM forte

Severity controla a intensidade das assinaturas.
Cada asset tem baseline próprio (offsets, ruído).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

# Configurações
N_ASSETS = 8
SAMPLES_PER_ASSET = 10000  # 10k samples por asset = 80k total
SAMPLING_RATE = 1000  # Hz
RPM_NOMINAL = 1800  # rotação nominal
FREQ_BASELINE = RPM_NOMINAL / 60.0  # 30 Hz (1× RPM)

MODES = ["normal_operation", "bearing_wear", "cavitation", "imbalance", "misalignment"]
SEVERITIES = ["normal", "early", "moderate", "severe", "failure"]

# Seeds para reprodutibilidade
np.random.seed(42)


def generate_spectral_signature(mode: str, severity_level: int, n_samples: int, asset_baseline: dict) -> dict:
    """
    Gera assinatura espectral realista para um modo e severidade.
    
    severity_level: 0=normal, 1=early, 2=moderate, 3=severe, 4=failure
    """
    dt = 1.0 / SAMPLING_RATE
    t = np.arange(n_samples) * dt
    
    # Baseline do asset (cada bomba tem características próprias)
    baseline_offset = asset_baseline["offset"]
    baseline_noise = asset_baseline["noise_level"]
    baseline_gain = asset_baseline["gain"]
    
    # Intensidade baseada em severity (0.1 para normal, até 2.0 para failure)
    intensity = 0.1 + severity_level * 0.475  # 0.1, 0.575, 1.05, 1.525, 2.0
    
    # Frequências fundamentais
    f1x = FREQ_BASELINE  # 1× RPM (30 Hz)
    f2x = 2 * f1x  # 2× RPM (60 Hz)
    f3x = 3 * f1x  # 3× RPM (90 Hz)
    
    # Vibração base (sempre presente)
    vibration = baseline_offset + np.sin(2 * np.pi * f1x * t) * baseline_gain
    
    if mode == "normal_operation":
        # Apenas baseline + ruído baixo
        vibration += np.random.randn(n_samples) * baseline_noise * 0.5
        pressure = 2.5 + np.random.randn(n_samples) * 0.05 + baseline_offset * 0.1
        flow = 100 + np.random.randn(n_samples) * 2.0
        temperature = 45 + np.random.randn(n_samples) * 1.0
        motor_current = 15 + np.random.randn(n_samples) * 0.3
        
    elif mode == "bearing_wear":
        # Alta frequência + sidebands + aumenta com severity
        bearing_freq = 120 + severity_level * 30  # 120-240 Hz
        vibration += np.sin(2 * np.pi * bearing_freq * t) * intensity * 0.8
        # Sidebands (modulação)
        vibration += np.sin(2 * np.pi * (bearing_freq + 5) * t) * intensity * 0.3
        vibration += np.sin(2 * np.pi * (bearing_freq - 5) * t) * intensity * 0.3
        # Ruído de alta frequência
        vibration += np.random.randn(n_samples) * baseline_noise * intensity * 2.0
        
        pressure = 2.5 - severity_level * 0.15 + np.random.randn(n_samples) * 0.1
        flow = 100 - severity_level * 8 + np.random.randn(n_samples) * 3.0
        temperature = 45 + severity_level * 5 + np.random.randn(n_samples) * 2.0
        motor_current = 15 + severity_level * 1.5 + np.random.randn(n_samples) * 0.5
        
    elif mode == "cavitation":
        # Ruído broadband + picos irregulares
        broadband = np.random.randn(n_samples) * intensity * 3.0
        # Adicionar picos irregulares (burst noise)
        burst_indices = np.random.choice(n_samples, size=int(n_samples * 0.1 * intensity), replace=False)
        broadband[burst_indices] += np.random.randn(len(burst_indices)) * intensity * 5.0
        vibration += broadband
        
        # Cavitação afeta muito a pressão e flow
        pressure = 2.5 - severity_level * 0.25 + np.random.randn(n_samples) * 0.15 * intensity
        flow = 100 - severity_level * 12 + np.random.randn(n_samples) * 5.0 * intensity
        temperature = 45 + severity_level * 3 + np.random.randn(n_samples) * 1.5
        motor_current = 15 + severity_level * 0.8 + np.random.randn(n_samples) * 0.4
        
    elif mode == "imbalance":
        # 1× RPM muito forte
        vibration += np.sin(2 * np.pi * f1x * t) * intensity * 2.5
        # Componente axial (fase diferente)
        vibration_y = np.sin(2 * np.pi * f1x * t + np.pi/2) * intensity * 2.0
        vibration_z = np.sin(2 * np.pi * f1x * t + np.pi/4) * intensity * 1.5
        
        pressure = 2.5 - severity_level * 0.08 + np.random.randn(n_samples) * 0.08
        flow = 100 - severity_level * 5 + np.random.randn(n_samples) * 2.5
        temperature = 45 + severity_level * 2 + np.random.randn(n_samples) * 1.0
        motor_current = 15 + severity_level * 1.2 + np.random.randn(n_samples) * 0.4
        
    elif mode == "misalignment":
        # 2× RPM dominante (harmónico forte)
        vibration += np.sin(2 * np.pi * f2x * t) * intensity * 2.0
        vibration += np.sin(2 * np.pi * f3x * t) * intensity * 0.8  # 3× também aparece
        # Componentes axiais fortes
        vibration_y = np.sin(2 * np.pi * f2x * t + np.pi/3) * intensity * 1.5
        vibration_z = np.sin(2 * np.pi * f1x * t) * intensity * 1.0
        
        pressure = 2.5 - severity_level * 0.10 + np.random.randn(n_samples) * 0.09
        flow = 100 - severity_level * 6 + np.random.randn(n_samples) * 3.0
        temperature = 45 + severity_level * 4 + np.random.randn(n_samples) * 1.5
        motor_current = 15 + severity_level * 1.0 + np.random.randn(n_samples) * 0.4
    
    else:
        raise ValueError(f"Modo desconhecido: {mode}")
    
    # Criar vibration_x, _y, _z para imbalance/misalignment
    if mode in ["imbalance", "misalignment"]:
        vibration_x = vibration
    else:
        vibration_x = vibration
        vibration_y = vibration * 0.7 + np.random.randn(n_samples) * baseline_noise * 0.5
        vibration_z = vibration * 0.5 + np.random.randn(n_samples) * baseline_noise * 0.3
    
    # Overall vibration (RMS-like)
    overall_vibration = np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2) / np.sqrt(3)
    
    # Ultrasonic noise (aumenta com degradação)
    ultrasonic = 30 + severity_level * 8 + np.random.randn(n_samples) * 3.0
    
    return {
        "overall_vibration": np.mean(overall_vibration),
        "vibration_x": np.mean(vibration_x),
        "vibration_y": np.mean(vibration_y),
        "vibration_z": np.mean(vibration_z),
        "pressure": np.mean(pressure),
        "flow": np.mean(flow),
        "temperature": np.mean(temperature),
        "motor_current": np.mean(motor_current),
        "ultrasonic_noise": np.mean(ultrasonic),
    }


def generate_synthetic_dataset(output_dir: Path):
    """
    Gera dataset sintético completo com assinaturas espectrais realistas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GERANDO DADOS SINTÉTICOS COM ASSINATURAS ESPECTRAIS REALISTAS")
    print("=" * 80)
    
    all_rows = []
    
    # Gerar dados para cada asset
    for asset_id in range(1, N_ASSETS + 1):
        asset_name = f"PUMP_{asset_id:02d}"
        
        # Cada asset tem baseline único
        asset_baseline = {
            "offset": np.random.uniform(-0.2, 0.2),
            "noise_level": np.random.uniform(0.05, 0.15),
            "gain": np.random.uniform(0.8, 1.2),
        }
        
        print(f"\n[{asset_name}] Baseline: offset={asset_baseline['offset']:.3f}, noise={asset_baseline['noise_level']:.3f}, gain={asset_baseline['gain']:.3f}")
        
        # Simular ciclo de vida do asset
        samples_per_mode = SAMPLES_PER_ASSET // len(MODES)
        
        for mode_idx, mode in enumerate(MODES):
            # Dentro de cada modo, passar por severidades
            samples_per_severity = samples_per_mode // len(SEVERITIES)
            
            for sev_idx, severity in enumerate(SEVERITIES):
                # RUL decresce com severity
                # Normal: RUL=1.0, Failure: RUL=0.0
                rul_base = 1.0 - (sev_idx / (len(SEVERITIES) - 1))
                
                # Health também decresce
                health_base = 1.0 - (sev_idx / (len(SEVERITIES) - 1))
                
                # Gerar samples para esta combinação
                for sample_idx in range(samples_per_severity):
                    # Timestamp sequencial
                    hours_elapsed = (mode_idx * samples_per_mode + sev_idx * samples_per_severity + sample_idx) * 0.1
                    timestamp = datetime(2024, 1, 1) + timedelta(hours=hours_elapsed)
                    
                    # Drift gradual dentro da severity
                    drift = sample_idx / samples_per_severity
                    rul = max(0.0, rul_base - drift * 0.1 + np.random.uniform(-0.02, 0.02))
                    health = max(0.0, health_base - drift * 0.05 + np.random.uniform(-0.01, 0.01))
                    
                    # Gerar sinais com assinatura espectral
                    signals = generate_spectral_signature(mode, sev_idx, n_samples=1000, asset_baseline=asset_baseline)
                    
                    # Criar row
                    row = {
                        "timestamp": timestamp.isoformat(),
                        "asset_id": asset_name,
                        "mode": mode,
                        "severity": severity,
                        "rul_minutes": rul * 1000,  # 0-1000 minutos
                        "health_index": health,
                        **signals
                    }
                    
                    all_rows.append(row)
        
        print(f"  Gerados {len(all_rows) - (asset_id - 1) * SAMPLES_PER_ASSET} samples")
    
    # Criar DataFrame
    df = pd.DataFrame(all_rows)
    
    # Salvar CSV principal
    output_file = output_dir / "sensors_log.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset salvo: {output_file}")
    print(f"   Total rows: {len(df)}")
    print(f"   Assets: {df['asset_id'].nunique()}")
    print(f"   Modes: {df['mode'].nunique()}")
    print(f"   Severities: {df['severity'].nunique()}")
    
    # Estatísticas por modo
    print("\n[DISTRIBUIÇÃO POR MODO]")
    print(df.groupby("mode").size())
    
    print("\n[DISTRIBUIÇÃO POR SEVERITY]")
    print(df.groupby("severity").size())
    
    # Salvar metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_assets": N_ASSETS,
        "samples_per_asset": SAMPLES_PER_ASSET,
        "total_samples": len(df),
        "sampling_rate": SAMPLING_RATE,
        "rpm_nominal": RPM_NOMINAL,
        "modes": MODES,
        "severities": SEVERITIES,
        "spectral_signatures": {
            "normal_operation": "baseline + low noise",
            "bearing_wear": "high freq (120-240Hz) + sidebands + noise",
            "cavitation": "broadband noise + burst peaks",
            "imbalance": "1× RPM dominant (30Hz)",
            "misalignment": "2× RPM dominant (60Hz) + 3× RPM",
        }
    }
    
    metadata_file = output_dir / "synthetic_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata salvo: {metadata_file}")
    
    return df


if __name__ == "__main__":
    output_dir = Path("logs")
    df = generate_synthetic_dataset(output_dir)
    print("\n" + "=" * 80)
    print("DATASET SINTÉTICO GERADO COM SUCESSO!")
    print("Pronto para treino com assinaturas espectrais realistas.")
    print("=" * 80)
