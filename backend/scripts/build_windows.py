#!/usr/bin/env python3
"""
build_windows.py
================
Script ETL para criar janelas prontas para treino a partir de timeseries processada.

Este script:
1. Carrega pump_timeseries_v3.csv/.parquet
2. Aplica normalização (StandardScaler)
3. Cria janelas deslizantes (FAST: espectrogramas, SLOW: estatísticas)
4. Salva como .npz pronto para treino

Output:
- pump_windows_train.npz: X_spec, X_aux, masks, y_severity, y_mode, y_rul, y_health
- pump_scalers.json: mean, std, min, max por coluna
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WindowConfig:
    """Configuração para criação de janelas."""
    
    # Janela temporal
    seq_len: int = 64           # Tamanho da janela em amostras
    hop: int = 8                # Hop entre janelas
    sample_rate_hz: float = 1.0  # Taxa de amostragem
    
    # STFT para FAST (vibração)
    n_fft: int = 32
    hop_stft: int = 4
    window_stft: str = "hann"
    
    # Sensores
    max_sensors_fast: int = 4   # Vibração: overall, x, y, z
    max_sensors_slow: int = 16  # Processo, elétrica, fluído
    
    # Estatísticas para AUX
    aux_stats: List[str] = None  # ["last", "mean", "std", "min", "max", "slope"]
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        if self.aux_stats is None:
            self.aux_stats = ["last", "mean", "std", "min", "max", "slope"]


# Definição de colunas por categoria
FAST_COLS = [
    "overall_vibration",
    "vibration_x", 
    "vibration_y",
    "vibration_z",
]

SLOW_COLS = [
    # Processo/Hidráulica
    "suction_pressure",
    "discharge_pressure", 
    "delta_p",
    "flow",
    "valve_position",
    "pump_speed_rpm",
    "head",
    
    # Elétrica
    "motor_current",
    "voltage_rms",
    "power_kw",
    "power_factor",
    "motor_temperature",
    
    # Fluído
    "fluid_temperature",
    "density",
    
    # Derivadas
    "hydraulic_power",
    "efficiency_est",
]

CONTEXT_COLS = [
    "run_state",
    "ambient_temperature",
]


def compute_stft_spectrogram(
    signal_data: np.ndarray,
    n_fft: int = 32,
    hop_length: int = 4,
    window: str = "hann"
) -> np.ndarray:
    """
    Calcula STFT real de um sinal temporal.
    
    Args:
        signal_data: Array 1D com o sinal
        n_fft: Tamanho da FFT
        hop_length: Hop entre frames
        window: Tipo de janela
    
    Returns:
        Espectrograma (n_freq, n_frames)
    """
    # Usar scipy.signal.stft
    f, t, Zxx = signal.stft(
        signal_data,
        fs=1.0,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        return_onesided=True
    )
    
    # Magnitude
    spec = np.abs(Zxx)
    
    return spec


def compute_window_stats(
    window_data: np.ndarray,
    stats: List[str]
) -> np.ndarray:
    """
    Calcula estatísticas de uma janela temporal.
    
    Args:
        window_data: Array (seq_len,) ou (seq_len, n_features)
        stats: Lista de estatísticas a calcular
    
    Returns:
        Array com estatísticas concatenadas
    """
    if window_data.ndim == 1:
        window_data = window_data.reshape(-1, 1)
    
    results = []
    
    for stat in stats:
        if stat == "last":
            results.append(window_data[-1, :])
        elif stat == "mean":
            results.append(np.mean(window_data, axis=0))
        elif stat == "std":
            results.append(np.std(window_data, axis=0) + 1e-8)
        elif stat == "min":
            results.append(np.min(window_data, axis=0))
        elif stat == "max":
            results.append(np.max(window_data, axis=0))
        elif stat == "slope":
            # Slope via regressão linear simples
            x = np.arange(len(window_data))
            slopes = []
            for col in range(window_data.shape[1]):
                coef = np.polyfit(x, window_data[:, col], 1)
                slopes.append(coef[0])
            results.append(np.array(slopes))
        elif stat == "range":
            results.append(np.max(window_data, axis=0) - np.min(window_data, axis=0))
        elif stat == "skew":
            from scipy.stats import skew
            results.append(skew(window_data, axis=0))
        elif stat == "kurtosis":
            from scipy.stats import kurtosis
            results.append(kurtosis(window_data, axis=0))
    
    return np.concatenate(results)


def create_windows_from_timeseries(
    df: pd.DataFrame,
    cfg: WindowConfig,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[Dict, StandardScaler, Dict]:
    """
    Cria janelas a partir de timeseries processada.
    
    Returns:
        (data_dict, scaler, stats_dict)
    """
    # Identificar colunas disponíveis
    fast_cols = [c for c in FAST_COLS if c in df.columns]
    slow_cols = [c for c in SLOW_COLS if c in df.columns]
    context_cols = [c for c in CONTEXT_COLS if c in df.columns]
    
    all_sensor_cols = fast_cols + slow_cols
    
    print(f"[INFO] FAST cols ({len(fast_cols)}): {fast_cols}")
    print(f"[INFO] SLOW cols ({len(slow_cols)}): {slow_cols}")
    
    # Normalização
    if fit_scaler:
        scaler = StandardScaler()
        df_norm = df.copy()
        df_norm[all_sensor_cols] = scaler.fit_transform(df[all_sensor_cols])
    else:
        df_norm = df.copy()
        df_norm[all_sensor_cols] = scaler.transform(df[all_sensor_cols])
    
    # Armazenar estatísticas do scaler
    scaler_stats = {
        "mean": dict(zip(all_sensor_cols, scaler.mean_.tolist())),
        "std": dict(zip(all_sensor_cols, scaler.scale_.tolist())),
    }
    
    # Listas para acumular janelas
    X_spec_list = []
    X_aux_list = []
    mask_list = []
    y_severity_list = []
    y_mode_list = []
    y_rul_list = []
    y_health_list = []
    
    # Calcular n_freq e n_frames com um sample de teste
    test_sig = np.random.randn(cfg.seq_len)
    test_spec = compute_stft_spectrogram(test_sig, cfg.n_fft, cfg.hop_stft, cfg.window_stft)
    n_freq = test_spec.shape[0]
    n_frames = test_spec.shape[1]
    print(f"[INFO] STFT shape: ({n_freq}, {n_frames})")
    
    # Processar por asset (preserva sequência temporal)
    for asset_id, group in df_norm.groupby("asset_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        
        if n < cfg.seq_len:
            print(f"[WARN] Asset {asset_id} has only {n} samples, skipping")
            continue
        
        # Iterar sobre janelas
        for i in range(0, n - cfg.seq_len + 1, cfg.hop):
            window = group.iloc[i:i + cfg.seq_len]
            
            # === FAST: Espectrogramas de vibração ===
            spec_data = np.zeros((cfg.max_sensors_fast, n_freq, n_frames), dtype=np.float32)
            
            for s_idx, col in enumerate(fast_cols[:cfg.max_sensors_fast]):
                sig = window[col].values
                spec = compute_stft_spectrogram(sig, cfg.n_fft, cfg.hop_stft, cfg.window_stft)
                # Copia o que couber
                h = min(spec.shape[0], n_freq)
                w = min(spec.shape[1], n_frames)
                spec_data[s_idx, :h, :w] = spec[:h, :w]
            
            # Adicionar canal (para CNN 2D)
            spec_data = spec_data[..., np.newaxis]  # (max_sensors, n_freq, n_frames, 1)
            
            # === SLOW: Estatísticas por janela ===
            slow_data = window[slow_cols].values  # (seq_len, n_slow)
            aux_features = compute_window_stats(slow_data, cfg.aux_stats)
            
            # Adicionar context (último valor)
            for ctx_col in context_cols:
                aux_features = np.append(aux_features, window[ctx_col].iloc[-1])
            
            # === Mask ===
            mask_fast = np.zeros(cfg.max_sensors_fast, dtype=np.float32)
            mask_fast[:len(fast_cols)] = 1.0
            
            # === Labels (último sample da janela) ===
            last_row = window.iloc[-1]
            
            # Acumular
            X_spec_list.append(spec_data)
            X_aux_list.append(aux_features.astype(np.float32))
            mask_list.append(mask_fast)
            y_severity_list.append(last_row["severity"])
            y_mode_list.append(last_row["mode"])
            y_rul_list.append(float(last_row["rul_minutes"]))
            y_health_list.append(float(last_row["health_index"]))
    
    # Converter para arrays
    data = {
        "X_spec": np.array(X_spec_list, dtype=np.float32),
        "X_aux": np.array(X_aux_list, dtype=np.float32),
        "mask": np.array(mask_list, dtype=np.float32),
        "y_severity": np.array(y_severity_list),
        "y_mode": np.array(y_mode_list),
        "y_rul": np.array(y_rul_list, dtype=np.float32),
        "y_health": np.array(y_health_list, dtype=np.float32),
    }
    
    stats = {
        "n_windows": len(X_spec_list),
        "spec_shape": data["X_spec"].shape,
        "aux_shape": data["X_aux"].shape,
        "fast_cols": fast_cols,
        "slow_cols": slow_cols,
        "context_cols": context_cols,
        "n_freq": n_freq,
        "n_frames": n_frames,
        "scaler": scaler_stats,
    }
    
    return data, scaler, stats


def encode_labels(
    y_severity: np.ndarray,
    y_mode: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Encode labels para inteiros."""
    
    severity_labels = sorted(np.unique(y_severity))
    mode_labels = sorted(np.unique(y_mode))
    
    sev_to_idx = {s: i for i, s in enumerate(severity_labels)}
    mode_to_idx = {m: i for i, m in enumerate(mode_labels)}
    
    y_sev_enc = np.array([sev_to_idx[s] for s in y_severity])
    y_mode_enc = np.array([mode_to_idx[m] for m in y_mode])
    
    return y_sev_enc, y_mode_enc, sev_to_idx, mode_to_idx


def save_windows(
    data: Dict,
    stats: Dict,
    output_dir: Path,
    rul_max_train: float = None
):
    """Salva janelas em formato NPZ + metadata JSON."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode labels
    y_sev_enc, y_mode_enc, sev_to_idx, mode_to_idx = encode_labels(
        data["y_severity"], data["y_mode"]
    )
    
    # Normalizar RUL e Health
    if rul_max_train is None:
        rul_max_train = float(data["y_rul"].max())
    
    y_rul_norm = data["y_rul"] / max(rul_max_train, 1.0)
    y_health_norm = data["y_health"] / 100.0
    
    # Salvar NPZ
    npz_path = output_dir / "pump_windows_train.npz"
    np.savez_compressed(
        npz_path,
        X_spec=data["X_spec"],
        X_aux=data["X_aux"],
        mask=data["mask"],
        y_severity=y_sev_enc,
        y_mode=y_mode_enc,
        y_rul=y_rul_norm,
        y_health=y_health_norm,
        # Raw labels para referência
        y_severity_raw=data["y_severity"],
        y_mode_raw=data["y_mode"],
        y_rul_raw=data["y_rul"],
        y_health_raw=data["y_health"],
    )
    print(f"[SAVE] Windows NPZ: {npz_path}")
    
    # Metadata
    metadata = {
        **stats,
        "rul_max_train": rul_max_train,
        "health_scale": 100.0,
        "severity_labels": list(sev_to_idx.keys()),
        "mode_labels": list(mode_to_idx.keys()),
        "sev_to_idx": sev_to_idx,
        "mode_to_idx": mode_to_idx,
        "severity_distribution": {
            str(k): int(v) for k, v in 
            zip(*np.unique(y_sev_enc, return_counts=True))
        },
        "mode_distribution": {
            str(k): int(v) for k, v in 
            zip(*np.unique(y_mode_enc, return_counts=True))
        },
    }
    
    meta_path = output_dir / "windows_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[SAVE] Metadata: {meta_path}")
    
    # Scaler separado
    scaler_path = output_dir / "pump_scalers.json"
    with open(scaler_path, "w") as f:
        json.dump(stats["scaler"], f, indent=2)
    print(f"[SAVE] Scalers: {scaler_path}")


def main():
    print("=" * 70)
    print("BUILD WINDOWS - ETL Pipeline")
    print("=" * 70)
    
    # Configuração
    cfg = WindowConfig(
        seq_len=64,
        hop=8,
        n_fft=32,
        hop_stft=4,
        max_sensors_fast=4,
        max_sensors_slow=16,
    )
    
    # Paths
    base_dir = Path(__file__).parent.parent
    
    # Tentar carregar parquet primeiro, depois CSV
    data_paths = [
        base_dir / "data" / "processed" / "pump_timeseries_v3.parquet",
        base_dir / "data" / "processed" / "pump_timeseries_v3.csv",
        base_dir / "datasets" / "pump_timeseries_v3.csv",
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            print(f"[LOAD] Loading: {path}")
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            break
    
    if df is None:
        print("[ERROR] No timeseries file found!")
        print("Run generate_dataset_v3_pump.py first.")
        return
    
    print(f"[INFO] Loaded {len(df)} samples")
    
    # Criar janelas
    print("\n[INFO] Creating windows...")
    data, scaler, stats = create_windows_from_timeseries(df, cfg, fit_scaler=True)
    
    print(f"[INFO] Created {stats['n_windows']} windows")
    print(f"[INFO] Spec shape: {stats['spec_shape']}")
    print(f"[INFO] Aux shape: {stats['aux_shape']}")
    
    # Salvar
    output_dir = base_dir / "data" / "processed"
    save_windows(data, stats, output_dir)
    
    # Também copiar para datasets/ para compatibilidade
    compat_dir = base_dir / "datasets"
    compat_dir.mkdir(exist_ok=True)
    
    import shutil
    for f in ["pump_windows_train.npz", "windows_metadata.json", "pump_scalers.json"]:
        src = output_dir / f
        dst = compat_dir / f
        if src.exists():
            shutil.copy(src, dst)
            print(f"[COPY] {dst}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Windows created!")
    print("=" * 70)


if __name__ == "__main__":
    main()
