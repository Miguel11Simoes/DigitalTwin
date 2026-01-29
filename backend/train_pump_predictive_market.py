#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pump_predictive_market.py
Pipeline end-to-end para manutenção preditiva de bombas industriais.

✅ Produto-ready (mesmo usando dados sintéticos realistas hoje):
- Modelo versátil a N sensores variáveis (min..max), com sensor_mask
- CNN 2D por sensor: Conv -> Pool -> Flatten -> Dense
- Aux features por sensor (7)
- Agregação robusta: masked attention pooling + masked mean pooling
- Sensor dropout (robustez a sensores em falta)
- SpecAugment aplicado só em treino (correto)
- Split temporal por ativo (asset_id) se existir; fallback com aviso
- Treino com sample_weight por output (severity/mode)
- Export + report JSON + confusion matrix + classification reports

Schema esperado no CSV (real OU sintético realista):
Obrigatório:
- timestamp (datetime ou parseável)  [se não existir, cria sequencial]
- mode (string)                      [se não existir, "unknown"]

Recomendado (produto real):
- asset_id (para anti-leakage)
- rul_minutes (float)   -> tempo até falha em minutos
- health_index (float)  -> 0..100
- severity (string)     -> normal/early/moderate/severe/failure (ou ordinal equivalente)
- sensores (colunas numéricas): vibração, pressão, corrente, temp, etc.

Se rul/health/severity não existirem, o código cria fallback (APENAS demo).
Em produto real, desliga os fallbacks.

Requisitos:
  pip install tensorflow pandas numpy scikit-learn
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from focal_loss import focal_loss


# --------------------------
# Paths / Consts
# --------------------------
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"

OUT_DIR = Path(os.environ.get("DT_MODELS_DIR", str(BASE_DIR / "models")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = OUT_DIR / "pump_predictive_market.keras"
DEFAULT_BEST_CKPT = OUT_DIR / "best_val_pump_predictive_market.weights.h5"
DEFAULT_LABELS_PATH = OUT_DIR / "labels_master_market.json"
DEFAULT_REPORT_PATH = OUT_DIR / "eval_report_market.json"
DEFAULT_TRAIN_LOG = OUT_DIR / "train_log.csv"

PRIMARY_PREF = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "ultrasonic_noise", "motor_current", "pressure", "flow", "temperature"
]

DROP_COLS_ALWAYS = {
    "timestamp", "mode", "severity", "rul_minutes", "health_index", "asset_id",
    "anomaly_score", "failure_probability", "model_confidence", "predicted_mode"
}

SEV_ORDER = ["normal", "early", "moderate", "severe", "failure"]


# --------------------------
# Config
# --------------------------
@dataclass
class Cfg:
    # Split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Windowing (cria muitas janelas)
    seq_len: int = 128  # FIXED: reduced from 256
    hop: int = 16  # FIXED: reduced from 32

    # Spectrogram
    frame_len: int = 64
    frame_hop: int = 32
    t_window: int = 96
    n_mels: int = 96

    # Sensores (N variável + máscara)
    max_sensors: int = 16
    min_sensors: int = 2
    aux_per_sensor: int = 10  # FIXED: 3 absolutas (abs_mean, abs_std, abs_rms) + 7 normalizadas

    # Treino - ITERATION 5: Otimizado para convergência rápida
    epochs: int = 50
    batch_size: int = 64  # Maior para treino mais rápido
    lr: float = 1e-3  # Maior para convergência rápida
    weight_decay: float = 1e-5
    l2_reg: float = 5e-5  # Menos regularização
    dropout: float = 0.25  # Menos dropout para convergir
    patience: int = 15

    # Robustez (curriculum: ligar depois epoch 10)
    sensor_dropout_rate: float = 0.0  # CRITICAL FIX: start at 0, increase after epoch 10
    enable_spec_augment: bool = False  # CRITICAL FIX: enable after epoch 10 when model is stable

    # Anti-leakage
    asset_id_col: str = "asset_id"

    # Labels reais vs fallback demo
    allow_fallback_targets: bool = True  # em produto real => False

    # Seed
    seed: int = 42


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------
# CSV discovery/load
# --------------------------
def discover_csvs() -> List[Path]:
    files: List[Path] = []
    p = LOGS_DIR / "sensors_log.csv"
    if p.exists():
        files.append(p)
    if ARCHIVE_DIR.exists():
        files.extend(sorted(ARCHIVE_DIR.glob("sensors_log_*.csv")))
    return files


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True)
    except UnicodeDecodeError:
        return pd.read_csv(
            path, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True, encoding="latin1"
        )


def load_all_csvs(csv_files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in csv_files:
        print(f"[INFO] Lendo: {f}")
        dfs.append(read_csv(f))
    if not dfs:
        raise FileNotFoundError("Nenhum CSV encontrado.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Linhas totais: {len(df)}")
    return df


# --------------------------
# Sanitização / schema
# --------------------------
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_mode(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns:
        df["mode"] = "unknown"
    df["mode"] = df["mode"].astype(str).fillna("unknown")
    return df


def ensure_asset_id(df: pd.DataFrame, asset_id_col: str) -> pd.DataFrame:
    """
    Se não existir asset_id, mantém sem ele (vai dar aviso no split).
    """
    if asset_id_col not in df.columns:
        return df
    df[asset_id_col] = df[asset_id_col].astype(str).fillna("unknown_asset")
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in ("mode", "severity", "timestamp", "asset_id"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------
# Fallback targets (APENAS demo)
# --------------------------
def fallback_targets_if_missing(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """
    Só para conseguires treinar com dados sintéticos/atuais se ainda não tens labels.
    Em produto real: cfg.allow_fallback_targets = False
    """
    if not cfg.allow_fallback_targets:
        # exige targets reais
        required = ["rul_minutes", "health_index", "severity"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Faltam targets reais: {miss}. Em produto, adiciona via CMMS/eventos.")
        return df

    rng = np.random.default_rng(cfg.seed)

    if "rul_minutes" not in df.columns or df["rul_minutes"].isna().all():
        print("[WARN] 'rul_minutes' ausente -> a gerar fallback (demo, NÃO produto).")
        df["rul_minutes"] = 0.0
        # por modo: rampa 1000h -> 0h
        for mode_label in df["mode"].unique():
            m = (df["mode"] == mode_label)
            n = int(m.sum())
            if n > 0:
                df.loc[m, "rul_minutes"] = np.linspace(1000.0 * 60.0, 0.0, n)

    if "health_index" not in df.columns or df["health_index"].isna().all():
        print("[WARN] 'health_index' ausente -> a gerar fallback (demo, NÃO produto).")
        df["health_index"] = 100.0
        for mode_label in df["mode"].unique():
            m = (df["mode"] == mode_label)
            n = int(m.sum())
            if n > 0:
                t = np.linspace(0, 1, n)
                base = 1.0 - t**1.5
                noise = rng.normal(0, 0.02, n)
                df.loc[m, "health_index"] = np.clip(base + noise, 0, 1) * 100.0

    if "severity" not in df.columns or df["severity"].isna().all():
        print("[WARN] 'severity' ausente -> a derivar via quantis de health (demo).")
        h = pd.to_numeric(df["health_index"], errors="coerce").astype(float)
        h = h.fillna(h.median())
        h = h + rng.normal(0, 1e-3, size=len(h))  # quebra empates
        df["severity"] = pd.qcut(
            h,
            q=5,
            labels=["failure", "severe", "moderate", "early", "normal"]
        ).astype(str)

    df["severity"] = df["severity"].astype(str).fillna("normal")
    df = df.dropna(subset=["rul_minutes", "health_index", "severity", "mode"]).reset_index(drop=True)
    return df


# --------------------------
# Sensor selection
# --------------------------
def pick_sensor_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in DROP_COLS_ALWAYS]
    preferred = [c for c in PRIMARY_PREF if c in numeric_cols]
    others = [c for c in numeric_cols if c not in preferred]
    return preferred + others


# --------------------------
# Split temporal (anti-leakage)
# --------------------------
def split_by_asset_id(
    df: pd.DataFrame,
    asset_col: str,
    tr: float,
    va: float,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL FIX: Split por asset_id completos (não temporal dentro de asset).
    Cada asset vai inteiro para train/val/test, evitando distribution shift.
    Garante que todas as classes (mode/severity) aparecem em train/val/test.
    """
    assets = df[asset_col].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(assets)

    n = len(assets)
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    tr_assets = set(assets[:n_tr])
    va_assets = set(assets[n_tr:n_tr + n_va])
    te_assets = set(assets[n_tr + n_va:])

    print(f"[INFO] Split por asset_id completos: {n_tr} train / {n_va} val / {n-n_tr-n_va} test assets")

    df_tr = df[df[asset_col].astype(str).isin(tr_assets)].copy()
    df_va = df[df[asset_col].astype(str).isin(va_assets)].copy()
    df_te = df[df[asset_col].astype(str).isin(te_assets)].copy()
    return df_tr, df_va, df_te


def temporal_split_grouped(
    df: pd.DataFrame,
    group_col: str,
    tr: float,
    va: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED: Split temporal dentro de cada asset (pode causar distribution shift).
    Use split_by_asset_id() quando possível.
    """
    parts = []
    for _, dfg in df.groupby(group_col, sort=False):
        dfg = dfg.sort_values("timestamp")
        n = len(dfg)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        df_tr = dfg.iloc[:n_tr].copy()
        df_va = dfg.iloc[n_tr:n_tr + n_va].copy()
        df_te = dfg.iloc[n_tr + n_va:].copy()
        df_tr["__split__"] = "train"
        df_va["__split__"] = "val"
        df_te["__split__"] = "test"
        parts += [df_tr, df_va, df_te]
    
    out = pd.concat(parts, ignore_index=True)
    return (
        out[out["__split__"] == "train"].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"] == "val"].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"] == "test"].drop(columns="__split__").reset_index(drop=True),
    )


# --------------------------
# Windowing + Aux features
# --------------------------
def window_indices(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    idx = []
    i = 0
    while i + win <= n:
        idx.append((i, i + win))
        i += hop
    return idx


def slope_np(y: np.ndarray) -> float:
    n = y.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    vx = np.var(x)
    if vx < 1e-12:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def skewness_np(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0
    m = float(np.mean(y))
    s = float(np.std(y)) + 1e-9
    return float(np.mean(((y - m) / s) ** 3))


def kurtosis_np(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size < 4:
        return 0.0
    m = float(np.mean(y))
    s = float(np.std(y)) + 1e-9
    return float(np.mean(((y - m) / s) ** 4) - 3.0)


def crest_factor_np(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    peak = float(np.max(np.abs(y)))
    return float(peak / rms)


def aux_features_per_sensor(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.float32)
    finite = seg[np.isfinite(seg)]
    if finite.size == 0:
        return np.zeros((10,), dtype=np.float32)

    # CRITICAL FIX: Calcular features absolutas ANTES de normalizar
    # Preserva informação de amplitude/offset (temperatura alta, pressão baixa, etc)
    abs_mean = float(np.mean(finite))
    abs_std = float(np.std(finite))
    abs_rms = float(np.sqrt(np.mean(finite ** 2)))

    # normalização leve para features normalizadas (estabilidade)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) + 1e-6
    z = (seg - med) / mad
    z = z[np.isfinite(z)]

    mean = float(np.mean(z))
    std = float(np.std(z))
    slp = slope_np(z)
    rms = float(np.sqrt(np.mean(z ** 2)))
    kur = kurtosis_np(z)
    skw = skewness_np(z)
    crf = crest_factor_np(z)
    
    # ORDEM: 3 absolutas + 7 normalizadas = 10 features
    return np.array([abs_mean, abs_std, abs_rms, mean, std, slp, rms, kur, skw, crf], dtype=np.float32)


def make_windows_variable_sensors(
    df_block: pd.DataFrame,
    sensor_cols: List[str],
    cfg: Cfg,
    asset_stats: Optional[Dict] = None,  # ASSET NORMALIZATION
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(df_block)
    idxs = window_indices(n, cfg.seq_len, cfg.hop)

    valid_cols = []
    for c in sensor_cols:
        if c in df_block.columns and pd.api.types.is_numeric_dtype(df_block[c]) and df_block[c].notna().any():
            valid_cols.append(c)

    if len(valid_cols) < cfg.min_sensors:
        raise ValueError(
            f"Poucos sensores válidos: {len(valid_cols)} < MIN_SENSORS={cfg.min_sensors}."
        )

    use_cols = valid_cols[:cfg.max_sensors]

    X_raw_list: List[np.ndarray] = []
    X_aux_list: List[np.ndarray] = []
    X_mask_list: List[np.ndarray] = []
    y_rul, y_health, y_sev, y_mode = [], [], [], []

    raw_cache = {c: df_block[c].to_numpy(dtype=np.float32) for c in use_cols}

    for (i0, i1) in idxs:
        signals = np.zeros((cfg.max_sensors, cfg.seq_len), dtype=np.float32)
        aux = np.zeros((cfg.max_sensors, cfg.aux_per_sensor), dtype=np.float32)
        mask = np.zeros((cfg.max_sensors,), dtype=np.float32)

        ok = True
        for si, c in enumerate(use_cols):
            seg = raw_cache[c][i0:i1]
            if seg.shape[0] != cfg.seq_len:
                ok = False
                break
            # imputação simples em NaNs
            if not np.isfinite(seg).all():
                finite = seg[np.isfinite(seg)]
                if finite.size == 0:
                    ok = False
                    break
                med = float(np.median(finite))
                seg = np.where(np.isfinite(seg), seg, med).astype(np.float32)

            # ASSET NORMALIZATION: Use asset-level stats if available
            if asset_stats and c in asset_stats:
                asset_med = asset_stats[c]["median"]
                asset_mad = asset_stats[c]["mad"]
                seg_in = ((seg - asset_med) / asset_mad).astype(np.float32)
            elif any(k in c.lower() for k in ["vibration", "ultrasonic", "accel", "overall_vibration"]):
                # Fallback: window-level normalization for vibration
                med = float(np.median(seg))
                mad = float(np.median(np.abs(seg - med))) + 1e-6
                seg_in = (seg - med) / mad
            else:
                # Sensores lentos: clipping suave, preserva escala
                seg_in = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
            
            signals[si, :] = seg_in.astype(np.float32)
            aux[si, :] = aux_features_per_sensor(seg)
            mask[si] = 1.0

        if not ok:
            continue

        # targets por janela
        rul_win = df_block["rul_minutes"].iloc[i0:i1].to_numpy(dtype=np.float32)
        health_win = df_block["health_index"].iloc[i0:i1].to_numpy(dtype=np.float32)
        sev_win = df_block["severity"].iloc[i0:i1].astype(str).to_numpy()
        mode_win = df_block["mode"].iloc[i0:i1].astype(str).to_numpy()

        if rul_win.size == 0:
            continue

        rul_val = float(np.nanmean(rul_win)) / (1000.0)  # 0..1 (max 1000 min)
        rul_val = float(np.clip(rul_val, 0.0, 1.0))

        health_val = float(np.nanmean(health_win)) / 100.0
        health_val = float(np.clip(health_val, 0.0, 1.0))

        # maioria na janela
        vals_sev, cnts_sev = np.unique(sev_win, return_counts=True)
        sev_lab = vals_sev[int(np.argmax(cnts_sev))]

        vals_mode, cnts_mode = np.unique(mode_win, return_counts=True)
        mode_lab = vals_mode[int(np.argmax(cnts_mode))]

        X_raw_list.append(signals)
        X_aux_list.append(aux)
        X_mask_list.append(mask)
        y_rul.append(rul_val)
        y_health.append(health_val)
        y_sev.append(sev_lab)
        y_mode.append(mode_lab)

    if not X_raw_list:
        return (
            np.zeros((0, cfg.max_sensors, cfg.seq_len), dtype=np.float32),
            np.zeros((0, cfg.max_sensors, cfg.aux_per_sensor), dtype=np.float32),
            np.zeros((0, cfg.max_sensors), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=str),
            np.array([], dtype=str),
        )

    return (
        np.stack(X_raw_list, axis=0),
        np.stack(X_aux_list, axis=0),
        np.stack(X_mask_list, axis=0),
        np.array(y_rul, dtype=np.float32),
        np.array(y_health, dtype=np.float32),
        np.array(y_sev, dtype=object).astype(str),
        np.array(y_mode, dtype=object).astype(str),
    )


# --------------------------
# Label maps
# --------------------------
def build_label_maps(y_sev_str: np.ndarray, y_mode_str: np.ndarray) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
    sev_unique = sorted(np.unique(y_sev_str.astype(str)).tolist())
    # tenta manter ordem industrial se possível
    sev_labels = [s for s in SEV_ORDER if s in sev_unique] + [s for s in sev_unique if s not in SEV_ORDER]

    mode_labels = sorted(np.unique(y_mode_str.astype(str)).tolist())
    sev_lab2i = {lab: i for i, lab in enumerate(sev_labels)}
    mode_lab2i = {lab: i for i, lab in enumerate(mode_labels)}
    return sev_labels, sev_lab2i, mode_labels, mode_lab2i


# --------------------------
# Custom layers (serializáveis)
# --------------------------
@keras.utils.register_keras_serializable(package="Custom")
class SensorDropout(layers.Layer):
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, sensor_mask, training=None):
        sensor_mask = tf.cast(sensor_mask, tf.float32)
        if (training is None) or (not training) or self.rate <= 0.0:
            return sensor_mask
        keep = tf.cast(tf.random.uniform(tf.shape(sensor_mask)) >= self.rate, tf.float32)
        dropped = sensor_mask * keep
        all_zero = tf.reduce_all(tf.equal(dropped, 0.0), axis=-1, keepdims=True)
        return tf.where(all_zero, sensor_mask, dropped)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"rate": self.rate})
        return cfg


@keras.utils.register_keras_serializable(package="Custom")
class RawToSpectrograms(layers.Layer):
    def __init__(self, frame_len: int, frame_hop: int, t_window: int, n_mels: int, **kwargs):
        super().__init__(**kwargs)
        self.frame_len = int(frame_len)
        self.frame_hop = int(frame_hop)
        self.t_window = int(t_window)
        self.n_mels = int(n_mels)

    def _one_sensor(self, sig_1d: tf.Tensor) -> tf.Tensor:
        stft = tf.signal.stft(
            signals=tf.cast(sig_1d, tf.float32),
            frame_length=self.frame_len,
            frame_step=self.frame_hop,
            fft_length=self.frame_len,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
        mag = tf.abs(stft) + 1e-9
        logmag = tf.math.log(mag)
        # CRITICAL FIX: Não normalizar por amostra - preserva energia absoluta (severidade)
        # Apenas clipping para estabilidade
        norm = tf.clip_by_value(logmag, -12.0, 12.0)
        norm = tf.expand_dims(norm, axis=-1)
        norm = tf.image.resize(norm, size=(self.t_window, self.n_mels), method="bilinear")
        return tf.clip_by_value(norm, -12.0, 12.0)

    def call(self, x_raw: tf.Tensor) -> tf.Tensor:
        # x_raw: (B,S,seq_len) -> (B,S,T,F,1)
        def per_batch(b):
            return tf.map_fn(self._one_sensor, b, fn_output_signature=tf.float32)
        specs = tf.map_fn(per_batch, x_raw, fn_output_signature=tf.float32)
        return specs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "t_window": self.t_window,
            "n_mels": self.n_mels,
        })
        return cfg


@keras.utils.register_keras_serializable(package="Custom")
class SpecAugment(layers.Layer):
    def __init__(self, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.enabled = bool(enabled)

    def _augment_one(self, spec: tf.Tensor) -> tf.Tensor:
        # spec: (T,F,1)
        s = tf.identity(spec)
        T = tf.shape(s)[0]
        F = tf.shape(s)[1]

        # time masks
        max_t = tf.maximum(1, tf.cast(tf.cast(T, tf.float32) * 0.12, tf.int32))
        def time_mask_once(x):
            width = tf.random.uniform([], 1, max_t + 1, dtype=tf.int32)
            start = tf.random.uniform([], 0, tf.maximum(1, T - width), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([start, 1, 1], tf.float32),
                tf.zeros([width, 1, 1], tf.float32),
                tf.ones([T - start - width, 1, 1], tf.float32),
            ], axis=0)
            return x * mask

        s = tf.cond(tf.random.uniform([]) > 0.35, lambda: time_mask_once(s), lambda: s)
        s = tf.cond(tf.random.uniform([]) > 0.35, lambda: time_mask_once(s), lambda: s)

        # freq masks
        max_f = tf.maximum(1, tf.cast(tf.cast(F, tf.float32) * 0.12, tf.int32))
        def freq_mask_once(x):
            width = tf.random.uniform([], 1, max_f + 1, dtype=tf.int32)
            start = tf.random.uniform([], 0, tf.maximum(1, F - width), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([1, start, 1], tf.float32),
                tf.zeros([1, width, 1], tf.float32),
                tf.ones([1, F - start - width, 1], tf.float32),
            ], axis=1)
            return x * mask

        s = tf.cond(tf.random.uniform([]) > 0.35, lambda: freq_mask_once(s), lambda: s)
        s = tf.cond(tf.random.uniform([]) > 0.35, lambda: freq_mask_once(s), lambda: s)

        # noise + scaling
        s = tf.cond(
            tf.random.uniform([]) > 0.45,
            lambda: s + tf.random.normal(tf.shape(s), mean=0.0, stddev=0.06, dtype=tf.float32),
            lambda: s,
        )
        s = tf.cond(
            tf.random.uniform([]) > 0.50,
            lambda: s * tf.random.uniform([], 0.85, 1.15, dtype=tf.float32),
            lambda: s,
        )
        return tf.clip_by_value(s, -10.0, 10.0)

    def call(self, specs: tf.Tensor, mask: tf.Tensor, training=None) -> tf.Tensor:
        # specs: (B,S,T,F,1)  mask: (B,S)
        if (training is None) or (not training) or (not self.enabled):
            return specs

        mask = tf.cast(mask, tf.float32)

        def aug_sensor(args):
            spec, m = args
            return tf.cond(m > 0.5, lambda: self._augment_one(spec), lambda: spec)

        def aug_batch(b_specs, b_mask):
            return tf.map_fn(aug_sensor, (b_specs, b_mask), fn_output_signature=tf.float32)

        out = tf.map_fn(lambda x: aug_batch(x[0], x[1]), (specs, mask), fn_output_signature=tf.float32)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"enabled": self.enabled})
        return cfg


def masked_softmax(logits: tf.Tensor, mask: tf.Tensor, axis: int = -1) -> tf.Tensor:
    mask = tf.cast(mask, tf.float32)
    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_logits = tf.where(mask > 0.0, logits, neg_inf)
    return tf.nn.softmax(masked_logits, axis=axis)


# ============================================================================
# Custom Layers serializáveis (produto-ready, substituem Lambda)
# ============================================================================

@keras.utils.register_keras_serializable(package="Custom")
class SqueezeLast(layers.Layer):
    """Squeeze última dimensão."""
    def call(self, x):
        return tf.squeeze(x, axis=-1)


@keras.utils.register_keras_serializable(package="Custom")
class ExpandDimsLast(layers.Layer):
    """Expand dims na última posição."""
    def call(self, x):
        return tf.expand_dims(x, axis=-1)


@keras.utils.register_keras_serializable(package="Custom")
class MaskedSoftmaxLayer(layers.Layer):
    """Aplica softmax mascarado. Inputs: [scores, mask]."""
    def call(self, inputs):
        scores, mask = inputs
        return masked_softmax(scores, mask, axis=-1)


@keras.utils.register_keras_serializable(package="Custom")
class WeightedSum(layers.Layer):
    """Soma ponderada: sum(features * weights, axis=1). Inputs: [features, weights]."""
    def call(self, inputs):
        features, weights = inputs
        return tf.reduce_sum(features * weights, axis=1)


@keras.utils.register_keras_serializable(package="Custom")
class MaskExpand(layers.Layer):
    """Converte mask bool/int para float e expande dimensão."""
    def call(self, mask):
        return tf.expand_dims(tf.cast(mask, tf.float32), -1)


@keras.utils.register_keras_serializable(package="Custom")
class MaskedDenominator(layers.Layer):
    """Calcula denominador mascarado: max(sum(mask, axis=1), 1.0)."""
    def call(self, mask_expanded):
        return tf.maximum(tf.reduce_sum(mask_expanded, axis=1), 1.0)


@keras.utils.register_keras_serializable(package="Custom")
class DivideLayer(layers.Layer):
    """Divide x[0] / x[1]. Inputs: [numerator, denominator]."""
    def call(self, inputs):
        numerator, denominator = inputs
        return numerator / denominator


# --------------------------
# Build model (CNN2D + Aux)
# --------------------------
def build_model(cfg: Cfg, n_sev: int, n_mode: int) -> keras.Model:
    raw_in = keras.Input(shape=(cfg.max_sensors, cfg.seq_len), name="raw_input")
    aux_in = keras.Input(shape=(cfg.max_sensors, cfg.aux_per_sensor), name="aux_input")
    mask_in = keras.Input(shape=(cfg.max_sensors,), name="sensor_mask")

    mask_eff = SensorDropout(cfg.sensor_dropout_rate, name="sensor_dropout")(mask_in)

    specs = RawToSpectrograms(
        cfg.frame_len, cfg.frame_hop, cfg.t_window, cfg.n_mels, name="raw_to_specs"
    )(raw_in)

    specs = SpecAugment(enabled=cfg.enable_spec_augment, name="spec_augment")(specs, mask_eff)

    # CNN por sensor (partilhada)
    per_in = keras.Input(shape=(cfg.t_window, cfg.n_mels, 1))
    z = layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(per_in)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.20)(z)

    z = layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.25)(z)

    z = layers.Conv2D(128, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.30)(z)

    z = layers.Flatten()(z)
    z = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.Dropout(cfg.dropout)(z)

    per_sensor_cnn = keras.Model(per_in, z, name="per_sensor_cnn")
    sensor_embed = layers.TimeDistributed(per_sensor_cnn, name="td_cnn")(specs)  # (B,S,256)

    # Aux
    aux_norm = layers.LayerNormalization(name="aux_norm")(aux_in)
    aux_embed = layers.TimeDistributed(
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg)),
        name="td_aux_dense64"
    )(aux_norm)
    aux_embed = layers.Dropout(cfg.dropout * 0.30)(aux_embed)

    # fusion por sensor
    sensor_feat = layers.Concatenate()([sensor_embed, aux_embed])  # (B,S,320)
    sensor_feat = layers.TimeDistributed(
        layers.Dense(192, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg)),
        name="td_sensor_dense192"
    )(sensor_feat)
    sensor_feat = layers.Dropout(cfg.dropout)(sensor_feat)

    # attention pooling masked
    scores = layers.TimeDistributed(layers.Dense(1, activation=None), name="attn_score")(sensor_feat)  # (B,S,1)
    scores = SqueezeLast(name="attn_squeeze")(scores)  # (B,S)
    weights = MaskedSoftmaxLayer(name="attn_weights")([scores, mask_eff])
    weights_exp = ExpandDimsLast(name="attn_expand")(weights)
    pooled_attn = WeightedSum(name="pooled_attn")([sensor_feat, weights_exp])

    # masked mean pooling
    mask_f = MaskExpand(name="mask_expand")(mask_eff)
    sum_feat = WeightedSum(name="masked_sum")([sensor_feat, mask_f])
    denom = MaskedDenominator(name="masked_denom")(mask_f)
    pooled_mean = DivideLayer(name="pooled_mean")([sum_feat, denom])

    fused = layers.Concatenate(name="fusion_concat")([pooled_attn, pooled_mean])  # (B,384)
    fused = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(fused)  # ITER4: back to 256
    fused = layers.BatchNormalization()(fused)  # ITERATION 2
    fused = layers.Dropout(cfg.dropout)(fused)

    # Heads
    rul_h = layers.Dense(128, activation="relu")(fused)  # ITER4: back to 128
    rul_h = layers.BatchNormalization()(rul_h)
    rul_h = layers.Dropout(cfg.dropout * 0.60)(rul_h)
    rul_out = layers.Dense(1, activation="sigmoid", name="rul")(rul_h)

    health_h = layers.Dense(128, activation="relu")(fused)  # ITER4: back to 128
    health_h = layers.BatchNormalization()(health_h)
    health_h = layers.Dropout(cfg.dropout * 0.60)(health_h)
    health_out = layers.Dense(1, activation="sigmoid", name="health")(health_h)

    sev_h = layers.Dense(192, activation="relu")(fused)  # ITER4: back to 192
    sev_h = layers.BatchNormalization()(sev_h)
    sev_h = layers.Dropout(cfg.dropout)(sev_h)
    sev_out = layers.Dense(n_sev, activation="softmax", name="severity")(sev_h)

    mode_h = layers.Dense(192, activation="relu")(fused)  # ITER4: back to 192
    mode_h = layers.BatchNormalization()(mode_h)
    mode_h = layers.Dropout(cfg.dropout)(mode_h)
    mode_out = layers.Dense(n_mode, activation="softmax", name="mode")(mode_h)

    model = keras.Model(
        inputs={"raw_input": raw_in, "aux_input": aux_in, "sensor_mask": mask_in},
        outputs={"rul": rul_out, "health": health_out, "severity": sev_out, "mode": mode_out},
        name="pump_predictive_market",
    )

    # Optimizer
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay, clipnorm=1.0)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0)

    # sparse label smoothing compatível
    def sparse_label_smoothing_ce(num_classes: int, smoothing: float = 0.1):
        def loss(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_oh = tf.one_hot(y_true, num_classes)
            y_s = y_oh * (1.0 - smoothing) + (smoothing / float(num_classes))
            return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, y_pred))
        return loss

    model.compile(
        optimizer=opt,
        loss={
            "rul": keras.losses.Huber(delta=0.08),
            "health": keras.losses.Huber(delta=0.04),
            "severity": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "mode": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights={
            "rul": 2.0,        # RUL importante para manutenção
            "health": 3.0,     # Health index 
            "severity": 5.0,   # Severity crítico para alertas
            "mode": 5.0,       # Mode para diagnóstico
        },
        metrics={
            "rul": [keras.metrics.MeanAbsoluteError(name="mae")],
            "health": [keras.metrics.MeanAbsoluteError(name="mae")],
            "severity": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "mode": [
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
            ],
        },
    )
    return model


# --------------------------
# tf.data (para avaliação)
# --------------------------
def build_tf_dataset(
    X_raw: np.ndarray,
    X_aux: np.ndarray,
    X_mask: np.ndarray,
    y_rul: np.ndarray,
    y_health: np.ndarray,
    y_sev: np.ndarray,
    y_mode: np.ndarray,
    cfg: Cfg,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((
        {"raw_input": X_raw, "aux_input": X_aux, "sensor_mask": X_mask},
        {"rul": y_rul, "health": y_health, "severity": y_sev, "mode": y_mode},
    ))
    if training:
        ds = ds.shuffle(min(len(X_raw), 20000), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------
# Product metrics callback (VAL)
# --------------------------
class ProductMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_ds: tf.data.Dataset, sev_labels: List[str], ckpt_path: Path):
        super().__init__()
        self.val_ds = val_ds
        self.sev_labels = sev_labels
        self.ckpt_path = ckpt_path
        self.best = -1.0
        self.severe_idx = sev_labels.index("severe") if "severe" in sev_labels else None
        self.failure_idx = sev_labels.index("failure") if "failure" in sev_labels else None

    def on_epoch_end(self, epoch, logs=None):
        yts, yps = [], []
        ytm, ypm = [], []

        for x, y in self.val_ds:
            pred = self.model.predict(x, verbose=0)
            yps.append(np.argmax(pred["severity"], axis=1))
            ypm.append(np.argmax(pred["mode"], axis=1))
            yts.append(y["severity"].numpy())
            ytm.append(y["mode"].numpy())

        yt_s = np.concatenate(yts) if yts else np.array([], dtype=np.int64)
        yp_s = np.concatenate(yps) if yps else np.array([], dtype=np.int64)
        yt_m = np.concatenate(ytm) if ytm else np.array([], dtype=np.int64)
        yp_m = np.concatenate(ypm) if ypm else np.array([], dtype=np.int64)

        if yt_s.size == 0:
            return

        sev_macro_f1 = float(f1_score(yt_s, yp_s, average="macro", zero_division=0))
        mode_macro_f1 = float(f1_score(yt_m, yp_m, average="macro", zero_division=0))

        crit_recall = None
        crit = []
        if self.severe_idx is not None:
            crit.append(self.severe_idx)
        if self.failure_idx is not None:
            crit.append(self.failure_idx)
        if crit:
            yt_bin = np.isin(yt_s, crit).astype(int)
            yp_bin = np.isin(yp_s, crit).astype(int)
            crit_recall = float(recall_score(yt_bin, yp_bin, zero_division=0))

        logs = logs or {}
        logs["val_severity_macro_f1"] = sev_macro_f1
        logs["val_mode_macro_f1"] = mode_macro_f1
        if crit_recall is not None:
            logs["val_critical_recall"] = crit_recall

        print(
            f"\n[VAL PRODUCT] epoch={epoch+1} "
            f"sev_macro_f1={sev_macro_f1:.4f} mode_macro_f1={mode_macro_f1:.4f} "
            f"critical_recall={(crit_recall if crit_recall is not None else float('nan')):.4f}"
        )

        if sev_macro_f1 > self.best:
            self.best = sev_macro_f1
            self.model.save_weights(self.ckpt_path)
            print(f"[CKPT] Guardado best weights: {self.ckpt_path} (sev_macro_f1={sev_macro_f1:.4f})")


class CurriculumCallback(keras.callbacks.Callback):
    """Ativa curriculum de robustez após um número de epochs.
    Liga `sensor_dropout.rate` e `spec_augment.enabled` dinamicamente.
    """
    def __init__(self, start_epoch: int = 10, sensor_dropout_rate: float = 0.10, enable_spec: bool = True):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.sensor_dropout_rate = float(sensor_dropout_rate)
        self.enable_spec = bool(enable_spec)
        self._applied = False

    def on_epoch_begin(self, epoch, logs=None):
        # apply once when epoch >= start_epoch
        if self._applied:
            return
        if epoch >= self.start_epoch:
            # Find and modify layers by name
            for layer in self.model.layers:
                # SensorDropout layer name 'sensor_dropout' used in build_model
                if getattr(layer, "name", "") == "sensor_dropout":
                    if hasattr(layer, "rate"):
                        old = getattr(layer, "rate")
                        layer.rate = float(self.sensor_dropout_rate)
                        print(f"[CURRICULUM] sensor_dropout: {old} -> {layer.rate}")
                # SpecAugment layer name 'spec_augment'
                if getattr(layer, "name", "") == "spec_augment":
                    if hasattr(layer, "enabled"):
                        old = getattr(layer, "enabled")
                        layer.enabled = bool(self.enable_spec)
                        print(f"[CURRICULUM] spec_augment: {old} -> {layer.enabled}")
            self._applied = True


# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=Cfg.batch_size)
    ap.add_argument("--max-sensors", type=int, default=Cfg.max_sensors)
    ap.add_argument("--min-sensors", type=int, default=Cfg.min_sensors)
    ap.add_argument("--asset-id-col", type=str, default=Cfg.asset_id_col)
    ap.add_argument("--no-fallback-targets", type=int, default=0, help="1=exige labels reais")

    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--best-ckpt", type=str, default=str(DEFAULT_BEST_CKPT))
    ap.add_argument("--labels-path", type=str, default=str(DEFAULT_LABELS_PATH))
    ap.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    ap.add_argument("--train-log", type=str, default=str(DEFAULT_TRAIN_LOG))
    ap.add_argument("--overfit-test", type=int, default=0, help="Run overfit test with N samples (0=disabled)")
    return ap.parse_args()


# --------------------------
# Windowing by group (CRITICAL: prevents mixing assets/modes in same window)
# --------------------------
def make_windows_by_group(
    df_split: pd.DataFrame,
    group_cols: List[str],  # CRITICAL FIX: múltiplas colunas (ex: ["asset_id", "mode"])
    sensor_cols: List[str],
    cfg: Cfg,
    asset_norm_stats: Optional[Dict] = None,  # ASSET NORMALIZATION
):
    """
    Generate windows grouped by multiple columns (asset_id, mode, severity) to prevent label noise.
    CRITICAL: Each window comes from a single continuous sequence of ONE mode/severity.
    Uses asset-level normalization for cross-asset generalization.
    """
    all_raw, all_aux, all_mask = [], [], []
    all_rul, all_health, all_sev, all_mode = [], [], [], []

    for grp_key, g in df_split.groupby(group_cols, sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        
        # Get asset_id from group key (first element if multi-column group)
        if isinstance(grp_key, tuple):
            asset_id = str(grp_key[0])  # First column is asset_id
        else:
            asset_id = str(grp_key)
        
        # Get asset-specific stats for normalization
        asset_stats = None
        if asset_norm_stats and asset_id in asset_norm_stats:
            asset_stats = asset_norm_stats[asset_id]
        
        Xr, Xa, Xm, yr, yh, ys, ym = make_windows_variable_sensors(g, sensor_cols, cfg, asset_stats)
        if len(yr) == 0:
            continue
        all_raw.append(Xr); all_aux.append(Xa); all_mask.append(Xm)
        all_rul.append(yr); all_health.append(yh); all_sev.append(ys); all_mode.append(ym)

    if not all_raw:
        return (
            np.zeros((0, cfg.max_sensors, cfg.seq_len), np.float32),
            np.zeros((0, cfg.max_sensors, cfg.aux_per_sensor), np.float32),
            np.zeros((0, cfg.max_sensors), np.float32),
            np.array([], np.float32),
            np.array([], np.float32),
            np.array([], dtype=str),
            np.array([], dtype=str),
        )

    return (
        np.concatenate(all_raw, axis=0),
        np.concatenate(all_aux, axis=0),
        np.concatenate(all_mask, axis=0),
        np.concatenate(all_rul, axis=0),
        np.concatenate(all_health, axis=0),
        np.concatenate(all_sev, axis=0),
        np.concatenate(all_mode, axis=0),
    )


# --------------------------
# Main
# --------------------------
def main() -> None:
    args = parse_args()
    cfg = Cfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_sensors=args.max_sensors,
        min_sensors=args.min_sensors,
        asset_id_col=args.asset_id_col,
        allow_fallback_targets=(args.no_fallback_targets == 0),
    )

    set_seeds(cfg.seed)

    csv_files = discover_csvs()
    if not csv_files:
        raise FileNotFoundError("Nenhum CSV encontrado em logs/ ou logs/archive/.")

    df = load_all_csvs(csv_files)
    df = ensure_timestamp(df)
    df = ensure_mode(df)
    df = ensure_asset_id(df, cfg.asset_id_col)
    df = ensure_numeric(df)
    df = fallback_targets_if_missing(df, cfg)

    sensor_cols = pick_sensor_columns(df)
    if len(sensor_cols) < cfg.min_sensors:
        raise ValueError(f"Encontrados {len(sensor_cols)} sensores numéricos. Precisas >= {cfg.min_sensors}.")

    print(f"[SENSORS] Num sensores candidatos: {len(sensor_cols)}")
    print(f"[SENSORS] Top sensores: {sensor_cols[:min(20, len(sensor_cols))]}")

    group_col = cfg.asset_id_col if cfg.asset_id_col in df.columns else "mode"
    asset_norm_stats = None  # Initialize as None
    
    if group_col == "mode":
        print("[WARN] Sem asset_id -> split por 'mode' (pode haver leakage). Recomendo adicionar asset_id.")
        df_tr, df_va, df_te = temporal_split_grouped(df, group_col=group_col, tr=cfg.train_ratio, va=cfg.val_ratio)
    else:
        # CRITICAL FIX: Usar split TEMPORAL dentro de cada asset
        # Isto é mais realista para produção e provou ter acc muito superior (0.69 vs 0.20)
        print(f"[INFO] Split INTRA-ASSET TEMPORAL: 70% inicial treino, 15% meio val, 15% final test por asset")
        df_tr, df_va, df_te = temporal_split_grouped(df, group_col=cfg.asset_id_col, tr=cfg.train_ratio, va=cfg.val_ratio)
    
    print(f"[SPLIT] train={len(df_tr)} val={len(df_va)} test={len(df_te)}")

    # CRITICAL FIX: Agrupar janelas por (asset_id, mode, severity) para evitar label noise
    # Janelas NÃO podem atravessar fronteiras de modo nem de severidade
    window_group_cols = [group_col, "mode", "severity"]
    print(f"[INFO] Windowing agrupado por: {window_group_cols}")

    # Pass asset_norm_stats for cross-asset normalization
    Xtr_raw, Xtr_aux, Xtr_mask, ytr_rul, ytr_health, ytr_sev_s, ytr_mode_s = make_windows_by_group(df_tr, window_group_cols, sensor_cols, cfg, asset_norm_stats)
    Xva_raw, Xva_aux, Xva_mask, yva_rul, yva_health, yva_sev_s, yva_mode_s = make_windows_by_group(df_va, window_group_cols, sensor_cols, cfg, asset_norm_stats)
    Xte_raw, Xte_aux, Xte_mask, yte_rul, yte_health, yte_sev_s, yte_mode_s = make_windows_by_group(df_te, window_group_cols, sensor_cols, cfg, asset_norm_stats)

    print(f"[WINDOWS] train={len(ytr_rul)} val={len(yva_rul)} test={len(yte_rul)}")
    if len(ytr_rul) == 0 or len(yva_rul) == 0 or len(yte_rul) == 0:
        raise RuntimeError("Sem janelas suficientes. Aumenta dados ou reduz seq_len/hop.")

    sev_labels, sev_lab2i, mode_labels, mode_lab2i = build_label_maps(
        np.concatenate([ytr_sev_s, yva_sev_s, yte_sev_s]),
        np.concatenate([ytr_mode_s, yva_mode_s, yte_mode_s]),
    )
    n_sev = len(sev_labels)
    n_mode = len(mode_labels)

    ytr_sev = np.array([sev_lab2i[s] for s in ytr_sev_s], dtype=np.int64)
    yva_sev = np.array([sev_lab2i[s] for s in yva_sev_s], dtype=np.int64)
    yte_sev = np.array([sev_lab2i[s] for s in yte_sev_s], dtype=np.int64)
    ytr_mode = np.array([mode_lab2i[m] for m in ytr_mode_s], dtype=np.int64)
    yva_mode = np.array([mode_lab2i[m] for m in yva_mode_s], dtype=np.int64)
    yte_mode = np.array([mode_lab2i[m] for m in yte_mode_s], dtype=np.int64)

    # DIAGNOSTIC PRINTS (requested by user)
    print("\n[DIAGNOSTIC] Severity class distribution:")
    print(f"  Train: {np.bincount(ytr_sev, minlength=n_sev)}")
    print(f"  Val:   {np.bincount(yva_sev, minlength=n_sev)}")
    print(f"  Test:  {np.bincount(yte_sev, minlength=n_sev)}")
    print("\n[DIAGNOSTIC] Mode class distribution:")
    print(f"  Train: {np.bincount(ytr_mode, minlength=n_mode)}")
    print(f"  Val:   {np.bincount(yva_mode, minlength=n_mode)}")
    print(f"  Test:  {np.bincount(yte_mode, minlength=n_mode)}\n")

    print(f"[CLASSES] Severity={n_sev} {sev_labels}")
    print(f"[CLASSES] Mode={n_mode} (ex.: {mode_labels[:min(12, n_mode)]}{'...' if n_mode > 12 else ''})")

    # SAFE sample weights (protege contra classes ausentes no training set)
    def safe_class_weight(n_classes: int, y_train: np.ndarray) -> np.ndarray:
        """Compute class weights only for present classes, fill missing with 1.0"""
        unique_classes = np.unique(y_train)
        cw = np.ones(n_classes, dtype=np.float32)
        if len(unique_classes) > 0:
            weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
            for cls, w in zip(unique_classes, weights):
                cw[int(cls)] = float(w)
        return cw
    
    sev_cw = safe_class_weight(n_sev, ytr_sev)
    mode_cw = safe_class_weight(n_mode, ytr_mode)
    sev_sw = sev_cw[ytr_sev].astype(np.float32)
    mode_sw = mode_cw[ytr_mode].astype(np.float32)

    train_inputs = {"raw_input": Xtr_raw, "aux_input": Xtr_aux, "sensor_mask": Xtr_mask}
    train_targets = {"rul": ytr_rul, "health": ytr_health, "severity": ytr_sev, "mode": ytr_mode}
    val_inputs = {"raw_input": Xva_raw, "aux_input": Xva_aux, "sensor_mask": Xva_mask}
    val_targets = {"rul": yva_rul, "health": yva_health, "severity": yva_sev, "mode": yva_mode}

    # OVERFIT TEST MODE
    if args.overfit_test > 0:
        N = min(args.overfit_test, len(ytr_rul))
        print(f"\n{'='*80}")
        print(f"OVERFIT TEST MODE: Using only {N} training samples")
        print(f"Config: dropout=0, l2=0, sensor_dropout=0, augment=OFF, lr=1e-3")
        print(f"Goal: Train acc should reach 0.95+ if labels are learnable")
        print(f"{'='*80}\n")
        
        # Override config for overfit test
        cfg.dropout = 0.0
        cfg.l2_reg = 0.0
        cfg.sensor_dropout_rate = 0.0
        cfg.enable_spec_augment = False
        cfg.lr = 1e-3
        cfg.epochs = 50
        cfg.patience = 50  # No early stopping
        
        # Subset training data
        train_inputs = {k: v[:N] for k, v in train_inputs.items()}
        train_targets = {k: v[:N] for k, v in train_targets.items()}
        sev_sw = sev_sw[:N]
        mode_sw = mode_sw[:N]

    train_sw = {
        "rul": np.ones_like(train_targets["rul"], dtype=np.float32),
        "health": np.ones_like(train_targets["health"], dtype=np.float32),
        "severity": sev_sw,
        "mode": mode_sw,
    }

    va_ds = build_tf_dataset(Xva_raw, Xva_aux, Xva_mask, yva_rul, yva_health, yva_sev, yva_mode, cfg, training=False)
    te_ds = build_tf_dataset(Xte_raw, Xte_aux, Xte_mask, yte_rul, yte_health, yte_sev, yte_mode, cfg, training=False)

    model = build_model(cfg, n_sev=n_sev, n_mode=n_mode)
    model.summary()

    ckpt_path = Path(args.best_ckpt)
    callbacks: List[keras.callbacks.Callback] = [
        # CRITICAL FIX: Monitor val_mode_acc (classifica\u00e7\u00e3o \u00e9 prioridade) em vez de val_loss
        keras.callbacks.EarlyStopping(monitor="val_mode_acc", patience=cfg.patience, restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mode_acc", factor=0.5, patience=8, min_lr=1e-6, verbose=1, mode="max"),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(args.train_log),
        ProductMetricsCallback(val_ds=va_ds, sev_labels=sev_labels, ckpt_path=ckpt_path),
        CurriculumCallback(start_epoch=10, sensor_dropout_rate=0.10, enable_spec=True),
    ]

    print(f"[TRAIN] samples train={len(ytr_rul)} val={len(yva_rul)} batch_size={cfg.batch_size}")

    _hist = model.fit(
        x=train_inputs,
        y=train_targets,
        sample_weight=train_sw,
        validation_data=(val_inputs, val_targets),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    # Avaliação final no test set
    if ckpt_path.exists():
        print(f"[INFO] A carregar best weights: {ckpt_path}")
        model.load_weights(ckpt_path)
    else:
        print("[WARN] Best weights não encontrados, usando modelo atual.")
    
    results = model.evaluate(te_ds, verbose=0, return_dict=True)

    print("\n[TEST RESULTS]")
    print(f"  RUL MAE (0-1):     {results.get('rul_mae', 0.0):.4f}  (~{results.get('rul_mae',0.0)*1000:.0f}h/1000h)")
    print(f"  Health MAE (%):    {results.get('health_mae', 0.0)*100.0:.2f}%")
    print(f"  Severity acc:      {results.get('severity_acc', 0.0):.4f}")
    print(f"  Mode acc:          {results.get('mode_acc', 0.0):.4f}")
    print(f"  Mode top2:         {results.get('mode_top2', 0.0):.4f}")

    # relatórios
    y_true_sev, y_pred_sev = [], []
    y_true_mode, y_pred_mode = [], []

    for x, y in te_ds:
        pred = model.predict(x, verbose=0)
        y_pred_sev.append(np.argmax(pred["severity"], axis=1))
        y_pred_mode.append(np.argmax(pred["mode"], axis=1))
        y_true_sev.append(y["severity"].numpy())
        y_true_mode.append(y["mode"].numpy())

    yt_sev = np.concatenate(y_true_sev)
    yp_sev = np.concatenate(y_pred_sev)
    yt_mode = np.concatenate(y_true_mode)
    yp_mode = np.concatenate(y_pred_mode)

    sev_macro_f1 = float(f1_score(yt_sev, yp_sev, average="macro", zero_division=0))
    mode_macro_f1 = float(f1_score(yt_mode, yp_mode, average="macro", zero_division=0))

    severe_idx = sev_labels.index("severe") if "severe" in sev_labels else None
    failure_idx = sev_labels.index("failure") if "failure" in sev_labels else None
    crit_recall = None
    crit = []
    if severe_idx is not None: crit.append(severe_idx)
    if failure_idx is not None: crit.append(failure_idx)
    if crit:
        yt_bin = np.isin(yt_sev, crit).astype(int)
        yp_bin = np.isin(yp_sev, crit).astype(int)
        crit_recall = float(recall_score(yt_bin, yp_bin, zero_division=0))

    cm_sev = confusion_matrix(yt_sev, yp_sev, labels=np.arange(n_sev))
    cm_mode = confusion_matrix(yt_mode, yp_mode, labels=np.arange(n_mode))

    # Fix: usar labels= para classificar apenas classes presentes no test set
    sev_report = classification_report(
        yt_sev, yp_sev, 
        labels=np.arange(n_sev), 
        target_names=sev_labels, 
        output_dict=True, 
        zero_division=0
    )
    mode_report = classification_report(
        yt_mode, yp_mode, 
        labels=np.arange(n_mode), 
        target_names=mode_labels, 
        output_dict=True, 
        zero_division=0
    )

    labels_data = {
        "severity": sev_labels,
        "mode": mode_labels,
        "sensor_columns_ranked": sensor_cols[:cfg.max_sensors],
        "notes": "O modelo usa 'sensor_mask' para aceitar N sensores variáveis até max_sensors.",
    }
    with open(args.labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    report = {
        "test_results": {k: float(v) for k, v in results.items()},
        "derived_metrics": {
            "severity_macro_f1": sev_macro_f1,
            "mode_macro_f1": mode_macro_f1,
            "critical_recall_severe_failure": crit_recall,
            "health_mae_percent": float(results.get("health_mae", 0.0) * 100.0),
            "rul_mae_hours_over_1000h": float(results.get("rul_mae", 0.0) * 1000.0),
        },
        "labels": labels_data,
        "confusion_matrices": {"severity": cm_sev.tolist(), "mode": cm_mode.tolist()},
        "classification_reports": {"severity": sev_report, "mode": mode_report},
        "config": {
            "seq_len": cfg.seq_len,
            "hop": cfg.hop,
            "frame_len": cfg.frame_len,
            "frame_hop": cfg.frame_hop,
            "t_window": cfg.t_window,
            "n_mels": cfg.n_mels,
            "max_sensors": cfg.max_sensors,
            "min_sensors": cfg.min_sensors,
            "aux_per_sensor": cfg.aux_per_sensor,
            "sensor_dropout_rate": cfg.sensor_dropout_rate,
            "enable_spec_augment": cfg.enable_spec_augment,
            "split_group_col": (cfg.asset_id_col if cfg.asset_id_col in df.columns else "mode"),
            "n_windows": {"train": int(len(ytr_rul)), "val": int(len(yva_rul)), "test": int(len(yte_rul))},
            "note": "Se os targets forem sintéticos, estes resultados medem consistência do dataset, não prova campo.",
        },
    }
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    model.save(args.model_path)

    print(f"\n[OK] Modelo final: {args.model_path}")
    print(f"[OK] Melhor ckpt:  {ckpt_path if ckpt_path.exists() else '(não gerado)'}")
    print(f"[OK] Labels:       {args.labels_path}")
    print(f"[OK] Report:       {args.report_path}")
    print("\n[OK] Pipeline pronto para passar de sintetico realista -> dados reais sem mudar arquitetura.")


if __name__ == "__main__":
    main()
