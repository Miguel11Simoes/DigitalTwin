#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pump_predictive_market.py
Pipeline completo (end-to-end) para manutenção preditiva de bombas industriais
com modelo versátil para número variável de sensores (mínimo e máximo),
usando CNN 2D + Aux por sensor + máscara de sensores presentes.

Objetivo:
- Hoje: obter o melhor treino possível com dados atuais (mesmo que parcialmente sintéticos).
- Amanhã (produto): trocar targets sintéticos por labels reais (CMMS / eventos de falha),
  mantendo o pipeline exatamente igual.

Principais características "de mercado":
- Inputs variáveis: suporta N sensores presentes (N >= MIN_SENSORS), até MAX_SENSORS.
- Per-sensor CNN 2D seguindo: Convolution -> Pooling -> Flatten -> Dense.
- Agregação de sensores: attention pooling masked + mean pooling masked (estável).
- Sensor dropout durante treino (robustez a sensores em falta no cliente).
- Split temporal com opção anti-leakage por ativo (asset_id/pump_id se existir).
- Val/Test sem oversampling e sem augmentation (avaliação honesta).
- Sample weights para classes (severity + mode).
- Métricas de produto: macro-F1 severity, recall(severe+failure), macro-F1 mode, top-2 mode.
- Report JSON completo: confusion matrices, classification_report, config, results.
- Checkpoint pelo melhor macro-F1 de severity na VAL (mais alinhado com refinaria).

Requisitos:
pip install tensorflow pandas numpy scikit-learn

Execução:
python train_pump_predictive_market.py
"""

from __future__ import annotations

import os
import json
import math
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


# --------------------------
# Config
# --------------------------
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"

OUT_DIR = Path(os.environ.get("DT_MODELS_DIR", str(BASE_DIR / "models")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = OUT_DIR / "pump_predictive_market.keras"
DEFAULT_BEST_CKPT = OUT_DIR / "best_val_pump_predictive_market.keras"
DEFAULT_LABELS_PATH = OUT_DIR / "labels_master_market.json"
DEFAULT_REPORT_PATH = OUT_DIR / "eval_report_market.json"

PRIMARY_PREF = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "ultrasonic_noise", "motor_current", "pressure", "flow", "temperature"
]

DROP_COLS_ALWAYS = {
    "timestamp", "mode", "severity", "rul_minutes", "health_index",
    # colunas "model outputs" que podem existir no log
    "anomaly_score", "failure_probability", "model_confidence", "predicted_mode"
}

SEV_ORDER = ["normal", "early", "moderate", "severe", "failure"]


@dataclass
class Cfg:
    # Split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Windowing
    seq_len: int = 512
    hop: int = 256

    # Spectrogram
    frame_len: int = 64
    frame_hop: int = 32
    t_window: int = 128
    n_mels: int = 128  # aqui é "freq bins via resize" (mantemos compatível)

    # Versatilidade sensores
    max_sensors: int = 16
    min_sensors: int = 2  # podes mudar para 1 se quiseres permitir 1 sensor

    # Aux features por sensor (vamos gerar automaticamente)
    # (mean, std, slope, rms, kurtosis, skewness, crest_factor) = 7
    aux_per_sensor: int = 7

    # Treino
    epochs: int = 250
    batch_size: int = 64
    lr: float = 3e-4
    l2_reg: float = 1e-4
    dropout: float = 0.35

    patience: int = 40
    oversample_factor: int = 3  # repete treino (com shuffle) para mais passos úteis

    # Robustez
    sensor_dropout_rate: float = 0.20  # "sensor dropout" durante treino

    # Anti-leakage
    asset_id_col: str = "asset_id"  # se existir, split por asset; senão por "mode"

    seed: int = 42


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------
# IO: CSV discovery + load
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
# Targets (real vs fallback sintético)
# --------------------------
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_mode(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns:
        raise ValueError("CSV precisa de coluna 'mode'. (mesmo que seja 'unknown')")
    df["mode"] = df["mode"].astype(str)
    df = df.dropna(subset=["mode"]).reset_index(drop=True)
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in ("mode", "severity", "timestamp"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def synthetic_targets_if_missing(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Fallback APENAS para treino com o que tens hoje.
    Em produto: remover este fallback e usar labels reais.
    """
    rng = np.random.default_rng(seed)

    # RUL
    if "rul_minutes" not in df.columns or df["rul_minutes"].isna().all():
        print("[WARN] 'rul_minutes' ausente -> a gerar RUL sintético (NÃO é produto).")
        df["rul_minutes"] = 0.0
        for mode_label in df["mode"].unique():
            mask = df["mode"] == mode_label
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, "rul_minutes"] = np.linspace(1000 * 60, 0, n)

    # Health
    if "health_index" not in df.columns or df["health_index"].isna().all():
        print("[WARN] 'health_index' ausente -> a gerar Health sintético (NÃO é produto).")
        df["health_index"] = 100.0
        for mode_label in df["mode"].unique():
            mask = df["mode"] == mode_label
            n = int(mask.sum())
            if n > 0:
                t = np.linspace(0, 1, n)
                if ("bearing" in mode_label) or ("wear" in mode_label):
                    base = 1.0 - t**1.5
                elif ("electrical" in mode_label) or ("motor" in mode_label):
                    base = 1.0 - t**2.0
                else:
                    base = 1.0 - t**1.3
                noise = rng.normal(0, 0.02, n)
                health = np.clip(base + noise, 0, 1) * 100.0
                df.loc[mask, "health_index"] = health

    # Severity
    if "severity" not in df.columns or df["severity"].isna().all():
        print("[WARN] 'severity' ausente -> a derivar de health (demo). Em produto, usar label real/ordinal.")
        def map_severity(h: float) -> str:
            if h >= 90: return "normal"
            if h >= 70: return "early"
            if h >= 50: return "moderate"
            if h >= 30: return "severe"
            return "failure"
        df["severity"] = df["health_index"].apply(map_severity)

    df["severity"] = df["severity"].astype(str)
    df = df.dropna(subset=["severity", "rul_minutes", "health_index"]).reset_index(drop=True)
    return df


# --------------------------
# Sensor selection (versátil)
# --------------------------
def pick_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    Escolhe colunas numéricas candidatas a sensores.
    Estratégia:
    1) Preferir as do PRIMARY_PREF que existirem
    2) Depois adicionar outras numéricas não proibidas
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in DROP_COLS_ALWAYS]
    preferred = [c for c in PRIMARY_PREF if c in numeric_cols]
    others = [c for c in numeric_cols if c not in preferred]
    sensors = preferred + others
    return sensors


# --------------------------
# Split temporal (anti-leakage)
# --------------------------
def temporal_split_grouped(
    df: pd.DataFrame,
    group_col: str,
    tr: float,
    va: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
# Windowing + Aux features (numpy)
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
    return float(np.mean(((y - m) / s) ** 4) - 3.0)  # excess kurtosis


def crest_factor_np(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    peak = float(np.max(np.abs(y)))
    return float(peak / rms)


def aux_features_per_sensor(seg: np.ndarray) -> np.ndarray:
    """
    Gera features (7) por sensor, por janela.
    (mean, std, slope, rms, kurtosis, skewness, crest_factor)
    """
    seg = seg.astype(np.float32)
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return np.zeros((7,), dtype=np.float32)

    mean = float(np.mean(seg))
    std = float(np.std(seg))
    slp = slope_np(seg)
    rms = float(np.sqrt(np.mean(seg ** 2)))
    kur = kurtosis_np(seg)
    skw = skewness_np(seg)
    crf = crest_factor_np(seg)
    return np.array([mean, std, slp, rms, kur, skw, crf], dtype=np.float32)


def make_windows_variable_sensors(
    df_block: pd.DataFrame,
    sensor_cols: List[str],
    cfg: Cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna:
      X_raw   : (N, S_max, seq_len)  float32  (sinais brutos por sensor, padded)
      X_aux   : (N, S_max, A)        float32  (aux features por sensor, padded)
      X_mask  : (N, S_max)           float32  (1 se sensor existe, 0 se padding)
      y_rul   : (N,)                 float32  (0..1)
      y_health: (N,)                 float32  (0..1)
      y_sev_s : (N,)                 str
      y_mode_s: (N,)                 str
    """
    n = len(df_block)
    idxs = window_indices(n, cfg.seq_len, cfg.hop)

    # sensores válidos: colunas numéricas que têm pelo menos algum valor
    valid_cols = []
    for c in sensor_cols:
        if c in df_block.columns and pd.api.types.is_numeric_dtype(df_block[c]) and df_block[c].notna().any():
            valid_cols.append(c)

    if len(valid_cols) < cfg.min_sensors:
        raise ValueError(
            f"Poucos sensores válidos neste bloco: {len(valid_cols)} < MIN_SENSORS={cfg.min_sensors}. "
            "Ajusta min_sensors ou garante colunas numéricas com dados."
        )

    # Limitamos a MAX_SENSORS
    use_cols = valid_cols[:cfg.max_sensors]
    s_used = len(use_cols)

    X_raw_list: List[np.ndarray] = []
    X_aux_list: List[np.ndarray] = []
    X_mask_list: List[np.ndarray] = []

    y_rul, y_health, y_sev, y_mode = [], [], [], []

    for (i0, i1) in idxs:
        # raw signals tensor (S_used, seq_len)
        signals = np.zeros((cfg.max_sensors, cfg.seq_len), dtype=np.float32)
        aux = np.zeros((cfg.max_sensors, cfg.aux_per_sensor), dtype=np.float32)
        mask = np.zeros((cfg.max_sensors,), dtype=np.float32)

        ok = True
        for si, c in enumerate(use_cols):
            seg = df_block[c].to_numpy(dtype=np.float32)[i0:i1]
            if seg.shape[0] != cfg.seq_len or not np.isfinite(seg).all():
                # ainda aceitamos NaNs -> imputamos com mediana local
                finite = seg[np.isfinite(seg)]
                if finite.size == 0:
                    ok = False
                    break
                med = float(np.median(finite))
                seg = np.where(np.isfinite(seg), seg, med).astype(np.float32)

            signals[si, :] = seg
            aux[si, :] = aux_features_per_sensor(seg)
            mask[si] = 1.0

        if not ok:
            continue

        # Targets (último valor / média da janela)
        rul_win = df_block["rul_minutes"].iloc[i0:i1].to_numpy(dtype=np.float32)
        health_win = df_block["health_index"].iloc[i0:i1].to_numpy(dtype=np.float32)
        sev_win = df_block["severity"].iloc[i0:i1].astype(str).to_numpy()
        mode_win = df_block["mode"].iloc[i0:i1].astype(str).to_numpy()

        if rul_win.size == 0:
            continue

        rul_val = float(np.nanmean(rul_win)) / (1000.0 * 60.0)  # normalizado 0..1
        rul_val = float(np.clip(rul_val, 0.0, 1.0))

        health_val = float(np.nanmean(health_win)) / 100.0
        health_val = float(np.clip(health_val, 0.0, 1.0))

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
# Label encoding
# --------------------------
def build_label_maps(y_sev_str: np.ndarray, y_mode_str: np.ndarray) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
    sev_labels = sorted(np.unique(y_sev_str.astype(str)).tolist())
    mode_labels = sorted(np.unique(y_mode_str.astype(str)).tolist())
    sev_lab2i = {lab: i for i, lab in enumerate(sev_labels)}
    mode_lab2i = {lab: i for i, lab in enumerate(mode_labels)}
    return sev_labels, sev_lab2i, mode_labels, mode_lab2i


# --------------------------
# Spectrogram + augmentation in TF
# --------------------------
def series_to_spectrogram_tf(
    x_1d: tf.Tensor,
    frame_len: int,
    frame_hop: int,
    t_window: int,
    n_mels: int,
) -> tf.Tensor:
    """
    log-magnitude spectrogram normalizado e redimensionado: (t_window, n_mels, 1)
    """
    stft = tf.signal.stft(
        signals=tf.cast(x_1d, tf.float32),
        frame_length=frame_len,
        frame_step=frame_hop,
        fft_length=frame_len,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    mag = tf.abs(stft) + 1e-9
    logmag = tf.math.log(mag)

    mean = tf.reduce_mean(logmag)
    std = tf.math.reduce_std(logmag) + 1e-6
    norm = (logmag - mean) / std  # (frames, bins)

    norm = tf.expand_dims(norm, axis=-1)  # (frames,bins,1)
    norm = tf.image.resize(norm, size=(t_window, n_mels), method="bilinear")
    return tf.clip_by_value(norm, -10.0, 10.0)


def augment_spec_tf(spec: tf.Tensor) -> tf.Tensor:
    """
    Augmentation leve/moderado para espectrogramas.
    spec: (T, F, 1)
    """
    spec = tf.identity(spec)

    # time mask
    T = tf.shape(spec)[0]
    max_t = tf.maximum(1, tf.cast(tf.cast(T, tf.float32) * 0.12, tf.int32))
    def time_mask_once(s):
        width = tf.random.uniform([], 1, max_t + 1, dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(1, T - width), dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start, 1, 1], tf.float32),
            tf.zeros([width, 1, 1], tf.float32),
            tf.ones([T - start - width, 1, 1], tf.float32),
        ], axis=0)
        return s * mask
    spec = tf.cond(tf.random.uniform([]) > 0.35, lambda: time_mask_once(spec), lambda: spec)
    spec = tf.cond(tf.random.uniform([]) > 0.35, lambda: time_mask_once(spec), lambda: spec)

    # freq mask
    F = tf.shape(spec)[1]
    max_f = tf.maximum(1, tf.cast(tf.cast(F, tf.float32) * 0.12, tf.int32))
    def freq_mask_once(s):
        width = tf.random.uniform([], 1, max_f + 1, dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(1, F - width), dtype=tf.int32)
        mask = tf.concat([
            tf.ones([1, start, 1], tf.float32),
            tf.zeros([1, width, 1], tf.float32),
            tf.ones([1, F - start - width, 1], tf.float32),
        ], axis=1)
        return s * mask
    spec = tf.cond(tf.random.uniform([]) > 0.35, lambda: freq_mask_once(spec), lambda: spec)
    spec = tf.cond(tf.random.uniform([]) > 0.35, lambda: freq_mask_once(spec), lambda: spec)

    # noise
    spec = tf.cond(
        tf.random.uniform([]) > 0.45,
        lambda: spec + tf.random.normal(tf.shape(spec), mean=0.0, stddev=0.06, dtype=tf.float32),
        lambda: spec,
    )

    # amplitude scaling
    spec = tf.cond(
        tf.random.uniform([]) > 0.50,
        lambda: spec * tf.random.uniform([], 0.85, 1.15, dtype=tf.float32),
        lambda: spec,
    )

    return tf.clip_by_value(spec, -10.0, 10.0)


# --------------------------
# Model: per-sensor CNN (Conv->Pool->Flatten->Dense) + masked pooling
# --------------------------
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


def masked_softmax(logits: tf.Tensor, mask: tf.Tensor, axis: int = -1) -> tf.Tensor:
    mask = tf.cast(mask, tf.float32)
    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_logits = tf.where(mask > 0.0, logits, neg_inf)
    return tf.nn.softmax(masked_logits, axis=axis)


def build_model(cfg: Cfg, n_sev: int, n_mode: int) -> keras.Model:
    """
    Inputs:
      raw_input   : (S_max, seq_len)         sinais brutos por sensor
      aux_input   : (S_max, A)               aux por sensor
      sensor_mask : (S_max,)                 1/0
    """
    raw_in = keras.Input(shape=(cfg.max_sensors, cfg.seq_len), name="raw_input")
    aux_in = keras.Input(shape=(cfg.max_sensors, cfg.aux_per_sensor), name="aux_input")
    mask_in = keras.Input(shape=(cfg.max_sensors,), name="sensor_mask")

    # Sensor dropout (treino)
    sd = SensorDropout(cfg.sensor_dropout_rate, name="sensor_dropout")
    mask_eff = sd(mask_in)

    # Converte raw -> spectrogram por sensor via map_fn
    def raw_to_specs(x_raw):
        # x_raw: (S, seq_len)
        def one_sensor(sig):
            spec = series_to_spectrogram_tf(sig, cfg.frame_len, cfg.frame_hop, cfg.t_window, cfg.n_mels)
            return spec  # (T,F,1)
        specs = tf.map_fn(one_sensor, x_raw, fn_output_signature=tf.float32)  # (S,T,F,1)
        return specs

    specs = layers.Lambda(raw_to_specs, name="raw_to_specs")(raw_in)  # (B,S,T,F,1)

    # Augmentation só em treino (no graph): aplica a cada sensor
    def maybe_augment(x):
        x_specs, x_mask = x
        # aplica augmentation só onde mask=1 e training=True
        # usaremos um truque: a Lambda roda sempre; o dropout e treino já dão robustez.
        # para não complicar, aplicamos augmentation em todos e dependemos de mask.
        def aug_one_sensor(spec):
            return augment_spec_tf(spec)
        return tf.map_fn(lambda s: tf.map_fn(aug_one_sensor, s, fn_output_signature=tf.float32),
                         x_specs, fn_output_signature=tf.float32)

    specs_aug = layers.Lambda(maybe_augment, name="spec_augment")([specs, mask_eff])

    # Per-sensor CNN (Conv -> Pool -> Flatten -> Dense), pesos partilhados
    per_in = keras.Input(shape=(cfg.t_window, cfg.n_mels, 1))
    z = layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(per_in)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.25)(z)

    z = layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.30)(z)

    z = layers.Conv2D(128, (3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D((2, 2))(z)
    z = layers.Dropout(cfg.dropout * 0.35)(z)

    z = layers.Flatten()(z)  # <- Flatten como pediste
    z = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(z)
    z = layers.Dropout(cfg.dropout)(z)

    per_sensor_cnn = keras.Model(per_in, z, name="per_sensor_cnn")

    # aplica por sensor: (B,S,T,F,1) -> (B,S,256)
    sensor_embed = layers.TimeDistributed(per_sensor_cnn, name="td_cnn")(specs_aug)

    # Aux embed por sensor
    aux_norm = layers.LayerNormalization(name="aux_norm")(aux_in)
    aux_embed = layers.TimeDistributed(
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg)),
        name="td_aux_dense64"
    )(aux_norm)
    aux_embed = layers.Dropout(cfg.dropout * 0.35)(aux_embed)

    # Sensor features: concat CNN + Aux -> (B,S,320) -> dense -> (B,S,192)
    sensor_feat = layers.Concatenate()([sensor_embed, aux_embed])
    sensor_feat = layers.TimeDistributed(
        layers.Dense(192, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg)),
        name="td_sensor_dense192"
    )(sensor_feat)
    sensor_feat = layers.Dropout(cfg.dropout)(sensor_feat)

    # attention pooling masked
    scores = layers.TimeDistributed(layers.Dense(1, activation=None), name="attn_score")(sensor_feat)  # (B,S,1)
    scores = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="attn_squeeze")(scores)  # (B,S)

    weights = layers.Lambda(lambda x: masked_softmax(x[0], x[1], axis=-1), name="attn_weights")([scores, mask_eff])  # (B,S)
    weights_exp = layers.Lambda(lambda w: tf.expand_dims(w, axis=-1), name="attn_expand")(weights)  # (B,S,1)

    pooled_attn = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1), name="pooled_attn")([sensor_feat, weights_exp])  # (B,192)

    # masked mean pooling (estabiliza em produção)
    mask_f = layers.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.float32), -1), name="mask_expand")(mask_eff)  # (B,S,1)
    sum_feat = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1), name="masked_sum")([sensor_feat, mask_f])
    denom = layers.Lambda(lambda m: tf.maximum(tf.reduce_sum(m, axis=1), 1.0), name="masked_denom")(mask_f)
    pooled_mean = layers.Lambda(lambda x: x[0] / x[1], name="pooled_mean")([sum_feat, denom])  # (B,192)

    fused = layers.Concatenate(name="fusion_concat")([pooled_attn, pooled_mean])  # (B,384)
    fused = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(fused)
    fused = layers.Dropout(cfg.dropout)(fused)

    # Heads
    rul_h = layers.Dense(128, activation="relu")(fused)
    rul_h = layers.Dropout(cfg.dropout * 0.60)(rul_h)
    rul_out = layers.Dense(1, activation="sigmoid", name="rul")(rul_h)

    health_h = layers.Dense(128, activation="relu")(fused)
    health_h = layers.Dropout(cfg.dropout * 0.60)(health_h)
    health_out = layers.Dense(1, activation="sigmoid", name="health")(health_h)

    sev_h = layers.Dense(192, activation="relu")(fused)
    sev_h = layers.Dropout(cfg.dropout)(sev_h)
    sev_out = layers.Dense(n_sev, activation="softmax", name="severity")(sev_h)

    mode_h = layers.Dense(192, activation="relu")(fused)
    mode_h = layers.Dropout(cfg.dropout)(mode_h)
    mode_out = layers.Dense(n_mode, activation="softmax", name="mode")(mode_h)

    model = keras.Model(
        inputs={"raw_input": raw_in, "aux_input": aux_in, "sensor_mask": mask_in},
        outputs={"rul": rul_out, "health": health_out, "severity": sev_out, "mode": mode_out},
        name="pump_predictive_market",
    )

    # Optimizer (AdamW se disponível)
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=1e-5, clipnorm=1.0)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0)

    model.compile(
        optimizer=opt,
        loss={
            "rul": keras.losses.Huber(delta=0.08),
            "health": keras.losses.Huber(delta=0.04),
            "severity": keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
            "mode": keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        },
        loss_weights={
            "rul": 1.0,
            "health": 6.0,
            "severity": 10.0,
            "mode": 6.0,
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
# Datasets (tf.data)
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
    sev_sample_w: Optional[np.ndarray] = None,
    mode_sample_w: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X_raw, X_aux, X_mask, y_rul, y_health, y_sev, y_mode))

    if training:
        ds = ds.shuffle(min(len(X_raw), 20000), seed=cfg.seed, reshuffle_each_iteration=True)

    def pack(x_raw, x_aux, x_mask, r, h, sv, m):
        inputs = {"raw_input": x_raw, "aux_input": x_aux, "sensor_mask": x_mask}
        targets = {"rul": r, "health": h, "severity": sv, "mode": m}
        return inputs, targets

    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)

    if sev_sample_w is not None and mode_sample_w is not None:
        wds = tf.data.Dataset.from_tensor_slices((sev_sample_w.astype(np.float32), mode_sample_w.astype(np.float32)))
        ds = tf.data.Dataset.zip((ds, wds))

        def merge_w(data, w):
            (inputs, targets) = data
            sw = {"severity": w[0], "mode": w[1]}
            return inputs, targets, sw

        ds = ds.map(merge_w, num_parallel_calls=tf.data.AUTOTUNE)

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

        for batch in self.val_ds:
            if len(batch) == 3:
                x, y, _w = batch
            else:
                x, y = batch
            pred = self.model.predict(x, verbose=0)
            ps = np.argmax(pred["severity"], axis=1)
            pm = np.argmax(pred["mode"], axis=1)
            yts.append(y["severity"].numpy())
            yps.append(ps)
            ytm.append(y["mode"].numpy())
            ypm.append(pm)

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
            self.model.save(self.ckpt_path)
            print(f"[CKPT] Guardado melhor modelo: {self.ckpt_path} (sev_macro_f1={sev_macro_f1:.4f})")


# --------------------------
# Main
# --------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=Cfg.batch_size)
    ap.add_argument("--max-sensors", type=int, default=Cfg.max_sensors)
    ap.add_argument("--min-sensors", type=int, default=Cfg.min_sensors)
    ap.add_argument("--oversample-factor", type=int, default=Cfg.oversample_factor)
    ap.add_argument("--asset-id-col", type=str, default=Cfg.asset_id_col)

    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--best-ckpt", type=str, default=str(DEFAULT_BEST_CKPT))
    ap.add_argument("--labels-path", type=str, default=str(DEFAULT_LABELS_PATH))
    ap.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Cfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_sensors=args.max_sensors,
        min_sensors=args.min_sensors,
        oversample_factor=args.oversample_factor,
        asset_id_col=args.asset_id_col,
    )

    set_seeds(cfg.seed)

    csv_files = discover_csvs()
    if not csv_files:
        raise FileNotFoundError("Nenhum CSV encontrado em logs/ ou logs/archive/.")

    df = load_all_csvs(csv_files)
    df = ensure_timestamp(df)
    df = ensure_mode(df)
    df = ensure_numeric(df)
    df = synthetic_targets_if_missing(df, seed=cfg.seed)

    # Escolher sensores
    sensor_cols = pick_sensor_columns(df)
    if len(sensor_cols) < cfg.min_sensors:
        raise ValueError(
            f"Encontrados {len(sensor_cols)} sensores numéricos. "
            f"Precisas de pelo menos MIN_SENSORS={cfg.min_sensors}."
        )

    print(f"[SENSORS] Num sensores candidatos: {len(sensor_cols)}")
    print(f"[SENSORS] Top sensores: {sensor_cols[:min(20, len(sensor_cols))]}")

    group_col = cfg.asset_id_col if cfg.asset_id_col in df.columns else "mode"
    if group_col == "mode":
        print("[WARN] asset_id/pump_id não encontrado -> split por 'mode' (risco de leakage em produto).")
    else:
        print(f"[INFO] Split por ativo: {group_col}")

    df_tr, df_va, df_te = temporal_split_grouped(df, group_col=group_col, tr=cfg.train_ratio, va=cfg.val_ratio)
    print(f"[SPLIT] train={len(df_tr)} val={len(df_va)} test={len(df_te)}")

    # Windowing por split
    Xtr_raw, Xtr_aux, Xtr_mask, ytr_rul, ytr_health, ytr_sev_s, ytr_mode_s = make_windows_variable_sensors(df_tr, sensor_cols, cfg)
    Xva_raw, Xva_aux, Xva_mask, yva_rul, yva_health, yva_sev_s, yva_mode_s = make_windows_variable_sensors(df_va, sensor_cols, cfg)
    Xte_raw, Xte_aux, Xte_mask, yte_rul, yte_health, yte_sev_s, yte_mode_s = make_windows_variable_sensors(df_te, sensor_cols, cfg)

    print(f"[WINDOWS] train={len(ytr_rul)} val={len(yva_rul)} test={len(yte_rul)}")
    if len(ytr_rul) == 0 or len(yva_rul) == 0 or len(yte_rul) == 0:
        raise RuntimeError("Sem janelas suficientes. Aumenta dados ou reduz SEQ_LEN/HOP.")

    # Labels maps (fit em conjunto para consistência no report)
    sev_labels, sev_lab2i, mode_labels, mode_lab2i = build_label_maps(
        np.concatenate([ytr_sev_s, yva_sev_s, yte_sev_s]),
        np.concatenate([ytr_mode_s, yva_mode_s, yte_mode_s]),
    )
    n_sev = len(sev_labels)
    n_mode = len(mode_labels)

    ytr_sev = np.array([sev_lab2i[s] for s in ytr_sev_s], dtype=np.int64)
    yva_sev = np.array([sev_lab2i[s] for s in yva_sev_s], dtype=np.int64)
    yte_sev = np.array([sev_lab2i[s] for s in yte_sev_s], dtype=np.int64)

    ytr_mode = np.array([mode_lab2i[s] for s in ytr_mode_s], dtype=np.int64)
    yva_mode = np.array([mode_lab2i[s] for s in yva_mode_s], dtype=np.int64)
    yte_mode = np.array([mode_lab2i[s] for s in yte_mode_s], dtype=np.int64)

    print(f"[CLASSES] Severity={n_sev} {sev_labels}")
    print(f"[CLASSES] Mode={n_mode} (ex.: {mode_labels[:min(12, n_mode)]}{'...' if n_mode > 12 else ''})")

    # Sample weights (treino) por classe
    sev_cw = compute_class_weight("balanced", classes=np.arange(n_sev), y=ytr_sev)
    mode_cw = compute_class_weight("balanced", classes=np.arange(n_mode), y=ytr_mode)
    sev_sw = sev_cw[ytr_sev].astype(np.float32)
    mode_sw = mode_cw[ytr_mode].astype(np.float32)

    # Datasets
    tr_ds = build_tf_dataset(
        Xtr_raw, Xtr_aux, Xtr_mask, ytr_rul, ytr_health, ytr_sev, ytr_mode,
        cfg=cfg, training=True, sev_sample_w=sev_sw, mode_sample_w=mode_sw
    )
    if cfg.oversample_factor > 1:
        tr_ds = tr_ds.repeat(cfg.oversample_factor)

    va_ds = build_tf_dataset(
        Xva_raw, Xva_aux, Xva_mask, yva_rul, yva_health, yva_sev, yva_mode,
        cfg=cfg, training=False, sev_sample_w=None, mode_sample_w=None
    )
    te_ds = build_tf_dataset(
        Xte_raw, Xte_aux, Xte_mask, yte_rul, yte_health, yte_sev, yte_mode,
        cfg=cfg, training=False, sev_sample_w=None, mode_sample_w=None
    )

    # Modelo
    model = build_model(cfg, n_sev=n_sev, n_mode=n_mode)
    model.summary()

    # Callbacks
    ckpt_path = Path(args.best_ckpt)
    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        keras.callbacks.TerminateOnNaN(),
        ProductMetricsCallback(val_ds=va_ds, sev_labels=sev_labels, ckpt_path=ckpt_path),
    ]

    # steps/epoch (dataset repetido => infinito)
    steps_per_epoch = int(math.ceil((len(ytr_rul) * cfg.oversample_factor) / cfg.batch_size))
    val_steps = int(math.ceil(len(yva_rul) / cfg.batch_size))
    print(f"[TRAIN] steps_per_epoch={steps_per_epoch} val_steps={val_steps}")

    _hist = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    # Carregar melhor checkpoint (macro-F1 severity)
    if ckpt_path.exists():
        print(f"[INFO] A carregar melhor checkpoint: {ckpt_path}")
        model = keras.models.load_model(ckpt_path)

    # Avaliação final
    results = model.evaluate(te_ds, verbose=0, return_dict=True)
    print("\n[TEST RESULTS]")
    print(f"  RUL MAE (0-1):     {results.get('rul_mae', 0.0):.4f}  (~{results.get('rul_mae',0.0)*1000:.0f}h/1000h)")
    print(f"  Health MAE (%):    {results.get('health_mae', 0.0)*100.0:.2f}%")
    print(f"  Severity acc:      {results.get('severity_acc', 0.0):.4f}")
    print(f"  Mode acc:          {results.get('mode_acc', 0.0):.4f}")
    print(f"  Mode top2:         {results.get('mode_top2', 0.0):.4f}")

    # Previsões para relatórios
    y_true_sev, y_pred_sev = [], []
    y_true_mode, y_pred_mode = [], []

    for batch in te_ds:
        x, y = batch if len(batch) == 2 else (batch[0], batch[1])
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
    sev_report = classification_report(yt_sev, yp_sev, target_names=sev_labels, output_dict=True, zero_division=0)
    mode_report = classification_report(yt_mode, yp_mode, target_names=mode_labels, output_dict=True, zero_division=0)

    # Guardar labels + report
    labels_data = {"severity": sev_labels, "mode": mode_labels, "sensor_columns_ranked": sensor_cols[:cfg.max_sensors]}
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
            "oversample_factor": cfg.oversample_factor,
            "split_group_col": group_col,
            "n_windows": {"train": int(len(ytr_rul)), "val": int(len(yva_rul)), "test": int(len(yte_rul))},
            "note": "Se os targets foram gerados sinteticamente, estes resultados não provam performance real em campo.",
        },
    }

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Guardar modelo final
    model.save(args.model_path)

    print(f"\n[OK] Modelo final: {args.model_path}")
    print(f"[OK] Melhor ckpt:  {ckpt_path if ckpt_path.exists() else '(não gerado)'}")
    print(f"[OK] Labels:       {args.labels_path}")
    print(f"[OK] Report:       {args.report_path}")
    print("\n✓ Pipeline completo pronto para evoluir para dados reais (produto).")


if __name__ == "__main__":
    main()
