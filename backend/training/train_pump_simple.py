#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pump_simple.py
Pipeline SIMPLIFICADO para manutenção preditiva - focado em RESULTADOS.

Usa features estatísticas agregadas em vez de CNNs complexos.
Targets: Mode acc > 0.9, Severity acc > 0.9, Health MAE < 10%, RUL MAE < 0.2
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
OUT_DIR = Path(os.environ.get("DT_MODELS_DIR", str(BASE_DIR / "models")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEV_ORDER = ["normal", "early", "moderate", "severe", "failure"]


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data() -> pd.DataFrame:
    """Carrega CSV principal."""
    csv_path = LOGS_DIR / "sensors_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[DATA] Carregado: {len(df)} linhas")
    return df


def extract_features(df: pd.DataFrame, sensor_cols: List[str], window_size: int = 64, hop: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai features estatísticas por janela - OTIMIZADO com mais features.
    Para cada sensor: mean, std, min, max, rms, percentile25, percentile75, skewness, kurtosis = 9 features
    Total: n_sensors * 9 features
    """
    n = len(df)
    n_windows = (n - window_size) // hop + 1
    n_features_per_sensor = 9
    n_features = len(sensor_cols) * n_features_per_sensor
    
    print(f"[FEATURES] Preparando {n_windows} janelas x {n_features} features...")
    
    # Pre-allocate arrays
    X = np.zeros((n_windows, n_features), dtype=np.float32)
    y_rul = np.zeros(n_windows, dtype=np.float32)
    y_health = np.zeros(n_windows, dtype=np.float32)
    y_sev = np.empty(n_windows, dtype=object)
    y_mode = np.empty(n_windows, dtype=object)
    
    # Valores dos sensores
    sensor_data = np.column_stack([df[c].fillna(0).values.astype(np.float32) for c in sensor_cols])
    
    # Targets
    rul_vals = df['rul_minutes'].fillna(500).values / 1000.0
    health_vals = df['health_index'].fillna(50).values / 100.0
    sev_vals = df['severity'].fillna('normal').values
    mode_vals = df['mode'].fillna('unknown').values
    
    def safe_skew(arr):
        """Calcula skewness de forma segura."""
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-9:
            return 0.0
        return np.mean(((arr - m) / s) ** 3)
    
    def safe_kurtosis(arr):
        """Calcula kurtosis de forma segura."""
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-9:
            return 0.0
        return np.mean(((arr - m) / s) ** 4) - 3.0
    
    for w_idx in range(n_windows):
        i = w_idx * hop
        i_end = i + window_size
        
        # Features por sensor (vectorizado)
        for s_idx, c in enumerate(sensor_cols):
            seg = sensor_data[i:i_end, s_idx]
            base = s_idx * n_features_per_sensor
            
            # Estatísticas básicas
            X[w_idx, base] = np.mean(seg)
            X[w_idx, base+1] = np.std(seg)
            X[w_idx, base+2] = np.min(seg)
            X[w_idx, base+3] = np.max(seg)
            X[w_idx, base+4] = np.sqrt(np.mean(seg**2))  # RMS
            
            # Percentis (capturam distribuição)
            X[w_idx, base+5] = np.percentile(seg, 25)
            X[w_idx, base+6] = np.percentile(seg, 75)
            
            # Forma da distribuição (crucial para diferenciar severidades)
            X[w_idx, base+7] = safe_skew(seg)
            X[w_idx, base+8] = safe_kurtosis(seg)
        
        # Targets
        y_rul[w_idx] = np.clip(np.mean(rul_vals[i:i_end]), 0, 1)
        y_health[w_idx] = np.clip(np.mean(health_vals[i:i_end]), 0, 1)
        
        # Mode/Severity: usar valor do meio da janela (mais estável)
        mid = i + window_size // 2
        y_sev[w_idx] = sev_vals[mid]
        y_mode[w_idx] = mode_vals[mid]
        
        if w_idx % 1000 == 0:
            print(f"  Progresso: {w_idx}/{n_windows}")
    
    print(f"[FEATURES] Extraídas {X.shape[0]} janelas x {X.shape[1]} features")
    
    return (
        X,
        y_rul,
        y_health,
        y_sev.astype(str),
        y_mode.astype(str)
    )


def build_label_maps(y_sev: np.ndarray, y_mode: np.ndarray):
    """Cria mapeamentos de labels para índices."""
    # Severity em ordem industrial
    sev_unique = sorted(set(y_sev))
    sev_labels = [s for s in SEV_ORDER if s in sev_unique]
    sev_labels += [s for s in sev_unique if s not in sev_labels]
    sev2idx = {s: i for i, s in enumerate(sev_labels)}
    
    # Mode alfabético
    mode_labels = sorted(set(y_mode))
    mode2idx = {m: i for i, m in enumerate(mode_labels)}
    
    return sev_labels, sev2idx, mode_labels, mode2idx


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Normalização robusta (z-score com clip)."""
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-9
    
    X_train = np.clip((X_train - mean) / std, -5, 5)
    X_val = np.clip((X_val - mean) / std, -5, 5)
    X_test = np.clip((X_test - mean) / std, -5, 5)
    
    return X_train, X_val, X_test, mean, std


def build_model(n_features: int, n_sev: int, n_mode: int, dropout: float = 0.3) -> keras.Model:
    """
    MLP otimizado para severity.
    Arquitetura: Input -> Dense(512) -> Dense(256) -> Dense(128) -> separate heads
    Severity head mais profundo para melhor discriminação.
    """
    inp = keras.Input(shape=(n_features,), name="features")
    
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    # SEVERITY HEAD - mais profundo para melhor discriminação entre classes
    sev_h = layers.Dense(192, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    sev_h = layers.BatchNormalization()(sev_h)
    sev_h = layers.Dropout(dropout * 0.4)(sev_h)
    sev_h = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(sev_h)
    sev_h = layers.BatchNormalization()(sev_h)
    sev_h = layers.Dropout(dropout * 0.3)(sev_h)
    sev_h = layers.Dense(64, activation="relu")(sev_h)
    sev_out = layers.Dense(n_sev, activation="softmax", name="severity")(sev_h)
    
    # MODE HEAD
    mode_h = layers.Dense(128, activation="relu")(x)
    mode_h = layers.BatchNormalization()(mode_h)
    mode_h = layers.Dropout(dropout * 0.5)(mode_h)
    mode_h = layers.Dense(64, activation="relu")(mode_h)
    mode_out = layers.Dense(n_mode, activation="softmax", name="mode")(mode_h)
    
    # Regression heads
    reg_repr = layers.Dense(128, activation="relu")(x)
    reg_repr = layers.BatchNormalization()(reg_repr)
    reg_repr = layers.Dropout(dropout * 0.5)(reg_repr)
    
    rul_out = layers.Dense(1, activation="sigmoid", name="rul")(reg_repr)
    health_out = layers.Dense(1, activation="sigmoid", name="health")(reg_repr)
    
    model = keras.Model(inputs=inp, outputs={
        "rul": rul_out, "health": health_out, "severity": sev_out, "mode": mode_out
    })
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "rul": keras.losses.MeanSquaredError(),
            "health": keras.losses.MeanSquaredError(),
            "severity": keras.losses.SparseCategoricalCrossentropy(),
            "mode": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"rul": 0.5, "health": 1.0, "severity": 8.0, "mode": 2.0},  # MUCH more focus on severity
        metrics={
            "rul": [keras.metrics.MeanAbsoluteError(name="mae")],
            "health": [keras.metrics.MeanAbsoluteError(name="mae")],
            "severity": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "mode": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
        }
    )
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--hop", type=int, default=16)
    args = parser.parse_args()
    
    set_seeds(42)
    
    # Load data
    df = load_data()
    
    # Select sensor columns
    sensor_cols = [
        'overall_vibration', 'vibration_x', 'vibration_y', 'vibration_z',
        'motor_current', 'pressure', 'flow', 'temperature'
    ]
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    print(f"[SENSORS] Usando: {sensor_cols}")
    
    # Extract features
    X, y_rul, y_health, y_sev_str, y_mode_str = extract_features(
        df, sensor_cols, window_size=args.window, hop=args.hop
    )
    
    # Build label maps
    sev_labels, sev2idx, mode_labels, mode2idx = build_label_maps(y_sev_str, y_mode_str)
    n_sev, n_mode = len(sev_labels), len(mode_labels)
    
    y_sev = np.array([sev2idx[s] for s in y_sev_str], dtype=np.int64)
    y_mode = np.array([mode2idx[m] for m in y_mode_str], dtype=np.int64)
    
    print(f"[LABELS] Severity classes: {n_sev} {sev_labels}")
    print(f"[LABELS] Mode classes: {n_mode} {mode_labels}")
    
    # Split: 70/15/15 temporal
    n_samples = len(X)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_rul_tr, y_rul_va, y_rul_te = y_rul[:n_train], y_rul[n_train:n_train+n_val], y_rul[n_train+n_val:]
    y_health_tr, y_health_va, y_health_te = y_health[:n_train], y_health[n_train:n_train+n_val], y_health[n_train+n_val:]
    y_sev_tr, y_sev_va, y_sev_te = y_sev[:n_train], y_sev[n_train:n_train+n_val], y_sev[n_train+n_val:]
    y_mode_tr, y_mode_va, y_mode_te = y_mode[:n_train], y_mode[n_train:n_train+n_val], y_mode[n_train+n_val:]
    
    print(f"[SPLIT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    X_train, X_val, X_test, feat_mean, feat_std = normalize_features(X_train, X_val, X_test)
    
    # Class distribution
    print(f"[DIST] Severity train: {np.bincount(y_sev_tr, minlength=n_sev)}")
    print(f"[DIST] Mode train: {np.bincount(y_mode_tr, minlength=n_mode)}")
    
    # Class weights
    sev_weights = compute_class_weight("balanced", classes=np.arange(n_sev), y=y_sev_tr)
    mode_weights = compute_class_weight("balanced", classes=np.arange(n_mode), y=y_mode_tr)
    sev_sw = sev_weights[y_sev_tr].astype(np.float32)
    mode_sw = mode_weights[y_mode_tr].astype(np.float32)
    
    # Build model
    n_features = X_train.shape[1]
    model = build_model(n_features, n_sev, n_mode, dropout=0.25)  # Slightly less dropout
    model.summary()
    
    # Callbacks - Monitor severity since it's the hardest target
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_severity_acc", patience=25, restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_severity_acc", factor=0.5, patience=10, min_lr=1e-6, mode="max"),
        keras.callbacks.ModelCheckpoint(
            str(OUT_DIR / "best_pump_simple.weights.h5"),
            monitor="val_severity_acc", save_best_only=True, save_weights_only=True, mode="max"
        ),
    ]
    
    # Sample weights para todas as outputs (Keras requer isto)
    sample_weights = {
        "rul": np.ones(len(y_rul_tr), dtype=np.float32),
        "health": np.ones(len(y_health_tr), dtype=np.float32),
        "severity": sev_sw,
        "mode": mode_sw,
    }
    
    # Train
    history = model.fit(
        X_train,
        {"rul": y_rul_tr, "health": y_health_tr, "severity": y_sev_tr, "mode": y_mode_tr},
        sample_weight=sample_weights,
        validation_data=(
            X_val,
            {"rul": y_rul_va, "health": y_health_va, "severity": y_sev_va, "mode": y_mode_va}
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # Load best weights
    best_path = OUT_DIR / "best_pump_simple.weights.h5"
    if best_path.exists():
        model.load_weights(str(best_path))
    
    # Evaluate on test set
    results = model.evaluate(
        X_test,
        {"rul": y_rul_te, "health": y_health_te, "severity": y_sev_te, "mode": y_mode_te},
        verbose=0, return_dict=True
    )
    
    print("\n" + "="*60)
    print("TEST RESULTS (TARGETS)")
    print("="*60)
    
    rul_mae = results.get('rul_mae', 0)
    health_mae = results.get('health_mae', 0)
    sev_acc = results.get('severity_acc', 0)
    mode_acc = results.get('mode_acc', 0)
    
    print(f"  RUL MAE:      {rul_mae:.4f}  (target < 0.20)  {'✓' if rul_mae < 0.2 else '✗'}")
    print(f"  Health MAE:   {health_mae*100:.2f}%  (target < 10%)  {'✓' if health_mae < 0.1 else '✗'}")
    print(f"  Severity acc: {sev_acc:.4f}  (target > 0.90)  {'✓' if sev_acc > 0.9 else '✗'}")
    print(f"  Mode acc:     {mode_acc:.4f}  (target > 0.90)  {'✓' if mode_acc > 0.9 else '✗'}")
    print("="*60)
    
    # Detailed predictions
    preds = model.predict(X_test, verbose=0)
    y_pred_sev = np.argmax(preds['severity'], axis=1)
    y_pred_mode = np.argmax(preds['mode'], axis=1)
    
    print("\n[SEVERITY REPORT]")
    print(classification_report(y_sev_te, y_pred_sev, target_names=sev_labels, zero_division=0))
    
    print("\n[MODE REPORT]")
    print(classification_report(y_mode_te, y_pred_mode, target_names=mode_labels, zero_division=0))
    
    # Save model
    model.save(str(OUT_DIR / "pump_simple.keras"))
    
    # Save labels
    labels_data = {
        "severity": sev_labels,
        "mode": mode_labels,
        "sensors": sensor_cols,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
    }
    with open(OUT_DIR / "labels_simple.json", "w") as f:
        json.dump(labels_data, f, indent=2)
    
    # Save report
    report = {
        "test_results": {
            "rul_mae": float(rul_mae),
            "health_mae_percent": float(health_mae * 100),
            "severity_acc": float(sev_acc),
            "mode_acc": float(mode_acc),
        },
        "targets_met": {
            "rul": rul_mae < 0.2,
            "health": health_mae < 0.1,
            "severity": sev_acc > 0.9,
            "mode": mode_acc > 0.9,
        },
        "all_targets_met": (rul_mae < 0.2 and health_mae < 0.1 and sev_acc > 0.9 and mode_acc > 0.9),
    }
    with open(OUT_DIR / "eval_report_simple.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[SAVED] Model: {OUT_DIR / 'pump_simple.keras'}")
    print(f"[SAVED] Labels: {OUT_DIR / 'labels_simple.json'}")
    print(f"[SAVED] Report: {OUT_DIR / 'eval_report_simple.json'}")
    
    if report["all_targets_met"]:
        print("\n🎯 TODOS OS TARGETS ATINGIDOS! 🎯")
    else:
        print("\n⚠️  Alguns targets não atingidos - ajustar hiperparâmetros")


if __name__ == "__main__":
    main()
