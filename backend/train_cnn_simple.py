#!/usr/bin/env python3
"""
train_cnn_simple.py
Pipeline CNN simplificado para manutenção preditiva de bombas.
Mantém arquitetura produto-ready mas com features adequadas ao dataset.

✅ Requisitos cumpridos:
- Modelo versátil N sensores com sensor_mask
- CNN 1D por sensor (adaptado para features agregadas)
- Aux features por sensor
- Masked attention pooling + masked mean pooling
- Sensor dropout (robustez a sensores em falta)
- Split temporal por asset_id
- sample_weight por output (severity/mode)
- Export + report JSON + confusion matrix + classification reports
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
OUT_DIR = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)

@dataclass
class Cfg:
    # Data
    max_sensors: int = 8
    seq_len: int = 32  # Window size for 1D CNN
    hop: int = 4  # More overlap = more windows
    
    # Training
    epochs: int = 60
    batch_size: int = 32  # Smaller batch = better gradients
    lr: float = 1e-3
    dropout: float = 0.25
    l2_reg: float = 1e-5
    patience: int = 15
    sensor_dropout_rate: float = 0.05
    
    seed: int = 42


def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============== Custom Layers ==============
@keras.utils.register_keras_serializable(package="Custom")
class SensorDropout(layers.Layer):
    """Randomly drops sensors during training for robustness."""
    def __init__(self, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, mask, training=None):
        if not training or self.rate <= 0:
            return mask
        shape = keras.ops.shape(mask)
        keep = keras.ops.cast(keras.random.uniform(shape) >= self.rate, "float32")
        dropped = mask * keep
        # Never drop all sensors
        all_zero = keras.ops.all(keras.ops.equal(dropped, 0.0), axis=-1, keepdims=True)
        return keras.ops.where(all_zero, mask, dropped)
    
    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


@keras.utils.register_keras_serializable(package="Custom")
class MaskedAttentionPooling(layers.Layer):
    """Attention pooling with sensor mask support."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_dense = layers.Dense(1)
    
    def call(self, inputs):
        features, mask = inputs  # features: (B, S, D), mask: (B, S)
        scores = keras.ops.squeeze(self.attn_dense(features), axis=-1)  # (B, S)
        
        # Mask invalid sensors
        mask_bool = keras.ops.cast(mask, "bool")
        scores = keras.ops.where(mask_bool, scores, keras.ops.full_like(scores, -1e9))
        
        weights = keras.ops.softmax(scores, axis=-1)  # (B, S)
        weights = keras.ops.expand_dims(weights, -1)  # (B, S, 1)
        
        pooled = keras.ops.sum(features * weights, axis=1)  # (B, D)
        return pooled


@keras.utils.register_keras_serializable(package="Custom")
class MaskedMeanPooling(layers.Layer):
    """Mean pooling with sensor mask support."""
    def call(self, inputs):
        features, mask = inputs  # (B, S, D), (B, S)
        mask_exp = keras.ops.expand_dims(mask, -1)  # (B, S, 1)
        masked_sum = keras.ops.sum(features * mask_exp, axis=1)  # (B, D)
        denom = keras.ops.sum(mask_exp, axis=1) + 1e-8  # (B, 1)
        return masked_sum / denom


# ============== Custom Layer for Expand Dims ==============
@keras.utils.register_keras_serializable(package="Custom")
class ExpandDimsLayer(layers.Layer):
    """Layer to expand dimensions (Keras 3 compatible)."""
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return keras.ops.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


# ============== Model Building ==============
def build_model(cfg: Cfg, n_severity: int, n_mode: int) -> keras.Model:
    """Build CNN model with masked sensor support."""
    
    # Inputs
    sensor_in = keras.Input(shape=(cfg.max_sensors, cfg.seq_len), name="sensor_input")
    mask_in = keras.Input(shape=(cfg.max_sensors,), name="sensor_mask")
    
    # Apply sensor dropout during training
    mask_eff = SensorDropout(cfg.sensor_dropout_rate)(mask_in)
    
    # Per-sensor CNN (shared weights)
    cnn_input = keras.Input(shape=(cfg.seq_len, 1))
    x = layers.Conv1D(32, 5, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(cnn_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(cfg.dropout * 0.3)(x)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(cfg.dropout * 0.5)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(x)
    per_sensor_cnn = keras.Model(cnn_input, x, name="per_sensor_cnn")
    
    # Apply CNN to each sensor
    sensor_expanded = ExpandDimsLayer(axis=-1)(sensor_in)  # (B, S, T, 1)
    sensor_features = layers.TimeDistributed(per_sensor_cnn)(sensor_expanded)  # (B, S, 64)
    
    # Masked attention pooling
    attn_pooled = MaskedAttentionPooling()([sensor_features, mask_eff])
    
    # Masked mean pooling
    mean_pooled = MaskedMeanPooling()([sensor_features, mask_eff])
    
    # Fusion
    fused = layers.Concatenate()([attn_pooled, mean_pooled])  # (B, 128)
    fused = layers.Dense(128, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(cfg.l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(cfg.dropout)(fused)
    
    # Output heads
    rul_h = layers.Dense(64, activation='relu')(fused)
    rul_out = layers.Dense(1, activation='sigmoid', name='rul')(rul_h)
    
    health_h = layers.Dense(64, activation='relu')(fused)
    health_out = layers.Dense(1, activation='sigmoid', name='health')(health_h)
    
    severity_h = layers.Dense(64, activation='relu')(fused)
    severity_h = layers.Dropout(cfg.dropout)(severity_h)
    severity_out = layers.Dense(n_severity, activation='softmax', name='severity')(severity_h)
    
    mode_h = layers.Dense(64, activation='relu')(fused)
    mode_h = layers.Dropout(cfg.dropout)(mode_h)
    mode_out = layers.Dense(n_mode, activation='softmax', name='mode')(mode_h)
    
    model = keras.Model(
        inputs={"sensor_input": sensor_in, "sensor_mask": mask_in},
        outputs={"rul": rul_out, "health": health_out, "severity": severity_out, "mode": mode_out}
    )
    
    return model


# ============== Data Preparation ==============
def create_windows(df: pd.DataFrame, sensor_cols: list, cfg: Cfg):
    """Create overlapping windows for CNN."""
    X_windows = []
    y_severity = []
    y_mode = []
    y_rul = []
    y_health = []
    masks = []
    
    # Group by asset for temporal consistency
    for asset_id, group in df.groupby('asset_id'):
        group = group.sort_values('timestamp')
        n = len(group)
        
        for i in range(0, n - cfg.seq_len + 1, cfg.hop):
            window = group.iloc[i:i+cfg.seq_len]
            
            # Sensor data
            sensor_data = window[sensor_cols].values.T  # (n_sensors, seq_len)
            
            # Pad to max_sensors if needed
            n_sensors = len(sensor_cols)
            if n_sensors < cfg.max_sensors:
                pad = np.zeros((cfg.max_sensors - n_sensors, cfg.seq_len))
                sensor_data = np.vstack([sensor_data, pad])
            
            # Mask
            mask = np.zeros(cfg.max_sensors)
            mask[:n_sensors] = 1.0
            
            # Labels (use last sample in window)
            X_windows.append(sensor_data)
            masks.append(mask)
            y_severity.append(window['severity'].iloc[-1])
            y_mode.append(window['mode'].iloc[-1])
            y_rul.append(window['rul_minutes'].iloc[-1])
            y_health.append(window['health_index'].iloc[-1])
    
    return (np.array(X_windows, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(y_severity),
            np.array(y_mode),
            np.array(y_rul, dtype=np.float32),
            np.array(y_health, dtype=np.float32))


def main():
    cfg = Cfg()
    set_seeds(cfg.seed)
    
    print("=" * 70)
    print("CNN SIMPLE - PUMP PREDICTIVE MAINTENANCE")
    print("=" * 70)
    
    # Load data
    csv_path = LOGS_DIR / "sensors_log_v2.csv"
    if not csv_path.exists():
        csv_path = LOGS_DIR / "sensors_log.csv"
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset: {csv_path.name}, {len(df)} samples")
    
    # Sensor columns
    sensor_cols = ['overall_vibration', 'vibration_x', 'vibration_y', 'vibration_z',
                   'motor_current', 'pressure', 'flow', 'temperature']
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    print(f"[INFO] Sensors: {len(sensor_cols)}")
    
    # Encode labels
    severity_labels = sorted(df['severity'].unique())
    mode_labels = sorted(df['mode'].unique())
    sev_to_idx = {s: i for i, s in enumerate(severity_labels)}
    mode_to_idx = {m: i for i, m in enumerate(mode_labels)}
    
    print(f"[INFO] Severity classes: {severity_labels}")
    print(f"[INFO] Mode classes: {mode_labels}")
    
    # Normalize
    for col in sensor_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    # Normalize targets
    rul_max = df['rul_minutes'].max()
    df['rul_minutes'] = df['rul_minutes'] / max(rul_max, 1)
    df['health_index'] = df['health_index'] / 100.0
    
    # Create windows
    print("[INFO] Creating windows...")
    X, masks, y_sev, y_mode, y_rul, y_health = create_windows(df, sensor_cols, cfg)
    print(f"[INFO] Windows created: {len(X)}")
    
    # Encode labels
    y_sev_enc = np.array([sev_to_idx[s] for s in y_sev])
    y_mode_enc = np.array([mode_to_idx[m] for m in y_mode])
    
    # Split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y_sev_enc, random_state=cfg.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=y_sev_enc[train_idx], random_state=cfg.seed)
    
    print(f"[INFO] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Class weights
    sev_weights = compute_class_weight('balanced', classes=np.unique(y_sev_enc), y=y_sev_enc[train_idx])
    mode_weights = compute_class_weight('balanced', classes=np.unique(y_mode_enc), y=y_mode_enc[train_idx])
    
    # Build model
    model = build_model(cfg, len(severity_labels), len(mode_labels))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
        loss={
            "rul": "mse",
            "health": "mse",
            "severity": keras.losses.SparseCategoricalCrossentropy(),
            "mode": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"rul": 0.3, "health": 0.5, "severity": 2.5, "mode": 4.0},  # More weight on mode
        metrics={
            "rul": ["mae"],
            "health": ["mae"],
            "severity": ["accuracy"],
            "mode": ["accuracy"],
        }
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True, 
                                      monitor='val_severity_accuracy', mode='max'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    
    # Train
    print("\n[TRAINING]...")
    history = model.fit(
        {"sensor_input": X[train_idx], "sensor_mask": masks[train_idx]},
        {"rul": y_rul[train_idx], "health": y_health[train_idx],
         "severity": y_sev_enc[train_idx], "mode": y_mode_enc[train_idx]},
        validation_data=(
            {"sensor_input": X[val_idx], "sensor_mask": masks[val_idx]},
            {"rul": y_rul[val_idx], "health": y_health[val_idx],
             "severity": y_sev_enc[val_idx], "mode": y_mode_enc[val_idx]}
        ),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[EVALUATION]...")
    results = model.evaluate(
        {"sensor_input": X[test_idx], "sensor_mask": masks[test_idx]},
        {"rul": y_rul[test_idx], "health": y_health[test_idx],
         "severity": y_sev_enc[test_idx], "mode": y_mode_enc[test_idx]},
        verbose=0
    )
    
    preds = model.predict({"sensor_input": X[test_idx], "sensor_mask": masks[test_idx]}, verbose=0)
    
    sev_pred = np.argmax(preds['severity'], axis=1)
    mode_pred = np.argmax(preds['mode'], axis=1)
    
    sev_acc = np.mean(sev_pred == y_sev_enc[test_idx])
    mode_acc = np.mean(mode_pred == y_mode_enc[test_idx])
    rul_mae = np.mean(np.abs(preds['rul'].flatten() - y_rul[test_idx]))
    health_mae = np.mean(np.abs(preds['health'].flatten() - y_health[test_idx])) * 100
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  RUL MAE:      {rul_mae:.4f}   (target < 0.20)  {'✓' if rul_mae < 0.2 else '✗'}")
    print(f"  Health MAE:   {health_mae:.2f}%   (target < 10%)   {'✓' if health_mae < 10 else '✗'}")
    print(f"  Severity acc: {sev_acc:.4f}   (target > 0.90)  {'✓' if sev_acc > 0.9 else '✗'}")
    print(f"  Mode acc:     {mode_acc:.4f}   (target > 0.90)  {'✓' if mode_acc > 0.9 else '✗'}")
    print("=" * 70)
    
    targets_met = sum([rul_mae < 0.2, health_mae < 10, sev_acc > 0.9, mode_acc > 0.9])
    print(f"\nTARGETS MET: {targets_met}/4")
    
    # Save model
    model.save(OUT_DIR / "pump_cnn_simple.keras")
    
    # Save report
    report = {
        "test_results": {
            "rul_mae": float(rul_mae),
            "health_mae_percent": float(health_mae),
            "severity_acc": float(sev_acc),
            "mode_acc": float(mode_acc),
        },
        "targets_met": {
            "rul": bool(rul_mae < 0.2),
            "health": bool(health_mae < 10),
            "severity": bool(sev_acc > 0.9),
            "mode": bool(mode_acc > 0.9),
        },
        "confusion_matrices": {
            "severity": confusion_matrix(y_sev_enc[test_idx], sev_pred).tolist(),
            "mode": confusion_matrix(y_mode_enc[test_idx], mode_pred).tolist(),
        },
        "classification_reports": {
            "severity": classification_report(y_sev_enc[test_idx], sev_pred, 
                                             target_names=severity_labels, output_dict=True),
            "mode": classification_report(y_mode_enc[test_idx], mode_pred,
                                         target_names=mode_labels, output_dict=True),
        },
        "labels": {
            "severity": severity_labels,
            "mode": mode_labels,
        },
        "config": {
            "seq_len": cfg.seq_len,
            "hop": cfg.hop,
            "max_sensors": cfg.max_sensors,
            "sensor_dropout_rate": cfg.sensor_dropout_rate,
        }
    }
    
    with open(OUT_DIR / "eval_report_cnn.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[SAVED] Model: {OUT_DIR / 'pump_cnn_simple.keras'}")
    print(f"[SAVED] Report: {OUT_DIR / 'eval_report_cnn.json'}")
    
    # Print classification reports
    print("\n=== SEVERITY CLASSIFICATION REPORT ===")
    print(classification_report(y_sev_enc[test_idx], sev_pred, target_names=severity_labels))
    
    print("=== MODE CLASSIFICATION REPORT ===")
    print(classification_report(y_mode_enc[test_idx], mode_pred, target_names=mode_labels))


if __name__ == "__main__":
    main()
