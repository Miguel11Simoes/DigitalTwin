#!/usr/bin/env python3
"""
train_pump_cnn2d_v2.py
======================
Modelo CNN 2D v2 para classificação de bombas industriais.
Baseado na arquitetura de sucesso de train_cnn_2d.py com melhorias A1-A12.

TARGETS OBRIGATÓRIOS:
- RUL MAE < 20% (target: 0.20)
- Health MAE < 10%  
- Severity Accuracy >= 90%
- Mode Accuracy >= 90%

Author: Industrial Pump Digital Twin
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] TensorFlow: {tf.__version__}")
print(f"[INFO] Keras: {keras.__version__}")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ModelConfig:
    """Configuração do modelo CNN 2D v2."""
    
    # Input shapes
    max_sensors: int = 8
    seq_len: int = 64
    hop: int = 8
    n_freq: int = 16
    n_time: int = 16
    aux_dim: int = 8
    
    # Conv blocks
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    # Dense layers
    dense_units: int = 128
    dropout: float = 0.3
    sensor_dropout_rate: float = 0.15
    
    # Output classes
    n_severity: int = 5
    n_mode: int = 5
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    
    # Loss weights
    loss_weight_severity: float = 3.0
    loss_weight_mode: float = 2.5
    loss_weight_rul: float = 1.0
    loss_weight_health: float = 1.5
    
    # Targets
    target_severity_acc: float = 0.90
    target_mode_acc: float = 0.90
    target_rul_mae: float = 0.20
    target_health_mae: float = 0.10
    
    # Seed
    seed: int = 42


# ===========================================================================
# CUSTOM LAYERS
# ===========================================================================

@keras.utils.register_keras_serializable(package="PumpV2")
class SensorDropout(layers.Layer):
    """Randomly drops entire sensors during training."""
    
    def __init__(self, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        x, mask = inputs
        batch_size = tf.shape(x)[0]
        n_sensors = tf.shape(x)[1]
        
        # Random mask for dropping sensors
        drop_mask = tf.random.uniform((batch_size, n_sensors, 1, 1, 1)) > self.rate
        drop_mask = tf.cast(drop_mask, x.dtype)
        
        # Combine with existing mask
        mask_expanded = tf.reshape(mask, (batch_size, n_sensors, 1, 1, 1))
        combined_mask = drop_mask * mask_expanded
        
        return x * combined_mask, tf.squeeze(combined_mask, axis=[2, 3, 4])
    
    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


@keras.utils.register_keras_serializable(package="PumpV2")
class MaskedGlobalPooling(layers.Layer):
    """Global Average Pooling that respects sensor mask."""
    
    def call(self, inputs):
        x, mask = inputs
        # x: (batch, n_sensors, features)
        # mask: (batch, n_sensors)
        
        mask_expanded = tf.expand_dims(mask, -1)
        x_masked = x * mask_expanded
        
        sum_features = tf.reduce_sum(x_masked, axis=1)
        n_active = tf.reduce_sum(mask, axis=1, keepdims=True)
        n_active = tf.maximum(n_active, 1.0)
        
        return sum_features / n_active
    
    def get_config(self):
        return super().get_config()


# ===========================================================================
# DATA PIPELINE
# ===========================================================================

class DataPipeline:
    """Pipeline de dados para treino."""
    
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.severity_labels = []
        self.mode_labels = []
        self.sev_to_idx = {}
        self.mode_to_idx = {}
        self.rul_max_train = 1.0
    
    def load_data(self, path: Path) -> pd.DataFrame:
        """Load dataset."""
        df = pd.read_csv(path)
        print(f"[LOAD] {len(df)} samples from {path.name}")
        return df
    
    def split_by_asset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by asset (no leakage)."""
        assets = df['asset_id'].unique().tolist()
        np.random.seed(self.cfg.seed)
        np.random.shuffle(assets)
        
        n_train = int(0.7 * len(assets))
        n_val = int(0.15 * len(assets))
        
        train_assets = assets[:n_train]
        val_assets = assets[n_train:n_train + n_val]
        test_assets = assets[n_train + n_val:]
        
        train_df = df[df['asset_id'].isin(train_assets)]
        val_df = df[df['asset_id'].isin(val_assets)]
        test_df = df[df['asset_id'].isin(test_assets)]
        
        print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def create_windows(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        fit_scaler: bool = False
    ) -> Tuple:
        """Create windows with pseudo-spectrograms."""
        
        X_spec = []
        X_aux = []
        masks = []
        y_severity = []
        y_mode = []
        y_rul = []
        y_health = []
        
        # Normalize sensors
        if fit_scaler:
            self.scaler.fit(df[sensor_cols])
            self.rul_max_train = df['rul_minutes'].max()
        
        df_norm = df.copy()
        df_norm[sensor_cols] = self.scaler.transform(df[sensor_cols])
        df_norm['rul_norm'] = df['rul_minutes'] / max(self.rul_max_train, 1)
        df_norm['health_norm'] = df['health_index'] / 100.0
        
        # Process by asset
        for asset_id, group in df_norm.groupby('asset_id'):
            group = group.sort_values('timestamp')
            n = len(group)
            
            for i in range(0, n - self.cfg.seq_len + 1, self.cfg.hop):
                window = group.iloc[i:i + self.cfg.seq_len]
                
                # Sensor data
                sensor_data = window[sensor_cols].values.T  # (n_sensors, seq_len)
                
                # Pad to max_sensors
                n_sensors = len(sensor_cols)
                if n_sensors < self.cfg.max_sensors:
                    pad = np.zeros((self.cfg.max_sensors - n_sensors, self.cfg.seq_len))
                    sensor_data = np.vstack([sensor_data, pad])
                
                # Create pseudo-spectrogram
                spec = self._create_spectrogram(sensor_data)
                
                # Mask
                mask = np.zeros(self.cfg.max_sensors)
                mask[:n_sensors] = 1.0
                
                # Aux features
                aux = self._extract_aux(window.iloc[-1], sensor_cols)
                
                # Labels
                X_spec.append(spec)
                X_aux.append(aux)
                masks.append(mask)
                y_severity.append(window['severity'].iloc[-1])
                y_mode.append(window['mode'].iloc[-1])
                y_rul.append(window['rul_norm'].iloc[-1])
                y_health.append(window['health_norm'].iloc[-1])
        
        return (
            np.array(X_spec, dtype=np.float32),
            np.array(X_aux, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(y_severity),
            np.array(y_mode),
            np.array(y_rul, dtype=np.float32),
            np.array(y_health, dtype=np.float32)
        )
    
    def _create_spectrogram(self, sensor_data: np.ndarray) -> np.ndarray:
        """Create pseudo-spectrogram from sensor data."""
        n_sensors, seq_len = sensor_data.shape
        n_freq = self.cfg.n_freq
        n_time = self.cfg.n_time
        
        spec = np.zeros((n_sensors, n_freq, n_time), dtype=np.float32)
        
        for s in range(n_sensors):
            signal = sensor_data[s]
            frame_len = seq_len // n_time
            
            for t in range(n_time):
                start = t * frame_len
                end = start + frame_len
                frame = signal[start:end]
                fft = np.abs(np.fft.rfft(frame, n=n_freq * 2))[:n_freq]
                spec[s, :, t] = fft
        
        return spec[..., np.newaxis]  # (n_sensors, n_freq, n_time, 1)
    
    def _extract_aux(self, row: pd.Series, sensor_cols: List[str]) -> np.ndarray:
        """Extract auxiliary features."""
        aux = []
        for col in sensor_cols[:self.cfg.aux_dim]:
            if col in row.index:
                aux.append(float(row[col]))
        while len(aux) < self.cfg.aux_dim:
            aux.append(0.0)
        return np.array(aux[:self.cfg.aux_dim], dtype=np.float32)
    
    def encode_labels(self, y_sev, y_mode, fit=False):
        """Encode labels."""
        if fit:
            self.severity_labels = sorted(np.unique(y_sev).tolist())
            self.mode_labels = sorted(np.unique(y_mode).tolist())
            self.sev_to_idx = {s: i for i, s in enumerate(self.severity_labels)}
            self.mode_to_idx = {m: i for i, m in enumerate(self.mode_labels)}
        
        y_sev_enc = np.array([self.sev_to_idx.get(s, 0) for s in y_sev])
        y_mode_enc = np.array([self.mode_to_idx.get(m, 0) for m in y_mode])
        
        return y_sev_enc, y_mode_enc


# ===========================================================================
# MODEL BUILDER
# ===========================================================================

def build_model(cfg: ModelConfig) -> Model:
    """Build CNN 2D model with masked pooling."""
    
    # Inputs
    input_spec = layers.Input(
        shape=(cfg.max_sensors, cfg.n_freq, cfg.n_time, 1),
        name="input_spec"
    )
    input_aux = layers.Input(shape=(cfg.aux_dim,), name="input_aux")
    input_mask = layers.Input(shape=(cfg.max_sensors,), name="input_mask")
    
    # Sensor dropout
    x, mask = SensorDropout(cfg.sensor_dropout_rate)([input_spec, input_mask])
    
    # Shared CNN per sensor using TimeDistributed
    cnn = keras.Sequential([
        layers.Conv2D(cfg.conv_filters[0], (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(cfg.conv_filters[1], (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(cfg.conv_filters[2], (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
    ], name="shared_cnn")
    
    # Apply CNN to each sensor
    x = layers.TimeDistributed(cnn, name="td_cnn")(x)
    
    # Masked pooling over sensors
    x = MaskedGlobalPooling()([x, mask])
    
    # Dense
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    
    # Aux branch
    aux = layers.Dense(32, activation="relu")(input_aux)
    
    # Concatenate
    combined = layers.Concatenate()([x, aux])
    combined = layers.Dense(cfg.dense_units, activation="relu")(combined)
    combined = layers.Dropout(cfg.dropout)(combined)
    
    # Output heads
    out_severity = layers.Dense(cfg.n_severity, activation="softmax", name="severity")(combined)
    out_mode = layers.Dense(cfg.n_mode, activation="softmax", name="mode")(combined)
    out_rul = layers.Dense(1, activation="sigmoid", name="rul")(combined)
    out_health = layers.Dense(1, activation="sigmoid", name="health")(combined)
    
    model = Model(
        inputs=[input_spec, input_aux, input_mask],
        outputs=[out_severity, out_mode, out_rul, out_health],
        name="PumpCNN2D_v2"
    )
    
    return model


# ===========================================================================
# TRAINING
# ===========================================================================

def compute_sample_weights(y_sev: np.ndarray, y_mode: np.ndarray, n_severity: int, n_mode: int) -> np.ndarray:
    """Compute sample weights balancing both severity and mode classes."""
    # Severity weights
    sev_weights = compute_class_weight("balanced", classes=np.arange(n_severity), y=y_sev)
    sev_weights_dict = {i: w for i, w in enumerate(sev_weights)}
    
    # Mode weights  
    mode_weights = compute_class_weight("balanced", classes=np.arange(n_mode), y=y_mode)
    mode_weights_dict = {i: w for i, w in enumerate(mode_weights)}
    
    # Combined sample weights (geometric mean)
    sample_weights = np.array([
        np.sqrt(sev_weights_dict[s] * mode_weights_dict[m])
        for s, m in zip(y_sev, y_mode)
    ])
    
    return sample_weights


def train_model(
    model: Model,
    train_data: Tuple,
    val_data: Tuple,
    cfg: ModelConfig
) -> keras.callbacks.History:
    """Train the model."""
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Sample weights (class_weight not supported in multi-output)
    sample_weights = compute_sample_weights(y_train[0], y_train[1], cfg.n_severity, cfg.n_mode)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
        loss={
            "severity": keras.losses.SparseCategoricalCrossentropy(),
            "mode": keras.losses.SparseCategoricalCrossentropy(),
            "rul": keras.losses.MeanSquaredError(),
            "health": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "severity": cfg.loss_weight_severity,
            "mode": cfg.loss_weight_mode,
            "rul": cfg.loss_weight_rul,
            "health": cfg.loss_weight_health,
        },
        metrics={
            "severity": ["accuracy"],
            "mode": ["accuracy"],
            "rul": ["mae"],
            "health": ["mae"],
        }
    )
    
    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Train with sample weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=cb_list,
        sample_weight=sample_weights,
        verbose=1
    )
    
    return history


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate_model(model: Model, test_data: Tuple, cfg: ModelConfig) -> Dict:
    """Evaluate model and check targets."""
    
    X_test, y_test = test_data
    
    # Predict
    preds = model.predict(X_test, verbose=0)
    
    y_pred_sev = np.argmax(preds[0], axis=1)
    y_pred_mode = np.argmax(preds[1], axis=1)
    y_pred_rul = preds[2].flatten()
    y_pred_health = preds[3].flatten()
    
    # Metrics
    sev_acc = accuracy_score(y_test[0], y_pred_sev)
    mode_acc = accuracy_score(y_test[1], y_pred_mode)
    rul_mae = mean_absolute_error(y_test[2], y_pred_rul)
    health_mae = mean_absolute_error(y_test[3], y_pred_health)
    
    results = {
        "severity_accuracy": sev_acc,
        "mode_accuracy": mode_acc,
        "rul_mae": rul_mae,
        "health_mae": health_mae,
    }
    
    # Check targets
    targets = {
        "severity": sev_acc >= cfg.target_severity_acc,
        "mode": mode_acc >= cfg.target_mode_acc,
        "rul": rul_mae <= cfg.target_rul_mae,
        "health": health_mae <= cfg.target_health_mae,
    }
    
    results["targets_met"] = targets
    results["all_met"] = all(targets.values())
    
    return results, y_pred_sev, y_pred_mode


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("PUMP CNN 2D v2 - Training Pipeline")
    print("=" * 70)
    
    # Config
    cfg = ModelConfig()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "datasets" / "sensors_log_v2.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Sensor columns
    SENSOR_COLS = [
        "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
        "motor_current", "pressure", "flow", "temperature"
    ]
    
    # Pipeline
    pipeline = DataPipeline(cfg)
    
    # Load data
    df = pipeline.load_data(data_path)
    
    # Split by asset
    train_df, val_df, test_df = pipeline.split_by_asset(df)
    
    # Create windows
    print("\n[WINDOWS] Creating training windows...")
    X_train = pipeline.create_windows(train_df, SENSOR_COLS, fit_scaler=True)
    X_val = pipeline.create_windows(val_df, SENSOR_COLS, fit_scaler=False)
    X_test = pipeline.create_windows(test_df, SENSOR_COLS, fit_scaler=False)
    
    print(f"[INFO] Train windows: {X_train[0].shape[0]}")
    print(f"[INFO] Val windows: {X_val[0].shape[0]}")
    print(f"[INFO] Test windows: {X_test[0].shape[0]}")
    
    # Encode labels
    y_sev_train, y_mode_train = pipeline.encode_labels(X_train[3], X_train[4], fit=True)
    y_sev_val, y_mode_val = pipeline.encode_labels(X_val[3], X_val[4])
    y_sev_test, y_mode_test = pipeline.encode_labels(X_test[3], X_test[4])
    
    # Update config
    cfg.n_severity = len(pipeline.severity_labels)
    cfg.n_mode = len(pipeline.mode_labels)
    
    print(f"[INFO] Severity classes: {cfg.n_severity} - {pipeline.severity_labels}")
    print(f"[INFO] Mode classes: {cfg.n_mode} - {pipeline.mode_labels}")
    
    # Prepare data
    X_train_inputs = [X_train[0], X_train[1], X_train[2]]
    y_train_outputs = [y_sev_train, y_mode_train, X_train[5], X_train[6]]
    
    X_val_inputs = [X_val[0], X_val[1], X_val[2]]
    y_val_outputs = [y_sev_val, y_mode_val, X_val[5], X_val[6]]
    
    X_test_inputs = [X_test[0], X_test[1], X_test[2]]
    y_test_outputs = [y_sev_test, y_mode_test, X_test[5], X_test[6]]
    
    # Build model
    print("\n[BUILD] Building model...")
    model = build_model(cfg)
    model.summary()
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = train_model(
        model,
        (X_train_inputs, y_train_outputs),
        (X_val_inputs, y_val_outputs),
        cfg
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    results, y_pred_sev, y_pred_mode = evaluate_model(
        model, (X_test_inputs, y_test_outputs), cfg
    )
    
    print("\n" + "-" * 50)
    print("TEST RESULTS")
    print("-" * 50)
    print(f"Severity Accuracy: {results['severity_accuracy']*100:.2f}%")
    print(f"Mode Accuracy:     {results['mode_accuracy']*100:.2f}%")
    print(f"RUL MAE:           {results['rul_mae']:.4f}")
    print(f"Health MAE:        {results['health_mae']*100:.2f}%")
    
    print("\n" + "-" * 50)
    print("TARGET VERIFICATION")
    print("-" * 50)
    for target, met in results['targets_met'].items():
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"{target}: {status}")
    
    print("\nSeverity Report:")
    print(classification_report(y_sev_test, y_pred_sev, target_names=[str(l) for l in pipeline.severity_labels]))
    
    print("\nMode Report:")
    print(classification_report(y_mode_test, y_pred_mode, target_names=[str(l) for l in pipeline.mode_labels]))
    
    # Save model
    if results['all_met']:
        print("\n[SUCCESS] All targets met! Saving model...")
        model.save(model_dir / "pump_cnn2d_v2.keras")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "results": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in results.items() if k != 'targets_met'},
            "targets_met": results['targets_met'],
            "config": asdict(cfg),
            "severity_labels": pipeline.severity_labels,
            "mode_labels": pipeline.mode_labels,
            "rul_max_train": float(pipeline.rul_max_train),
        }
        
        with open(model_dir / "train_report_v2.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"[SAVE] Model: {model_dir / 'pump_cnn2d_v2.keras'}")
        print(f"[SAVE] Report: {model_dir / 'train_report_v2.json'}")
    else:
        print("\n[WARNING] Some targets not met!")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
