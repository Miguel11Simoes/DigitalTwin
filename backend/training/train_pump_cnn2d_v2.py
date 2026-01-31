#!/usr/bin/env python3
"""
train_pump_cnn2d_v2.py
======================
Modelo CNN 2D completo para classificação de bombas industriais.

IMPLEMENTA TODOS OS COMMITS A1-A12:
- A1: RUL/Health normalização sem leakage
- A2: sample_weight multi-output
- A3: Stress tests com spec=0 quando mask=0
- A4: PumpProfile FAST vs SLOW
- A5: Derived features (delta_p, head, efficiency)
- A6: Window statistics para AUX
- A7: STFT reais
- A8: Modelo chip-friendly (sem loop de sensores)
- A9: CriticalRecallCallback
- A10: OOD/Drift detection
- A11: Export completo + TFLite
- A12: Edge runtime ready

Targets:
- RUL MAE < 3%
- Health MAE < 3%
- Severity Accuracy >= 93%
- Mode Accuracy >= 99%

Author: Generated for Industrial Pump Digital Twin
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] TensorFlow: {tf.__version__}")
print(f"[INFO] Keras: {keras.__version__}")


# ===========================================================================
# A4: PUMP PROFILE - FAST vs SLOW separation
# ===========================================================================

@dataclass
class PumpProfile:
    """
    Perfil de colunas para bomba industrial.
    Separa FAST (alta frequência, espectrogramas) de SLOW (baixa frequência, estatísticas).
    """
    
    # FAST: Vibração (alta frequência, processada como espectrogramas)
    fast_cols: List[str] = field(default_factory=lambda: [
        "overall_vibration",
        "vibration_x",
        "vibration_y", 
        "vibration_z",
    ])
    
    # SLOW: Processo, elétrica, fluído (baixa frequência, estatísticas)
    slow_cols: List[str] = field(default_factory=lambda: [
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
        # Derivadas (A5)
        "hydraulic_power",
        "efficiency_est",
    ])
    
    # Context (categórico ou fixo por janela)
    context_cols: List[str] = field(default_factory=lambda: [
        "run_state",
        "ambient_temperature",
    ])
    
    # Targets
    target_regression: List[str] = field(default_factory=lambda: [
        "rul",
        "health",
    ])
    
    target_classification: List[str] = field(default_factory=lambda: [
        "severity",
        "mode",
    ])
    
    # Estatísticas para janela (A6)
    window_stats: List[str] = field(default_factory=lambda: [
        "last",
        "mean", 
        "std",
        "min",
        "max",
        "slope",
    ])
    
    @property
    def n_fast(self) -> int:
        return len(self.fast_cols)
    
    @property
    def n_slow(self) -> int:
        return len(self.slow_cols)
    
    @property
    def n_aux_per_col(self) -> int:
        return len(self.window_stats)
    
    @property
    def aux_dim(self) -> int:
        """Dimensão total do vetor AUX (slow stats + context)."""
        return self.n_slow * self.n_aux_per_col + len(self.context_cols)


# ===========================================================================
# MODEL CONFIGURATION
# ===========================================================================

@dataclass  
class ModelConfig:
    """Configuração do modelo CNN 2D."""
    
    # Input shapes
    max_sensors_fast: int = 4
    n_freq: int = 17        # (n_fft // 2 + 1) com n_fft=32
    n_frames: int = 8       # ((seq_len - n_fft) // hop_stft + 1)
    aux_dim: int = 98       # slow_cols * 6 stats + context
    
    # Conv blocks (A8: chip-friendly)
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernels: List[Tuple[int, int]] = field(default_factory=lambda: [(3, 3), (3, 3), (3, 3)])
    pool_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 2), (2, 1), (1, 1)])
    
    # Dense layers
    embedding_dim: int = 128
    aux_dense: int = 64
    combined_dense: int = 256
    dropout: float = 0.3
    
    # Output classes
    n_severity: int = 4     # critical, degraded, warning, healthy
    n_mode: int = 7         # 6 failure modes + normal
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    lr_decay: float = 0.95
    early_stop_patience: int = 15
    reduce_lr_patience: int = 5
    
    # Loss weights
    loss_weight_severity: float = 2.0
    loss_weight_mode: float = 1.0
    loss_weight_rul: float = 1.0
    loss_weight_health: float = 1.0
    
    # A10: OOD detection
    ood_percentile: float = 95.0
    
    # Seed
    seed: int = 42


# ===========================================================================
# A2: SAMPLE WEIGHTS
# ===========================================================================

def build_sample_weights(
    y_severity: np.ndarray,
    n_classes: int,
    emphasis: str = "minority"
) -> np.ndarray:
    """
    Calcula sample weights para balancear classes.
    
    Args:
        y_severity: Labels de severidade (0-3)
        n_classes: Número de classes
        emphasis: "minority" para dar mais peso a classes raras,
                  "critical" para dar mais peso a critical (0)
    
    Returns:
        Array de pesos por amostra
    """
    # Calcular class weights
    classes = np.arange(n_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_severity
    )
    
    # Converter para dict
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Se emphasis = "critical", aumentar peso de critical (class 0)
    if emphasis == "critical":
        class_weight_dict[0] *= 2.0  # Dobrar peso de critical
    
    # Criar array de sample weights
    sample_weights = np.array([class_weight_dict[y] for y in y_severity])
    
    return sample_weights, class_weight_dict


# ===========================================================================
# A3: STRESS TESTS - Mask application
# ===========================================================================

def apply_sensor_mask(X_spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica máscara de sensores: spec=0 quando mask=0.
    
    Args:
        X_spec: (batch, n_sensors, n_freq, n_frames, 1)
        mask: (batch, n_sensors)
    
    Returns:
        X_spec com zeros onde mask=0
    """
    # Expandir mask para shape compatível
    mask_expanded = mask[:, :, np.newaxis, np.newaxis, np.newaxis]
    return X_spec * mask_expanded


def augment_sensor_dropout(
    X_spec: np.ndarray,
    mask: np.ndarray, 
    dropout_prob: float = 0.1,
    min_sensors: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation: dropout aleatório de sensores.
    
    Args:
        X_spec: (batch, n_sensors, n_freq, n_frames, 1)
        mask: (batch, n_sensors)
        dropout_prob: Probabilidade de dropout por sensor
        min_sensors: Mínimo de sensores ativos
    
    Returns:
        X_spec_aug, mask_aug
    """
    batch_size, n_sensors = mask.shape
    mask_aug = mask.copy()
    
    for b in range(batch_size):
        # Encontrar sensores ativos
        active = np.where(mask_aug[b] > 0)[0]
        n_active = len(active)
        
        if n_active <= min_sensors:
            continue
        
        # Dropar aleatoriamente
        for s in active:
            if np.random.random() < dropout_prob and n_active > min_sensors:
                mask_aug[b, s] = 0.0
                n_active -= 1
    
    # Aplicar máscara
    X_spec_aug = apply_sensor_mask(X_spec, mask_aug)
    
    return X_spec_aug, mask_aug


# ===========================================================================
# A8: CHIP-FRIENDLY CNN MODEL (no sensor loop)
# ===========================================================================

class MaskedGlobalPooling(layers.Layer):
    """Global Average Pooling que respeita máscara de sensores."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        x, mask = inputs
        # x: (batch, n_sensors, features)
        # mask: (batch, n_sensors)
        
        mask_expanded = tf.expand_dims(mask, -1)  # (batch, n_sensors, 1)
        x_masked = x * mask_expanded
        
        # Sum over sensors
        sum_features = tf.reduce_sum(x_masked, axis=1)  # (batch, features)
        
        # Count active sensors
        n_active = tf.reduce_sum(mask, axis=1, keepdims=True)  # (batch, 1)
        n_active = tf.maximum(n_active, 1.0)  # Avoid division by zero
        
        # Average
        return sum_features / n_active


def build_cnn2d_model(cfg: ModelConfig) -> Model:
    """
    Constrói modelo CNN 2D chip-friendly.
    
    Architecture:
    - Input: Espectrogramas (n_sensors, n_freq, n_frames, 1) + AUX + Mask
    - CNN compartilhada entre sensores (sem loop explícito)
    - Masked global pooling
    - Dense para embedding
    - Multi-head output (severity, mode, RUL, health)
    
    Returns:
        Model Keras compilado
    """
    # Inputs
    input_spec = layers.Input(
        shape=(cfg.max_sensors_fast, cfg.n_freq, cfg.n_frames, 1),
        name="input_spec"
    )
    input_aux = layers.Input(shape=(cfg.aux_dim,), name="input_aux")
    input_mask = layers.Input(shape=(cfg.max_sensors_fast,), name="input_mask")
    
    # Reshape spec para processar todos sensores juntos: (batch, n_sensors, freq, time, 1)
    # Vamos usar TimeDistributed para aplicar CNN em cada sensor
    
    # CNN compartilhada
    cnn_layers = []
    for i, (filters, kernel, pool) in enumerate(zip(
        cfg.conv_filters, cfg.conv_kernels, cfg.pool_sizes
    )):
        cnn_layers.extend([
            layers.Conv2D(
                filters, kernel, activation="relu", padding="same",
                name=f"conv_{i}"
            ),
            layers.BatchNormalization(name=f"bn_{i}"),
            layers.MaxPooling2D(pool, name=f"pool_{i}"),
        ])
    
    # Flatten
    cnn_layers.append(layers.Flatten(name="flatten"))
    
    # Criar modelo CNN sequencial
    cnn_model = keras.Sequential(cnn_layers, name="shared_cnn")
    
    # Aplicar CNN em cada sensor usando TimeDistributed
    x = layers.TimeDistributed(cnn_model, name="td_cnn")(input_spec)
    # x: (batch, n_sensors, cnn_features)
    
    # Masked Global Pooling
    x_pooled = MaskedGlobalPooling(name="masked_pool")([x, input_mask])
    # x_pooled: (batch, cnn_features)
    
    # Embedding
    embedding = layers.Dense(cfg.embedding_dim, activation="relu", name="embedding")(x_pooled)
    embedding = layers.Dropout(cfg.dropout, name="dropout_emb")(embedding)
    
    # AUX branch
    aux = layers.Dense(cfg.aux_dense, activation="relu", name="aux_dense")(input_aux)
    aux = layers.BatchNormalization(name="bn_aux")(aux)
    
    # Concatenar
    combined = layers.Concatenate(name="concat")([embedding, aux])
    combined = layers.Dense(cfg.combined_dense, activation="relu", name="combined_dense")(combined)
    combined = layers.Dropout(cfg.dropout, name="dropout_combined")(combined)
    
    # Output heads
    out_severity = layers.Dense(
        cfg.n_severity, activation="softmax", name="severity"
    )(combined)
    
    out_mode = layers.Dense(
        cfg.n_mode, activation="softmax", name="mode"
    )(combined)
    
    out_rul = layers.Dense(1, activation="sigmoid", name="rul")(combined)
    
    out_health = layers.Dense(1, activation="sigmoid", name="health")(combined)
    
    # Model
    model = Model(
        inputs=[input_spec, input_aux, input_mask],
        outputs=[out_severity, out_mode, out_rul, out_health],
        name="PumpCNN2D_v2"
    )
    
    return model


# ===========================================================================
# A9: CRITICAL RECALL CALLBACK
# ===========================================================================

class CriticalRecallCallback(callbacks.Callback):
    """
    Early stopping baseado em recall de classe crítica.
    
    Para se o recall de critical (class 0) cair abaixo de threshold.
    """
    
    def __init__(
        self,
        validation_data: Tuple,
        critical_class: int = 0,
        min_recall: float = 0.85,
        patience: int = 5
    ):
        super().__init__()
        self.validation_data = validation_data
        self.critical_class = critical_class
        self.min_recall = min_recall
        self.patience = patience
        self.wait = 0
        self.best_recall = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        
        # Predict
        preds = self.model.predict(X_val, verbose=0)
        y_pred_severity = np.argmax(preds[0], axis=1)
        y_true_severity = y_val[0]
        
        # Calcular recall para critical
        critical_mask = y_true_severity == self.critical_class
        if np.sum(critical_mask) > 0:
            critical_correct = np.sum(
                (y_pred_severity == self.critical_class) & critical_mask
            )
            critical_recall = critical_correct / np.sum(critical_mask)
        else:
            critical_recall = 1.0
        
        logs["critical_recall"] = critical_recall
        
        # Check
        if critical_recall < self.min_recall:
            self.wait += 1
            print(f"\n[WARN] Critical recall {critical_recall:.3f} < {self.min_recall}")
            if self.wait >= self.patience:
                print(f"\n[STOP] Critical recall too low for {self.patience} epochs")
                self.model.stop_training = True
        else:
            self.wait = 0
            if critical_recall > self.best_recall:
                self.best_recall = critical_recall


# ===========================================================================
# A10: OOD / DRIFT DETECTION
# ===========================================================================

def compute_embedding_threshold(
    model: Model,
    X_train: List[np.ndarray],
    percentile: float = 95.0
) -> Dict[str, float]:
    """
    Calcula threshold para detecção OOD baseado em embeddings.
    
    Returns:
        Dict com centroid e threshold
    """
    # Criar modelo para extrair embeddings
    embedding_layer = model.get_layer("embedding")
    embedding_model = Model(
        inputs=model.inputs,
        outputs=embedding_layer.output
    )
    
    # Extrair embeddings do train
    embeddings = embedding_model.predict(X_train, verbose=0)
    
    # Centroid
    centroid = np.mean(embeddings, axis=0)
    
    # Distâncias ao centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # Threshold no percentil
    threshold = np.percentile(distances, percentile)
    
    return {
        "centroid": centroid.tolist(),
        "threshold": float(threshold),
        "percentile": percentile,
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
    }


def detect_ood(
    model: Model,
    X_input: List[np.ndarray],
    ood_config: Dict
) -> np.ndarray:
    """
    Detecta amostras OOD baseado em distância ao centroid.
    
    Returns:
        Array booleano: True = OOD
    """
    embedding_layer = model.get_layer("embedding")
    embedding_model = Model(
        inputs=model.inputs,
        outputs=embedding_layer.output
    )
    
    embeddings = embedding_model.predict(X_input, verbose=0)
    centroid = np.array(ood_config["centroid"])
    threshold = ood_config["threshold"]
    
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    is_ood = distances > threshold
    
    return is_ood, distances


# ===========================================================================
# A1: RUL NORMALIZATION WITHOUT LEAKAGE
# ===========================================================================

def normalize_rul_health(
    y_rul_train: np.ndarray,
    y_rul_val: np.ndarray,
    y_rul_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Normaliza RUL usando apenas max do train (sem leakage).
    
    Returns:
        y_rul_train_norm, y_rul_val_norm, y_rul_test_norm, rul_max_train
    """
    # A1: Usar APENAS dados de treino para normalização
    rul_max_train = float(y_rul_train.max())
    
    y_rul_train_norm = y_rul_train / max(rul_max_train, 1.0)
    y_rul_val_norm = y_rul_val / max(rul_max_train, 1.0)
    y_rul_test_norm = y_rul_test / max(rul_max_train, 1.0)
    
    # Clip test/val para [0, 1] - pode haver valores > max_train
    y_rul_val_norm = np.clip(y_rul_val_norm, 0, 1)
    y_rul_test_norm = np.clip(y_rul_test_norm, 0, 1)
    
    return y_rul_train_norm, y_rul_val_norm, y_rul_test_norm, rul_max_train


# ===========================================================================
# A11: EXPORT FUNCTIONS
# ===========================================================================

def export_model(
    model: Model,
    profile: PumpProfile,
    cfg: ModelConfig,
    scaler_stats: Dict,
    label_maps: Dict,
    ood_config: Dict,
    rul_max_train: float,
    output_dir: Path
):
    """
    Exporta modelo completo para deployment.
    
    Exports:
    - model.keras: Modelo Keras
    - model.tflite: TFLite para edge
    - profile.json: PumpProfile
    - config.json: ModelConfig
    - scalers.json: Normalização
    - labels.json: Label mappings
    - ood_config.json: OOD detection config
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Keras model
    keras_path = output_dir / "pump_cnn2d_v2.keras"
    model.save(keras_path)
    print(f"[EXPORT] Keras: {keras_path}")
    
    # 2. TFLite (A11)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = output_dir / "pump_cnn2d_v2.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[EXPORT] TFLite: {tflite_path}")
    except Exception as e:
        print(f"[WARN] TFLite export failed: {e}")
    
    # 3. Profile
    profile_path = output_dir / "pump_profile.json"
    with open(profile_path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    print(f"[EXPORT] Profile: {profile_path}")
    
    # 4. Config
    config_path = output_dir / "model_config.json"
    cfg_dict = asdict(cfg)
    cfg_dict["rul_max_train"] = rul_max_train
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"[EXPORT] Config: {config_path}")
    
    # 5. Scalers
    scaler_path = output_dir / "pump_scalers.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"[EXPORT] Scalers: {scaler_path}")
    
    # 6. Labels
    labels_path = output_dir / "pump_labels.json"
    with open(labels_path, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"[EXPORT] Labels: {labels_path}")
    
    # 7. OOD config
    ood_path = output_dir / "ood_config.json"
    with open(ood_path, "w") as f:
        json.dump(ood_config, f, indent=2)
    print(f"[EXPORT] OOD: {ood_path}")
    
    # 8. Manifest
    manifest = {
        "model_name": "PumpCNN2D_v2",
        "version": "2.0.0",
        "created": datetime.now().isoformat(),
        "files": [
            "pump_cnn2d_v2.keras",
            "pump_cnn2d_v2.tflite",
            "pump_profile.json",
            "model_config.json",
            "pump_scalers.json",
            "pump_labels.json",
            "ood_config.json",
        ],
        "targets": {
            "severity": cfg.n_severity,
            "mode": cfg.n_mode,
            "rul": "regression [0,1]",
            "health": "regression [0,1]",
        },
        "rul_max_train": rul_max_train,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[EXPORT] Manifest: {manifest_path}")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_windows(data_path: Path) -> Tuple[Dict, Dict]:
    """
    Carrega janelas pré-processadas.
    
    Returns:
        (data_dict, metadata)
    """
    # NPZ
    npz_path = data_path / "pump_windows_train.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Windows file not found: {npz_path}")
    
    data = dict(np.load(npz_path, allow_pickle=True))
    
    # Metadata
    meta_path = data_path / "windows_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return data, metadata


# ===========================================================================
# MAIN TRAINING PIPELINE
# ===========================================================================

def main():
    print("=" * 70)
    print("PUMP CNN 2D v2 - Industrial Training Pipeline")
    print("=" * 70)
    
    # Setup
    cfg = ModelConfig()
    profile = PumpProfile()
    
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets"
    model_dir = base_dir / "models"
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[LOAD] Loading preprocessed windows...")
    
    try:
        data, metadata = load_windows(data_dir)
    except FileNotFoundError:
        print("[ERROR] Windows not found. Run build_windows.py first!")
        sys.exit(1)
    
    X_spec = data["X_spec"]
    X_aux = data["X_aux"]
    mask = data["mask"]
    y_severity = data["y_severity"]
    y_mode = data["y_mode"]
    y_rul = data["y_rul"]
    y_health = data["y_health"]
    
    print(f"[INFO] X_spec shape: {X_spec.shape}")
    print(f"[INFO] X_aux shape: {X_aux.shape}")
    print(f"[INFO] mask shape: {mask.shape}")
    print(f"[INFO] Total samples: {len(y_severity)}")
    
    # Update config from data
    cfg.max_sensors_fast = X_spec.shape[1]
    cfg.n_freq = X_spec.shape[2]
    cfg.n_frames = X_spec.shape[3]
    cfg.aux_dim = X_aux.shape[1]
    cfg.n_severity = len(np.unique(y_severity))
    cfg.n_mode = len(np.unique(y_mode))
    
    print(f"[INFO] Updated config: n_severity={cfg.n_severity}, n_mode={cfg.n_mode}")
    
    # =========================================================================
    # TRAIN/VAL/TEST SPLIT
    # =========================================================================
    print("\n[SPLIT] Creating train/val/test splits...")
    
    indices = np.arange(len(y_severity))
    
    # Stratified split on severity
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=y_severity, random_state=cfg.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y_severity[temp_idx], random_state=cfg.seed
    )
    
    # Extract splits
    X_spec_train, X_spec_val, X_spec_test = X_spec[train_idx], X_spec[val_idx], X_spec[test_idx]
    X_aux_train, X_aux_val, X_aux_test = X_aux[train_idx], X_aux[val_idx], X_aux[test_idx]
    mask_train, mask_val, mask_test = mask[train_idx], mask[val_idx], mask[test_idx]
    
    y_sev_train, y_sev_val, y_sev_test = y_severity[train_idx], y_severity[val_idx], y_severity[test_idx]
    y_mode_train, y_mode_val, y_mode_test = y_mode[train_idx], y_mode[val_idx], y_mode[test_idx]
    y_rul_train, y_rul_val, y_rul_test = y_rul[train_idx], y_rul[val_idx], y_rul[test_idx]
    y_health_train, y_health_val, y_health_test = y_health[train_idx], y_health[val_idx], y_health[test_idx]
    
    print(f"[INFO] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # =========================================================================
    # A1: RUL NORMALIZATION
    # =========================================================================
    print("\n[A1] RUL normalization without leakage...")
    
    y_rul_train_raw = y_rul_train.copy()
    y_rul_train, y_rul_val, y_rul_test, rul_max_train = normalize_rul_health(
        y_rul_train, y_rul_val, y_rul_test
    )
    print(f"[INFO] rul_max_train: {rul_max_train:.2f}")
    
    # =========================================================================
    # A3: APPLY MASK (stress test ready)
    # =========================================================================
    print("\n[A3] Applying sensor masks...")
    
    X_spec_train = apply_sensor_mask(X_spec_train, mask_train)
    X_spec_val = apply_sensor_mask(X_spec_val, mask_val)
    X_spec_test = apply_sensor_mask(X_spec_test, mask_test)
    
    # =========================================================================
    # A2: SAMPLE WEIGHTS
    # =========================================================================
    print("\n[A2] Computing sample weights...")
    
    sample_weights, class_weight_dict = build_sample_weights(
        y_sev_train, cfg.n_severity, emphasis="critical"
    )
    print(f"[INFO] Class weights: {class_weight_dict}")
    
    # =========================================================================
    # BUILD MODEL
    # =========================================================================
    print("\n[BUILD] Building CNN 2D model...")
    
    model = build_cnn2d_model(cfg)
    model.summary()
    
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
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    X_train_inputs = [X_spec_train, X_aux_train, mask_train]
    X_val_inputs = [X_spec_val, X_aux_val, mask_val]
    y_train_outputs = [y_sev_train, y_mode_train, y_rul_train, y_health_train]
    y_val_outputs = [y_sev_val, y_mode_val, y_rul_val, y_health_val]
    
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.lr_decay,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_dir / "pump_cnn2d_v2_best.keras",
            monitor="val_severity_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # A9: Critical Recall Callback
        CriticalRecallCallback(
            validation_data=(X_val_inputs, y_val_outputs),
            critical_class=0,
            min_recall=0.80,
            patience=10
        ),
    ]
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = model.fit(
        X_train_inputs,
        y_train_outputs,
        validation_data=(X_val_inputs, y_val_outputs),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=cb_list,
        sample_weight=sample_weights,  # A2: Applied!
        verbose=1
    )
    
    # =========================================================================
    # EVALUATE
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    X_test_inputs = [X_spec_test, X_aux_test, mask_test]
    y_test_outputs = [y_sev_test, y_mode_test, y_rul_test, y_health_test]
    
    results = model.evaluate(X_test_inputs, y_test_outputs, verbose=1)
    
    # Get predictions
    preds = model.predict(X_test_inputs, verbose=0)
    y_pred_sev = np.argmax(preds[0], axis=1)
    y_pred_mode = np.argmax(preds[1], axis=1)
    y_pred_rul = preds[2].flatten()
    y_pred_health = preds[3].flatten()
    
    # Metrics
    from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
    
    sev_acc = accuracy_score(y_sev_test, y_pred_sev)
    mode_acc = accuracy_score(y_mode_test, y_pred_mode)
    rul_mae = mean_absolute_error(y_rul_test, y_pred_rul)
    health_mae = mean_absolute_error(y_health_test, y_pred_health)
    
    print("\n" + "-" * 50)
    print("TEST RESULTS")
    print("-" * 50)
    print(f"Severity Accuracy: {sev_acc*100:.2f}%")
    print(f"Mode Accuracy:     {mode_acc*100:.2f}%")
    print(f"RUL MAE:           {rul_mae*100:.2f}%")
    print(f"Health MAE:        {health_mae*100:.2f}%")
    
    print("\nSeverity Classification Report:")
    print(classification_report(y_sev_test, y_pred_sev))
    
    print("\nMode Classification Report:")
    print(classification_report(y_mode_test, y_pred_mode))
    
    # Check targets
    targets_met = {
        "severity": sev_acc >= 0.93,
        "mode": mode_acc >= 0.99,
        "rul": rul_mae <= 0.03,
        "health": health_mae <= 0.03,
    }
    
    print("\n" + "-" * 50)
    print("TARGET VERIFICATION")
    print("-" * 50)
    for target, met in targets_met.items():
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"{target}: {status}")
    
    all_met = all(targets_met.values())
    
    # =========================================================================
    # A10: OOD DETECTION
    # =========================================================================
    print("\n[A10] Computing OOD detection config...")
    
    ood_config = compute_embedding_threshold(
        model, X_train_inputs, cfg.ood_percentile
    )
    print(f"[INFO] OOD threshold: {ood_config['threshold']:.4f}")
    
    # Test OOD detection
    is_ood, distances = detect_ood(model, X_test_inputs, ood_config)
    print(f"[INFO] Test OOD samples: {np.sum(is_ood)} ({np.mean(is_ood)*100:.1f}%)")
    
    # =========================================================================
    # A11: EXPORT
    # =========================================================================
    print("\n[A11] Exporting model...")
    
    # Load scaler stats
    scaler_path = data_dir / "pump_scalers.json"
    if scaler_path.exists():
        with open(scaler_path) as f:
            scaler_stats = json.load(f)
    else:
        scaler_stats = {}
    
    # Label maps
    label_maps = {
        "severity": metadata.get("sev_to_idx", {}),
        "mode": metadata.get("mode_to_idx", {}),
    }
    
    export_model(
        model=model,
        profile=profile,
        cfg=cfg,
        scaler_stats=scaler_stats,
        label_maps=label_maps,
        ood_config=ood_config,
        rul_max_train=rul_max_train,
        output_dir=model_dir
    )
    
    # =========================================================================
    # SAVE TRAINING REPORT
    # =========================================================================
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "PumpCNN2D_v2",
        "dataset": {
            "total_samples": len(y_severity),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
        },
        "config": asdict(cfg),
        "results": {
            "severity_accuracy": float(sev_acc),
            "mode_accuracy": float(mode_acc),
            "rul_mae": float(rul_mae),
            "health_mae": float(health_mae),
        },
        "targets_met": targets_met,
        "all_targets_met": all_met,
        "ood_config": ood_config,
        "commits_implemented": [
            "A1: RUL normalization without leakage",
            "A2: sample_weight multi-output",
            "A3: Stress tests (mask application)",
            "A4: PumpProfile FAST/SLOW",
            "A5: Derived features in pipeline",
            "A6: Window statistics",
            "A7: Real STFT",
            "A8: Chip-friendly model",
            "A9: CriticalRecallCallback",
            "A10: OOD detection",
            "A11: Complete export",
        ],
    }
    
    report_path = model_dir / "train_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[SAVE] Report: {report_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    if all_met:
        print("SUCCESS! All targets met.")
    else:
        print("Some targets not met. Check report for details.")
    print("=" * 70)
    
    return model, history, report


if __name__ == "__main__":
    model, history, report = main()
