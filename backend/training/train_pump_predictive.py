# train_pump_predictive.py
# Multi-task CNN para manutenção preditiva de bombas industriais
# Outputs: RUL, Health Index, Severity, Mode
# Baseado em espectrograma + features auxiliares

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# --------------------------
# Config
# --------------------------
BASE_DIR      = Path(__file__).parent
LOGS_DIR      = BASE_DIR / "logs"
ARCHIVE_DIR   = LOGS_DIR / "archive"
OUT_DIR       = Path(os.environ.get("DT_MODELS_DIR", str(BASE_DIR / "models")))
MODEL_PATH    = OUT_DIR / "pump_predictive.keras"
LABELS_PATH   = OUT_DIR / "labels_master.json"
REPORT_PATH   = OUT_DIR / "eval_report.json"

# Carregar todos os CSVs disponíveis
CSV_FILES = [
    LOGS_DIR / "sensors_log.csv",
    ARCHIVE_DIR / "sensors_log_20250930_110827.csv",
    ARCHIVE_DIR / "sensors_log_20251001_063731.csv",
    ARCHIVE_DIR / "sensors_log_20251002_005621.csv",
]

# Split temporal
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Janela - Otimizada para balancear samples e generalização
SEQ_LEN   = 512
HOP       = 256    # Aumentado para menos overlap (menos overfitting)
FRAME_LEN = 64
FRAME_HOP = 32
T_WINDOW  = 128
N_MELS    = 128

# Treino - Configuração otimizada
EPOCHS      = 500  # Mais epochs para convergência gradual
BATCH_SIZE  = 64   # Batch maior para melhor generalização
SEED        = 42
LR          = 1e-3  # LR inicial maior com schedule agressivo
PATIENCE    = 80    # Muita paciência para permitir plateaus
L2_REG      = 1e-4  # Regularização L2

np.random.seed(SEED)
tf.random.set_seed(SEED)

PRIMARY_PREF = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "ultrasonic_noise", "motor_current", "pressure", "flow", "temperature"
]

DROP_AUX = {
    "anomaly_score", "failure_probability", "health_index", "rul_minutes", 
    "model_confidence", "predicted_mode"
}

# --------------------------
# Data Augmentation para Espectrogramas - VERSÃO REFORÇADA
# --------------------------
def augment_spectrogram_numpy(spec):
    """
    Aplica augmentation MAIS AGRESSIVO a espectrogramas usando NumPy.
    - Time masking: múltiplos blocos até 30% do tempo
    - Frequency masking: múltiplos blocos até 30% das frequências
    - Gaussian noise: até 8% de ruído
    - Time stretch: simula variações de velocidade
    - Amplitude scaling: variações de intensidade
    """
    spec = spec.copy()
    
    # Time masking - 2-3 blocos pequenos (mais efetivo que 1 grande)
    num_time_masks = np.random.randint(2, 4)
    for _ in range(num_time_masks):
        if np.random.rand() > 0.3:
            time_mask_param = int(T_WINDOW * 0.15)  # Blocos menores
            t = np.random.randint(1, time_mask_param)
            t0 = np.random.randint(0, T_WINDOW - t)
            spec[t0:t0+t, :, :] = 0
    
    # Frequency masking - 2-3 blocos pequenos
    num_freq_masks = np.random.randint(2, 4)
    for _ in range(num_freq_masks):
        if np.random.rand() > 0.3:
            freq_mask_param = int(N_MELS * 0.15)
            f = np.random.randint(1, freq_mask_param)
            f0 = np.random.randint(0, N_MELS - f)
            spec[:, f0:f0+f, :] = 0
    
    # Gaussian noise mais forte
    if np.random.rand() > 0.4:
        noise = np.random.normal(0.0, 0.08, spec.shape).astype(np.float32)  # 8% em vez de 5%
        spec = spec + noise
    
    # Amplitude scaling (simula variações de intensidade do sinal)
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.85, 1.15)
        spec = spec * scale
    
    return np.clip(spec, -10, 10).astype(np.float32)  # Clip extremos

# --------------------------
# Focal Loss for imbalanced classes - SPARSE VERSION
# --------------------------
def sparse_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss to handle class imbalance - accepts sparse integer labels.
    Focuses training on hard examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Flatten y_true if needed (sometimes comes as (batch, 1), sometimes as (batch,))
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        
        # Get probabilities for true class
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, indices)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = K.pow(1.0 - p_t, gamma)
        
        # Compute cross entropy loss
        ce = -K.log(p_t)
        
        # Apply focal weight and alpha
        loss = alpha * focal_weight * ce
        return loss
    
    return focal_loss_fixed

# --------------------------
# Load data
# --------------------------
print(f"[INFO] A carregar múltiplos CSVs...")
dfs = []
for csv_file in CSV_FILES:
    if not csv_file.exists():
        print(f"[WARN] Ficheiro não encontrado: {csv_file}")
        continue
    print(f"[INFO] A ler: {csv_file}")
    try:
        df_temp = pd.read_csv(csv_file, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True)
        dfs.append(df_temp)
    except UnicodeDecodeError:
        df_temp = pd.read_csv(csv_file, sep=None, engine="python", on_bad_lines="warn",
                         skipinitialspace=True, encoding="latin1")
        dfs.append(df_temp)

if not dfs:
    raise FileNotFoundError("Nenhum CSV encontrado!")

print(f"[INFO] Total de ficheiros carregados: {len(dfs)}")
df = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Total de linhas: {len(df)}")

if "mode" not in df.columns:
    raise ValueError("CSV precisa de coluna 'mode'.")

# Criar targets preditivos se não existirem
if "rul_minutes" not in df.columns:
    # Sintético: decai de 1000h até 0 por modo
    df["rul_minutes"] = 0.0
    for mode_label in df["mode"].unique():
        mask = df["mode"] == mode_label
        n = mask.sum()
        if n > 0:
            df.loc[mask, "rul_minutes"] = np.linspace(1000*60, 0, n)

# Forçar recriação de health_index e severity para dados sintéticos melhores
print("[INFO] A gerar health_index e severity sintéticos...")
np.random.seed(SEED)

# Normalizar health_index para 0-1 (facilita treino)
df["health_index_raw"] = 100.0
for mode_label in df["mode"].unique():
    mask = df["mode"] == mode_label
    n = mask.sum()
    if n > 0:
        t = np.linspace(0, 1, n)
        if "bearing" in mode_label or "wear" in mode_label:
            base_health = 1.0 - t**1.5
        elif "electrical" in mode_label or "motor" in mode_label:
            base_health = 1.0 - t**2.0
        else:
            base_health = 1.0 - t**1.3
        noise = np.random.normal(0, 0.02, n)
        health = np.clip(base_health + noise, 0, 1)
        df.loc[mask, "health_index_raw"] = health

# Criar coluna normalizada para treino
df["health_index"] = df["health_index_raw"] * 100.0  # Manter 0-100 para compatibilidade

# Mapear de health_index com limiares ajustados para distribuição melhor
# CRITICAL: Mais agressivo para balancear classes
def map_severity(h):
    if h >= 90: return "normal"      # Apenas muito saudável
    elif h >= 70: return "early"     # Começa a degradar
    elif h >= 50: return "moderate"  # Degradação moderada
    elif h >= 30: return "severe"    # Degradação severa
    else: return "failure"           # Falha iminente
df["severity"] = df["health_index"].apply(map_severity)

print(f"[INFO] Distribuição de severity: {df['severity'].value_counts().to_dict()}")

if "timestamp" not in df.columns:
    df["timestamp"] = np.arange(len(df))

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

for c in df.columns:
    if c not in ("mode", "severity", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["mode", "severity"]).reset_index(drop=True)
df["mode"] = df["mode"].astype(str)
df["severity"] = df["severity"].astype(str)

# Escolher canal primário
primary_col = None
for c in PRIMARY_PREF:
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
        primary_col = c
        break
if not primary_col:
    primary_col = "synthetic_primary"
    df[primary_col] = 0.0
print(f"[PRIMARY] {primary_col}")

# Auxiliares
aux_candidates = [
    c for c in df.columns
    if (c not in (primary_col, "timestamp", "mode", "severity", "rul_minutes", "health_index"))
    and pd.api.types.is_numeric_dtype(df[c])
    and (c not in DROP_AUX)
]

# --------------------------
# Split temporal por classe
# --------------------------
def temporal_split(df_in: pd.DataFrame, tr=0.70, va=0.15):
    parts = []
    for lab, dfg in df_in.groupby("mode", sort=False):
        dfg = dfg.sort_values("timestamp")
        n = len(dfg)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        df_tr = dfg.iloc[:n_tr].copy()
        df_va = dfg.iloc[n_tr:n_tr+n_va].copy()
        df_te = dfg.iloc[n_tr+n_va:].copy()
        df_tr["__split__"] = "train"
        df_va["__split__"] = "val"
        df_te["__split__"] = "test"
        parts += [df_tr, df_va, df_te]
    out = pd.concat(parts, ignore_index=True)
    return (
        out[out["__split__"]=="train"].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"]=="val"  ].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"]=="test" ].drop(columns="__split__").reset_index(drop=True),
    )

df_tr, df_va, df_te = temporal_split(df, TRAIN_RATIO, VAL_RATIO)
print(f"[SPLIT] train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")

# --------------------------
# Janela → espectrograma + aux + targets
# --------------------------
def window_indices(n: int, win: int, hop: int):
    idx = []
    i = 0
    while i + win <= n:
        idx.append((i, i+win))
        i += hop
    return idx

def series_to_spectrogram(x: np.ndarray):
    stft = tf.signal.stft(
        signals=tf.convert_to_tensor(x, dtype=tf.float32),
        frame_length=FRAME_LEN,
        frame_step=FRAME_HOP,
        fft_length=FRAME_LEN,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    mag = tf.abs(stft) + 1e-9
    logmag = tf.math.log(mag)
    mean = tf.reduce_mean(logmag)
    std  = tf.math.reduce_std(logmag) + 1e-6
    norm = (logmag - mean) / std
    spec_hw = tf.expand_dims(norm, axis=-1)
    spec_hw = tf.image.resize(spec_hw, size=(T_WINDOW, N_MELS), method="bilinear")
    out = tf.squeeze(spec_hw, axis=-1).numpy()
    return out.astype("float32")

def slope(y: np.ndarray):
    n = y.shape[0]
    if n < 2: return 0.0
    x = np.arange(n, dtype=np.float32)
    vx = np.var(x)
    if vx < 1e-12: return 0.0
    return float(np.cov(x, y, bias=True)[0,1] / vx)

def make_windows(df_block: pd.DataFrame):
    """
    Retorna: X_spec, X_aux, y_rul, y_health, y_sev_str, y_mode_str
    """
    p = df_block[primary_col].to_numpy(dtype=np.float32)
    n = len(p)
    idxs = window_indices(n, SEQ_LEN, HOP)
    
    X_spec, X_aux = [], []
    Y_rul, Y_health, Y_sev, Y_mode = [], [], [], []
    
    for (i0, i1) in idxs:
        seg = p[i0:i1]
        if not np.isfinite(seg).all():
            continue
        
        spec = series_to_spectrogram(seg)
        X_spec.append(spec)
        
        # Aux features
        aux_feats = []
        for c in aux_candidates:
            arr = df_block[c].to_numpy(dtype=np.float32)[i0:i1]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                aux_feats += [0.0, 0.0, 0.0]
            else:
                aux_feats += [float(np.mean(arr)), float(np.std(arr)), slope(arr)]
        if not aux_feats:
            aux_feats = [0.0]
        X_aux.append(np.array(aux_feats, dtype=np.float32))
        
        # Targets: último valor da janela (ou média)
        rul_win = df_block["rul_minutes"].iloc[i0:i1].to_numpy()
        health_win = df_block["health_index"].iloc[i0:i1].to_numpy()
        sev_win = df_block["severity"].iloc[i0:i1].astype(str).to_numpy()
        mode_win = df_block["mode"].iloc[i0:i1].astype(str).to_numpy()
        
        if rul_win.size == 0:
            continue
        
        # RUL normalizado (0-1, assumindo 0-1000h)
        rul_val = float(np.nanmean(rul_win)) / (1000.0 * 60.0)
        rul_val = np.clip(rul_val, 0.0, 1.0)
        Y_rul.append(rul_val)
        
        # Health normalizado (0-1 em vez de 0-100 para facilitar treino)
        health_val = float(np.nanmean(health_win)) / 100.0
        health_val = np.clip(health_val, 0.0, 1.0)
        Y_health.append(health_val)
        
        # Severity/Mode: moda
        vals_sev, cnts_sev = np.unique(sev_win, return_counts=True)
        Y_sev.append(vals_sev[np.argmax(cnts_sev)])
        
        vals_mode, cnts_mode = np.unique(mode_win, return_counts=True)
        Y_mode.append(vals_mode[np.argmax(cnts_mode)])
    
    if not X_spec:
        maxF = len(aux_candidates)*3 or 1
        return (np.zeros((0, T_WINDOW, N_MELS), dtype=np.float32),
                np.zeros((0, maxF), dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=str),
                np.array([], dtype=str))
    
    maxF = len(aux_candidates)*3 if aux_candidates else 1
    X_aux = [np.pad(a, (0, maxF - a.shape[0])) for a in X_aux]
    
    return (
        np.stack(X_spec, axis=0),
        np.stack(X_aux, axis=0),
        np.array(Y_rul, dtype=np.float32),
        np.array(Y_health, dtype=np.float32),
        np.array(Y_sev, dtype=object).astype(str),
        np.array(Y_mode, dtype=object).astype(str)
    )

Xtr_spec, Xtr_aux, ytr_rul, ytr_health, ytr_sev, ytr_mode = make_windows(df_tr)
Xva_spec, Xva_aux, yva_rul, yva_health, yva_sev, yva_mode = make_windows(df_va)
Xte_spec, Xte_aux, yte_rul, yte_health, yte_sev, yte_mode = make_windows(df_te)

print(f"[WINDOWS] train={len(ytr_rul)}, val={len(yva_rul)}, test={len(yte_rul)}")

# --------------------------
# Data Generator com Class-Balanced Augmentation
# --------------------------
class AugmentedDataGenerator(keras.utils.Sequence):
    """
    Custom data generator com CLASS-BALANCED sampling.
    Garante que cada batch tem representação igual de todas as classes severity.
    """
    def __init__(self, Xspec, Xaux, yrul, yhealth, ysev, ymode, 
                 batch_size=32, augment=True, oversample_factor=5, **kwargs):
        super().__init__(**kwargs)
        self.Xspec_base = Xspec
        self.Xaux_base = Xaux
        self.yrul_base = yrul
        self.yhealth_base = yhealth
        self.ysev_base = ysev
        self.ymode_base = ymode
        self.batch_size = batch_size
        self.augment = augment
        self.oversample_factor = oversample_factor
        
        # Agrupar índices por classe de severity (tarefa mais problemática)
        self.sev_indices = {}
        for sev_class in np.unique(ysev):
            self.sev_indices[sev_class] = np.where(ysev == sev_class)[0].tolist()
        
        self.sev_classes = list(self.sev_indices.keys())
        self.n_sev_classes = len(self.sev_classes)
        
        # Calcular steps garantindo que todas as classes sejam igualmente vistas
        max_samples_per_class = max(len(idxs) for idxs in self.sev_indices.values())
        self.steps_per_epoch = int(np.ceil(max_samples_per_class * self.n_sev_classes * oversample_factor / batch_size))
        
        print(f"[BALANCED_GEN] {self.n_sev_classes} severity classes, {self.steps_per_epoch} steps/epoch")
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # Balanced sampling: igual número de amostras por classe severity
        samples_per_class = max(1, self.batch_size // self.n_sev_classes)
        
        batch_specs, batch_aux = [], []
        batch_rul, batch_health, batch_sev, batch_mode = [], [], [], []
        
        for sev_class in self.sev_classes:
            class_idxs = self.sev_indices[sev_class]
            # Sample com replacement (permite oversampling de classes minoritárias)
            selected = np.random.choice(class_idxs, size=samples_per_class, replace=True)
            
            for sample_idx in selected:
                spec = self.Xspec_base[sample_idx].copy()
                
                # Aplicar augmentation forte
                if self.augment:
                    spec_3d = spec[..., np.newaxis]
                    spec = augment_spectrogram_numpy(spec_3d)
                else:
                    spec = spec[..., np.newaxis]
                
                batch_specs.append(spec)
                batch_aux.append(self.Xaux_base[sample_idx])
                batch_rul.append(self.yrul_base[sample_idx])
                batch_health.append(self.yhealth_base[sample_idx])
                batch_sev.append(self.ysev_base[sample_idx])
                batch_mode.append(self.ymode_base[sample_idx])
        
        return (
            {"spec_input": np.array(batch_specs), "aux_input": np.array(batch_aux)},
            {"rul": np.array(batch_rul), "health": np.array(batch_health),
             "severity": np.array(batch_sev), "mode": np.array(batch_mode)}
        )
    
    def on_epoch_end(self):
        pass  # Sampling já é randomizado

# USAR class-balanced sampling para combater desbalanceamento severo
print(f"[INFO] Usando data augmentation avançado + oversampling inteligente")

# --------------------------
# Encode labels
# --------------------------
def encode_labels(*ys):
    merged = np.concatenate([y for y in ys if y.size > 0], axis=0).astype(str)
    labels = sorted(np.unique(merged).tolist())
    lab2i = {lab:i for i, lab in enumerate(labels)}
    ys_i = [np.array([lab2i[s] for s in y.astype(str)], dtype="int64") for y in ys]
    return labels, ys_i

sev_labels, (ytr_sev_i, yva_sev_i, yte_sev_i) = encode_labels(ytr_sev, yva_sev, yte_sev)
mode_labels, (ytr_mode_i, yva_mode_i, yte_mode_i) = encode_labels(ytr_mode, yva_mode, yte_mode)

n_sev = len(sev_labels)
n_mode = len(mode_labels)
print(f"[CLASSES] Severity={n_sev} {sev_labels}, Mode={n_mode} {mode_labels}")

# Calcular class weights para balancear as classes
from sklearn.utils.class_weight import compute_class_weight
sev_class_weights = compute_class_weight('balanced', classes=np.arange(n_sev), y=ytr_sev_i)
mode_class_weights = compute_class_weight('balanced', classes=np.arange(n_mode), y=ytr_mode_i)
sev_class_weight_dict = {i: w for i, w in enumerate(sev_class_weights)}
mode_class_weight_dict = {i: w for i, w in enumerate(mode_class_weights)}
print(f"[CLASS_WEIGHTS] Severity: {sev_class_weight_dict}")
print(f"[CLASS_WEIGHTS] Mode: {mode_class_weight_dict}")

# --------------------------
# Criar Data Generators
# --------------------------
F_AUX = Xtr_aux.shape[1] if Xtr_aux.ndim == 2 and Xtr_aux.shape[0] > 0 else 1

# Train generator com augmentation e oversampling
train_gen = AugmentedDataGenerator(
    Xtr_spec, Xtr_aux, ytr_rul, ytr_health, ytr_sev_i, ytr_mode_i,
    batch_size=BATCH_SIZE,
    augment=True,  # Aplica augmentation
    oversample_factor=5  # 5x mais samples por epoch com augmentation
)

# Validation generator sem augmentation
val_gen = AugmentedDataGenerator(
    Xva_spec, Xva_aux, yva_rul, yva_health, yva_sev_i, yva_mode_i,
    batch_size=BATCH_SIZE,
    augment=False,
    oversample_factor=1
)

# Test dataset (mantém tf.data para avaliação)
test_ds = tf.data.Dataset.from_tensor_slices((Xte_spec, Xte_aux, yte_rul, yte_health, yte_sev_i, yte_mode_i))
test_ds = test_ds.map(lambda s, a, r, h, sv, m: (
    (tf.expand_dims(s, -1), a),
    {"rul": r, "health": h, "severity": sv, "mode": m}
), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"[DATAGEN] Train batches/epoch: {len(train_gen)} (with 5x oversample + augmentation)")
print(f"[DATAGEN] Val batches/epoch: {len(val_gen)}")

# --------------------------
# Multi-task Model
# --------------------------
def conv_block(x, f, k=3, dropout_rate=0.3):
    """Conv block com regularização forte"""
    x = layers.Conv2D(f, (k, k), padding="same", use_bias=False,
                     kernel_regularizer=keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout em cada bloco
    return x

def build_model(t_window, n_mels, f_aux, n_sev, n_mode):
    spec_in = keras.Input(shape=(t_window, n_mels, 1), name="spec_input")
    aux_in  = keras.Input(shape=(f_aux,), name="aux_input")
    
    # CNN backbone com MAIS capacidade (voltando a arquitetura mais complexa mas com regularização)
    x = conv_block(spec_in, 64, 3, dropout_rate=0.2)  # 64 em vez de 32
    x = conv_block(x, 64, 3, dropout_rate=0.2)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    
    x = conv_block(x, 128, 3, dropout_rate=0.3)  # 128 em vez de 64
    x = conv_block(x, 128, 3, dropout_rate=0.3)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    
    x = conv_block(x, 256, 3, dropout_rate=0.4)  # 256 em vez de 128
    x = conv_block(x, 256, 3, dropout_rate=0.4)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    
    x = conv_block(x, 512, 3, dropout_rate=0.5)  # 512 para mais capacidade
    x = layers.GlobalAveragePooling2D()(x)
    
    # Aux features - MAIS capacidade
    aux = layers.LayerNormalization()(aux_in)
    aux = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(aux)
    aux = layers.Dropout(0.4)(aux)
    aux = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(aux)
    aux = layers.Dropout(0.4)(aux)
    
    # Fusion - MAIS capacidade
    fused = layers.Concatenate()([x, aux])
    fused = layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    fused = layers.Dropout(0.5)(fused)  # Dropout forte
    fused = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    fused = layers.Dropout(0.5)(fused)
    
    # Task-specific heads - MAIS PROFUNDOS para tarefas difíceis
    # RUL (prioridade baixa) - 2 camadas
    rul = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    rul = layers.Dropout(0.3)(rul)
    rul = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(rul)
    rul = layers.Dropout(0.3)(rul)
    rul_out = layers.Dense(1, activation="sigmoid", name="rul")(rul)
    
    # Health (PRIORIDADE MÁXIMA) - 3 camadas
    health = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    health = layers.Dropout(0.4)(health)
    health = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(health)
    health = layers.Dropout(0.4)(health)
    health = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(health)
    health = layers.Dropout(0.3)(health)
    health_out = layers.Dense(1, activation="sigmoid", name="health")(health)
    
    # Severity (PRIORIDADE CRÍTICA) - 4 camadas MAIS PROFUNDAS
    sev = layers.Dense(384, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    sev = layers.Dropout(0.5)(sev)
    sev = layers.Dense(192, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(sev)
    sev = layers.Dropout(0.5)(sev)
    sev = layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(sev)
    sev = layers.Dropout(0.4)(sev)
    sev = layers.Dense(48, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(sev)
    sev = layers.Dropout(0.3)(sev)
    sev_out = layers.Dense(n_sev, activation="softmax", name="severity")(sev)
    
    # Mode (ALTA PRIORIDADE) - 4 camadas para multi-classe difícil
    mode = layers.Dense(384, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(fused)
    mode = layers.Dropout(0.5)(mode)
    mode = layers.Dense(192, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(mode)
    mode = layers.Dropout(0.5)(mode)
    mode = layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(mode)
    mode = layers.Dropout(0.4)(mode)
    mode = layers.Dense(48, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REG))(mode)
    mode = layers.Dropout(0.3)(mode)
    mode_out = layers.Dense(n_mode, activation="softmax", name="mode")(mode)
    
    model = keras.Model(
        inputs=[spec_in, aux_in],
        outputs={"rul": rul_out, "health": health_out, "severity": sev_out, "mode": mode_out},
        name="pump_predictive"
    )
    
    # Label Smoothing CrossEntropy para prevenir overconfidence
    def label_smoothing_ce(y_true, y_pred, smoothing=0.1):
        """Cross-entropy with label smoothing for sparse integer labels."""
        y_true = tf.reshape(y_true, [-1])
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        
        # One-hot com smoothing
        y_true_oh = tf.one_hot(y_true, num_classes)
        y_true_smooth = y_true_oh * (1.0 - smoothing) + (smoothing / tf.cast(num_classes, tf.float32))
        
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred))
    
    # Compile com LOSS WEIGHTS REDUZIDOS + LABEL SMOOTHING
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),  # LR reduzido para estabilidade
        loss={
            "rul": "mse",
            "health": "mse",
            "severity": lambda yt, yp: label_smoothing_ce(yt, yp, 0.1),  # Label smoothing!
            "mode": lambda yt, yp: label_smoothing_ce(yt, yp, 0.1)
        },
        loss_weights={
            "rul": 0.5,
            "health": 12.0,   # Reduzido de 50 → 12
            "severity": 10.0, # Reduzido de 20 → 10 (PRIORIDADE MÁXIMA)
            "mode": 6.0       # Reduzido de 15 → 6
        },
        metrics={
            "rul": ["mae"],
            "health": ["mae"],
            "severity": ["accuracy"],
            "mode": ["accuracy"]
        }
    )
    return model

model = build_model(T_WINDOW, N_MELS, F_AUX, n_sev, n_mode)
model.summary()

# --------------------------
# Train
# --------------------------
# Train com Generators
# --------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, mode="min"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=15, min_lr=1e-8, verbose=1, mode="min"),  # Mais agressivo
    keras.callbacks.TerminateOnNaN(),
    # Monitorar health MAE especificamente
    keras.callbacks.ReduceLROnPlateau(monitor="val_health_mae", factor=0.5, patience=20, min_lr=1e-8, verbose=1, mode="min"),
]

history = model.fit(
    train_gen,  # Usar generator em vez de dataset
    validation_data=val_gen,  # Usar generator para validation também
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# --------------------------
# Evaluate
# --------------------------
results = model.evaluate(test_ds, verbose=0, return_dict=True)
print("\n[TEST RESULTS]")
print(f"  RUL MAE: {results.get('rul_mae', 0):.4f}")
# Converter health MAE de volta para percentual (estava em 0-1)
health_mae_normalized = results.get('health_mae', 0)
health_mae_percent = health_mae_normalized * 100.0
print(f"  Health MAE: {health_mae_percent:.2f}%")
print(f"  Severity accuracy: {results.get('severity_accuracy', 0):.4f}")
print(f"  Mode accuracy: {results.get('mode_accuracy', 0):.4f}")

# Confusion matrices
def collect_predictions(ds):
    y_true_sev, y_pred_sev = [], []
    y_true_mode, y_pred_mode = [], []
    for (xs, ys) in ds:
        preds = model.predict(xs, verbose=0)
        y_pred_sev.append(np.argmax(preds["severity"], axis=1))
        y_pred_mode.append(np.argmax(preds["mode"], axis=1))
        y_true_sev.append(ys["severity"].numpy())
        y_true_mode.append(ys["mode"].numpy())
    if not y_true_sev:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(y_true_sev), np.concatenate(y_pred_sev),
            np.concatenate(y_true_mode), np.concatenate(y_pred_mode))

yt_sev, yp_sev, yt_mode, yp_mode = collect_predictions(test_ds)

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm_sev = confusion_matrix(yt_sev, yp_sev, n_sev)
cm_mode = confusion_matrix(yt_mode, yp_mode, n_mode)

print("\n[Severity Confusion Matrix]")
print(cm_sev)
print("\n[Mode Confusion Matrix]")
print(cm_mode)

# --------------------------
# Save
# --------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

labels_data = {
    "severity": sev_labels,
    "mode": mode_labels
}
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(labels_data, f, ensure_ascii=False, indent=2)

report = {
    "test_results": {k: float(v) for k, v in results.items()},
    "rul_mae": float(results.get("rul_mae", 0)),
    "health_mae": float(results.get("health_mae", 0)),
    "severity_accuracy": float(results.get("severity_accuracy", 0)),
    "mode_accuracy": float(results.get("mode_accuracy", 0)),
    "labels": labels_data,
    "confusion_matrices": {
        "severity": cm_sev.tolist(),
        "mode": cm_mode.tolist()
    },
    "config": {
        "primary_signal": primary_col,
        "seq_len": SEQ_LEN,
        "t_window": T_WINDOW,
        "n_mels": N_MELS,
        "aux_features": aux_candidates,
        "n_windows": {"train": int(len(ytr_rul)), "val": int(len(yva_rul)), "test": int(len(yte_rul))}
    }
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Modelo: {MODEL_PATH}")
print(f"[OK] Labels: {LABELS_PATH}")
print(f"[OK] Report: {REPORT_PATH}")
print("\n✓ Modelo preditivo completo para manutenção!")
