# train_tf.py — versão com CNN 2D (log-Mel) + late-fusion tabular e fallback para MLP
from __future__ import annotations
import json, os, math, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==============================
# Config
# ==============================
CSV_PATH = os.environ.get("DT_CSV", "logs/sensors_log.csv")
OUT_DIR = Path(os.environ.get("DT_MODELS_DIR", "models"))
MODEL_MODE_ONLY = OUT_DIR / "pump_mode_classifier.keras"       # compatível com backend atual
MODEL_MULTITASK = OUT_DIR / "pump_cnn_multitask.keras"         # modo+severidade (futuro)
SCALER_PATH = OUT_DIR / "scaler.json"
LABELS_PATH = OUT_DIR / "labels.json"
REPORT_PATH = OUT_DIR / "eval_report.json"
LABELS_MASTER_PATH = OUT_DIR / "labels_master.json"

# proporções temporais
TRAIN_RATIO = float(os.environ.get("DT_TRAIN_RATIO", 0.70))
VAL_RATIO   = float(os.environ.get("DT_VAL_RATIO",   0.15))
TEST_RATIO  = float(os.environ.get("DT_TEST_RATIO",  0.15))

# treino
EPOCHS = int(os.environ.get("DT_EPOCHS", 60))
BATCH_SIZE = int(os.environ.get("DT_BATCH_SIZE", 128))
SEED = int(os.environ.get("DT_SEED", 42))

# engenharia de janelas (apenas para MLP fallback e/ou enriquecer tabular)
ROLL_WIN = int(os.environ.get("DT_ROLL_WIN", 60))
MIN_STD = 1e-8

# CNN/audio
USE_CNN = int(os.environ.get("DT_USE_CNN", 1)) == 1     # tenta usar CNN se possível
FS = int(os.environ.get("DT_FS", 10000))                # Hz
WIN = int(os.environ.get("DT_STFT_WIN", 1024))
HOP = int(os.environ.get("DT_STFT_HOP", 256))
N_MELS = int(os.environ.get("DT_N_MELS", 96))
FMIN = float(os.environ.get("DT_FMIN", 5.0))
FMAX = float(os.environ.get("DT_FMAX", 4000.0))
FIXED_SAMPLES = int(os.environ.get("DT_FIXED_SAMPLES", 20480))  # ~2.05 s a 10 kHz

# reproducibilidade
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.RandomState(SEED)

# sensores candidatos (o script usa os que existirem no CSV)
CANDIDATE_SENSORS = [
    "temperature","pressure","flow","overall_vibration","ultrasonic_noise","motor_current",
    "density","viscosity","ferrous_particles","rpm",
    "bearing_temp_DE","bearing_temp_NDE","casing_temp",
    "suction_pressure","discharge_pressure","delta_p",
    "gas_volume_fraction",
    "current_A","current_B","current_C","power_factor","frequency","torque_est",
    "oil_temp","oil_water_ppm","particle_count","oil_TAN",
    "seal_temp","seal_flush_pressure","leakage_rate",
    "shaft_displacement","noise_dBA",
]

# colunas proibidas (derivadas/heurísticas/UI)
DROP_NUMERIC = {
    "anomaly_score","failure_probability","health_index","rul_minutes","model_confidence",
    "predicted_mode"
}

# ==============================
# Helpers gerais
# ==============================
def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, ddof=0, keepdims=True)
    std  = np.where(std < MIN_STD, 1.0, std)
    return mean, std

def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std

def class_weights_from_counts(y_int: np.ndarray, n_classes: int) -> Dict[int, float]:
    counts = np.bincount(y_int, minlength=n_classes).astype(float)
    return {i: (len(y_int) / (n_classes * counts[i])) if counts[i] > 0 else 0.0 for i in range(n_classes)}

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
    n = cm.shape[0]; out = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        out.append(dict(precision=prec, recall=rec, f1=f1, support=int(cm[i, :].sum())))
    return out

def add_window_feats(df_in: pd.DataFrame, sensors: List[str], win: int) -> pd.DataFrame:
    df = df_in.copy()
    extras = {}
    if {"discharge_pressure","suction_pressure"}.issubset(df.columns):
        extras["delta_p_rt"] = df["discharge_pressure"] - df["suction_pressure"]
    if {"motor_current","flow"}.issubset(df.columns):
        extras["amps_per_flow"] = df["motor_current"] / df["flow"].replace(0, 1e-6)
    base_cols = [c for c in sensors if c in df.columns]
    if extras:
        for k, v in extras.items():
            df[k] = v
        base_cols += list(extras.keys())
    if not base_cols:
        return df
    roll = df[base_cols].rolling(window=win, min_periods=win)
    m = roll.mean().add_suffix(f"_m{win}")
    s = roll.std().add_suffix(f"_s{win}")
    mn = roll.min().add_suffix(f"_min{win}")
    mx = roll.max().add_suffix(f"_max{win}")
    sl = df[base_cols].diff(win).div(float(win)).add_suffix(f"_slope{win}")
    df_out = pd.concat([df, m, s, mn, mx, sl], axis=1)
    df_out = df_out.iloc[win-1:].reset_index(drop=True)
    return df_out

def plot_training(h, title="Training"):
    acc = h.history.get('mode_accuracy', h.history.get('accuracy', []))
    val_acc = h.history.get('val_mode_accuracy', h.history.get('val_accuracy', []))
    loss = h.history.get('loss', [])
    val_loss = h.history.get('val_loss', [])
    epochs = range(1, len(loss)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(epochs, acc, 'o-', label='train'); plt.plot(epochs, val_acc, 'o-', label='val')
    plt.title('Accuracy (mode)'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend()
    plt.subplot(1,2,2); plt.plot(epochs, loss, 'o-', label='train'); plt.plot(epochs, val_loss, 'o-', label='val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()

# ==============================
# Leitura CSV
# ==============================
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"Não encontrei {CSV_PATH}")
print(f"A ler CSV: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True)
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn",
                     skipinitialspace=True, encoding="latin1")

if "mode" not in df.columns:
    raise ValueError("CSV precisa de uma coluna 'mode'.")

# timestamp
if "timestamp" not in df.columns: df["timestamp"] = np.arange(len(df))
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# tipos numéricos no resto
for c in df.columns:
    if c not in ("mode","timestamp","wave_path"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["mode"]).reset_index(drop=True)
df["mode"] = df["mode"].astype(str)

# ------------------------------
# Severidade (opcional) a partir do sufixo _early/_moderate/_severe
# ------------------------------
def extract_mode_and_sev(s: str) -> Tuple[str, str]:
    s = str(s).lower()
    for tag in ("_early","_moderate","_severe"):
        if s.endswith(tag):
            return s.replace(tag,""), tag[1:]
    return s, "moderate"  # default

base_modes, severities = zip(*[extract_mode_and_sev(m) for m in df["mode"]])
df["mode_base"] = list(base_modes)
df["severity"] = list(severities)

# ==============================
# Split temporal por classe (usando 'mode_base')
# ==============================
def temporal_split_per_class(df, train_ratio=0.70, val_ratio=0.15):
    parts = []
    for lab, dfg in df.groupby("mode_base", sort=False):
        dfg = dfg.sort_values("timestamp")
        n = len(dfg)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        df_tr = dfg.iloc[:n_train].copy()
        df_va = dfg.iloc[n_train:n_train+n_val].copy()
        df_te = dfg.iloc[n_train+n_val:].copy()
        df_tr["__split__"]="train"; df_va["__split__"]="val"; df_te["__split__"]="test"
        parts += [df_tr, df_va, df_te]
    out = pd.concat(parts, ignore_index=True)
    return (
        out[out["__split__"]=="train"].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"]=="val"  ].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"]=="test" ].drop(columns="__split__").reset_index(drop=True),
    )

df_train, df_val, df_test = temporal_split_per_class(df, TRAIN_RATIO, VAL_RATIO)
print(f"Split: train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
for name, dfx in [("train", df_train),("val", df_val),("test", df_test)]:
    missing = [c for c in df["mode_base"].unique() if c not in set(dfx["mode_base"])]
    if missing: print(f"[WARN] {name} sem classes: {missing}")

# ==============================
# Features tabulares (sempre geradas)
# ==============================
present_sensors = [s for s in CANDIDATE_SENSORS if s in df.columns and s not in DROP_NUMERIC]
df_train_w = add_window_feats(df_train, present_sensors, ROLL_WIN)
df_val_w   = add_window_feats(df_val,   present_sensors, ROLL_WIN)
df_test_w  = add_window_feats(df_test,  present_sensors, ROLL_WIN)

def choose_features(dfw: pd.DataFrame) -> List[str]:
    cols = []
    for c in dfw.columns:
        if c in ("mode","mode_base","severity","timestamp","wave_path"): continue
        if c in DROP_NUMERIC: continue
        if pd.api.types.is_numeric_dtype(dfw[c]): cols.append(c)
    return cols

feature_cols = choose_features(df_train_w)
if not feature_cols:
    raise ValueError("Sem features numéricas após engenharia de janelas.")

def df_to_Xy(dfw: pd.DataFrame, feature_cols: List[str]):
    X = dfw[feature_cols].astype("float32").replace([np.inf,-np.inf], np.nan)
    mask = X.notna().all(axis=1) & dfw["mode_base"].notna() & dfw["severity"].notna()
    X = X[mask]
    y_mode = dfw.loc[mask, "mode_base"].astype(str).values
    y_sev  = dfw.loc[mask, "severity"].astype(str).values
    wpaths = dfw.loc[mask, "wave_path"].astype(str).values if "wave_path" in dfw.columns else np.array([""]*mask.sum())
    return X.values, y_mode, y_sev, wpaths

X_tr, y_tr_mode_str, y_tr_sev_str, W_tr = df_to_Xy(df_train_w, feature_cols)
X_va, y_va_mode_str, y_va_sev_str, W_va = df_to_Xy(df_val_w,   feature_cols)
X_te, y_te_mode_str, y_te_sev_str, W_te = df_to_Xy(df_test_w,  feature_cols)

# Labels
labels_mode = sorted(pd.unique(pd.Series(np.concatenate([y_tr_mode_str,y_va_mode_str,y_te_mode_str]))))
labels_sev  = ["early","moderate","severe"]
lab2int_mode = {l:i for i,l in enumerate(labels_mode)}
lab2int_sev  = {l:i for i,l in enumerate(labels_sev)}
y_tr_mode = np.array([lab2int_mode[s] for s in y_tr_mode_str], dtype="int64")
y_va_mode = np.array([lab2int_mode[s] for s in y_va_mode_str], dtype="int64")
y_te_mode = np.array([lab2int_mode[s] for s in y_te_mode_str], dtype="int64")
y_tr_sev  = np.array([lab2int_sev[s]  for s in y_tr_sev_str ], dtype="int64")
y_va_sev  = np.array([lab2int_sev[s]  for s in y_va_sev_str ], dtype="int64")
y_te_sev  = np.array([lab2int_sev[s]  for s in y_te_sev_str ], dtype="int64")

print("Classes (modes):", labels_mode)
print("Classes (severity):", labels_sev)
print(f"Features: {len(feature_cols)}")

# Scaler (fit no treino)
mean, std = standardize_fit(X_tr)
X_tr = standardize_transform(X_tr, mean, std)
X_va = standardize_transform(X_va, mean, std)
X_te = standardize_transform(X_te, mean, std)

# ==============================
# Áudio / vibração — loaders
# ==============================
def load_wave_any(path: str, target_len: int = FIXED_SAMPLES, fs: int = FS) -> np.ndarray:
    if not path or not isinstance(path, str) or not Path(path).exists():
        return np.zeros((target_len,), dtype=np.float32)
    p = Path(path)
    try:
        if p.suffix.lower() == ".npy":
            x = np.load(p).astype(np.float32).squeeze()
        elif p.suffix.lower() == ".wav":
            raw = tf.io.read_file(str(p))
            wav, rate = tf.audio.decode_wav(raw)
            x = tf.squeeze(wav, axis=-1).numpy().astype(np.float32)
            if int(rate.numpy()) != fs and len(x) > 0:
                # reamostragem simples (nearest) — suficiente p/ treino; podes trocar por resample sinc se precisares
                idx = np.linspace(0, len(x)-1, int(len(x)*fs/float(rate.numpy())), dtype=np.int64)
                x = x[idx]
        else:
            return np.zeros((target_len,), dtype=np.float32)
    except Exception:
        return np.zeros((target_len,), dtype=np.float32)
    # pad/truncate
    if len(x) >= target_len:
        return x[:target_len]
    out = np.zeros((target_len,), dtype=np.float32)
    out[:len(x)] = x
    return out

def stft_logmel(batch_wave):  # batch_wave: [B, T]
    stft = tf.signal.stft(batch_wave, frame_length=WIN, frame_step=HOP,
                          fft_length=WIN, window_fn=tf.signal.hann_window)
    mag = tf.abs(stft) + 1e-8
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=WIN//2 + 1,
        sample_rate=FS, lower_edge_hertz=FMIN, upper_edge_hertz=FMAX
    )
    power = tf.square(mag)
    mel = tf.tensordot(power, mel_w, axes=1)
    mel.set_shape(mag.shape[:-1].concatenate([N_MELS]))
    logmel = tf.math.log(mel + 1e-6)                 # [B, frames, Mels]
    logmel = tf.transpose(logmel, [0,2,1])           # [B, Mels, frames]
    logmel = tf.expand_dims(logmel, -1)              # [B, Mels, frames, 1]
    return logmel

# detetar se dá para usar CNN
HAS_WAVES = ("wave_path" in df_train_w.columns) and (any([Path(p).exists() for p in W_tr[:50]]))
USE_CNN_EFF = USE_CNN and HAS_WAVES
if USE_CNN and not HAS_WAVES:
    print("[WARN] USE_CNN=1 mas não há wave_path válido — a usar MLP tabular.")

# ==============================
# Modelos
# ==============================
def build_mlp(input_dim: int, n_modes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(512, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.20)(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.10)(x)
    out_mode = layers.Dense(n_modes, activation="softmax", name="mode")(x)
    # pequena cabeça de severidade opcional (ajuda a regularizar, mas não quebra compatibilidade)
    out_sev  = layers.Dense(3, activation="softmax", name="severity")(x)
    model = keras.Model(inputs=inputs, outputs=[out_mode, out_sev])
    losses = {"mode":"sparse_categorical_crossentropy", "severity":"sparse_categorical_crossentropy"}
    lw = {"mode":1.0, "severity":0.3}
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss=losses, loss_weights=lw,
                  metrics={"mode":["accuracy"], "severity":["accuracy"]})
    return model

def build_cnn2d_multitask(n_modes: int, tab_dim: int) -> keras.Model:
    spec_in = keras.Input(shape=(N_MELS, None, 1), name="spec")
    tab_in  = keras.Input(shape=(tab_dim,), name="tabular")

    def block(x, f):
        x = layers.Conv2D(f, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        return x

    x = spec_in
    x = block(x, 32); x = block(x, 32); x = layers.MaxPool2D((2,2))(x)
    x = block(x, 64); x = block(x, 64); x = layers.MaxPool2D((2,2))(x)
    x = block(x, 128); x = block(x, 128); x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(192, (3,3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)  # [B, 192]

    t = layers.LayerNormalization()(tab_in)
    t = layers.Dense(64, activation="relu")(t)
    t = layers.Dense(64, activation="relu")(t)

    h = layers.Concatenate()([x, t])
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.30)(h)

    mode_out     = layers.Dense(n_modes, activation="softmax", name="mode")(h)
    severity_out = layers.Dense(3, activation="softmax", name="severity")(h)

    model = keras.Model([spec_in, tab_in], [mode_out, severity_out])
    losses = {
        "mode": "sparse_categorical_crossentropy",
        "severity": "sparse_categorical_crossentropy",
    }
    lw = {"mode": 1.0, "severity": 0.4}
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=losses, loss_weights=lw,
        metrics={"mode":["accuracy"], "severity":["accuracy"]}
    )
    return model

# ==============================
# Datasets
# ==============================
def make_ds_mlp(X, y_mode, y_sev, batch=BATCH_SIZE, train=True):
    ds = tf.data.Dataset.from_tensor_slices((X, {"mode": y_mode, "severity": y_sev}))
    if train: ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def make_ds_cnn(X_tab, W_paths, y_mode, y_sev, batch=BATCH_SIZE, train=True):
    def gen():
        for xt, wp, ym, ys in zip(X_tab, W_paths, y_mode, y_sev):
            w = load_wave_any(wp, FIXED_SAMPLES, FS)  # np.float32 [T]
            yield (xt.astype(np.float32), w.astype(np.float32), np.int64(ym), np.int64(ys))
    output_signature = (
        tf.TensorSpec(shape=(X_tab.shape[1],), dtype=tf.float32),
        tf.TensorSpec(shape=(FIXED_SAMPLES,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if train: ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    def map_to_inputs(tab, wave, ym, ys):
        spec = stft_logmel(tf.expand_dims(wave, 0))  # [1, M, F, 1]
        spec = tf.squeeze(spec, axis=0)              # [M, F, 1]
        return ({"spec": spec, "tabular": tab}, {"mode": ym, "severity": ys})
    ds = ds.map(map_to_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================
# Treino
# ==============================
OUT_DIR.mkdir(parents=True, exist_ok=True)

if USE_CNN_EFF:
    print(">> Treino CNN 2D + Tabular (multitarefa)")
    model = build_cnn2d_multitask(n_modes=len(labels_mode), tab_dim=X_tr.shape[1])
    ds_tr = make_ds_cnn(X_tr, W_tr, y_tr_mode, y_tr_sev, train=True)
    ds_va = make_ds_cnn(X_va, W_va, y_va_mode, y_va_sev, train=False)
    callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_mode_accuracy",
        mode="max",              # <-- acrescenta isto
        patience=8,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    ),
        ]

    history = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    # avaliação (modo)
    ds_te = make_ds_cnn(X_te, W_te, y_te_mode, y_te_sev, train=False)
    eval_out = model.evaluate(ds_te, return_dict=True, verbose=0)
    test_acc_mode = float(eval_out.get("mode_accuracy", 0.0))
    test_acc_sev  = float(eval_out.get("severity_accuracy", 0.0))
else:
    print(">> Treino MLP tabular (fallback)")
    model = build_mlp(input_dim=X_tr.shape[1], n_modes=len(labels_mode))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_mode_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]
    ds_tr = make_ds_mlp(X_tr, y_tr_mode, y_tr_sev, train=True)
    ds_va = make_ds_mlp(X_va, y_va_mode, y_va_sev, train=False)
    history = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    # avaliação (modo)
    ds_te = make_ds_mlp(X_te, y_te_mode, y_te_sev, train=False)
    eval_out = model.evaluate(ds_te, return_dict=True, verbose=0)
    test_acc_mode = float(eval_out.get("mode_accuracy", eval_out.get("accuracy", 0.0)))
    test_acc_sev  = float(eval_out.get("severity_accuracy", 0.0))

# ==============================
# Predições no test (para matriz confusão)
# ==============================
def predict_modes_on_test():
    if USE_CNN_EFF:
        # iterar test para obter logits de modo
        preds = []
        for batch in ds_te:
            out = model.predict(batch[0], verbose=0)
            probs = out[0]  # "mode"
            preds.append(np.argmax(probs, axis=1))
        y_pred = np.concatenate(preds)
    else:
        # ds_te já é (X -> {mode,severity})
        probs = model.predict(ds_te, verbose=0)[0]
        y_pred = np.argmax(probs, axis=1)
    return y_pred

y_pred_mode = predict_modes_on_test()
cm_mode = confusion_matrix(y_te_mode, y_pred_mode, n_classes=len(labels_mode))
metrics_pc_mode = per_class_metrics(cm_mode)

# ==============================
# Permutation Importance (tabular only)
# ==============================
from collections import defaultdict
def _accuracy_mode(model, Xtab, y, batch_size=512):
    if USE_CNN_EFF:
        # usar W_te real, spec fixo, só baralhamos tabular X
        preds = []
        def ds_from_X(Xtmp):
            return make_ds_cnn(Xtmp, W_te, y, y_te_sev, train=False)
        base_ds = ds_from_X(Xtab)
        for b in base_ds:
            out = model.predict(b[0], verbose=0)
            probs = out[0]
            preds.append(np.argmax(probs, axis=1))
        pred = np.concatenate(preds)
        return float(np.mean(pred == y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((Xtab,)).batch(batch_size)
        probs = model.predict(ds, verbose=0)[0]
        pred = np.argmax(probs, axis=1)
        return float(np.mean(pred == y))

def permutation_importance_tab(model, Xtab, y, n_repeats=3, batch_size=512, rng_seed=SEED):
    rng_local = np.random.RandomState(rng_seed)
    base_acc = _accuracy_mode(model, Xtab, y, batch_size)
    n_features = Xtab.shape[1]
    drops = defaultdict(list)
    Xw = Xtab.copy()
    for j in range(n_features):
        orig = Xw[:, j].copy()
        for _ in range(n_repeats):
            rng_local.shuffle(Xw[:, j])
            acc = _accuracy_mode(model, Xw, y, batch_size)
            drops[j].append(max(0.0, base_acc - acc))
            Xw[:, j] = orig
    imp = {int(j): {"mean_drop": float(np.mean(vals)), "std_drop": float(np.std(vals))}
           for j, vals in drops.items()}
    return imp, base_acc

print("\n[Permutation Importance] tabular no test set…")
imp_test_raw, base_test_acc = permutation_importance_tab(model, X_te, y_te_mode, n_repeats=3, batch_size=512, rng_seed=SEED+1)
def _rank_importances(imp_dict: Dict[int, Dict[str, float]], feature_names: List[str]):
    items = []
    for j, stats in imp_dict.items():
        items.append({"feature": feature_names[int(j)], "mean_drop": stats["mean_drop"], "std_drop": stats["std_drop"]})
    items.sort(key=lambda x: x["mean_drop"], reverse=True)
    return items
imp_test = _rank_importances(imp_test_raw, feature_cols)
try:
    topk = min(25, len(imp_test))
    names = [r["feature"] for r in imp_test[:topk]][::-1]
    vals  = [r["mean_drop"] for r in imp_test[:topk]][::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(names, vals)
    plt.xlabel("Accuracy drop (permutation) — mode")
    plt.title("Permutation importance — tabular (test, top-25)")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "feature_importance_test_top25.png", dpi=150)
except Exception as e:
    print("[WARN] Falhou plot de importâncias:", e)

# ==============================
# Guardar modelos e artefactos
# ==============================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Modelo multitarefa (seja MLP ou CNN)
model.save(MODEL_MULTITASK)

# 2) Versão MODE-ONLY compatível (mesmo backbone, só cabeça de modo)
def export_mode_only_from(model_mt: keras.Model) -> keras.Model:
    # identifica a saída 'mode' e reconstrói um model com a(s) mesma(s) entrada(s)
    if isinstance(model_mt.output, (list, tuple)):
        # assume que outputs[0] é "mode"
        mode_out = model_mt.outputs[0]
    else:
        mode_out = model_mt.output
    model_mode = keras.Model(inputs=model_mt.inputs, outputs=mode_out)
    # compila só para poder avaliar/usar facilmente no backend se necessário
    model_mode.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model_mode

model_mode_only = export_mode_only_from(model)
model_mode_only.save(MODEL_MODE_ONLY)

# scaler/labels
with open(SCALER_PATH, "w", encoding="utf-8") as f:
    json.dump({"feature_names": feature_cols,
               "mean": mean.squeeze().tolist(),
               "std": std.squeeze().tolist()}, f, ensure_ascii=False, indent=2)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(labels_mode, f, ensure_ascii=False, indent=2)

# labels master overview (opcional)
all_classes_overview = None
if Path(LABELS_MASTER_PATH).exists():
    try:
        master = [str(x) for x in json.loads(Path(LABELS_MASTER_PATH).read_text(encoding="utf-8"))]
        seen = set(labels_mode)
        unseen = [x for x in master if x not in seen]
        all_classes_overview = {
            "total_possible": len(master),
            "seen_in_training": labels_mode,
            "unseen": unseen
        }
    except Exception as e:
        print(f"[WARN] labels_master.json inválido: {e}")

# relatório
report = {
    "use_cnn": bool(USE_CNN_EFF),
    "test_accuracy_mode": float(test_acc_mode),
    "test_accuracy_severity": float(test_acc_sev),
    "classes_mode": labels_mode,
    "classes_severity": labels_sev,
    "confusion_matrix_mode": cm_mode.tolist(),
    "per_class_mode": {labels_mode[i]: m for i, m in enumerate(per_class_metrics(cm_mode))},
    "n_samples": {
        "total": int(len(df)),
        "train": int(len(df_train_w)),
        "val": int(len(df_val_w)),
        "test": int(len(df_test_w)),
    },
    "features": feature_cols,
    "roll_window": ROLL_WIN,
    "splits": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO},
    "permutation_importance_tabular_test": imp_test,
    "all_classes_overview": all_classes_overview
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nModelos guardados em:\n - {MODEL_MODE_ONLY}\n - {MODEL_MULTITASK}")
print(f"Scaler em: {SCALER_PATH}")
print(f"Labels  em: {LABELS_PATH}")
print(f"Relatório em: {REPORT_PATH}")

# gráfico do treino (mostra metrica de modo)
plot_training(history, title="Training (mode)")