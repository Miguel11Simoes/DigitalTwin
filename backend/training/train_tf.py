# train_tf.py
# CNN (Conv2D em espectrograma) + features auxiliares por janela
# - Só usa CNN (sem MLP tabular antigo)
# - Sem dependências externas (usa tf.signal)
# - tf.data correto: ((spec[T,N,1], aux[F]), y)
# - Guarda modelo/labels/report em models/

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# Config (env overrides)
# --------------------------
CSV_PATH      = os.environ.get("DT_CSV", "logs/sensors_log.csv")
OUT_DIR       = Path(os.environ.get("DT_MODELS_DIR", "models"))
MODEL_PATH    = OUT_DIR / "pump_mode_cnn.keras"
LABELS_PATH   = OUT_DIR / "labels.json"
REPORT_PATH   = OUT_DIR / "eval_report.json"

# split temporal por classe
TRAIN_RATIO = float(os.environ.get("DT_TRAIN_RATIO", 0.70))
VAL_RATIO   = float(os.environ.get("DT_VAL_RATIO",   0.15))
TEST_RATIO  = float(os.environ.get("DT_TEST_RATIO",  0.15))

# janela deslizante em amostras da série primária
SEQ_LEN   = int(os.environ.get("DT_SEQ_LEN",   1024))   # comprimento da janela em amostras
HOP       = int(os.environ.get("DT_HOP",       512))    # avanço entre janelas
# STFT -> espectrograma
FRAME_LEN = int(os.environ.get("DT_FRAME_LEN", 64))
FRAME_HOP = int(os.environ.get("DT_FRAME_HOP", 32))
# reshape final do espectrograma para a CNN
T_WINDOW  = int(os.environ.get("DT_T_WINDOW", 128))
N_MELS    = int(os.environ.get("DT_N_MELS",   128))     # “mels” aqui = bins reamostrados (sem banco mel)

# treino
EPOCHS      = int(os.environ.get("DT_EPOCHS", 40))
BATCH_SIZE  = int(os.environ.get("DT_BATCH_SIZE", 64))
SEED        = int(os.environ.get("DT_SEED", 42))
LR          = float(os.environ.get("DT_LR", 1e-3))

np.random.seed(SEED)
tf.random.set_seed(SEED)

# candidatos a “canal primário” (ordem de preferência)
PRIMARY_PREF = [
    "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
    "ultrasonic_noise", "motor_current", "pressure", "flow", "temperature"
]

# colunas a ignorar nos auxiliares (UI/derivadas)
DROP_AUX = {
    "anomaly_score", "failure_probability", "health_index", "rul_minutes", "model_confidence",
    "predicted_mode"
}

# --------------------------
# IO & preparação
# --------------------------
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"Não encontrei {CSV_PATH}")

print(f"[INFO] A ler: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True)
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn",
                     skipinitialspace=True, encoding="latin1")

if "mode" not in df.columns:
    raise ValueError("CSV precisa de coluna 'mode'.")

if "timestamp" not in df.columns:
    df["timestamp"] = np.arange(len(df))

# normalizar dtypes / ordenar por tempo
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
# força numérico no resto
for c in df.columns:
    if c not in ("mode", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["mode"]).reset_index(drop=True)
df["mode"] = df["mode"].astype(str)

# escolher canal primário
primary_col = None
for c in PRIMARY_PREF:
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
        primary_col = c
        break
if not primary_col:
    # Se não existir nenhum, cria uma série sintética a zero
    primary_col = "synthetic_primary"
    df[primary_col] = 0.0
print(f"[PRIMARY] Usar canal: {primary_col}")

# auxiliares possíveis (numéricos, excluindo primary/timestamp/mode e DROP_AUX)
aux_candidates = [
    c for c in df.columns
    if (c not in (primary_col, "timestamp", "mode"))
    and pd.api.types.is_numeric_dtype(df[c])
    and (c not in DROP_AUX)
]

# --------------------------
# Split temporal por classe
# --------------------------
def temporal_split_per_class(df_in: pd.DataFrame,
                             train_ratio=0.70, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for lab, dfg in df_in.groupby("mode", sort=False):
        dfg = dfg.sort_values("timestamp")
        n = len(dfg)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
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

df_tr, df_va, df_te = temporal_split_per_class(df, TRAIN_RATIO, VAL_RATIO)
print(f"[SPLIT] train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")

# --------------------------
# Janela → espectrograma + aux
# --------------------------
def window_indices(n: int, win: int, hop: int) -> List[Tuple[int,int]]:
    idx = []
    i = 0
    while i + win <= n:
        idx.append((i, i+win))
        i += hop
    return idx

def series_to_spectrogram(x: np.ndarray) -> np.ndarray:
    """
    x: [SEQ_LEN] float32
    devolve spec: [T_WINDOW, N_MELS] float32
    """
    # STFT: [time, freq]
    stft = tf.signal.stft(
        signals=tf.convert_to_tensor(x, dtype=tf.float32),
        frame_length=FRAME_LEN,
        frame_step=FRAME_HOP,
        fft_length=FRAME_LEN,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )  # [T, F]
    mag = tf.abs(stft) + 1e-9
    logmag = tf.math.log(mag)
    # normaliza por janela (z-score)
    mean = tf.reduce_mean(logmag)
    std  = tf.math.reduce_std(logmag) + 1e-6
    norm = (logmag - mean) / std
    # reescala para [T_WINDOW, N_MELS]
    spec_hw = tf.expand_dims(norm, axis=-1)         # [T,F,1]
    spec_hw = tf.image.resize(spec_hw, size=(T_WINDOW, N_MELS), method="bilinear")
    out = tf.squeeze(spec_hw, axis=-1).numpy()      # [T,N]
    return out.astype("float32")

def slope(y: np.ndarray) -> float:
    n = y.shape[0]
    if n < 2: return 0.0
    x = np.arange(n, dtype=np.float32)
    # cov(x,y)/var(x)
    vx = np.var(x)
    if vx < 1e-12: return 0.0
    return float(np.cov(x, y, bias=True)[0,1] / vx)

def make_windows_block(df_block: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Para um bloco (train/val/test):
      - gera janelas sobre primary_col -> espectrograma [T,N]
      - extrai features auxiliares (mean/std/slope) para cada aux_candidates
      - escolhe rótulo (mais frequente na janela)
    retorna: X_spec [W, T, N], X_aux [W, F], y_str [W]
    """
    p = df_block[primary_col].to_numpy(dtype=np.float32)
    n = len(p)
    idxs = window_indices(n, SEQ_LEN, HOP)
    X_spec, X_aux, Y = [], [], []
    for (i0, i1) in idxs:
        seg = p[i0:i1]
        if not np.isfinite(seg).all():  # salta janelas com NaN/Inf
            continue
        # espectrograma
        spec = series_to_spectrogram(seg)  # [T_WINDOW, N_MELS]
        X_spec.append(spec)

        # auxiliares: mean/std/slope por coluna (na mesma janela)
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

        # label: moda na janela (ou último valor válido)
        w_modes = df_block["mode"].iloc[i0:i1].astype(str).to_numpy()
        if w_modes.size == 0:
            continue
        # moda simples
        vals, cnts = np.unique(w_modes, return_counts=True)
        y_lab = vals[np.argmax(cnts)]
        Y.append(y_lab)

    if not X_spec:
        return np.zeros((0, T_WINDOW, N_MELS), dtype=np.float32), \
               np.zeros((0, len(aux_candidates)*3 or 1), dtype=np.float32), \
               np.array([], dtype=str)

    # pad aux para dimensão fixa (caso sem aux_candidates)
    maxF = len(aux_candidates)*3 if aux_candidates else 1
    X_aux = [np.pad(a, (0, maxF - a.shape[0])) for a in X_aux]

    return (
        np.stack(X_spec, axis=0),
        np.stack(X_aux, axis=0),
        np.array(Y, dtype=object).astype(str)
    )

Xtr_spec, Xtr_aux, ytr = make_windows_block(df_tr)
Xva_spec, Xva_aux, yva = make_windows_block(df_va)
Xte_spec, Xte_aux, yte = make_windows_block(df_te)
print(f"[WINDOWS] train={len(ytr)}, val={len(yva)}, test={len(yte)} (T={T_WINDOW})")

# --------------------------
# Labels
# --------------------------
def encode_labels(*ys: np.ndarray) -> Tuple[List[str], List[np.ndarray]]:
    merged = np.concatenate([y for y in ys if y.size > 0], axis=0).astype(str)
    labels = sorted(np.unique(merged).tolist())
    lab2i = {lab:i for i, lab in enumerate(labels)}
    ys_i = [np.array([lab2i[s] for s in y.astype(str)], dtype="int64") for y in ys]
    return labels, ys_i

labels, (ytr_i, yva_i, yte_i) = encode_labels(ytr, yva, yte)
n_classes = len(labels)
print(f"[CLASSES] {n_classes} → {labels}")

# --------------------------
# tf.data
# --------------------------
F_AUX = Xtr_aux.shape[1] if Xtr_aux.ndim == 2 and Xtr_aux.shape[0] > 0 else (Xva_aux.shape[1] if Xva_aux.shape[0] > 0 else (Xte_aux.shape[1] if Xte_aux.shape[0] > 0 else 1))

def to_tf_dataset(X_spec, X_aux, y, batch_size, shuffle=True):
    """
    Converte numpy -> tf.data.Dataset, adicionando canal ao spectrograma.
    Saída: ((spec[T,N,1], aux[F]), y)
    """
    if X_spec.size == 0:
        # dataset vazio mas válido
        ds = tf.data.Dataset.from_tensors(((tf.zeros((T_WINDOW, N_MELS, 1), tf.float32),
                                            tf.zeros((F_AUX,), tf.float32)), tf.constant(0, tf.int64)))
        return ds.take(0)

    ds = tf.data.Dataset.from_tensor_slices((X_spec, X_aux, y))
    ds = ds.map(lambda s, a, t: ((tf.expand_dims(s, -1), a), t),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, X_spec.shape[0]))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = to_tf_dataset(Xtr_spec, Xtr_aux, ytr_i, BATCH_SIZE, shuffle=True)
val_ds   = to_tf_dataset(Xva_spec, Xva_aux, yva_i, BATCH_SIZE, shuffle=False)
test_ds  = to_tf_dataset(Xte_spec, Xte_aux, yte_i, BATCH_SIZE, shuffle=False)

# --------------------------
# Modelo (CNN 2D + Aux)
# --------------------------
def conv_block(x, f, k=3):
    x = layers.Conv2D(f, (k, k), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_model(t_window: int, n_mels: int, f_aux: int, n_classes: int) -> keras.Model:
    spec_in = keras.Input(shape=(t_window, n_mels, 1), name="spec")
    aux_in  = keras.Input(shape=(f_aux,), name="aux")

    x = conv_block(spec_in, 32, 3)
    x = conv_block(x, 32, 3)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = conv_block(x, 64, 3)
    x = conv_block(x, 64, 3)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = conv_block(x, 128, 3)
    x = conv_block(x, 128, 3)

    x = layers.GlobalAveragePooling2D()(x)  # [128]

    # cabeça para auxiliares (pequena)
    a = layers.LayerNormalization()(aux_in)
    a = layers.Dense(128, activation="relu")(a)

    h = layers.Concatenate()([x, a])        # [128 + 128]
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.15)(h)

    out = layers.Dense(n_classes, activation="softmax", name="probs")(h)

    model = keras.Model(inputs=[spec_in, aux_in], outputs=out, name="pump_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),  # compatível com TF sem label_smoothing
        metrics=["accuracy"]
    )
    return model

model = build_model(T_WINDOW, N_MELS, F_AUX, n_classes)
model.summary()

# --------------------------
# Treino
# --------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# --------------------------
# Avaliação
# --------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"[TEST] accuracy={test_acc:.4f}")

# Confusion matrix
def collect_all(ds):
    y_true, y_pred = [], []
    for (xs, ys) in ds:
        probs = model.predict(xs, verbose=0)
        yp = np.argmax(probs, axis=1)
        y_true.append(ys.numpy())
        y_pred.append(yp)
    if not y_true:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)

yt, yp = collect_all(test_ds)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix(yt, yp, n_classes)

def per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
    n = cm.shape[0]
    out = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append(dict(precision=prec, recall=rec, f1=f1, support=int(cm[i, :].sum())))
    return out

metrics_pc = per_class_metrics(cm)

print("\nConfusion matrix (linhas=verdade, colunas=pred):")
with np.printoptions(linewidth=180):
    print(cm)
print("\nMétricas por classe:")
for i, lab in enumerate(labels):
    m = metrics_pc[i]
    print(f"- {lab:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")

# --------------------------
# Guardar
# --------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

report = {
    "test_accuracy": float(test_acc),
    "classes": labels,
    "confusion_matrix": cm.tolist(),
    "per_class": {labels[i]: metrics_pc[i] for i in range(n_classes)},
    "primary_signal": primary_col,
    "seq_len": SEQ_LEN,
    "hop": HOP,
    "frame_len": FRAME_LEN,
    "frame_hop": FRAME_HOP,
    "t_window": T_WINDOW,
    "n_mels": N_MELS,
    "aux_features": aux_candidates,
    "splits": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO},
    "n_windows": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))}
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Modelo: {MODEL_PATH}")
print(f"[OK] Labels: {LABELS_PATH}")
print(f"[OK] Report: {REPORT_PATH}")
