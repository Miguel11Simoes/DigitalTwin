
# train_tf.py
# Treino genérico para TODOS os sensores numéricos e TODOS os modes do sensors_log.csv
# - Split temporal (evita fuga de informação)
# - Features de janela (mean/std/min/max/slope) calculadas por split (sem inflacionar)
# - Permutation importance (val e test) gravada em JSON/PNG
# - Bloco all_classes_overview no eval_report.json com base em models/labels_master.json (se existir)
# - Compatível com Python 3.10+ e TF 2.16+ (Keras 3)
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# -----------------------------
# Config
# -----------------------------
CSV_PATH = os.environ.get("DT_CSV", "logs/sensors_log.csv")
OUT_DIR = Path(os.environ.get("DT_MODELS_DIR", "models"))
MODEL_PATH = OUT_DIR / "pump_mode_classifier.keras"
SCALER_PATH = OUT_DIR / "scaler.json"
LABELS_PATH = OUT_DIR / "labels.json"
REPORT_PATH = OUT_DIR / "eval_report.json"
LABELS_MASTER_PATH = OUT_DIR / "labels_master.json"  # <— novo
# proporções temporais (treino/val/test em ordem cronológica)
TRAIN_RATIO = float(os.environ.get("DT_TRAIN_RATIO", 0.70))
VAL_RATIO   = float(os.environ.get("DT_VAL_RATIO",   0.15))
TEST_RATIO  = float(os.environ.get("DT_TEST_RATIO",  0.15))
EPOCHS = int(os.environ.get("DT_EPOCHS", 80))
BATCH_SIZE = int(os.environ.get("DT_BATCH_SIZE", 256))
SEED = int(os.environ.get("DT_SEED", 42))
# janela para features temporais (em nº de amostras)
ROLL_WIN = int(os.environ.get("DT_ROLL_WIN", 60))
MIN_STD = 1e-8
# reproducibilidade
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.RandomState(SEED)
# sensores “candidatos” — o script usa apenas os que existirem no CSV
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
# colunas “proibidas” (heurísticas/UI) que não entram no treino
DROP_NUMERIC = {
    "anomaly_score", "failure_probability",
    "health_index", "rul_minutes", "model_confidence",
    "predicted_mode"  # se o backend gravar esta coluna
}
# -----------------------------
# Helpers
# -----------------------------
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
def build_mlp(input_dim: int, n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(512, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.20)(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.10)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
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
    """
    Gera estatísticas da janela para as colunas disponíveis (mean/std/min/max/slope).
    Retorna df cortado (drop primeiras win-1 linhas).
    """
    df = df_in.copy()
    # extras derivadas
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
    # corta as primeiras linhas sem janela completa
    df_out = df_out.iloc[win-1:].reset_index(drop=True)
    return df_out
def plot_training(h):
    acc = h.history.get('accuracy', [])
    val_acc = h.history.get('val_accuracy', [])
    loss = h.history.get('loss', [])
    val_loss = h.history.get('val_loss', [])
    epochs = range(1, len(acc)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(epochs, acc, 'o-', label='train'); plt.plot(epochs, val_acc, 'o-', label='val')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend()
    plt.subplot(1,2,2); plt.plot(epochs, loss, 'o-', label='train'); plt.plot(epochs, val_loss, 'o-', label='val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()
# -----------------------------
# Carregar CSV
# -----------------------------
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"Não encontrei {CSV_PATH}")
print(f"A ler CSV: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn", skipinitialspace=True)
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", on_bad_lines="warn",
                     skipinitialspace=True, encoding="latin1")
if "mode" not in df.columns:
    raise ValueError("CSV precisa de uma coluna 'mode' com as classes.")
if "timestamp" not in df.columns:
    df["timestamp"] = np.arange(len(df))  # fallback: índice temporal sintético
# parse/ordenar por tempo
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
# força numérico no resto
for c in df.columns:
    if c not in ("mode", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
# remove linhas com mode nulo
df = df.dropna(subset=["mode"]).reset_index(drop=True)
df["mode"] = df["mode"].astype(str)
# -----------------------------
# Split temporal POR CLASSE (70/15/15 dentro de cada 'mode')
# -----------------------------
def temporal_split_per_class(df, train_ratio=0.70, val_ratio=0.15):
    parts = []
    for lab, dfg in df.groupby("mode", sort=False):
        dfg = dfg.sort_values("timestamp")
        n = len(dfg)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        n_test  = max(1, n - n_train - n_val)
        df_tr = dfg.iloc[:n_train].copy()
        df_va = dfg.iloc[n_train:n_train + n_val].copy()
        df_te = dfg.iloc[n_train + n_val:].copy()
        df_tr["__split__"] = "train"; df_va["__split__"] = "val"; df_te["__split__"] = "test"
        parts += [df_tr, df_va, df_te]
    out = pd.concat(parts, ignore_index=True)
    return (
        out[out["__split__"] == "train"].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"] == "val"  ].drop(columns="__split__").reset_index(drop=True),
        out[out["__split__"] == "test" ].drop(columns="__split__").reset_index(drop=True),
    )
df_train, df_val, df_test = temporal_split_per_class(df, TRAIN_RATIO, VAL_RATIO)
print(f"Split temporal por classe: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
print("Train counts:\n", df_train["mode"].value_counts())
print("Val counts:\n",   df_val["mode"].value_counts())
print("Test counts:\n",  df_test["mode"].value_counts())
for name, dframe in [("train", df_train), ("val", df_val), ("test", df_test)]:
    missing = [c for c in df["mode"].unique() if c not in set(dframe["mode"])]
    if missing:
        print(f"[WARN] {name} sem estas classes:", missing)
# -----------------------------
# Features de janela (por split, sem inflacionar)
# -----------------------------
present_sensors = [s for s in CANDIDATE_SENSORS if s in df.columns and s not in DROP_NUMERIC]
df_train_w = add_window_feats(df_train, present_sensors, ROLL_WIN)
df_val_w   = add_window_feats(df_val,   present_sensors, ROLL_WIN)
df_test_w  = add_window_feats(df_test,  present_sensors, ROLL_WIN)
# -----------------------------
# Seleção de features
# -----------------------------
def choose_features(dfw: pd.DataFrame) -> List[str]:
    cols = []
    for c in dfw.columns:
        if c in ("mode", "timestamp"):
            continue
        if c in DROP_NUMERIC:
            continue
        if pd.api.types.is_numeric_dtype(dfw[c]):
            cols.append(c)
    return cols
feature_cols = choose_features(df_train_w)
if not feature_cols:
    raise ValueError("Sem features numéricas após engenharia de janelas.")
def clean_split(dfw: pd.DataFrame, feature_cols: List[str]):
    X = dfw[feature_cols].astype("float32").copy()
    # troca ±Inf por NaN e remove linhas com qualquer NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & dfw["mode"].notna()
    X = X[mask]
    y = dfw.loc[mask, "mode"].astype(str)
    return X.values, y.values
# … depois de definires feature_cols e label_to_int:
X_train, y_train_str = clean_split(df_train_w, feature_cols)
X_val,   y_val_str   = clean_split(df_val_w,   feature_cols)
X_test,  y_test_str  = clean_split(df_test_w,  feature_cols)
# garantir que os rótulos mapeiam
labels = sorted(pd.unique(
    pd.concat([pd.Series(y_train_str), pd.Series(y_val_str), pd.Series(y_test_str)]).dropna()
).tolist())
label_to_int = {lab: i for i, lab in enumerate(labels)}
y_train = np.array([label_to_int[s] for s in y_train_str], dtype="int64")
y_val   = np.array([label_to_int[s] for s in y_val_str],   dtype="int64")
y_test  = np.array([label_to_int[s] for s in y_test_str],  dtype="int64")
print(f"Total de amostras: {len(df)}")
print(f"Nº de features (sensores + janelas): {len(feature_cols)}")
print("Features (exemplo):", ", ".join(feature_cols[: min(40, len(feature_cols))]))
print("Classes (modes):", labels)
# -----------------------------
# Scaler (fit no treino)
# -----------------------------
mean, std = standardize_fit(X_train)
X_train = standardize_transform(X_train, mean, std)
X_val   = standardize_transform(X_val,   mean, std)
X_test  = standardize_transform(X_test,  mean, std)
# -----------------------------
# Modelo
# -----------------------------
n_classes = len(labels)
model = build_mlp(input_dim=X_train.shape[1], n_classes=n_classes)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
]
cw = class_weights_from_counts(y_train, n_classes)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw,
    callbacks=callbacks,
    verbose=2,
)
# -----------------------------
# Avaliação
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_test, y_pred, n_classes)
metrics_pc = per_class_metrics(cm)
print("\nConfusion matrix (linhas = verdade, colunas = predito):")
with np.printoptions(linewidth=160):
    print(cm)
print("\nMétricas por classe:")
for i, lab in enumerate(labels):
    m = metrics_pc[i]
    print(f"- {lab:20s} | precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} support={m['support']}")
# =============================
# Permutation Importance (val/test)
# =============================
from collections import defaultdict
def _accuracy(model, X, y, batch_size=1024):
    probs = model.predict(X, verbose=0, batch_size=batch_size)
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))
def permutation_importance(model,
                           X: np.ndarray,
                           y: np.ndarray,
                           n_repeats: int = 5,
                           batch_size: int = 1024,
                           rng_seed: int = 42) -> Tuple[Dict[int, Dict[str, float]], float]:
    rng_local = np.random.RandomState(rng_seed)
    base_acc = _accuracy(model, X, y, batch_size=batch_size)
    n_features = X.shape[1]
    drops = defaultdict(list)
    X_work = X.copy()
    for j in range(n_features):
        col = X[:, j].copy()
        for _ in range(n_repeats):
            rng_local.shuffle(X_work[:, j])          # baralha só esta coluna
            acc = _accuracy(model, X_work, y, batch_size=batch_size)
            drops[j].append(max(0.0, base_acc - acc))
            X_work[:, j] = col                        # repõe a coluna
    imp = {int(j): {"mean_drop": float(np.mean(vals)), "std_drop": float(np.std(vals))}
           for j, vals in drops.items()}
    return imp, base_acc
def _rank_importances(imp_dict: Dict[int, Dict[str, float]], feature_names: List[str]):
    items = []
    for j, stats in imp_dict.items():
        items.append({
            "feature": feature_names[int(j)],
            "mean_drop": stats["mean_drop"],
            "std_drop": stats["std_drop"]
        })
    items.sort(key=lambda x: x["mean_drop"], reverse=True)
    return items
print("\n[Permutation Importance] a calcular no validation set...")
imp_val_raw, base_val_acc = permutation_importance(model, X_val, y_val, n_repeats=5, batch_size=1024, rng_seed=SEED)
imp_val = _rank_importances(imp_val_raw, feature_cols)
print(f"Accuracy (val) baseline: {base_val_acc:.4f}")
print("Top-20 (val):")
for row in imp_val[:20]:
    print(f"  {row['feature']:<40s} Δacc={row['mean_drop']:.4f} ±{row['std_drop']:.4f}")
print("\n[Permutation Importance] a calcular no test set...")
imp_test_raw, base_test_acc = permutation_importance(model, X_test, y_test, n_repeats=5, batch_size=1024, rng_seed=SEED+1)
imp_test = _rank_importances(imp_test_raw, feature_cols)
print(f"Accuracy (test) baseline: {base_test_acc:.4f}")
print("Top-20 (test):")
for row in imp_test[:20]:
    print(f"  {row['feature']:<40s} Δacc={row['mean_drop']:.4f} ±{row['std_drop']:.4f}")
# guardar importâncias
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "feature_importance_val.json", "w", encoding="utf-8") as f:
    json.dump(imp_val, f, ensure_ascii=False, indent=2)
with open(OUT_DIR / "feature_importance_test.json", "w", encoding="utf-8") as f:
    json.dump(imp_test, f, ensure_ascii=False, indent=2)
# gráfico rápido (top-25 no test)
try:
    topk = min(25, len(imp_test))
    names = [r["feature"] for r in imp_test[:topk]][::-1]
    vals  = [r["mean_drop"] for r in imp_test[:topk]][::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(names, vals)
    plt.xlabel("Accuracy drop (permutation)")
    plt.title("Permutation importance — test (top-25)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_test_top25.png", dpi=150)
except Exception as e:
    print("[WARN] Falhou plot de importâncias:", e)
# -----------------------------
# Guardar artefactos
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
with open(SCALER_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"feature_names": feature_cols,
         "mean": mean.squeeze().tolist(),
         "std": std.squeeze().tolist()},
        f, ensure_ascii=False, indent=2
    )
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
# -------- NOVO: bloco all_classes_overview a partir de labels_master.json --------
all_classes_overview = None
if LABELS_MASTER_PATH.exists():
    try:
        with open(LABELS_MASTER_PATH, "r", encoding="utf-8") as f:
            master = json.load(f)
        master = [str(x) for x in master]
        seen = set(labels)
        unseen = [x for x in master if x not in seen]
        all_classes_overview = {
            "total_possible": len(master),
            "seen_in_training": labels,
            "unseen": unseen
        }
    except Exception as e:
        print(f"[WARN] labels_master.json inválido: {e}")
# preparar report (serializável)
report = {
    "test_accuracy": float(test_acc) if np.isfinite(test_acc) else None,
    "classes": labels,
    "class_weights": {labels[i]: float(cw.get(i, 0.0)) for i in range(n_classes)},
    "confusion_matrix": cm.tolist(),
    "per_class": {labels[i]: metrics_pc[i] for i in range(n_classes)},
    "n_samples": {
        "total": int(len(df)),
        "train": int(len(df_train_w)),
        "val": int(len(df_val_w)),
        "test": int(len(df_test_w)),
    },
    "features": feature_cols,
    "roll_window": ROLL_WIN,
    "splits": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO},
    "permutation_importance": {
        "val": imp_val,
        "test": imp_test
    },
    "all_classes_overview": all_classes_overview  # <— aparece se labels_master existir
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nModelo guardado em: {MODEL_PATH}")
print(f"Scaler guardado em: {SCALER_PATH}")
print(f"Labels    em: {LABELS_PATH}")
print(f"Relatório  em: {REPORT_PATH}")
# -----------------------------
# Plots do treino
# -----------------------------
plot_training(history)
