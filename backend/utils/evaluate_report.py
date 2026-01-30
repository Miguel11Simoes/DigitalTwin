
# evaluate_report.py
# Lê models/eval_report.json e gera:
# - impressão das métricas
# - confusion_matrix.png (normalizável)
# - f1_per_class.png
# - metrics_per_class.csv
#
# Uso:
#   python evaluate_report.py
#   python evaluate_report.py --report models/eval_report.json --normalize row --show
#   python evaluate_report.py --outdir artefacts --dpi 200
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------
# Utils
# -----------------------------
def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei o relatório: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def ensure_labels_and_cm(report: dict) -> Tuple[List[str], np.ndarray]:
    # labels
    labels = report.get("classes")
    if not labels:
        # tenta usar as chaves de per_class (ordem alfabética)
        per_class = report.get("per_class", {})
        labels = sorted(list(per_class.keys()))
    # confusion matrix
    cm = np.array(report.get("confusion_matrix", []), dtype=float)
    if cm.size == 0:
        raise ValueError("Relatório não contém 'confusion_matrix'. Recorre o treino para gerar.")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Matriz de confusão não é quadrada: {cm.shape}")
    if len(labels) != cm.shape[0]:
        # tenta alinhar com tamanho da matriz
        labels = labels[: cm.shape[0]]
    return labels, cm
def normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cm
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "row":
            sums = cm.sum(axis=1, keepdims=True)
            out = np.divide(cm, sums, where=sums != 0)
        elif mode == "col":
            sums = cm.sum(axis=0, keepdims=True)
            out = np.divide(cm, sums, where=sums != 0)
        elif mode == "all":
            total = cm.sum()
            out = cm / total if total > 0 else cm
        else:
            out = cm
    return out
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    out_png: Path,
    normalize: str = "none",
    dpi: int = 150,
    show: bool = False,
):
    cm_norm = normalize_cm(cm, normalize)
    fig_w = max(8, min(16, 0.6 * len(labels)))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm_norm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_ix = np.arange(len(labels))
    plt.xticks(tick_ix, labels, rotation=45, ha="right")
    plt.yticks(tick_ix, labels)
    # anotações
    fmt = ".2f" if normalize != "none" else ".0f"
    thresh = cm_norm.max() / 2 if cm_norm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            txt = format(val, fmt)
            plt.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8,
                color="white" if val > thresh else "black",
            )
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    if show:
        plt.show()
    plt.close()
def plot_f1_bar(
    per_class: Dict[str, Dict[str, float]],
    labels: List[str],
    title: str,
    out_png: Path,
    dpi: int = 150,
    show: bool = False,
):
    f1_vals = [float(per_class.get(c, {}).get("f1", 0.0)) for c in labels]
    fig_h = max(5, min(12, 0.35 * len(labels)))
    plt.figure(figsize=(10, fig_h))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, f1_vals)
    plt.yticks(y_pos, labels)
    plt.xlabel("F1-score")
    plt.title(title)
    plt.xlim(0, 1)
    for i, v in enumerate(f1_vals):
        plt.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    if show:
        plt.show()
    plt.close()
def save_metrics_csv(per_class: Dict[str, Dict[str, float]], labels: List[str], out_csv: Path):
    # guarda em CSV: classe, precision, recall, f1, support
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for c in labels:
            m = per_class.get(c, {})
            prec = float(m.get("precision", 0.0))
            rec = float(m.get("recall", 0.0))
            f1 = float(m.get("f1", 0.0))
            sup = int(m.get("support", 0))
            f.write(f"{c},{prec:.6f},{rec:.6f},{f1:.6f},{sup}\n")
# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Avaliar e plotar relatório do modelo.")
    ap.add_argument("--report", type=str, default="models/eval_report.json", help="Caminho para eval_report.json")
    ap.add_argument("--outdir", type=str, default=None, help="Diretoria de saída (por defeito, a do report)")
    ap.add_argument("--normalize", type=str, choices=["none", "row", "col", "all"], default="none",
                    help="Normalização da matriz de confusão")
    ap.add_argument("--dpi", type=int, default=150, help="DPI para as figuras")
    ap.add_argument("--show", action="store_true", help="Mostrar figuras na janela gráfica")
    args = ap.parse_args()
    report_path = Path(args.report)
    report = load_report(report_path)
    labels, cm = ensure_labels_and_cm(report)
    outdir = Path(args.outdir) if args.outdir else report_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    test_acc = float(report.get("test_accuracy", float("nan")))
    per_class = report.get("per_class", {})
    print("\n=== Avaliação do Modelo ===")
    print(f"Relatório: {report_path}")
    print(f"Acurácia (teste): {test_acc:.3f}")
    ns = report.get("n_samples", {})
    if ns:
        print(f"Amostras: total={ns.get('total','?')} | train={ns.get('train','?')} | val={ns.get('val','?')} | test={ns.get('test','?')}")
    print("\nMétricas por classe:")
    for c in labels:
        m = per_class.get(c, {})
        print(f"- {c:30s} | prec={m.get('precision',0):.3f} rec={m.get('recall',0):.3f} f1={m.get('f1',0):.3f} support={int(m.get('support',0))}")
    # Figuras
    cm_png = outdir / ("confusion_matrix" + (f"_{args.normalize}" if args.normalize != "none" else "") + ".png")
    plot_confusion_matrix(
        cm, labels,
        title=f"Matriz de Confusão ({args.normalize})",
        out_png=cm_png,
        normalize=args.normalize,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"[OK] Confusion matrix -> {cm_png}")
    f1_png = outdir / "f1_per_class.png"
    plot_f1_bar(per_class, labels, title="F1 por classe (teste)", out_png=f1_png, dpi=args.dpi, show=args.show)
    print(f"[OK] F1 barplot -> {f1_png}")
    # CSV com métricas por classe
    out_csv = outdir / "metrics_per_class.csv"
    save_metrics_csv(per_class, labels, out_csv)
    print(f"[OK] CSV métricas -> {out_csv}")
    # Opcional: guarda uma cópia “flat” dos metadados úteis
    meta_json = outdir / "report_meta.json"
    meta = {
        "test_accuracy": test_acc,
        "classes": labels,
        "n_features": len(report.get("features", [])),
        "roll_window": report.get("roll_window"),
        "splits": report.get("splits"),
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Meta -> {meta_json}")
if __name__ == "__main__":
    main()


