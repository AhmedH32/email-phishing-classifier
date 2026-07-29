import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Computes core classification metrics for phishing evaluation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }
    return metrics


def print_metrics_summary(title: str, metrics: Dict[str, float]) -> None:
    """Prints formatted summary of performance metrics."""
    print("\n" + "=" * 50)
    print(f"📊 {title} Summary")
    print("=" * 50)
    print(f"  • Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall:    {metrics['recall']:.4f}")
    print(f"  • F1 Score:  {metrics['f1']:.4f}")
    print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 50)


def save_experiment_artifacts(
    output_dir: str,
    history: Dict[str, List[float]],
    final_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    exp_name: str = "exp1",
) -> None:
    """Saves plots and structured JSON results for ablation study documentation."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save JSON Metrics Log
    metrics_payload = {
        "experiment": exp_name,
        "final_metrics": final_metrics,
        "history": history,
    }
    json_path = os.path.join(output_dir, f"metrics_{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"📄 Metrics log saved to {json_path}")

    # 2. Plot Training & Validation Loss/F1 Curves
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(
        epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4"
    )
    ax1.plot(
        epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e"
    )
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(
        epochs, history["val_f1"], "d-", label="Val F1 Score", color="#2ca02c"
    )
    ax2.set_title("Validation F1 Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "loss_f1_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"📈 Loss curves saved to {curve_path}")

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{cm[i, j]:,}",
                va="center",
                ha="center",
                size="large",
                weight="bold",
            )

    plt.title(f"Confusion Matrix ({exp_name.upper()})", pad=20)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"])
    ax.set_yticklabels(["Legitimate (0)", "Phishing (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to {cm_path}")

    # 4. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="#d62728",
        lw=2,
        label=f"Model (AUC = {final_metrics['roc_auc']:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({exp_name.upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"📉 ROC curve saved to {roc_path}")