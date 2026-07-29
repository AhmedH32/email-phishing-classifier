from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, float]:
    """Calculates standardized classification metrics for ablation comparison."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def print_metrics_summary(exp_name: str, metrics: Dict[str, float]):
    """Prints formatted metrics block to console."""
    print(f"\n==========================================")
    print(f"📊 ABLATION METRICS: {exp_name}")
    print(f"==========================================")
    print(f"  ├─ Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  ├─ Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  ├─ Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  ├─ F1-Score:  {metrics['f1'] * 100:.2f}%")
    if "roc_auc" in metrics:
        print(f"  └─ ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"==========================================\n")