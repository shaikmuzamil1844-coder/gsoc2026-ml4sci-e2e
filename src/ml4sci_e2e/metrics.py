from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def inverse_fpr_at_target_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.70,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float("inf") if fpr[idx] <= 0 else float(1.0 / fpr[idx])


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    y_pred = (y_score > threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
