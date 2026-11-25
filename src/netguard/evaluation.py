from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def evaluate_binary_classifier(
    model: XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    attack_categories: pd.Series | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive binary NIDS metrics and per-category breakdown."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if attack_categories is not None:
        cat_series = pd.Series(attack_categories).reset_index(drop=True)
        breakdown: dict[str, dict[str, Any]] = {}
        for category in sorted(cat_series.unique()):
            mask = (cat_series == category).to_numpy()
            total_cat = int(mask.sum())
            pred_attack = int((y_pred[mask] == 1).sum())
            pred_normal = int((y_pred[mask] == 0).sum())

            if category == "Normal":
                correct_rate = pred_normal / total_cat if total_cat > 0 else 0.0
            else:
                correct_rate = pred_attack / total_cat if total_cat > 0 else 0.0

            breakdown[str(category)] = {
                "total": total_cat,
                "predicted_attack": pred_attack,
                "predicted_normal": pred_normal,
                "correct_rate": round(float(correct_rate), 6),
            }
        metrics["category_breakdown"] = breakdown

    return metrics
