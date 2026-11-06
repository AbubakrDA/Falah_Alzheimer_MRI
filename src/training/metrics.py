"""NeuroVision Clinical Evaluation Metrics Module

Computes rigorous medical imaging metrics:
- Accuracy & Balanced Accuracy
- Sensitivity (Recall) & Specificity
- Precision & F1-Score (Macro and Per-Class)
- ROC-AUC and PR-AUC (One-vs-Rest for multiclass)
- Confusion Matrix
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes a full battery of clinical diagnostic metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    num_classes = len(class_names) if class_names else len(np.unique(y_true))
    classes = class_names or [f"Class_{i}" for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Compute Sensitivity and Specificity per class (One-vs-Rest)
    per_class_metrics = {}
    sensitivities = []
    specificities = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        c_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        c_f1 = (2 * c_prec * sensitivity) / (c_prec + sensitivity) if (c_prec + sensitivity) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        per_class_metrics[classes[c]] = {
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(c_prec), 4),
            "f1_score": round(float(c_f1), 4),
            "support": int(np.sum(cm[c, :])),
        }

    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))

    # ROC-AUC & PR-AUC
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary classification
                probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc = float(roc_auc_score(y_true, probs_pos))
                pr_auc = float(average_precision_score(y_true, probs_pos))
            else:
                # Multiclass One-vs-Rest
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                pr_auc = float(average_precision_score(
                    np.eye(num_classes)[y_true], y_prob, average="macro"
                ))
        except Exception:
            pass

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "sensitivity_macro": round(macro_sens, 4),
        "specificity_macro": round(macro_spec, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }

