from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

class EvaluationMetrics:
    """
    Computes rigorous biomedical classification metrics for binary and multiclass tasks.
    Safely handles edge cases, zero division, and class imbalances.
    """

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        confidence_level: float = 0.95,
        n_bootstraps: int = 200
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true, dtype=np.int32)
        y_prob = np.asarray(y_prob, dtype=np.float32)
        y_pred = np.argmax(y_prob, axis=1)

        num_classes = len(class_names)
        
        # 1. Overall Metrics
        acc = float(accuracy_score(y_true, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

        # Precision, recall, f1
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # 2. Per-class metrics
        prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0
        )

        # Compute per-class specificity
        # specificity = TN / (TN + FP)
        conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        specificity_per_class = []
        for i in range(num_classes):
            tn = np.sum(np.delete(np.delete(conf_mat, i, axis=0), i, axis=1))
            fp = np.sum(conf_mat[:, i]) - conf_mat[i, i]
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(spec)

        per_class_results = {}
        for idx, name in enumerate(class_names):
            per_class_results[name] = {
                "precision": float(prec_per_class[idx]),
                "recall_sensitivity": float(rec_per_class[idx]),
                "specificity": float(specificity_per_class[idx]),
                "f1_score": float(f1_per_class[idx]),
                "support": int(support[idx])
            }

        # 3. ROC-AUC and PR-AUC
        roc_auc_macro = None
        roc_curves_data = {}
        pr_curves_data = {}

        try:
            if num_classes == 2:
                # Binary
                roc_auc_macro = float(roc_auc_score(y_true, y_prob[:, 1]))
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_curves_data["Demented"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc_macro}
                
                prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
                pr_auc_val = float(auc(rec, prec))
                pr_curves_data["Demented"] = {"precision": prec.tolist(), "recall": rec.tolist(), "auc": pr_auc_val}
            else:
                # Multiclass (One-vs-Rest)
                y_true_onehot = np.eye(num_classes)[y_true]
                roc_auc_macro = float(roc_auc_score(y_true_onehot, y_prob, multi_class="ovr", average="macro"))

                for i, name in enumerate(class_names):
                    if np.sum(y_true_onehot[:, i]) > 0:
                        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
                        class_auc = float(auc(fpr, tpr))
                        roc_curves_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": class_auc}
                        
                        prec, rec, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
                        pr_auc_val = float(auc(rec, prec))
                        pr_curves_data[name] = {"precision": prec.tolist(), "recall": rec.tolist(), "auc": pr_auc_val}
        except Exception:
            roc_auc_macro = 0.0

        # 4. Bootstrap Confidence Interval for Macro-F1
        boot_f1s = []
        rng = np.random.default_rng(42)
        n_samples = len(y_true)
        if n_samples > 10:
            for _ in range(n_bootstraps):
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                b_true = y_true[boot_idx]
                b_pred = y_pred[boot_idx]
                _, _, bf1, _ = precision_recall_fscore_support(b_true, b_pred, average="macro", zero_division=0)
                boot_f1s.append(bf1)
            alpha = (1.0 - confidence_level) / 2.0
            ci_lower = float(np.percentile(boot_f1s, alpha * 100))
            ci_upper = float(np.percentile(boot_f1s, (1.0 - alpha) * 100))
        else:
            ci_lower, ci_upper = float(f1_macro), float(f1_macro)

        return {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc_macro": roc_auc_macro,
            "f1_macro_ci_95": (ci_lower, ci_upper),
            "per_class": per_class_results,
            "confusion_matrix": conf_mat.tolist(),
            "roc_curves": roc_curves_data,
            "pr_curves": pr_curves_data
        }

    @classmethod
    def plot_confusion_matrix(cls, conf_matrix: List[List[int]], class_names: List[str], title: str = "Confusion Matrix") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = np.array(conf_matrix)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylabel("Ground Truth", fontsize=10)
        ax.set_xlabel("Model Prediction", fontsize=10)
        plt.tight_layout()
        return fig

    @classmethod
    def plot_roc_curves(cls, roc_data: Dict[str, Any], title: str = "ROC Curves (One-vs-Rest)") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cname, data in roc_data.items():
            fpr = data["fpr"]
            tpr = data["tpr"]
            c_auc = data["auc"]
            ax.plot(fpr, tpr, label=f"{cname} (AUC = {c_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax.set_title(title, fontsize=12, pad=12)
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
