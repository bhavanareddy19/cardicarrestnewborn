"""Evaluation metrics: AUC, accuracy, precision, recall, F1, confusion matrix."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.utils import to_categorical

from src.config import CLASS_NAMES, NUM_CLASSES


class MetricsCalculator:
    """Compute comprehensive evaluation metrics for multi-class classification."""

    def compute_multiclass_auc(
        self,
        y_true_onehot: np.ndarray,
        y_pred_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-class and macro/weighted One-vs-Rest AUC.

        Args:
            y_true_onehot: One-hot encoded true labels (N, 3).
            y_pred_probs: Predicted probabilities (N, 3).

        Returns:
            Dict with per-class AUC, macro AUC, and weighted AUC.
        """
        per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = roc_auc_score(
                y_true_onehot[:, i], y_pred_probs[:, i]
            )

        macro = roc_auc_score(
            y_true_onehot, y_pred_probs, multi_class="ovr", average="macro"
        )
        weighted = roc_auc_score(
            y_true_onehot, y_pred_probs, multi_class="ovr", average="weighted"
        )

        return {"per_class": per_class, "macro": macro, "weighted": weighted}

    def full_report(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
    ) -> Dict:
        """Generate a complete evaluation report.

        Args:
            y_true: Integer label array (N,).
            y_pred_probs: Predicted probabilities (N, 3).

        Returns:
            Dict with accuracy, precision, recall, F1, AUC breakdown,
            confusion matrix, and classification report string.
        """
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_onehot = to_categorical(y_true, NUM_CLASSES)

        auc_results = self.compute_multiclass_auc(y_true_onehot, y_pred_probs)

        return {
            "accuracy": accuracy_score(y_true, y_pred_classes),
            "precision_macro": precision_score(
                y_true, y_pred_classes, average="macro"
            ),
            "recall_macro": recall_score(
                y_true, y_pred_classes, average="macro"
            ),
            "f1_macro": f1_score(y_true, y_pred_classes, average="macro"),
            "auc": auc_results,
            "confusion_matrix": confusion_matrix(y_true, y_pred_classes),
            "classification_report": classification_report(
                y_true, y_pred_classes, target_names=CLASS_NAMES
            ),
        }
