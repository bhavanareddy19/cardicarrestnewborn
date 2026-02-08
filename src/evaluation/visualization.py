"""Visualization utilities: ROC curves, confusion matrix, training history."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical

from src.config import CLASS_NAMES, LOGS_DIR, NUM_CLASSES


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot One-vs-Rest ROC curves for all classes.

    Args:
        y_true: Integer labels (N,).
        y_pred_probs: Predicted probabilities (N, 3).
        save_path: If provided, save plot to this path.
    """
    y_true_onehot = to_categorical(y_true, NUM_CLASSES)

    plt.figure(figsize=(10, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("One-vs-Rest ROC Curves -- Ensemble Prediction", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap confusion matrix.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        save_path: If provided, save plot to this path.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix -- Ensemble Prediction", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ensemble_comparison(
    model_names: List[str],
    individual_aucs: List[float],
    ensemble_auc: float,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing individual model AUCs vs ensemble AUC.

    Args:
        model_names: List of model names.
        individual_aucs: AUC for each individual model.
        ensemble_auc: AUC for the ensemble.
        save_path: If provided, save plot to this path.
    """
    names = model_names + ["ENSEMBLE"]
    aucs = individual_aucs + [ensemble_auc]
    colors = ["#3498db"] * len(model_names) + ["#e74c3c"]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(names, aucs, color=colors, edgecolor="black", linewidth=0.5)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Macro AUC", fontsize=12)
    plt.title("Individual Model AUC vs Ensemble AUC", fontsize=14)
    plt.ylim(min(aucs) - 0.05, 1.0)

    for bar, val in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(
    histories: Dict[str, dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot training/validation loss curves for multiple models.

    Args:
        histories: Dict mapping model name -> keras history.history dict.
        save_path: If provided, save plot to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, hist in histories.items():
        if "loss" in hist:
            axes[0].plot(hist["loss"], label=name, alpha=0.7)
        if "val_loss" in hist:
            axes[1].plot(hist["val_loss"], label=name, alpha=0.7)

    axes[0].set_title("Training Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
