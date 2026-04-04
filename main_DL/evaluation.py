"""
Evaluation utilities for DL churn model.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def ensure_results_dir(results_dir: str = "results") -> str:
    """Create result directory if it does not exist."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate standard binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "Accuracy": float(accuracy),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "AUC": float(roc_auc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "fpr": fpr,
        "tpr": tpr,
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics to terminal."""
    print("\n" + "=" * 50)
    print("DL MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy:    {metrics['Accuracy'] * 100:.2f}%")
    print(f"Sensitivity: {metrics['Sensitivity'] * 100:.2f}%")
    print(f"Specificity: {metrics['Specificity'] * 100:.2f}%")
    print(f"AUC:         {metrics['AUC']:.4f}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, results_dir: str) -> None:
    """Save confusion matrix figure."""
    ensure_results_dir(results_dir)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - DL MLP")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_dl.png"), dpi=150)
    plt.close()


def plot_roc_curve(metrics: dict, results_dir: str) -> None:
    """Save ROC curve figure."""
    ensure_results_dir(results_dir)

    plt.figure(figsize=(7, 6))
    plt.plot(
        metrics["fpr"],
        metrics["tpr"],
        lw=2,
        label=f"MLP (AUC = {metrics['AUC']:.4f})",
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - DL MLP")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve_dl.png"), dpi=150)
    plt.close()


def plot_training_history(history, results_dir: str) -> None:
    """Save training history (loss and AUC)."""
    ensure_results_dir(results_dir)

    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict.get("loss", []), label="Train loss")
    plt.plot(epochs, history_dict.get("val_loss", []), label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict.get("auc", []), label="Train AUC")
    plt.plot(epochs, history_dict.get("val_auc", []), label="Val AUC")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history_dl.png"), dpi=150)
    plt.close()


def save_metrics_json(metrics: dict, results_dir: str) -> None:
    """Save scalar metrics as JSON for reporting."""
    ensure_results_dir(results_dir)

    payload = {
        "Accuracy": metrics["Accuracy"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "AUC": metrics["AUC"],
        "TP": metrics["TP"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
    }

    with open(os.path.join(results_dir, "metrics_dl.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def generate_all_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    history,
    results_dir: str,
) -> None:
    """Generate all evaluation artifacts."""
    plot_confusion_matrix(y_true, y_pred, results_dir)
    plot_roc_curve(metrics, results_dir)
    plot_training_history(history, results_dir)
    save_metrics_json(metrics, results_dir)

    print(f"Saved evaluation outputs to: {results_dir}")
