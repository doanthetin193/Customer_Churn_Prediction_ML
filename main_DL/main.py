"""
Entry point for deep learning churn pipeline (MLP).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import explore_data, load_data
from data_preprocessing import preprocess_data
from evaluation import calculate_metrics, generate_all_outputs, print_metrics
from models_dl import find_optimal_threshold, predict_mlp, train_mlp


def main() -> dict:
    """Run the full DL experiment pipeline."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - DEEP LEARNING (MLP)")
    print("=" * 60)

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    print("\nSTEP 1: Loading data")
    df = load_data()
    explore_data(df)

    # Always use class_weight (no SMOTE) for fair comparison with ML branch
    # and to match the paper's approach of preserving natural class distribution.
    use_smote = False

    print("\nSTEP 2: Preprocessing data")
    data = preprocess_data(df, use_smote=use_smote)

    print("\nSTEP 3: Building train/validation split")
    X_train, X_val, y_train, y_val = train_test_split(
        data["X_train"],
        data["y_train"],
        test_size=0.2,
        random_state=42,
        stratify=data["y_train"],
    )

    # Focal loss (alpha=0.75) handles class imbalance directly — no class_weight needed.
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    print("\nSTEP 4: Training MLP")
    model, history = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weight=None,
        epochs=200,
        batch_size=32,
        patience=30,
    )

    print("\nSTEP 4b: Finding optimal classification threshold on validation set")
    optimal_threshold, youden_j = find_optimal_threshold(model, X_val, y_val)

    print("\nSTEP 5: Evaluating on test set")
    y_pred, y_prob = predict_mlp(model, data["X_test"], threshold=optimal_threshold)
    metrics = calculate_metrics(data["y_test"], y_pred, y_prob)
    metrics["threshold"] = optimal_threshold
    metrics["youden_j"] = youden_j
    print_metrics(metrics)
    print(f"  Threshold used: {optimal_threshold:.2f}")

    print("\nSTEP 6: Saving artifacts")
    generate_all_outputs(data["y_test"], y_pred, metrics, history, results_dir)

    model_path = os.path.join(results_dir, "mlp_churn_model.keras")
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    print("\nPipeline completed successfully.")
    return metrics


if __name__ == "__main__":
    main()
