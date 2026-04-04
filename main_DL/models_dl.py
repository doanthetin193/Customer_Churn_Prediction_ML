"""
Deep learning model definitions and utilities.

Improvements over baseline:
- Wider architecture: 256 → 128 → 64 → 32
- LeakyReLU activations (avoid dying ReLU)
- L2 kernel regularization on all Dense layers
- Label smoothing in BinaryCrossentropy loss
- Optimal threshold search: maximize Youden's J on validation set
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for main_DL. Install dependencies from main_DL/requirements.txt"
    ) from exc


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(
    input_dim: int,
    hidden_units: tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.4,
    learning_rate: float = 5e-4,
    l2_lambda: float = 3e-4,
    label_smoothing: float = 0.0,
) -> keras.Model:
    """Build an improved feed-forward neural network for binary classification.

    Architecture:
        Input → [Dense → BN → LeakyReLU → Dropout] x3 → Output

    Key improvements vs baseline:
        - 3-block depth (256→128→64): fewer parameters, less overfitting
        - LeakyReLU(α=0.1) prevents dead neurons
        - Higher L2 (3e-4) + Dropout (0.4) to close val/test gap
        - Lower LR (5e-4) for more stable convergence
    """
    regularizer = keras.regularizers.l2(l2_lambda)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            # Block 1
            keras.layers.Dense(hidden_units[0], kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dropout(dropout_rate),
            # Block 2
            keras.layers.Dense(hidden_units[1], kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dropout(dropout_rate),
            # Block 3
            keras.layers.Dense(hidden_units[2], kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dropout(dropout_rate * 0.5),
            # Output
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def find_optimal_threshold(
    model: keras.Model, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float, float]:
    """Find the classification threshold that maximises Youden's J statistic
    on the validation set.

    Youden's J = Sensitivity + Specificity - 1

    Returns (best_threshold, best_J).
    """
    y_prob = model.predict(X_val, verbose=0).reshape(-1)
    thresholds = np.arange(0.05, 0.95, 0.01)

    best_j, best_t = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)

    print(f"  Optimal threshold: {best_t:.2f}  (Youden's J = {best_j:.4f})")
    return best_t, best_j


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: dict[int, float] | None = None,
    epochs: int = 150,
    batch_size: int = 64,
    patience: int = 20,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train MLP with early stopping and LR scheduling."""
    set_seed(42)

    model = build_mlp(input_dim=X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=max(5, patience // 4),
            min_lr=5e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def predict_mlp(
    model: keras.Model, X: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and probabilities."""
    y_prob = model.predict(X, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob

