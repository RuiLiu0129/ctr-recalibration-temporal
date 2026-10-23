"""Training / prediction helpers and metric computation shared by all arms."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .calibration import clip_probs, expected_calibration_error
from .models import build_model


def evaluate_predictions(y: np.ndarray, p: np.ndarray, ece_bins: int = 15) -> dict:
    """AUC, log loss and ECE for one (labels, probabilities) pair."""
    y = np.asarray(y)
    p = clip_probs(p)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, ece_bins),
    }


def train_model(model_name: str, encoder, X_train, y_train, cfg: dict,
                max_train_rows: int | None = None, seed: int = 42):
    """Build and fit one model; optionally subsample the training rows first.

    In the Plan-A streaming pipeline the training-row cap is applied during
    data preparation (``data_pipeline``), so ``max_train_rows`` is normally
    None and X_train / y_train are read-only memmaps passed straight to the
    model's streaming ``fit``.
    """
    if max_train_rows is not None and len(X_train) > max_train_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=max_train_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    model = build_model(model_name, encoder, cfg)
    model.fit(X_train, y_train)
    return model
