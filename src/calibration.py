"""Calibration metrics and recalibration.

- ``expected_calibration_error``: ECE with equal-width bins.
- ``reliability_table``: per-bin counts / mean prediction / empirical rate,
  exported to CSV for reliability diagrams.
- Recalibrators (Platt scaling, isotonic regression) are fitted on a small
  recent validation slice and applied to a test bucket's raw probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-7


def clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per-bin reliability data for one (labels, probabilities) pair."""
    y = np.asarray(y, dtype=np.float64)
    p = clip_probs(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        rows.append({
            "bin": b,
            "bin_lo": edges[b],
            "bin_hi": edges[b + 1],
            "count": count,
            "mean_pred": float(p[mask].mean()) if count else np.nan,
            "frac_pos": float(y[mask].mean()) if count else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / N) * |frac_pos_b - mean_pred_b| over equal-width bins."""
    table = reliability_table(y, p, n_bins)
    total = table["count"].sum()
    if total == 0:
        return float("nan")
    filled = table.dropna(subset=["mean_pred"])
    weights = filled["count"] / total
    return float((weights * (filled["frac_pos"] - filled["mean_pred"]).abs()).sum())


class PlattRecalibrator:
    """Platt scaling: logistic regression on the logit of the raw probability."""

    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=1000)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = clip_probs(p)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, p_calib: np.ndarray, y_calib: np.ndarray) -> "PlattRecalibrator":
        self.clf.fit(self._logit(p_calib), y_calib)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._logit(p))[:, 1]


class IsotonicRecalibrator:
    """Isotonic regression recalibration (out-of-bounds clipped)."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_calib: np.ndarray, y_calib: np.ndarray) -> "IsotonicRecalibrator":
        self.iso.fit(clip_probs(p_calib), y_calib)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.iso.predict(clip_probs(p)), dtype=np.float64)


RECALIBRATORS = {"platt": PlattRecalibrator, "isotonic": IsotonicRecalibrator}


def fit_recalibrator(kind: str, p_calib: np.ndarray, y_calib: np.ndarray):
    """Fit a recalibrator of the given kind on a calibration slice's raw
    predicted probabilities and true labels."""
    if kind not in RECALIBRATORS:
        raise ValueError(f"unknown recalibrator {kind!r} (expected {sorted(RECALIBRATORS)})")
    return RECALIBRATORS[kind]().fit(p_calib, y_calib)
