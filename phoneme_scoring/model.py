"""Portable Ridge artifact: numbers in JSON, never untrusted pickle/joblib."""
from __future__ import annotations

import numpy as np

from .common import FEATURE_DIM, FEATURE_VERSION, METHOD_VERSION, PHONES, PREPROCESS_VERSION, ScoringError, dump_json, load_json


def fit_ridge(x, y, alpha=10.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be positive and finite")
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or y.shape != (len(x),) or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Invalid train features or normalized targets")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    # Train-only score-bin frequencies. Square-root balancing with cap 5.
    bins = np.digitize(y, [.25, .75])
    count = np.bincount(bins, minlength=3)
    bin_weights = np.minimum(5, np.sqrt(len(y) / np.maximum(count, 1)))
    weights = bin_weights[bins]
    weights /= weights.mean()
    center_x = np.average(z, weights=weights, axis=0)
    center_y = float(np.average(y, weights=weights))
    a = (z - center_x) * np.sqrt(weights[:, None])
    b = (y - center_y) * np.sqrt(weights)
    coef = np.linalg.solve(a.T @ a + float(alpha) * np.eye(FEATURE_DIM), a.T @ b)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "intercept": float(center_y - center_x @ coef), "alpha": float(alpha), "score_bin_counts_train": count.tolist()}


class RidgeCalibrator:
    def __init__(self, artifact):
        try:
            artifact = load_json(artifact) if isinstance(artifact, (str, bytes)) or hasattr(artifact, "read_text") else artifact
            if artifact["method_version"] not in {METHOD_VERSION, "gop-ridge-v1"} or artifact["feature_version"] != FEATURE_VERSION or artifact["preprocess_version"] != PREPROCESS_VERSION or artifact["phone_order"] != list(PHONES):
                raise ValueError("Incompatible calibrator feature/preprocessing version")
            if not artifact.get("extractor_id") or not artifact.get("threshold_locked_on_validation"):
                raise ValueError("Calibrator needs acoustic identity and locked validation threshold")
            self.mean, self.scale, self.coef = [np.asarray(artifact[n], float) for n in ("mean", "scale", "coef")]
            if any(a.shape != (FEATURE_DIM,) or not np.isfinite(a).all() for a in (self.mean, self.scale, self.coef)) or (self.scale <= 0).any():
                raise ValueError("Invalid Ridge coefficients/scaler")
            self.intercept = float(artifact["intercept"])
            self.threshold = float(artifact["practice_threshold_norm"])
            if not np.isfinite([self.intercept, self.threshold]).all() or not 0 <= self.threshold <= np.nextafter(1., np.inf):
                raise ValueError("Invalid intercept/threshold")
            self.extractor_id = artifact["extractor_id"]
            self.artifact = artifact
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ScoringError("model_not_ready", str(error)) from error

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ScoringError("alignment_failed", "Invalid acoustic features")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        if not np.isfinite(prediction).all():
            raise ScoringError("model_not_ready", "Calibrator produced nonfinite predictions")
        return np.clip(prediction, 0, 1)

    def save(self, path):
        dump_json(path, self.artifact)
