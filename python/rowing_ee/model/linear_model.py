"""Linear energy model P = w . x (mirror of lib/core/LinearModel).

Training is out of scope for this first pass; this class only evaluates and
stores weights. Until a trained model exists, `LinearModel.placeholder(D)`
returns all-zero weights, so every stroke is clamped to resting power.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LinearModel:
    def __init__(self, w: np.ndarray, *, placeholder: bool = False, feature_hash: int | None = None):
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.placeholder = placeholder
        self.feature_hash = feature_hash

    @property
    def D(self) -> int:
        return self.w.size

    @classmethod
    def placeholder_for(cls, D: int, feature_hash: int | None = None) -> "LinearModel":
        return cls(np.zeros(D), placeholder=True, feature_hash=feature_hash)

    @classmethod
    def load(cls, path: Path | str) -> "LinearModel":
        """CSV with one weight per line (intercept first), or .npz with arrays w and feature_hash."""
        p = Path(path)
        if p.suffix == ".npz":
            z = np.load(p)
            fh = int(z["feature_hash"]) if "feature_hash" in z else None
            return cls(z["w"], feature_hash=fh)
        return cls(np.loadtxt(p, delimiter=",", ndmin=1))

    def save_csv(self, path: Path | str) -> None:
        np.savetxt(path, self.w, delimiter=",", fmt="%.17g")

    def raw(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def predict(self, x: np.ndarray, p_rest: float, p_max: float) -> tuple[float, float, bool]:
        """Returns (P used, P raw, flag_high). P is capped at p_max, floored at p_rest."""
        p_raw = self.raw(x)
        flag_high = p_raw > p_max
        p = min(p_raw, p_max)
        p = max(p, p_rest)
        return p, p_raw, flag_high

    @staticmethod
    def fold_normalization(w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Bake z-scoring into the weights so the device can use raw features.

        y = b + sum_d w_d (x_d - mu_d) / sigma_d
          = (b - sum_d w_d mu_d / sigma_d) + sum_d (w_d / sigma_d) x_d

        w, mu, sigma cover the non-intercept features (x[1:]); returns the full
        weight vector with the folded intercept at index 0.
        """
        w = np.asarray(w, dtype=np.float64)
        w_raw = w / sigma
        b_raw = b - float(np.sum(w * mu / sigma))
        return np.concatenate(([b_raw], w_raw))
