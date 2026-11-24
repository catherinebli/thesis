"""One stroke -> feature vector (mirror of lib/core/Featurizer).

    x = [1, mass_kg, height_m, T_stroke_s, ch0 bins 0..B-1, ch1 bins 0..B-1, ...]

Bin k of a stroke with L samples averages samples [floor(kL/B), floor((k+1)L/B)).
The firmware stores filtered frames as float32, so the segment passed in here is
float32 too; means are accumulated in float64 on both sides.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..subject import Subject


class Featurizer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.B = cfg.bins
        self.D = cfg.D

    def bin_edges(self, L: int) -> np.ndarray:
        return (np.arange(self.B + 1) * L) // self.B

    def extract(self, seg: np.ndarray, T_s: float, subject: Subject) -> np.ndarray:
        """seg: (L, n_ch) float32 filtered samples of one stroke."""
        L = seg.shape[0]
        if L < self.B:
            raise ValueError(f"stroke has {L} samples, fewer than {self.B} bins")
        edges = self.bin_edges(L)
        seg64 = seg.astype(np.float64)
        bins = np.empty((self.cfg.n_ch, self.B))
        for k in range(self.B):
            a, b = edges[k], edges[k + 1]
            bins[:, k] = seg64[a:b].sum(axis=0) / (b - a)
        x = np.empty(self.D)
        x[:4] = (1.0, subject.mass_kg, subject.height_m, T_s)
        x[4:] = bins.reshape(-1)          # channel-major: index 4 + c*B + k
        return x
