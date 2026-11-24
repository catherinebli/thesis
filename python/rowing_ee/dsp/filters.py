"""Causal Butterworth low-pass bank (mirror of lib/core/FilterBank).

Each channel runs the same cascade of second-order sections in direct form II
transposed, which is exactly what scipy.signal.sosfilt computes. The firmware
implements the same recurrence, so the two agree sample for sample:

    y      = b0*x + z1
    z1_new = b1*x - a1*y + z2
    z2_new = b2*x - a2*y
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import PipelineConfig


def design_sos(cfg: PipelineConfig) -> np.ndarray:
    return cfg.sos()


class FilterBank:
    def __init__(self, sos: np.ndarray, n_ch: int):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.n_ch = n_ch
        # state layout matches scipy's zi for filtering along axis 0: (n_sections, 2, n_ch)
        self.z = np.zeros((self.sos.shape[0], 2, n_ch))

    def reset(self, x0: np.ndarray) -> None:
        """Steady-state initial conditions for a constant input x0 (no start-up transient).

        For a constant input u, a section settles at y = g*u with
        g = (b0 + b1 + b2) / (1 + a1 + a2); solving the recurrence for the states gives
        z1 = (b1 + b2)*u - (a1 + a2)*y and z2 = b2*u - a2*y. Each section feeds the next.
        """
        u = np.asarray(x0, dtype=np.float64).copy()
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            y = u * (b0 + b1 + b2) / (1.0 + a1 + a2)
            self.z[s, 0] = (b1 + b2) * u - (a1 + a2) * y
            self.z[s, 1] = b2 * u - a2 * y
            u = y

    def step(self, x: np.ndarray) -> np.ndarray:
        """One sample for every channel (streaming path)."""
        u = np.asarray(x, dtype=np.float64)
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            y = b0 * u + self.z[s, 0]
            self.z[s, 0] = b1 * u - a1 * y + self.z[s, 1]
            self.z[s, 1] = b2 * u - a2 * y
            u = y
        return u

    def run(self, X: np.ndarray) -> np.ndarray:
        """A whole (N, n_ch) block at once (batch path); continues from the current state."""
        Y, self.z = signal.sosfilt(self.sos, np.asarray(X, dtype=np.float64), axis=0, zi=self.z)
        return Y
