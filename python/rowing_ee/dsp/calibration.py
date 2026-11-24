"""Counts -> SI units, gyro bias removal, mount rotation (mirror of lib/core/Calibration)."""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig


class Calibration:
    """Gyro bias from a still window at power-on, then a fixed per-sample transform.

    Bias: the gyro mean over `still_n` consecutive frames. If any gyro axis has a
    standard deviation above `max_still_gyro_std` in that window, the window is
    thrown away and the next `still_n` frames are tried (the person moved).
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        n = cfg.n_imu
        self.acc_idx = np.array([6 * i + k for i in range(n) for k in range(3)])
        self.gyr_idx = np.array([6 * i + 3 + k for i in range(n) for k in range(3)])
        self.R = cfg.mount.astype(np.float64)        # (n_imu, 3, 3), sensor -> segment
        self.reset()

    def reset(self) -> None:
        self.sum = np.zeros(3 * self.cfg.n_imu)
        self.sumsq = np.zeros(3 * self.cfg.n_imu)
        self.count = 0
        self.bias = np.zeros(3 * self.cfg.n_imu)
        self.done = False
        self.windows_tried = 0

    # ------------------------------------------------------------- streaming
    def add_still(self, raw: np.ndarray) -> bool:
        """Accumulate one frame of the still window. Returns True when the bias is set."""
        g = raw[self.gyr_idx].astype(np.float64) * self.cfg.gyro_scale
        self.sum += g
        self.sumsq += g * g
        self.count += 1
        if self.count < self.cfg.still_n:
            return False
        self.windows_tried += 1
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        if np.max(std) <= self.cfg.max_still_gyro_std:
            self.bias = mean
            self.done = True
            return True
        self.sum[:] = 0.0
        self.sumsq[:] = 0.0
        self.count = 0
        return False

    # ----------------------------------------------------------------- batch
    def fit_bias(self, raw: np.ndarray) -> int:
        """Batch version of add_still over a whole log.

        Returns the index of the frame that completed calibration, or -1 if no
        window was still enough. Accumulates sequentially like the firmware.
        """
        self.reset()
        for i in range(raw.shape[0]):
            if self.add_still(raw[i]):
                return i
        return -1

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """(..., n_ch) int16 counts -> (..., n_ch) float64 segment-frame SI values."""
        x = np.asarray(raw, dtype=np.float64)
        out = np.empty_like(x)
        acc = x[..., self.acc_idx] * self.cfg.accel_scale
        gyr = x[..., self.gyr_idx] * self.cfg.gyro_scale - self.bias
        for i in range(self.cfg.n_imu):
            R = self.R[i]
            out[..., 6 * i:6 * i + 3] = acc[..., 3 * i:3 * i + 3] @ R.T
            out[..., 6 * i + 3:6 * i + 6] = gyr[..., 3 * i:3 * i + 3] @ R.T
        return out
