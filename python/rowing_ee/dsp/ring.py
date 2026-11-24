"""Fixed-size buffer of filtered frames (mirror of lib/core/FrameRing).

Frames are stored as float32, like the firmware (which keeps 10 s x 30 channels
in RAM2). Index by absolute sample number n.
"""
from __future__ import annotations

import numpy as np


class FrameRing:
    def __init__(self, capacity: int, n_ch: int):
        self.cap = capacity
        self.data = np.zeros((capacity, n_ch), dtype=np.float32)
        self.err = np.zeros(capacity, dtype=np.uint8)
        self.count = 0
        self.newest = -1

    def push(self, n: int, y: np.ndarray, err: int) -> None:
        k = n % self.cap
        self.data[k] = y            # float64 -> float32, same rounding as the firmware's cast
        self.err[k] = err
        self.newest = n
        self.count = min(self.count + 1, self.cap)

    def oldest(self) -> int:
        return self.newest - self.count + 1

    def contains(self, n0: int, n1: int) -> bool:
        """True if every sample in [n0, n1) is still in the buffer."""
        return self.count > 0 and n0 <= n1 and n0 >= self.oldest() and n1 <= self.newest + 1

    def slice(self, n0: int, n1: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n0, n1) % self.cap
        return self.data[idx], self.err[idx]
