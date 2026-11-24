"""Online drive-peak picker that segments strokes (mirror of lib/core/StrokeDetector).

The input is the sign-corrected, filtered shank sagittal gyro. One big positive
pulse per stroke comes from the leg drive. A candidate peak is tracked while the
signal is above the threshold; it is confirmed once the signal falls below
`hyst * thr`. Consecutive confirmed peaks bound one stroke.

Keep this sample-by-sample (do not swap in scipy.signal.find_peaks): the
firmware runs this exact state machine, and parity depends on it.
"""
from __future__ import annotations

import math

from ..config import PipelineConfig

NONE, FIRST, STROKE = 0, 1, 2


def median(vals: list[float]) -> float:
    """Median with the even-count rule the firmware uses (mean of the middle two)."""
    v = sorted(vals)
    m = len(v)
    return v[m // 2] if m % 2 else 0.5 * (v[m // 2 - 1] + v[m // 2])


class StrokeDetector:
    RECENT = 5

    def __init__(self, cfg: PipelineConfig):
        self.thr_min = cfg.thr_min
        self.thr_frac = cfg.thr_frac
        self.hyst = cfg.hyst
        self.min_n = cfg.min_stroke_n
        self.reset()

    def reset(self) -> None:
        self.cand_val = -math.inf
        self.cand_n = -1
        self.last_peak_n = -1
        self.recent: list[float] = []

    def threshold(self) -> float:
        if not self.recent:
            return self.thr_min
        return max(self.thr_min, self.thr_frac * median(self.recent))

    def step(self, s: float, n: int) -> tuple[int, int, int]:
        """Feed one sample. Returns (event, n0, n1).

        FIRST:  first confirmed peak of the session, at n0 (start of the first stroke)
        STROKE: a stroke spanning samples [n0, n1)
        NONE:   nothing yet (n0 = n1 = -1)
        """
        thr = self.threshold()
        if s > thr and s > self.cand_val:
            self.cand_val, self.cand_n = s, n
            return NONE, -1, -1
        if self.cand_n >= 0 and s < self.hyst * thr:
            event, n0, n1 = NONE, -1, -1
            accept = True
            if self.last_peak_n < 0:
                event, n0, n1 = FIRST, self.cand_n, self.cand_n
            elif self.cand_n - self.last_peak_n >= self.min_n:
                event, n0, n1 = STROKE, self.last_peak_n, self.cand_n
            else:
                accept = False            # second bump inside one stroke: ignore it
            if accept:
                self.last_peak_n = self.cand_n
                self.recent.append(self.cand_val)
                if len(self.recent) > self.RECENT:
                    self.recent.pop(0)
            self.cand_val, self.cand_n = -math.inf, -1
            return event, n0, n1
        return NONE, -1, -1
