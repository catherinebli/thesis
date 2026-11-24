"""Watts per stroke -> joules -> kcal (mirror of lib/core/EnergyMeter).

Time is counted in samples (n), so both sides integrate identically. Strokes
tile the time axis from one drive peak to the next, so no interval is counted
twice:

    E = sum_n P_n T_n + P_rest T_rest,   kcal = E / 4184
"""
from __future__ import annotations

REST, ROWING = 0, 1
J_PER_KCAL = 4184.0


class EnergyMeter:
    def __init__(self, fs: int, rest_timeout_n: int, p_rest: float):
        self.fs = float(fs)
        self.rest_timeout_n = rest_timeout_n
        self.p_rest = p_rest
        self.begin(0)

    def begin(self, n: int) -> None:
        self.E = 0.0
        self.n_acc = n            # time is accounted up to (not including) sample n_acc
        self.mode = REST
        self.p_last = self.p_rest
        self.n_last_peak = n

    def start_bout(self, n: int) -> None:
        """First catch after the session start or a pause: bill the gap at resting power."""
        self.E += self.p_rest * (n - self.n_acc) / self.fs
        self.n_acc = n
        self.mode = ROWING
        self.n_last_peak = n

    def add_stroke(self, n0: int, n1: int, p: float, valid: bool) -> float:
        """Bill [n_acc, n1) at the stroke's power. Invalid strokes hold the last estimate."""
        if not valid:
            p = self.p_last
        self.E += p * (n1 - self.n_acc) / self.fs
        self.n_acc = n1
        if valid:
            self.p_last = p
        self.n_last_peak = n1
        return p

    def tick(self, n: int) -> None:
        if self.mode == ROWING and (n - self.n_last_peak) > self.rest_timeout_n:
            self.mode = REST

    def live_joules(self, n: int) -> float:
        p = self.p_last if self.mode == ROWING else self.p_rest
        return self.E + p * (n - self.n_acc) / self.fs

    def live_kcal(self, n: int) -> float:
        return self.live_joules(n) / J_PER_KCAL

    def end(self, n: int) -> float:
        """Close the session at sample n (exclusive); returns total kcal."""
        p = self.p_last if self.mode == ROWING else self.p_rest
        self.E += p * (n - self.n_acc) / self.fs
        self.n_acc = n
        return self.E / J_PER_KCAL
