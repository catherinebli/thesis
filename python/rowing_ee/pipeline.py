"""Raw IMU frames -> per-stroke metabolic power -> kcal (mirror of lib/core/Pipeline).

Two entry points that give identical results:

* Pipeline.on_sample(): one frame at a time, the same call sequence as the
  firmware's Pipeline::onSample(). Use it to reason about the device.
* run_session(): a whole raw log at once. Calibration, unit conversion and
  filtering are vectorized; the detector and energy accounting still run sample
  by sample, so strokes and kcal match the streaming path exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import PipelineConfig
from .dsp.calibration import Calibration
from .dsp.detector import FIRST, STROKE, StrokeDetector
from .dsp.features import Featurizer
from .dsp.filters import FilterBank
from .dsp.ring import FrameRing
from .energy import REST, EnergyMeter
from .io.raw_log import RawLog
from .model.linear_model import LinearModel
from .subject import Subject

CSV_HEADER = "n0,n1,T_s,P_w,P_raw_w,valid,flag_high,kcal_total"


@dataclass
class StrokeRecord:
    n0: int
    n1: int
    T_s: float
    P_w: float          # power billed for this stroke (held value if invalid)
    P_raw_w: float      # model output before floor/cap (nan if not scored)
    valid: bool
    flag_high: bool
    kcal_total: float   # cumulative kcal at the end of this stroke

    def csv_row(self) -> str:
        return (f"{self.n0},{self.n1},{self.T_s:.17g},{self.P_w:.17g},{self.P_raw_w:.17g},"
                f"{int(self.valid)},{int(self.flag_high)},{self.kcal_total:.17g}")


@dataclass
class SessionResult:
    strokes: list[StrokeRecord]
    kcal_total: float
    n_first: int
    n_end: int                       # exclusive
    fs: int
    calibrated_n: int                # -1 if the still window never passed
    p_rest_w: float
    placeholder_model: bool
    detect_signal: Optional[np.ndarray] = None   # filtered, sign-corrected detector channel
    detect_n: Optional[np.ndarray] = None
    peaks_n: list[int] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.n_end - self.n_first) / self.fs

    def summary(self) -> dict:
        valid = [s for s in self.strokes if s.valid]
        return {
            "duration_s": round(self.duration_s, 2),
            "kcal_total": round(self.kcal_total, 3),
            "strokes": len(self.strokes),
            "valid_strokes": len(valid),
            "mean_power_w": round(float(np.mean([s.P_w for s in valid])), 1) if valid else None,
            "spm": round(60.0 / float(np.mean([s.T_s for s in valid])), 1) if valid else None,
            "rest_power_w": round(self.p_rest_w, 1),
            "calibrated_at_s": round((self.calibrated_n - self.n_first) / self.fs, 2)
            if self.calibrated_n >= 0 else None,
            "placeholder_model": self.placeholder_model,
        }

    def to_csv(self, path: Path | str) -> None:
        lines = [CSV_HEADER] + [s.csv_row() for s in self.strokes]
        lines.append(f"# end,{self.n_end},{self.kcal_total:.17g}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_est_csv(path: Path | str) -> tuple[list[StrokeRecord], Optional[float]]:
    """Read est_###.csv written by the firmware (or by the native replay tool)."""
    strokes, kcal_total = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("n0"):
            continue
        if line.startswith("# end"):
            kcal_total = float(line.split(",")[2])
            continue
        if line.startswith("#"):
            continue
        f = line.split(",")
        strokes.append(StrokeRecord(int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4]),
                                    bool(int(f[5])), bool(int(f[6])), float(f[7])))
    return strokes, kcal_total


class Pipeline:
    def __init__(self, cfg: PipelineConfig, model: LinearModel, subject: Subject):
        if model.D != cfg.D:
            raise ValueError(f"model has {model.D} weights, config expects {cfg.D}")
        self.cfg = cfg
        self.model = model
        self.subject = subject
        self.p_rest = subject.rest_watts(cfg.k_rest)
        self.cal = Calibration(cfg)
        self.fb = FilterBank(cfg.sos(), cfg.n_ch)
        self.det = StrokeDetector(cfg)
        self.feat = Featurizer(cfg)
        self.energy = EnergyMeter(cfg.fs, cfg.rest_timeout_n, self.p_rest)
        self.ring = FrameRing(cfg.ring_capacity, cfg.n_ch)
        self.started = False
        self.filter_init = False
        self.calibrated_n = -1

    # ------------------------------------------------------------- streaming
    def on_sample(self, n: int, raw: np.ndarray, err: int) -> Optional[StrokeRecord]:
        if not self.started:
            self.energy.begin(n)
            self.started = True
        if not self.cal.done:
            if self.cal.add_still(raw):
                self.calibrated_n = n
            return None
        x = self.cal.apply(raw)
        if not self.filter_init:
            self.fb.reset(x)
            self.filter_init = True
        y = self.fb.step(x)
        self.ring.push(n, y, err)
        ev, n0, n1 = self.det.step(self.cfg.detect_sign * float(y[self.cfg.detect_channel]), n)
        rec = None
        if ev == FIRST:
            self.energy.start_bout(n0)
        elif ev == STROKE:
            rec = self._on_stroke(n0, n1, self.ring.contains, self.ring.slice)
        self.energy.tick(n)
        return rec

    def end(self, n_end: int) -> float:
        return self.energy.end(n_end)

    # ------------------------------------------------------------ shared core
    def _on_stroke(self, n0: int, n1: int, contains: Callable[[int, int], bool],
                   get: Callable[[int, int], tuple[np.ndarray, np.ndarray]]) -> Optional[StrokeRecord]:
        L = n1 - n0
        if self.energy.mode == REST or L > self.cfg.max_stroke_n:
            self.energy.start_bout(n1)          # first stroke after a pause: no estimate
            return None
        valid = contains(n0, n1)
        seg = None
        if valid:
            seg, errs = get(n0, n1)
            valid = np.count_nonzero(errs) / L <= self.cfg.max_err_frac
        p_raw, p, flag = math.nan, math.nan, False
        T = L / self.cfg.fs
        if valid:
            x = self.feat.extract(seg, T, self.subject)
            p, p_raw, flag = self.model.predict(x, self.p_rest, self.cfg.p_max_w)
        used = self.energy.add_stroke(n0, n1, p, valid)
        return StrokeRecord(n0, n1, T, used, p_raw, valid, flag, self.energy.live_kcal(n1))

    # ------------------------------------------------------------------ batch
    @classmethod
    def run_session(cls, log: RawLog, cfg: PipelineConfig, model: LinearModel,
                    subject: Subject, keep_signal: bool = True) -> SessionResult:
        if log.raw.shape[1] != cfg.n_ch:
            raise ValueError(f"log has {log.raw.shape[1]} channels, config expects {cfg.n_ch}")
        p = cls(cfg, model, subject)
        n_all = log.n
        n_first, n_end = int(n_all[0]), int(n_all[-1]) + 1
        p.energy.begin(n_first)
        p.started = True

        ic = p.cal.fit_bias(log.raw)
        strokes: list[StrokeRecord] = []
        peaks: list[int] = []
        S = nn = None
        if ic >= 0:
            p.calibrated_n = int(n_all[ic])
            start = ic + 1
            if start < len(n_all):
                X = p.cal.apply(log.raw[start:])
                p.fb.reset(X[0])
                p.filter_init = True
                Y = p.fb.run(X)
                Y32 = Y.astype(np.float32)
                ee = log.err[start:]
                nn = n_all[start:]
                S = cfg.detect_sign * Y[:, cfg.detect_channel]
                base = int(nn[0])
                cap = cfg.ring_capacity
                for i in range(len(nn)):
                    ni = base + i
                    ev, n0, n1 = p.det.step(float(S[i]), ni)
                    if ev == FIRST:
                        p.energy.start_bout(n0)
                        peaks.append(n0)
                    elif ev == STROKE:
                        peaks.append(n1)
                        oldest = max(base, ni - cap + 1)
                        rec = p._on_stroke(
                            n0, n1,
                            lambda a, b, o=oldest, now=ni: a >= o and b <= now + 1 and a <= b,
                            lambda a, b: (Y32[a - base:b - base], ee[a - base:b - base]),
                        )
                        if rec is not None:
                            strokes.append(rec)
                    p.energy.tick(ni)
        kcal = p.energy.end(n_end)
        return SessionResult(
            strokes=strokes, kcal_total=kcal, n_first=n_first, n_end=n_end, fs=cfg.fs,
            calibrated_n=p.calibrated_n, p_rest_w=p.p_rest, placeholder_model=model.placeholder,
            detect_signal=S if keep_signal else None, detect_n=nn if keep_signal else None,
            peaks_n=peaks,
        )


def run_streaming(log: RawLog, cfg: PipelineConfig, model: LinearModel, subject: Subject) -> SessionResult:
    """Frame-by-frame replay through Pipeline.on_sample (slow; used to test run_session)."""
    p = Pipeline(cfg, model, subject)
    strokes = []
    for i in range(len(log)):
        rec = p.on_sample(int(log.n[i]), log.raw[i], int(log.err[i]))
        if rec is not None:
            strokes.append(rec)
    n_end = int(log.n[-1]) + 1
    kcal = p.end(n_end)
    return SessionResult(strokes=strokes, kcal_total=kcal, n_first=int(log.n[0]), n_end=n_end,
                         fs=cfg.fs, calibrated_n=p.calibrated_n, p_rest_w=p.p_rest,
                         placeholder_model=model.placeholder)
