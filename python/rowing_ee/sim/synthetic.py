"""Synthetic rowing logs in the firmware's raw format, for testing without hardware.

This is a plumbing test, not physiology: each IMU gets a periodic, zero-mean
stroke pattern (a fast drive pulse and a slower recovery), gravity in a fixed
segment orientation, sensor noise and a gyro bias. Bouts, rests, I2C read
failures and missed ticks can be injected to exercise every code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import G0, PipelineConfig

# per-IMU (gz amplitude rad/s, phase offset, gravity direction in segment frame)
_PROFILE = {
    "forearm":   (2.5, 0.25, (0.3, -0.2, 0.93)),
    "upper_arm": (1.8, 0.20, (0.9, 0.1, 0.42)),
    "chest":     (1.2, 0.10, (0.95, 0.3, 0.0)),
    "thigh":     (2.0, 0.00, (0.4, 0.9, 0.17)),
    "shank":     (3.0, 0.00, (0.9, 0.4, 0.17)),
}
DRIVE = 0.35   # fraction of the stroke spent on the drive


def stroke_shape(phi: np.ndarray) -> np.ndarray:
    """Zero-mean stroke pulse: sin^2 drive bump, then a smaller, longer recovery dip."""
    phi = np.mod(phi, 1.0)
    rec_amp = DRIVE / (1.0 - DRIVE)       # makes the cycle mean exactly zero
    return np.where(phi < DRIVE,
                    np.sin(np.pi * phi / DRIVE) ** 2,
                    -rec_amp * np.sin(np.pi * (phi - DRIVE) / (1.0 - DRIVE)) ** 2)


@dataclass
class SyntheticSession:
    n: np.ndarray
    t_us: np.ndarray
    raw: np.ndarray
    err: np.ndarray
    expected_strokes: int     # complete strokes the detector should report


def make_session(cfg: PipelineConfig, *, still_s: float = 5.0,
                 bouts: tuple = ((90.0, 20.0, 1.0),), rest_s: float = 20.0,
                 tail_s: float = 5.0, seed: int = 0, err_rate: float = 0.0,
                 gap_prob: float = 0.0, bias_dps: float = 0.5,
                 t0_us: int = 4_290_000_000) -> SyntheticSession:
    """bouts: sequence of (duration_s, strokes_per_min, intensity)."""
    rng = np.random.default_rng(seed)
    fs = cfg.fs

    # timeline: per-sample phase and intensity (intensity 0 = sitting still)
    phase_parts, inten_parts = [np.zeros(int(still_s * fs))], [np.zeros(int(still_s * fs))]
    expected = 0
    for k, (dur, spm, inten) in enumerate(bouts):
        m = int(dur * fs)
        t = np.arange(m) / fs
        phase_parts.append(rng.uniform(0.4, 0.9) + t * spm / 60.0)
        inten_parts.append(np.full(m, float(inten)))
        # strokes between consecutive drive peaks that fall inside the bout
        peaks = np.floor(phase_parts[-1][-1] - DRIVE / 2) - np.ceil(phase_parts[-1][0] - DRIVE / 2) + 1
        expected += int(peaks) - 1
        gap = rest_s if k < len(bouts) - 1 else tail_s
        phase_parts.append(np.zeros(int(gap * fs)))
        inten_parts.append(np.zeros(int(gap * fs)))
    phase = np.concatenate(phase_parts)
    inten = np.concatenate(inten_parts)
    N = phase.size

    seg = np.zeros((N, cfg.n_ch))
    for i, imu in enumerate(cfg.imus):
        amp, off, gdir = _PROFILE.get(imu.name, (1.5, 0.15 * i, (0.6, 0.6, 0.53)))
        g = np.asarray(gdir) / np.linalg.norm(gdir) * G0
        sh = stroke_shape(phase + off) * inten
        sh2 = stroke_shape(phase + off + 0.3) * inten
        b = 6 * i
        seg[:, b + 0] = g[0] + 2.0 * sh2
        seg[:, b + 1] = g[1] - 1.5 * sh
        seg[:, b + 2] = g[2] + 0.5 * sh2
        seg[:, b + 3] = 0.15 * amp * sh2
        seg[:, b + 4] = 0.10 * amp * sh
        seg[:, b + 5] = amp * sh
    seg[:, 0::6] += rng.normal(0, 0.03, (N, cfg.n_imu))
    seg[:, 1::6] += rng.normal(0, 0.03, (N, cfg.n_imu))
    seg[:, 2::6] += rng.normal(0, 0.03, (N, cfg.n_imu))
    for k in (3, 4, 5):
        seg[:, k::6] += rng.normal(0, 0.01, (N, cfg.n_imu))

    # segment frame -> sensor frame (R maps sensor -> segment, so sensor = R^T seg)
    sens = np.empty_like(seg)
    for i in range(cfg.n_imu):
        R = cfg.mount[i].astype(np.float64)
        sens[:, 6 * i:6 * i + 3] = seg[:, 6 * i:6 * i + 3] @ R
        sens[:, 6 * i + 3:6 * i + 6] = seg[:, 6 * i + 3:6 * i + 6] @ R
    bias = rng.uniform(-bias_dps, bias_dps, (cfg.n_imu, 3)) * np.pi / 180.0
    for i in range(cfg.n_imu):
        sens[:, 6 * i + 3:6 * i + 6] += bias[i]

    raw = np.empty((N, cfg.n_ch), dtype=np.int16)
    for i in range(cfg.n_imu):
        a = np.round(sens[:, 6 * i:6 * i + 3] / cfg.accel_scale)
        w = np.round(sens[:, 6 * i + 3:6 * i + 6] / cfg.gyro_scale)
        raw[:, 6 * i:6 * i + 3] = np.clip(a, -32768, 32767)
        raw[:, 6 * i + 3:6 * i + 6] = np.clip(w, -32768, 32767)

    # inject failures exactly the way the firmware records them (hold last value, set bit)
    err = np.zeros(N, dtype=np.uint8)
    still_n = int(still_s * fs)
    for k in range(still_n + 1, N):
        if gap_prob and rng.random() < gap_prob:
            raw[k] = raw[k - 1]
            err[k] = 0x80
            continue
        if err_rate:
            for i in range(cfg.n_imu):
                if rng.random() < err_rate:
                    raw[k, 6 * i:6 * i + 6] = raw[k - 1, 6 * i:6 * i + 6]
                    err[k] |= 1 << i
    n = np.arange(N, dtype=np.int64)
    t_us = t0_us + n * (1_000_000 // fs) + rng.integers(-30, 31, N)
    return SyntheticSession(n=n, t_us=t_us, raw=raw, err=err, expected_strokes=expected)
