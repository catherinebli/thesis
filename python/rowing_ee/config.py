"""Pipeline configuration.

One YAML file (config/pipeline.yaml) drives both the Python pipeline and, via
scripts/gen_headers.py, the Teensy firmware. Everything derived here (sample
counts, scale factors, filter coefficients, the feature hash) is computed once
and written into the firmware header, so both sides use identical numbers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"

AXES = ("ax", "ay", "az", "gx", "gy", "gz")
G0 = 9.80665                       # m/s^2 per g
DEG2RAD = np.pi / 180.0
ACCEL_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
GYRO_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
ACCEL_FS_SEL = {2: 0, 4: 1, 8: 2, 16: 3}
GYRO_FS_SEL = {250: 0, 500: 1, 1000: 2, 2000: 3}


@dataclass(frozen=True)
class ImuSpec:
    name: str
    bus: int
    mux: Optional[int]   # PCA9548 address, None = no mux
    channel: int
    addr: int


def _check_mount(name: str, m: np.ndarray) -> None:
    """A mount matrix must be a signed permutation with det = +1 (a proper rotation)."""
    if m.shape != (3, 3) or not np.all(np.isin(m, (-1, 0, 1))):
        raise ValueError(f"mount[{name}] must be 3x3 with entries in -1, 0, 1")
    if not (np.all(np.abs(m).sum(axis=0) == 1) and np.all(np.abs(m).sum(axis=1) == 1)):
        raise ValueError(f"mount[{name}] must have exactly one nonzero per row and column")
    if round(np.linalg.det(m)) != 1:
        raise ValueError(f"mount[{name}] has det -1: that is a mirror, not a rotation")


class PipelineConfig:
    """Validated view of pipeline.yaml plus every derived constant."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.fs = int(raw["sampling"]["fs_hz"])

        self.imus = [
            ImuSpec(
                name=str(d["name"]),
                bus=int(d["bus"]),
                mux=None if d.get("mux") is None else int(d["mux"]),
                channel=int(d.get("channel", 0)),
                addr=int(d["addr"]),
            )
            for d in raw["imus"]
        ]
        names = [i.name for i in self.imus]
        if len(set(names)) != len(names):
            raise ValueError("IMU names must be unique")
        for i in self.imus:
            if i.bus not in (0, 1, 2):
                raise ValueError(f"{i.name}: bus must be 0, 1 or 2")
            if i.mux is not None and not (0x70 <= i.mux <= 0x77 and 0 <= i.channel <= 7):
                raise ValueError(f"{i.name}: mux must be 0x70-0x77 with channel 0-7")
        self.n_imu = len(self.imus)
        self.n_ch = 6 * self.n_imu
        self.channel_names = [f"{i.name}_{a}" for i in self.imus for a in AXES]

        icm = raw["icm20948"]
        self.i2c_hz = int(icm["i2c_hz"])
        self.accel_fs_g = int(icm["accel_fs_g"])
        self.gyro_fs_dps = int(icm["gyro_fs_dps"])
        self.dlpf_cfg = int(icm["dlpf_cfg"])
        self.odr_div = int(icm["odr_div"])
        if self.accel_fs_g not in ACCEL_LSB_PER_G or self.gyro_fs_dps not in GYRO_LSB_PER_DPS:
            raise ValueError("unsupported ICM-20948 full-scale setting")
        if not 0 <= self.dlpf_cfg <= 7:
            raise ValueError("dlpf_cfg must be 0-7")
        self.accel_scale = G0 / ACCEL_LSB_PER_G[self.accel_fs_g]          # m/s^2 per count
        self.gyro_scale = DEG2RAD / GYRO_LSB_PER_DPS[self.gyro_fs_dps]    # rad/s per count

        mount = raw["mount"]
        self.mount = np.zeros((self.n_imu, 3, 3), dtype=np.int64)
        for k, i in enumerate(self.imus):
            m = np.asarray(mount[i.name], dtype=np.int64)
            _check_mount(i.name, m)
            self.mount[k] = m

        cal = raw["calibration"]
        self.still_n = int(round(float(cal["still_s"]) * self.fs))
        self.max_still_gyro_std = float(cal["max_still_gyro_std"])

        self.filter_order = int(raw["filter"]["order"])
        self.cutoff_hz = float(raw["filter"]["cutoff_hz"])
        if not 0 < self.cutoff_hz < self.fs / 2:
            raise ValueError("filter cutoff must be below Nyquist")

        det = raw["detector"]
        if det["imu"] not in names:
            raise ValueError(f"detector.imu '{det['imu']}' is not in imus")
        if det["axis"] not in AXES:
            raise ValueError(f"detector.axis must be one of {AXES}")
        self.detect_channel = 6 * names.index(det["imu"]) + AXES.index(det["axis"])
        self.detect_sign = 1 if int(det["sign"]) >= 0 else -1
        self.thr_min = float(det["thr_min"])
        self.thr_frac = float(det["thr_frac"])
        self.hyst = float(det["hyst"])
        self.min_stroke_n = int(round(float(det["min_stroke_s"]) * self.fs))
        self.max_stroke_n = int(round(float(det["max_stroke_s"]) * self.fs))

        self.bins = int(raw["features"]["bins"])
        self.max_err_frac = float(raw["features"]["max_err_frac"])
        if self.min_stroke_n < self.bins:
            raise ValueError("min_stroke_s * fs must be >= bins so every bin has a sample")
        self.D = 4 + self.n_ch * self.bins

        en = raw["energy"]
        self.k_rest = float(en["k_rest"])
        self.rest_timeout_n = int(round(float(en["rest_timeout_s"]) * self.fs))
        self.p_max_w = float(en["p_max_w"])
        if self.rest_timeout_n <= self.max_stroke_n:
            raise ValueError("rest_timeout_s must be longer than max_stroke_s")

        self.ring_capacity = int(round(float(raw["buffer"]["seconds"]) * self.fs))
        if self.ring_capacity <= self.max_stroke_n:
            raise ValueError("buffer.seconds must exceed max_stroke_s")

        self.subject_defaults = dict(raw["subject_defaults"])

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "PipelineConfig":
        p = Path(path) if path is not None else DEFAULT_CONFIG
        with open(p, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    # ---------------------------------------------------------------- derived
    def sos(self) -> np.ndarray:
        """Second-order sections, rows [b0, b1, b2, 1, a1, a2] (scipy convention)."""
        return signal.butter(self.filter_order, self.cutoff_hz, fs=self.fs, output="sos")

    def feature_hash(self) -> int:
        """32-bit hash of every setting that changes the feature vector.

        Bus, mux and address wiring, energy settings and subject defaults are
        excluded: they do not change what the model sees.
        """
        subset = {
            "fs_hz": self.fs,
            "imus": [i.name for i in self.imus],
            "icm": {"dlpf_cfg": self.dlpf_cfg, "odr_div": self.odr_div},
            "mount": {i.name: self.mount[k].tolist() for k, i in enumerate(self.imus)},
            "calibration": self.raw["calibration"],
            "filter": self.raw["filter"],
            "detector": self.raw["detector"],
            "bins": self.bins,
        }
        blob = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode()
        return int(hashlib.sha256(blob).hexdigest()[:8], 16)

    def feature_names(self) -> list[str]:
        names = ["bias", "mass_kg", "height_m", "stroke_s"]
        names += [f"{ch}_b{k:02d}" for ch in self.channel_names for k in range(self.bins)]
        return names
