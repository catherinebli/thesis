"""Reader and writer for raw_###.bin, the file the Teensy logs to the SD card.

Layout (little-endian, packed), documented in docs/06_firmware_and_data_formats.md:

    header, 32 bytes
        char[4]  magic        "REE1"
        uint16   version      1
        uint16   header_bytes 32
        uint16   record_bytes 69
        uint8    n_ch         30
        uint8    n_imu        5
        uint16   fs_hz        100
        uint16   reserved
        uint32   feature_hash
        uint32   session
        uint32   boot_ms
        uint8[4] pad
    records, 69 bytes each
        uint32   n            sample index, +1 every 10 ms tick
        uint32   t_us         micros() at read time (wraps every 71.6 min)
        int16[30] raw         ax ay az gx gy gz counts per IMU, sensor frame
        uint8    err          bit i = IMU i read failed (value held), bit 7 = missed tick (frame held)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"REE1"
VERSION = 1
HEADER_BYTES = 32
ERR_GAP = 0x80
HEADER_FMT = "<4sHHHBBHHIII4s"   # 32 bytes
assert struct.calcsize(HEADER_FMT) == HEADER_BYTES


def record_dtype(n_ch: int) -> np.dtype:
    return np.dtype([("n", "<u4"), ("t_us", "<u4"), ("raw", "<i2", (n_ch,)), ("err", "u1")])


def unwrap_u32(t: np.ndarray) -> np.ndarray:
    """Undo uint32 wraparound (micros() wraps every 2^32 us = 71.6 min)."""
    t = t.astype(np.int64)
    wraps = np.concatenate(([0], np.cumsum(np.diff(t) < 0)))
    return t + (wraps << 32)


@dataclass
class RawLog:
    header: dict
    n: np.ndarray          # (N,) int64 sample index, contiguous
    t_us: np.ndarray       # (N,) int64, unwrapped
    raw: np.ndarray        # (N, n_ch) int16 counts
    err: np.ndarray        # (N,) uint8
    dropped_tail: int = 0  # records discarded after the first sequence break
    path: str = field(default="")

    def __len__(self) -> int:
        return len(self.n)

    @classmethod
    def read(cls, path: Path | str) -> "RawLog":
        data = Path(path).read_bytes()
        if len(data) < HEADER_BYTES:
            raise ValueError(f"{path}: too short for a header")
        (magic, version, header_bytes, record_bytes, n_ch, n_imu, fs_hz, _res,
         feature_hash, session, boot_ms, _pad) = struct.unpack_from(HEADER_FMT, data, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, not a raw_###.bin log")
        if version != VERSION or header_bytes != HEADER_BYTES:
            raise ValueError(f"{path}: unsupported log version {version}")
        dt = record_dtype(n_ch)
        if record_bytes != dt.itemsize:
            raise ValueError(f"{path}: record size {record_bytes} != expected {dt.itemsize}")
        body = data[HEADER_BYTES:]
        n_rec = len(body) // dt.itemsize
        rec = np.frombuffer(body[: n_rec * dt.itemsize], dtype=dt)

        # The firmware preallocates the file, so a log that was not closed cleanly
        # ends in zeros or stale bytes. Keep records up to the first break in n.
        keep = n_rec
        if n_rec > 1:
            breaks = np.nonzero(np.diff(rec["n"].astype(np.int64)) != 1)[0]
            if breaks.size:
                keep = int(breaks[0]) + 1
        rec = rec[:keep]
        header = dict(version=version, n_ch=n_ch, n_imu=n_imu, fs_hz=fs_hz,
                      feature_hash=feature_hash, session=session, boot_ms=boot_ms)
        return cls(header=header, n=rec["n"].astype(np.int64), t_us=unwrap_u32(rec["t_us"]),
                   raw=np.array(rec["raw"], dtype=np.int16), err=np.array(rec["err"], dtype=np.uint8),
                   dropped_tail=n_rec - keep, path=str(path))

    def timing_report(self) -> dict:
        """Sample timing and error summary: run this on every new log before trusting it."""
        rep = {"samples": len(self), "duration_s": len(self) / self.header["fs_hz"],
               "gap_filled": int(np.count_nonzero(self.err & ERR_GAP)),
               "dropped_tail_records": self.dropped_tail}
        if len(self) > 1:
            dt = np.diff(self.t_us) / 1000.0
            rep.update(dt_ms_mean=float(dt.mean()), dt_ms_std=float(dt.std()),
                       dt_ms_max=float(dt.max()), dt_ms_min=float(dt.min()))
        rep["err_per_imu"] = [int(np.count_nonzero(self.err & (1 << i)))
                              for i in range(self.header["n_imu"])]
        return rep


def write_raw_log(path: Path | str, n: np.ndarray, t_us: np.ndarray, raw: np.ndarray,
                  err: np.ndarray, *, fs_hz: int, n_imu: int, feature_hash: int,
                  session: int = 0, boot_ms: int = 0) -> None:
    """Write a log in the firmware's format (used by the synthetic generator and tests)."""
    n_ch = raw.shape[1]
    dt = record_dtype(n_ch)
    rec = np.zeros(len(n), dtype=dt)
    rec["n"] = np.asarray(n, dtype=np.int64) & 0xFFFFFFFF
    rec["t_us"] = np.asarray(t_us, dtype=np.int64) & 0xFFFFFFFF
    rec["raw"] = raw
    rec["err"] = err
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, HEADER_BYTES, dt.itemsize, n_ch, n_imu,
                         fs_hz, 0, feature_hash & 0xFFFFFFFF, session, boot_ms, b"\0" * 4)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())
