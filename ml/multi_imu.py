"""Five-IMU suit support for the offline pipeline.

The suit firmware (firmware/icm20948_mux5) and recorder (tools/record_piece.py)
write one CSV per piece: one row per 100 Hz frame, all five IMUs side by side

    sample, t_s, forearm_ax_mps2 ... forearm_gz_rads, upperarm_..., chest_...,
    thigh_..., shank_..., read_us

A failed read is logged as NaN for that IMU only. Because every IMU shares the
row's timestamp, the IMUs never need syncing to each other.

imu_preprocessing / stroke_detection / erg_stroke_sync work on one IMU's six
columns (ax_mps2 .. gz_rads). This module bridges the two:

1. load_suit      split the suit file into one canonical frame per IMU
2. lead_in_bias   gyro offset of each IMU from the still lead-in
3. clean_suit     clean each IMU with clean_imu, then put all of them back on
                  one 100 Hz grid with prefixed columns ("wide" frame)
4. ref_view       single-IMU view of one segment (thigh by default) with the same
                  row index as the wide frame: strokes detected and synced on it
                  cut all five IMUs at the same samples
5. process_piece  IMU file + PM5 stroke file -> erg-labelled 30-bin features for
                  every valid stroke (5 IMUs x 6 channels x 30 bins = 900, plus
                  weight and height)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import erg_stroke_sync as es
import imu_preprocessing as ip
import stroke_detection as sd

SEGMENTS = ("forearm", "upperarm", "chest", "thigh", "shank")
REF_SEGMENT = "thigh"       # big, clean sagittal rotation every stroke
FS = 100.0

# icm20948_mux5.ino settings (for the saturation check)
ACC_RANGE_G = 8.0
GYR_RANGE_RADS = float(np.deg2rad(1000.0))


def suit_columns(segments) -> list[str]:
    """Prefixed signal columns in feature order: segment by segment, 6 channels each."""
    return [f"{s}_{c}" for s in segments for c in ip.SIGNALS]


def find_segments(df: pd.DataFrame) -> list[str]:
    """Segments that have all six columns in a suit file, in SEGMENTS order first."""
    names = []
    for c in df.columns:
        m = re.match(r"^(.+)_ax_mps2$", str(c))
        if m and all(f"{m.group(1)}_{s}" in df.columns for s in ip.SIGNALS):
            names.append(m.group(1))
    known = [n for n in SEGMENTS if n in names]
    return known + [n for n in names if n not in known]


def read_suit_csv(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source, comment="#")


def load_suit(source, segments=None) -> dict[str, pd.DataFrame]:
    """Suit CSV -> {segment: canonical 6-DoF frame (t_s, sample, ax_mps2 .. gz_rads)}."""
    df = read_suit_csv(source)
    segs = list(segments) if segments else find_segments(df)
    if not segs:
        raise ValueError(f"no <segment>_ax_mps2 .. _gz_rads column groups in {list(df.columns)[:8]} ...")
    out = {}
    for s in segs:
        cols = {c: f"{s}_{c}" for c in ip.SIGNALS}
        missing = [v for v in cols.values() if v not in df]
        if missing:
            raise ValueError(f"segment {s!r} is missing columns {missing}")
        out[s] = ip.load_imu(df, columns=cols, acc_unit="mps2", gyr_unit="rads")
    return out


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, stop) of the longest True run, or (0, 0)."""
    if not mask.any():
        return 0, 0
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


def lead_in_bias(raw: pd.DataFrame, fs: float = FS, win_s: float = 1.0,
                 search_s: float = 30.0, gyr_std_tol: float = 0.06,
                 acc_std_tol: float = 0.4) -> dict | None:
    """Gyro offset {gx_rads: b, ...} from the still lead-in, or None if there isn't one.

    Searches the first search_s seconds for the longest stretch where every gyro
    axis and |accel| barely change (rolling std over win_s below the
    tolerances), and averages the gyro there. It tests the spread of the gyro,
    not its size, so it still works when the offset is several deg/s
    (imu_preprocessing.find_still needs |w| < 0.1 rad/s including the offset).
    """
    t = raw["t_s"].to_numpy()
    sel = raw.loc[t <= t[0] + search_s].reset_index(drop=True)
    n = max(3, int(round(win_s * fs)))
    if len(sel) < 2 * n:
        return None
    gstd = sel[ip.GYR].rolling(n, center=True, min_periods=n).std().max(axis=1, skipna=False)
    a = pd.Series(np.linalg.norm(sel[ip.ACC].to_numpy(), axis=1))
    astd = a.rolling(n, center=True, min_periods=n).std()
    quiet = ((gstd < gyr_std_tol) & (astd < acc_std_tol)).to_numpy()
    s, e = _longest_run(quiet)
    if e - s < n // 2:                       # need about 1.5 s of stillness
        return None
    lo, hi = max(0, s - n // 2), min(len(sel), e + n // 2)
    return sel.loc[lo:hi - 1, ip.GYR].mean().to_dict()


def clean_suit(raw: dict, fs: float = FS, remove_bias: bool = True, **clean_kwargs):
    """Clean every IMU and merge them on one grid. Returns (wide, info).

    wide: t_s, segment, <seg>_ax_mps2 .. <seg>_gz_rads for each IMU. 'segment'
    numbers the stretches where all IMUs have data (-1 = at least one missing),
    so a stroke that crosses any IMU's dropout is rejected.
    info: {segment: clean_imu info + 'bias_source'}.
    """
    t0 = min(float(r["t_s"].iloc[0]) for r in raw.values())
    frames, info = {}, {}
    for s, r in raw.items():
        bias = lead_in_bias(r, fs) if remove_bias else None
        c, inf = ip.clean_imu(r, fs=fs, remove_gyro_bias=remove_bias, gyro_bias=bias, **clean_kwargs)
        if not remove_bias:
            inf["bias_source"] = "off"
        elif bias is not None:
            inf["bias_source"] = "lead-in"
        else:
            inf["bias_source"] = "still periods" if inf.get("gyro_bias") is not None else "none"
        frames[s], info[s] = c, inf

    k = {s: np.rint((c["t_s"].to_numpy() - t0) * fs).astype(int) for s, c in frames.items()}
    n = int(max(v.max() for v in k.values())) + 1
    wide = pd.DataFrame({"t_s": t0 + np.arange(n) / fs})
    for s, c in frames.items():
        for col in ip.SIGNALS:
            v = np.full(n, np.nan)
            v[k[s]] = c[col].to_numpy()
            wide[f"{s}_{col}"] = v
    bad = wide[suit_columns(frames)].isna().any(axis=1).to_numpy()
    seg = np.cumsum(np.diff(np.concatenate([[1], bad.astype(int)])) == -1)
    wide.insert(1, "segment", np.where(bad, -1, seg))
    return wide, info


def ref_view(wide: pd.DataFrame, segment: str = REF_SEGMENT) -> pd.DataFrame:
    """One segment as a canonical single-IMU frame (t_s, segment, ax_mps2 .. gz_rads).

    Same row index as `wide`, so start_idx / end_idx found on it cut every IMU.
    """
    cols = {f"{segment}_{c}": c for c in ip.SIGNALS}
    missing = [c for c in cols if c not in wide]
    if missing:
        raise ValueError(f"reference segment {segment!r} not in the suit data")
    return wide[["t_s", "segment", *cols]].rename(columns=cols)


def piece_quality(raw: dict, fs: float = FS) -> dict:
    """quality_report for every IMU, with the suit's configured ranges."""
    return {s: ip.quality_report(r, fs_nominal=fs, acc_range_g=ACC_RANGE_G,
                                 gyr_range_rads=GYR_RANGE_RADS, verbose=False)
            for s, r in raw.items()}


def process_piece(imu_source, pm5_source, *, segments=None, ref: str = REF_SEGMENT,
                  fs: float = FS, weight=None, height=None, offset_s=None,
                  skip_first: int = 5, n_bins: int = sd.N_BINS, method: str = "resample",
                  detect_kwargs: dict | None = None) -> dict:
    """One piece: suit CSV + Concept2 stroke CSV -> labelled 30-bin features.

    Strokes are detected on the reference segment and matched to the PM5 log
    (erg_stroke_sync.sync_strokes); each valid stroke is then binned for every
    IMU with the same boundaries.

    Returns a dict with:
      raw, quality, clean_info  per-IMU input, audit and cleaning info
      wide, ref                 merged clean frame and the reference view
      erg                       PM5 log with reliability flags
      strokes, diag             synced strokes and sync diagnostics
      binned                    [n_valid x n_bins x 6*n_imu]
      table                     one row per valid stroke: erg values + weight,
                                height + <seg>_<channel>_bNN features
      columns                   feature signal columns in order
    """
    raw = load_suit(imu_source, segments)
    if ref not in raw:
        raise ValueError(f"reference segment {ref!r} not in {list(raw)}")
    quality = piece_quality(raw, fs)
    wide, clean_info = clean_suit(raw, fs)
    rv = ref_view(wide, ref)
    erg = es.load_c2_strokes(pm5_source, skip_first=skip_first)
    strokes, diag = es.sync_strokes(rv, erg, fs=fs, offset_s=offset_s, detect_kwargs=detect_kwargs)
    cols = suit_columns(raw)
    binned, table = es.synced_features(wide, strokes, n_bins=n_bins, method=method,
                                       weight=weight, height=height, cols=cols)
    return dict(raw=raw, quality=quality, clean_info=clean_info, wide=wide, ref=rv, erg=erg,
                strokes=strokes, diag=diag, binned=binned, table=table, columns=cols)
