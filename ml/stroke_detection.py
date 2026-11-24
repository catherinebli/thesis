"""Rowing-stroke detection and 30-bin discretisation of 6-DoF erg IMU data.

Mirrors the Slade et al. gait pipeline (see estimation.py: checkPeaks /
binData / processRawGait), with heel strikes replaced by rowing catches:

  Slade (walking)                        here (erg rowing)
  -------------------------------------  ----------------------------------------
  shank gyro-z, low-pass 6 Hz            dominant-rotation gyro signal, low-pass
  find_peaks(height, min distance)       find_peaks(prominence, distance from the
                                         measured stroke rate)
  gait cycle = heel strike -> next one   stroke = catch -> next catch
  signal.resample(cycle, 30)             signal.resample(stroke, 30)
  flatten [features x bins]              flatten [features x bins]

Input is the cleaned frame from imu_preprocessing (t_s, ax..az_mps2,
gx..gz_rads, optional 'segment'). Works for any sensor placement/orientation:
by default the detection signal is the first principal component of the gyro,
i.e. angular velocity about the segment's dominant rotation axis (the sagittal
plane axis for a thigh, shank or trunk sensor on the erg).

Catch definition: rowing alternates a fast drive (~1/3 of the stroke) and a
slower recovery. The detection signal is signed so the drive is positive
(the larger, sharper excursion). Each drive produces one peak; the catch is the
zero crossing just before it, where the segment reverses from the recovery
direction into the drive. Use event="peak" to cut at the drive peaks instead
(closest analogue of Slade's raw peak cut).

Typical use:
    clean = pd.read_csv("..._clean.csv")
    strokes, binned, feats = segment_strokes(clean)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from imu_preprocessing import ACC, GYR, SIGNALS, G, estimate_fs

N_BINS = 30                    # Slade et al.
STROKE_BAND_HZ = (0.25, 1.0)   # 15-60 strokes/min covers rest paddling -> sprint


# --------------------------------------------------------------------------- #
# Detection signal
# --------------------------------------------------------------------------- #
def _lowpass(x, fs, cutoff, order=4):
    sos = signal.butter(order, min(cutoff, 0.45 * fs), fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _fill_nans(x):
    """Linear fill so filters/PCA can run across short NaN gaps."""
    s = pd.Series(x)
    return s.interpolate(limit_direction="both").to_numpy()


def detection_signal(df: pd.DataFrame, channel: str = "auto") -> tuple[np.ndarray, str]:
    """1-D signal used to find strokes. Returns (x, description).

    channel:
      "auto" / "gyr_pc1": projection of the gyro onto its first principal
                          axis (placement-independent).
      a column name, e.g. "gy_rads", or "-gy_rads" to flip it.
    """
    if channel in ("auto", "gyr_pc1"):
        w = np.column_stack([_fill_nans(df[c].to_numpy()) for c in GYR])
        w = w - w.mean(axis=0)
        _, _, vt = np.linalg.svd(w, full_matrices=False)
        axis = vt[0]
        return w @ axis, f"gyro PC1 axis {np.round(axis, 2).tolist()}"
    sign = -1.0 if channel.startswith("-") else 1.0
    col = channel.lstrip("-")
    return sign * _fill_nans(df[col].to_numpy()), channel


def orient_drive_positive(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Flip x so the drive (faster, larger angular velocity) is positive."""
    xc = x - np.median(x)
    pos, neg = np.percentile(xc, 99), -np.percentile(xc, 1)
    s = 1 if pos >= neg else -1
    return s * x, s


def estimate_stroke_rate(x: np.ndarray, fs: float,
                         band: tuple[float, float] = STROKE_BAND_HZ) -> float:
    """Dominant stroke frequency (Hz) from the autocorrelation peak.

    Autocorrelation is used instead of a spectral peak because the stroke
    waveform is strongly non-sinusoidal (fast drive, slow recovery).
    """
    x = x - x.mean()
    n = len(x)
    ac = signal.fftconvolve(x, x[::-1], mode="full")[n - 1:]
    ac /= ac[0]
    lo, hi = int(fs / band[1]), min(int(fs / band[0]), n - 1)
    pk, _ = signal.find_peaks(ac[lo:hi])
    if len(pk) == 0:
        raise ValueError("no periodic stroke pattern found in the stroke band")
    lag = lo + pk[np.argmax(ac[lo:hi][pk])]
    return fs / lag


# --------------------------------------------------------------------------- #
# Stroke detection
# --------------------------------------------------------------------------- #
def detect_strokes(df: pd.DataFrame, fs: float | None = None,
                   channel: str = "auto", event: str = "catch",
                   lowpass_hz: float = 3.0, min_dist_frac: float = 0.6,
                   prominence_frac: float = 0.3, valid_range: tuple = (0.6, 1.5),
                   return_signal: bool = False):
    """Find stroke boundaries. Returns a DataFrame with one row per stroke.

    fs:              sample rate (default: from t_s).
    channel:         see detection_signal().
    event:           "catch" (zero crossing before each drive peak, default)
                     or "peak" (drive peak itself).
    lowpass_hz:      low-pass for the *detection signal only* (Slade used 6 Hz
                     for walking; rowing is ~2x slower).
    min_dist_frac:   min spacing between drives, as a fraction of the
                     estimated stroke period (Slade: fixed 0.6 s).
    prominence_frac: min peak prominence as a fraction of the signal's
                     5-95 % range (replaces Slade's fixed 70 deg/s threshold so
                     it works for any sensor location / rower).
    valid_range:     a stroke is flagged invalid if its duration is outside
                     valid_range x the median duration, or it spans a data gap.

    Columns: start_idx, end_idx (exclusive, = next stroke's start_idx),
    t_start, t_end, duration_s, rate_spm, peak_idx, valid.
    """
    fs = fs or estimate_fs(df)
    x_raw, desc = detection_signal(df, channel)
    x = _lowpass(x_raw, fs, lowpass_hz)
    x, sign = orient_drive_positive(x)
    # The segment returns to the same angle every stroke, so its angular
    # velocity averages to zero over whole strokes: centre on the mean (robust
    # to residual gyro bias). Not the median - the long, slow recovery would
    # drag it negative and shift the catch earlier.
    x = x - x.mean()

    f_stroke = estimate_stroke_rate(x, fs)
    period = fs / f_stroke                                     # samples
    rng = np.percentile(x, 95) - np.percentile(x, 5)
    peaks, _ = signal.find_peaks(x, distance=max(1, int(min_dist_frac * period)),
                                 prominence=prominence_frac * rng)
    peaks = peaks[x[peaks] > 0]

    if event == "peak":
        bounds = peaks
    elif event == "catch":
        bounds = []
        look = int(0.6 * period)
        for p in peaks:
            lo = max(0, p - look)
            neg = np.flatnonzero(x[lo:p] <= 0)
            if len(neg):                       # last sample at/below zero, +1
                bounds.append(lo + neg[-1] + 1)
            else:                              # never crossed: use local min
                bounds.append(lo + int(np.argmin(x[lo:p])))
        bounds = np.asarray(bounds, int)
    else:
        raise ValueError("event must be 'catch' or 'peak'")

    t = df["t_s"].to_numpy()
    seg = df["segment"].to_numpy() if "segment" in df else np.zeros(len(df))
    nan_any = df[SIGNALS].isna().any(axis=1).to_numpy()
    rows = []
    for k in range(len(bounds) - 1):
        s, e = int(bounds[k]), int(bounds[k + 1])
        rows.append({"start_idx": s, "end_idx": e, "t_start": t[s], "t_end": t[e],
                     "duration_s": t[e] - t[s], "peak_idx": int(peaks[k]),
                     "gap": bool(nan_any[s:e].any() or (seg[s:e] != seg[s]).any())})
    strokes = pd.DataFrame(rows, columns=["start_idx", "end_idx", "t_start", "t_end",
                                          "duration_s", "peak_idx", "gap"])
    strokes["rate_spm"] = 60.0 / strokes["duration_s"]
    if len(strokes):
        med = strokes.loc[~strokes.gap, "duration_s"].median()
        ok_len = strokes.duration_s.between(valid_range[0] * med, valid_range[1] * med)
        strokes["valid"] = ok_len & ~strokes.gap
    else:
        strokes["valid"] = pd.Series(dtype=bool)
    strokes = strokes.drop(columns="gap")
    strokes.attrs.update(fs=fs, channel=desc, sign=sign, event=event,
                         est_rate_spm=60 * f_stroke)
    if return_signal:
        return strokes, x
    return strokes


# --------------------------------------------------------------------------- #
# Binning (Slade: each cycle -> 30 bins)
# --------------------------------------------------------------------------- #
def bin_stroke(block: np.ndarray, n_bins: int = N_BINS, method: str = "resample") -> np.ndarray:
    """[time x features] -> [n_bins x features].

    method:
      "resample": scipy.signal.resample (Fourier), exactly as Slade's binData.
      "interp":   linear interpolation at n_bins evenly spaced points.
      "mean":     mean of the samples falling in each of n_bins equal-time bins.
    """
    n = len(block)
    if method == "resample":
        return signal.resample(block, n_bins, axis=0)
    if method == "interp":
        src = np.arange(n) / n
        dst = (np.arange(n_bins) + 0.5) / n_bins
        return np.column_stack([np.interp(dst, src, block[:, j]) for j in range(block.shape[1])])
    if method == "mean":
        edges = np.linspace(0, n, n_bins + 1).round().astype(int)
        return np.vstack([block[a:max(b, a + 1)].mean(axis=0)
                          for a, b in zip(edges[:-1], edges[1:])])
    raise ValueError("method must be 'resample', 'interp' or 'mean'")


def bin_strokes(df: pd.DataFrame, strokes: pd.DataFrame, cols: list[str] = SIGNALS,
                n_bins: int = N_BINS, method: str = "resample",
                only_valid: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Bin every stroke. Returns (binned [n_strokes x n_bins x n_cols], strokes_used)."""
    use = strokes[strokes.valid] if only_valid else strokes
    data = df[cols].to_numpy()
    binned = np.stack([bin_stroke(data[s:e], n_bins, method)
                       for s, e in zip(use.start_idx, use.end_idx)]) if len(use) \
        else np.empty((0, n_bins, len(cols)))
    return binned, use.reset_index(drop=True)


def stroke_features(binned: np.ndarray, cols: list[str] = SIGNALS,
                    weight: float | None = None, height: float | None = None) -> pd.DataFrame:
    """Flatten to one row per stroke in Slade's order: all bins of feature 1,
    then all bins of feature 2, ... (i.e. [features x bins].flatten()).

    If weight/height are given they are prepended as columns, as in Slade's
    model input ([1, weight, height, bins...]; add the bias term in the model).
    """
    n, nb, nf = binned.shape
    flat = binned.transpose(0, 2, 1).reshape(n, nf * nb)
    names = [f"{c}_b{b:02d}" for c in cols for b in range(nb)]
    out = pd.DataFrame(flat, columns=names)
    if height is not None:
        out.insert(0, "height", height)
    if weight is not None:
        out.insert(0, "weight", weight)
    return out


def segment_strokes(df: pd.DataFrame, n_bins: int = N_BINS, method: str = "resample",
                    cols: list[str] = SIGNALS, weight=None, height=None, **detect_kwargs):
    """detect_strokes -> bin_strokes -> stroke_features in one call.

    Returns (strokes, binned, features); `strokes` has all detected strokes
    (see the 'valid' column), binned/features only the valid ones, row-aligned
    with strokes[strokes.valid].
    """
    strokes = detect_strokes(df, **detect_kwargs)
    binned, _ = bin_strokes(df, strokes, cols, n_bins, method)
    feats = stroke_features(binned, cols, weight, height)
    return strokes, binned, feats


# --------------------------------------------------------------------------- #
# Synthetic erg data (for testing the detector against known catches)
# --------------------------------------------------------------------------- #
def simulate_erg_imu(duration_s: float = 60.0, fs: float = 100.0,
                     spm: tuple[float, float] = (20.0, 30.0), drive_s: float = 0.8,
                     amp_rad: float = 1.0, noise: float = 0.05, seed: int = 0,
                     lead_s: float = 1.0):
    """Thigh-like IMU during erg rowing with a known catch time for each stroke.

    Segment angle swings amp_rad between catch and finish; drive takes ~drive_s,
    recovery fills the rest of the stroke (so higher rates shorten recovery,
    as on a real erg). Stroke rate ramps linearly over spm. The rower sits
    still at the catch for lead_s before the first stroke. The rotation axis
    and the sensor's mounting orientation are random.
    Returns (df in canonical 6-DoF format, true catch times in s).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    theta = np.zeros_like(t)
    catches, tc = [], lead_s
    while tc < duration_s:
        catches.append(tc)
        T = 60.0 / np.interp(tc, [lead_s, duration_s], spm)
        d = min(drive_s, 0.45 * T)
        u = (t - tc) / d
        v = (t - tc - d) / (T - d)
        m1, m2 = (u >= 0) & (u < 1), (v >= 0) & (v < 1)
        theta[m1] = amp_rad * 0.5 * (1 + np.cos(np.pi * u[m1]))   # catch -> finish
        theta[m2] = amp_rad * 0.5 * (1 - np.cos(np.pi * v[m2]))   # finish -> catch
        tc += T
    theta[t < catches[0]] = amp_rad
    omega = np.gradient(theta, 1 / fs)

    # random rotation axis in a random sensor frame
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    axis = q[:, 0]
    gyr = omega[:, None] * axis[None, :]
    # gravity seen in sensor frame rotates about the axis by theta (Rodrigues)
    g0 = q[:, 1] * G
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    acc = g0 * c + np.cross(axis, g0) * s + axis * (axis @ g0) * (1 - c)
    acc += 0.3 * np.gradient(omega, 1 / fs)[:, None] * q[:, 2]    # tangential accel
    gyr += rng.normal(0, noise, gyr.shape)
    acc += rng.normal(0, 3 * noise, acc.shape)
    df = pd.DataFrame(np.column_stack([t, acc, gyr]), columns=["t_s"] + ACC + GYR)
    return df, np.asarray(catches)
