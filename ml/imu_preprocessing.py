"""Preprocessing for 6-DoF IMU recordings of a rower on an erg.

Only the tri-axial accelerometer and gyroscope are used (6 DoF). Any other
channels in the file (magnetometer, temperature, quaternions, ...) are
dropped on load -- the magnetometer is unreliable next to an erg's flywheel
and steel frame anyway.

Works with any IMU CSV / DataFrame, not just our logger's format:
  * column names are auto-detected from common spellings (ax, acc_x, AccX,
    accel_x_g, gyro_z_dps, ...) or can be given explicitly via `columns=`;
  * accel in g or m/s^2, gyro in deg/s or rad/s (auto-detected or explicit);
  * time in s / ms / us, absolute datetimes, or no time column at all
    (then `fs` must be given);
  * sample rate is measured from the timestamps if not given.

Everything is converted to a canonical frame:
    t_s, ax_mps2, ay_mps2, az_mps2, gx_rads, gy_rads, gz_rads  [+ sample]

Typical use:
    clean, report, info = preprocess_imu("some_rower.csv")

or step by step:
    raw = load_imu(path)                  # canonical 6-DoF frame
    report = quality_report(raw)
    clean, info = clean_imu(raw)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

G = 9.80665

ACC = ["ax_mps2", "ay_mps2", "az_mps2"]
GYR = ["gx_rads", "gy_rads", "gz_rads"]
SIGNALS = ACC + GYR                       # the 6 DoF kept

# Erg-motion defaults. Rowing stroke rate is ~16-40 spm (0.3-0.7 Hz); body
# segment motion has essentially all its power below ~10 Hz, while the
# catch/finish transients and flywheel/chain vibration sit above that.
DEFAULT_ACC_CUTOFF = 12.0   # Hz
DEFAULT_GYR_CUTOFF = 12.0   # Hz


# --------------------------------------------------------------------------- #
# Loading / standardising arbitrary IMU input
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# regexes applied to the normalised (lowercase, alphanumeric-only) name
#   <sensor prefix><axis><optional unit suffix>, e.g. ax, AccX, accel_x_g,
#   gyro_z_dps, Gyroscope X (rad/s)
_ACC_PRE = r"(a|acc|accel|accelerometer|acceleration|linacc)"
_GYR_PRE = r"(g|gyr|gyro|gyroscope|w|omega|angvel|angularvelocity|rot)"
_UNIT_SUF = r"(mps2|ms2|mss|g|rads|rad|radpersec|dps|degs|deg|degpersec|raw)?"
_PATTERNS = {
    f"{p}{ax}_{u}": rf"^{pre}{ax}{_UNIT_SUF}$"
    for p, pre, u in [("a", _ACC_PRE, "mps2"), ("g", _GYR_PRE, "rads")]
    for ax in "xyz"
}
_TIME_PATTERN = r"^(t|time|timestamp|elapsed|elapsedtime|secondselapsed)(s|sec|ms|us)?$"
_SAMPLE_NAMES = ["sample", "samplenumber", "samplecount", "counter", "packetcounter"]


def _find_column(cols: list[str], pattern: str) -> str | None:
    hits = [c for c in cols if re.match(pattern, _norm(c))]
    return hits[0] if len(hits) >= 1 else None


def _unit_from_name(name: str, kind: str) -> str | None:
    n = _norm(name)
    if kind == "acc":
        if re.search(r"(mps2|ms2|mss)$", n):
            return "mps2"
        if n.endswith("g"):
            return "g"
    elif kind == "gyr":
        if re.search(r"(dps|degs|degpersec|deg)$", n):
            return "degs"
        if re.search(r"(rads|radpersec|rad)$", n):
            return "rads"
    elif kind == "time":
        if n.endswith("us"):
            return "us"
        if n.endswith("ms"):
            return "ms"
        if n.endswith("s"):
            return "s"
    return None


def _guess_acc_unit(a: np.ndarray) -> str:
    # gravity alone gives |a| ~ 1 g or ~ 9.81 m/s^2
    med = np.nanmedian(np.linalg.norm(a, axis=1))
    return "g" if med < 3.0 else "mps2"


def _guess_gyr_unit(w: np.ndarray) -> str:
    # Body-worn sensors during rowing stay well under ~20 rad/s (~1150 deg/s).
    # Values regularly above that are almost certainly deg/s.
    p99 = np.nanpercentile(np.abs(w), 99)
    return "degs" if p99 > 20.0 else "rads"


def load_imu(source, columns: dict | None = None, fs: float | None = None,
             acc_unit: str = "auto", gyr_unit: str = "auto",
             time_unit: str = "auto", verbose: bool = False) -> pd.DataFrame:
    """Load IMU data and return a canonical 6-DoF DataFrame.

    source:    path to a CSV, or an existing DataFrame.
    columns:   optional explicit mapping {canonical_name: name_in_file} for any
               of t_s, ax_mps2..az_mps2, gx_rads..gz_rads, sample. Anything not
               given is auto-detected.
    fs:        sample rate (Hz). Required only if there is no time column.
    acc_unit:  "auto" | "mps2" | "g"
    gyr_unit:  "auto" | "rads" | "degs"
    time_unit: "auto" | "s" | "ms" | "us"  (ignored for datetime columns)

    Magnetometer, temperature and any other columns are discarded.
    Detected mappings/units are stored in df.attrs["source_info"].
    """
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    cols = list(df.columns)
    columns = dict(columns or {})
    src: dict = {}

    # --- accel / gyro columns ------------------------------------------------
    for canon, pat in _PATTERNS.items():
        if canon not in columns:
            hit = canon if canon in cols else _find_column(cols, pat)
            if hit is None:
                raise ValueError(f"could not find a column for {canon}; columns are "
                                 f"{cols}. Pass columns={{'{canon}': '<name>'}}.")
            columns[canon] = hit
    if len({columns[c] for c in SIGNALS}) != 6:
        raise ValueError(f"accel/gyro column detection is ambiguous: "
                         f"{ {c: columns[c] for c in SIGNALS} }; pass columns= explicitly.")

    out = pd.DataFrame({c: pd.to_numeric(df[columns[c]], errors="coerce") for c in SIGNALS})

    # --- units ---------------------------------------------------------------
    if acc_unit == "auto":
        acc_unit = (_unit_from_name(columns["ax_mps2"], "acc")
                    or _guess_acc_unit(out[ACC].to_numpy()))
    if acc_unit == "g":
        out[ACC] *= G
    elif acc_unit != "mps2":
        raise ValueError(f"acc_unit must be 'mps2' or 'g', got {acc_unit!r}")

    if gyr_unit == "auto":
        gyr_unit = (_unit_from_name(columns["gx_rads"], "gyr")
                    or _guess_gyr_unit(out[GYR].to_numpy()))
    if gyr_unit == "degs":
        out[GYR] = np.deg2rad(out[GYR])
    elif gyr_unit != "rads":
        raise ValueError(f"gyr_unit must be 'rads' or 'degs', got {gyr_unit!r}")

    # --- time ----------------------------------------------------------------
    tcol = columns.get("t_s")
    if tcol is None:
        tcol = _find_column(cols, _TIME_PATTERN)
    if tcol is not None:
        traw = df[tcol]
        if pd.api.types.is_numeric_dtype(traw):
            t = traw.to_numpy(float)
            if time_unit == "auto":
                time_unit = _unit_from_name(tcol, "time") or "s"
                # sanity: a 'time' column in ms/us shows up as huge steps
                med_dt = np.nanmedian(np.diff(t))
                if time_unit == "s" and med_dt > 0.5:
                    time_unit = "ms" if med_dt < 500 else "us"
            t = t / {"s": 1.0, "ms": 1e3, "us": 1e6}[time_unit]
        else:
            ts = pd.to_datetime(traw, errors="coerce")
            t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
            time_unit = "datetime"
        out.insert(0, "t_s", t - np.nanmin(t))
    elif fs is not None:
        out.insert(0, "t_s", np.arange(len(out)) / fs)
        time_unit = f"none (synthesised at {fs} Hz)"
    else:
        raise ValueError("no time column found; pass fs= or columns={'t_s': '<name>'}.")

    # --- optional sample counter (for dropped-sample detection) -------------
    scol = columns.get("sample")
    if scol is None:
        by_norm = {_norm(c): c for c in cols}
        scol = next((by_norm[n] for n in _SAMPLE_NAMES if n in by_norm), None)
    if scol is not None:
        out.insert(0, "sample", pd.to_numeric(df[scol], errors="coerce"))

    out = out.dropna(subset=["t_s"]).reset_index(drop=True)
    src.update(columns={k: v for k, v in columns.items()}, time_column=tcol,
               sample_column=scol, acc_unit=acc_unit, gyr_unit=gyr_unit,
               time_unit=time_unit,
               dropped_columns=[c for c in cols if c not in
                                set(columns.values()) | {tcol, scol}])
    out.attrs["source_info"] = src
    if verbose:
        print(f"acc: {[columns[c] for c in ACC]} [{acc_unit}]  "
              f"gyr: {[columns[c] for c in GYR]} [{gyr_unit}]  "
              f"time: {tcol} [{time_unit}]")
        if src["dropped_columns"]:
            print(f"dropped: {src['dropped_columns']}")
    return out


def estimate_fs(df: pd.DataFrame) -> float:
    """Sample rate from the median timestamp spacing."""
    dt = np.diff(np.unique(df["t_s"].to_numpy()))
    return float(1.0 / np.median(dt))


# --------------------------------------------------------------------------- #
# Quality audit (does not modify data)
# --------------------------------------------------------------------------- #
def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def find_still(df: pd.DataFrame, fs: float, win_s: float = 1.0,
               acc_tol: float = 0.4, gyr_tol: float = 0.1) -> np.ndarray:
    """Boolean mask of samples where the sensor is stationary.

    Stationary = |a| within acc_tol (m/s^2) of g and |w| < gyr_tol (rad/s),
    sustained for at least win_s seconds. On the erg this is typically only
    the setup/rest period before the first stroke or after the last one.
    """
    a = np.linalg.norm(df[ACC].to_numpy(), axis=1)
    w = np.linalg.norm(df[GYR].to_numpy(), axis=1)
    raw = (np.abs(a - G) < acc_tol) & (w < gyr_tol)
    n = max(1, int(win_s * fs))
    still = np.zeros_like(raw)
    edges = np.diff(np.concatenate([[0], raw.astype(int), [0]]))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s >= n:
            still[s:e] = True
    return still


def quality_report(df: pd.DataFrame, fs_nominal: float | None = None,
                   acc_range_g: float | None = None,
                   gyr_range_rads: float | None = None,
                   verbose: bool = True) -> dict:
    """Summarise data quality problems of a canonical 6-DoF frame.

    fs_nominal: expected sample rate; if None it is estimated from the data.
    acc_range_g / gyr_range_rads: configured full-scale range of the sensor.
    If given, samples within 2% of full scale are counted as saturated.
    """
    t_raw = df["t_s"].to_numpy()
    r: dict = {}
    r["non_monotonic"] = int((np.diff(t_raw) <= 0).sum())   # in logged order
    t = np.unique(t_raw)                                      # sorted, deduped
    dt = np.diff(t)
    med_dt = np.median(dt)
    fs_nominal = fs_nominal or float(1 / med_dt)

    # timing
    r["n_samples"] = len(df)
    r["duration_s"] = float(t[-1] - t[0])
    r["fs_measured"] = float(1 / med_dt)
    r["fs_nominal"] = fs_nominal
    r["dt_jitter_ms"] = float(dt.std() * 1e3)
    r["gaps"] = int((dt > 1.5 / fs_nominal).sum())
    r["largest_gap_s"] = float(dt.max())
    if "sample" in df:
        ds = np.diff(df["sample"].to_numpy())
        r["dropped_samples"] = int((ds[ds > 1] - 1).sum())
        r["duplicate_samples"] = int((ds == 0).sum())

    # per-channel checks
    r["nan_count"] = {c: int(df[c].isna().sum()) for c in SIGNALS if df[c].isna().any()}
    r["repeat_frac"] = {}
    r["longest_flatline"] = {}
    for c in SIGNALS:
        same = np.diff(df[c].to_numpy()) == 0
        r["repeat_frac"][c] = float(same.mean())
        r["longest_flatline"][c] = _longest_run(same) + 1

    # saturation
    r["saturated"] = {}
    if acc_range_g:
        lim = 0.98 * acc_range_g * G
        for c in ACC:
            r["saturated"][c] = int((df[c].abs() >= lim).sum())
    if gyr_range_rads:
        lim = 0.98 * gyr_range_rads
        for c in GYR:
            r["saturated"][c] = int((df[c].abs() >= lim).sum())

    # physical sanity
    a = np.linalg.norm(df[ACC].to_numpy(), axis=1)
    r["acc_norm_mean"] = float(np.nanmean(a))
    still = find_still(df.fillna(0), fs_nominal)
    r["still_frac"] = float(still.mean())
    if still.any():
        r["acc_norm_still"] = float(a[still].mean())       # should be ~9.81
        r["gyro_bias_rads"] = df.loc[still, GYR].mean().round(4).to_dict()

    # warnings
    w = []
    if abs(r["fs_measured"] - fs_nominal) > 0.02 * fs_nominal:
        w.append(f"sample rate {r['fs_measured']:.1f} Hz != nominal {fs_nominal} Hz")
    if r["non_monotonic"]:
        w.append(f"{r['non_monotonic']} non-monotonic timestamps")
    if r["gaps"]:
        w.append(f"{r['gaps']} timing gaps (largest {r['largest_gap_s']*1e3:.0f} ms)")
    if r.get("dropped_samples"):
        w.append(f"{r['dropped_samples']} dropped samples (sample counter)")
    if r["nan_count"]:
        w.append(f"NaNs: {r['nan_count']}")
    for c in SIGNALS:
        if r["longest_flatline"][c] >= 5:
            w.append(f"{c} flatlined for {r['longest_flatline'][c]} samples (stuck sensor / bus error?)")
    if any(r["saturated"].values()):
        w.append(f"saturation: { {k: v for k, v in r['saturated'].items() if v} }")
    if "acc_norm_still" in r and abs(r["acc_norm_still"] - G) > 0.3:
        w.append(f"|a| at rest = {r['acc_norm_still']:.2f} m/s^2 (expect {G:.2f}) -> accel calibration needed")
    if not still.any():
        w.append("no stationary period found -> gyro bias can't be estimated from this file "
                 "(record a few seconds sitting still before rowing, or pass gyro_bias=)")
    if abs(r["acc_norm_mean"] - G) > 3:
        w.append(f"mean |a| = {r['acc_norm_mean']:.1f} m/s^2 -> check accel units (g vs m/s^2)")
    r["warnings"] = w

    if verbose:
        print(f"{r['n_samples']} samples, {r['duration_s']:.2f} s, "
              f"fs = {r['fs_measured']:.2f} Hz (jitter {r['dt_jitter_ms']:.3f} ms)")
        print(f"|a| mean {r['acc_norm_mean']:.2f} m/s^2; still {r['still_frac']:.0%} of the time")
        print("WARNINGS:" if w else "No warnings.")
        for x in w:
            print("  -", x)
    return r


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #
def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop duplicate / non-increasing timestamps."""
    df = df.sort_values("t_s", kind="stable")
    df = df[~df["t_s"].duplicated(keep="first")]
    return df.reset_index(drop=True)


def resample_uniform(df: pd.DataFrame, fs: float, max_gap_s: float = 0.05,
                     cols: list[str] | None = None) -> pd.DataFrame:
    """Linearly interpolate onto a uniform time grid.

    Gaps longer than max_gap_s are NOT filled; samples in them become NaN and
    a 'segment' column labels each contiguous good stretch.
    """
    cols = cols or SIGNALS
    df = df.dropna(subset=cols)
    t = df["t_s"].to_numpy()
    t_new = np.arange(t[0], t[-1] + 0.5 / fs, 1 / fs)
    out = pd.DataFrame({"t_s": t_new})
    for c in cols:
        out[c] = np.interp(t_new, t, df[c].to_numpy())

    # blank out long gaps
    dt = np.diff(t)
    bad = np.zeros(len(t_new), bool)
    for i in np.flatnonzero(dt > max_gap_s):
        bad |= (t_new > t[i]) & (t_new < t[i + 1])
    out.loc[bad, cols] = np.nan
    seg = np.cumsum(np.diff(np.concatenate([[1], bad.astype(int)])) == -1)
    out["segment"] = np.where(bad, -1, seg)
    return out


def hampel(x: np.ndarray, half_win: int = 3, n_sigma: float = 6.0):
    """Replace isolated spikes with the local median. Returns (y, mask).

    Threshold = n_sigma * max(local MAD, global sample-to-sample spread), so
    smooth-but-fast motion (e.g. the drive phase) is not flagged; only
    single-sample glitches far outside normal motion are replaced.
    """
    s = pd.Series(x)
    w = 2 * half_win + 1
    med = s.rolling(w, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=1).median()
    step = 1.4826 * np.median(np.abs(np.diff(x)))       # robust std of diff
    thresh = n_sigma * np.maximum(1.4826 * mad, step)
    mask = (s - med).abs() > thresh
    y = s.where(~mask, med).to_numpy()
    return y, mask.to_numpy()


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (offline only; filtfilt is non-causal)."""
    cutoff = min(cutoff, 0.45 * fs)       # keep below Nyquist for low-rate IMUs
    sos = signal.butter(order, cutoff, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _apply_by_segment(df, cols, fn, min_len: int = 30):
    out = df.copy()
    for seg, idx in df.groupby("segment").groups.items():
        if seg < 0:
            continue
        block = df.loc[idx, cols].to_numpy()
        if len(block) < min_len:       # too short to filter reliably
            out.loc[idx, cols] = np.nan
            continue
        out.loc[idx, cols] = fn(block)
    return out


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def clean_imu(raw: pd.DataFrame, fs: float | None = None,
              acc_cutoff: float = DEFAULT_ACC_CUTOFF,
              gyr_cutoff: float = DEFAULT_GYR_CUTOFF,
              max_gap_s: float | None = None,
              despike: bool = True, remove_gyro_bias: bool = True,
              gyro_bias: dict | None = None):
    """Clean a canonical 6-DoF frame (from load_imu). Returns (clean_df, info).

    Steps: fix timestamps -> uniform resample (long gaps -> NaN segments)
    -> despike accel/gyro -> gyro bias removal -> low-pass.

    fs:        output sample rate; defaults to the measured rate of the input.
    max_gap_s: longest gap that is interpolated; defaults to 5 samples.
    gyro_bias: optional {gx_rads: b, ...} from a separate static recording.
               If None and remove_gyro_bias, bias is estimated from still
               periods in this file (skipped if there are none).
    """
    info: dict = {}
    df = fix_timestamps(raw[["t_s"] + SIGNALS])
    info["dropped_bad_timestamps"] = len(raw) - len(df)

    fs = fs or estimate_fs(df)
    max_gap_s = max_gap_s if max_gap_s is not None else 5.0 / fs
    info["fs"] = fs

    df = resample_uniform(df, fs, max_gap_s)
    info["n_segments"] = int(df.loc[df.segment >= 0, "segment"].nunique())
    info["gap_samples"] = int((df.segment < 0).sum())

    if despike:
        info["spikes_replaced"] = {}
        for c in SIGNALS:
            ok = df[c].notna().to_numpy()
            y, m = hampel(df.loc[ok, c].to_numpy())
            df.loc[ok, c] = y
            info["spikes_replaced"][c] = int(m.sum())

    if remove_gyro_bias:
        if gyro_bias is None:
            still = find_still(df.fillna(0), fs)
            if still.sum() >= fs:          # need >= 1 s of stillness
                gyro_bias = df.loc[still, GYR].mean().to_dict()
        if gyro_bias is not None:
            for c in GYR:
                df[c] = df[c] - gyro_bias[c]
        info["gyro_bias"] = gyro_bias      # None => not removed

    min_len = max(30, int(0.3 * fs))
    df = _apply_by_segment(df, ACC, lambda b: lowpass(b, fs, acc_cutoff), min_len)
    df = _apply_by_segment(df, GYR, lambda b: lowpass(b, fs, gyr_cutoff), min_len)

    df["acc_norm"] = np.linalg.norm(df[ACC].to_numpy(), axis=1)
    df["gyr_norm"] = np.linalg.norm(df[GYR].to_numpy(), axis=1)
    return df, info


def preprocess_imu(source, *, columns: dict | None = None,
                   fs_in: float | None = None, fs_out: float | None = None,
                   acc_unit: str = "auto", gyr_unit: str = "auto",
                   time_unit: str = "auto", out_path=None,
                   verbose: bool = True, **clean_kwargs):
    """One-call pipeline for any rower's IMU file: load -> audit -> clean.

    source:  CSV path or DataFrame.
    fs_in:   input sample rate (only needed if the file has no time column;
             also used as the nominal rate for the quality report).
    fs_out:  output sample rate (default: measured input rate).
    out_path: if given, the cleaned frame is written there as CSV.
    clean_kwargs: passed to clean_imu (acc_cutoff, gyr_cutoff, gyro_bias, ...).

    Returns (clean_df, report, info).
    """
    raw = load_imu(source, columns=columns, fs=fs_in, acc_unit=acc_unit,
                   gyr_unit=gyr_unit, time_unit=time_unit, verbose=verbose)
    report = quality_report(raw, fs_nominal=fs_in, verbose=verbose)
    clean, info = clean_imu(raw, fs=fs_out, **clean_kwargs)
    info["source"] = raw.attrs.get("source_info", {})
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(out_path, index=False)
    return clean, report, info


def preprocess_folder(in_dir, out_dir, pattern: str = "*.csv", **kwargs) -> pd.DataFrame:
    """Run preprocess_imu on every matching file; returns a summary table."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    rows = []
    for p in sorted(in_dir.glob(pattern)):
        try:
            _, rep, inf = preprocess_imu(p, out_path=out_dir / p.name,
                                         verbose=False, **kwargs)
            rows.append({"file": p.name, "fs": round(inf["fs"], 2),
                         "duration_s": round(rep["duration_s"], 1),
                         "acc_unit": inf["source"].get("acc_unit"),
                         "gyr_unit": inf["source"].get("gyr_unit"),
                         "gaps": rep["gaps"], "segments": inf["n_segments"],
                         "spikes": sum(inf.get("spikes_replaced", {}).values()),
                         "gyro_bias_removed": inf.get("gyro_bias") is not None,
                         "warnings": "; ".join(rep["warnings"])})
        except Exception as e:                      # keep going on bad files
            rows.append({"file": p.name, "error": str(e)})
    return pd.DataFrame(rows)
