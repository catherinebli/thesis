"""Erg-guided stroke segmentation: use the Concept2 PM5 stroke log to say
*which* strokes happened, and the simultaneous IMU to say *exactly when*.

Why both
--------
* The PM5 stroke export (Concept2 logbook CSV) has one row per stroke with
  power, pace, stroke rate, distance - but its timestamps are quantised to
  0.5 s ticks (intervals come out as 1.5 / 2.0 s at ~36 spm), so a single
  stroke's time is only good to +-0.25 s. The first few strokes (start
  sequence, flywheel spin-up) are also unreliable.
* The IMU resolves the catch to ~10 ms, but on its own it can't tell you
  which erg stroke (and therefore which power value) a cycle belongs to.

Pipeline
--------
1. load_c2_strokes   : read the PM5 CSV, flag unreliable strokes
                       (first `skip_first`, missing spm, rate outliers,
                       double-logged strokes).
2. imu_onset         : PM5 timer starts on the first pull of the flywheel, so
                       t_erg = 0 <-> the IMU's first movement after the rower
                       sits still. This is the coarse clock offset tau0.
                       (Or pass offset_s from a manual sync.)
3. fine offset       : the PM logs each stroke a roughly constant delay after
                       its catch (around the end of the drive). Search one
                       stroke period for the single offset Delta that best
                       lines up all *reliable* erg strokes with the IMU-detected
                       catches: catch_k ~= t_erg_k + Delta.
4. local catch search: for every erg stroke, look for the IMU catch (zero
                       crossing into the drive) within +-window of its predicted
                       time. A linear fit (offset + clock drift) of the matches
                       refines the prediction and the search is repeated.
5. strokes           : stroke k = [catch_k, catch_{k+1}), output only for
                       reliable erg strokes whose own and next catch were both
                       found. Each row carries the erg's watts / pace / spm etc.
                       The table plugs straight into stroke_detection.bin_strokes.

Assumption to keep in mind: the PM logs a stroke between slightly before its
catch and late in its recovery (search window d in [-0.15, 0.85] x period;
log_delay_s includes ~0.25 s average rounding to the PM tick).
Check `diag["log_delay_s"]` and the ambiguity table; if a neighbouring
one-stroke shift scores almost as well, sync manually (offset_s=...).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import signal

import stroke_detection as sd
from imu_preprocessing import GYR, estimate_fs

PM_TICK_S = 0.5     # resolution of PM5 stroke timestamps


# --------------------------------------------------------------------------- #
# 1. Concept2 stroke log
# --------------------------------------------------------------------------- #
_C2_COLS = {  # normalised substring -> canonical name
    "number": "erg_stroke", "time": "t_erg", "distance": "dist_m",
    "pace": "pace_500m_s", "watts": "watts", "calhr": "cal_hr",
    "strokerate": "spm", "heartrate": "hr",
}


def load_c2_strokes(path, skip_first: int = 5, rate_tol: float = 0.2,
                    rate_win: int = 15) -> pd.DataFrame:
    """Read a Concept2 stroke-data CSV and flag reliable strokes.

    skip_first: number of strokes at the start to treat as unreliable
                (start sequence / flywheel spin-up).
    rate_tol:   a stroke is a rate outlier if its spm differs from the centred
                rolling median (rate_win strokes) by more than this fraction.
    """
    raw = pd.read_csv(path)
    ren = {}
    for c in raw.columns:
        n = re.sub(r"[^a-z]", "", c.lower())
        for key, canon in _C2_COLS.items():
            if n.startswith(key) and canon not in ren.values():
                ren[c] = canon
                break
    df = raw.rename(columns=ren)
    if "t_erg" not in df:
        raise ValueError(f"no time column found in {list(raw.columns)}")
    if "erg_stroke" not in df:
        df.insert(0, "erg_stroke", np.arange(1, len(df) + 1))
    df = df.sort_values("t_erg").reset_index(drop=True)

    reasons = pd.Series("", index=df.index)
    reasons[df.index < skip_first] += "start;"
    if "spm" in df:
        reasons[df.spm.isna()] += "no_spm;"
        med = df.spm.rolling(rate_win, center=True, min_periods=3).median()
        reasons[(df.spm - med).abs() > rate_tol * med] += "rate_outlier;"
    dt = df.t_erg.diff()
    reasons[dt < 0.6 * dt.median()] += "double_logged;"
    df["unreliable_reason"] = reasons.str.rstrip(";")
    df["reliable"] = reasons == ""
    return df


# --------------------------------------------------------------------------- #
# 2. Coarse sync: IMU movement onset
# --------------------------------------------------------------------------- #
def imu_onset(clean: pd.DataFrame, fs: float | None = None,
              lp_hz: float = 2.0, rise_frac: float = 0.25,
              rest_frac: float = 0.05) -> float:
    """Time (s) the rower starts moving: the PM5 timer starts on the first pull.

    Smoothed |gyro| (each axis low-passed first, so sensor noise doesn't bias
    the magnitude) is compared with its rest level (5th percentile) and its
    rowing level (95th percentile). Finds the first time it rises rise_frac of
    the way from rest to rowing, then backs up to where it last sat within
    rest_frac of rest (the actual start of the motion).
    """
    fs = fs or estimate_fs(clean)
    g = np.column_stack([sd._lowpass(sd._fill_nans(clean[c].to_numpy()), fs, lp_hz)
                         for c in GYR])
    w = np.linalg.norm(g, axis=1)
    rest, ref = np.percentile(w, 5), np.percentile(w, 95)
    above = np.flatnonzero(w > rest + rise_frac * (ref - rest))
    if len(above) == 0:
        raise ValueError("no movement found in IMU data")
    i = above[0]
    quiet = np.flatnonzero(w[:i] < rest + rest_frac * (ref - rest))
    j = quiet[-1] + 1 if len(quiet) else 0
    return float(clean.t_s.iloc[j])


# --------------------------------------------------------------------------- #
# 3-4. Fine alignment + local catch search
# --------------------------------------------------------------------------- #
def _nearest_dist(a, b):
    """For each a, distance to nearest b (b sorted)."""
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.minimum(np.abs(a - b[idx - 1]), np.abs(a - b[idx]))


def _offset_score(t_erg, catches, delta, sigma=0.15):
    r = _nearest_dist(t_erg + delta, catches)
    return float(np.exp(-0.5 * (r / sigma) ** 2).mean())


def _find_catch(x, t, t_pred, win_s, look_ahead, min_peak):
    """Upward zero crossing of x (into the drive) nearest t_pred, followed by a
    real drive peak. Returns sample index or -1."""
    lo = np.searchsorted(t, t_pred - win_s)
    hi = np.searchsorted(t, t_pred + win_s)
    lo, hi = max(lo, 1), min(hi, len(x) - 1)
    if hi <= lo:
        return -1
    seg = x[lo - 1:hi + 1]
    up = np.flatnonzero((seg[:-1] <= 0) & (seg[1:] > 0)) + lo
    up = [i for i in up if x[i:i + look_ahead].max(initial=-np.inf) > min_peak]
    if not up:
        return -1
    return int(min(up, key=lambda i: abs(t[i] - t_pred)))


def sync_strokes(clean: pd.DataFrame, erg: pd.DataFrame, fs: float | None = None,
                 offset_s: float | None = None, win_s: float = 0.4,
                 delay_range: tuple = (-0.15, 0.85), dur_tol: float = 0.25,
                 detect_kwargs: dict | None = None):
    """Align the PM5 stroke log to the IMU and return erg-labelled strokes.

    clean:      cleaned 6-DoF IMU frame (imu_preprocessing output).
    erg:        output of load_c2_strokes.
    offset_s:   IMU time at which the PM5 timer read 0. If None, taken from
                imu_onset() (rower must sit still before the first pull).
    win_s:      half-width of the per-stroke catch search (>= PM tick/2 + slack).
    delay_range: allowed PM logging delay after the catch, x stroke period.
    dur_tol:    stroke flagged invalid if IMU duration differs from the
                erg's 60/spm by more than this fraction.

    Returns (strokes, diag). `strokes` has stroke_detection's columns
    (start_idx, end_idx, t_start, t_end, duration_s, rate_spm, valid) plus the
    erg columns, one row per erg stroke that got an IMU catch.
    """
    fs = fs or estimate_fs(clean)
    t = clean.t_s.to_numpy()
    detect_kwargs = detect_kwargs or {}
    imu, x = sd.detect_strokes(clean, fs=fs, event="catch", return_signal=True,
                               **detect_kwargs)
    imu_catches = np.r_[imu.t_start.to_numpy(), imu.t_end.iloc[-1:]] if len(imu) else np.array([])
    if len(imu_catches) < 3:
        raise ValueError("too few IMU strokes detected to sync")
    period = float(np.median(np.diff(imu_catches)))
    min_peak = 0.3 * np.median(x[imu.peak_idx])
    look_ahead = int(0.6 * period * fs)

    diag: dict = {"imu_strokes_detected": len(imu), "imu_period_s": period}

    # --- coarse: PM timer zero in IMU time ----------------------------------
    if offset_s is None:
        tau0 = imu_onset(clean, fs)
        diag["sync"] = "imu_onset"
        if tau0 < 1.0:
            diag["warning"] = ("IMU movement starts <1 s into the recording - no still "
                               "lead-in, onset sync is unreliable; pass offset_s.")
    else:
        tau0 = float(offset_s)
        diag["sync"] = "manual"
    diag["tau0_s"] = tau0

    # --- fine: one offset Delta, catch ~= t_erg + Delta ---------------------
    rel = erg[erg.reliable]
    t_rel = rel.t_erg.to_numpy()
    d_lo, d_hi = delay_range[0] * period, delay_range[1] * period
    grid = np.arange(tau0 - d_hi, tau0 - d_lo, 0.01)
    scores = np.array([_offset_score(t_rel, imu_catches, d) for d in grid])
    delta = float(grid[np.argmax(scores)])
    diag["log_delay_s"] = tau0 - delta          # PM log time after the catch
    diag["offset_score"] = float(scores.max())


    # --- local catch search, two passes (2nd uses offset+drift fit) ---------
    t_erg_all = erg.t_erg.to_numpy()
    use = np.flatnonzero(erg.index >= erg.index[erg.reliable].min()) if erg.reliable.any() \
        else np.arange(len(erg))
    pred = t_erg_all + delta
    for _ in range(2):
        cidx = np.full(len(erg), -1)
        for k in use:
            cidx[k] = _find_catch(x, t, pred[k], win_s, look_ahead, min_peak)
        ok = (cidx >= 0) & erg.reliable.to_numpy()
        if ok.sum() >= 5:
            b, a = np.polyfit(t_erg_all[ok], t[cidx[ok]], 1)
            pred = a + b * t_erg_all
    diag["clock_drift_ppm"] = float((b - 1) * 1e6) if ok.sum() >= 5 else np.nan
    resid = t[cidx[ok]] - pred[ok]
    diag["resid_std_s"] = float(resid.std())
    diag["reliable_matched"] = f"{ok.sum()}/{int(erg.reliable.sum())}"

    # Label check. In a steady piece, shifting every label by one stroke fits
    # the timing almost as well, so which erg stroke is which rests on the
    # start anchor. Count IMU catches from movement onset to the first reliable
    # erg stroke's catch: it should equal the number of erg strokes before it.
    if ok.any():
        k0 = int(np.flatnonzero(ok)[0])
        n_imu = int(((imu_catches >= tau0 - 0.3) & (imu_catches < t[cidx[k0]] - 0.3)).sum())
        diag["start_count_check"] = {"erg_strokes_before": k0, "imu_catches_before": n_imu}
        if n_imu != k0:
            diag["warning"] = (diag.get("warning", "") + f" Start count mismatch: {k0} erg "
                               f"strokes but {n_imu} IMU catches before the first reliable "
                               "stroke - labels may be shifted (short start strokes can be "
                               "missed by the IMU detector; check the plot).").strip()

    # --- assemble strokes: [catch_k, catch_{k+1}) ---------------------------
    rows = []
    erg_cols = [c for c in ["erg_stroke", "t_erg", "dist_m", "pace_500m_s", "watts",
                            "cal_hr", "spm", "hr", "reliable", "unreliable_reason"] if c in erg]
    for k in range(len(erg) - 1):
        s, e = cidx[k], cidx[k + 1]
        if s < 0:
            continue
        row = erg.iloc[k][erg_cols].to_dict()
        row.update(start_idx=int(s), end_idx=int(e) if e > s else -1,
                   t_catch=t[s], catch_resid_s=t[s] - pred[k])
        rows.append(row)
    st = pd.DataFrame(rows)
    has_end = st.end_idx > 0
    st["t_start"] = st.t_catch
    st["t_end"] = np.where(has_end, t[st.end_idx.clip(lower=0)], np.nan)
    st["duration_s"] = st.t_end - st.t_start
    st["rate_spm"] = 60.0 / st.duration_s
    gap = np.zeros(len(st), bool)
    if "segment" in clean:
        seg = clean.segment.to_numpy()
        gap = np.array([e > 0 and (seg[s:e] != seg[s]).any()
                        for s, e in zip(st.start_idx, st.end_idx)])
    dur_ok = np.ones(len(st), bool)
    if "spm" in st:
        dur_ok = (st.duration_s - 60 / st.spm).abs() <= dur_tol * 60 / st.spm
    st["valid"] = st.reliable.astype(bool) & has_end & ~gap & dur_ok
    st.attrs.update(fs=fs, **{k: v for k, v in diag.items() if not isinstance(v, dict)})
    diag["valid_strokes"] = int(st.valid.sum())
    return st, diag


def synced_features(clean, strokes, n_bins: int = sd.N_BINS, method: str = "resample",
                    weight=None, height=None,
                    targets=("watts", "spm", "pace_500m_s", "cal_hr", "hr")):
    """Bin valid synced strokes (Slade, 30 bins) and attach erg targets.

    Returns (binned [n x n_bins x 6], table) where table has stroke ids,
    timing, the 180 binned features and the erg values for each stroke.
    """
    binned, used = sd.bin_strokes(clean, strokes, n_bins=n_bins, method=method)
    feats = sd.stroke_features(binned, weight=weight, height=height)
    meta = used[["erg_stroke", "t_erg", "t_start", "duration_s", "rate_spm"]
                + [c for c in targets if c in used]].reset_index(drop=True)
    return binned, pd.concat([meta, feats], axis=1)


# --------------------------------------------------------------------------- #
# Synthetic PM5 log for testing
# --------------------------------------------------------------------------- #
def simulate_pm5_log(true_catches, onset_s, log_delay_s=0.7, tick=PM_TICK_S,
                     n_bad_start=4, seed=0) -> pd.DataFrame:
    """PM5-style stroke log for known IMU catch times.

    The PM timer starts at onset_s (IMU time); each stroke is logged
    log_delay_s after its catch, rounded up to the PM's 0.5 s tick; the first
    n_bad_start strokes get start-sequence garbage (no spm / jittered times).
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(true_catches)
    phase = rng.uniform(0, tick)
    t_log = c + log_delay_s - onset_s
    t_q = np.ceil((t_log - phase) / tick) * tick + phase
    T = np.diff(c, append=c[-1] + np.diff(c)[-1])
    spm = np.round(60 / T).astype(float)
    t_q[:n_bad_start] = np.sort(t_q[:n_bad_start] + rng.uniform(-0.5, 0.5, n_bad_start))
    spm[:2] = np.nan
    spm[2:n_bad_start] *= rng.uniform(1.3, 1.8, max(0, n_bad_start - 2))
    watts = np.round(300 + 20 * rng.normal(size=len(c)))
    return pd.DataFrame({"Number": np.arange(1, len(c) + 1),
                         "Time (seconds)": np.round(t_q, 1),
                         "Watts": watts, "Stroke Rate": spm})
