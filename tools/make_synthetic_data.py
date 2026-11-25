"""Make a synthetic data folder in the exact formats the real pipeline reads.

Suit CSVs as tools/record_piece.py saves them, Concept2 stroke exports, and
subjects.csv / pieces.csv. Use it to test build_dataset.py and train_model.py
without the suit. The numbers are made up: power follows drive speed and body
size through an invented formula, so a good fit here proves the plumbing, not
the physiology.

    python tools/make_synthetic_data.py --out data_synth
    python ml/build_dataset.py --data data_synth --out out_synth
    python ml/train_model.py --dataset out_synth/dataset.csv --out out_synth/model

Each subject rows three pieces at 20 spm (easy / moderate / hard: a faster drive
at higher power) after a 10 s still lead-in. Each IMU has its own mounting
orientation (the same for everyone, as in the suit), a gyro offset, and a few
failed reads; some pieces have a 150 ms USB stall.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
import erg_stroke_sync as es  # noqa: E402
import imu_preprocessing as ip  # noqa: E402
import stroke_detection as sd  # noqa: E402

SEGMENTS = ["forearm", "upperarm", "chest", "thigh", "shank"]
AMP_RAD = {"forearm": 1.3, "upperarm": 1.0, "chest": 0.6, "thigh": 1.2, "shank": 1.0}
SEED = {s: 10 + i for i, s in enumerate(SEGMENTS)}       # fixed mounting per segment
INTENSITIES = [("easy", 1.10), ("moderate", 0.95), ("hard", 0.82)]   # (label, drive time s)
FS = 100.0


def piece_power(drive_s: float, size: float, technique: float) -> float:
    """Invented: ~130 / 185 / 260 W for an average rower at the three drive times."""
    return 163.0 * drive_s ** -2.36 * size ** 2 * (1 + technique)


def write_suit_csv(path: Path, df: pd.DataFrame):
    """Same number formats as icm20948_mux5.ino (NaN prints as nan)."""
    fmt = ["%d", "%.6f"] + ["%.4f", "%.4f", "%.4f", "%.5f", "%.5f", "%.5f"] * len(SEGMENTS) + ["%d"]
    np.savetxt(path, df.to_numpy(), fmt=fmt, delimiter=",", header=",".join(df.columns), comments="")


def pm5_export(catches, onset_s, watts, rng) -> pd.DataFrame:
    """Concept2 Logbook stroke export: timing from erg_stroke_sync.simulate_pm5_log."""
    log = es.simulate_pm5_log(catches, onset_s=onset_s, log_delay_s=float(rng.uniform(0.6, 0.9)),
                              n_bad_start=4, seed=int(rng.integers(1 << 30)))
    w = np.array(watts, float)
    w[:4] *= np.linspace(0.55, 0.9, 4)                      # flywheel spin-up
    t = log["Time (seconds)"].to_numpy()
    speed = (w / 2.80) ** (1 / 3)                           # Concept2: P = 2.80 v^3
    dist = np.cumsum(speed * np.diff(np.r_[0.0, t]))
    return pd.DataFrame({"Number": log["Number"], "Time (seconds)": t,
                         "Distance (meters)": np.round(dist, 1),
                         "Pace (seconds)": np.round(500 / speed, 1),
                         "Watts": np.round(w).astype(int),
                         "Cal/Hr": np.round(4 * 0.8604 * w + 300).astype(int),
                         "Stroke Rate": log["Stroke Rate"].round(),
                         "Heart Rate": np.nan})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=2.0, help="rowing time per piece")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)

    subjects, pieces = [], []
    for k in range(args.subjects):
        sid = f"S{k + 1:02d}"
        size = float(rng.uniform(0.9, 1.1))
        technique = float(rng.normal(0, 0.04))              # power not visible in kinematics
        subjects.append(dict(subject=sid, weight_kg=round(60 + 150 * (size - 0.9) + rng.normal(0, 3), 1),
                             height_m=round(1.65 + 1.2 * (size - 0.9) + rng.normal(0, 0.02), 2),
                             age=int(rng.integers(19, 30)), sex=str(rng.choice(["M", "F"]))))
        (args.out / "raw" / sid).mkdir(exist_ok=True)
        for j, (label, drive) in enumerate(INTENSITIES):
            pid = f"P{j + 1}"
            dur = 10 + 60 * args.minutes
            sims = {}
            for s in SEGMENTS:
                sim, catches = sd.simulate_erg_imu(duration_s=dur, fs=FS, spm=(19.5, 20.5), drive_s=drive,
                                                   amp_rad=AMP_RAD[s] * size, noise=0.003,
                                                   seed=SEED[s], lead_s=10.0)
                sims[s] = sim
            n = len(sims["thigh"])
            df = pd.DataFrame({"sample": np.arange(n), "t_s": np.arange(n) / FS + rng.normal(0, 30e-6, n)})
            df.loc[0, "t_s"] = 0.0
            for s in SEGMENTS:
                gb = rng.normal(0, 0.05, 3)                 # gyro offset, ~3 deg/s
                for c in ip.ACC:
                    df[f"{s}_{c}"] = sims[s][c].to_numpy() + rng.normal(0, 0.02, n)
                for i, c in enumerate(ip.GYR):
                    df[f"{s}_{c}"] = sims[s][c].to_numpy() + gb[i] + rng.normal(0, 0.003, n)
                bad = rng.random(n) < 0.0005                # failed I2C reads
                df.loc[bad, [f"{s}_{c}" for c in ip.SIGNALS]] = np.nan
            df["read_us"] = rng.integers(2100, 2500, n)
            if (k + j) % 4 == 0:                            # USB stall: 150 ms of frames missing
                i0 = int(rng.integers(int(30 * FS), n - 500))
                df = df.drop(index=range(i0, i0 + 15)).reset_index(drop=True)
                df["sample"] = np.arange(len(df))
            imu_rel = f"raw/{sid}/{sid}_{pid}_imu.csv"
            write_suit_csv(args.out / imu_rel, df)

            p = piece_power(drive, size, technique)
            watts = p * (1 + rng.normal(0, 0.03, len(catches)))
            pm5_rel = f"raw/{sid}/{sid}_{pid}_pm5.csv"
            pm5_export(catches, catches[0], watts, rng).to_csv(args.out / pm5_rel, index=False)
            pieces.append(dict(subject=sid, piece=pid, intensity=label, drag_factor=120, imu_file=imu_rel,
                               pm5_file=pm5_rel, offset_s="", ref_segment="", use=1, notes="synthetic"))
        print(f"{sid}: 3 pieces, {', '.join(f'{piece_power(d, size, technique):.0f} W' for _, d in INTENSITIES)}")
    pd.DataFrame(subjects).to_csv(args.out / "subjects.csv", index=False)
    pd.DataFrame(pieces).to_csv(args.out / "pieces.csv", index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
