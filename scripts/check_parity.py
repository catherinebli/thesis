#!/usr/bin/env python3
"""Prove the C++ core and the Python package agree on the same raw log.

    # build the native replay of firmware/lib/core and compare on a log
    python scripts/check_parity.py data/raw/synthetic.bin
    # compare against the est_###.csv the Teensy itself wrote
    python scripts/check_parity.py data/raw/raw_000.bin --est data/est_from_sd/est_000.csv --tol 1e-4

Exit code 0 = every stroke boundary matches and every power is within --tol (relative).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from rowing_ee.config import PipelineConfig
from rowing_ee.io.raw_log import RawLog
from rowing_ee.model.export import placeholder_model
from rowing_ee.model.linear_model import LinearModel
from rowing_ee.pipeline import Pipeline, read_est_csv
from rowing_ee.subject import Subject

NATIVE = ROOT / "firmware" / "test" / "native"


def build_replay() -> Path:
    subprocess.run(["make", "-s", "-C", str(NATIVE), "replay"], check=True)
    return NATIVE / "replay"


def subject_cfg_text(s: Subject) -> str:
    return (f"id={s.id}\nmass_kg={s.mass_kg!r}\nheight_m={s.height_m!r}\n"
            f"age={s.age!r}\nsex={s.sex}\n")


def rel(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def compare(py, cpp, kcal_py, kcal_cpp, tol) -> list[str]:
    problems = []
    if len(py) != len(cpp):
        problems.append(f"stroke count differs: python {len(py)} vs c++ {len(cpp)}")
    for i, (a, b) in enumerate(zip(py, cpp)):
        if (a.n0, a.n1, a.valid) != (b.n0, b.n1, b.valid):
            problems.append(f"stroke {i}: boundaries/validity differ {(a.n0, a.n1, a.valid)} vs {(b.n0, b.n1, b.valid)}")
            break
        for name in ("P_w", "P_raw_w", "kcal_total"):
            d = rel(getattr(a, name), getattr(b, name))
            if d > tol:
                problems.append(f"stroke {i} {name}: {getattr(a, name)!r} vs {getattr(b, name)!r} (rel {d:.2e})")
    if kcal_cpp is not None and rel(kcal_py, kcal_cpp) > tol:
        problems.append(f"total kcal: python {kcal_py!r} vs c++ {kcal_cpp!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--weights", help="weights CSV used by both sides (default placeholder)")
    ap.add_argument("--est", help="compare against this est.csv instead of running the native replay")
    ap.add_argument("--mass", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--age", type=float)
    ap.add_argument("--sex")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    base = Subject.from_dict(cfg.subject_defaults)
    subj = Subject(base.id, args.mass or base.mass_kg, args.height or base.height_m,
                   args.age or base.age, (args.sex or base.sex).upper()[:1])
    model = LinearModel.load(args.weights) if args.weights else placeholder_model(cfg)
    py = Pipeline.run_session(RawLog.read(args.log), cfg, model, subj, keep_signal=False)

    if args.est:
        cpp, kcal_cpp = read_est_csv(args.est)
    else:
        replay = build_replay()
        with tempfile.TemporaryDirectory() as td:
            scfg = Path(td) / "subject.cfg"
            scfg.write_text(subject_cfg_text(subj))
            out = Path(td) / "est_cpp.csv"
            cmd = [str(replay), args.log, str(out), "--subject", str(scfg)]
            if args.weights:
                cmd += ["--weights", args.weights]
            subprocess.run(cmd, check=True)
            cpp, kcal_cpp = read_est_csv(out)

    problems = compare(py.strokes, cpp, py.kcal_total, kcal_cpp, args.tol)
    print(f"python: {len(py.strokes)} strokes, {py.kcal_total:.6f} kcal | "
          f"c++: {len(cpp)} strokes, {kcal_cpp if kcal_cpp is None else round(kcal_cpp, 6)} kcal")
    if problems:
        print("PARITY FAILED")
        for p in problems[:20]:
            print("  " + p)
        return 1
    print(f"PARITY OK (tolerance {args.tol:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
