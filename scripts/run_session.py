#!/usr/bin/env python3
"""Raw SD log -> per-stroke power and total kcal.

    python scripts/run_session.py data/raw/raw_000.bin --subject S01
    python scripts/run_session.py data/raw/raw_000.bin --mass 72 --height 1.83 --age 22 --sex M --plot

Subject lookup order: --mass/--height/... flags, then --subject ID in
data/subjects.yaml, then config subject_defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from rowing_ee.config import PipelineConfig
from rowing_ee.io.raw_log import RawLog
from rowing_ee.model.export import placeholder_model
from rowing_ee.model.linear_model import LinearModel
from rowing_ee.pipeline import Pipeline
from rowing_ee.subject import Subject, load_subjects


def resolve_subject(args, cfg: PipelineConfig) -> Subject:
    base = Subject.from_dict(cfg.subject_defaults)
    if args.subject:
        subjects = load_subjects(args.subjects)
        if args.subject not in subjects:
            raise SystemExit(f"{args.subject} not found in {args.subjects}")
        base = subjects[args.subject]
    return Subject(
        id=base.id,
        mass_kg=args.mass if args.mass is not None else base.mass_kg,
        height_m=args.height if args.height is not None else base.height_m,
        age=args.age if args.age is not None else base.age,
        sex=(args.sex or base.sex).upper()[:1],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--weights", help="trained weights (CSV/NPZ); default: placeholder zeros")
    ap.add_argument("--subject", help="subject ID in --subjects")
    ap.add_argument("--subjects", default=str(ROOT / "data" / "subjects.yaml"))
    ap.add_argument("--mass", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--age", type=float)
    ap.add_argument("--sex")
    ap.add_argument("--out", help="per-stroke CSV (default: data/est/<log name>.csv)")
    ap.add_argument("--plot", action="store_true", help="also write a PNG next to the CSV")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    log = RawLog.read(args.log)
    if log.header["feature_hash"] != cfg.feature_hash():
        print(f"note: log was recorded with feature hash 0x{log.header['feature_hash']:08X}, "
              f"current config is 0x{cfg.feature_hash():08X}")
    model = placeholder_model(cfg) if not args.weights else LinearModel.load(args.weights)
    subject = resolve_subject(args, cfg)

    res = Pipeline.run_session(log, cfg, model, subject)
    out = Path(args.out) if args.out else ROOT / "data" / "est" / (Path(args.log).stem + ".csv")
    res.to_csv(out)

    print(json.dumps({"timing": log.timing_report(), "result": res.summary()}, indent=2))
    if model.placeholder:
        print("NOTE: placeholder weights -> every stroke is billed at resting power until a model is trained.")
    print(f"per-stroke CSV: {out}")
    if args.plot:
        from plot_session import plot_session
        png = out.with_suffix(".png")
        plot_session(res, png)
        print(f"plot: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
