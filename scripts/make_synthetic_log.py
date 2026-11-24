#!/usr/bin/env python3
"""Write a synthetic raw_###.bin so the whole pipeline can run without hardware.

    python scripts/make_synthetic_log.py data/raw/synthetic.bin
    python scripts/make_synthetic_log.py data/raw/synthetic.bin --spm 20 24 --bout-s 120 --err-rate 0.002
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from rowing_ee.config import PipelineConfig
from rowing_ee.io.raw_log import write_raw_log
from rowing_ee.sim.synthetic import make_session


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--spm", type=float, nargs="+", default=[20.0], help="one bout per value")
    ap.add_argument("--bout-s", type=float, default=90.0)
    ap.add_argument("--rest-s", type=float, default=20.0)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--err-rate", type=float, default=0.0, help="per-IMU read failure probability")
    ap.add_argument("--gap-prob", type=float, default=0.0, help="per-frame missed-tick probability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    bouts = tuple((args.bout_s, s, args.intensity) for s in args.spm)
    sess = make_session(cfg, bouts=bouts, rest_s=args.rest_s, seed=args.seed,
                        err_rate=args.err_rate, gap_prob=args.gap_prob)
    write_raw_log(args.out, sess.n, sess.t_us, sess.raw, sess.err, fs_hz=cfg.fs,
                  n_imu=cfg.n_imu, feature_hash=cfg.feature_hash())
    print(f"wrote {args.out}: {len(sess.n)} samples ({len(sess.n) / cfg.fs:.1f} s), "
          f"~{sess.expected_strokes} strokes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
