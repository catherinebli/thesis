#!/usr/bin/env python3
"""config/pipeline.yaml (+ optional trained weights) -> firmware/include/*.h

    python scripts/gen_headers.py                      # placeholder weights
    python scripts/gen_headers.py --weights model.csv  # trained weights, one per line, intercept first
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from rowing_ee.config import PipelineConfig
from rowing_ee.model.export import placeholder_model, write_config_header, write_weights_header
from rowing_ee.model.linear_model import LinearModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--weights", help="CSV or NPZ of D weights (normalization folded in)")
    ap.add_argument("--out", default=str(ROOT / "firmware" / "include"))
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    model = placeholder_model(cfg) if not args.weights else LinearModel.load(args.weights)
    if model.feature_hash is not None and model.feature_hash != cfg.feature_hash():
        print(f"warning: weights were trained with feature hash 0x{model.feature_hash:08X}, "
              f"config is 0x{cfg.feature_hash():08X}; the firmware will refuse to compile")
    write_config_header(cfg, f"{args.out}/PipelineConfig.h")
    write_weights_header(model, cfg, f"{args.out}/ModelWeights.h")
    kind = "placeholder (all zeros)" if model.placeholder else args.weights
    print(f"wrote {args.out}/PipelineConfig.h and ModelWeights.h  D={cfg.D}  "
          f"feature_hash=0x{cfg.feature_hash():08X}  weights={kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
