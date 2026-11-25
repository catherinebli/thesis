"""Build the stroke-level dataset from every recorded piece.

Inputs, in the data folder (default: data/ at the repo root):
  subjects.csv  subject, weight_kg, height_m, age, sex
  pieces.csv    subject, piece, intensity, drag_factor, imu_file, pm5_file,
                offset_s, ref_segment, use, notes
                (imu_file / pm5_file relative to the data folder; offset_s,
                ref_segment, use and notes may be blank)

Outputs, in the out folder (default: out/ at the repo root):
  dataset.csv     one row per valid stroke: subject, piece, intensity, the PM5
                  values for that stroke (watts, spm, pace ...), weight, height and
                  the 900 binned IMU features (5 IMUs x 6 channels x 30 bins)
  pieces_qc.csv   one row per piece: sync diagnostics, data problems, review flag
  figures/<subject>_<piece>.png   check sheet for each piece

Usage, from the repo root:
  python ml/build_dataset.py                      # everything in data/pieces.csv
  python ml/build_dataset.py --subject S01        # one subject, e.g. the pilot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import multi_imu as mi
import suit_plots

ROOT = Path(__file__).resolve().parents[1]
NO = {"0", "n", "no", "false", "skip", "x"}


def _use(v) -> bool:
    """pieces.csv 'use' column: blank or 1 = include; 0 / no / skip = leave out."""
    s = str(v).strip().lower()
    if s in NO:
        return False
    try:
        return float(s) != 0          # "0.0" after a round trip through Excel
    except ValueError:
        return True


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def read_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(data_dir / "subjects.csv", dtype={"subject": str})
    pieces = pd.read_csv(data_dir / "pieces.csv", dtype={"subject": str, "piece": str})
    for name, df, need in [("subjects.csv", subjects, ["subject", "weight_kg", "height_m"]),
                           ("pieces.csv", pieces, ["subject", "piece", "imu_file", "pm5_file"])]:
        missing = [c for c in need if c not in df]
        if missing:
            raise SystemExit(f"{name} is missing columns {missing}")
    dup = subjects.subject[subjects.subject.duplicated()].tolist()
    if dup:
        raise SystemExit(f"subjects.csv lists these subjects twice: {dup}")
    return subjects.set_index("subject"), pieces


def piece_qc(res: dict) -> dict:
    """Sync diagnostics and data problems for one processed piece."""
    diag, erg, strokes, wide = res["diag"], res["erg"], res["strokes"], res["wide"]
    v = strokes[strokes.valid]
    q = dict(duration_s=round(float(wide.t_s.iloc[-1] - wide.t_s.iloc[0]), 1),
             erg_strokes=len(erg), erg_reliable=int(erg.reliable.sum()),
             imu_strokes=diag.get("imu_strokes_detected"), valid_strokes=int(len(v)),
             reliable_matched=diag.get("reliable_matched"), sync=diag.get("sync"),
             tau0_s=round(diag.get("tau0_s", np.nan), 2),
             log_delay_s=round(diag.get("log_delay_s", np.nan), 3),
             resid_std_ms=round(1e3 * diag.get("resid_std_s", np.nan), 1),
             clock_drift_ppm=round(diag.get("clock_drift_ppm", np.nan), 1),
             mean_watts=round(float(v.watts.mean()), 1) if "watts" in v and len(v) else np.nan,
             mean_spm=round(float(v.spm.mean()), 1) if "spm" in v and len(v) else np.nan)
    scc = diag.get("start_count_check", {})
    q["start_count_ok"] = bool(scc) and scc.get("erg_strokes_before") == scc.get("imu_catches_before")
    q["sync_warning"] = diag.get("warning", "")

    no_bias = [s for s, inf in res["clean_info"].items() if inf.get("bias_source") != "lead-in"]
    q["no_lead_in_bias"] = ",".join(no_bias)
    timing = next(iter(res["quality"].values()))            # every IMU shares the timestamps
    q["timing_gaps"] = timing["gaps"]
    q["largest_gap_ms"] = round(1e3 * timing["largest_gap_s"])
    failed = {s: max(rep.get("nan_count", {}).values(), default=0) for s, rep in res["quality"].items()}
    q["failed_reads"] = ", ".join(f"{s} {n}" for s, n in failed.items() if n)
    warns, sat = [], []
    for i, (s, rep) in enumerate(res["quality"].items()):
        sat += [f"{s}:{c}" for c, n in rep.get("saturated", {}).items() if n]
        for w in rep["warnings"]:
            if w.startswith(("NaNs", "saturation")) or "no stationary period" in w:
                continue                     # counted above / bias comes from the lead-in
            if any(k in w for k in ("sample rate", "non-monotonic", "timing gaps", "dropped samples")):
                if i == 0:
                    warns.append(w)
                continue
            warns.append(f"{s}: {w}")
    q["saturated"] = ",".join(sat)
    q["quality_warnings"] = " | ".join(warns)
    reasons = []
    if q["valid_strokes"] < 10:
        reasons.append("few valid strokes")
    if not q["start_count_ok"]:
        reasons.append("start count mismatch")
    if not np.isnan(q["resid_std_ms"]) and q["resid_std_ms"] > 200:
        reasons.append("large sync residual")
    if q["sync_warning"]:
        reasons.append("sync warning")
    if no_bias:
        reasons.append("no still lead-in")
    q["review"] = "; ".join(reasons)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--subject", action="append", help="only these subjects (repeatable)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    subjects, pieces = read_tables(args.data)
    if args.subject:
        pieces = pieces[pieces.subject.isin(args.subject)]
    if "use" in pieces:
        pieces = pieces[pieces["use"].map(_use)]
    if pieces.empty:
        raise SystemExit("no pieces to process")
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    rows, qc = [], []
    for _, p in pieces.iterrows():
        subj, piece = str(p.subject), str(p.piece)
        info = dict(subject=subj, piece=piece,
                    intensity=p.get("intensity", ""), drag_factor=p.get("drag_factor", np.nan))
        print(f"{subj} {piece}: ", end="", flush=True)
        try:
            if subj not in subjects.index:
                raise ValueError(f"subject {subj} is not in subjects.csv")
            s = subjects.loc[subj]
            ref = mi.REF_SEGMENT if _blank(p.get("ref_segment")) else str(p.ref_segment).strip()
            off = None if _blank(p.get("offset_s")) else float(p.offset_s)
            imu, pm5 = args.data / str(p.imu_file), args.data / str(p.pm5_file)
            for f in (imu, pm5):
                if not f.exists():
                    raise FileNotFoundError(f"{f} not found")
            res = mi.process_piece(imu, pm5, ref=ref, offset_s=off,
                                   weight=float(s.weight_kg), height=float(s.height_m))
        except Exception as e:                      # keep going; the QC table says why
            qc.append({**info, "status": "error", "review": f"{type(e).__name__}: {e}"})
            print(f"ERROR {type(e).__name__}: {e}")
            continue
        q = piece_qc(res)
        qc.append({**info, "status": "ok", **q})
        t = res["table"]
        t.insert(0, "subject", subj)
        t.insert(1, "piece", piece)
        t.insert(2, "intensity", info["intensity"])
        t.insert(3, "drag_factor", info["drag_factor"])
        for c in ("age", "sex"):
            if c in subjects:
                t.insert(4, c, s.get(c))
        rows.append(t)
        if not args.no_figures:
            suit_plots.piece_sheet(res, f"{subj} {piece} ({info['intensity']})",
                                   figdir / f"{subj}_{piece}.png")
        print(f"{q['valid_strokes']} strokes, {q['mean_watts']} W" + (f"  REVIEW: {q['review']}" if q["review"] else ""))

    qc = pd.DataFrame(qc)
    qc.to_csv(args.out / "pieces_qc.csv", index=False)
    if not rows:
        print("no piece produced strokes; see pieces_qc.csv")
        return 1
    ds = pd.concat(rows, ignore_index=True)
    ds.to_csv(args.out / "dataset.csv", index=False)
    n_feat = ds.filter(regex=r"_b\d\d$").shape[1]
    print(f"\n{len(ds)} strokes from {ds.subject.nunique()} subjects / "
          f"{ds.groupby(['subject', 'piece']).ngroups} pieces, {n_feat} IMU features each "
          f"-> {args.out / 'dataset.csv'}")
    flagged = qc[qc.review.fillna("") != ""]
    if len(flagged):
        print(f"{len(flagged)} piece(s) to review -> {args.out / 'pieces_qc.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
