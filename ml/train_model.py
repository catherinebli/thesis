"""Train and validate the IMU -> PM5 power model (Slade-style linear model).

Validation is leave-one-subject-out: each subject's strokes are predicted by a
model that never saw that subject. The model is ridge regression on
standardized features; the ridge strength is chosen inside each training fold
(cross-validated by subject), so the held-out subject never influences it.

Two baselines show what the IMUs add:
  mean   predict the training mean power (no information at all)
  body   ridge on weight + height only (everything except the IMUs)

Errors are reported per stroke and per piece (mean power over the piece, which
also sets the energy: kcal = sum(P * T) / efficiency).

Outputs, in --out (default out/model):
  metrics.json      stroke- and piece-level errors for each model
  predictions.csv   held-out prediction for every stroke
  pieces.csv        mean PM5 vs predicted power and kcal for every piece
  weights.csv       final model trained on all data, in Slade's order
                    [bias, weight, height, features ...] (see features.txt)
  features.txt      feature order matching weights.csv (after the bias)
  model_info.json   ridge strength, data size, feature hash
  pred_vs_pm5.png, pieces.png
  ablation.csv, ablation.png   with --ablation: same validation per IMU subset

Usage, from the repo root:
  python ml/train_model.py                    # uses out/dataset.csv
  python ml/train_model.py --ablation
  python ml/train_model.py --cv piece         # one subject only (e.g. the pilot)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import suit_plots

ROOT = Path(__file__).resolve().parents[1]
FEATURE_RE = re.compile(r"^(?P<seg>.+)_(?P<ch>[ag][xyz]_(?:mps2|rads))_b\d\d$")
ALPHAS = np.logspace(-1, 5, 13)
ETA = 0.25            # gross efficiency assumed for kcal (the PM5 uses the same factor of 4)
J_PER_KCAL = 4184.0


# --------------------------------------------------------------------------- #
def feature_columns(df: pd.DataFrame, segments=None) -> tuple[list[str], list[str]]:
    """(body columns, binned IMU columns) in dataset order, optionally only some segments."""
    body = [c for c in ("weight", "height") if c in df and df[c].notna().all()]
    bins = [c for c in df.columns if (m := FEATURE_RE.match(c))
            and (segments is None or m.group("seg") in segments)]
    return body, bins


def segments_in(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        m = FEATURE_RE.match(c)
        if m and m.group("seg") not in out:
            out.append(m.group("seg"))
    return out


def fit_ridge(X, y, groups, alphas=ALPHAS):
    """Standardize + ridge, strength chosen by grouped CV. Returns (model, alpha)."""
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    n_groups = len(np.unique(groups))
    if n_groups >= 3:
        cv, kw = GroupKFold(n_splits=min(5, n_groups)), {"groups": groups}
    else:
        cv, kw = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0), {}
    gs = GridSearchCV(pipe, {"ridge__alpha": alphas}, cv=cv, scoring="neg_mean_absolute_error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, y, **kw)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


def held_out_predictions(df, cols, y, groups, alphas=ALPHAS):
    """Prediction for every stroke from a model that never saw its group."""
    pred, chosen = np.full(len(y), np.nan), []
    X = df[cols].to_numpy(float) if cols else None
    for g in pd.unique(groups):
        test = groups == g
        if not cols:                                   # mean baseline
            pred[test] = y[~test].mean()
            continue
        model, a = fit_ridge(X[~test], y[~test], groups[~test], alphas)
        pred[test] = model.predict(X[test])
        chosen.append(a)
    return pred, chosen


def stroke_metrics(y, p) -> dict:
    e = p - y
    return dict(mae_w=float(np.mean(np.abs(e))), rmse_w=float(np.sqrt(np.mean(e ** 2))),
                mape_pct=float(100 * np.mean(np.abs(e) / y)), bias_w=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def piece_table(df, pred, target="watts") -> pd.DataFrame:
    """Mean power and energy per piece, PM5 vs predicted (same strokes)."""
    d = df[["subject", "piece", "intensity", "duration_s"]].copy()
    d["actual"], d["pred"] = df[target].to_numpy(float), pred
    d["work_j"], d["work_pred_j"] = d.actual * d.duration_s, d.pred * d.duration_s
    g = d.groupby(["subject", "piece"], sort=False).agg(
        intensity=("intensity", "first"), strokes=("actual", "size"), seconds=("duration_s", "sum"),
        actual=("actual", "mean"), pred=("pred", "mean"), work_j=("work_j", "sum"),
        work_pred_j=("work_pred_j", "sum")).reset_index()
    g["kcal_pm5"] = g.work_j / ETA / J_PER_KCAL
    g["kcal_pred"] = g.work_pred_j / ETA / J_PER_KCAL
    g["abs_pct"] = 100 * (g.pred - g.actual).abs() / g.actual
    g["label"] = g.subject + " " + g.piece
    return g.drop(columns=["work_j", "work_pred_j"])


def piece_metrics(pt) -> dict:
    return dict(mape_pct=float(pt.abs_pct.mean()), worst_pct=float(pt.abs_pct.max()),
                within_10pct=f"{int((pt.abs_pct <= 10).sum())}/{len(pt)}")


def evaluate(df, cols, y, groups, alphas=ALPHAS, target="watts") -> tuple[dict, np.ndarray]:
    pred, chosen = held_out_predictions(df, cols, y, groups, alphas)
    pt = piece_table(df, pred, target)
    m = dict(stroke=stroke_metrics(y, pred), piece=piece_metrics(pt))
    if chosen:
        m["alpha_median"] = float(np.median(chosen))
    return m, pred


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "out" / "dataset.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "model")
    ap.add_argument("--target", default="watts")
    ap.add_argument("--cv", choices=["subject", "piece"], default="subject",
                    help="hold out one subject (default) or one piece at a time")
    ap.add_argument("--ablation", action="store_true", help="also validate each IMU subset")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.dataset, dtype={"subject": str, "piece": str})
    df["intensity"] = df["intensity"].fillna("") if "intensity" in df else ""
    body, bins = feature_columns(df)
    if not bins:
        raise SystemExit("no binned feature columns (<seg>_<channel>_bNN) in the dataset")
    keep = df[[args.target, "duration_s"] + body + bins].notna().all(axis=1)
    if (~keep).any():
        print(f"dropping {int((~keep).sum())} strokes with missing values")
    df = df[keep].reset_index(drop=True)
    y = df[args.target].to_numpy(float)

    cv = args.cv
    if cv == "subject" and df.subject.nunique() < 3:
        print(f"only {df.subject.nunique()} subject(s): holding out one piece at a time instead")
        cv = "piece"
    groups = (df.subject if cv == "subject" else df.subject + "/" + df.piece).to_numpy()
    if len(np.unique(groups)) < 2:
        raise SystemExit("need at least two subjects (or pieces) to validate")
    n_sub, n_pc = df.subject.nunique(), df.groupby(["subject", "piece"]).ngroups
    print(f"{len(df)} strokes, {n_sub} subjects, {n_pc} pieces; {len(body)} body + {len(bins)} IMU features; "
          f"held out by {cv}")

    args.out.mkdir(parents=True, exist_ok=True)
    models = {"mean baseline": [], "weight + height": body, "IMU model": body + bins}
    metrics, preds = {}, {}
    for name, cols in models.items():
        if name == "weight + height" and not body:
            continue
        metrics[name], preds[name] = evaluate(df, cols, y, groups, target=args.target)

    out = df[["subject", "piece", "intensity"] + [c for c in ("erg_stroke", "t_start", "duration_s") if c in df]].copy()
    out["pm5_" + args.target] = y
    for name, p in preds.items():
        out["pred_" + name.split()[0].lower().replace("+", "")] = p
    out.to_csv(args.out / "predictions.csv", index=False)
    pt = piece_table(df, preds["IMU model"], args.target)
    pt.to_csv(args.out / "pieces.csv", index=False)

    # final model on everything, saved in Slade's layout
    cols = body + bins
    model, alpha = fit_ridge(df[cols].to_numpy(float), y, groups)
    sc, rg = model.named_steps["scale"], model.named_steps["ridge"]
    w = rg.coef_ / sc.scale_
    b = rg.intercept_ - float(np.sum(rg.coef_ * sc.mean_ / sc.scale_))
    X = df[cols].to_numpy(float)
    assert np.allclose(X @ w + b, model.predict(X), rtol=1e-6, atol=1e-6)
    np.savetxt(args.out / "weights.csv", np.r_[b, w], delimiter=",")
    (args.out / "features.txt").write_text("\n".join(cols) + "\n")
    fhash = hashlib.sha1("\n".join(cols).encode()).hexdigest()[:10]
    info = dict(target=args.target, alpha=alpha, strokes=len(df), subjects=n_sub, pieces=n_pc,
                features=len(cols), feature_hash=fhash, cv=cv, efficiency_for_kcal=ETA,
                weights_layout="[bias, " + ", ".join(cols[:2]) + ", ...] (see features.txt)")
    (args.out / "model_info.json").write_text(json.dumps(info, indent=2))

    if args.ablation:
        segs = segments_in(df)
        subsets = {"all IMUs": segs}
        named = {"legs (thigh + shank)": ["thigh", "shank"], "arms (forearm + upper arm)": ["forearm", "upperarm"]}
        subsets.update({k: v for k, v in named.items() if all(s in segs for s in v)})
        subsets.update({s: [s] for s in segs})
        rows = []
        for name, ss in subsets.items():
            _, b_s = feature_columns(df, ss)
            m, _ = evaluate(df, body + b_s, y, groups, target=args.target)
            rows.append(dict(subset=name, imu_features=len(b_s), piece_mape=m["piece"]["mape_pct"],
                             stroke_mape=m["stroke"]["mape_pct"], stroke_mae_w=m["stroke"]["mae_w"]))
            print(f"  ablation {name:28s} piece {m['piece']['mape_pct']:5.1f} %   stroke {m['stroke']['mape_pct']:5.1f} %")
        if body:
            rows.append(dict(subset="no IMUs (weight + height)", imu_features=0,
                             piece_mape=metrics["weight + height"]["piece"]["mape_pct"],
                             stroke_mape=metrics["weight + height"]["stroke"]["mape_pct"],
                             stroke_mae_w=metrics["weight + height"]["stroke"]["mae_w"]))
        ab = pd.DataFrame(rows).sort_values("piece_mape")
        ab.to_csv(args.out / "ablation.csv", index=False)
        suit_plots.ablation_bars(ab, args.out / "ablation.png")
        metrics["ablation"] = ab.to_dict(orient="records")

    metrics["_data"] = info
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imu_m = metrics["IMU model"]
    suit_plots.predicted_vs_actual(y, preds["IMU model"], df.subject.to_numpy(),
                                   f"Held-out strokes (by {cv}): MAE {imu_m['stroke']['mae_w']:.0f} W, "
                                   f"{imu_m['stroke']['mape_pct']:.1f} %", args.out / "pred_vs_pm5.png")
    suit_plots.piece_bars(pt, f"Mean power per piece, held out by {cv}: "
                          f"{imu_m['piece']['mape_pct']:.1f} % mean error", args.out / "pieces.png")

    print(f"\n{'model':18s} {'stroke MAE W':>12s} {'stroke err %':>12s} {'piece err %':>11s} {'worst %':>8s} {'<=10 %':>7s}")
    for name, m in metrics.items():
        if name.startswith("_") or name == "ablation":
            continue
        s, p = m["stroke"], m["piece"]
        print(f"{name:18s} {s['mae_w']:12.1f} {s['mape_pct']:12.1f} {p['mape_pct']:11.1f} {p['worst_pct']:8.1f} {p['within_10pct']:>7s}")
    print(f"\nfinal model: ridge alpha {alpha:g}, {len(cols)} features -> {args.out / 'weights.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
