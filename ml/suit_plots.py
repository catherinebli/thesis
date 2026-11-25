"""Figures for the 5-IMU pipeline: one check sheet per piece (build_dataset.py)
and model results (train_model.py). Saved as PNG, no display needed."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import imu_preprocessing as ip  # noqa: E402
import stroke_detection as sd  # noqa: E402


def piece_sheet(res: dict, title: str, path, window_s: float = 25.0):
    """Sync check + per-IMU stroke profiles for one processed piece (multi_imu.process_piece)."""
    rv, strokes, diag, erg = res["ref"], res["strokes"], res["diag"], res["erg"]
    binned, cols = res["binned"], res["columns"]
    segs = list(res["raw"])
    t = rv["t_s"].to_numpy()
    _, x = sd.detect_strokes(rv, return_signal=True)
    tau0 = diag["tau0_s"]

    fig = plt.figure(figsize=(13, 4.2 + 2.1 * len(segs)), constrained_layout=True)
    gs = fig.add_gridspec(2 + len(segs), 2, height_ratios=[1.3, 1.0] + [1.0] * len(segs))

    ax = fig.add_subplot(gs[0, :])
    m = (t >= tau0 - 3) & (t <= tau0 + window_s)
    ax.plot(t[m], x[m], color="k", lw=0.8, label="reference IMU detection signal (drive +)")
    ax.axvline(tau0, color="C1", lw=1.5, label="movement onset = PM5 t 0")
    e = erg["t_erg"].to_numpy() + tau0
    e = e[(e >= t[m][0]) & (e <= t[m][-1])]
    ax.plot(e, np.full(len(e), np.nanmax(x[m]) * 1.1), "v", color="C0", ms=5, label="PM5 stroke log time")
    s = strokes[(strokes.t_start >= t[m][0]) & (strokes.t_start <= t[m][-1])]
    ok = s.valid.to_numpy(bool)
    ax.plot(s.t_start[ok], x[s.start_idx[ok]], "o", color="C2", ms=5, label="synced catch (used)")
    ax.plot(s.t_start[~ok], x[s.start_idx[~ok]], "o", color="C3", ms=5, label="catch (not used)")
    ax.set_xlim(t[m][0], t[m][-1])
    ax.set_xlabel("IMU time (s)")
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_title(f"{title}: start of the piece, first {window_s:.0f} s after onset", fontsize=10)

    v = strokes[strokes.valid]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(v.t_start, v.catch_resid_s * 1e3, ".", ms=3)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("IMU time (s)")
    ax.set_ylabel("catch - predicted (ms)")
    ax.set_title(f"Sync residual: std {1e3 * diag['resid_std_s']:.0f} ms, "
                 f"matched {diag['reliable_matched']}", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    if "watts" in erg:
        ax.plot(erg.t_erg + tau0, erg.watts, color="0.75", lw=0.8, label="PM5, all strokes")
    if "watts" in v:
        ax.plot(v.t_start, v.watts, ".", ms=4, color="C0", label="strokes used")
    ax.set_xlabel("IMU time (s)")
    ax.set_ylabel("PM5 power (W)")
    ax.legend(fontsize=7)
    ax.set_title(f"{len(v)} strokes used, mean {v.watts.mean():.0f} W" if "watts" in v and len(v)
                 else "no valid strokes", fontsize=9)

    pct = np.arange(binned.shape[1]) / binned.shape[1] * 100 if len(binned) else np.array([])
    for i, seg in enumerate(segs):
        for j, (group, unit) in enumerate([(ip.ACC, "m/s²"), (ip.GYR, "rad/s")]):
            ax = fig.add_subplot(gs[2 + i, j])
            for k, c in enumerate(group):
                if not len(binned):
                    break
                y = binned[:, :, cols.index(f"{seg}_{c}")]
                mu, sdv = y.mean(0), y.std(0)
                ax.plot(pct, mu, color=f"C{k}", lw=1.5, label=c.split("_")[0])
                ax.fill_between(pct, mu - sdv, mu + sdv, color=f"C{k}", alpha=0.15)
            ax.set_ylabel(f"{seg}\n{unit}", fontsize=9)
            if i == 0:
                ax.set_title("accelerometer" if j == 0 else "gyroscope", fontsize=9)
                ax.legend(fontsize=7, ncol=3, loc="upper right")
            if i == len(segs) - 1:
                ax.set_xlabel("% stroke (catch -> next catch), mean ± SD")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def predicted_vs_actual(y, pred, groups, title, path, unit="W"):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    for k, g in enumerate(sorted(set(groups))):
        m = np.asarray(groups) == g
        ax.plot(y[m], pred[m], ".", ms=4, alpha=0.6, color=f"C{k % 10}", label=str(g))
    lim = [min(np.min(y), np.min(pred)), max(np.max(y), np.max(pred))]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel(f"PM5 ({unit})")
    ax.set_ylabel(f"predicted ({unit})")
    ax.set_title(title, fontsize=10)
    if len(set(groups)) <= 12:
        ax.legend(fontsize=7, ncol=2, title="held-out", title_fontsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def piece_bars(pieces, title, path):
    """pieces: DataFrame with label, actual, pred (mean W per piece)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(pieces) + 2), 4), constrained_layout=True)
    xx = np.arange(len(pieces))
    ax.bar(xx - 0.2, pieces["actual"], 0.4, label="PM5")
    ax.bar(xx + 0.2, pieces["pred"], 0.4, label="IMU model (held out)")
    ax.set_xticks(xx)
    ax.set_xticklabels(pieces["label"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean power over the piece (W)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ablation_bars(table, path):
    """table: DataFrame with subset, piece_mape (sorted)."""
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(table) + 1.5), constrained_layout=True)
    ax.barh(table["subset"], table["piece_mape"])
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute error per piece (% of PM5 power), held-out subjects")
    ax.set_title("Which IMUs carry the signal", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
