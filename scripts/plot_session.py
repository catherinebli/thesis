#!/usr/bin/env python3
"""Three-panel session plot: detector signal with peaks, power per stroke, cumulative kcal.

    python scripts/plot_session.py data/raw/raw_000.bin --subject S01
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np


def plot_session(res, path: Path | str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = res.fs
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    if res.detect_signal is not None:
        t = (res.detect_n - res.n_first) / fs
        ax[0].plot(t, res.detect_signal, lw=0.8, color="#185FA5")
        pk = np.asarray(res.peaks_n)
        if pk.size:
            i = pk - int(res.detect_n[0])
            ok = (i >= 0) & (i < len(res.detect_signal))
            ax[0].plot((pk[ok] - res.n_first) / fs, res.detect_signal[i[ok]], "o", ms=3, color="#D85A30")
    ax[0].set_ylabel("shank gz (rad/s)")
    ax[0].set_title("Detector signal and confirmed drive peaks")

    tc = np.array([(s.n1 - res.n_first) / fs for s in res.strokes])
    pw = np.array([s.P_w for s in res.strokes])
    val = np.array([s.valid for s in res.strokes], dtype=bool)
    if tc.size:
        # draw each stroke as a flat segment over its own interval; pauses stay blank
        for s in res.strokes:
            ax[1].plot([(s.n0 - res.n_first) / fs, (s.n1 - res.n_first) / fs], [s.P_w, s.P_w],
                       color="#534AB7", lw=1.4)
        ax[1].plot(tc[~val], pw[~val], "x", color="#A32D2D", label="held (invalid stroke)")
    ax[1].axhline(res.p_rest_w, color="gray", ls="--", lw=0.8, label="resting floor")
    ax[1].set_ylabel("metabolic power (W)")
    ax[1].legend(loc="upper right", frameon=False)

    kc = np.array([s.kcal_total for s in res.strokes])
    if tc.size:
        ax[2].plot(np.r_[0, tc, res.duration_s], np.r_[0, kc, res.kcal_total], color="#0F6E56")
    ax[2].set_ylabel("cumulative kcal")
    ax[2].set_xlabel("time (s)")
    if res.placeholder_model:
        fig.suptitle("Placeholder weights: power is the resting floor until a model is trained", color="#854F0B")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.argv.append("--plot")
    from run_session import main
    raise SystemExit(main())
