"""Unit tests for config, calibration, filtering, detection and features."""
import math

import numpy as np
import pytest
from scipy import signal

from rowing_ee.config import PipelineConfig
from rowing_ee.dsp.calibration import Calibration
from rowing_ee.dsp.detector import FIRST, NONE, STROKE, StrokeDetector, median
from rowing_ee.dsp.features import Featurizer
from rowing_ee.dsp.filters import FilterBank


def test_config_derived(cfg):
    assert cfg.n_ch == 30 and cfg.D == 4 + 30 * cfg.bins == 904
    assert cfg.channel_names[cfg.detect_channel] == "shank_gz"
    assert len(cfg.feature_names()) == cfg.D
    assert cfg.feature_hash() == cfg.feature_hash()


def test_mirror_mount_rejected(cfg):
    raw = dict(cfg.raw)
    raw["mount"] = dict(raw["mount"], shank=[[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    with pytest.raises(ValueError, match="mirror"):
        PipelineConfig(raw)


def test_feature_hash_ignores_wiring(cfg):
    raw = dict(cfg.raw)
    raw["imus"] = [dict(d, bus=2, mux=None) for d in cfg.raw["imus"]]
    assert PipelineConfig(raw).feature_hash() == cfg.feature_hash()
    raw2 = dict(cfg.raw, features={"bins": 15, "max_err_frac": 0.05})
    assert PipelineConfig(raw2).feature_hash() != cfg.feature_hash()


def test_calibration_bias_and_units(cfg):
    cal = Calibration(cfg)
    rng = np.random.default_rng(0)
    bias_counts = rng.integers(-20, 20, cfg.n_imu * 3)
    frames = np.zeros((cfg.still_n, cfg.n_ch), dtype=np.int16)
    frames[:, cal.gyr_idx] = bias_counts
    frames[:, cal.acc_idx] = int(round(9.80665 / cfg.accel_scale))
    assert cal.fit_bias(frames) == cfg.still_n - 1
    np.testing.assert_allclose(cal.bias, bias_counts * cfg.gyro_scale)
    out = cal.apply(frames[0])
    np.testing.assert_allclose(out[cal.gyr_idx], 0.0, atol=1e-15)
    np.testing.assert_allclose(out[cal.acc_idx], 9.80665, atol=cfg.accel_scale)


def test_calibration_retries_when_moving(cfg):
    cal = Calibration(cfg)
    moving = np.zeros((cfg.still_n, cfg.n_ch), dtype=np.int16)
    moving[:, cal.gyr_idx[0]] = (np.arange(cfg.still_n) % 2) * 2000   # ~2 rad/s jitter
    still = np.zeros((cfg.still_n, cfg.n_ch), dtype=np.int16)
    assert cal.fit_bias(np.vstack([moving, still])) == 2 * cfg.still_n - 1
    assert cal.windows_tried == 2


def test_filter_step_matches_block_and_scipy(cfg):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, cfg.n_ch)) + 9.8
    a, b = FilterBank(cfg.sos(), cfg.n_ch), FilterBank(cfg.sos(), cfg.n_ch)
    a.reset(X[0]); b.reset(X[0])
    Y_step = np.array([a.step(x) for x in X])
    Y_block = b.run(X)
    np.testing.assert_allclose(Y_step, Y_block, rtol=0, atol=1e-12)
    # our steady-state start equals scipy's sosfilt_zi scaled by the first sample
    zi = signal.sosfilt_zi(cfg.sos())[:, :, None] * X[0][None, None, :]
    c = FilterBank(cfg.sos(), cfg.n_ch)
    c.reset(X[0])
    np.testing.assert_allclose(c.z, zi, atol=1e-12)


def test_filter_constant_input_has_no_transient(cfg):
    f = FilterBank(cfg.sos(), cfg.n_ch)
    x = np.full(cfg.n_ch, 9.80665)
    f.reset(x)
    for _ in range(50):
        np.testing.assert_allclose(f.step(x), x, rtol=1e-12)


def test_filter_cutoff_is_minus_3db_single_pass(cfg):
    _, h = signal.sosfreqz(cfg.sos(), worN=[cfg.cutoff_hz], fs=cfg.fs)
    assert abs(abs(h[0]) ** 2 - 0.5) < 1e-6


def test_median_even_and_odd():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_detector_pulse_train(cfg):
    det = StrokeDetector(cfg)
    fs, T = cfg.fs, 3.0
    t = np.arange(int(30 * fs)) / fs
    s = np.where(np.mod(t, T) < 1.0, 3.0 * np.sin(np.pi * np.mod(t, T) / 1.0) ** 2, -0.3)
    events = [(n, *det.step(v, n)) for n, v in enumerate(s)]
    firsts = [e for e in events if e[1] == FIRST]
    strokes = [e for e in events if e[1] == STROKE]
    assert len(firsts) == 1
    assert len(strokes) == 9
    lengths = [e[3] - e[2] for e in strokes]
    assert all(abs(L - T * fs) <= 1 for L in lengths)


def test_detector_ignores_double_bump(cfg):
    det = StrokeDetector(cfg)
    s = np.zeros(1000)
    for c in (100, 160, 400):            # 160 is only 0.6 s after 100: a second bump
        s[c - 5:c + 6] = 3.0 - np.abs(np.arange(-5, 6)) * 0.2
    ev = [det.step(v, n) for n, v in enumerate(s)]
    kinds = [e[0] for e in ev if e[0] != NONE]
    assert kinds == [FIRST, STROKE]
    stroke = [e for e in ev if e[0] == STROKE][0]
    assert (stroke[1], stroke[2]) == (100, 400)


def test_features_bin_means(cfg, subject):
    feat = Featurizer(cfg)
    L = 300
    seg = np.tile(np.arange(L, dtype=np.float32)[:, None], (1, cfg.n_ch))
    x = feat.extract(seg, L / cfg.fs, subject)
    assert x.shape == (cfg.D,)
    assert tuple(x[:4]) == (1.0, subject.mass_kg, subject.height_m, 3.0)
    edges = (np.arange(cfg.bins + 1) * L) // cfg.bins
    expect = [(edges[k] + edges[k + 1] - 1) / 2 for k in range(cfg.bins)]
    np.testing.assert_allclose(x[4:4 + cfg.bins], expect)
    assert math.isclose(x[4 + cfg.bins * 7 + 3], expect[3])     # channel-major layout
