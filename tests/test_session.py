"""Energy accounting, log I/O, end-to-end pipeline, and C++ parity."""
import subprocess

import numpy as np
import pytest

from conftest import HAVE_CXX, ROOT
from rowing_ee.energy import ROWING, EnergyMeter
from rowing_ee.io.raw_log import ERR_GAP, RawLog, unwrap_u32, write_raw_log
from rowing_ee.model.export import placeholder_model
from rowing_ee.model.linear_model import LinearModel
from rowing_ee.pipeline import Pipeline, read_est_csv, run_streaming
from rowing_ee.subject import Subject, parse_subject_cfg


def random_model(cfg, seed=7):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, cfg.D) * 0.3
    w[0] = 380.0
    return LinearModel(w)


# ------------------------------------------------------------------ energy
def test_energy_tiles_time_exactly():
    e = EnergyMeter(fs=100, rest_timeout_n=800, p_rest=100.0)
    e.begin(0)
    e.start_bout(500)                       # 5 s of rest before the first catch
    e.add_stroke(500, 800, 400.0, True)     # 3 s at 400 W
    e.add_stroke(800, 1100, 0.0, False)     # invalid: holds 400 W
    assert e.mode == ROWING
    for n in range(1100, 2000):
        e.tick(n)
    kcal = e.end(2000)                      # 9 s of rest after the timeout
    expect_J = 100 * 5 + 400 * 3 + 400 * 3 + 100 * 9
    assert abs(kcal * 4184 - expect_J) < 1e-9


def test_mifflin_st_jeor():
    s = Subject("x", 70.0, 1.80, 22, "M")
    kcal_day = 10 * 70 + 625 * 1.80 - 5 * 22 + 5
    assert abs(s.bmr_watts() - kcal_day * 4184 / 86400) < 1e-12
    assert Subject("x", 70.0, 1.80, 22, "F").bmr_watts() < s.bmr_watts()


def test_subject_cfg_parsing():
    base = Subject("S00", 70, 1.8, 22, "M")
    s = parse_subject_cfg("id=S07\nmass_kg = 64.5  # lightweight\nheight_m=1.75\nsex=f\nbogus\n", base)
    assert (s.id, s.mass_kg, s.height_m, s.age, s.sex) == ("S07", 64.5, 1.75, 22, "F")


# ------------------------------------------------------------------ log io
def test_unwrap_micros():
    t = np.array([4294967000, 4294967200, 100, 300], dtype=np.uint32)
    assert list(np.diff(unwrap_u32(t))) == [200, 196, 200]   # 2^32 - 4294967200 + 100 = 196


def test_log_roundtrip_and_truncated_tail(cfg, tmp_path):
    n = np.arange(100)
    raw = np.arange(100 * cfg.n_ch, dtype=np.int64).reshape(100, cfg.n_ch).astype(np.int16)
    err = np.zeros(100, dtype=np.uint8)
    err[5] = ERR_GAP
    p = tmp_path / "x.bin"
    write_raw_log(p, n, n * 10000, raw, err, fs_hz=100, n_imu=cfg.n_imu, feature_hash=123)
    with open(p, "ab") as f:                 # preallocated garbage after an unclean stop
        f.write(b"\0" * 69 * 10)
    log = RawLog.read(p)
    assert len(log) == 100 and log.dropped_tail == 10
    assert np.array_equal(log.raw, raw) and log.header["feature_hash"] == 123
    assert log.timing_report()["gap_filled"] == 1


# ---------------------------------------------------------------- pipeline
def test_strokes_detected_with_faults(cfg, subject, faulty_log):
    path, sess = faulty_log
    res = Pipeline.run_session(RawLog.read(path), cfg, random_model(cfg), subject)
    assert abs(len(res.strokes) - sess.expected_strokes) <= 3       # +-1 per bout edge
    assert sum(not s.valid for s in res.strokes) <= 3
    assert res.calibrated_n == cfg.still_n - 1


def test_batch_equals_streaming(cfg, subject, faulty_log):
    log = RawLog.read(faulty_log[0])
    m = random_model(cfg)
    a = Pipeline.run_session(log, cfg, m, subject)
    b = run_streaming(log, cfg, m, subject)
    assert [(s.n0, s.n1, s.valid) for s in a.strokes] == [(s.n0, s.n1, s.valid) for s in b.strokes]
    np.testing.assert_allclose([s.P_w for s in a.strokes], [s.P_w for s in b.strokes], rtol=1e-12)
    assert abs(a.kcal_total - b.kcal_total) < 1e-9


def test_placeholder_bills_rest_for_whole_session(cfg, subject, faulty_log):
    log = RawLog.read(faulty_log[0])
    res = Pipeline.run_session(log, cfg, placeholder_model(cfg), subject)
    expect = subject.rest_watts(cfg.k_rest) * len(log) / cfg.fs / 4184
    assert abs(res.kcal_total - expect) < 1e-9
    assert all(s.P_w == pytest.approx(res.p_rest_w) for s in res.strokes)


def test_fold_normalization_matches_zscored_model():
    rng = np.random.default_rng(2)
    X = rng.normal(5, 2, (50, 6))
    mu, sd = X.mean(0), X.std(0)
    w, b = rng.normal(size=6), 3.0
    y_z = b + ((X - mu) / sd) @ w
    wf = LinearModel.fold_normalization(w, b, mu, sd)
    y_raw = np.c_[np.ones(50), X] @ wf
    np.testing.assert_allclose(y_raw, y_z, rtol=1e-12)


# ------------------------------------------------------------- C++ parity
@pytest.mark.skipif(not HAVE_CXX, reason="needs g++ and make")
def test_cpp_replay_matches_python(cfg, subject, faulty_log, tmp_path):
    native = ROOT / "firmware" / "test" / "native"
    subprocess.run(["make", "-s", "-C", str(native), "replay"], check=True)
    m = random_model(cfg)
    wcsv = tmp_path / "w.csv"
    m.save_csv(wcsv)
    scfg = tmp_path / "subject.cfg"
    scfg.write_text(f"id={subject.id}\nmass_kg={subject.mass_kg!r}\nheight_m={subject.height_m!r}\n"
                    f"age={subject.age!r}\nsex={subject.sex}\n")
    out = tmp_path / "est.csv"
    subprocess.run([str(native / "replay"), str(faulty_log[0]), str(out), "--weights", str(wcsv),
                    "--subject", str(scfg)], check=True, capture_output=True)
    cpp, kcal_cpp = read_est_csv(out)
    py = Pipeline.run_session(RawLog.read(faulty_log[0]), cfg, m, subject)
    assert [(s.n0, s.n1, s.valid) for s in cpp] == [(s.n0, s.n1, s.valid) for s in py.strokes]
    np.testing.assert_allclose([s.P_w for s in cpp], [s.P_w for s in py.strokes], rtol=1e-9)
    assert abs(kcal_cpp - py.kcal_total) / py.kcal_total < 1e-9


@pytest.mark.skipif(not HAVE_CXX, reason="needs g++ and make")
def test_hal_and_main_compile_against_stubs():
    subprocess.run(["make", "-s", "-C", str(ROOT / "firmware" / "test" / "native"), "hal-check"], check=True)


@pytest.mark.skipif(not HAVE_CXX, reason="needs g++")
def test_generated_headers_match_config(cfg, tmp_path):
    from rowing_ee.model.export import write_config_header, write_weights_header
    write_config_header(cfg, tmp_path / "PipelineConfig.h")
    write_weights_header(placeholder_model(cfg), cfg, tmp_path / "ModelWeights.h")
    src = tmp_path / "t.cpp"
    src.write_text(
        '#include "PipelineConfig.h"\n#include "ModelWeights.h"\n'
        f"static_assert(PC_D == {cfg.D}, \"D\");\n"
        f"static_assert(PC_DETECT_CH == {cfg.detect_channel}, \"ch\");\n"
        "static_assert(MW_FEATURE_HASH == PC_FEATURE_HASH, \"hash\");\n"
        "int main() { return (int)(PC_SOS[0][0] * 0) + MW_IS_PLACEHOLDER - 1; }\n")
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Werror", "-I", str(tmp_path), str(src),
                    "-o", str(tmp_path / "t")], check=True)
    assert subprocess.run([str(tmp_path / "t")]).returncode == 0
    # the committed headers must be in sync with config/pipeline.yaml
    committed = (ROOT / "firmware" / "include" / "PipelineConfig.h").read_text()
    assert committed == (tmp_path / "PipelineConfig.h").read_text(), "run scripts/gen_headers.py"
