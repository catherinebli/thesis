import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from rowing_ee.config import PipelineConfig  # noqa: E402
from rowing_ee.io.raw_log import write_raw_log  # noqa: E402
from rowing_ee.sim.synthetic import make_session  # noqa: E402
from rowing_ee.subject import Subject  # noqa: E402

HAVE_CXX = shutil.which("g++") is not None and shutil.which("make") is not None


@pytest.fixture(scope="session")
def cfg():
    return PipelineConfig.load(ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="session")
def subject(cfg):
    return Subject.from_dict(cfg.subject_defaults)


@pytest.fixture(scope="session")
def faulty_log(cfg, tmp_path_factory):
    """Three bouts at different rates with I2C failures and missed ticks injected."""
    sess = make_session(cfg, bouts=((70.0, 20.0, 1.0), (60.0, 26.0, 0.8), (50.0, 18.0, 1.2)),
                        rest_s=15.0, seed=11, err_rate=0.003, gap_prob=0.001)
    p = tmp_path_factory.mktemp("logs") / "faulty.bin"
    write_raw_log(p, sess.n, sess.t_us, sess.raw, sess.err, fs_hz=cfg.fs, n_imu=cfg.n_imu,
                  feature_hash=cfg.feature_hash())
    return p, sess
