"""Lets the scripts run from a fresh clone without `pip install -e python/`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "python"
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))
