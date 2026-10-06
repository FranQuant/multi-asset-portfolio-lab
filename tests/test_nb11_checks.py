"""Notebook 11 check runs on the real price cache: the 23 α-family runs and 60/40 rebuilt for notebook 11
reproduce the registered Sharpe ratios in all three windows (4338 / 3504 / 834 days)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]


@pytest.fixture(scope="module")
def h11():
    """Notebook 11's helpers, imported with notebooks/ on sys.path only for the import."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb11 as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    return h


def test_build_data_matches_registered_sharpe(h11):
    data = h11.build_data()
    XW = data["XW"]
    assert [len(Xw) for Xw in XW.values()] == [4338, 3504, 834]
    gate = h11.repro_gate(XW)
    assert len(gate) == 72 and gate["ok"].all()
