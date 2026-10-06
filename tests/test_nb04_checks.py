"""Design checks from notebook 04 §5, run on the real price cache at 2014-12-31,
2017-12-31 and 2022-12-31."""
import sys
from pathlib import Path

import pandas as pd
import pytest

import maplab as ml
from maplab import BlackLitterman

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]

DATES = [pd.Timestamp(d) for d in ["2014-12-31", "2017-12-31", "2022-12-31"]]


@pytest.fixture(scope="module")
def nb():
    """Panel built exactly as notebook 04's data cell does."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        import _nb04_helpers as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    return {"h": h, "panel": panel}


def test_bl_k0_is_ew_under_every_estimator(nb):
    devs = {}
    for asof in DATES:
        for name, est in [("sample", ml.sample_cov), ("ledoit_wolf", ml.ledoit_wolf_cov), ("oas", ml.oas_cov)]:
            w = BlackLitterman(cov_estimator=est, k=0.0)(nb["panel"], asof)
            devs[(asof.date(), name)] = float((w - 1.0 / 13).abs().max())
    print(f"max |w - 1/13| = {max(devs.values()):.2e}")
    assert max(devs.values()) <= 1e-5, devs


def test_ito_term_below_preset_rule(nb):
    t = nb["h"].ito_table(nb["panel"], DATES)
    print(f"max |dmu| = {t['max_abs_dmu_bp'].max():.2f} bp/yr; max halfL1 = {t['halfL1_dw_%'].max():.2f}%")
    assert (t["max_abs_dmu_bp"] <= 10).all(), t
    assert (t["halfL1_dw_%"] <= 1.0).all(), t
