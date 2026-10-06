"""Arithmetic checks from notebook 07 §4, run on the real price cache at all 208 rebalances
under the sample and Ledoit–Wolf covariance and both bisections."""
import sys
from pathlib import Path

import pytest

import maplab as ml

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]


@pytest.fixture(scope="module")
def nb():
    """Panel and rebalance dates built exactly as notebook 07's data cell does."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb07 as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    rdates = ml.rebalance_dates(log_returns.index)
    rdates = rdates[rdates >= ml.first_eligible_rebalance(log_returns.index)]
    return {"h": h, "panel": panel, "rdates": rdates}


def test_hrp_arithmetic_checks(nb):
    assert len(nb["rdates"]) == 208
    H, pos_spreads, cophs, root_sides = nb["h"].mechanism_checks(nb["panel"], nb["rdates"])
    dev_cols = ["sum_dev", "split_tree", "split_pos", "diag_ivp_tree", "diag_ivp_pos",
                "scale_tree", "scale_pos", "perm_tree"]
    count_cols = ["perm_valid", "link_ok", "qd_match", "dist_ok"]
    print("; ".join(f"max {c} = {H[c].max():.2e}" for c in dev_cols)
          + f"; min w = {H['min_w'].min():.2e}; "
          + "; ".join(f"min {c} = {int(H[c].min())}" for c in count_cols))
    tol = 1e-12
    for _, row in H.iterrows():
        for c in dev_cols:
            assert row[c] <= tol, (c, row)
        assert row["min_w"] > 0, row
        for c in count_cols:
            assert row[c] == 208, (c, row)
