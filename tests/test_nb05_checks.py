"""Solver checks from notebook 05 §4, run on the real price cache at all 208 rebalances
under the sample and Ledoit–Wolf covariance."""
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
    """Panel and rebalance dates built exactly as notebook 05's data cell does."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb05 as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    rdates = ml.rebalance_dates(log_returns.index)
    rdates = rdates[rdates >= ml.first_eligible_rebalance(log_returns.index)]
    return {"h": h, "panel": panel, "rdates": rdates}


def test_mdp_solver_checks(nb):
    assert len(nb["rdates"]) == 208
    mech_table = nb["h"].mechanism_checks(nb["panel"], nb["rdates"])
    print(f"max identity diff = {mech_table['max_identity_diff'].max():.2e}; "
          f"min DR margin = {mech_table['min_dr_margin'].min():.4f}")
    for _, row in mech_table.iterrows():
        assert row["max_identity_diff"] <= 1e-5, row
        assert row["min_dr_margin"] >= -1e-10, row
