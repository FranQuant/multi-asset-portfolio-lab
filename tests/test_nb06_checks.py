"""Solver checks from notebook 06 §4, run on the real price cache at all 208 rebalances
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
    """Panel and rebalance dates built exactly as notebook 06's data cell does."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        import _nb06_helpers as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    rdates = ml.rebalance_dates(log_returns.index)
    rdates = rdates[rdates >= ml.first_eligible_rebalance(log_returns.index)]
    return {"h": h, "panel": panel, "rdates": rdates}


def test_erc_solver_checks(nb):
    assert len(nb["rdates"]) == 208
    mech = nb["h"].mechanism_checks(nb["panel"], nb["rdates"])
    print(f"max rc-share dev = {mech['max_rc_share_dev'].max():.2e}; "
          f"max Euler = {mech['max_euler'].max():.2e}; "
          f"max identA = {mech['max_identA'].max():.2e}; "
          f"max identB = {mech['max_identB'].max():.2e}; "
          f"max rho-bar dev = {mech['max_rho_bar_dev'].max():.2e}; "
          f"min vol margin GMV = {mech['min_vol_margin_gmv'].min():.2e}; "
          f"min vol margin EW = {mech['min_vol_margin_ew'].min():.2e}; "
          f"min w = {mech['min_w'].min():.2e}; "
          f"sweeps median {mech['median_sweeps'].to_dict()} max {mech['max_sweeps'].to_dict()}")
    for _, row in mech.iterrows():
        assert row["max_rc_share_dev"] <= 1e-11, row
        assert row["max_euler"] <= 1e-12, row
        assert row["max_identA"] <= 1e-11, row
        assert row["max_identB"] <= 1e-11, row
        assert row["max_rho_bar_dev"] <= 1e-11, row
        assert row["min_vol_margin_gmv"] >= -1e-11, row
        assert row["min_vol_margin_ew"] >= -1e-11, row
        assert row["min_w"] > 0, row
