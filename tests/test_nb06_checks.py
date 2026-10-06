"""Solver checks from notebook 06 §4, run on the real price cache at all 208 rebalances
under the sample and Ledoit–Wolf covariance."""
import sys
from pathlib import Path

import pandas as pd
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
        from helpers import nb06 as h
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


def test_nb06_beta_decomposition_reproduces_the_difference(nb):
    """The §7 Δα + Δβ·E[mkt] decomposition reproduces each pair's realised full-window difference."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import common as cm
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    panel = nb["panel"]
    prices = ml.load_prices()
    simple_returns = ml.to_simple_returns(prices)
    rf_daily = ml.load_rf_returns()["BIL"]
    split_ts = pd.Timestamp(ml.TRAIN_TEST_SPLIT)

    strategies = {
        "ERC(S)": ml.EqualRiskContribution(cov_estimator=ml.sample_cov),
        "GMV(S)": ml.GMV(cov_estimator=ml.sample_cov),
        "EW": ml.EqualWeight(),
        "60/40": ml.FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
        "MDP(S)": ml.MostDiversified(cov_estimator=ml.sample_cov),
        "ERC(LW)": ml.EqualRiskContribution(cov_estimator=ml.ledoit_wolf_cov),
        "IV": ml.InverseVol(),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = ml.backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    summary_table = cm.summary_by_window(results, split_ts, rf_daily, cost_drag=True)
    pairs = [
        ("ERC(S)", "GMV(S)"),
        ("ERC(S)", "MDP(S)"),
        ("ERC(S)", "EW"),
        ("ERC(LW)", "ERC(S)"),
        ("ERC(S)", "IV"),
    ]

    paired_table, paired_series = ml.inference.paired_table(results, pairs, split_ts, summary_table)
    MKT_simple = (simple_returns["SPY"] - rf_daily).dropna()

    capm_table = ml.inference.capm_table(results, ["ERC(S)", "GMV(S)", "MDP(S)", "EW", "60/40", "ERC(LW)", "IV"], MKT_simple, rf_daily)
    diff_capm_table = ml.inference.diff_regression(paired_series, pairs, MKT_simple)
    decomp = ml.inference.beta_decomposition(capm_table, paired_table, diff_capm_table, pairs,
                                             results, rf_daily, MKT_simple)
    assert len(decomp) == len(pairs)
    assert all(abs(r["identity_check"]) < 1e-9 for r in decomp.values())
