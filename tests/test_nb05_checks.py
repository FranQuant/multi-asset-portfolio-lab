"""Solver checks from notebook 05 §4, run on the real price cache at all 208 rebalances
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


def test_nb05_beta_decomposition_reproduces_the_difference(nb):
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
        "GMV(S)": ml.GMV(cov_estimator=ml.sample_cov),
        "MaxSharpe(S)": ml.MaxSharpe(cov_estimator=ml.sample_cov),
        "EW": ml.EqualWeight(),
        "60/40": ml.FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
        "MDP(S)": ml.MostDiversified(cov_estimator=ml.sample_cov),
        "MDP(LW)": ml.MostDiversified(cov_estimator=ml.ledoit_wolf_cov),
        "IV": ml.InverseVol(),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = ml.backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    summary_table = cm.summary_by_window(results, split_ts, rf_daily, cost_drag=True)
    pairs = [
        ("MDP(S)", "GMV(S)"),
        ("MDP(S)", "EW"),
        ("MDP(LW)", "MDP(S)"),
        ("MDP(S)", "IV"),
    ]

    paired_table, paired_series = ml.inference.paired_table(results, pairs, split_ts, summary_table)
    MKT_simple = (simple_returns["SPY"] - rf_daily).dropna()

    capm_table = ml.inference.capm_table(results, ["MDP(S)", "GMV(S)", "EW", "60/40", "MDP(LW)", "IV"], MKT_simple, rf_daily)
    diff_capm_table = ml.inference.diff_regression(paired_series, pairs, MKT_simple)
    decomp = ml.inference.beta_decomposition(capm_table, paired_table, diff_capm_table, pairs,
                                             results, rf_daily, MKT_simple)
    assert len(decomp) == len(pairs)
    assert all(abs(r["identity_check"]) < 1e-9 for r in decomp.values())
