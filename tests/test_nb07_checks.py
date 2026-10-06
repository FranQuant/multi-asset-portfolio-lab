"""Arithmetic checks from notebook 07 §4, run on the real price cache at all 208 rebalances
under the sample and Ledoit–Wolf covariance and both bisections."""
import sys
from pathlib import Path

import numpy as np
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


def test_nb07_beta_decomposition_reproduces_the_difference(nb):
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
        "HRP(S)": ml.HierarchicalRiskParity(cov_estimator=ml.sample_cov),
        "GMV(S)": ml.GMV(cov_estimator=ml.sample_cov),
        "EW": ml.EqualWeight(),
        "60/40": ml.FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
        "MDP(S)": ml.MostDiversified(cov_estimator=ml.sample_cov),
        "ERC(S)": ml.EqualRiskContribution(cov_estimator=ml.sample_cov),
        "HRP(LW)": ml.HierarchicalRiskParity(cov_estimator=ml.ledoit_wolf_cov),
        "HRP[pos](S)": ml.HierarchicalRiskParity(bisection="positional"),
        "HRP[ward](S)": ml.HierarchicalRiskParity(linkage="ward"),
        "IVP": ml.InverseVariance(),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = ml.backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    summary_table = cm.summary_by_window(results, split_ts, rf_daily, cost_drag=True)
    pairs = [
        ("HRP(S)", "GMV(S)"),
        ("HRP(S)", "MDP(S)"),
        ("HRP(S)", "ERC(S)"),
        ("HRP(S)", "EW"),
        ("HRP(LW)", "HRP(S)"),
        ("HRP(S)", "IVP"),
        ("HRP[pos](S)", "HRP(S)"),
        ("HRP[ward](S)", "HRP(S)"),
    ]

    paired_table, paired_series = ml.inference.paired_table(results, pairs, split_ts, summary_table)
    MKT_simple = (simple_returns["SPY"] - rf_daily).dropna()

    capm_table = ml.inference.capm_table(results, list(strategies), MKT_simple, rf_daily)
    diff_capm_table = ml.inference.diff_regression(paired_series, pairs, MKT_simple)
    decomp = ml.inference.beta_decomposition(capm_table, paired_table, diff_capm_table, pairs,
                                             results, rf_daily, MKT_simple)
    assert len(decomp) == len(pairs)
    assert all(abs(r["identity_check"]) < 1e-9 for r in decomp.values())


def test_nb07_hrp_weight_is_product_of_tree_path_shares(nb):
    """UUP's HRP weight equals the product of its shares along the tree path, at the three §4 dates."""
    h, panel = nb["h"], nb["panel"]
    DATES = [pd.Timestamp(d) for d in ["2014-12-31", "2017-12-31", "2022-12-31"]]
    snap = h.snapshot(panel, DATES)
    for asof in DATES:
        s = snap[asof]
        Sigma_np, iU = s["Sigma_np"], s["iU"]
        tp = h.tree_path(Sigma_np, s["info"]["Z"], iU)
        assert abs(np.prod([a for _, _, a in tp]) - s["W"]["HRP"][iU]) <= 1e-12, asof
