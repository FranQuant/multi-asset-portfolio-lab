"""Design checks from notebook 04 §5, run on the real price cache at 2014-12-31,
2017-12-31 and 2022-12-31."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf, empirical_covariance

import maplab as ml
from maplab import BlackLitterman, EqualWeight, GMV, MaxSharpe, backtest

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
        from helpers import nb04 as h
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


def test_ledoit_wolf_eigenvalues_are_shrunk_sample_eigenvalues(nb):
    """At 2022-12-31 the Ledoit–Wolf eigenvalues are (1 − δ)·λ(S) + δ·m, with m = tr(S)/N."""
    asof_split = pd.Timestamp("2022-12-31")
    rets_split = nb["panel"].slice(asof_split, "returns", ml.COV_LOOKBACK)[ml.UNIVERSE]
    S = empirical_covariance(rets_split.to_numpy())
    m = np.trace(S) / len(ml.UNIVERSE)
    lw_fit = LedoitWolf().fit(rets_split.to_numpy())
    delta_lw_split = lw_fit.shrinkage_
    lhs_lw = ml.ledoit_wolf_cov(rets_split).to_numpy() / ml.TRADING_DAYS

    lam_S = np.linalg.eigvalsh(S)
    lam_LW = np.linalg.eigvalsh(lhs_lw)
    eig_gap = np.max(np.abs(lam_LW - ((1 - delta_lw_split) * lam_S + delta_lw_split * m)))
    assert eig_gap <= 1e-12


def test_ledoit_wolf_beta_decomposition_reproduces_the_difference(nb):
    """The §6 Δα + Δβ·E[mkt] split reproduces each registered LW − S pair's realised full-window difference."""
    panel = nb["panel"]
    simple_returns = ml.to_simple_returns(ml.load_prices())
    rf_daily = ml.load_rf_returns()["BIL"]

    strategies = {
        "GMV(S)": GMV(cov_estimator=ml.sample_cov),
        "GMV(LW)": GMV(cov_estimator=ml.ledoit_wolf_cov),
        "MS(S)": MaxSharpe(cov_estimator=ml.sample_cov),
        "MS(LW)": MaxSharpe(cov_estimator=ml.ledoit_wolf_cov),
        "BL(S)": BlackLitterman(cov_estimator=ml.sample_cov, k=0.1),
        "BL(LW)": BlackLitterman(cov_estimator=ml.ledoit_wolf_cov, k=0.1),
    }
    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    MKT_simple = (simple_returns["SPY"] - rf_daily).dropna()
    capm_table = ml.inference.capm_table(results, ["GMV(S)", "GMV(LW)", "MS(S)", "MS(LW)", "BL(S)", "BL(LW)"],
                                         MKT_simple, rf_daily)
    for est_name, base_name, label in [("GMV(LW)", "GMV(S)", "GMV"), ("MS(LW)", "MS(S)", "MaxSharpe"), ("BL(LW)", "BL(S)", "BL(0.1)")]:
        alpha_base = float(capm_table.loc[base_name, "alpha_ann"])
        beta_base = float(capm_table.loc[base_name, "beta"])
        alpha_est = float(capm_table.loc[est_name, "alpha_ann"])
        beta_est = float(capm_table.loc[est_name, "beta"])

        net_est = results[est_name]["net"]
        net_base = results[base_name]["net"]
        idx = net_est.index.intersection(net_base.index)
        diff = net_est.loc[idx] - net_base.loc[idx]
        realized_diff_full = float(diff.mean() * ml.TRADING_DAYS)

        idx = net_est.index.intersection(rf_daily.index).intersection(MKT_simple.index)
        mkt_ann_mean = float(MKT_simple.loc[idx].mean() * ml.TRADING_DAYS)

        d_alpha = alpha_est - alpha_base
        d_beta = beta_est - beta_base
        beta_part = d_beta * mkt_ann_mean
        assert abs(realized_diff_full - (d_alpha + beta_part)) < 1e-9, label
