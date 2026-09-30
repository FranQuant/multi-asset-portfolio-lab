"""Unit tests for maplab.overlay.vol_overlay — synthetic data only, no real cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.contract import COST_BPS, TRADING_DAYS
from maplab.overlay import vol_overlay


def make_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = np.where(np.arange(n) % 200 < 100, 0.006, 0.02)   # calm/turbulent regimes
    base = pd.Series(rng.normal(0.0004, vol), index=idx, name="base")
    rf = pd.Series(rng.uniform(0.0, 0.0002, n), index=idx, name="rf")
    return base, rf


def test_exported_at_package_level():
    assert ml.vol_overlay is vol_overlay


def test_identity_when_target_huge():
    base, rf = make_series()
    net, c, diag = vol_overlay(base, rf, fixed_target=1e9, target="fixed")
    assert (c == 1.0).all()
    pd.testing.assert_series_equal(net, base, check_exact=True)
    assert diag["total_cost"] == 0.0
    assert (diag["turnover"] == 0.0).all()


def test_no_lookahead():
    base, rf = make_series()
    k = 400
    _, c0, _ = vol_overlay(base, rf)
    base2, rf2 = base.copy(), rf.copy()
    base2.iloc[k:] += 0.05
    rf2.iloc[k:] += 0.001
    _, c1, _ = vol_overlay(base2, rf2)
    pd.testing.assert_series_equal(c0.iloc[: k + 1], c1.iloc[: k + 1], check_exact=True)
    assert not np.allclose(c0.iloc[k + 1:], c1.iloc[k + 1:])


def test_warmup_is_one():
    base, rf = make_series()
    _, c, _ = vol_overlay(base, rf, min_history=252)
    assert (c.iloc[:252] == 1.0).all()
    assert (c.iloc[252:] < 1.0).any()


@pytest.mark.parametrize("cap", [1.0, 1.5])
def test_bounds_after_warmup(cap):
    base, rf = make_series()
    _, c, diag = vol_overlay(base, rf, cap=cap)
    post = c.iloc[252:]
    assert (post > 0).all() and (post <= cap).all()
    assert 0.0 <= diag["share_braking"] <= 1.0
    assert diag["share_braking"] == float((post < 1.0).mean())


def test_hand_computed_five_day_example():
    rf_v = 0.001
    base = pd.Series([0.011, -0.009, 0.021, -0.019, 0.011],
                     index=pd.bdate_range("2020-01-01", periods=5))
    rf = pd.Series(rf_v, index=base.index)
    T = 0.1
    net, c, diag = vol_overlay(base, rf, window=2, min_history=2, target="fixed",
                               fixed_target=T, cost_bps=10.0)
    # x = [.010, -.010, .020, -.020, .010]; 2-day std = |dx|/sqrt(2); annualised * sqrt(252)
    s = np.sqrt(126.0)
    sig = [np.nan, 0.02 * s, 0.03 * s, 0.04 * s, 0.03 * s]
    r = base.to_numpy()
    exp_c = [1.0, 1.0, T / sig[1], T / sig[2], T / sig[3]]
    np.testing.assert_allclose(diag["sigma_hat"].to_numpy(), sig, rtol=1e-12)
    np.testing.assert_allclose(c.to_numpy(), exp_c, rtol=1e-12)

    d_prev = 1.0
    for t in range(5):
        g = exp_c[t] * r[t] + (1 - exp_c[t]) * rf_v
        d = exp_c[t] * (1 + r[t]) / (1 + g)
        tau = abs(exp_c[t] - d_prev)
        cost = tau * 10.0 / 1e4
        assert np.isclose(diag["turnover"].iloc[t], tau, rtol=1e-12, atol=1e-15)
        assert np.isclose(net.iloc[t], g - cost, rtol=1e-12, atol=1e-15)
        d_prev = d
    # days 0-1: c = 1, no drift, no trade, net == base
    assert (diag["turnover"].iloc[:2] == 0.0).all()
    np.testing.assert_array_equal(net.iloc[:2].to_numpy(), r[:2])
    # day 2 by hand: c = 0.1/(0.02*sqrt(126)); tau = 1 - c (d_1 = 1)
    c2 = 0.1 / (0.02 * s)
    assert np.isclose(diag["turnover"].iloc[2], 1 - c2, rtol=1e-12)
    g2 = c2 * 0.021 + (1 - c2) * 0.001
    assert np.isclose(net.iloc[2], g2 - (1 - c2) * 10.0 / 1e4, rtol=1e-12)
    assert np.isclose(diag["total_cost"], diag["turnover"].sum() * 10.0 / 1e4, rtol=1e-12)
    assert np.isclose(diag["share_braking"], 1.0)


def test_expanding_sigma_star_values():
    base, rf = make_series()
    _, c, diag = vol_overlay(base, rf)
    sh, ss = diag["sigma_hat"], diag["sigma_star"]
    for t in (252, 300, 450, 599):
        assert np.isclose(ss.iloc[t], sh.iloc[:t].dropna().mean(), rtol=1e-12, atol=0)
        assert np.isclose(c.iloc[t], min(1.0, ss.iloc[t] / sh.iloc[t - 1]), rtol=1e-12, atol=0)


def test_inverse_variance_is_squared_ratio():
    base, rf = make_series()
    kw = dict(cap=1e9, target="fixed", fixed_target=0.1)
    _, c_vol, _ = vol_overlay(base, rf, scaling="inverse_vol", **kw)
    _, c_var, _ = vol_overlay(base, rf, scaling="inverse_variance", **kw)
    np.testing.assert_allclose(c_var.iloc[252:], c_vol.iloc[252:] ** 2, rtol=1e-12)
    assert (c_var.iloc[:252] == 1.0).all()


def test_leverage_negative_cash_leg():
    base, rf = make_series(n=300)
    net, c, diag = vol_overlay(base, rf, cap=1.5, target="fixed", fixed_target=10.0,
                               min_history=100, window=21)
    assert (c.iloc[:100] == 1.0).all()
    assert (c.iloc[100:] == 1.5).all()          # ratio >> cap -> capped at 1.5
    t = 150
    r, f = base.iloc[t], rf.iloc[t]
    g = 1.5 * r + (1 - 1.5) * f                  # cash leg is -0.5 * rf (borrowing)
    assert np.isclose(g, 1.5 * r - 0.5 * f, rtol=1e-15)
    # constant c = 1.5 after day 100: tau_t = |1.5 - d_{t-1}|
    d_prev = 1.5 * (1 + base.iloc[t - 1]) / (1 + 1.5 * base.iloc[t - 1] - 0.5 * rf.iloc[t - 1])
    tau = abs(1.5 - d_prev)
    assert np.isclose(diag["turnover"].iloc[t], tau, rtol=1e-12)
    assert np.isclose(net.iloc[t], g - tau * COST_BPS / 1e4, rtol=1e-12)
    # first levered day: tau = |1.5 - 1.0| = 0.5 exactly (d_99 = 1)
    assert np.isclose(diag["turnover"].iloc[100], 0.5, rtol=1e-12)


def test_zero_lagged_vol_gives_cap():
    idx = pd.bdate_range("2020-01-01", periods=10)
    base = pd.Series(0.001, index=idx)
    rf = pd.Series(0.0005, index=idx)          # x constant -> sigma_hat == 0
    _, c, _ = vol_overlay(base, rf, window=3, min_history=3, cap=1.5,
                          target="fixed", fixed_target=0.1)
    assert (c.iloc[:3] == 1.0).all()
    assert (c.iloc[3:] == 1.5).all()


def test_cost_identity_random_series():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2018-01-01", periods=800)
    base = pd.Series(rng.normal(0.0003, 0.012, 800), index=idx)
    rf = pd.Series(0.0001, index=idx)
    for cap in (1.0, 1.5):
        _, _, diag = vol_overlay(base, rf, cap=cap, cost_bps=25.0)
        assert np.isclose(diag["total_cost"], diag["turnover"].sum() * 25.0 / 1e4,
                          rtol=0, atol=1e-12)
        assert diag["turnover"].sum() > 0
        assert diag["params"]["cost_bps"] == 25.0 and diag["params"]["cap"] == cap


def test_validation_errors():
    base, rf = make_series(n=400)
    bad = base.copy()
    bad.iloc[10] = np.nan
    with pytest.raises(ValueError):
        vol_overlay(bad, rf)
    with pytest.raises(ValueError):                       # missing rf dates
        vol_overlay(base, rf.iloc[:-5])
    rf_nan = rf.copy()
    rf_nan.iloc[3] = np.nan
    with pytest.raises(ValueError):
        vol_overlay(base, rf_nan)
    with pytest.raises(ValueError):
        vol_overlay(base, rf, scaling="inverse_std")
    with pytest.raises(ValueError):
        vol_overlay(base, rf, target="rolling")
    with pytest.raises(ValueError):                       # fixed needs fixed_target
        vol_overlay(base, rf, target="fixed")
    with pytest.raises(ValueError):
        vol_overlay(base, rf, target="fixed", fixed_target=0.0)
    with pytest.raises(ValueError):
        vol_overlay(base, rf, window=30, min_history=20)
