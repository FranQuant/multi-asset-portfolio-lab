"""Unit tests for maplab.models — synthetic data only, no real cache."""
from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.contract import LONG_ONLY, LONG_SHORT, COV_LOOKBACK, TRADING_DAYS
from maplab.data import Panel
from maplab.models import (
    Strategy,
    GMV,
    MaxSharpe,
    BetaTargetMinVar,
    BlackLitterman,
    EqualWeight,
    gmv_closed_form,
    tangency_closed_form,
    _min_variance_long_only,
)


UNIVERSE = ml.UNIVERSE
N_ASSETS = len(UNIVERSE)


def make_log_returns(n_days=300, mean=0.0005, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(loc=mean, scale=vol, size=(n_days, N_ASSETS))
    return pd.DataFrame(data, index=idx, columns=UNIVERSE)


def make_panel(**kwargs):
    log_returns = make_log_returns(**kwargs)
    return Panel({"returns": log_returns}), log_returns


def make_panel_with_rf(n_days=300, mean=0.0005, vol=0.01, rf_mean=0.00002, rf_vol=0.0001, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    log_returns = pd.DataFrame(
        rng.normal(loc=mean, scale=vol, size=(n_days, N_ASSETS)), index=idx, columns=UNIVERSE,
    )
    rf = pd.DataFrame(
        rng.normal(loc=rf_mean, scale=rf_vol, size=n_days), index=idx, columns=["BIL"],
    )
    panel = Panel({"returns": log_returns, "rf": rf})
    return panel, log_returns, rf


# ── a. GMV validity + matches closed form on a diagonal-dominant Sigma ──────

def test_gmv_weights_valid():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]
    w = GMV()(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()


def test_gmv_matches_closed_form_on_diagonal_dominant_sigma():
    rng = np.random.default_rng(1)
    diag_vals = rng.uniform(0.02, 0.05, size=N_ASSETS)
    Sigma = pd.DataFrame(np.diag(diag_vals), index=UNIVERSE, columns=UNIVERSE)

    closed = gmv_closed_form(Sigma)
    assert (closed > 0).all()  # confirms Sigma's unconstrained GMV is all-positive

    numeric = pd.Series(_min_variance_long_only(Sigma), index=UNIVERSE)
    # SLSQP's stopping criterion here is the relative objective-value
    # decrease (ftol=1e-12), not gradient noise — the analytic jacobian
    # barely moves this number (~4.51e-6 either way) — so achievable
    # precision on x is ~5e-6, not 1e-6.
    assert np.allclose(numeric.to_numpy(), closed.to_numpy(), atol=1e-5, rtol=0)


# ── b. tangency_closed_form: hand-computable 2-asset case + denom<=0 guard ──

def test_tangency_closed_form_two_asset_analytic():
    mu = pd.Series([0.10, 0.06], index=["A", "B"])
    Sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    rf = 0.02

    w = tangency_closed_form(mu, Sigma, rf)
    # Sigma^-1 (mu - rf) = [0.08/0.04, 0.04/0.09] = [2, 4/9]; normalize by the sum.
    expected = pd.Series([9 / 11, 2 / 11], index=["A", "B"])
    pd.testing.assert_series_equal(w, expected, atol=1e-10, check_names=False)


def test_tangency_closed_form_raises_on_nonpositive_denominator():
    mu = pd.Series([0.01, 0.01], index=["A", "B"])
    Sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    rf = 0.05  # both excess returns negative -> denominator <= 0
    with pytest.raises(ValueError):
        tangency_closed_form(mu, Sigma, rf)


# ── c. MaxSharpe ex-ante Sharpe dominates GMV and EW on the same (mu, Sigma) ─

def test_maxsharpe_exante_sharpe_dominates_gmv_and_ew():
    panel, log_returns = make_panel(seed=2)
    asof = log_returns.index[COV_LOOKBACK]

    gmv = GMV()
    msr = MaxSharpe(rf=0.0)
    ew = EqualWeight()

    mu, Sigma = gmv._estimate(panel, asof)
    rf_ann = msr._rf_ann(panel, asof)
    assert (mu - rf_ann).max() > 0  # sanity: non-degenerate for this seed

    def exante_sharpe(w):
        w = w.reindex(UNIVERSE)
        Sigma_np = Sigma.to_numpy()
        ret = float(w.to_numpy() @ mu.to_numpy())
        vol = float(np.sqrt(w.to_numpy() @ Sigma_np @ w.to_numpy()))
        return (ret - rf_ann) / vol

    s_gmv = exante_sharpe(gmv(panel, asof))
    s_msr = exante_sharpe(msr(panel, asof))
    s_ew = exante_sharpe(ew(panel, asof))

    assert s_msr >= s_gmv - 1e-8
    assert s_msr >= s_ew - 1e-8


# ── d. Degenerate case: all-negative expected returns -> GMV fallback ───────

def test_maxsharpe_degenerate_case_falls_back_to_gmv(caplog):
    panel, log_returns = make_panel(mean=-0.05, vol=0.01, seed=3)
    asof = log_returns.index[COV_LOOKBACK]

    gmv = GMV()
    msr = MaxSharpe(rf=0.0)

    with caplog.at_level(logging.WARNING, logger="maplab.models"):
        w_msr = msr(panel, asof)
    w_gmv = gmv(panel, asof)

    pd.testing.assert_series_equal(w_msr, w_gmv, atol=1e-8, check_names=False)
    assert msr.fallback_dates == [asof]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ── e. Retry path on a transient SLSQP failure; RuntimeError if it persists ──

def test_maxsharpe_retries_once_on_solver_failure(monkeypatch):
    import scipy.optimize as opt
    from maplab import models as models_mod

    panel, log_returns = make_panel(seed=4)
    asof = log_returns.index[COV_LOOKBACK]
    msr = MaxSharpe(rf=0.0)

    real_minimize = opt.minimize
    calls = {"n": 0}

    class FakeResult:
        success = False
        message = "synthetic linesearch failure"
        x = None
        fun = None

    def flaky_minimize(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResult()
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(models_mod.opt, "minimize", flaky_minimize)

    w = msr(panel, asof)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()
    assert msr.retry_dates == [asof]
    assert calls["n"] == 2


def test_maxsharpe_raises_if_retry_also_fails(monkeypatch):
    from maplab import models as models_mod

    panel, log_returns = make_panel(seed=5)
    asof = log_returns.index[COV_LOOKBACK]
    msr = MaxSharpe(rf=0.0)

    class FakeResult:
        success = False
        message = "synthetic persistent failure"
        x = None
        fun = None

    def always_fails(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(models_mod.opt, "minimize", always_fails)

    with pytest.raises(RuntimeError):
        msr(panel, asof)
    assert msr.retry_dates == [asof]


# ── e2. MaxSharpe rf="panel": trailing mean×252 off the panel's "rf" frame ──

def test_maxsharpe_rf_panel_uses_trailing_mean():
    _, log_returns = make_panel(seed=8)
    rng = np.random.default_rng(9)
    rf_series = pd.DataFrame(
        rng.normal(loc=0.00005, scale=0.0001, size=len(log_returns)),
        index=log_returns.index, columns=["BIL"],
    )
    panel = Panel({"returns": log_returns, "rf": rf_series})
    asof = log_returns.index[COV_LOOKBACK]

    msr_panel = MaxSharpe(rf="panel")
    rf_ann = msr_panel._rf_ann(panel, asof)

    expected_rf_ann = float(
        rf_series.loc[rf_series.index < asof].iloc[-COV_LOOKBACK:].mean().iloc[0] * TRADING_DAYS
    )
    assert np.isclose(rf_ann, expected_rf_ann)

    w_panel = msr_panel(panel, asof)
    w_float = MaxSharpe(rf=expected_rf_ann)(panel, asof)
    pd.testing.assert_series_equal(w_panel, w_float, atol=1e-8, check_names=False)


def test_maxsharpe_rf_panel_without_rf_frame_raises():
    panel, log_returns = make_panel(seed=10)
    asof = log_returns.index[COV_LOOKBACK]
    msr = MaxSharpe(rf="panel")
    with pytest.raises(ValueError):
        msr(panel, asof)


# ── e3. ann_sharpe / summary: rf required; constant daily Series == annual float

def test_ann_sharpe_constant_rf_series_matches_annual_float():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0006, 0.01, size=300), index=idx)
    rf_daily = 0.00008
    rf_series = pd.Series(rf_daily, index=idx)
    rf_annual = rf_daily * TRADING_DAYS

    s_series = ml.ann_sharpe(r, rf_series)
    s_float = ml.ann_sharpe(r, rf_annual)
    assert np.isclose(s_series, s_float, atol=1e-10)


def test_ann_sharpe_and_summary_require_rf():
    r = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(TypeError):
        ml.ann_sharpe(r)
    with pytest.raises(TypeError):
        ml.summary(r)


# ── f. No look-ahead: garbage in rows at/after asof must not change weights ─

@pytest.mark.parametrize("strategy_cls", [GMV, functools.partial(MaxSharpe, rf=0.0)])
def test_no_lookahead(strategy_cls):
    panel, log_returns = make_panel(seed=6)
    asof = log_returns.index[COV_LOOKBACK]

    w_before = strategy_cls()(panel, asof)

    tainted = log_returns.copy()
    tainted.loc[tainted.index >= asof] = 1e6
    tainted_panel = Panel({"returns": tainted})
    w_after = strategy_cls()(tainted_panel, asof)

    pd.testing.assert_series_equal(w_before, w_after, atol=1e-12, check_names=False)


# ── g. _validate: NaN / bad sum / negative-under-LONG_ONLY / LONG_SHORT passthrough

class _DummyLongOnly(Strategy):
    name = "DummyLO"
    family = "Benchmark"
    constraint = LONG_ONLY

    def __init__(self, weights):
        super().__init__()
        self._weights = weights

    def predict_weights(self, panel, asof):
        return self._weights


class _DummyLongShort(Strategy):
    name = "DummyLS"
    family = "Overlay"
    constraint = LONG_SHORT

    def __init__(self, weights):
        super().__init__()
        self._weights = weights

    def predict_weights(self, panel, asof):
        return self._weights


def test_validate_raises_on_nan():
    vals = [np.nan] + [1.0 / (N_ASSETS - 1)] * (N_ASSETS - 1)
    strat = _DummyLongOnly(pd.Series(vals, index=UNIVERSE))
    with pytest.raises(ValueError):
        strat(None, None)


def test_validate_raises_on_bad_sum():
    vals = [0.5] * N_ASSETS  # sums to 13, not 1
    strat = _DummyLongOnly(pd.Series(vals, index=UNIVERSE))
    with pytest.raises(ValueError):
        strat(None, None)


def test_validate_raises_on_negative_under_long_only():
    vals = [-0.5] + [1.5 / (N_ASSETS - 1)] * (N_ASSETS - 1)
    strat = _DummyLongOnly(pd.Series(vals, index=UNIVERSE))
    with pytest.raises(ValueError):
        strat(None, None)


def test_validate_long_short_passthrough_unchanged():
    vals = [1.5, -0.5] + [0.0] * (N_ASSETS - 2)
    strat = _DummyLongShort(pd.Series(vals, index=UNIVERSE))
    w = strat(None, None)
    assert w.iloc[0] == 1.5
    assert w.iloc[1] == -0.5
    assert (w.iloc[2:] == 0.0).all()
    assert np.isclose(w.sum(), 1.0)


# ── h. _estimate raises when the slice has fewer than lookback rows ─────────

def test_estimate_raises_on_insufficient_history():
    panel, log_returns = make_panel(n_days=50, seed=7)
    asof = log_returns.index[10]  # only 10 rows of history precede this
    strat = GMV()
    with pytest.raises(ValueError):
        strat._estimate(panel, asof)


# ── i. Labels ────────────────────────────────────────────────────────────────

def test_labels():
    assert GMV().label == "GMV(sample)"
    assert MaxSharpe(cov_estimator=ml.ledoit_wolf_cov).label == "MaxSharpe(ledoit_wolf)"
    assert EqualWeight().label == "EqualWeight"


# ── j. BetaTargetMinVar ───────────────────────────────────────────────────────

def test_beta_target_minvar_weights_valid():
    panel, log_returns, rf = make_panel_with_rf(seed=10)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar(beta_target=0.3)

    w = strat(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()

    beta = strat._beta_vector(panel, asof)
    achieved = float(beta.reindex(UNIVERSE).to_numpy() @ w.to_numpy())
    assert achieved >= 0.3 - 1e-8


def test_beta_target_minvar_market_beta_is_one():
    panel, log_returns, rf = make_panel_with_rf(seed=11)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar()

    beta = strat._beta_vector(panel, asof)
    assert np.isclose(beta["SPY"], 1.0, atol=1e-10)


def test_beta_target_minvar_zero_target_matches_gmv_and_is_slack():
    panel, log_returns, rf = make_panel_with_rf(seed=12)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar(beta_target=0.0)

    w = strat(panel, asof)
    w_gmv = GMV()(panel, asof)

    pd.testing.assert_series_equal(w, w_gmv, atol=1e-6, check_names=False)
    assert strat.slack_dates == [asof]


def test_beta_target_minvar_infeasible_falls_back_to_gmv(caplog):
    panel, log_returns, rf = make_panel_with_rf(seed=13)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar(beta_target=10.0)

    with caplog.at_level(logging.WARNING, logger="maplab.models"):
        w = strat(panel, asof)
    w_gmv = GMV()(panel, asof)

    pd.testing.assert_series_equal(w, w_gmv, atol=1e-8, check_names=False)
    assert strat.infeasible_dates == [asof]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_beta_target_minvar_raises_without_rf_frame():
    panel, log_returns = make_panel(seed=14)  # no "rf" frame
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar()
    with pytest.raises(ValueError):
        strat(panel, asof)


def test_beta_target_minvar_raises_on_market_not_in_universe():
    panel, log_returns, rf = make_panel_with_rf(seed=15)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BetaTargetMinVar(market="NOTATICKER")
    with pytest.raises(ValueError):
        strat(panel, asof)


# ── k. BlackLitterman ─────────────────────────────────────────────────────

def test_black_litterman_weights_valid():
    panel, log_returns, rf = make_panel_with_rf(seed=20)
    asof = log_returns.index[COV_LOOKBACK]
    w = BlackLitterman()(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()


def test_black_litterman_zero_k_matches_ew_and_mu_bl_equals_pi():
    panel, log_returns, rf = make_panel_with_rf(seed=21)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman(k=0.0)

    w = strat(panel, asof)
    w_ew = pd.Series(1.0 / N_ASSETS, index=UNIVERSE)
    pd.testing.assert_series_equal(w, w_ew, atol=1e-5, check_names=False, check_exact=False)

    post = strat.posterior(panel, asof)
    assert np.allclose(post["mu_bl"].to_numpy(), post["pi"].to_numpy(), atol=1e-14, rtol=0)


def test_black_litterman_prior_identity_is_ew():
    panel, log_returns, rf = make_panel_with_rf(seed=22)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman()
    post = strat.posterior(panel, asof)

    w = tangency_closed_form(post["pi"], post["Sigma"], 0.0)
    w_ew = pd.Series(1.0 / N_ASSETS, index=UNIVERSE)
    pd.testing.assert_series_equal(w, w_ew, atol=1e-10, check_names=False)


def test_black_litterman_tau_invariance():
    panel, log_returns, rf = make_panel_with_rf(seed=23)
    asof = log_returns.index[COV_LOOKBACK]

    mu_bl_lo = BlackLitterman(tau=0.01).posterior(panel, asof)["mu_bl"]
    mu_bl_hi = BlackLitterman(tau=1.0).posterior(panel, asof)["mu_bl"]
    assert np.allclose(mu_bl_lo.to_numpy(), mu_bl_hi.to_numpy(), rtol=1e-10, atol=1e-14)


def test_black_litterman_closed_form_matches_derivation():
    panel, log_returns, rf = make_panel_with_rf(seed=24)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman(k=0.2)
    post = strat.posterior(panel, asof)

    Sigma_np = post["Sigma"].to_numpy()
    delta = post["delta"]
    w_ew = post["w_ew"].to_numpy()
    signs = post["signs"].to_numpy()
    sigma = post["sigma"].to_numpy()

    lhs = np.linalg.solve(Sigma_np, post["mu_bl"].to_numpy())
    D = np.diag(np.diag(Sigma_np))
    rhs = delta * w_ew + 0.2 * np.linalg.solve(Sigma_np + D, signs * sigma)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_black_litterman_views_match_independent_momentum_and_q():
    panel, log_returns, rf = make_panel_with_rf(seed=25)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman(k=0.2, mom_skip=21)
    post = strat.posterior(panel, asof)

    rets = log_returns.loc[log_returns.index < asof].iloc[-COV_LOOKBACK:]
    rf_slice = rf.loc[rf.index < asof].iloc[-COV_LOOKBACK:]
    rf_log = np.log1p(rf_slice.iloc[:, 0])
    excess = rets.sub(rf_log, axis=0)
    momentum = excess.iloc[: COV_LOOKBACK - 21].sum()
    signs_expected = np.sign(momentum.to_numpy())

    assert np.array_equal(post["signs"].to_numpy(), signs_expected)
    diff = (post["q"] - post["pi"]).to_numpy()
    expected_diff = 0.2 * signs_expected * post["sigma"].to_numpy()
    assert np.allclose(diff, expected_diff, atol=1e-14, rtol=0)


def test_black_litterman_degenerate_falls_back_to_gmv(caplog):
    panel, log_returns, rf = make_panel_with_rf(mean=-0.005, vol=0.01, seed=16)
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman(k=10)

    post = strat.posterior(panel, asof)
    assert post["mu_bl"].max() <= 0

    with caplog.at_level(logging.WARNING, logger="maplab.models"):
        w = strat(panel, asof)
    w_gmv = pd.Series(_min_variance_long_only(post["Sigma"]), index=post["Sigma"].columns)

    pd.testing.assert_series_equal(w, w_gmv, atol=1e-8, check_names=False)
    assert strat.fallback_dates == [asof]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_black_litterman_concentration_bookkeeping():
    panel, log_returns, rf = make_panel_with_rf(seed=20)
    asof = log_returns.index[COV_LOOKBACK]

    strat_100 = BlackLitterman(k=0.2, effn_floor=100.0)
    w_100 = strat_100(panel, asof)
    assert strat_100.concentration_dates == [asof]

    strat_default = BlackLitterman(k=0.2)
    w_default = strat_default(panel, asof)

    assert np.allclose(w_default.to_numpy(), w_100.to_numpy(), atol=0.0, rtol=0.0)


def test_black_litterman_raises_without_rf_frame():
    panel, log_returns = make_panel(seed=26)  # no "rf" frame
    asof = log_returns.index[COV_LOOKBACK]
    strat = BlackLitterman()
    with pytest.raises(ValueError):
        strat(panel, asof)


def test_black_litterman_raises_on_invalid_mom_skip():
    with pytest.raises(ValueError):
        BlackLitterman(mom_skip=COV_LOOKBACK)


def test_black_litterman_label():
    assert BlackLitterman().label == "BL(k=0.1)(sample)"
