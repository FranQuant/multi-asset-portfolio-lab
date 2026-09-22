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
    MostDiversified,
    EqualRiskContribution,
    HierarchicalRiskParity,
    EqualWeight,
    gmv_closed_form,
    tangency_closed_form,
    mdp_closed_form,
    risk_contributions,
    _min_variance_long_only,
    _mdp_long_only,
    _erc_long_only,
    _hrp_long_only,
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


# ── k. MostDiversified ────────────────────────────────────────────────────

def test_mdp_weights_valid():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]
    strat = MostDiversified()
    w = strat(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()
    assert strat.retry_dates == []


def test_mdp_maximizes_dr():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()
    sigma_np = np.sqrt(np.diag(Sigma_np))

    def dr(w: pd.Series) -> float:
        w_np = w.reindex(UNIVERSE).to_numpy()
        return float(w_np @ sigma_np) / np.sqrt(float(w_np @ Sigma_np @ w_np))

    dr_mdp = dr(MostDiversified()(panel, asof))
    dr_ew = dr(EqualWeight()(panel, asof))
    dr_gmv = dr(GMV()(panel, asof))

    rng = np.random.default_rng(0)
    draws = rng.dirichlet(np.full(N_ASSETS, 1.0), size=5000)
    dr_draws = (draws @ sigma_np) / np.sqrt(np.einsum("ij,jk,ik->i", draws, Sigma_np, draws))
    best_baseline = max(dr_ew, dr_gmv, float(np.max(dr_draws)))

    assert dr_mdp >= best_baseline - 1e-10


def test_mdp_constant_correlation_is_inverse_vol():
    rng = np.random.default_rng(30)
    vols = rng.uniform(0.05, 0.30, size=N_ASSETS)
    rho = 0.3
    C = np.full((N_ASSETS, N_ASSETS), rho)
    np.fill_diagonal(C, 1.0)
    D = np.diag(vols)
    Sigma = pd.DataFrame(D @ C @ D, index=UNIVERSE, columns=UNIVERSE)

    w, retried = _mdp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"))
    expected = (1.0 / vols) / np.sum(1.0 / vols)
    assert np.allclose(w, expected, atol=1e-5, rtol=0)
    assert not retried

    w_closed = mdp_closed_form(Sigma)
    assert np.allclose(w_closed.to_numpy(), expected, atol=1e-10, rtol=0)


def test_mdp_equal_vols_is_gmv():
    rng = np.random.default_rng(31)
    L = rng.normal(size=(N_ASSETS, 3))
    raw = L @ L.T + np.eye(N_ASSETS) * 5.0
    d = np.sqrt(np.diag(raw))
    C = raw / np.outer(d, d)

    vol = 0.1
    Sigma = pd.DataFrame((vol ** 2) * C, index=UNIVERSE, columns=UNIVERSE)

    w_mdp, retried = _mdp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"))
    w_gmv = _min_variance_long_only(Sigma)
    assert np.allclose(w_mdp, w_gmv, atol=1e-5, rtol=0)
    assert not retried


def test_mdp_risk_weights_equal_gmv_on_correlation():
    panel, log_returns = make_panel(seed=32)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()
    sigma_np = np.sqrt(np.diag(Sigma_np))
    D_inv = np.diag(1.0 / sigma_np)
    C = pd.DataFrame(D_inv @ Sigma_np @ D_inv, index=UNIVERSE, columns=UNIVERSE)

    w_mdp, _ = _mdp_long_only(Sigma, "test", asof)
    x = (w_mdp * sigma_np) / float(w_mdp @ sigma_np)

    w_gmv_c = _min_variance_long_only(C)
    assert np.allclose(x, w_gmv_c, atol=1e-5, rtol=0)


def test_mdp_risk_weights_scale_invariant():
    panel, log_returns = make_panel(seed=33)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()

    rng = np.random.default_rng(34)
    k = rng.uniform(0.5, 2.0, size=N_ASSETS)
    K = np.diag(k)
    Sigma2_np = K @ Sigma_np @ K
    Sigma2 = pd.DataFrame(Sigma2_np, index=UNIVERSE, columns=UNIVERSE)

    w1, _ = _mdp_long_only(Sigma, "test", asof)
    sigma1 = np.sqrt(np.diag(Sigma_np))
    x1 = (w1 * sigma1) / float(w1 @ sigma1)

    w2, _ = _mdp_long_only(Sigma2, "test", asof)
    sigma2 = np.sqrt(np.diag(Sigma2_np))
    x2 = (w2 * sigma2) / float(w2 @ sigma2)

    assert np.allclose(x1, x2, atol=1e-5, rtol=0)


def test_mdp_labels():
    assert MostDiversified().label == "MDP(sample)"
    assert MostDiversified(cov_estimator=ml.ledoit_wolf_cov).label == "MDP(ledoit_wolf)"


# ── l. EqualRiskContribution ─────────────────────────────────────────────

def test_erc_weights_valid():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]
    strat = EqualRiskContribution()
    w = strat(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert w.min() > 0
    assert len(strat.sweeps) == 1


def test_erc_equal_risk_contributions():
    panel, log_returns = make_panel(seed=40)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    w, _ = _erc_long_only(Sigma, "test", asof)
    w_series = pd.Series(w, index=Sigma.columns)

    rc = risk_contributions(w_series, Sigma)
    shares = rc / rc.sum()
    assert np.allclose(shares.to_numpy(), 1.0 / N_ASSETS, atol=1e-11, rtol=0)

    port_vol = np.sqrt(float(w @ Sigma.to_numpy() @ w))
    assert np.isclose(rc.sum(), port_vol, atol=1e-12, rtol=0)


def test_erc_constant_correlation_is_inverse_vol():
    rng = np.random.default_rng(41)
    vols = rng.uniform(0.05, 0.30, size=N_ASSETS)
    for rho in (0.0, 0.3, 0.7):
        C = np.full((N_ASSETS, N_ASSETS), rho)
        np.fill_diagonal(C, 1.0)
        D = np.diag(vols)
        Sigma = pd.DataFrame(D @ C @ D, index=UNIVERSE, columns=UNIVERSE)

        w, _ = _erc_long_only(Sigma, "test", pd.Timestamp("2020-01-01"))
        expected = (1.0 / vols) / np.sum(1.0 / vols)
        assert np.max(np.abs(w - expected)) <= 1e-11


def test_erc_two_assets_is_inverse_vol():
    v = np.array([0.08, 0.22])
    for rho in (-0.5, 0.0, 0.8):
        S = np.array([
            [v[0] ** 2, rho * v[0] * v[1]],
            [rho * v[0] * v[1], v[1] ** 2],
        ])
        Sigma = pd.DataFrame(S, index=["A", "B"], columns=["A", "B"])
        w, _ = _erc_long_only(Sigma, "test", pd.Timestamp("2020-01-01"))
        expected = (1.0 / v) / np.sum(1.0 / v)
        assert np.max(np.abs(w - expected)) <= 1e-11


def test_erc_vol_between_gmv_and_ew():
    panel, log_returns = make_panel(seed=42)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()

    w_erc, _ = _erc_long_only(Sigma, "test", asof)
    w_gmv = _min_variance_long_only(Sigma)
    w_ew = np.full(N_ASSETS, 1.0 / N_ASSETS)

    def port_vol(w):
        return np.sqrt(float(w @ Sigma_np @ w))

    assert port_vol(w_gmv) <= port_vol(w_erc) + 1e-12
    assert port_vol(w_erc) <= port_vol(w_ew) + 1e-12


def test_erc_risk_weights_equal_erc_on_correlation():
    panel, log_returns = make_panel(seed=43)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()
    sigma_np = np.sqrt(np.diag(Sigma_np))
    D_inv = np.diag(1.0 / sigma_np)
    C = pd.DataFrame(D_inv @ Sigma_np @ D_inv, index=UNIVERSE, columns=UNIVERSE)

    w_erc, _ = _erc_long_only(Sigma, "test", asof)
    x = (w_erc * sigma_np) / float(w_erc @ sigma_np)

    w_erc_c, _ = _erc_long_only(C, "test", asof)
    assert np.max(np.abs(x - w_erc_c)) <= 1e-11


def test_erc_scale_equivariance():
    panel, log_returns = make_panel(seed=44)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()

    rng = np.random.default_rng(45)
    k = rng.uniform(0.5, 2.0, size=N_ASSETS)
    K = np.diag(k)
    Sigma2 = pd.DataFrame(K @ Sigma_np @ K, index=UNIVERSE, columns=UNIVERSE)

    w1, _ = _erc_long_only(Sigma, "test", asof)
    w2, _ = _erc_long_only(Sigma2, "test", asof)
    w2_back = (w2 * k) / np.sum(w2 * k)

    assert np.max(np.abs(w1 - w2_back)) <= 1e-11


def test_erc_weight_times_beta_is_one_over_n():
    panel, log_returns = make_panel(seed=46)
    asof = log_returns.index[COV_LOOKBACK]

    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()

    w, _ = _erc_long_only(Sigma, "test", asof)
    port_var = float(w @ Sigma_np @ w)
    beta = (Sigma_np @ w) / port_var

    assert np.max(np.abs(w * beta - 1.0 / N_ASSETS)) <= 1e-11


def test_erc_raises_if_not_converged():
    panel, log_returns = make_panel(seed=47)
    asof = log_returns.index[COV_LOOKBACK]
    _, Sigma = GMV()._estimate(panel, asof)

    with pytest.raises(RuntimeError):
        _erc_long_only(Sigma, "test", asof, maxiter=1)


def test_erc_labels():
    assert EqualRiskContribution().label == "ERC(sample)"
    assert EqualRiskContribution(cov_estimator=ml.ledoit_wolf_cov).label == "ERC(ledoit_wolf)"


# ── m. HierarchicalRiskParity ─────────────────────────────────────────────

def _ivp_var_independent(Sigma_np, idx):
    idx = list(idx)
    sub = Sigma_np[np.ix_(idx, idx)]
    p = 1.0 / np.diag(sub)
    p = p / p.sum()
    return float(p @ sub @ p)


def test_hrp_weights_valid():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]
    strat = HierarchicalRiskParity()
    w = strat(panel, asof)
    assert list(w.index) == UNIVERSE
    assert np.isclose(w.sum(), 1.0)
    assert w.min() > 0
    assert len(strat.orders) == 1


def test_hrp_order_is_permutation():
    panel, log_returns = make_panel()
    asof = log_returns.index[COV_LOOKBACK]
    _, Sigma = GMV()._estimate(panel, asof)

    for bisection in ("tree", "positional"):
        for linkage in ("single", "average", "ward"):
            _, info = _hrp_long_only(Sigma, "test", asof, linkage=linkage, bisection=bisection)
            assert np.array_equal(np.sort(info["order"]), np.arange(N_ASSETS))


def test_hrp_split_identity():
    panel, log_returns = make_panel(seed=48)
    asof = log_returns.index[COV_LOOKBACK]
    _, Sigma = GMV()._estimate(panel, asof)
    Sigma_np = Sigma.to_numpy()

    for bisection in ("tree", "positional"):
        w, info = _hrp_long_only(Sigma, "test", asof, bisection=bisection)
        for L, R, a in info["splits"]:
            V_L = _ivp_var_independent(Sigma_np, L)
            V_R = _ivp_var_independent(Sigma_np, R)
            expected_a = V_R / (V_L + V_R)
            assert abs(a - expected_a) <= 1e-12
            mass_L = w[L].sum()
            mass_R = w[R].sum()
            assert abs(mass_L / (mass_L + mass_R) - expected_a) <= 1e-12


def test_hrp_diagonal_is_inverse_variance():
    rng = np.random.default_rng(50)
    vols = rng.uniform(0.05, 0.30, size=N_ASSETS)
    Sigma = pd.DataFrame(np.diag(vols ** 2), index=UNIVERSE, columns=UNIVERSE)
    expected = (1.0 / vols ** 2) / np.sum(1.0 / vols ** 2)
    expected_gmv = gmv_closed_form(Sigma).to_numpy()

    for bisection in ("tree", "positional"):
        w, _ = _hrp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"), bisection=bisection)
        assert np.max(np.abs(w - expected)) <= 1e-12
        assert np.max(np.abs(w - expected_gmv)) <= 1e-12


def test_hrp_two_assets_is_inverse_variance():
    v = np.array([0.08, 0.22])
    for rho in (-0.5, 0.0, 0.8):
        S = np.array([
            [v[0] ** 2, rho * v[0] * v[1]],
            [rho * v[0] * v[1], v[1] ** 2],
        ])
        Sigma = pd.DataFrame(S, index=["A", "B"], columns=["A", "B"])
        expected = (1.0 / v ** 2) / np.sum(1.0 / v ** 2)
        for bisection in ("tree", "positional"):
            w, _ = _hrp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"), bisection=bisection)
            assert np.max(np.abs(w - expected)) <= 1e-12


def test_hrp_exchangeable_positional_depends_only_on_split_sizes():
    n = 13
    labels = [f"A{i}" for i in range(n)]
    vol = 0.15

    def V(k, rho):
        return (1.0 - rho) / k + rho

    def rec(k, mass, rho):
        if k == 1:
            return [mass]
        h = k // 2
        a = V(k - h, rho) / (V(h, rho) + V(k - h, rho))
        return rec(h, mass * a, rho) + rec(k - h, mass * (1.0 - a), rho)

    for rho, check_uniform in ((0.0, True), (0.5, False)):
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1.0)
        Sigma = pd.DataFrame(vol ** 2 * C, index=labels, columns=labels)
        w, _ = _hrp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"), bisection="positional")
        expected = np.array(rec(n, 1.0, rho))
        if check_uniform:
            assert np.max(np.abs(w - 1.0 / n)) <= 1e-12
        else:
            assert np.max(np.abs(np.sort(w) - np.sort(expected))) <= 1e-12
            assert np.max(np.abs(w - 1.0 / n)) > 1e-3


def test_hrp_uniform_scale_invariance():
    panel, log_returns = make_panel(seed=49)
    asof = log_returns.index[COV_LOOKBACK]
    _, Sigma = GMV()._estimate(panel, asof)
    Sigma2 = Sigma * 7.3

    for bisection in ("tree", "positional"):
        w1, _ = _hrp_long_only(Sigma, "test", asof, bisection=bisection)
        w2, _ = _hrp_long_only(Sigma2, "test", asof, bisection=bisection)
        assert np.max(np.abs(w1 - w2)) <= 1e-12


def test_hrp_tree_permutation_equivariance():
    panel, log_returns = make_panel(seed=49)
    asof = log_returns.index[COV_LOOKBACK]
    _, Sigma = GMV()._estimate(panel, asof)
    w0, _ = _hrp_long_only(Sigma, "test", asof, bisection="tree")

    rng = np.random.default_rng(51)
    for _ in range(5):
        p = rng.permutation(N_ASSETS)
        Sigma_perm = Sigma.iloc[p, p]
        w_perm, _ = _hrp_long_only(Sigma_perm, "test", asof, bisection="tree")
        wb = np.empty(N_ASSETS)
        wb[p] = w_perm
        assert np.max(np.abs(wb - w0)) <= 1e-12


def test_hrp_root_singleton_weight_and_correlation_blindness():
    n = 13
    labels = [f"A{i}" for i in range(n)]
    rng = np.random.default_rng(52)
    vols = rng.uniform(0.05, 0.30, size=n)

    C = np.full((n, n), 0.6)
    C[-1, :] = -0.3
    C[:, -1] = -0.3
    np.fill_diagonal(C, 1.0)
    D = np.diag(vols)
    Sigma = pd.DataFrame(D @ C @ D, index=labels, columns=labels)

    w, info = _hrp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"), bisection="tree")
    Z = info["Z"]
    assert 12 in (int(Z[-1, 0]), int(Z[-1, 1]))

    V_rest = _ivp_var_independent(Sigma.to_numpy(), list(range(n - 1)))
    expected_w12 = V_rest / (vols[-1] ** 2 + V_rest)
    assert abs(w[12] - expected_w12) <= 1e-12

    C_cf = C.copy()
    C_cf[12, :] = 0.0
    C_cf[:, 12] = 0.0
    C_cf[12, 12] = 1.0
    Sigma_cf = pd.DataFrame(D @ C_cf @ D, index=labels, columns=labels)
    w_cf, info_cf = _hrp_long_only(Sigma_cf, "test", pd.Timestamp("2020-01-01"), bisection="tree")
    Z_cf = info_cf["Z"]
    assert 12 in (int(Z_cf[-1, 0]), int(Z_cf[-1, 1]))
    assert abs(w_cf[12] - w[12]) <= 1e-12


def test_hrp_raises_on_bad_options():
    with pytest.raises(ValueError):
        HierarchicalRiskParity(linkage="complete")
    with pytest.raises(ValueError):
        HierarchicalRiskParity(bisection="halves")

    Sigma = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"],
    )
    with pytest.raises(ValueError):
        _hrp_long_only(Sigma, "test", pd.Timestamp("2020-01-01"))


def test_hrp_labels():
    assert HierarchicalRiskParity().label == "HRP(sample)"
    assert HierarchicalRiskParity(cov_estimator=ml.ledoit_wolf_cov).label == "HRP(ledoit_wolf)"
    assert HierarchicalRiskParity(bisection="positional").label == "HRP[positional](sample)"
    assert HierarchicalRiskParity(linkage="ward").label == "HRP[ward](sample)"
    assert HierarchicalRiskParity(dist_of_dist=False).label == "HRP[direct](sample)"
