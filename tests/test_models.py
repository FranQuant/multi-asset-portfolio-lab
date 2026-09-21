"""Unit tests for maplab.models — synthetic data only, no real cache."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.contract import LONG_ONLY, LONG_SHORT, COV_LOOKBACK
from maplab.data import Panel
from maplab.models import (
    Strategy,
    GMV,
    MaxSharpe,
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
    msr = MaxSharpe()
    ew = EqualWeight()

    mu, Sigma = gmv._estimate(panel, asof)
    assert (mu - msr.rf).max() > 0  # sanity: non-degenerate for this seed

    def exante_sharpe(w):
        w = w.reindex(UNIVERSE)
        Sigma_np = Sigma.to_numpy()
        ret = float(w.to_numpy() @ mu.to_numpy())
        vol = float(np.sqrt(w.to_numpy() @ Sigma_np @ w.to_numpy()))
        return (ret - msr.rf) / vol

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
    msr = MaxSharpe()

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
    msr = MaxSharpe()

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
    msr = MaxSharpe()

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


# ── f. No look-ahead: garbage in rows at/after asof must not change weights ─

@pytest.mark.parametrize("strategy_cls", [GMV, MaxSharpe])
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
