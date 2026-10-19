"""Unit tests for maplab.roster batch 2 — synthetic data only, no real cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maplab.backtest import backtest
from maplab.contract import COV_LOOKBACK
from maplab.data import Panel
from maplab.models import GMV, MaxSharpe, EqualWeight, _tangency_long_only
from maplab.roster import (
    AdversarialDiversifier, MaxDDConstrained, MaxSortino, QuarterlyHold, ResampledEF,
    RobustMaxSharpe, TPATwoFactor, TrailingEnsemble, MinimumCorrelation, on_universe,
    max_drawdown_uncompounded, _max_return_dd_limit, _robust_tangency, tracking_error,
)
from test_roster import U18, B18, make_panel, check_valid


def asof_of(logr, k=5):
    return logr.index[COV_LOOKBACK + k]


# ── Robust MV ───────────────────────────────────────────────────────────────

def test_robust_kappa_zero_is_max_sharpe_and_kappa_shrinks_toward_diversified():
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    w_ms = on_universe(MaxSharpe, U18)()(panel, asof)
    w0 = RobustMaxSharpe(universe=U18, kappa=0.0)(panel, asof)
    assert np.abs(w0 - w_ms).max() < 1e-5
    r = RobustMaxSharpe(universe=U18, kappa=1.0)
    w1 = r(panel, asof)
    check_valid(w1)
    H = lambda w: -np.sum(w[w > 0] * np.log(w[w > 0]))
    assert H(w1) >= H(w_ms) - 1e-9 or r.fallback_dates == [asof]


def test_robust_infeasible_falls_back_to_gmv_and_records():
    panel, logr = make_panel()
    asof = asof_of(logr)
    r = RobustMaxSharpe(universe=U18, kappa=1e6)
    w = r(panel, asof)
    assert r.fallback_dates == [asof]
    assert np.allclose(w, on_universe(GMV, U18)()(panel, asof), atol=1e-8)


def test_robust_worst_case_sharpe_beats_random_portfolios():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.normal(size=(n, n)); S = A @ A.T / n * 0.04 + np.eye(n) * 0.01
    ex = np.array([0.08, 0.05, 0.02, 0.06, -0.01, 0.03])
    th = np.sqrt(np.diag(S))
    kappa = 0.5
    w = _robust_tangency(ex, S, th, kappa, "t", None)
    wc = lambda v: (v @ ex - kappa * np.linalg.norm(th * v)) / np.sqrt(v @ S @ v)
    for _ in range(3000):
        assert wc(rng.dirichlet(np.ones(n))) <= wc(w) + 1e-9


# ── Resampled EF ────────────────────────────────────────────────────────────

def test_resampled_is_deterministic_per_asof_and_valid():
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    a = ResampledEF(universe=U18, n_draws=40)(panel, asof)
    b = ResampledEF(universe=U18, n_draws=40)(panel, asof)
    check_valid(a)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    c = ResampledEF(universe=U18, n_draws=40, seed=1)(panel, asof)
    assert np.abs(a - c).max() > 1e-6
    w_ms = on_universe(MaxSharpe, U18)()(panel, asof)
    assert (a > 1e-6).sum() >= (w_ms > 1e-6).sum()                    # averaging diversifies


# ── Sortino ─────────────────────────────────────────────────────────────────

def test_sortino_beats_random_portfolios_on_its_own_ratio():
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    s = MaxSortino(universe=U18, rf=0.0)
    w = s(panel, asof).to_numpy()
    check_valid(pd.Series(w, index=U18))
    mu, _ = s._estimate(panel, asof)
    X = np.expm1(logr.loc[logr.index < asof].iloc[-COV_LOOKBACK:].to_numpy())
    ratio = lambda v: (v @ mu.to_numpy()) / np.sqrt(np.mean(np.minimum(X @ v, 0) ** 2))
    rng = np.random.default_rng(1)
    for _ in range(2000):
        assert ratio(rng.dirichlet(np.ones(18))) <= ratio(w) + 1e-9


def test_sortino_degenerate_falls_back_to_gmv():
    panel, logr = make_panel()
    asof = asof_of(logr)
    s = MaxSortino(universe=U18, rf=10.0)
    s(panel, asof)
    assert s.fallback_dates == [asof]


# ── Max drawdown ────────────────────────────────────────────────────────────

def test_maxdd_respects_limit_and_beats_feasible_portfolios():
    panel, logr = make_panel(seed=2)
    R = np.expm1(logr.iloc[:252].to_numpy())
    limit = 0.5 * min(max_drawdown_uncompounded(R, np.eye(18)[j]) for j in range(17))
    w = _max_return_dd_limit(R, limit, "t", None)
    assert max_drawdown_uncompounded(R, w) <= limit + 1e-9
    rng = np.random.default_rng(2)
    for _ in range(3000):
        v = rng.dirichlet(np.ones(18))
        if max_drawdown_uncompounded(R, v) <= limit:
            assert R.mean(0) @ v <= R.mean(0) @ w + 1e-12


def test_maxdd_infeasible_limit_falls_back_to_min_drawdown():
    panel, logr = make_panel(seed=2)
    asof = asof_of(logr)
    m = MaxDDConstrained(universe=U18, limit=-1.0)                   # impossible: drawdown >= 0
    w = m(panel, asof)
    check_valid(w)
    assert m.fallback_dates == [asof]


# ── TPA ─────────────────────────────────────────────────────────────────────

def test_tpa_keeps_factor_exposure_within_residual_budget_and_beats_replication():
    panel, logr = make_panel(seed=3)
    asof = asof_of(logr)
    t = TPATwoFactor(universe=U18, rf=0.0, factors=("US_LC", "UST_INT"), residual_budget=0.02)
    w = t(panel, asof).to_numpy()
    check_valid(pd.Series(w, index=U18))
    alpha, B, S, idx = t.factor_model(panel, asof)
    assert np.allclose(B.T @ w, [0.6, 0.4], atol=1e-8)
    assert np.allclose(B[U18.index("US_LC")], [1, 0], atol=1e-10)   # factor assets load on themselves
    assert np.sqrt(w @ S @ w) <= 0.02 + 1e-8
    naive = np.zeros(18); naive[idx] = [0.6, 0.4]
    assert naive @ S @ naive < 1e-20                                 # the replication has no residual risk
    assert alpha @ w >= alpha @ naive - 1e-12
    with pytest.raises(ValueError):
        TPATwoFactor(universe=U18, target=(0.7, 0.4))


# ── Adversarial ─────────────────────────────────────────────────────────────

def test_adversarial_meets_floor_and_is_far_from_centroid():
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    members = [on_universe(EqualWeight, U18)(), on_universe(GMV, U18)(), MinimumCorrelation(universe=U18)]
    adv = AdversarialDiversifier(members, rf=0.0)
    w = adv(panel, asof).to_numpy()
    check_valid(pd.Series(w, index=U18))
    mu, Sigma = adv._estimate(panel, asof)
    ex, S = mu.to_numpy(), Sigma.to_numpy()
    wt, _ = _tangency_long_only(ex, S, "t", asof)
    floor = 0.75 * (wt @ ex) / np.sqrt(wt @ S @ wt)
    assert (w @ ex) / np.sqrt(w @ S @ w) >= floor - 1e-9
    c = adv.centroid[asof].to_numpy()
    tv = lambda v: (v - c) @ S @ (v - c)
    rng = np.random.default_rng(4)
    for _ in range(3000):
        v = rng.dirichlet(np.ones(18) * 0.3)
        if (v @ ex) / np.sqrt(v @ S @ v) >= floor:
            assert tv(v) <= tv(w) + 1e-12


def test_adversarial_from_precomputed_weights_equals_live():
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    live = AdversarialDiversifier([on_universe(EqualWeight, U18)(), on_universe(GMV, U18)()], rf=0.0)
    wl = {"EqualWeight": pd.DataFrame([on_universe(EqualWeight, U18)()(panel, asof)], index=[asof]),
          "GMV(sample)": pd.DataFrame([on_universe(GMV, U18)()(panel, asof)], index=[asof])}
    pre = AdversarialDiversifier(member_weights=wl, universe=U18, rf=0.0)
    assert np.allclose(live(panel, asof), pre(panel, asof))
    with pytest.raises(ValueError):
        AdversarialDiversifier(universe=U18)


# ── Trailing ensembles ──────────────────────────────────────────────────────

def _members_run(n_days=1300, seed=7, end="2024-09-30"):
    # the harness's last segment starts at a calendar month-end label, so the data must end on
    # one that is a business day (2024-09-30 Mon, 2022-08-31 Wed)
    panel, logr = make_panel(n_days=n_days, seed=seed)
    end = pd.Timestamp(end)
    assert logr.index[-1] >= end
    logr = logr.loc[:end]
    panel = Panel({"returns": logr, "rf": panel.slice(end + pd.Timedelta(days=1), "rf")})
    simple = np.expm1(logr)
    ms = [on_universe(EqualWeight, U18)(), on_universe(GMV, U18)(), MinimumCorrelation(universe=U18)]
    W, R = {}, {}
    for m in ms:
        net, wlog, _ = backtest(m, simple, panel)
        W[m.label], R[m.label] = wlog, net
    rf = np.expm1(panel.slice(end + pd.Timedelta(days=1), "rf")["BIL"])
    return panel, simple, W, pd.DataFrame(R), rf


@pytest.mark.parametrize("method", ["backtest_sharpe", "composite_score", "meta_optimization"])
def test_trailing_ensembles_warm_up_then_weight_out_of_sample(method):
    panel, simple, W, R, rf = _members_run()
    ens = TrailingEnsemble(W, R, rf, method, universe=U18, window_months=12, te_reference=B18)
    net, wlog, _ = backtest(ens, simple, panel)
    assert np.allclose(wlog.sum(axis=1), 1.0)
    assert ens.warmup_dates and ens.warmup_dates[0] == wlog.index[0]
    late = [d for d in wlog.index if d not in ens.warmup_dates]
    assert late, "never left warm-up"
    d = late[-1]
    mix = ens.mix[d]
    assert np.isclose(mix.sum(), 1) and (mix >= -1e-12).all()
    # out of sample: changing member returns on/after d does not change the mix at d
    R2 = R.copy(); R2.loc[R2.index >= d] *= -5
    ens2 = TrailingEnsemble(W, R2, rf, method, universe=U18, window_months=12, te_reference=B18)
    assert np.allclose(ens2(panel, d), wlog.loc[d], atol=1e-12)


def test_backtest_sharpe_mix_is_proportional_to_positive_sharpe():
    panel, simple, W, R, rf = _members_run()
    ens = TrailingEnsemble(W, R, rf, "backtest_sharpe", universe=U18, window_months=12)
    d = list(W.values())[0].index[-1]
    ens(panel, d)
    h = R.loc[(R.index >= d - pd.DateOffset(months=12)) & (R.index < d)].sub(rf.reindex(R.index).fillna(0), axis=0)
    sr = (h.mean() / h.std() * np.sqrt(252)).clip(lower=0)
    if sr.sum() > 0:
        assert np.allclose(ens.mix[d], sr / sr.sum())


# ── Quarterly hold ──────────────────────────────────────────────────────────

def test_quarterly_hold_trades_only_at_quarter_ends():
    panel, simple, W, R, rf = _members_run(n_days=700, end="2022-08-31")
    q = QuarterlyHold(on_universe(GMV, U18)())
    net, wlog, diag = backtest(q, simple, panel)
    turn = diag["turnover_per_rebalance"]
    off = turn[[d.month not in (3, 6, 9, 12) for d in turn.index]].iloc[1:]
    assert len(off) > 0 and off.max() < 1e-12
    on = turn[[d.month in (3, 6, 9, 12) for d in turn.index]]
    assert on.max() > 1e-4


# ── Tangency fallback (macOS linesearch edge) ───────────────────────────────

def test_tangency_robust_rescales_on_failure_with_same_answer(monkeypatch):
    import maplab.roster as r
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)); S = A @ A.T / 6 * 0.04 + np.eye(6) * 0.01
    ex = np.array([0.08, 0.05, 0.02, 0.06, -0.01, 0.03])
    w_ref = r._tangency_robust(ex, S, "t", None)
    real, calls, hits = r._tangency_long_only, {"n": 0}, []

    def flaky(e, Sig, label, asof, on_retry=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Positive directional derivative for linesearch")
        return real(e, Sig, label, asof, on_retry)

    monkeypatch.setattr(r, "_tangency_long_only", flaky)
    w = r._tangency_robust(ex, S, "t", None, on_rescale=lambda: hits.append(1))
    assert hits == [1] and np.abs(w - w_ref).max() < 1e-7


def test_resampled_records_rescaled_draws(monkeypatch):
    import maplab.roster as r
    panel, logr = make_panel(seed=5)
    asof = asof_of(logr)
    base = ResampledEF(universe=U18, n_draws=10)(panel, asof)
    real, calls = r._tangency_long_only, {"n": 0}

    def flaky(e, Sig, label, asof_, on_retry=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("Positive directional derivative for linesearch")
        return real(e, Sig, label, asof_, on_retry)

    monkeypatch.setattr(r, "_tangency_long_only", flaky)
    st = ResampledEF(universe=U18, n_draws=10)
    w = st(panel, asof)
    assert st.retry_dates == [asof] and np.abs(w - base).max() < 1e-6
