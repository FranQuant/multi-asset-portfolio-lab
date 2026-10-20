"""Forced-failure tests for maplab.roster's solver fallback (synthetic data only).

On macOS (Accelerate) SLSQP sometimes stops at the optimum and still reports failure. `_fallback` accepts
a point only if it is feasible and passes the KKT certificate. Two patched SLSQPs:
  flagged  — returns its real answer but success=False (the macOS case): must be accepted, unchanged
  stalled  — returns its start point, success=False: trust-constr must reach the optimum where it can;
             where it cannot (max entropy), the fallback must RAISE, never return a wrong point.
References: the clean SLSQP solutions (checked against cvxpy in test_roster_reference.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.optimize as sopt

import maplab.roster as r
from test_roster_reference import scenarios

W_TOL = 1e-5


@pytest.fixture
def flagged_slsqp(monkeypatch):
    real = sopt.minimize

    def fake(fun, x0, *a, method=None, **kw):
        res = real(fun, x0, *a, method=method, **kw)
        if method == "SLSQP":
            res.success, res.message = False, "forced: Positive directional derivative for linesearch"
        return res

    monkeypatch.setattr(r.opt, "minimize", fake)
    import maplab.models as m
    monkeypatch.setattr(m.opt, "minimize", fake)


@pytest.fixture
def stalled_slsqp(monkeypatch):
    real = sopt.minimize

    def fake(fun, x0, *a, method=None, **kw):
        if method == "SLSQP":
            return sopt.OptimizeResult(x=np.asarray(x0, float).copy(), success=False, status=8,
                                       message="forced: Positive directional derivative for linesearch",
                                       fun=fun(np.asarray(x0, float)))
        return real(fun, x0, *a, method=method, **kw)

    monkeypatch.setattr(r.opt, "minimize", fake)
    import maplab.models as m
    monkeypatch.setattr(m.opt, "minimize", fake)


def problem(seed=11):
    R = scenarios(seed)
    S = np.cov(R, rowvar=False) * 252
    mu = R.mean(0) * 252
    return R, mu - mu.mean() + 0.02, S


def clean(fn):
    return fn()


@pytest.mark.parametrize("seed", [11, 12])
def test_tangency_falls_back_to_trust_constr(seed, request):
    R, ex, S = problem(seed)
    w0 = r._tangency_robust(ex, S, "t", None)
    request.getfixturevalue("stalled_slsqp")
    hits = []
    w1 = r._tangency_robust(ex, S, "t", None, on_rescale=lambda: hits.append(1))
    assert hits and np.abs(w1 - w0).max() <= W_TOL


@pytest.mark.parametrize("seed", [11, 12])
def test_max_entropy_flagged_at_optimum_is_accepted(seed, request):
    R, ex, S = problem(seed)
    wt = r._tangency_robust(ex, S, "t", None)
    floor = 0.95 * (wt @ ex) / np.sqrt(wt @ S @ wt)
    w0 = r._max_entropy_sharpe_floor(ex, S, floor, "t", None, x_retry=wt)
    request.getfixturevalue("flagged_slsqp")
    w1 = r._max_entropy_sharpe_floor(ex, S, floor, "t", None, x_retry=wt)
    assert np.abs(w1 - w0).max() <= W_TOL


def test_max_entropy_stalled_raises_instead_of_a_wrong_point(request):
    R, ex, S = problem(11)
    wt = r._tangency_robust(ex, S, "t", None)
    floor = 0.95 * (wt @ ex) / np.sqrt(wt @ S @ wt)
    request.getfixturevalue("stalled_slsqp")
    with pytest.raises(RuntimeError, match="KKT-certified"):
        r._max_entropy_sharpe_floor(ex, S, floor, "t", None, x_retry=wt)


@pytest.mark.parametrize("fixture", ["flagged_slsqp", "stalled_slsqp"])
def test_tangency_paths(fixture, request):
    R, ex, S = problem(12)
    w0 = r._tangency_robust(ex, S, "t", None)
    request.getfixturevalue(fixture)
    assert np.abs(r._tangency_robust(ex, S, "t", None) - w0).max() <= W_TOL


def test_kkt_certificate_separates_optimum_from_perturbed():
    R, ex, S = problem(11)
    w = r._tangency_robust(ex, S, "t", None)
    y = w / (ex @ w)
    b, cons = [(0.0, None)] * len(y), [{"type": "eq", "fun": lambda y: ex @ y - 1, "jac": lambda y: ex}]
    assert r.kkt_residual(y, lambda y: 2 * S @ y, b, cons) <= r.KKT_TOL
    yp = y.copy(); yp[np.argmax(y)] *= 1.01; yp /= ex @ yp
    assert r.kkt_residual(yp, lambda y: 2 * S @ y, b, cons) > r.KKT_TOL


def test_robust_tangency_flagged_accepted_stalled_raises(request, monkeypatch):
    R, ex, S = problem(11)
    th = np.sqrt(np.diag(S))
    w0 = r._robust_tangency(ex, S, th, 0.5, "t", None)
    with monkeypatch.context() as mp:
        real = sopt.minimize

        def flagged(fun, x0, *a, method=None, **kw):
            res = real(fun, x0, *a, method=method, **kw)
            if method == "SLSQP":
                res.success = False
            return res
        mp.setattr(r.opt, "minimize", flagged)
        assert np.abs(r._robust_tangency(ex, S, th, 0.5, "t", None) - w0).max() <= W_TOL
    request.getfixturevalue("stalled_slsqp")                     # trust-constr cannot certify here
    with pytest.raises(RuntimeError, match="KKT-certified"):
        r._robust_tangency(ex, S, th, 0.5, "t", None)


def test_sortino_falls_back(request):
    R, ex, S = problem(12)
    X = R - 0.02 / 252
    w0 = r._max_sortino(ex, X, "t", None)
    request.getfixturevalue("stalled_slsqp")
    w1 = r._max_sortino(ex, X, "t", None)
    ratio = lambda v: (v @ ex) / np.sqrt(np.mean(np.minimum(X @ v, 0) ** 2))
    assert abs(ratio(w1) - ratio(w0)) / abs(ratio(w0)) <= 1e-7


def test_tpa_falls_back(request):
    from maplab.data import Panel
    R = scenarios(13)
    names = [f"A{i:02d}" for i in range(R.shape[1])]
    idx = pd.bdate_range("2020-01-01", periods=len(R) + 1)
    panel = Panel({"returns": pd.DataFrame(np.log1p(R), index=idx[:-1], columns=names)})
    mk = lambda: r.TPATwoFactor(universe=names, rf=0.0, lookback=len(R), factors=("A00", "A01"), residual_budget=0.03)
    w0 = mk()(panel, idx[-1]).to_numpy()
    request.getfixturevalue("stalled_slsqp")
    t = mk()
    w1 = t(panel, idx[-1]).to_numpy()
    alpha, B, Sres, _ = t.factor_model(panel, idx[-1])
    assert t.retry_dates == [idx[-1]]
    assert np.allclose(B.T @ w1, [0.6, 0.4], atol=1e-7) and np.sqrt(w1 @ Sres @ w1) <= 0.03 + 1e-7
    assert alpha @ w1 >= alpha @ w0 - 1e-7 * max(abs(alpha @ w0), 1e-6)


def test_es_parity_flagged_at_optimum_is_accepted(request):
    R = scenarios(11)[:, :8]
    w0 = r._es_parity_long_only(R, 0.95, "t", None)
    request.getfixturevalue("flagged_slsqp")
    w1 = r._es_parity_long_only(R, 0.95, "t", None)
    assert r.es_parity_kkt_residual(R, w1, 0.95) < 1e-4
    assert np.abs(w1 - w0).max() <= 1e-4


def test_fallback_raises_when_nothing_is_feasible():
    with pytest.raises(RuntimeError, match="KKT-certified"):
        r._fallback(lambda x: x @ x, lambda x: 2 * x, [np.array([0.5, 0.5])], [(0, 1), (0, 1)],
                    [{"type": "eq", "fun": lambda x: x.sum() - 5.0, "jac": lambda x: np.ones(2)}], [], "t", None)
