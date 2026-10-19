"""Reference checks of maplab.roster's solvers — synthetic data only, no real cache.

  MinCVaR         vs PyPortfolioOpt 1.6.0 EfficientCVaR.min_cvar (same Rockafellar–Uryasev LP)
  TailRiskParity  vs cvxpy: min ES(y) - (1/n) sum log y  (no PyPortfolioOpt equivalent)
  MaxEntropy      vs cvxpy: max sum entr(w) s.t. sum w = 1, w >= 0, w'ex >= floor ||L'w||
MinimumCorrelation has no PyPortfolioOpt equivalent; it is pinned to Varadi's published
spreadsheet example in test_roster.py.

Reference precision: CLARABEL with tol_gap_abs = tol_gap_rel = tol_feas = 1e-12 (as
test_reference.py). Gates: objective |f_maplab - f_ref| / |f_ref| <= 1e-8 and maplab not worse
by more than 1e-9 (relative); weights max|dw| <= 1e-5 where the optimum is unique (tail-risk
parity and max entropy are strictly convex). Min-CVaR weights are REPORT-ONLY: the LP optimum
need not be unique, so only the objective is gated. Max entropy's "not worse" gate uses a
reference on a 1e-8-tightened floor (CLARABEL can end slightly infeasible; see the test).

Scenarios: 252 x 17 fat-tailed (Student-t, 4 df) correlated daily returns, three seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cvxpy")
pytest.importorskip("pypfopt")
from pypfopt import EfficientCVaR  # noqa: E402

from maplab.models import _tangency_long_only  # noqa: E402
from maplab.roster import (  # noqa: E402
    _es_parity_long_only, _max_entropy_sharpe_floor, _min_cvar_long_only, historical_cvar,
)

SOLVER = "CLARABEL"
CLARABEL_OPTIONS = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
OBJ_TOL, NOT_WORSE, W_TOL = 1e-8, 1e-9, 1e-5
SEEDS = [11, 12, 13]
BETA = 0.95


def scenarios(seed, T=252, n=17, df=4):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, 3))
    vols = rng.uniform(0.003, 0.02, n)
    z = rng.standard_t(df, size=(T, n)) / np.sqrt(df / (df - 2))
    f = rng.standard_t(df, size=(T, 3)) / np.sqrt(df / (df - 2))
    return 0.0003 + vols * (0.6 * z + 0.4 * (f @ L.T) / np.sqrt((L ** 2).sum(1)))


@pytest.mark.parametrize("seed", SEEDS)
def test_min_cvar_matches_pypfopt(seed):
    R = scenarios(seed)
    w, obj = _min_cvar_long_only(R, BETA, "ref", None)
    ef = EfficientCVaR(None, pd.DataFrame(R), beta=BETA, weight_bounds=(0, 1),
                       solver=SOLVER, solver_options=CLARABEL_OPTIONS)
    ef.min_cvar()
    w_ref = np.array(list(ef.clean_weights(cutoff=0, rounding=None).values()))
    f_ref = historical_cvar(R, w_ref, BETA)
    f_ml = historical_cvar(R, w, BETA)
    assert abs(f_ml - f_ref) / abs(f_ref) <= OBJ_TOL
    assert f_ml <= f_ref * (1 + NOT_WORSE)
    print(f"seed {seed}: CVaR maplab {f_ml:.10f} ref {f_ref:.10f}  max|dw| {np.abs(w - w_ref).max():.2e} (report-only)")


def _es_parity_ref(R):
    T, n = R.shape
    k = 1.0 / ((1.0 - BETA) * T)
    y, a, u = cp.Variable(n), cp.Variable(), cp.Variable(T)
    obj = a + k * cp.sum(u) - cp.sum(cp.log(y)) / n
    prob = cp.Problem(cp.Minimize(obj), [u >= -R @ y - a, u >= 0])
    prob.solve(solver=SOLVER, **CLARABEL_OPTIONS)
    yv = y.value
    return yv / yv.sum(), lambda yy: historical_cvar(R, yy, BETA) - np.log(yy).sum() / n, yv


@pytest.mark.parametrize("seed", SEEDS)
def test_tail_risk_parity_matches_cvxpy(seed):
    R = scenarios(seed)
    w = _es_parity_long_only(R, BETA, "ref", None)
    w_ref, f, y_ref = _es_parity_ref(R)
    # compare the barrier objective at each solution's optimal scale (ES(y) = 1 at the optimum)
    y_ml = w / historical_cvar(R, w, BETA)
    f_ml, f_ref = f(y_ml), f(y_ref)
    assert abs(f_ml - f_ref) / abs(f_ref) <= OBJ_TOL
    assert f_ml <= f_ref + NOT_WORSE * abs(f_ref)
    assert np.abs(w - w_ref).max() <= W_TOL


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("floor_frac", [0.75, 0.95])
def test_max_entropy_matches_cvxpy(seed, floor_frac):
    R = scenarios(seed)
    mu = R.mean(0) * 252
    excess = mu - mu.mean() + 0.02                         # a spread of positive and negative excess
    S = np.cov(R, rowvar=False) * 252
    w_t, _ = _tangency_long_only(excess, S, "ref", None)
    floor = floor_frac * (w_t @ excess) / np.sqrt(w_t @ S @ w_t)
    w = _max_entropy_sharpe_floor(excess, S, floor, "ref", None, x_retry=w_t)
    Lc = np.linalg.cholesky(S)
    H = lambda v: -np.sum(v * np.log(np.clip(v, 1e-300, None)))
    sr = lambda v: (v @ excess) / np.sqrt(v @ S @ v)

    def ref(fl):
        x = cp.Variable(len(excess))
        cp.Problem(cp.Maximize(cp.sum(cp.entr(x))),
                   [cp.sum(x) == 1, x >= 0, excess @ x >= fl * cp.norm(Lc.T @ x)]).solve(
            solver=SOLVER, **CLARABEL_OPTIONS)
        return x.value

    w_ref = ref(floor)
    # CLARABEL can stop a few 1e-9 short of the Sharpe floor ("optimal_inaccurate"), which
    # buys it ~1e-9 of entropy; "not worse" is therefore gated against a reference solved on a
    # floor tightened by 1e-8, which is feasible for the original problem.
    w_ref_feasible = ref(floor + 1e-8)
    assert sr(w_ref_feasible) >= floor
    assert sr(w) >= floor - 1e-12
    assert abs(H(w) - H(w_ref)) / abs(H(w_ref)) <= OBJ_TOL
    assert H(w) >= H(w_ref_feasible) * (1 - NOT_WORSE)
    assert np.abs(w - w_ref).max() <= W_TOL
