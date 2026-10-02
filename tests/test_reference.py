"""Reference checks of maplab's solvers against PyPortfolioOpt 1.6.0 (cvxpy / CLARABEL)
on synthetic covariances — synthetic data only, no real cache.

Install with the `ref` extra: pip install -e ".[ref]"

Reference precision: every EfficientFrontier uses CLARABEL with tol_gap_abs =
tol_gap_rel = tol_feas = 1e-12 (default tolerances are ~1e-8 and would make the
reference, not maplab, the limiting side).

Gates (maplab is never given a looser gate than the reference):
  Objective, all solvers, all cases:
    |f_maplab - f_ref| / |f_ref| <= 1e-8, AND maplab is not worse:
      GMV variance       f_maplab <= f_ref * (1 + 1e-9)   (1e-8 for case C only, see below)
      MaxSharpe / MDP    f_maplab >= f_ref * (1 - 1e-9)   (Sharpe / diversification ratio)
  Weights: max|dw| <= 1e-5 wherever the optimum is identified: every case for
    MaxSharpe and MDP, and GMV cases A, B, D. GMV case C is REPORT-ONLY for weights
    (max|dw| and cond(Sigma) are printed): Sigma_C contains an asset equal to the
    average of two others plus 1e-8 variance noise, so the long-only minimum-variance
    surface is nearly flat along that direction and the weights are not identified at
    1e-5 by any solver; only the objective is.
  Known limitation (GMV case C): at cond(Sigma) ~ 1.4e7, `_min_variance_long_only`
    (SLSQP, ftol 1e-12) stops ~4.5e-9 (relative) above the exact optimum, 1.8045559325789e-3
    (equality-constrained solution, all weights >= 0); the reference is within 5e-11 of it.
    Hence the 1e-8 "not worse" margin for that case. Not seen on real data (cond ~ 300:
    maplab was never worse than the reference by more than 1e-9 at the three check dates).
    The solver is left unchanged because the registered results depend on it.
  HRP (closed-form, no solver): positional / direct / single vs HRPOpt on a jittered
    block case (exact ties in the plain block case make leaf order solver-arbitrary):
    precondition min gap between sorted linkage heights >= 1e-6 on both sides, leaf
    order identical, max|dw| <= 1e-12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pypfopt")

import scipy.cluster.hierarchy as sch  # noqa: E402
from pypfopt import EfficientFrontier, HRPOpt  # noqa: E402

from maplab.models import (  # noqa: E402
    _hrp_long_only,
    _mdp_long_only,
    _min_variance_long_only,
    _tangency_long_only,
)

N = 13
NAMES = [f"A{i:02d}" for i in range(N)]  # zero-padded: HRPOpt sorts its output by label
SOLVER = "CLARABEL"
OBJ_TOL = 1e-8
W_TOL = 1e-5
HRP_W_TOL = 1e-12
NOT_WORSE = 1e-9
NOT_WORSE_GMV_C = 1e-8  # known limitation, see module docstring
CLARABEL_OPTIONS = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
MIN_HEIGHT_GAP = 1e-6
GMV_WEIGHTS_IDENTIFIED = {"A": True, "B": True, "C": False, "D": True}
CASES = ["A", "B", "C", "D"]
D_SEED = 0  # first seed in range(200) whose long-only GMV has >= 5 zero weights


# ── synthetic covariance builders (annualised vols 5-30%) ───────────────────

def _vols(rng):
    return rng.uniform(0.05, 0.30, N)


def _frame(S):
    return pd.DataFrame(S, index=NAMES, columns=NAMES)


def _from_corr(C, vols):
    return _frame(np.outer(vols, vols) * C)


def sigma_a(seed=101):
    """Well-conditioned: Wishart from a 13x1000 normal draw."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 1000))
    W = X @ X.T / 1000
    d = np.sqrt(np.diag(W))
    return _from_corr(W / np.outer(d, d), _vols(rng))


def sigma_b(seed=102):
    """Block-correlated: 3 blocks (5, 4, 4), rho 0.8 inside / 0.2 between, mixed vols."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat([0, 1, 2], [5, 4, 4])
    C = np.where(blocks[:, None] == blocks[None, :], 0.8, 0.2)
    np.fill_diagonal(C, 1.0)
    return _from_corr(C, _vols(rng))


def sigma_c(seed=103):
    """Near-singular: asset 12 = 0.5*asset0 + 0.5*asset1 + tiny noise (var 1e-8)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N - 1, 1000))
    W = X @ X.T / 1000
    d = np.sqrt(np.diag(W))
    v = rng.uniform(0.05, 0.30, N - 1)
    S12 = np.outer(v, v) * (W / np.outer(d, d))
    row = 0.5 * S12[0] + 0.5 * S12[1]
    var = 0.25 * S12[0, 0] + 0.25 * S12[1, 1] + 0.5 * S12[0, 1] + 1e-8
    S = np.zeros((N, N))
    S[: N - 1, : N - 1] = S12
    S[N - 1, : N - 1] = row
    S[: N - 1, N - 1] = row
    S[N - 1, N - 1] = var
    return _frame(S)


def sigma_d(seed=D_SEED):
    """One-factor Σ (rescaled to vols 5-30%) whose long-only GMV has >= 5 zero weights."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.3, 1.5, N)
    idio = rng.uniform(0.02, 0.25, N)
    S0 = np.outer(beta, beta) * 0.15**2 + np.diag(idio**2)
    d = np.sqrt(np.diag(S0))
    return _from_corr(S0 / np.outer(d, d), _vols(rng))


def sigma_b_jittered(seed=104):
    """Case B with a seeded symmetric +-0.02 perturbation of the off-diagonal correlations
    (breaks the exact linkage ties of B; asserted positive definite)."""
    rng = np.random.default_rng(seed)
    S = sigma_b()
    vols = np.sqrt(np.diag(S.to_numpy()))
    C = S.to_numpy() / np.outer(vols, vols)
    E = rng.uniform(-0.02, 0.02, (N, N))
    E = np.triu(E, 1)
    C = C + E + E.T
    assert np.linalg.eigvalsh(C).min() > 0
    return _from_corr(C, vols)


_BUILDERS = {"A": sigma_a, "B": sigma_b, "C": sigma_c, "D": sigma_d}


def build_sigma(case: str) -> pd.DataFrame:
    return _BUILDERS[case]()


def excess_positive() -> np.ndarray:
    return np.random.default_rng(201).uniform(0.02, 0.12, N)


def excess_mixed() -> np.ndarray:
    e = np.random.default_rng(0).normal(0.03, 0.06, N)
    assert (e < 0).sum() >= 3 and e.max() > 0
    return e


_EXCESS = {"positive": excess_positive, "mixed": excess_mixed}


# ── helpers ──────────────────────────────────────────────────────────────────

def _ef(mu, Sigma):
    return EfficientFrontier(mu, Sigma, weight_bounds=(0, 1), solver=SOLVER,
                             solver_options=dict(CLARABEL_OPTIONS))


def _ref_w(weights: dict) -> np.ndarray:
    return np.array([weights[k] for k in NAMES])


def _var(w, S):
    return float(w @ S @ w)


def _sharpe(w, e, S):
    return float(w @ e) / np.sqrt(float(w @ S @ w))


def _rel(a, b):
    return abs(a - b) / abs(b)


def _report(case, m, r):
    return f"case {case}: max|dw|={np.max(np.abs(m - r)):.3e}"


def _check(case, name, f_m, f_r, w_m, w_r, *, minimise, weights_identified=True, cond=None,
           not_worse=NOT_WORSE):
    """Objective gate (relative diff + maplab-not-worse) and weight gate (or report-only)."""
    dw = float(np.max(np.abs(w_m - w_r)))
    rel = _rel(f_m, f_r)
    print(f"[{name} {case}] max|dw|={dw:.3e} rel_obj_diff={rel:.3e}"
          + (f" cond(Sigma)={cond:.3e}" if cond is not None else "")
          + ("" if weights_identified else "  (weights report-only)"))
    assert rel <= OBJ_TOL, f"{name} {case}: rel obj diff {rel:.3e} > {OBJ_TOL:.0e}"
    if minimise:
        assert f_m <= f_r * (1 + not_worse), f"{name} {case}: maplab worse: {f_m!r} vs ref {f_r!r}"
    else:
        assert f_m >= f_r * (1 - not_worse), f"{name} {case}: maplab worse: {f_m!r} vs ref {f_r!r}"
    if weights_identified:
        assert dw <= W_TOL, f"{name} {case}: max|dw|={dw:.3e} > {W_TOL:.0e}"


# ── builders sanity ──────────────────────────────────────────────────────────

def test_case_d_gmv_has_at_least_five_zero_weights():
    w = _min_variance_long_only(sigma_d())
    assert (w < 1e-8).sum() >= 5


def test_case_c_is_near_singular():
    cond = np.linalg.cond(sigma_c().to_numpy())
    print(f"cond(Sigma_C) = {cond:.3e}")
    assert cond > 1e6


# ── GMV ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES)
def test_gmv_matches_pyportfolioopt(case):
    S = build_sigma(case)
    Sn = S.to_numpy()
    w_m = _min_variance_long_only(S)
    w_r = _ref_w(_ef(None, S).min_volatility())
    _check(case, "GMV", _var(w_m, Sn), _var(w_r, Sn), w_m, w_r, minimise=True,
           weights_identified=GMV_WEIGHTS_IDENTIFIED[case], cond=np.linalg.cond(Sn),
           not_worse=NOT_WORSE_GMV_C if case == "C" else NOT_WORSE)


# ── MaxSharpe ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("excess", list(_EXCESS))
@pytest.mark.parametrize("case", CASES)
def test_maxsharpe_matches_pyportfolioopt(case, excess):
    S = build_sigma(case)
    Sn = S.to_numpy()
    e = _EXCESS[excess]()
    w_m, _ = _tangency_long_only(e, Sn, "ref", None)
    w_r = _ref_w(_ef(pd.Series(e, index=NAMES), S).max_sharpe(risk_free_rate=0))
    _check(f"{case}/{excess}", "MaxSharpe", _sharpe(w_m, e, Sn), _sharpe(w_r, e, Sn),
           w_m, w_r, minimise=False)


# ── MDP ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES)
def test_mdp_matches_pyportfolioopt(case):
    S = build_sigma(case)
    Sn = S.to_numpy()
    sig = np.sqrt(np.diag(Sn))
    w_m, _ = _mdp_long_only(S, "ref", None)
    w_r = _ref_w(_ef(pd.Series(sig, index=NAMES), S).max_sharpe(risk_free_rate=0))
    dr = lambda w: float(w @ sig) / np.sqrt(float(w @ Sn @ w))  # noqa: E731
    _check(case, "MDP", dr(w_m), dr(w_r), w_m, w_r, minimise=False)


# ── HRP ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def _scipy_linkage_methods_shim(monkeypatch):
    """PyPortfolioOpt 1.6.0 reads a private scipy attribute removed in scipy 1.18
    (`scipy.cluster.hierarchy._LINKAGE_METHODS`); restore it. Validation only, no
    effect on numbers. Used by the HRP tests only."""
    if not hasattr(sch, "_LINKAGE_METHODS"):
        monkeypatch.setattr(
            sch, "_LINKAGE_METHODS",
            {"single": 0, "complete": 1, "average": 2, "centroid": 3,
             "median": 4, "ward": 5, "weighted": 6},
            raising=False,
        )


def _hrp_pair(S, **kw):
    w_m, info = _hrp_long_only(S, "ref", None, **kw)
    ref = HRPOpt(cov_matrix=S)
    w_r = _ref_w(ref.optimize("single"))
    order_r = list(sch.to_tree(ref.clusters, rd=False).pre_order())
    return w_m, info, w_r, order_r, ref.clusters


def _min_height_gap(Z):
    h = np.sort(Z[:, 2])
    return float(np.diff(h).min())


HRP_CASES = {"A": sigma_a, "B_jittered": sigma_b_jittered, "C": sigma_c, "D": sigma_d}


@pytest.mark.parametrize("case", list(HRP_CASES))
def test_hrp_positional_direct_matches_hrpopt(case, _scipy_linkage_methods_shim):
    S = HRP_CASES[case]()
    w_m, info, w_r, order_r, Z_r = _hrp_pair(
        S, linkage="single", bisection="positional", dist_of_dist=False)
    gap_m, gap_r = _min_height_gap(info["Z"]), _min_height_gap(Z_r)
    assert min(gap_m, gap_r) >= MIN_HEIGHT_GAP, (
        f"case {case}: precondition failed — linkage heights are (near-)tied, leaf order is "
        f"not identified. Smallest gap between sorted linkage heights: maplab {gap_m:.3e}, "
        f"HRPOpt {gap_r:.3e} (need >= {MIN_HEIGHT_GAP:.0e})"
    )
    order_m = [int(i) for i in info["order"]]
    assert order_m == order_r, (
        f"case {case}: HRP leaf order differs from HRPOpt. Tie diagnosis — smallest gap "
        f"between sorted linkage heights: maplab {gap_m:.3e}, HRPOpt {gap_r:.3e}; "
        f"max|dZ_height|={np.max(np.abs(info['Z'][:, 2] - Z_r[:, 2])):.3e}\n"
        f"maplab order {order_m}\nHRPOpt order {order_r}"
    )
    dw = float(np.max(np.abs(w_m - w_r)))
    print(f"[HRP {case}] max|dw|={dw:.3e} min height gap maplab={gap_m:.3e} hrpopt={gap_r:.3e}")
    assert dw <= HRP_W_TOL, _report(case, w_m, w_r)


def test_registered_hrp_default_differs_from_hrpopt_on_block_case(_scipy_linkage_methods_shim):
    """The registered HRP (tree bisection, distance-of-distances) is a variant of the
    literature algorithm that HRPOpt implements; on case B they must not coincide."""
    S = sigma_b()
    w_m, _, w_r, _, _ = _hrp_pair(S, linkage="single", bisection="tree", dist_of_dist=True)
    assert np.max(np.abs(w_m - w_r)) > 1e-6
