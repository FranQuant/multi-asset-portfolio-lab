"""Paper roster: methods and CIO ensembles from Ang, Azimbayev & Kim (2026), "The
Self-Driving Portfolio" (Exhibit 3, §3.3, §3.5, §4.1), that maplab did not already have.

Batch 1 (this module):
  Methods      MinimumCorrelation (Varadi et al. 2012; SIT `min.corr`), MinCVaR
               (Rockafellar & Uryasev 2000), TailRiskParity (equal historical-ES budgets;
               Boudt, Carl & Peterson 2013 use modified ES), MaxEntropy (Shannon entropy of
               weights subject to a Sharpe floor, as in §3.3), VolTarget (weight-level
               inverse-variance scaling into a cash asset; Moreira & Muir 2017).
  Wrappers     ExcludeAssets: run any method on a sub-universe (e.g. without cash), zero
               elsewhere.
  IPS          TECap: ex-ante tracking error vs a benchmark capped by shrinking toward it.
  Ensembles    Ensemble(method=...): simple average, inverse tracking error vs a benchmark,
               inverse tracking error vs the members' centroid, per-asset trimmed mean.
  Universe     Every class takes `universe=`; `on_universe(cls, tickers)` re-targets an
               existing maplab class (EqualWeight, FixedWeight, GMV, ...) without editing it.

Conventions (CLAUDE.md): estimation on the Panel's LOG returns over `lookback` days strictly
before asof; scenario-based methods (CVaR, ES) convert them exactly to simple returns
(expm1), because portfolio returns are linear in simple returns. SLSQP with ftol=1e-12,
maxiter=1000 and analytic Jacobians; LPs use HiGHS. Fallbacks are logged and recorded in
`fallback_dates`; solver failures raise.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm, rankdata, trim_mean

from .contract import LONG_ONLY, COV_LOOKBACK, TRADING_DAYS
from .covariance import sample_cov
from .data import Panel
from .models import Strategy, MaxSharpe, _tangency_long_only

logger = logging.getLogger("maplab.roster")

SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}


def on_universe(cls: type[Strategy], universe: list[str]) -> type[Strategy]:
    """Subclass of an existing Strategy class whose `universe` is `universe`.

    e.g. `on_universe(EqualWeight, U18)()` or `on_universe(FixedWeight, U18)(w, name="60/40")`.
    Name, family and logic are inherited unchanged.
    """
    return type(cls.__name__, (cls,), {"universe": list(universe), "__module__": __name__})


class RosterStrategy(Strategy):
    """Strategy base for this module: adds `universe=` and a simple-return scenario slice."""

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 universe: list[str] | None = None):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        if universe is not None:
            self.universe = list(universe)
        self.fallback_dates: list[pd.Timestamp] = []
        self.retry_dates: list[pd.Timestamp] = []

    def _scenarios(self, panel: Panel, asof: pd.Timestamp) -> np.ndarray:
        """T x n SIMPLE daily returns over the lookback window strictly before asof."""
        logr = panel.slice(asof, "returns", self.lookback)[self.universe]
        if len(logr) != self.lookback or logr.isna().any().any():
            raise ValueError(
                f"{self.label}: insufficient/NaN return history for asof={asof} "
                f"(got {len(logr)} rows, need {self.lookback})"
            )
        return np.expm1(logr.to_numpy())


# ── C. Minimum correlation ──────────────────────────────────────────────────

def _min_corr_weights(Sigma: pd.DataFrame, power: float = 1.0) -> np.ndarray:
    """Minimum Correlation Algorithm (Varadi, Kapler, Bee & Rittenhouse 2012), transliterated
    from the Systematic Investor Toolbox `min.corr` (SIT/R/strategy.r):

    1. off-diagonal correlations c_ij -> a_ij = 1 - Phi((c_ij - mean c) / sd c)   (diag 0)
    2. row means of a over j != i; rank them so the largest (least correlated) gets rank 1;
       rank weights r = rank^power / sum
    3. combined = r' A, normalised; 4. divide by volatility, normalise.
    n = 2 gives inverse volatility (SIT special case). If all pairwise correlations are equal
    (sd = 0) step 1 is undefined; every asset is then equally (de)correlated and the result is
    inverse volatility.
    """
    S = Sigma.to_numpy()
    n = S.shape[0]
    vol = np.sqrt(np.diag(S))
    if n == 1:
        return np.ones(1)
    if n == 2:
        return (1 / vol) / np.sum(1 / vol)
    C = S / np.outer(vol, vol)
    iu = np.triu_indices(n, 1)
    c = C[iu]
    sd = c.std(ddof=1)
    if sd == 0:
        return (1 / vol) / np.sum(1 / vol)
    A = np.zeros((n, n))
    A[iu] = 1.0 - norm.cdf(c, loc=c.mean(), scale=sd)
    A = A + A.T
    avg = A.sum(axis=1) / (n - 1)
    rank = rankdata(-avg) ** power          # R rank(): ties averaged, as rankdata default
    combined = (rank / rank.sum()) @ A
    x = (combined / combined.sum()) / vol
    return x / x.sum()


class MinimumCorrelation(RosterStrategy):
    """Varadi et al. (2012) Minimum Correlation Algorithm (SIT `min.corr`, power 1)."""

    name = "MinCorr"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 universe=None, power: float = 1.0):
        super().__init__(cov_estimator, lookback, universe)
        self.power = power

    def predict_weights(self, panel, asof):
        _, Sigma = self._estimate(panel, asof)
        return pd.Series(_min_corr_weights(Sigma, self.power), index=Sigma.columns)


# ── D. CVaR minimisation and tail-risk parity ───────────────────────────────

def _min_cvar_long_only(R: np.ndarray, beta: float, label: str, asof) -> tuple[np.ndarray, float]:
    """Long-only minimum CVaR_beta of the loss -R w (Rockafellar & Uryasev 2000) as an LP:

        min  a + 1/((1-beta) T) sum_t u_t
        s.t. u_t >= -r_t'w - a,  u >= 0,  sum w = 1,  0 <= w <= 1,  a free.
    Returns (w, CVaR) with CVaR as a positive daily loss.
    """
    T, n = R.shape
    k = 1.0 / ((1.0 - beta) * T)
    cost = np.concatenate([np.zeros(n), [1.0], np.full(T, k)])
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
    bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T
    res = opt.linprog(cost, A_ub=A_ub, b_ub=np.zeros(T), A_eq=A_eq, b_eq=[1.0],
                      bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"{label}: HiGHS failed at asof={asof}: {res.message}")
    w = np.clip(res.x[:n], 0.0, None)
    return w / w.sum(), float(res.fun)


def historical_cvar(R: np.ndarray, w: np.ndarray, beta: float) -> float:
    """CVaR_beta of the loss -R w via the Rockafellar–Uryasev formula, minimised over a
    (exact: the minimiser is a beta-quantile of the losses, so scanning the observed losses
    suffices)."""
    loss = -R @ w
    T = len(loss)
    k = 1.0 / ((1.0 - beta) * T)
    return float(min(a + k * np.clip(loss - a, 0.0, None).sum() for a in loss))


class MinCVaR(RosterStrategy):
    """Long-only minimum historical CVaR at level beta (default 95%) on simple returns."""

    name = "MinCVaR"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 universe=None, beta: float = 0.95):
        super().__init__(cov_estimator, lookback, universe)
        self.beta = beta

    def predict_weights(self, panel, asof):
        R = self._scenarios(panel, asof)
        w, _ = _min_cvar_long_only(R, self.beta, self.label, asof)
        return pd.Series(w, index=self.universe)


def _es_parity_long_only(R: np.ndarray, beta: float, label: str, asof) -> np.ndarray:
    """Equal historical expected-shortfall budgets.

    ES_beta(y) (Rockafellar–Uryasev) is convex and positively homogeneous, so the minimiser of
        ES_beta(y) - (1/n) sum_i log y_i,   y > 0
    satisfies y_i * g_i = 1/n for some subgradient g of ES at y (Euler with equal budgets;
    total ES = 1); w = y / sum(y). The barrier makes the problem strictly convex, so w is
    unique. Historical ES is piecewise linear, so equality holds through a subgradient, i.e.
    for some admissible tail weighting of the scenarios at the VaR kink, not for one fixed
    tail set (see `es_parity_kkt_residual`). Solved with SLSQP on the RU epigraph form.

    The problem is unbounded if some asset never loses in the window (its ES contribution
    is always negative: more of it always lowers ES) — e.g. T-bills. Such assets must be
    excluded by the caller; this raises if one is present.
    """
    T, n = R.shape
    never_loses = np.where((R >= 0).all(axis=0))[0]
    if len(never_loses):
        raise ValueError(f"{label}: asset column(s) {list(never_loses)} never lose in the window at "
                         f"asof={asof}; ES parity is unbounded — exclude them (e.g. cash)")
    k = 1.0 / ((1.0 - beta) * T)
    # start: inverse-vol y scaled so ES(y) ~ 1, a = VaR, u = excess losses
    vol = R.std(axis=0, ddof=1)
    y0 = (1 / vol) / np.sum(1 / vol)
    es0 = historical_cvar(R, y0, beta)
    y0 = y0 / es0 if es0 > 0 else y0
    loss0 = -R @ y0
    a0 = np.quantile(loss0, beta)
    x0 = np.concatenate([y0, [a0], np.clip(loss0 - a0, 0.0, None)])

    def f(x):
        return x[n] + k * x[n + 1:].sum() - np.log(x[:n]).sum() / n

    def jac(x):
        g = np.empty_like(x)
        g[:n] = -1.0 / (n * x[:n])
        g[n] = 1.0
        g[n + 1:] = k
        return g

    # u_t + a + r_t'y >= 0
    G = np.hstack([R, np.ones((T, 1)), np.eye(T)])
    cons = [{"type": "ineq", "fun": lambda x: G @ x, "jac": lambda x: G}]
    bounds = [(1e-12, None)] * n + [(None, None)] + [(0.0, None)] * T
    res = opt.minimize(f, x0, jac=jac, method="SLSQP", bounds=bounds,
                       constraints=cons, options=SLSQP_OPTIONS)
    if not res.success:
        raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message}")
    y = res.x[:n]
    return y / y.sum()


def es_parity_kkt_residual(R: np.ndarray, w: np.ndarray, beta: float, tie_tol: float = 1e-7) -> float:
    """Optimality certificate for `_es_parity_long_only`: the smallest max-abs deviation, over
    admissible tail weightings p (RU subgradients: p_t = 1/((1-beta)T) for losses above VaR,
    0 below, anything in between at the kink, sum 1), of the ES budgets w_i (-R'p)_i / ES
    from 1/n. Solved as an LP; ~0 at the optimum. Losses within `tie_tol` (relative to ES)
    of VaR count as tied.
    """
    T, n = R.shape
    k = 1.0 / ((1.0 - beta) * T)
    loss = -R @ w
    es = historical_cvar(R, w, beta)
    var = np.sort(loss)[::-1][int(np.floor((1 - beta) * T))] if (1 - beta) * T < T else loss.min()
    # VaR = the RU minimiser a*: any loss quantile at level beta; use the one RU's scan picks
    cands = [a for a in loss if np.isclose(a + k * np.clip(loss - a, 0, None).sum(), es, rtol=0, atol=1e-15 + 1e-12 * abs(es))]
    lo, hi = min(cands), max(cands)
    tol = tie_tol * abs(es)
    ub = np.where(loss > hi + tol, k, np.where(loss < lo - tol, 0.0, k))
    lb = np.where(loss > hi + tol, k, 0.0)
    # variables: p (T), s (1); minimise s s.t. |w_i (-R'p)_i / es - 1/n| <= s
    M = (-(R * w).T) / es                       # n x T
    c = np.concatenate([np.zeros(T), [1.0]])
    A_ub = np.vstack([np.hstack([M, -np.ones((n, 1))]), np.hstack([-M, -np.ones((n, 1))])])
    b_ub = np.concatenate([np.full(n, 1.0 / n), np.full(n, -1.0 / n)])
    A_eq = np.concatenate([np.ones(T), [0.0]])[None, :]
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=list(zip(lb, ub)) + [(0, None)], method="highs")
    if res.status != 0:
        raise RuntimeError(f"es_parity_kkt_residual: LP failed: {res.message}")
    return float(res.fun)


def es_contributions(R: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    """Euler contributions to historical ES_beta: w_i times minus the mean of asset i's return
    over the tail scenarios, the tail weighted as in the RU formula (the worst floor((1-beta)T)
    losses with weight 1 and the next one with the fractional remainder). Sums to ES."""
    loss = -R @ w
    T = len(loss)
    m = (1.0 - beta) * T
    order = np.argsort(-loss)
    p = np.zeros(T)
    full = int(np.floor(m))
    p[order[:full]] = 1.0
    if full < T and m > full:
        p[order[full]] = m - full
    p = p / m
    return w * (-(p @ R))


class TailRiskParity(RosterStrategy):
    """Equal budgets of historical ES at level beta (default 95%) over the non-excluded assets;
    `exclude` (e.g. the cash asset, whose ES contribution is always negative) gets weight 0.

    Boudt, Carl & Peterson (2013) budget modified (Cornish–Fisher) ES; this implementation
    uses historical ES on the same lookback window — a disclosed simplification.
    """

    name = "TailRP"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 universe=None, beta: float = 0.95, exclude: list[str] | tuple = ()):
        super().__init__(cov_estimator, lookback, universe)
        unknown = [t for t in exclude if t not in self.universe]
        if unknown:
            raise ValueError(f"TailRiskParity: excluded ticker(s) not in universe: {unknown}")
        self.beta, self.exclude = beta, list(exclude)

    def predict_weights(self, panel, asof):
        R = self._scenarios(panel, asof)
        keep = [i for i, t in enumerate(self.universe) if t not in self.exclude]
        w = np.zeros(len(self.universe))
        w[keep] = _es_parity_long_only(R[:, keep], self.beta, self.label, asof)
        return pd.Series(w, index=self.universe)


# ── E. Maximum entropy with a Sharpe floor ──────────────────────────────────

def _max_entropy_sharpe_floor(excess: np.ndarray, Sigma: np.ndarray, floor: float,
                              label: str, asof, x_retry: np.ndarray | None = None,
                              on_retry=None) -> np.ndarray:
    """max -sum w log w  s.t. sum w = 1, w >= 0, w'excess / sqrt(w'Sigma w) >= floor.

    For floor >= 0 the Sharpe constraint is convex (a concave function >= 0), so the problem
    is convex. If equal weights satisfy the floor they are the optimum. SLSQP starts at equal
    weights; on failure it logs, calls `on_retry`, and retries once from `x_retry` (a feasible
    point, e.g. the tangency portfolio), raising if that also fails — as `_tangency_long_only`.
    """
    n = len(excess)
    ew = np.full(n, 1.0 / n)

    # Constraint as a Sharpe deficit SR(w) - floor >= 0: the same convex feasible set
    # (a superlevel set of the quasi-concave Sharpe ratio), but O(1)-scaled, which SLSQP
    # needs — in return units (~1e-3) it stalls a hair outside the boundary.
    def g(w):
        return (w @ excess) / np.sqrt(w @ Sigma @ w) - floor

    if g(ew) >= 0:
        return ew

    def g_jac(w):
        s = np.sqrt(w @ Sigma @ w)
        return excess / s - (w @ excess) * (Sigma @ w) / s ** 3

    ones = np.ones(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones},
            {"type": "ineq", "fun": g, "jac": g_jac}]
    bounds = [(1e-12, 1.0)] * n

    def solve(x0):
        return opt.minimize(lambda w: np.sum(w * np.log(w)), x0, jac=lambda w: np.log(w) + 1.0,
                            method="SLSQP", bounds=bounds, constraints=cons, options=SLSQP_OPTIONS)

    res = solve(ew)
    if not res.success or g(res.x) < -1e-9:
        if x_retry is None:
            raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message} (g={g(res.x):.3e})")
        logger.warning("%s: SLSQP failed at asof=%s (%s); retrying from the feasible start",
                       label, asof, res.message)
        if on_retry is not None:
            on_retry()
        x0 = np.clip(x_retry, 1e-12, None)
        res = solve(x0 / x0.sum())
        if not res.success or g(res.x) < -1e-9:
            raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message} (g={g(res.x):.3e})")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


class MaxEntropy(RosterStrategy):
    """Maximum Shannon entropy of weights subject to Sharpe >= floor_frac * SR(max-Sharpe)
    (paper §3.3; floor_frac is not given in the paper — default 0.75, the adversarial
    diversifier's floor). Expected returns: maplab's historical mu (log mean + var/2) and rf
    exactly as MaxSharpe. If no asset has positive expected excess return the floor is
    undefined: equal weights (the unconstrained maximum), logged in fallback_dates.
    """

    name = "MaxEntropy"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 universe=None, rf: float | str = "panel", floor_frac: float = 0.75):
        super().__init__(cov_estimator, lookback, universe)
        self.rf = rf
        self.floor_frac = floor_frac
        self._ms = MaxSharpe(cov_estimator=cov_estimator, lookback=lookback, rf=rf)

    def predict_weights(self, panel, asof):
        mu, Sigma = self._estimate(panel, asof)
        excess = (mu - self._ms._rf_ann(panel, asof)).to_numpy()
        S = Sigma.to_numpy()
        n = len(excess)
        if excess.max() <= 0:
            logger.warning("%s: no positive expected excess return at asof=%s; equal weights",
                           self.label, asof)
            self.fallback_dates.append(asof)
            return pd.Series(1.0 / n, index=Sigma.columns)
        w_t, _ = _tangency_long_only(excess, S, self.label, asof,
                                     on_retry=lambda: self.retry_dates.append(asof))
        sr_max = (w_t @ excess) / np.sqrt(w_t @ S @ w_t)
        w = _max_entropy_sharpe_floor(excess, S, self.floor_frac * sr_max, self.label, asof,
                                      x_retry=w_t, on_retry=lambda: self.retry_dates.append(asof))
        return pd.Series(w, index=Sigma.columns)


# ── A. Volatility targeting (weight level) ──────────────────────────────────

class VolTarget(RosterStrategy):
    """Risky base portfolio scaled by trailing variance, remainder in a cash asset
    (Moreira & Muir 2017, "Volatility-Managed Portfolios", eq. 1: weight c / sigma_hat^2).

    sigma_hat^2 = b' Sigma_w b, Sigma_w from the last `vol_window` days (default 22: their
    previous-month realised variance); b = `base` weights over the non-cash assets (default
    equal weight), renormalised without cash. Exposure
        "inverse_variance" (default, M&M):  e = min(cap, target_vol^2 / sigma_hat^2)
        "inverse_vol":                      e = min(cap, target_vol / sigma_hat)
    w = e b + (1 - e) e_cash. M&M choose c so the managed portfolio's full-sample vol matches
    buy-and-hold — look-ahead in a backtest; here c = target_vol^2 is fixed ex ante (the paper's
    IPS band midpoint, 10%). cap = 1 (no leverage; M&M's constrained variant) keeps it
    long-only. Unlike maplab's `vol_overlay` (a return-stream overlay), this produces weights,
    so it can enter ensembles. The cash asset must be in `universe`.
    """

    name = "VolTarget"
    family = "Overlay"
    constraint = LONG_ONLY
    SCALINGS = ("inverse_variance", "inverse_vol")

    def __init__(self, cov_estimator=sample_cov, universe=None, *, cash: str,
                 base: dict[str, float] | None = None, target_vol: float = 0.10,
                 vol_window: int = 22, cap: float = 1.0, scaling: str = "inverse_variance"):
        super().__init__(cov_estimator, vol_window, universe)
        if cash not in self.universe:
            raise ValueError(f"VolTarget: cash asset {cash!r} not in universe")
        if not 0 < cap <= 1:
            raise ValueError("VolTarget: cap must be in (0, 1] (long-only)")
        if scaling not in self.SCALINGS:
            raise ValueError(f"VolTarget: scaling must be one of {self.SCALINGS}")
        risky = [t for t in self.universe if t != cash]
        if base is None:
            b = pd.Series(1.0, index=risky)
        else:
            unknown = [t for t in base if t not in self.universe]
            if unknown or min(base.values()) < 0:
                raise ValueError(f"VolTarget: base must be non-negative over the universe (unknown: {unknown})")
            b = pd.Series(base, dtype=float).reindex(risky, fill_value=0.0)
        if b.sum() <= 0:
            raise ValueError("VolTarget: base has no risky weight")
        self.cash, self.target_vol, self.cap, self.scaling = cash, target_vol, cap, scaling
        self.b = b / b.sum()
        self.exposure: dict[pd.Timestamp, float] = {}

    def predict_weights(self, panel, asof):
        _, Sigma = self._estimate(panel, asof)
        risky = list(self.b.index)
        var = float(self.b @ Sigma.loc[risky, risky] @ self.b)
        if var <= 0:
            e = self.cap
        elif self.scaling == "inverse_variance":
            e = min(self.cap, self.target_vol ** 2 / var)
        else:
            e = min(self.cap, self.target_vol / np.sqrt(var))
        self.exposure[asof] = e
        w = (e * self.b).reindex(self.universe, fill_value=0.0)
        w[self.cash] += 1.0 - e
        return w


# ── Sub-universe wrapper ────────────────────────────────────────────────────

class ExcludeAssets(RosterStrategy):
    """Run `base` on its own (smaller) universe and give every other asset of `universe` zero
    weight — e.g. risk-only methods on the risky classes, cash excluded:
        ExcludeAssets(on_universe(GMV, RISKY)(), universe=U18)
    Label, name and family are the base's; `excluded` lists the zero-weight assets.
    """

    constraint = LONG_ONLY

    def __init__(self, base: Strategy, universe: list[str]):
        missing = [t for t in base.universe if t not in universe]
        if missing:
            raise ValueError(f"ExcludeAssets: base universe has ticker(s) outside universe: {missing}")
        super().__init__(base.cov_estimator, base.lookback, universe)
        self.base = base
        self.name, self.family = base.label, base.family
        self.excluded = [t for t in self.universe if t not in base.universe]

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel, asof):
        return self.base(panel, asof).reindex(self.universe, fill_value=0.0)


# ── IPS: tracking-error cap ─────────────────────────────────────────────────

def tracking_error(w: np.ndarray, b: np.ndarray, Sigma: np.ndarray) -> float:
    d = w - b
    return float(np.sqrt(max(d @ Sigma @ d, 0.0)))


class TECap(RosterStrategy):
    """Ex-ante tracking error vs `benchmark` capped at `cap` (paper IPS: 6% vs 60/40).

    w_capped = b + lam (w - b), lam = min(1, cap / TE(w)): the smallest move toward the
    benchmark along the active direction that meets the cap. It applies the same rule to any
    method, preserves long-only and full investment, and leaves compliant portfolios
    unchanged. Sigma comes from the wrapper's own estimator and lookback. Records lam and the
    pre-cap TE per asof.
    """

    constraint = LONG_ONLY

    def __init__(self, base: Strategy, benchmark: dict[str, float], cap: float = 0.06,
                 cov_estimator=sample_cov, lookback: int = COV_LOOKBACK):
        super().__init__(cov_estimator, lookback, base.universe)
        unknown = [t for t in benchmark if t not in self.universe]
        if unknown:
            raise ValueError(f"TECap: benchmark ticker(s) not in universe: {unknown}")
        if abs(sum(benchmark.values()) - 1.0) > 1e-12 or min(benchmark.values()) < 0:
            raise ValueError("TECap: benchmark weights must be non-negative and sum to 1")
        self.base, self.cap = base, cap
        self.b = pd.Series(benchmark, dtype=float).reindex(self.universe, fill_value=0.0)
        self.name = f"TE≤{cap:.0%}[{base.label}]"
        self.family = base.family
        self.te_before: dict[pd.Timestamp, float] = {}
        self.lam: dict[pd.Timestamp, float] = {}

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel, asof):
        w = self.base(panel, asof).reindex(self.universe)
        _, Sigma = self._estimate(panel, asof)
        te = tracking_error(w.to_numpy(), self.b.to_numpy(), Sigma.loc[self.universe, self.universe].to_numpy())
        lam = 1.0 if te <= self.cap else self.cap / te
        self.te_before[asof], self.lam[asof] = te, lam
        return self.b + lam * (w - self.b)


# ── CIO ensembles ───────────────────────────────────────────────────────────

ENSEMBLE_METHODS = ("average", "inverse_te_benchmark", "inverse_te_centroid", "trimmed_mean")


class Ensemble(RosterStrategy):
    """Combine the target weights of `members` at each rebalance (paper §3.5, Exhibit 8).

    average               mean of member weights
    inverse_te_benchmark  member weight ∝ 1 / TE(w_k vs benchmark)  (the third-party
                          re-implementation's definition)
    inverse_te_centroid   member weight ∝ 1 / TE(w_k vs centroid = mean of members)  (the
                          paper's text: "closer to the ensemble centroid")
    trimmed_mean          per-asset mean after dropping the `trim` fraction of member weights
                          from each tail (scipy.stats.trim_mean), renormalised to sum 1

    TE uses the ensemble's own Sigma (estimator, lookback). A TE below `te_floor` is floored
    there (a member sitting on the reference would otherwise take infinite weight) and the
    date recorded in fallback_dates. Member weights and the member mix are logged per asof.
    Members are called fresh at every asof; pass fresh instances.
    """

    family = "Ensemble"
    constraint = LONG_ONLY

    def __init__(self, members: list[Strategy], method: str, *, benchmark: dict[str, float] | None = None,
                 trim: float = 0.10, te_floor: float = 1e-4,
                 cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, name: str | None = None):
        if method not in ENSEMBLE_METHODS:
            raise ValueError(f"Ensemble: unknown method {method!r}; choose from {ENSEMBLE_METHODS}")
        if not members:
            raise ValueError("Ensemble: no members")
        universe = list(members[0].universe)
        if any(list(m.universe) != universe for m in members):
            raise ValueError("Ensemble: members must share one universe (same order)")
        if method == "inverse_te_benchmark" and benchmark is None:
            raise ValueError("Ensemble: inverse_te_benchmark needs a benchmark")
        super().__init__(cov_estimator, lookback, universe)
        self.members, self.method, self.trim, self.te_floor = members, method, trim, te_floor
        self.b = None if benchmark is None else pd.Series(benchmark, dtype=float).reindex(universe, fill_value=0.0)
        self.name = name or f"Ens[{method}]"
        self.member_weights: dict[pd.Timestamp, pd.DataFrame] = {}
        self.mix: dict[pd.Timestamp, pd.Series] = {}

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel, asof):
        W = pd.DataFrame({m.label: m(panel, asof).reindex(self.universe) for m in self.members}).T
        if W.index.duplicated().any():
            raise ValueError(f"{self.label}: duplicate member labels {list(W.index[W.index.duplicated()])}")
        self.member_weights[asof] = W
        Wn = W.to_numpy()
        K = len(W)
        if self.method == "average":
            mix = np.full(K, 1.0 / K)
        elif self.method == "trimmed_mean":
            w = trim_mean(Wn, proportiontocut=self.trim, axis=0)
            self.mix[asof] = pd.Series(np.nan, index=W.index)
            return pd.Series(w / w.sum(), index=self.universe)
        else:
            _, Sigma = self._estimate(panel, asof)
            S = Sigma.loc[self.universe, self.universe].to_numpy()
            ref = self.b.to_numpy() if self.method == "inverse_te_benchmark" else Wn.mean(axis=0)
            te = np.array([tracking_error(Wn[k], ref, S) for k in range(K)])
            if (te < self.te_floor).any():
                logger.warning("%s: TE below floor for %s at asof=%s", self.label,
                               list(W.index[te < self.te_floor]), asof)
                self.fallback_dates.append(asof)
                te = np.maximum(te, self.te_floor)
            mix = (1 / te) / np.sum(1 / te)
        self.mix[asof] = pd.Series(mix, index=W.index)
        return pd.Series(mix @ Wn, index=self.universe)


# ════════════════════════════════════════════════════════════════════════════
# Batch 2 — remaining paper methods, trailing-performance ensembles, quarterly
# ════════════════════════════════════════════════════════════════════════════

from .models import _min_variance_long_only  # noqa: E402


class _ExcessMixin:
    """(excess mu, Sigma, rf) with maplab's mu and MaxSharpe's rf convention."""

    def _init_rf(self, rf):
        self.rf = rf
        self._ms = MaxSharpe(cov_estimator=self.cov_estimator, lookback=self.lookback, rf=rf)

    def _excess(self, panel, asof):
        mu, Sigma = self._estimate(panel, asof)
        rf_ann = self._ms._rf_ann(panel, asof)
        return (mu - rf_ann).to_numpy(), Sigma, rf_ann

    def _gmv_fallback(self, Sigma, asof, why):
        logger.warning("%s: %s at asof=%s; falling back to GMV", self.label, why, asof)
        self.fallback_dates.append(asof)
        return pd.Series(_min_variance_long_only(Sigma), index=Sigma.columns)


# ── B. Robust mean-variance (worst-case Sharpe) ─────────────────────────────

def _robust_tangency(excess: np.ndarray, Sigma: np.ndarray, theta_sd: np.ndarray, kappa: float,
                     label: str, asof) -> np.ndarray:
    """Long-only max worst-case Sharpe over the box-free ellipsoid
    {mu : ||(mu - mu_hat) / theta_sd|| <= kappa} (diagonal Theta):
        max_w (excess'w - kappa ||theta_sd * w||) / sqrt(w'Sigma w)
    Homogeneous of degree 0, so (Goldfarb & Iyengar 2003 style):
        min y'Sigma y  s.t.  excess'y - kappa ||theta_sd * y|| >= 1,  y >= 0;  w = y / sum y.
    Caller checks feasibility: sqrt(sum_i max(excess_i,0)^2 / theta_sd_i^2) > kappa.
    """
    n = len(excess)
    pos = np.clip(excess, 0.0, None)
    v = pos / theta_sd ** 2                       # maximiser of excess'w / ||theta_sd w|| over w >= 0
    y0 = v / (excess @ v - kappa * np.linalg.norm(theta_sd * v)) * 1.001

    def g(y):
        return excess @ y - kappa * np.linalg.norm(theta_sd * y) - 1.0

    def g_jac(y):
        return excess - kappa * (theta_sd ** 2 * y) / np.linalg.norm(theta_sd * y)

    res = opt.minimize(lambda y: y @ Sigma @ y, y0, jac=lambda y: 2 * Sigma @ y, method="SLSQP",
                       bounds=[(0.0, None)] * n, constraints=[{"type": "ineq", "fun": g, "jac": g_jac}],
                       options=SLSQP_OPTIONS)
    if not res.success or g(res.x) < -1e-9:
        raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message}")
    y = np.clip(res.x, 0.0, None)
    return y / y.sum()


class RobustMaxSharpe(_ExcessMixin, RosterStrategy):
    """Max worst-case Sharpe with expected returns in a kappa-standard-error ellipsoid around
    maplab's historical mu. Theta = diag(Sigma) * TRADING_DAYS / lookback (the sampling variance
    of each annualised mean, assets taken separately). With Theta proportional to Sigma the
    worst-case Sharpe is SR - kappa and the method would equal MaxSharpe, hence the diagonal.
    Infeasible (no portfolio with positive worst-case excess) -> GMV, logged.
    """

    name = "RobustMV"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, universe=None,
                 rf: float | str = "panel", kappa: float = 1.0):
        super().__init__(cov_estimator, lookback, universe)
        self._init_rf(rf)
        self.kappa = kappa

    def predict_weights(self, panel, asof):
        excess, Sigma, _ = self._excess(panel, asof)
        S = Sigma.to_numpy()
        theta_sd = np.sqrt(np.diag(S) * TRADING_DAYS / self.lookback)
        if np.sqrt(np.sum((np.clip(excess, 0, None) / theta_sd) ** 2)) <= self.kappa:
            return self._gmv_fallback(Sigma, asof, f"no portfolio has positive worst-case excess (kappa={self.kappa})")
        return pd.Series(_robust_tangency(excess, S, theta_sd, self.kappa, self.label, asof), index=Sigma.columns)


# ── B. Resampled efficient frontier (Michaud) ───────────────────────────────

class ResampledEF(_ExcessMixin, RosterStrategy):
    """Michaud (1998) resampling, max-Sharpe variant: draw `n_draws` samples of `lookback` daily
    log returns from N(mean, cov) of the window, re-estimate (mu, Sigma) with maplab's
    conventions, solve long-only max Sharpe (GMV if a draw has no positive excess), average.
    The RNG is seeded from (seed, asof) so a weight never depends on call order.
    """

    name = "ResampledEF"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, universe=None,
                 rf: float | str = "panel", n_draws: int = 500, seed: int = 20261019):
        super().__init__(cov_estimator, lookback, universe)
        self._init_rf(rf)
        self.n_draws, self.seed = n_draws, seed
        self.degenerate_draws: dict[pd.Timestamp, int] = {}

    def predict_weights(self, panel, asof):
        logr = panel.slice(asof, "returns", self.lookback)[self.universe]
        if len(logr) != self.lookback or logr.isna().any().any():
            raise ValueError(f"{self.label}: insufficient/NaN return history for asof={asof}")
        rf_ann = self._ms._rf_ann(panel, asof)
        m, C = logr.mean().to_numpy(), logr.cov().to_numpy()
        vals, vecs = np.linalg.eigh(C)
        L = vecs * np.sqrt(np.clip(vals, 0.0, None))
        rng = np.random.default_rng([self.seed, pd.Timestamp(asof).toordinal()])
        acc, degenerate = np.zeros(len(self.universe)), 0
        for _ in range(self.n_draws):
            X = pd.DataFrame(m + rng.standard_normal((self.lookback, len(m))) @ L.T, columns=self.universe)
            Sig = self.cov_estimator(X)
            mu = X.mean() * TRADING_DAYS + 0.5 * np.diag(Sig.to_numpy())
            ex = (mu - rf_ann).to_numpy()
            if ex.max() <= 0:
                degenerate += 1
                acc += _min_variance_long_only(Sig)
            else:
                w, _ = _tangency_long_only(ex, Sig.to_numpy(), self.label, asof)
                acc += w
        self.degenerate_draws[asof] = degenerate
        w = acc / self.n_draws
        return pd.Series(w / w.sum(), index=self.universe)


# ── B. Mean-downside (Sortino) ──────────────────────────────────────────────

def _max_sortino(excess: np.ndarray, X: np.ndarray, label: str, asof) -> np.ndarray:
    """Long-only max excess'w / DD(w), DD = sqrt(mean(min(X w, 0)^2)) on daily excess returns X
    (minimum acceptable return = rf). Homogeneous of degree 0, so
        min mean(min(X y, 0)^2)  s.t.  excess'y = 1, y >= 0;  w = y / sum y   (convex, C1).
    """
    n, T = len(excess), len(X)
    i0 = int(np.argmax(excess))
    y0 = np.zeros(n); y0[i0] = 1.0 / excess[i0]
    # Scale the objective to O(1) at a reference point: daily downside variances are ~1e-5,
    # where SLSQP's ftol stops ~1e-3 (relative) short of the optimum.
    pos = np.clip(excess, 0.0, None)
    y_ref = pos / (excess @ pos)
    d_ref = np.minimum(X @ y_ref, 0.0)
    scale = T / (d_ref @ d_ref) if d_ref @ d_ref > 0 else float(T)

    def f(y):
        d = np.minimum(X @ y, 0.0)
        return scale * (d @ d) / T

    def jac(y):
        return 2.0 * scale / T * X.T @ np.minimum(X @ y, 0.0)

    res = opt.minimize(f, y0, jac=jac, method="SLSQP", bounds=[(0.0, None)] * n,
                       constraints=[{"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess}],
                       options=SLSQP_OPTIONS)
    if not res.success:
        res = opt.minimize(f, y_ref, jac=jac, method="SLSQP", bounds=[(0.0, None)] * n,
                           constraints=[{"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess}],
                           options=SLSQP_OPTIONS)
        if not res.success:
            raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message}")
    y = np.clip(res.x, 0.0, None)
    return y / y.sum()


class MaxSortino(_ExcessMixin, RosterStrategy):
    """Mean-downside (Sortino & van der Meer 1991): max (mu - rf) / downside deviation below rf,
    mu = maplab's historical mu, downside on the window's simple daily returns minus rf/252.
    No positive expected excess -> GMV, logged."""

    name = "Sortino"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, universe=None,
                 rf: float | str = "panel"):
        super().__init__(cov_estimator, lookback, universe)
        self._init_rf(rf)

    def predict_weights(self, panel, asof):
        excess, Sigma, rf_ann = self._excess(panel, asof)
        if excess.max() <= 0:
            return self._gmv_fallback(Sigma, asof, "no positive expected excess return")
        X = self._scenarios(panel, asof) - rf_ann / TRADING_DAYS
        return pd.Series(_max_sortino(excess, X, self.label, asof), index=self.universe)


# ── D. Max-drawdown-constrained (Chekhlov, Uryasev & Zabarankin) ────────────

def max_drawdown_uncompounded(R: np.ndarray, w: np.ndarray) -> float:
    """Largest drop of the uncompounded cumulative return path (CUZ 2005), as a positive number,
    measured from the start (peak >= 0)."""
    c = np.cumsum(R @ w)
    peak = np.maximum.accumulate(np.maximum(c, 0.0))
    return float(np.max(peak - c))


def _max_return_dd_limit(R: np.ndarray, limit: float, label: str, asof) -> np.ndarray:
    """Long-only max mean return s.t. max drawdown of the uncompounded path <= limit (LP):
        c_t = C_t w (C = cumulative returns),  u_t >= c_t,  u_t >= u_{t-1},  u_t >= 0,  u_t - c_t <= limit.
    """
    T, n = R.shape
    C = np.cumsum(R, axis=0)
    cost = np.concatenate([-R.mean(axis=0), np.zeros(T)])
    I = np.eye(T)
    D = np.eye(T) - np.eye(T, k=-1)                       # (u_t - u_{t-1}); first row u_0
    A_ub = np.vstack([np.hstack([C, -I]),                  # c_t - u_t <= 0
                      np.hstack([np.zeros((T, n)), -D]),   # u_{t-1} - u_t <= 0 (row 0: -u_0 <= 0)
                      np.hstack([-C, I])])                 # u_t - c_t <= limit
    b_ub = np.concatenate([np.zeros(T), np.zeros(T), np.full(T, limit)])
    A_eq = np.concatenate([np.ones(n), np.zeros(T)])[None, :]
    res = opt.linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0.0, 1.0)] * n + [(0.0, None)] * T, method="highs")
    if res.status != 0:
        raise RuntimeError(f"{label}: HiGHS failed at asof={asof}: {res.message}")
    w = np.clip(res.x[:n], 0.0, None)
    return w / w.sum()


class MaxDDConstrained(RosterStrategy):
    """Max historical mean return s.t. the window's uncompounded max drawdown <= `limit`
    (default 0.25, the paper's IPS soft drawdown limit). Infeasible (every portfolio breaches)
    -> min-drawdown portfolio (LP with the limit as objective), logged."""

    name = "MaxDD"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, universe=None,
                 limit: float = 0.25):
        super().__init__(cov_estimator, lookback, universe)
        self.limit = limit

    def predict_weights(self, panel, asof):
        R = self._scenarios(panel, asof)
        try:
            w = _max_return_dd_limit(R, self.limit, self.label, asof)
        except RuntimeError:
            logger.warning("%s: drawdown limit %.2f infeasible at asof=%s; min-drawdown portfolio",
                           self.label, self.limit, asof)
            self.fallback_dates.append(asof)
            w = _min_drawdown(R, self.label, asof)
        return pd.Series(w, index=self.universe)


def _min_drawdown(R: np.ndarray, label: str, asof) -> np.ndarray:
    T, n = R.shape
    C = np.cumsum(R, axis=0)
    I = np.eye(T)
    D = np.eye(T) - np.eye(T, k=-1)
    # vars w, u, z: min z s.t. c - u <= 0, u_{t-1} - u_t <= 0, u - c - z <= 0
    cost = np.concatenate([np.zeros(n + T), [1.0]])
    A_ub = np.vstack([np.hstack([C, -I, np.zeros((T, 1))]),
                      np.hstack([np.zeros((T, n)), -D, np.zeros((T, 1))]),
                      np.hstack([-C, I, -np.ones((T, 1))])])
    A_eq = np.concatenate([np.ones(n), np.zeros(T + 1)])[None, :]
    res = opt.linprog(cost, A_ub=A_ub, b_ub=np.zeros(3 * T), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0.0, 1.0)] * n + [(0.0, None)] * (T + 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"{label}: HiGHS failed at asof={asof}: {res.message}")
    w = np.clip(res.x[:n], 0.0, None)
    return w / w.sum()


# ── D. Total Portfolio Approach, two factors ────────────────────────────────

class TPATwoFactor(_ExcessMixin, RosterStrategy):
    """Deterministic two-factor TPA (after Ang, Brandt & Denison 2014; this construction is ours).

    Regress each class's daily excess return on an intercept and two factor excess returns
    (default US_LC = equity, UST_INT = bonds) over the window. The reference portfolio fixes the
    factor exposures (default 0.6 equity, 0.4 bonds); within them, hold whatever earns the most
    factor-adjusted expected return (intercepts, annualised) inside an active-risk budget on the
    non-factor (residual) volatility, default the IPS's 6%:
        max alpha'w  s.t.  B'w = target, 1'w = 1, w >= 0, sqrt(w' S_res w) <= budget.
    Always feasible: target_eq * equity + target_bd * bond has exposures equal to the target and
    zero residual risk (the factor assets load only on themselves). Minimising residual risk
    alone would always return that replication, hence the return objective.
    """

    name = "TPA"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, universe=None,
                 rf: float | str = "panel", factors: tuple[str, str] = ("US_LC", "UST_INT"),
                 target: tuple[float, float] = (0.6, 0.4), residual_budget: float = 0.06):
        super().__init__(cov_estimator, lookback, universe)
        missing = [f for f in factors if f not in self.universe]
        if missing:
            raise ValueError(f"TPATwoFactor: factor asset(s) not in universe: {missing}")
        if abs(sum(target) - 1.0) > 1e-12:
            raise ValueError("TPATwoFactor: target exposures must sum to 1 (fully invested replication)")
        self._init_rf(rf)
        self.factors, self.target, self.budget = list(factors), np.asarray(target, dtype=float), residual_budget
        self.betas: dict[pd.Timestamp, pd.DataFrame] = {}
        self.alphas: dict[pd.Timestamp, pd.Series] = {}

    def factor_model(self, panel, asof):
        rf_ann = self._ms._rf_ann(panel, asof)
        X = self._scenarios(panel, asof) - rf_ann / TRADING_DAYS
        idx = [self.universe.index(f) for f in self.factors]
        F = np.column_stack([np.ones(len(X)), X[:, idx]])
        coef, *_ = np.linalg.lstsq(F, X, rcond=None)          # 3 x n
        S_res = np.cov(X - F @ coef, rowvar=False) * TRADING_DAYS
        return coef[0] * TRADING_DAYS, coef[1:].T, S_res, idx

    def predict_weights(self, panel, asof):
        alpha, B, S, idx = self.factor_model(panel, asof)
        self.betas[asof] = pd.DataFrame(B, index=self.universe, columns=self.factors)
        self.alphas[asof] = pd.Series(alpha, index=self.universe)
        n = len(self.universe)
        w0 = np.zeros(n); w0[idx] = self.target
        ones = np.ones(n)
        b2 = self.budget ** 2
        # alpha scaled to O(1) for SLSQP; the constraint is in variance units / budget^2
        a_scale = 1.0 / max(np.abs(alpha).max(), 1e-12)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones},
                {"type": "eq", "fun": lambda w: B.T @ w - self.target, "jac": lambda w: B.T},
                {"type": "ineq", "fun": lambda w: 1.0 - (w @ S @ w) / b2, "jac": lambda w: -2 * S @ w / b2}]
        res = opt.minimize(lambda w: -a_scale * (alpha @ w), w0, jac=lambda w: -a_scale * alpha,
                           method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=cons, options=SLSQP_OPTIONS)
        if not res.success:
            raise RuntimeError(f"{self.label}: SLSQP failed at asof={asof}: {res.message}")
        w = np.clip(res.x, 0.0, None)
        return pd.Series(w / w.sum(), index=self.universe)


# ── Member weights: live members or precomputed weight logs ─────────────────

def _member_matrix(owner, panel, asof) -> pd.DataFrame:
    """K x n member target weights at asof: from live `members` (called now) or from
    precomputed `member_weights` (label -> weights_log from backtest, rows = rebalance dates,
    each computed with data strictly before its date)."""
    if owner.members is not None:
        W = pd.DataFrame({m.label: m(panel, asof).reindex(owner.universe) for m in owner.members}).T
    else:
        asof = pd.Timestamp(asof)
        rows = {}
        for lab, wl in owner.member_weights.items():
            if asof not in wl.index:
                raise KeyError(f"{owner.label}: member {lab!r} has no weights at {asof.date()}")
            rows[lab] = wl.loc[asof].reindex(owner.universe).fillna(0.0)
        W = pd.DataFrame(rows).T
    if W.index.duplicated().any():
        raise ValueError(f"{owner.label}: duplicate member labels")
    return W


def _check_members(members, member_weights):
    if (members is None) == (member_weights is None):
        raise ValueError("pass exactly one of `members` (live) or `member_weights` (precomputed)")


# ── E. Adversarial diversifier ──────────────────────────────────────────────

class AdversarialDiversifier(_ExcessMixin, RosterStrategy):
    """Paper §3.3: maximise tracking variance (w - c)'Sigma(w - c) vs the centroid c of the
    other methods' weights, s.t. Sharpe(w) >= floor_frac * SR(MaxSharpe) (0.75), long-only.
    Maximising a convex function: deterministic multi-start SLSQP from every single-asset
    vertex that meets the floor plus the tangency portfolio; best feasible local optimum wins.
    No positive expected excess (floor undefined) -> the single asset farthest from c, logged.
    """

    name = "Adversarial"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, members: list[Strategy] | None = None, *, member_weights: dict | None = None,
                 universe=None, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 rf: float | str = "panel", floor_frac: float = 0.75):
        _check_members(members, member_weights)
        if universe is None:
            universe = members[0].universe if members is not None else None
        super().__init__(cov_estimator, lookback, universe)
        self._init_rf(rf)
        self.members, self.member_weights, self.floor_frac = members, member_weights, floor_frac
        self.centroid: dict[pd.Timestamp, pd.Series] = {}

    def predict_weights(self, panel, asof):
        W = _member_matrix(self, panel, asof)
        c = W.mean().to_numpy()
        self.centroid[asof] = pd.Series(c, index=self.universe)
        excess, Sigma, _ = self._excess(panel, asof)
        S = Sigma.to_numpy()
        n = len(c)
        tv = lambda w: (w - c) @ S @ (w - c)
        if excess.max() <= 0:
            logger.warning("%s: no positive expected excess at asof=%s; farthest vertex", self.label, asof)
            self.fallback_dates.append(asof)
            return pd.Series(np.eye(n)[int(np.argmax([tv(e) for e in np.eye(n)]))], index=self.universe)
        w_t, _ = _tangency_long_only(excess, S, self.label, asof)
        floor = self.floor_frac * (w_t @ excess) / np.sqrt(w_t @ S @ w_t)

        def g(w):
            return (w @ excess) / np.sqrt(w @ S @ w) - floor

        def g_jac(w):
            s = np.sqrt(w @ S @ w)
            return excess / s - (w @ excess) * (S @ w) / s ** 3

        ones = np.ones(n)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones},
                {"type": "ineq", "fun": g, "jac": g_jac}]
        starts = [e for e in np.eye(n) if g(e) >= 0] + [w_t]
        best, best_val = None, -np.inf
        for x0 in starts:
            res = opt.minimize(lambda w: -tv(w), x0, jac=lambda w: -2 * S @ (w - c), method="SLSQP",
                               bounds=[(0.0, 1.0)] * n, constraints=cons, options=SLSQP_OPTIONS)
            x = np.clip(res.x, 0.0, None); x = x / x.sum()
            if g(x) >= -1e-9 and tv(x) > best_val:
                best, best_val = x, tv(x)
        if best is None:
            raise RuntimeError(f"{self.label}: no feasible start converged at asof={asof}")
        return pd.Series(best, index=self.universe)


# ── CIO ensembles on trailing member performance ────────────────────────────

TRAILING_METHODS = ("backtest_sharpe", "composite_score", "meta_optimization")
COMPOSITE_WEIGHTS = {"sharpe": 25.0, "ips": 15.0, "diversification": 15.0, "robustness": 15.0}


class TrailingEnsemble(RosterStrategy):
    """Ensembles that weight members by their own past results (paper §3.5, §4.5).

    Inputs are precomputed: `member_weights` (label -> weights_log) and `member_returns`
    (daily NET returns per member label, from maplab's backtest). Each member's return on day t
    uses only information before t, so using returns strictly before asof is out of sample.
    `rf` = daily simple risk-free returns (BIL). Window = `window_months` before asof; while a
    member's history is shorter (warm-up) the ensemble is the simple average (logged).

    backtest_sharpe    mix ∝ max(trailing Sharpe, 0); all <= 0 -> average (logged)
    meta_optimization  members' trailing excess returns as assets, long-only max Sharpe (maplab
                       tangency solver); no positive mean -> GMV over members (logged)
    composite_score    paper rubric without its LLM-judged parts (regime fit 20, CMA use 10),
                       renormalised: Sharpe 25, IPS compliance 15, diversification 15,
                       robustness 15. Each metric -> cross-sectional percentile rank in (0, 1];
                       mix ∝ weighted score.
        IPS compliance  mean of [ex-ante TE vs te_reference <= te_cap, trailing ann. vol in
                        vol_band, trailing max drawdown >= dd_limit]
        diversification exp(entropy of weights) / n (effective number of assets, as the paper's
                        "18 for the 18-asset universe")
        robustness      minus mean one-way target-to-target turnover over the window
    """

    family = "Ensemble"
    constraint = LONG_ONLY

    def __init__(self, member_weights: dict[str, pd.DataFrame], member_returns: pd.DataFrame,
                 rf: pd.Series, method: str, *, universe: list[str], window_months: int = 36,
                 te_reference: dict[str, float] | None = None, te_cap: float = 0.06,
                 vol_band: tuple[float, float] = (0.08, 0.12), dd_limit: float = -0.25,
                 cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, name: str | None = None):
        if method not in TRAILING_METHODS:
            raise ValueError(f"TrailingEnsemble: unknown method {method!r}; choose from {TRAILING_METHODS}")
        missing = [k for k in member_weights if k not in member_returns.columns]
        if missing:
            raise ValueError(f"TrailingEnsemble: no returns for member(s) {missing}")
        if method == "composite_score" and te_reference is None:
            raise ValueError("TrailingEnsemble: composite_score needs te_reference")
        super().__init__(cov_estimator, lookback, universe)
        self.members = None
        self.member_weights, self.method, self.window_months = member_weights, method, window_months
        self.R = member_returns[list(member_weights)]
        self.rf = rf
        self.b = None if te_reference is None else pd.Series(te_reference, dtype=float).reindex(self.universe, fill_value=0.0)
        self.te_cap, self.vol_band, self.dd_limit = te_cap, vol_band, dd_limit
        self.name = name or f"Ens[{method}]"
        self.mix: dict[pd.Timestamp, pd.Series] = {}
        self.scores: dict[pd.Timestamp, pd.DataFrame] = {}
        self.warmup_dates: list[pd.Timestamp] = []

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel, asof):
        asof = pd.Timestamp(asof)
        W = _member_matrix(self, panel, asof)
        K = len(W)
        start = asof - pd.DateOffset(months=self.window_months)
        first_valid = self.R.apply(lambda s: s.first_valid_index()).max()
        if first_valid is None or first_valid > start:
            self.warmup_dates.append(asof)
            mix = np.full(K, 1.0 / K)
            self.mix[asof] = pd.Series(mix, index=W.index)
            return pd.Series(mix @ W.to_numpy(), index=self.universe)
        hist = self.R.loc[(self.R.index >= start) & (self.R.index < asof)]
        ex = hist.sub(self.rf.reindex(hist.index).fillna(0.0), axis=0)
        sr = (ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)).reindex(W.index).to_numpy()

        if self.method == "backtest_sharpe":
            pos = np.clip(np.nan_to_num(sr), 0.0, None)
            if pos.sum() == 0:
                self.fallback_dates.append(asof)
                mix = np.full(K, 1.0 / K)
            else:
                mix = pos / pos.sum()
        elif self.method == "meta_optimization":
            Sm = ex[W.index].cov().to_numpy() * TRADING_DAYS
            mum = ex[W.index].mean().to_numpy() * TRADING_DAYS
            if mum.max() <= 0:
                self.fallback_dates.append(asof)
                mix = _min_variance_long_only(pd.DataFrame(Sm))
            else:
                mix, _ = _tangency_long_only(mum, Sm, self.label, asof)
        else:
            _, Sigma = self._estimate(panel, asof)
            S = Sigma.loc[self.universe, self.universe].to_numpy()
            Wn = W.to_numpy()
            te = np.array([tracking_error(Wn[k], self.b.to_numpy(), S) for k in range(K)])
            h = hist[W.index]
            vol = (h.std(ddof=1) * np.sqrt(TRADING_DAYS)).to_numpy()
            wealth = (1 + h).cumprod()
            mdd = (wealth / wealth.cummax() - 1).min().to_numpy()
            ips = ((te <= self.te_cap + 1e-12).astype(float)
                   + ((vol >= self.vol_band[0]) & (vol <= self.vol_band[1])).astype(float)
                   + (mdd >= self.dd_limit).astype(float)) / 3.0
            p = np.clip(Wn, 1e-300, None)
            div = np.exp(-(Wn * np.log(p)).sum(axis=1)) / Wn.shape[1]
            turn = []
            for lab in W.index:
                wl = self.member_weights[lab]
                wl = wl.loc[(wl.index >= start) & (wl.index <= asof)]
                turn.append(0.5 * wl.diff().abs().sum(axis=1).iloc[1:].mean() if len(wl) > 1 else 0.0)
            rob = -np.asarray(turn)
            metrics = pd.DataFrame({"sharpe": np.nan_to_num(sr, nan=-np.inf), "ips": ips,
                                    "diversification": div, "robustness": rob}, index=W.index)
            ranks = metrics.rank(pct=True)
            wts = pd.Series(COMPOSITE_WEIGHTS)
            score = (ranks[wts.index] * wts).sum(axis=1) / wts.sum()
            self.scores[asof] = metrics.assign(score=score)
            mix = (score / score.sum()).to_numpy()
        self.mix[asof] = pd.Series(mix, index=W.index)
        return pd.Series(mix @ W.to_numpy(), index=self.universe)


# ── Quarterly rebalancing wrapper ───────────────────────────────────────────

class QuarterlyHold(RosterStrategy):
    """Rebalance `base` only at quarter-end rebalance dates (months 3/6/9/12). In other months
    return the previous quarterly target DRIFTED with realised returns exactly as maplab's
    backtest drifts it (w <- w (1 + r) / sum), so the harness trades nothing: one-way turnover
    ~ 0 up to floating point. Requires the backtest's simple returns to equal expm1 of the
    panel's log returns (true when both come from one price frame)."""

    constraint = LONG_ONLY

    def __init__(self, base: Strategy, months: tuple[int, ...] = (3, 6, 9, 12)):
        super().__init__(base.cov_estimator, base.lookback, base.universe)
        self.base, self.months = base, tuple(months)
        self.name, self.family = f"Q[{base.label}]", base.family
        self._last: tuple[pd.Timestamp, pd.Series] | None = None

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel, asof):
        asof = pd.Timestamp(asof)
        if self._last is None or asof.month in self.months:
            w = self.base(panel, asof).reindex(self.universe)
            self._last = (asof, w)
            return w
        last_date, w = self._last
        logr = panel.slice(asof, "returns")
        seg = np.expm1(logr.loc[logr.index >= last_date, self.universe])
        w = w.copy()
        for _, r in seg.iterrows():
            w = w * (1.0 + r)
            w = w / w.sum()
        self._last = (asof, w)
        return w
