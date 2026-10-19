"""Paper roster: methods and CIO ensembles from Ang, Azimbayev & Kim (2026), "The
Self-Driving Portfolio" (Exhibit 3, §3.3, §3.5, §4.1), that maplab did not already have.

Batch 1 (this module):
  Methods      MinimumCorrelation (Varadi et al. 2012; SIT `min.corr`), MinCVaR
               (Rockafellar & Uryasev 2000), TailRiskParity (equal historical-ES budgets;
               Boudt, Carl & Peterson 2013 use modified ES), MaxEntropy (Shannon entropy of
               weights subject to a Sharpe floor, as in §3.3), VolTarget (weight-level vol
               targeting into a cash asset; Moreira & Muir 2017).
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
    """Equal-weight risky portfolio scaled to a volatility target, remainder in a cash asset.

    sigma_hat = sqrt(b' Sigma_w b), b = 1/N over the non-cash assets, Sigma_w from the last
    `vol_window` days (default 21, as in Moreira & Muir 2017's one-month realised variance);
    exposure c = min(cap, target_vol / sigma_hat); w = c b + (1 - c) e_cash. cap = 1 (no
    leverage) keeps it long-only. Unlike maplab's `vol_overlay` (a return-stream overlay),
    this produces weights, so it can enter ensembles. The cash asset must be in `universe`.
    """

    name = "VolTarget"
    family = "Overlay"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, universe=None, *, cash: str,
                 target_vol: float = 0.10, vol_window: int = 21, cap: float = 1.0):
        super().__init__(cov_estimator, vol_window, universe)
        if cash not in self.universe:
            raise ValueError(f"VolTarget: cash asset {cash!r} not in universe")
        if not 0 < cap <= 1:
            raise ValueError("VolTarget: cap must be in (0, 1] (long-only)")
        self.cash, self.target_vol, self.cap = cash, target_vol, cap
        self.exposure: dict[pd.Timestamp, float] = {}

    def predict_weights(self, panel, asof):
        _, Sigma = self._estimate(panel, asof)
        risky = [t for t in self.universe if t != self.cash]
        b = pd.Series(1.0 / len(risky), index=risky)
        sig = float(np.sqrt(b @ Sigma.loc[risky, risky] @ b))
        c = self.cap if sig == 0 else min(self.cap, self.target_vol / sig)
        self.exposure[asof] = c
        w = (c * b).reindex(self.universe, fill_value=0.0)
        w[self.cash] += 1.0 - c
        return w


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
