"""Model classes: the strategies scored by the shared backtest harness.

Each `Strategy` instance is directly usable as `backtest`'s `weight_fn` (it
implements `__call__(panel, asof) -> pd.Series`). Notebooks are the teaching
layer — they instantiate these classes, run them through `ml.backtest`, and
narrate the results; the model logic itself lives here so every notebook that
uses "GMV" or "MaxSharpe" means exactly the same thing.
"""
from __future__ import annotations

import abc
import logging

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from .contract import LONG_ONLY, LONG_SHORT, UNIVERSE, TRADING_DAYS, COV_LOOKBACK
from .covariance import sample_cov
from .data import Panel

logger = logging.getLogger("maplab.models")


class Strategy(abc.ABC):
    """Base class for a weight-generating rule scored by `ml.backtest`.

    Subclasses set the class attributes `name`, `family`, and `constraint`,
    and implement `predict_weights`. An instance is callable with the same
    signature `backtest` expects for `weight_fn`.
    """

    name: str
    family: str
    constraint: str = LONG_ONLY
    universe: list[str] = UNIVERSE

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK):
        self.cov_estimator = cov_estimator
        self.lookback = lookback

    @property
    def label(self) -> str:
        est_name = self.cov_estimator.__name__
        if est_name.endswith("_cov"):
            est_name = est_name[: -len("_cov")]
        return f"{self.name}({est_name})"

    @abc.abstractmethod
    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        """Return TARGET weights (indexed by ticker) using only data strictly
        before `asof` (enforced by the Panel)."""
        raise NotImplementedError

    def __call__(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        return self._validate(self.predict_weights(panel, asof))

    def _validate(self, w: pd.Series) -> pd.Series:
        w = w.reindex(self.universe)
        if w.isna().any():
            raise ValueError(f"{self.label}: NaN weight(s) for {list(w[w.isna()].index)}")
        if abs(w.sum() - 1.0) > 1e-6:
            raise ValueError(f"{self.label}: weights sum to {w.sum()!r}, expected 1.0")
        if self.constraint == LONG_ONLY and (w < -1e-8).any():
            raise ValueError(f"{self.label}: negative weight(s) under LONG_ONLY: {w[w < -1e-8].to_dict()}")
        if self.constraint == LONG_ONLY:
            w = w.clip(lower=0.0)
            w = w / w.sum()
        return w

    def _estimate(self, panel: Panel, asof: pd.Timestamp) -> tuple[pd.Series, pd.DataFrame]:
        """Estimate (mu, Sigma) on `self.universe` from the lookback window of
        LOG returns strictly before `asof`.

        mu is converted from log to arithmetic (mu_log + 0.5*var) because the
        harness scores strategies on arithmetic (simple) returns — feeding a
        log-return mean straight into a mean-variance objective would be
        systematically biased downward.
        """
        rets = panel.slice(asof, "returns", self.lookback)[self.universe]
        if len(rets) != self.lookback or rets.isna().any().any():
            raise ValueError(
                f"{self.label}: insufficient/NaN return history for asof={asof} "
                f"(got {len(rets)} rows, need {self.lookback})"
            )
        Sigma = self.cov_estimator(rets)  # already annualised
        mu = rets.mean() * TRADING_DAYS + 0.5 * pd.Series(np.diag(Sigma.to_numpy()), index=Sigma.index)
        return mu, Sigma


def _min_variance_long_only(Sigma: pd.DataFrame) -> np.ndarray:
    """Long-only global minimum-variance weights: minimise w'Σw s.t. sum w=1, w>=0.

    With Σ annualised, w'Σw is on the order of 3e-4 for a typical GMV
    portfolio, so the default SLSQP ftol=1e-6 is far too loose relative to
    the objective's scale and leaves the solver free to wander among
    near-equal-objective solutions — adding spurious turnover between
    rebalances that isn't a real change in the minimum-variance portfolio.
    """
    n = Sigma.shape[0]
    Sigma_np = Sigma.to_numpy()
    x0 = np.full(n, 1.0 / n)
    ones = np.ones(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones}]
    bounds = [(0.0, 1.0)] * n
    res = opt.minimize(
        lambda w: w @ Sigma_np @ w, x0, jac=lambda w: 2 * Sigma_np @ w, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"_min_variance_long_only: SLSQP failed: {res.message}")
    return res.x


def _tangency_long_only(excess: np.ndarray, Sigma_np: np.ndarray,
                        label: str, asof, on_retry=None) -> tuple[np.ndarray, bool]:
    """Long-only max-Sharpe weights via the convex reformulation
    min y'Σy s.t. excess'y = 1, y >= 0; w = y / sum(y).
    Caller must ensure excess.max() > 0 (degenerate case handled by caller).
    Returns (w, retried). On SLSQP failure logs a warning, retries once from
    the interior x0, and raises RuntimeError if that also fails.
    `on_retry`, if given, is called the moment a retry is committed to
    (before the retry attempt runs), so callers can record the retry even
    if the retry itself goes on to raise."""
    excess_np = excess
    n = len(excess_np)
    i0 = int(np.argmax(excess_np))
    y0 = np.zeros(n)
    y0[i0] = 1.0 / excess_np[i0]

    cons = [{"type": "eq", "fun": lambda y: excess_np @ y - 1.0, "jac": lambda y: excess_np}]
    bounds = [(0.0, None)] * n
    res = opt.minimize(
        lambda y: y @ Sigma_np @ y, y0, jac=lambda y: 2 * Sigma_np @ y, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    retried = False
    if not res.success:
        # SLSQP occasionally reports failure (e.g. "Positive directional
        # derivative for linesearch") within a hair of the true optimum
        # when the argmax-only x0 starts right at a boundary — a
        # linesearch quirk, not a bad solution. Retry once from a
        # feasible interior point spread over all positive-excess
        # assets; only raise if that also fails.
        logger.warning(
            "%s: SLSQP failed at asof=%s (%s); retrying with interior x0",
            label, asof, res.message,
        )
        retried = True
        if on_retry is not None:
            on_retry()
        pos = np.clip(excess_np, 0.0, None)
        y0_retry = pos / (excess_np @ pos)
        res = opt.minimize(
            lambda y: y @ Sigma_np @ y, y0_retry, jac=lambda y: 2 * Sigma_np @ y, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"{label}: SLSQP failed at asof={asof}: {res.message}")
    y = res.x
    w = y / y.sum()
    return w, retried


def _mdp_long_only(Sigma: pd.DataFrame, label: str, asof, on_retry=None) -> tuple[np.ndarray, bool]:
    """Long-only max diversification ratio DR(w) = w'σ/√(w'Σw) s.t. 1'w=1,
    w>=0, solved exactly as the tangency QP with excess := σ (σ>0, so never
    degenerate). Identity: risk weights x = w∘σ/(w'σ) equal the long-only
    GMV on the correlation matrix C = D⁻¹ΣD⁻¹. Special cases: constant
    correlation ⇒ inverse vol; equal vols ⇒ GMV. Mechanism only — no
    empirical claims.
    """
    sigma = np.sqrt(np.diag(Sigma.to_numpy()))
    return _tangency_long_only(sigma, Sigma.to_numpy(), label, asof, on_retry=on_retry)


def _erc_long_only(Sigma: pd.DataFrame, label: str, asof, tol: float = 1e-12,
                    maxiter: int = 1000) -> tuple[np.ndarray, int]:
    """Long-only equal-risk-contribution (ERC) portfolio: w_i (Sigma w)_i equal for
    all i, 1'w = 1, w > 0. Solved via the strictly convex surrogate
        min_{y>0}  0.5 y'Sigma y - (1/N) sum_i log y_i,      w = y / 1'y,
    whose stationarity condition y_i (Sigma y)_i = 1/N gives equal risk
    contributions exactly; the log barrier makes w > 0 automatic, so unlike the
    tangency/MDP QPs there is no boundary case and no fallback. The solution is
    unique. Cyclical coordinate descent: with c_i = sum_{j != i} Sigma_ij y_j,
    each coordinate solves Sigma_ii y_i^2 + c_i y_i - 1/N = 0, taking the
    positive root. Identities: risk weights x = w*sigma/(w'sigma) equal ERC run
    on the correlation matrix C = D^-1 Sigma D^-1; w_i * beta_i,p = 1/N where
    beta_i,p = (Sigma w)_i / w'Sigma w. Special cases: constant correlation =>
    inverse vol (any N); N = 2 => inverse vol for any rho. Scale equivariance:
    w(K Sigma K) is proportional to K^-1 w(Sigma).
    """
    Sigma_np = Sigma.to_numpy()
    n = Sigma_np.shape[0]
    diag = np.diag(Sigma_np)
    if (diag <= 0).any():
        raise ValueError(f"{label}: non-positive diagonal in Sigma at asof={asof}")

    b = 1.0 / n
    sigma = np.sqrt(diag)
    y = (1.0 / sigma) / np.sum(1.0 / sigma)

    dev = np.inf
    for sweep in range(1, maxiter + 1):
        for i in range(n):
            c = Sigma_np[i] @ y - Sigma_np[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * Sigma_np[i, i] * b)) / (2.0 * Sigma_np[i, i])
        dev = np.max(np.abs(y * (Sigma_np @ y) - b))
        if dev <= tol:
            w = y / y.sum()
            return w, sweep

    raise RuntimeError(
        f"{label}: ERC coordinate descent failed to converge at asof={asof} "
        f"after {maxiter} sweeps (max|y_i(Sy)_i - 1/N| = {dev:.3e})"
    )


_HRP_LINKAGES = ("single", "average", "ward")
_HRP_BISECTIONS = ("tree", "positional")


def _hrp_cluster_var(Sigma_np: np.ndarray, idx) -> float:
    """Variance of the inverse-variance (IVP) portfolio on the sub-block idx."""
    idx = list(idx)
    sub = Sigma_np[np.ix_(idx, idx)]
    p = 1.0 / np.diag(sub)
    p = p / p.sum()
    return float(p @ sub @ p)


def _hrp_long_only(Sigma: pd.DataFrame, label: str, asof, linkage: str = "single",
                   bisection: str = "tree", dist_of_dist: bool = True) -> tuple[np.ndarray, dict]:
    """Long-only hierarchical risk parity (HRP). Closed-form arithmetic, no solver.

    Recipe: C = D^-1 Sigma D^-1 (clipped to [-1, 1], unit diagonal);
    d_ij = sqrt((1 - rho_ij) / 2); if dist_of_dist, cluster on the Euclidean
    distance between columns of d (the literature's recipe), else on d directly;
    hierarchical linkage (single / average / ward); then recursive bisection.
    At every split, capital goes to the two sides in inverse proportion to the
    variance of each side's inverse-variance portfolio:
        alpha_L = V_R / (V_L + V_R).
    bisection="tree" splits at each dendrogram node (a function of Sigma up to
    linkage ties, permutation-equivariant). bisection="positional" splits the
    leaf order into halves at floor(n/2), as in the literature's original
    algorithm; its weights depend on the leaf orientation and therefore on the
    input column order.

    Only the diagonal blocks Sigma_LL, Sigma_RR enter an allocation; the cross
    block Sigma_LR never does - it only shapes the tree. Identities: diagonal
    Sigma (C = I) => inverse-variance weights for any tree and either bisection
    (= long-only GMV); N = 2 => inverse variance for any rho; uniform scaling
    c*Sigma leaves weights unchanged. Every weight is strictly positive.

    Returns (w, info) with info = {"order", "Z", "splits"}; splits is a list of
    (left_indices, right_indices, alpha_left).
    """
    if linkage not in _HRP_LINKAGES:
        raise ValueError(f"{label}: unknown linkage {linkage!r}; expected one of {_HRP_LINKAGES}")
    if bisection not in _HRP_BISECTIONS:
        raise ValueError(f"{label}: unknown bisection {bisection!r}; expected one of {_HRP_BISECTIONS}")
    Sigma_np = Sigma.to_numpy(dtype=float)
    n = Sigma_np.shape[0]
    if n < 2:
        raise ValueError(f"{label}: HRP needs at least 2 assets, got {n} at asof={asof}")
    diag = np.diag(Sigma_np)
    if (diag <= 0).any():
        raise ValueError(f"{label}: non-positive diagonal in Sigma at asof={asof}")

    sigma = np.sqrt(diag)
    C = Sigma_np / np.outer(sigma, sigma)
    C = 0.5 * (C + C.T)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    D = np.sqrt(0.5 * (1.0 - C))
    np.fill_diagonal(D, 0.0)
    y = pdist(D, metric="euclidean") if dist_of_dist else squareform(D, checks=True)
    Z = sch.linkage(y, method=linkage)
    order = sch.leaves_list(Z)

    w = np.ones(n)
    splits = []
    if bisection == "tree":
        stack = [sch.to_tree(Z)]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                continue
            left, right = node.get_left(), node.get_right()
            L, R = left.pre_order(), right.pre_order()
            v_l, v_r = _hrp_cluster_var(Sigma_np, L), _hrp_cluster_var(Sigma_np, R)
            a = 1.0 - v_l / (v_l + v_r)
            w[L] *= a
            w[R] *= 1.0 - a
            splits.append((L, R, a))
            stack += [left, right]
    else:
        items = [list(order)]
        while items:
            nxt = []
            for it in items:
                if len(it) <= 1:
                    continue
                h = len(it) // 2
                L, R = it[:h], it[h:]
                v_l, v_r = _hrp_cluster_var(Sigma_np, L), _hrp_cluster_var(Sigma_np, R)
                a = 1.0 - v_l / (v_l + v_r)
                w[L] *= a
                w[R] *= 1.0 - a
                splits.append((L, R, a))
                nxt += [L, R]
            items = nxt
    return w, {"order": order, "Z": Z, "splits": splits}


class GMV(Strategy):
    """Long-only global minimum-variance portfolio."""

    name = "GMV"
    family = "Risk-based"
    constraint = LONG_ONLY

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        w = _min_variance_long_only(Sigma)
        return pd.Series(w, index=Sigma.columns)


class MaxSharpe(Strategy):
    """Long-only maximum-Sharpe (tangency) portfolio.

    Solved via the standard convex reformulation (minimise y'Σy subject to
    excess'y = 1, y >= 0; w = y / sum(y)) rather than maximising the Sharpe
    ratio directly: the Sharpe ratio itself is non-convex (a ratio of a
    linear and a quadratic form), but this QP is convex and exact whenever
    at least one asset has positive expected excess return.
    """

    name = "MaxSharpe"
    family = "Return-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK, rf: float | str = "panel"):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        self.rf = rf
        self.fallback_dates: list[pd.Timestamp] = []
        self.retry_dates: list[pd.Timestamp] = []

    def _rf_ann(self, panel: Panel, asof: pd.Timestamp) -> float:
        """Annualized risk-free rate for this asof.

        rf="panel" (default): trailing mean of the panel's "rf" frame (daily
        simple BIL returns) over the same lookback window as (mu, Sigma),
        annualized ×252. A float bypasses the panel entirely (used by tests).
        """
        if isinstance(self.rf, str) and self.rf == "panel":
            if "rf" not in panel:
                raise ValueError(
                    f"{self.label}: panel has no 'rf' frame at asof={asof}; "
                    "pass rf as a float or add an 'rf' frame to the panel"
                )
            rf_slice = panel.slice(asof, "rf", self.lookback)
            if len(rf_slice) < self.lookback:
                raise ValueError(
                    f"{self.label}: insufficient 'rf' history at asof={asof} "
                    f"(got {len(rf_slice)} rows, need {self.lookback})"
                )
            return float(rf_slice.mean().iloc[0] * TRADING_DAYS)
        return float(self.rf)

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        mu, Sigma = self._estimate(panel, asof)
        rf_ann = self._rf_ann(panel, asof)
        excess = mu - rf_ann

        if excess.max() <= 0:
            # Degenerate case: no long-only portfolio has positive expected
            # excess return. Maximizing Sharpe here would just chase
            # whichever asset happens to have the least-negative excess
            # return and the lowest vol — noise, not signal. Fall back to
            # GMV on the same Sigma instead of raising or chasing volatility.
            logger.warning(
                "MaxSharpe: degenerate excess returns at asof=%s (max excess=%.6f); "
                "falling back to GMV weights", asof, excess.max(),
            )
            self.fallback_dates.append(asof)
            w = _min_variance_long_only(Sigma)
            return pd.Series(w, index=Sigma.columns)

        Sigma_np = Sigma.to_numpy()
        excess_np = excess.to_numpy()
        w, retried = _tangency_long_only(
            excess_np, Sigma_np, self.label, asof,
            on_retry=lambda: self.retry_dates.append(asof),
        )
        return pd.Series(w, index=Sigma.columns)


class BetaTargetMinVar(Strategy):
    """Long-only minimum-variance with a CAPM beta floor.

    Under CAPM, beta is the only priced source of risk: once you have fixed
    the beta you want, every other source of variance is estimation noise to
    be minimised away. There is no mu here at all, so unlike MaxSharpe this
    can't become an error-maximizer over a noisy expected-return estimate —
    the whole optimisation runs on Sigma and beta alone. beta_target is fixed
    a priori (0.3 is the default); 0.5 is a sensitivity check, not a second
    "better" choice.

    Problem: min w'Sigma w  s.t.  sum(w)=1, w>=0, beta'w >= beta_target.
    """

    family = "Risk-based"
    constraint = LONG_ONLY
    universe: list[str] = UNIVERSE

    def __init__(
        self,
        cov_estimator=sample_cov,
        lookback: int = COV_LOOKBACK,
        beta_target: float = 0.3,
        market: str = "SPY",
    ):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        self.beta_target = beta_target
        self.market = market
        self.name = f"BetaMinVar(β≥{beta_target})"
        self.slack_dates: list[pd.Timestamp] = []
        self.infeasible_dates: list[pd.Timestamp] = []
        self.retry_dates: list[pd.Timestamp] = []

    def _beta_vector(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        """CAPM betas of `self.universe` against `self.market`, on the same
        trailing 252d window (strictly before `asof`, no look-ahead) as
        `_estimate`'s Sigma.

        excess_i = log_return_i - log1p(rf), rf the panel's daily simple BIL
        return; beta_i = cov(excess_i, excess_market) / var(excess_market),
        ddof=1 (pandas' default), consistent with Sigma's `returns.cov()`.
        By construction beta_market == 1 exactly.
        """
        if self.market not in self.universe:
            raise ValueError(f"{self.label}: market={self.market!r} not in universe {self.universe}")
        if "rf" not in panel:
            raise ValueError(
                f"{self.label}: panel has no 'rf' frame at asof={asof}; "
                "add an 'rf' frame to the panel"
            )
        rets = panel.slice(asof, "returns", self.lookback)[self.universe]
        rf_slice = panel.slice(asof, "rf", self.lookback)
        if len(rets) != self.lookback or rets.isna().any().any():
            raise ValueError(
                f"{self.label}: insufficient/NaN return history for asof={asof} "
                f"(got {len(rets)} rows, need {self.lookback})"
            )
        if len(rf_slice) != self.lookback or rf_slice.isna().any().any():
            raise ValueError(
                f"{self.label}: insufficient/NaN 'rf' history for asof={asof} "
                f"(got {len(rf_slice)} rows, need {self.lookback})"
            )
        rf_log = np.log1p(rf_slice.iloc[:, 0])
        excess = rets.sub(rf_log, axis=0)
        cov_with_mkt = excess.cov()[self.market]
        var_mkt = float(excess[self.market].var())
        return cov_with_mkt / var_mkt

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        beta = self._beta_vector(panel, asof)

        if beta.max() < self.beta_target:
            # No long-only combination (a convex combination of the beta_i's)
            # can reach the target. Fall back to GMV rather than raise or
            # silently violate the floor.
            logger.warning(
                "%s: infeasible beta target at asof=%s (max beta=%.4f < target=%.4f); "
                "falling back to GMV weights", self.label, asof, beta.max(), self.beta_target,
            )
            self.infeasible_dates.append(asof)
            w = _min_variance_long_only(Sigma)
            return pd.Series(w, index=Sigma.columns)

        w_gmv = _min_variance_long_only(Sigma)
        beta_gmv = float(beta.reindex(Sigma.columns).to_numpy() @ w_gmv)
        if beta_gmv - self.beta_target > 1e-6:
            # GMV already clears the beta floor: the inequality constraint
            # isn't binding, so the constrained optimum IS the GMV optimum.
            self.slack_dates.append(asof)
            return pd.Series(w_gmv, index=Sigma.columns)

        n = Sigma.shape[0]
        Sigma_np = Sigma.to_numpy()
        beta_np = beta.reindex(Sigma.columns).to_numpy()
        ones = np.ones(n)
        x0 = np.full(n, 1.0 / n)
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones},
            {"type": "ineq", "fun": lambda w: beta_np @ w - self.beta_target, "jac": lambda w: beta_np},
        ]
        bounds = [(0.0, 1.0)] * n
        res = opt.minimize(
            lambda w: w @ Sigma_np @ w, x0, jac=lambda w: 2 * Sigma_np @ w, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            # Same rationale as MaxSharpe's retry: an occasional SLSQP
            # linesearch quirk near the true optimum, not a bad solution.
            # Retry once from the highest-beta asset (guaranteed feasible,
            # since beta.max() >= beta_target was already checked above).
            logger.warning(
                "%s: SLSQP failed at asof=%s (%s); retrying with beta-tilted x0",
                self.label, asof, res.message,
            )
            self.retry_dates.append(asof)
            i0 = int(np.argmax(beta_np))
            y0_retry = np.zeros(n)
            y0_retry[i0] = 1.0
            res = opt.minimize(
                lambda w: w @ Sigma_np @ w, y0_retry, jac=lambda w: 2 * Sigma_np @ w, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if not res.success:
                raise RuntimeError(f"{self.label}: SLSQP failed at asof={asof}: {res.message}")
        return pd.Series(res.x, index=Sigma.columns)


class BlackLitterman(Strategy):
    """Long-only Black-Litterman tangency portfolio with an equal-weight
    prior and vol-scaled 12-1 momentum sign views.

    Prior: EW-implied (Pi = delta * Sigma @ w_EW), not CAPM/SPY-implied — a
    no-view BL posterior then collapses exactly to EW, the repo's benchmark,
    so the views are measured directly against it. A CAPM/SPY prior would
    instead make the no-view portfolio 100% SPY, which obscures what the
    views themselves are doing.

    Views: P = I (one view per asset), Q = Pi + k * sign(mom) * sigma — a
    fixed, vol-scaled tilt in the direction of trailing 12-1 excess momentum,
    not the trailing return itself as Q (that would just be the sample-mean
    mu again, with none of BL's shrinkage toward the prior).

    He-Litterman view uncertainty, Omega = diag(diag(tau*Sigma)), so tau
    cancels out of mu_BL exactly; only k and sr_ref (through delta) matter
    for the resulting weights. k=0.1 is the registered setting; k=0.2 is a
    sensitivity check. At k>=0.2, absolute risk-off views (as in 2022)
    concentrate weight in the few positive-momentum assets, because the
    long-only, fully-invested portfolio has no cash asset to retreat to.
    """

    family = "Return-based"
    constraint = LONG_ONLY
    universe: list[str] = UNIVERSE

    def __init__(
        self,
        cov_estimator=sample_cov,
        lookback: int = COV_LOOKBACK,
        k: float = 0.1,
        sr_ref: float = 0.3,
        mom_skip: int = 21,
        tau: float = 0.05,
        effn_floor: float = 5.0,
    ):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        if not (0 <= mom_skip < lookback):
            raise ValueError(
                f"BlackLitterman: mom_skip must satisfy 0 <= mom_skip < lookback={lookback}, "
                f"got {mom_skip}"
            )
        if not (sr_ref > 0):
            raise ValueError(f"BlackLitterman: sr_ref must be > 0, got {sr_ref}")
        if not (k >= 0):
            raise ValueError(f"BlackLitterman: k must be >= 0, got {k}")
        if not (tau > 0):
            raise ValueError(f"BlackLitterman: tau must be > 0, got {tau}")
        self.k = k
        self.sr_ref = sr_ref
        self.mom_skip = mom_skip
        self.tau = tau
        self.effn_floor = effn_floor
        self.name = f"BL(k={k})"
        self.fallback_dates: list[pd.Timestamp] = []
        self.retry_dates: list[pd.Timestamp] = []
        self.concentration_dates: list[pd.Timestamp] = []

    def posterior(self, panel: Panel, asof: pd.Timestamp) -> dict:
        """Compute the EW prior, momentum views, and He-Litterman posterior
        mean at `asof`. Public so notebooks can inspect the intermediate
        quantities (Pi, views, mu_BL) directly rather than just the final
        weights.
        """
        if "rf" not in panel:
            raise ValueError(
                f"{self.label}: panel has no 'rf' frame at asof={asof}; "
                "add an 'rf' frame to the panel"
            )
        _, Sigma = self._estimate(panel, asof)
        Sigma = Sigma.loc[self.universe, self.universe]
        Sigma_np = Sigma.to_numpy()

        rets = panel.slice(asof, "returns", self.lookback)[self.universe]
        rf_slice = panel.slice(asof, "rf", self.lookback)
        if len(rf_slice) != self.lookback or rf_slice.isna().any().any():
            raise ValueError(
                f"{self.label}: insufficient/NaN 'rf' history for asof={asof} "
                f"(got {len(rf_slice)} rows, need {self.lookback})"
            )

        n = len(self.universe)
        w_ew = pd.Series(1.0 / n, index=self.universe)
        sigma_ew = float(np.sqrt(w_ew.to_numpy() @ Sigma_np @ w_ew.to_numpy()))
        delta = self.sr_ref / sigma_ew
        pi = pd.Series(delta * (Sigma_np @ w_ew.to_numpy()), index=self.universe)
        sigma = pd.Series(np.sqrt(np.diag(Sigma_np)), index=self.universe)

        rf_log = np.log1p(rf_slice.iloc[:, 0])
        excess = rets.sub(rf_log, axis=0)
        momentum = excess.iloc[: self.lookback - self.mom_skip].sum()
        signs = pd.Series(np.sign(momentum.to_numpy()), index=self.universe)
        q = pi + self.k * signs * sigma

        tS = self.tau * Sigma_np
        Omega = np.diag(np.diag(tS))
        mu_bl_np = pi.to_numpy() + tS @ np.linalg.solve(tS + Omega, (q - pi).to_numpy())
        mu_bl = pd.Series(mu_bl_np, index=self.universe)

        return {
            "Sigma": Sigma, "sigma": sigma, "w_ew": w_ew, "delta": delta,
            "pi": pi, "momentum": momentum, "signs": signs, "q": q, "mu_bl": mu_bl,
        }

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        post = self.posterior(panel, asof)
        Sigma = post["Sigma"]
        mu_bl = post["mu_bl"]

        if mu_bl.max() <= 0:
            # Same rationale as MaxSharpe's degenerate case: no long-only
            # portfolio has positive expected excess return under the BL
            # posterior. Fall back to GMV on the same Sigma.
            logger.warning(
                "%s: degenerate BL posterior at asof=%s (max mu_bl=%.6f); "
                "falling back to GMV weights", self.label, asof, mu_bl.max(),
            )
            self.fallback_dates.append(asof)
            w = _min_variance_long_only(Sigma)
            return pd.Series(w, index=Sigma.columns)

        w, _ = _tangency_long_only(
            mu_bl.to_numpy(), Sigma.to_numpy(), self.label, asof,
            on_retry=lambda: self.retry_dates.append(asof),
        )
        effN = 1.0 / np.sum(w ** 2)
        if effN < self.effn_floor:
            self.concentration_dates.append(asof)
            logger.info(
                "%s: concentrated BL portfolio at asof=%s (effN=%.4f < floor=%.4f)",
                self.label, asof, effN, self.effn_floor,
            )
        return pd.Series(w, index=Sigma.columns)


class MostDiversified(Strategy):
    """Long-only maximum diversification ratio (MDP) portfolio."""

    name = "MDP"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        self.retry_dates: list[pd.Timestamp] = []

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        w, _ = _mdp_long_only(
            Sigma, self.label, asof,
            on_retry=lambda: self.retry_dates.append(asof),
        )
        return pd.Series(w, index=Sigma.columns)


class EqualRiskContribution(Strategy):
    """Long-only equal risk contribution (ERC) portfolio."""

    name = "ERC"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK):
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        self.sweeps: list[int] = []

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        w, n_sweeps = _erc_long_only(Sigma, self.label, asof)
        self.sweeps.append(n_sweeps)
        return pd.Series(w, index=Sigma.columns)


class HierarchicalRiskParity(Strategy):
    """Long-only hierarchical risk parity. Default = tree-following bisection,
    single linkage, distance-of-distances (the registered nb07 configuration)."""

    name = "HRP"
    family = "Risk-based"
    constraint = LONG_ONLY

    def __init__(self, cov_estimator=sample_cov, lookback: int = COV_LOOKBACK,
                 linkage: str = "single", bisection: str = "tree", dist_of_dist: bool = True):
        if linkage not in _HRP_LINKAGES:
            raise ValueError(f"unknown linkage {linkage!r}; expected one of {_HRP_LINKAGES}")
        if bisection not in _HRP_BISECTIONS:
            raise ValueError(f"unknown bisection {bisection!r}; expected one of {_HRP_BISECTIONS}")
        super().__init__(cov_estimator=cov_estimator, lookback=lookback)
        self.linkage = linkage
        self.bisection = bisection
        self.dist_of_dist = dist_of_dist
        self.orders: list[np.ndarray] = []

    @property
    def label(self) -> str:
        base = super().label
        tags = []
        if self.bisection != "tree":
            tags.append(self.bisection)
        if self.linkage != "single":
            tags.append(self.linkage)
        if not self.dist_of_dist:
            tags.append("direct")
        if not tags:
            return base
        return f"{self.name}[{','.join(tags)}]{base[len(self.name):]}"

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        w, info = _hrp_long_only(Sigma, self.label, asof, linkage=self.linkage,
                                 bisection=self.bisection, dist_of_dist=self.dist_of_dist)
        self.orders.append(info["order"])
        return pd.Series(w, index=Sigma.columns)


class EqualWeight(Strategy):
    """1/N over the universe. No estimation, no lookback dependency."""

    name = "EqualWeight"
    family = "Benchmark"
    constraint = LONG_ONLY

    @property
    def label(self) -> str:
        return self.name

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        n = len(self.universe)
        return pd.Series(1.0 / n, index=self.universe)


class InverseVol(Strategy):
    """Long-only inverse-vol weights w ∝ 1/σ, σ = sqrt(diag Σ) from the
    strategy's own covariance estimator (sample by default, as in nb05/nb06)."""

    name = "IV"
    family = "Risk-based"
    constraint = LONG_ONLY

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        sigma = np.sqrt(np.diag(Sigma.to_numpy()))
        w = (1.0 / sigma) / np.sum(1.0 / sigma)
        return pd.Series(w, index=Sigma.columns)


class InverseVariance(Strategy):
    """Long-only inverse-variance weights w ∝ 1/diag(Σ) -- HRP's reference
    case (HRP on diag(Σ) = IVP). Sample covariance by default, as in nb07."""

    name = "IVP"
    family = "Risk-based"
    constraint = LONG_ONLY

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        _, Sigma = self._estimate(panel, asof)
        diag = np.diag(Sigma.to_numpy())
        w = (1.0 / diag) / np.sum(1.0 / diag)
        return pd.Series(w, index=Sigma.columns)


# ── Teaching and diagnostic functions (unconstrained closed forms and risk contributions)
# Not used by any Strategy above (both are long-only, solved numerically);
# these are for the notebook's exposition and for tests, to show what the
# unconstrained solutions look like in closed form.

def gmv_closed_form(Sigma: pd.DataFrame) -> pd.Series:
    """Unconstrained global minimum-variance weights: Σ⁻¹1 / 1'Σ⁻¹1."""
    ones = np.ones(Sigma.shape[0])
    Sigma_inv_ones = np.linalg.solve(Sigma.to_numpy(), ones)
    w = Sigma_inv_ones / (ones @ Sigma_inv_ones)
    return pd.Series(w, index=Sigma.columns)


def tangency_closed_form(mu: pd.Series, Sigma: pd.DataFrame, rf: float) -> pd.Series:
    """Unconstrained tangency portfolio: Σ⁻¹(μ−rf) / 1'Σ⁻¹(μ−rf)."""
    excess = (mu - rf).to_numpy()
    Sigma_inv_excess = np.linalg.solve(Sigma.to_numpy(), excess)
    denom = np.ones(len(excess)) @ Sigma_inv_excess
    if denom <= 0:
        raise ValueError(f"tangency_closed_form: denominator 1'Σ⁻¹(μ−rf) = {denom!r} <= 0")
    w = Sigma_inv_excess / denom
    return pd.Series(w, index=Sigma.columns)


def mdp_closed_form(Sigma: pd.DataFrame) -> pd.Series:
    """Unconstrained max diversification ratio weights: Σ⁻¹σ / 1'Σ⁻¹σ
    (may short)."""
    sigma = np.sqrt(np.diag(Sigma.to_numpy()))
    Sigma_inv_sigma = np.linalg.solve(Sigma.to_numpy(), sigma)
    denom = np.ones(len(sigma)) @ Sigma_inv_sigma
    if denom <= 0:
        raise ValueError(f"mdp_closed_form: denominator 1'Σ⁻¹σ = {denom!r} <= 0")
    w = Sigma_inv_sigma / denom
    return pd.Series(w, index=Sigma.columns)


def risk_contributions(w: pd.Series | np.ndarray, Sigma: pd.DataFrame) -> pd.Series:
    """Total risk contribution of each asset: RC_i = w_i (Σw)_i / √(w'Σw),
    so that sum_i RC_i = √(w'Σw) exactly (Euler's theorem for the
    homogeneous-of-degree-1 risk measure σ_p)."""
    if isinstance(w, pd.Series):
        w_np = w.reindex(Sigma.columns).to_numpy()
    else:
        w_np = np.asarray(w)
    Sigma_np = Sigma.to_numpy()
    var = float(w_np @ Sigma_np @ w_np)
    if var <= 0:
        raise ValueError(f"risk_contributions: w'Σw = {var!r} <= 0")
    rc = (w_np * (Sigma_np @ w_np)) / np.sqrt(var)
    return pd.Series(rc, index=Sigma.columns)
