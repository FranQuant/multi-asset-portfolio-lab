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
        if not res.success:
            # SLSQP occasionally reports failure (e.g. "Positive directional
            # derivative for linesearch") within a hair of the true optimum
            # when the argmax-only x0 starts right at a boundary — a
            # linesearch quirk, not a bad solution. Retry once from a
            # feasible interior point spread over all positive-excess
            # assets; only raise if that also fails.
            logger.warning(
                "%s: SLSQP failed at asof=%s (%s); retrying with interior x0",
                self.label, asof, res.message,
            )
            self.retry_dates.append(asof)
            pos = np.clip(excess_np, 0.0, None)
            y0_retry = pos / (excess_np @ pos)
            res = opt.minimize(
                lambda y: y @ Sigma_np @ y, y0_retry, jac=lambda y: 2 * Sigma_np @ y, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if not res.success:
                raise RuntimeError(f"{self.label}: SLSQP failed at asof={asof}: {res.message}")
        y = res.x
        w = y / y.sum()
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


# ── Pure teaching functions (unconstrained closed forms) ────────────────────
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
