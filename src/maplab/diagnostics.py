"""Portfolio diagnostics shared by the notebooks.

Lifted verbatim from the notebook-local helpers of notebook 07 (§1 helper
cell, §3.2, Figure F1, Figure F3) so every notebook computes effN, DR,
n_uncorr, ... with the same formulas and the same operation order.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.optimize as opt

from .contract import ASSET_GROUPS, GROUP_OF

logger = logging.getLogger("maplab.diagnostics")


def _as_np(Sigma) -> np.ndarray:
    return Sigma.to_numpy() if isinstance(Sigma, pd.DataFrame) else np.asarray(Sigma)


def effective_n(w) -> float:
    """Effective N = 1 / sum(w_i^2)."""
    w = np.asarray(w)
    return float(1.0 / (w ** 2).sum())


def half_l1(a, b) -> float:
    """Half-L1 distance 0.5 * sum|a - b| between two weight vectors."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series) and not a.index.equals(b.index):
        raise ValueError("half_l1: Series indexes are not aligned")
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def port_vol(w, Sigma) -> float:
    """Ex-ante portfolio vol sqrt(w'Σw)."""
    Sigma_np = _as_np(Sigma)
    w = np.asarray(w)
    return float(np.sqrt(w @ Sigma_np @ w))


def diversification_ratio(w, Sigma) -> float:
    """DR = w'σ / sqrt(w'Σw), σ = sqrt(diag Σ)."""
    Sigma_np = _as_np(Sigma)
    sigma_np = np.sqrt(np.diag(Sigma_np))
    w = np.asarray(w)
    return float(w @ sigma_np) / port_vol(w, Sigma_np)


def n_uncorrelated(w, Sigma) -> float:
    # Exp of the entropy of p_k = (e_k'w)^2 lambda_k / w'Sigma w, eigh(Sigma), lambda clipped at 0.
    Sigma_np = _as_np(Sigma)
    w = np.asarray(w)
    lam, E = np.linalg.eigh(Sigma_np)
    lam = np.clip(lam, 0.0, None)
    v = (E.T @ w) ** 2 * lam
    total = v.sum()
    p = v / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def max_rc_share(w, Sigma) -> float:
    """Largest share of total variance contributed by one asset."""
    Sigma_np = _as_np(Sigma)
    w = np.asarray(w)
    rc = w * (Sigma_np @ w)
    return float(np.max(rc) / rc.sum())


def weights_by_group(wlog: pd.DataFrame) -> pd.DataFrame:
    """Sum a (date x ticker) weight log into (date x asset group) columns,
    in ASSET_GROUPS order, missing groups filled with 0."""
    grouped = wlog.T.groupby(GROUP_OF).sum().T
    return grouped.reindex(columns=list(ASSET_GROUPS)).fillna(0.0)


def _min_variance_at_target(Sigma_np, mu_np, target, x0):
    n = Sigma_np.shape[0]
    ones = np.ones(n)
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones},
        {"type": "eq", "fun": lambda w: w @ mu_np - target, "jac": lambda w: mu_np},
    ]
    bounds = [(0.0, 1.0)] * n
    res = opt.minimize(
        lambda w: w @ Sigma_np @ w, x0, jac=lambda w: 2 * Sigma_np @ w, method="SLSQP",
        bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x if res.success else None


def long_only_frontier(Sigma, mu, n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Long-only min-variance frontier traced over target returns
    linspace(mu.min, mu.max, n_points), EW start, warm-started; targets where
    SLSQP fails are skipped (and logged). Returns (vols, rets)."""
    Sigma_np = _as_np(Sigma)
    mu_np = mu.to_numpy() if isinstance(mu, pd.Series) else np.asarray(mu)
    n = Sigma_np.shape[0]

    targets = np.linspace(mu_np.min(), mu_np.max(), n_points)
    x0 = np.full(n, 1.0 / n)
    frontier_ret, frontier_vol = [], []
    for t in targets:
        w = _min_variance_at_target(Sigma_np, mu_np, t, x0)
        if w is None:
            logger.warning("long_only_frontier: SLSQP failed at target %r; point skipped", t)
            continue
        x0 = w
        frontier_ret.append(w @ mu_np)
        frontier_vol.append(np.sqrt(w @ Sigma_np @ w))
    frontier_ret = np.array(frontier_ret)
    frontier_vol = np.array(frontier_vol)
    return frontier_vol, frontier_ret


def diversification_table(weights: dict, Sigma, highlight: str = "UUP", tickers=None) -> pd.DataFrame:
    """One row per weight vector: w_<highlight>, effN, n_uncorr, max_RC_share,
    DR, exante_vol (nb07 §5.2). `tickers` defaults to Sigma's columns."""
    if tickers is None:
        if not isinstance(Sigma, pd.DataFrame):
            raise ValueError("diversification_table: pass tickers when Sigma is not a DataFrame")
        tickers = list(Sigma.columns)
    tickers = list(tickers)
    Sigma_np = _as_np(Sigma)
    iU = tickers.index(highlight)

    rows = {}
    for name, w in weights.items():
        w = np.asarray(w)
        rows[name] = {
            f"w_{highlight}": float(w[iU]), "effN": effective_n(w), "n_uncorr": n_uncorrelated(w, Sigma_np),
            "max_RC_share": max_rc_share(w, Sigma_np), "DR": diversification_ratio(w, Sigma_np),
            "exante_vol": port_vol(w, Sigma_np),
        }
    return pd.DataFrame(rows).T
