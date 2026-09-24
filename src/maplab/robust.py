"""HAC, Sharpe-difference, bootstrap and multiple-testing tools for the
Phase 2 comparison notebooks.

All inputs are daily EXCESS returns unless stated. numpy/pandas/math only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contract import TRADING_DAYS
from .inference import ols


def _as_array(x, name: str) -> np.ndarray:
    a = np.asarray(x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else x, dtype=float)
    if np.isnan(a).any():
        raise ValueError(f"{name}: contains NaN")
    return a


def nw_lag(T: int) -> int:
    """Newey-West rule-of-thumb lag floor(4 * (T/100)^(2/9))."""
    return int(math.floor(4 * (T / 100) ** (2 / 9)))


def newey_west_lrv(U, L: int, demean: bool = True) -> np.ndarray:
    """Bartlett long-run covariance Γ0 + Σ_{j=1..L} (1 − j/(L+1))(Γj + Γj'),
    Γj = U[j:]' U[:-j] / T. 1-D U -> (1, 1)."""
    U = _as_array(U, "newey_west_lrv")
    if U.ndim == 1:
        U = U[:, None]
    if demean:
        U = U - U.mean(axis=0)
    T = U.shape[0]
    S = U.T @ U / T
    for j in range(1, L + 1):
        G = U[j:].T @ U[:-j] / T
        S = S + (1 - j / (L + 1)) * (G + G.T)
    return S


def hac_mean_t(x, L: int | None = None) -> dict:
    """Mean with Newey-West se = sqrt(lrv / n); L=None -> nw_lag(n)."""
    x = _as_array(x, "hac_mean_t")
    n = len(x)
    L = nw_lag(n) if L is None else L
    m = float(x.mean())
    se = float(np.sqrt(newey_west_lrv(x, L)[0, 0] / n))
    return {"mean": m, "se": se, "t": m / se, "L": L, "n": n}


def ols_hac(y: pd.Series, X: pd.DataFrame, L: int | None = None) -> dict:
    """OLS y ~ 1 + X with Newey-West covariance
    V = (X'X)^-1 (n·S) (X'X)^-1, S = lrv of X_t e_t (not demeaned), no dof
    correction (L=0 is HC0). Coefficients are inference.ols's."""
    fit = ols(y, X)
    coef = fit["coef"]
    y_np = _as_array(y, "ols_hac y")
    X_np = _as_array(X, "ols_hac X")
    n = len(y_np)
    L = nw_lag(n) if L is None else L
    Xd = np.column_stack([np.ones(n), X_np])
    e = y_np - Xd @ coef.to_numpy()
    S = newey_west_lrv(Xd * e[:, None], L, demean=False)
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    V = XtX_inv @ (n * S) @ XtX_inv
    se = pd.Series(np.sqrt(np.diag(V)), index=coef.index)
    return {"coef": coef, "se": se, "tstat": coef / se, "n": n, "L": L}


def sharpe_diff_hac(xa, xb, L: int | None = None, trading_days: int = TRADING_DAYS) -> dict:
    """Delta-method HAC test of SR_a − SR_b (ddof=0 moments, y_t = (xa, xb,
    xa², xb²)), annualized by √trading_days; p two-sided normal."""
    if isinstance(xa, pd.Series) and isinstance(xb, pd.Series) and not xa.index.equals(xb.index):
        raise ValueError("sharpe_diff_hac: xa and xb are not aligned")
    xa = _as_array(xa, "sharpe_diff_hac xa")
    xb = _as_array(xb, "sharpe_diff_hac xb")
    if xa.shape != xb.shape or xa.ndim != 1:
        raise ValueError(f"sharpe_diff_hac: length mismatch {xa.shape} vs {xb.shape}")
    T = len(xa)
    L = nw_lag(T) if L is None else L
    ma, mb = xa.mean(), xb.mean()
    ga, gb = (xa ** 2).mean(), (xb ** 2).mean()
    sa, sb = np.sqrt(ga - ma ** 2), np.sqrt(gb - mb ** 2)
    ann = np.sqrt(trading_days)
    d_sr = (ma / sa - mb / sb) * ann
    grad = np.array([ga / sa ** 3, -gb / sb ** 3, -ma / (2 * sa ** 3), mb / (2 * sb ** 3)])
    y = np.column_stack([xa, xb, xa ** 2, xb ** 2])
    Psi = newey_west_lrv(y, L)
    se = float(np.sqrt(grad @ Psi @ grad / T) * ann)
    t = float(d_sr / se) if se > 0 else float("nan")
    return {"d_sr": float(d_sr), "se": se, "t": t, "p": math.erfc(abs(t) / math.sqrt(2)),
            "sr_a": float(ma / sa * ann), "sr_b": float(mb / sb * ann), "L": L, "n": T}


def bootstrap_index_chunks(T: int, B: int, mean_block: float, seed: int, chunk: int = 100):
    """Stationary-bootstrap indices, (≤chunk, T) int arrays totalling B rows:
    a new block starts w.p. 1/mean_block at a uniform index, else previous+1
    mod T. One default_rng(seed) for the whole run."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    t = np.arange(T)
    done = 0
    while done < B:
        c = min(chunk, B - done)
        new = rng.random((c, T)) < p
        new[:, 0] = True
        starts = rng.integers(0, T, size=(c, T))
        bstart = np.maximum.accumulate(np.where(new, t, 0), axis=1)
        yield (np.take_along_axis(starts, bstart, axis=1) + t - bstart) % T
        done += c


def bootstrap_sharpe(X, chunks, trading_days: int = TRADING_DAYS) -> np.ndarray:
    """(B, k) annualized Sharpe ratios (ddof=1) of X (T, k) per resample."""
    X = _as_array(X, "bootstrap_sharpe")
    if X.ndim == 1:
        X = X[:, None]
    out = []
    for idx in chunks:
        Xb = X[idx]
        out.append(Xb.mean(axis=1) / Xb.std(axis=1, ddof=1) * np.sqrt(trading_days))
    return np.concatenate(out, axis=0)


def bootstrap_capm_alpha(Y, x, chunks, trading_days: int = TRADING_DAYS) -> np.ndarray:
    """(B, k) annualized CAPM alphas (daily intercept × trading_days, as in
    inference.capm_table) of each column of Y (T, k) on the shared regressor
    x (T,) per resample."""
    Y = _as_array(Y, "bootstrap_capm_alpha Y")
    x = _as_array(x, "bootstrap_capm_alpha x")
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != len(x):
        raise ValueError(f"bootstrap_capm_alpha: length mismatch {Y.shape[0]} vs {len(x)}")
    n = len(x)
    out = []
    for idx in chunks:
        xb = x[idx]
        Yb = Y[idx]
        xbar = xb.sum(axis=1) / n
        Sxx = (xb * xb).sum(axis=1)
        ybar = Yb.sum(axis=1) / n
        Sxy = np.einsum("dt,dtk->dk", xb, Yb)
        beta = (Sxy - n * xbar[:, None] * ybar) / (Sxx - n * xbar ** 2)[:, None]
        out.append((ybar - beta * xbar[:, None]) * trading_days)
    return np.concatenate(out, axis=0)


def percentile_ci(draws, level: float = 0.95):
    """Percentile interval (lo, hi) per column of draws (B, k)."""
    draws = np.asarray(draws, dtype=float)
    q = 100 * (1 - level) / 2
    return np.percentile(draws, q, axis=0), np.percentile(draws, 100 - q, axis=0)


def bootstrap_p(d_hat, d_star) -> np.ndarray:
    """Two-sided bootstrap p per column: mean(|d* − d̂| ≥ |d̂|)."""
    d_hat = np.asarray(d_hat, dtype=float)
    d_star = np.asarray(d_star, dtype=float)
    return (np.abs(d_star - d_hat) >= np.abs(d_hat)).mean(axis=0)


def romano_wolf(t_hat, t_star) -> np.ndarray:
    """Romano-Wolf step-down adjusted p (original order). t_star (B, k) are
    centred bootstrap t-statistics."""
    t_hat = np.atleast_1d(np.asarray(t_hat, dtype=float))
    t_star = np.asarray(t_star, dtype=float).reshape(-1, len(t_hat))
    order = np.argsort(-np.abs(t_hat), kind="stable")
    abs_star = np.abs(t_star[:, order])
    # max over hypotheses ranked j..k, for each j
    tail_max = np.maximum.accumulate(abs_star[:, ::-1], axis=1)[:, ::-1]
    p_step = (tail_max >= np.abs(t_hat[order])).mean(axis=0)
    p_mono = np.maximum.accumulate(p_step)
    out = np.empty_like(p_mono)
    out[order] = p_mono
    return out


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p (original order)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p (original order)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    raw = np.minimum(1.0, m * p[order] / np.arange(1, m + 1))
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = adj
    return out


def forecast_bias_decomposition(realized_vol, exante_vol) -> dict:
    """q = log(s/σ̂): q_bar, vbar = mean(s²/σ̂²), half_log_vbar = ½·log vbar,
    J = half_log_vbar − q_bar (≥ 0, the Jensen gap)."""
    s = _as_array(realized_vol, "forecast_bias_decomposition realized_vol")
    sig = _as_array(exante_vol, "forecast_bias_decomposition exante_vol")
    q_bar = float(np.mean(np.log(s / sig)))
    vbar = float(np.mean((s / sig) ** 2))
    half_log_vbar = 0.5 * math.log(vbar)
    return {"q_bar": q_bar, "vbar": vbar, "half_log_vbar": half_log_vbar, "J": half_log_vbar - q_bar}


def segment_block_bootstrap_mean(values, block: int = 12, B: int = 10000, *, seed: int) -> np.ndarray:
    """(B,) bootstrap draws of the mean of a per-segment sequence, circular
    moving blocks of length `block`."""
    v = _as_array(values, "segment_block_bootstrap_mean")
    n = len(v)
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(B, -1)[:, :n]
    return v[idx].mean(axis=1)
