"""Volatility overlay: scale a method's net return stream by trailing volatility.

Takes a base method's daily NET returns (from `backtest()`) and the BIL daily
returns, and blends the two with a risky fraction c_t chosen from trailing
volatility. Definitions (implemented exactly):

- Inputs: base_net = daily net returns; rf = BIL daily returns, reindexed to
  base_net.index. NaN in base_net or in the reindexed rf raises ValueError.
- Excess returns: x_t = base_net_t - rf_t.
- sigma_hat_t = std(x over the `window` days ending at t, ddof=1) * sqrt(TRADING_DAYS).
- Exposure for day t uses information through t-1 only:
    target "expanding_mean": sigma_star_t = mean of all valid sigma_hat_s, s <= t-1;
    target "fixed":          sigma_star_t = fixed_target (required, > 0).
    ratio_t = sigma_star_t / sigma_hat_{t-1}   ("inverse_vol") or its square
              ("inverse_variance").
    c_t = min(cap, ratio_t). No floor. If sigma_hat_{t-1} == 0, c_t = cap.
    Warm-up: c_t = 1.0 while fewer than `min_history` observations of base_net
    exist strictly before t.
- Gross overlay return: g_t = c_t * base_net_t + (1 - c_t) * rf_t. With c = 1
  this is base_net bit-for-bit; for cap > 1 the negative cash leg is borrowing
  at rf.
- Drift and trade: after day t the risky fraction drifts to
  d_t = c_t * (1 + base_net_t) / (1 + g_t). Before the first day the previous
  fraction is 1.0. One-way turnover on day t: tau_t = |c_t - d_{t-1}|
  (tau_first = |c_first - 1.0|). Cost on day t: cost_t = tau_t * cost_bps / 1e4.
- Net overlay return: n_t = g_t - cost_t.

Approximation we accept: the base method's own trading costs are already inside
base_net and scale with c (a smaller risky sleeve trades less); the overlay
only adds the cost of moving the risky fraction itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import COST_BPS, TRADING_DAYS

SCALINGS = ("inverse_vol", "inverse_variance")
TARGETS = ("expanding_mean", "fixed")


def vol_overlay(
    base_net: pd.Series,
    rf: pd.Series,
    *,
    window: int = 21,
    min_history: int = 252,
    cap: float = 1.0,
    scaling: str = "inverse_vol",
    target: str = "expanding_mean",
    fixed_target: float | None = None,
    cost_bps: float = COST_BPS,
) -> tuple[pd.Series, pd.Series, dict]:
    """Apply the volatility overlay to a net return stream.

    Parameters
    ----------
    base_net : daily NET returns of the base method (from backtest())
    rf : BIL daily returns; reindexed to base_net.index
    window : trailing days for sigma_hat (>= 2)
    min_history : warm-up length; c = 1.0 for the first `min_history` days
                  (>= window, so sigma_hat_{t-1} exists after warm-up)
    cap : maximum risky fraction (> 1 borrows at rf)
    scaling : "inverse_vol" or "inverse_variance"
    target : "expanding_mean" or "fixed"
    fixed_target : annualised vol target, required (> 0) when target == "fixed"
    cost_bps : one-way cost in bps on the overlay's own turnover

    Returns
    -------
    net : pd.Series       daily overlay returns, net of overlay costs
    exposure : pd.Series  risky fraction c_t
    diag : dict           turnover, total_cost, sigma_hat, sigma_star,
                          share_braking, params
    """
    if scaling not in SCALINGS:
        raise ValueError(f"vol_overlay: scaling must be one of {SCALINGS}, got {scaling!r}")
    if target not in TARGETS:
        raise ValueError(f"vol_overlay: target must be one of {TARGETS}, got {target!r}")
    if target == "fixed" and (fixed_target is None or not fixed_target > 0):
        raise ValueError(f"vol_overlay: target='fixed' needs fixed_target > 0, got {fixed_target!r}")
    if window < 2:
        raise ValueError(f"vol_overlay: window must be >= 2, got {window}")
    if min_history < window:
        raise ValueError(f"vol_overlay: min_history ({min_history}) must be >= window ({window})")
    if not cap > 0:
        raise ValueError(f"vol_overlay: cap must be > 0, got {cap}")

    rf = rf.reindex(base_net.index)
    if base_net.isna().any():
        raise ValueError("vol_overlay: base_net contains NaN")
    if rf.isna().any():
        raise ValueError("vol_overlay: rf has NaN after reindexing to base_net.index (missing dates?)")

    idx = base_net.index
    n = len(idx)
    r = base_net.to_numpy(dtype=float)
    f = rf.to_numpy(dtype=float)

    x = base_net - rf
    sigma_hat = x.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    sigma_lag = sigma_hat.shift(1)
    if target == "expanding_mean":
        sigma_star = sigma_hat.expanding().mean().shift(1)
    else:
        sigma_star = pd.Series(float(fixed_target), index=idx)

    lag = sigma_lag.to_numpy()
    star = sigma_star.to_numpy()
    c = np.ones(n)
    for i in range(min_history, n):
        if lag[i] == 0.0:
            c[i] = cap
        else:
            ratio = star[i] / lag[i]
            if scaling == "inverse_variance":
                ratio = ratio ** 2
            c[i] = min(cap, ratio)

    g = c * r + (1.0 - c) * f
    d = c * (1.0 + r) / (1.0 + g)
    d_prev = np.concatenate(([1.0], d[:-1]))
    tau = np.abs(c - d_prev)
    cost = tau * cost_bps / 1e4
    if not np.isclose(cost.sum(), tau.sum() * cost_bps / 1e4, rtol=0, atol=1e-12):
        raise RuntimeError(
            f"Cost booking mismatch: cost.sum()={cost.sum()!r}, "
            f"expected={tau.sum() * cost_bps / 1e4!r}"
        )

    net = pd.Series(g - cost, index=idx, name=base_net.name)
    exposure = pd.Series(c, index=idx, name="exposure")
    post = c[min_history:]
    diag = {
        "turnover": pd.Series(tau, index=idx, name="turnover"),
        "total_cost": float(cost.sum()),
        "sigma_hat": sigma_hat,
        "sigma_star": sigma_star,
        "share_braking": float((post < 1.0).mean()) if len(post) else float("nan"),
        "params": dict(
            window=window, min_history=min_history, cap=cap, scaling=scaling,
            target=target, fixed_target=fixed_target, cost_bps=cost_bps,
        ),
    }
    return net, exposure, diag
