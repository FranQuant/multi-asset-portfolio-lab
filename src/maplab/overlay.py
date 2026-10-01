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
- Update rule (`update`): c_t above is the TARGET exposure; h_t is the fraction
  actually held.
    "daily":     h_t = c_t (every day).
    "month_end": the target is acted on only at the FIRST trading day s of each
                 calendar month in base_net.index: h_s = c_s (information through
                 the last trading day of the previous month). On every other day
                 t, h_t = d_{t-1}: the risky fraction drifts freely and nothing
                 is traded. The first day of the sample counts as a month start.
                 If s is inside the warm-up, c_s = 1.0, so that whole month stays
                 at (drifted) 1.0.
- Gross overlay return: g_t = h_t * base_net_t + (1 - h_t) * rf_t. With h = 1
  this is base_net bit-for-bit; for cap > 1 the negative cash leg is borrowing
  at rf.
- Drift and trade: after day t the risky fraction drifts to
  d_t = h_t * (1 + base_net_t) / (1 + g_t). Before the first day the previous
  fraction is 1.0. One-way turnover on day t: tau_t = |h_t - d_{t-1}|
  (tau_first = |h_first - 1.0|). Cost on day t: cost_t = tau_t * cost_bps / 1e4.
- Net overlay return: n_t = g_t - cost_t.
- The returned exposure is h; share_braking is computed on h after warm-up;
  diag["target_c"] holds c_t.

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
UPDATES = ("daily", "month_end")


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
    update: str = "daily",
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
    update : "daily" (hold the target each day) or "month_end" (re-target only
             on the first trading day of each calendar month, drift otherwise)

    Returns
    -------
    net : pd.Series       daily overlay returns, net of overlay costs
    exposure : pd.Series  held risky fraction h_t
    diag : dict           turnover, total_cost, sigma_hat, sigma_star,
                          share_braking, target_c, params
    """
    if scaling not in SCALINGS:
        raise ValueError(f"vol_overlay: scaling must be one of {SCALINGS}, got {scaling!r}")
    if target not in TARGETS:
        raise ValueError(f"vol_overlay: target must be one of {TARGETS}, got {target!r}")
    if update not in UPDATES:
        raise ValueError(f"vol_overlay: update must be one of {UPDATES}, got {update!r}")
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

    if update == "daily":
        h = c
        g = h * r + (1.0 - h) * f
        d = h * (1.0 + r) / (1.0 + g)
    else:
        per = idx.to_period("M")
        month_start = np.ones(n, dtype=bool)
        month_start[1:] = per[1:] != per[:-1]
        h = np.empty(n)
        g = np.empty(n)
        d = np.empty(n)
        d_last = 1.0
        for i in range(n):
            h[i] = c[i] if month_start[i] else d_last
            g[i] = h[i] * r[i] + (1.0 - h[i]) * f[i]
            d[i] = h[i] * (1.0 + r[i]) / (1.0 + g[i])
            d_last = d[i]
    d_prev = np.concatenate(([1.0], d[:-1]))
    tau = np.abs(h - d_prev)
    cost = tau * cost_bps / 1e4
    if not np.isclose(cost.sum(), tau.sum() * cost_bps / 1e4, rtol=0, atol=1e-12):
        raise RuntimeError(
            f"Cost booking mismatch: cost.sum()={cost.sum()!r}, "
            f"expected={tau.sum() * cost_bps / 1e4!r}"
        )

    net = pd.Series(g - cost, index=idx, name=base_net.name)
    exposure = pd.Series(h, index=idx, name="exposure")
    post = h[min_history:]
    diag = {
        "turnover": pd.Series(tau, index=idx, name="turnover"),
        "total_cost": float(cost.sum()),
        "sigma_hat": sigma_hat,
        "sigma_star": sigma_star,
        "share_braking": float((post < 1.0).mean()) if len(post) else float("nan"),
        "target_c": pd.Series(c, index=idx, name="target_c"),
        "params": dict(
            window=window, min_history=min_history, cap=cap, scaling=scaling,
            target=target, fixed_target=fixed_target, cost_bps=cost_bps, update=update,
        ),
    }
    return net, exposure, diag
