"""Performance metrics — identical definitions for every strategy.

All functions take a daily return Series and return a scalar (annualized where
relevant). Defined once so no notebook can quietly use a different Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import TRADING_DAYS


def ann_return(r: pd.Series) -> float:
    return float(r.mean() * TRADING_DAYS)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def ann_sharpe(r: pd.Series, rf: float | pd.Series | pd.DataFrame) -> float:
    """Annualized Sharpe ratio, excess over a risk-free rate.

    `rf` is REQUIRED — a float (annual RF) or a pd.Series/1-column DataFrame
    of daily simple RF returns (e.g. `load_rf_returns()["BIL"]`). For a
    Series/DataFrame, `r` and `rf` are aligned on `r`'s index (inner join);
    if the aligned length is < 90% of len(r), that's a coverage problem, not
    a silent approximation, so this raises.
    """
    if isinstance(rf, pd.DataFrame):
        rf = rf.iloc[:, 0]
    if isinstance(rf, pd.Series):
        aligned_r, aligned_rf = r.align(rf, join="inner")
        if len(aligned_r) < 0.9 * len(r):
            raise ValueError(
                f"ann_sharpe: aligned rf coverage {len(aligned_r)}/{len(r)} "
                f"({len(aligned_r) / len(r):.1%}) is below the 90% floor"
            )
        excess = aligned_r - aligned_rf
    else:
        excess = r - rf / TRADING_DAYS
    vol = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(excess.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def calmar(r: pd.Series) -> float:
    dd = abs(max_drawdown(r))
    return float(ann_return(r) / dd) if dd > 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean())


def ann_turnover(turnover_per_rebalance: pd.Series) -> float:
    """Annualized one-way turnover from the per-rebalance turnover series.

    Exact, not inferred: average one-way turnover per rebalance × the actual
    number of rebalances per year implied by the calendar (≈12 for monthly).
    Excludes the first rebalance (turnover from a zero portfolio is just the
    initial build, not ongoing trading).
    """
    if turnover_per_rebalance is None or len(turnover_per_rebalance) < 2:
        return np.nan
    ongoing = turnover_per_rebalance.iloc[1:]
    idx = turnover_per_rebalance.index
    years = (idx[-1] - idx[0]).days / 365.25
    rebals_per_year = (len(turnover_per_rebalance) - 1) / years if years > 0 else np.nan
    return float(ongoing.mean() * rebals_per_year)


def summary(
    r: pd.Series,
    turnover_per_rebalance: pd.Series | None = None,
    *,
    rf: float | pd.Series | pd.DataFrame,
) -> dict:
    return {
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": ann_sharpe(r, rf),
        "max_dd": max_drawdown(r),
        "calmar": calmar(r),
        "hit_rate": hit_rate(r),
        "ann_turnover": ann_turnover(turnover_per_rebalance)
        if turnover_per_rebalance is not None else np.nan,
    }
