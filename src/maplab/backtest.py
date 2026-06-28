"""The one backtest loop every model is scored on.

A weight function has the signature:

    weight_fn(panel: Panel, asof: pd.Timestamp) -> pd.Series

returning TARGET weights (indexed by ticker) using ONLY data strictly before
`asof` (the Panel enforces this). The loop:

  1. Sets target weights at each month-end rebalance.
  2. Lets those weights DRIFT with realized returns between rebalances.
  3. At the next rebalance, charges transaction cost on the REAL trade
     (new target - drifted actual), not target-to-target.
  4. Compounds on SIMPLE returns (weighted sum of simple returns is the only
     correct portfolio return).

This is what makes the comparison fair and the costs honest: same dates, same
drift mechanics, same cost model for every method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import REBALANCE, COST_BPS, WARMUP_DAYS, TRADING_DAYS
from .data import Panel


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Month-end rebalance dates available within the return index."""
    return index.to_series().resample(REBALANCE).last().dropna().index


def first_eligible_rebalance(index: pd.DatetimeIndex) -> pd.Timestamp:
    """First rebalance after the warm-up window — the common scoring start.

    Every strategy starts scoring here so methods with different internal
    lookbacks still begin on the same date (fair comparison).
    """
    warmup_cutoff = index[min(WARMUP_DAYS, len(index) - 1)]
    rdates = rebalance_dates(index)
    eligible = rdates[rdates >= warmup_cutoff]
    return eligible[0] if len(eligible) else rdates[-1]


def backtest(
    weight_fn,
    simple_returns: pd.DataFrame,
    panel: Panel,
    cost_bps: float = COST_BPS,
    start: str | pd.Timestamp | None = None,
):
    """Run one strategy through the shared harness.

    Parameters
    ----------
    weight_fn : callable(panel, asof) -> pd.Series of target weights
    simple_returns : daily SIMPLE returns (for compounding)
    panel : Panel (the no-look-ahead data source the weight_fn reads)
    start : optional explicit scoring start; defaults to first eligible
            rebalance after warm-up.

    Returns
    -------
    net_returns : pd.Series   daily portfolio returns, net of costs
    weights_log : pd.DataFrame target weights at each rebalance date
    diag : dict               turnover series + realized cost, for analysis
    """
    rets = simple_returns
    cols = list(rets.columns)

    scoring_start = pd.Timestamp(start) if start else first_eligible_rebalance(rets.index)
    rdates = rebalance_dates(rets.index)
    rdates = rdates[rdates >= scoring_start]

    # daily portfolio return and daily cost, assembled segment by segment
    port_ret = pd.Series(0.0, index=rets.index)
    cost_daily = pd.Series(0.0, index=rets.index)

    weights_log: dict[pd.Timestamp, pd.Series] = {}
    turnover_log: dict[pd.Timestamp, float] = {}

    actual_w = pd.Series(0.0, index=cols)   # drifted, realized weights held now

    for k, asof in enumerate(rdates):
        target = weight_fn(panel, asof).reindex(cols).fillna(0.0)
        weights_log[asof] = target

        # one-way turnover = 0.5 * sum|target - drifted actual|
        one_way = 0.5 * (target - actual_w).abs().sum()
        turnover_log[asof] = float(one_way)
        if asof in cost_daily.index:
            cost_daily.loc[asof] += one_way * cost_bps / 1e4

        # segment until next rebalance (exclusive) or end of sample
        seg_end = rdates[k + 1] if k + 1 < len(rdates) else rets.index[-1] + pd.Timedelta(days=1)
        seg = rets.loc[(rets.index >= asof) & (rets.index < seg_end)]

        # let weights drift across the segment; portfolio return each day is
        # the dot of CURRENT (drifted) weights with that day's asset returns
        w = target.copy()
        for dt, day_ret in seg.iterrows():
            port_ret.loc[dt] += float((w * day_ret).sum())
            # drift: each asset's weight grows by its own return, renormalize
            w = w * (1.0 + day_ret)
            tot = w.sum()
            if tot != 0:
                w = w / tot
        actual_w = w  # carry drifted weights into the next rebalance's cost calc

    net = (port_ret - cost_daily).loc[scoring_start:]
    wlog = pd.DataFrame(weights_log).T
    wlog.index.name = "rebalance_date"
    tlog = pd.Series(turnover_log, name="one_way_turnover")
    tlog.index.name = "rebalance_date"

    diag = {
        "turnover_per_rebalance": tlog,
        "total_cost": float(cost_daily.loc[scoring_start:].sum()),
        "scoring_start": scoring_start,
        "n_rebalances": len(rdates),
    }
    return net, wlog, diag
