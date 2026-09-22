"""Contract-level guarantees: panel-only tickers never leak into a model or
the backtest. Synthetic data only, no real cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.contract import UNIVERSE, RF_TICKER, FACTOR_TICKERS
from maplab.data import Panel
from maplab.models import EqualWeight


def test_rf_and_factor_tickers_disjoint_from_universe():
    universe_set = set(UNIVERSE)
    assert not (universe_set & {RF_TICKER}), f"RF_TICKER {RF_TICKER!r} must not be in UNIVERSE"
    assert not (universe_set & set(FACTOR_TICKERS)), (
        f"FACTOR_TICKERS {FACTOR_TICKERS!r} must not intersect UNIVERSE"
    )


def _make_universe_returns(seed=0):
    rng = np.random.default_rng(seed)
    # 2021-03-31 is itself a business day *and* a calendar month-end, so the
    # last rebalance's holding segment (dates >= that calendar label) is
    # non-empty -- backtest()'s rebalance labels are calendar month-ends,
    # not necessarily trading days.
    idx = pd.bdate_range("2020-01-01", "2021-03-31")
    data = rng.normal(0.0005, 0.01, size=(len(idx), len(UNIVERSE)))
    return pd.DataFrame(data, index=idx, columns=UNIVERSE)


def test_backtest_uses_exactly_universe_columns():
    log_returns = _make_universe_returns(seed=1)
    simple_returns = log_returns  # synthetic, shape/columns are what's under test
    panel = Panel({"returns": log_returns})

    net, wlog, diag = ml.backtest(EqualWeight(), simple_returns, panel)

    assert list(wlog.columns) == UNIVERSE


def test_backtest_raises_on_non_allocatable_columns():
    log_returns = _make_universe_returns(seed=2)
    panel = Panel({"returns": log_returns})

    contaminated = log_returns.copy()
    rng = np.random.default_rng(3)
    contaminated["IWM"] = rng.normal(0.0005, 0.01, size=len(contaminated))
    contaminated["BIL"] = rng.normal(0.00002, 0.0002, size=len(contaminated))

    with pytest.raises(ValueError, match="non-allocatable"):
        ml.backtest(EqualWeight(), contaminated, panel)
