"""Data loading and the no-look-ahead Panel gatekeeper.

The Panel is the single point through which strategies see data. Its `slice`
method refuses to return anything dated on or after `asof`, which makes
look-ahead bias a structural impossibility rather than a thing you remember to
avoid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import UNIVERSE, RF_TICKER, FACTOR_TICKERS, START, END, TRADING_DAYS, FILL_POLICY


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find the repo (marked by data/ and src/)."""
    cur = (Path.cwd() if start is None else Path(start)).resolve()
    for path in (cur, *cur.parents):
        if (path / "data").exists() and (path / "src").exists():
            return path
    raise FileNotFoundError("Could not locate repo root (expected data/ and src/).")


def load_prices(path: Path | None = None) -> pd.DataFrame:
    """Load the cached wide price panel (index=Date, columns=tickers).

    Looks for data/cache/prices.parquet. If absent, raises with a clear
    pointer to scripts/build_panel.py (or the synthetic fallback in nb 00).
    """
    root = find_repo_root()
    path = path or root / "data" / "cache" / "prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/build_panel.py to ingest real data, "
            "or use the synthetic fallback in notebook 00 for a dry run."
        )
    px = pd.read_parquet(path)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().loc[START:END, UNIVERSE]
    # Missing-data policy (stated in contract): forward-fill only. No back-fill
    # — that would leak future prices into the past (look-ahead).
    if FILL_POLICY == "ffill":
        px = px.ffill()
    return px


def load_rf_returns(path: Path | None = None) -> pd.DataFrame:
    """Load the risk-free daily simple-return series.

    BIL (contract.RF_TICKER, 1-3m T-bills) is the risk-free numeraire, never
    in UNIVERSE, so no model allocates to cash. This reads it out of the same
    cached panel used by `load_prices`, forward-fills, and converts to simple
    daily returns. One-column DataFrame named RF_TICKER, index aligned to
    `load_prices()`.
    """
    root = find_repo_root()
    path = path or root / "data" / "cache" / "prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/build_panel.py to ingest real data, "
            "or use the synthetic fallback in notebook 00 for a dry run."
        )
    px = pd.read_parquet(path)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().loc[START:END, [RF_TICKER]]
    if FILL_POLICY == "ffill":
        px = px.ffill()
    rf = px.pct_change().dropna(how="all")
    return rf


def load_factor_returns(path: Path | None = None) -> pd.DataFrame:
    """Load daily log returns of the panel-only factor ETFs.

    FACTOR_TICKERS (contract.FACTOR_TICKERS, currently IWM/IWD/IWF) are used
    to construct notebook 02's SIZE/VALUE factor diagnostics — never in
    UNIVERSE, never allocatable. Reads the same cached panel used by
    `load_prices`, forward-fills, and converts to log returns (additive,
    consistent with the rest of the harness's estimation convention).
    """
    root = find_repo_root()
    path = path or root / "data" / "cache" / "prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/build_panel.py to ingest real data, "
            "or use the synthetic fallback in notebook 00 for a dry run."
        )
    px = pd.read_parquet(path)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().loc[START:END, FACTOR_TICKERS]
    if FILL_POLICY == "ffill":
        px = px.ffill()
    return np.log(px).diff().dropna(how="all")


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns — for ESTIMATION (mu, Sigma). Additive, well-behaved."""
    return np.log(prices).diff().dropna(how="all")


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns — for COMPOUNDING the backtest.

    Portfolio return is the weighted sum of SIMPLE returns. The backtest must
    use these, never log returns.
    """
    return prices.pct_change().dropna(how="all")


class Panel:
    """A bundle of aligned time-indexed frames with a no-look-ahead slicer.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        e.g. {"prices": ..., "returns": ..., "regimes": ...}
    """

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = {k: v.sort_index() for k, v in frames.items()}

    def __contains__(self, key: str) -> bool:
        return key in self._frames

    def slice(
        self,
        asof: pd.Timestamp,
        kind: str = "returns",
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """Return rows STRICTLY BEFORE `asof` (no look-ahead), optionally the
        last `lookback` rows of them.
        """
        if kind not in self._frames:
            raise KeyError(f"Panel has no frame '{kind}'. Have: {list(self._frames)}")
        df = self._frames[kind]
        past = df.loc[df.index < pd.Timestamp(asof)]
        if lookback is not None:
            past = past.iloc[-lookback:]
        return past

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._frames["returns"].index
