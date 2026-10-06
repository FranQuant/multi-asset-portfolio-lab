"""The shipped price panel is the exact file the results were produced from."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

PRICES = Path(__file__).resolve().parents[1] / "data" / "cache" / "prices.parquet"

SHA256 = "329ef8a708730dafc0d8ffb4ec9d42ebbc56dca545bdd03fb9fd1b658daf09e0"
SHAPE = (4611, 17)
FIRST, LAST = pd.Timestamp("2008-01-02"), pd.Timestamp("2026-04-30")
COLUMNS = {"SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "VNQ", "DBC", "GLD", "UUP", "BIL",
           "IWM", "IWD", "IWF"}

pytestmark = pytest.mark.skipif(not PRICES.exists(), reason="data/cache/prices.parquet not found")


def test_prices_sha256():
    assert hashlib.sha256(PRICES.read_bytes()).hexdigest() == SHA256


def test_prices_shape_dates_columns():
    df = pd.read_parquet(PRICES)
    assert df.shape == SHAPE
    assert (df.index[0], df.index[-1]) == (FIRST, LAST)
    assert set(df.columns) == COLUMNS
