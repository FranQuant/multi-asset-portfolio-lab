"""Notebook 09 checks run on the real price cache: the 8 core runs reproduce notebook 08's Sharpe
ratios in all three windows, and the grid's 252-day runs hold the core runs' weights on the common dates."""
import sys
from pathlib import Path

import pandas as pd
import pytest

import maplab as ml

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]


@pytest.fixture(scope="module")
def nb():
    """Data, core runs and grid built exactly as notebook 09's data, §3 and §4 cells do."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb08 as h8
        from helpers import nb09 as h9
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    simple_returns = ml.to_simple_returns(prices)
    rf_daily = ml.load_rf_returns()["BIL"]
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    split_ts = pd.Timestamp(ml.TRAIN_TEST_SPLIT)
    rdates = ml.rebalance_dates(log_returns.index)
    rdates = rdates[rdates >= ml.first_eligible_rebalance(log_returns.index)]
    assert len(rdates) == 208
    common_start = h9.common_start(log_returns.index, 756)
    return {"h8": h8, "h9": h9, "simple_returns": simple_returns, "rf_daily": rf_daily, "split_ts": split_ts,
            "log_returns": log_returns, "panel": panel,
            "core": h9.build_core_runs(simple_returns, panel),
            "common_start": common_start,
            "grid": h9.build_grid_runs(simple_returns, panel, common_start)}


def test_core_runs_match_nb08(nb):
    h8, h9 = nb["h8"], nb["h9"]
    summary = h8.summary_table(nb["core"], h9.CORE, nb["split_ts"], nb["rf_daily"])
    ml.inference.check_reproduction(summary, {m: h8.REPRO_KEYS[m] for m in h9.CORE})


def test_grid_252_weights_equal_core(nb):
    assert nb["common_start"] == pd.Timestamp("2011-01-31")
    assert nb["h9"].weight_crosscheck(nb["grid"], nb["core"]).max() < 1e-12


def test_forecast_bias_decomposition_q_bar_equals_mean_log_ratio(nb):
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb07 as h7
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    CORE = nb["h9"].CORE
    rdates = ml.rebalance_dates(nb["log_returns"].index)
    rdates = rdates[rdates >= ml.first_eligible_rebalance(nb["log_returns"].index)]
    q_table, n_dropped, _ = h7.forecast_bias_table(nb["panel"], rdates, nb["core"], nb["simple_returns"], CORE)
    q_mean = q_table.groupby("method")["q"].mean()
    for m in CORE:
        seg = q_table[q_table["method"] == m].sort_values("asof")
        d = ml.robust.forecast_bias_decomposition(seg["realized_vol"], seg["exante_vol"])
        assert abs(d["q_bar"] - float(q_mean[m])) <= 1e-12, m
