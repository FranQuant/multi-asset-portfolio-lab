"""Notebook 10 checks run on the real price cache: the 8 core runs and 60/40 reproduce notebook 08's Sharpe
ratios in all three windows, and the overlays hold their identity, rebuild and go-live properties."""
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
    """Data, base runs and overlays built exactly as notebook 10's data, §3 and overlay cells do."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        import _nb08_helpers as h8
        import _nb10_helpers as h10
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    simple_returns = ml.to_simple_returns(prices)
    rf_daily = ml.load_rf_returns()["BIL"]
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    split_ts = pd.Timestamp(ml.TRAIN_TEST_SPLIT)
    runs = h10.build_base_runs(simple_returns, panel)
    return {"h8": h8, "h10": h10, "rf_daily": rf_daily, "split_ts": split_ts, "runs": runs,
            "ov": h10.build_overlays(runs, rf_daily)}


def test_base_runs_match_nb08(nb):
    h8, h10 = nb["h8"], nb["h10"]
    summary = h8.summary_table(nb["runs"], h10.CORE, nb["split_ts"], nb["rf_daily"])
    ml.inference.check_reproduction(summary, {k: h8.REPRO_KEYS[k] for k in h10.CORE})
    bench_summary = h8.summary_table(nb["runs"], h10.BENCH, nb["split_ts"], nb["rf_daily"])
    ml.inference.check_reproduction(bench_summary, h8.BENCH_REPRO_KEYS)


def test_overlay_identity(nb):
    h10 = nb["h10"]
    idt = h10.identity_table(nb["runs"], nb["ov"], nb["rf_daily"])
    assert idt["c1_identical"].all() and idt["warmup_c_all_one"].all() and idt["index_equal"].all()
    assert float(idt["rebuild_max_abs"].max()) <= 1e-15
    assert (idt["live_date"] == h10.LIVE).all()
    assert (idt["h_min"] > 0).all() and (idt["h_max"] <= 1.0).all()
