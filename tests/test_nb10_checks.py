"""Notebook 10 checks run on the real price cache: the 8 core runs and 60/40 reproduce notebook 08's Sharpe
ratios in all three windows, and the overlays hold their identity, rebuild and go-live properties."""
import sys
from pathlib import Path

import numpy as np
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
        from helpers import nb08 as h8
        from helpers import nb10 as h10
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


def test_overlay_exposure_properties(nb):
    """h equals c, the braking share rebuilds from the c path, and c never exceeds 1."""
    ex = nb["h10"].exposure_table(nb["ov"], nb["split_ts"])
    exf = ex.xs("full", level="window")
    for lab, o in nb["ov"].items():
        assert np.array_equal(o["h"].to_numpy(), o["c"].to_numpy())
        assert abs(o["diag"]["share_braking"] - exf.loc[lab, "share_braking"]) <= 1e-15
    assert (exf["mean_c"] <= 1.0).all()


def test_overlay_lowest_c_falls_in_march_april_2020(nb):
    ex = nb["h10"].exposure_table(nb["ov"], nb["split_ts"])
    exf = ex.xs("full", level="window")
    assert exf["min_c_date"].between(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-04-07")).all()


def test_perf_table_sharpe_matches_scoring_frames(nb):
    h10, split_ts, rf_daily = nb["h10"], nb["split_ts"], nb["rf_daily"]
    all_runs = {**nb["runs"], **nb["ov"]}
    XW = h10.scoring_frames(all_runs, h10.BASES + h10.VMPS, rf_daily, split_ts)
    ORDER = [x for m in h10.BASES for x in (m, h10.vmp(m))]
    perf = h10.perf_table(all_runs, ORDER, split_ts, rf_daily)
    perf = perf.reindex(pd.MultiIndex.from_product([ORDER, list(h10.WINDOWS)], names=["strategy", "window"]))
    PERF_COLS = ["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "ann_turnover", "cost_drag"]

    gap = 0.0
    for w, Xw in XW.items():
        sr = Xw.mean() / Xw.std(ddof=1) * np.sqrt(ml.TRADING_DAYS)
        gap = max(gap, float((sr[ORDER] - perf.xs(w, level="window").loc[ORDER, "sharpe"]).abs().max()))
    assert gap <= 1e-10 and not perf[PERF_COLS].isna().any().any()


def test_sensitivity_registered_variant_equals_overlay_minus_base(nb):
    h10, split_ts, rf_daily = nb["h10"], nb["split_ts"], nb["rf_daily"]
    all_runs = {**nb["runs"], **nb["ov"]}
    ORDER = [x for m in h10.BASES for x in (m, h10.vmp(m))]
    perf = h10.perf_table(all_runs, ORDER, split_ts, rf_daily)
    perf = perf.reindex(pd.MultiIndex.from_product([ORDER, list(h10.WINDOWS)], names=["strategy", "window"]))
    sens = h10.sensitivity_table(nb["runs"], rf_daily, split_ts, nb["ov"])
    sf = sens.xs("full", level="window")

    P_full = perf.xs("full", level="window")
    reg_gap = max(abs(float(sf.loc[("registered", n), "dSR_vs_base"])
                      - float(P_full.loc[h10.vmp(n), "sharpe"] - P_full.loc[n, "sharpe"])) for n in h10.BASES)
    assert reg_gap <= 1e-10
    assert (sf["dSR_vs_registered"].xs("registered", level="variant") == 0.0).all()
