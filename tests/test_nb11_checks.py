"""Notebook 11 check runs on the real price cache: the 23 α-family runs and 60/40 rebuilt for notebook 11
reproduce the registered Sharpe ratios in all three windows (4338 / 3504 / 834 days)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]


def shown(x, value, dp):
    """x equals `value` at the precision the notebook prints (dp decimals)."""
    return abs(float(x) - value) <= 0.5 * 10 ** -dp + 1e-12


@pytest.fixture(scope="module")
def h11():
    """Notebook 11's helpers, imported with notebooks/ on sys.path only for the import."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb11 as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    return h


def test_build_data_matches_registered_sharpe(h11):
    data = h11.build_data()
    XW = data["XW"]
    assert [len(Xw) for Xw in XW.values()] == [4338, 3504, 834]
    gate = h11.repro_gate(XW)
    assert len(gate) == 72 and gate["ok"].all()


@pytest.fixture(scope="module")
def tabs(h11):
    """D1–D4 tables computed exactly as notebook 11's cells do."""
    XW = h11.build_data()["XW"]
    return {"d1": h11.d1_table(XW), "d2": h11.d2_table(XW), "d3": h11.d3_table(XW), "d4": h11.d4_search(XW)}


def test_d4_best_run_and_search_size(tabs):
    d4 = tabs["d4"]
    assert d4["best"] == "BMV(0.5)"
    assert shown(d4["sr_star"], 0.0541, 4) and shown(d4["sr_star"] * np.sqrt(252), 0.86, 2)
    assert len(d4["srk"]) == 23 and d4["srk"].idxmax() == "BMV(0.5)"
    assert d4["nK"] == 3 and shown(d4["k_er"], 2.67, 2)
    assert np.isclose(d4["var_all"], 2.198e-05, rtol=5e-4) and np.isclose(d4["v0"], 2.239e-04, rtol=5e-4)


def test_d4_deflated_sharpe_cases(tabs):
    dsr = tabs["d4"]["dsr"].set_index("case")
    expected = {   # K, V, SR0,K, s0,K, DSR as printed in the notebook table
        "K=23, V_all":          (23, 2.198e-05, 0.0092, 0.0024, 1.0000),
        "K=nK, V_all":          (3, 2.198e-05, 0.0040, 0.0035, 1.0000),
        "K=round(K_er), V_all": (3, 2.198e-05, 0.0040, 0.0035, 1.0000),
        "sharp-null K=23, V0":  (23, 2.239e-04, 0.0294, 0.0077, 0.9993),
        "sharp-null K=nK, V0":  (3, 2.239e-04, 0.0128, 0.0112, 0.9999),
    }
    assert set(dsr.index) == set(expected)
    for case, (K, var, sr0k, s0k, d) in expected.items():
        row = dsr.loc[case]
        assert row["K"] == K and np.isclose(row["var"], var, rtol=5e-4), case
        assert shown(row["sr0K"], sr0k, 4) and shown(row["s0K"], s0k, 4) and shown(row["dsr"], d, 4), case


def test_d1_psr_and_mintrl_ranges(tabs):
    d1 = tabs["d1"].set_index(["run", "window"])
    full = d1.xs("full", level="window")
    assert full["PSR0"].idxmin() == "BL(0.1,S)" and shown(full["PSR0"].min(), 0.9960, 4)
    assert shown(full["PSR0"].max(), 1.0000, 4)
    assert shown(d1.xs("train (<= split)", level="window")["PSR0"].min(), 0.9864, 4)
    test = d1.xs("test (> split)", level="window")["PSR0"]
    for run, v in {"EW": 0.9501, "GMV(S)": 0.9208, "HRP(S)": 0.9451, "BL(0.1,S)": 0.8967}.items():
        assert shown(test[run], v, 4), run
    assert int((test.round(4) >= 0.95).sum()) == 6 and len(test) == 9
    assert shown(full["gen_over_iid"].min(), 0.95, 2) and shown(full["gen_over_iid"].max(), 1.03, 2)
    assert shown(full["gen_over_NW"].min(), 0.89, 2) and shown(full["gen_over_NW"].max(), 1.04, 2)
    # full-window MinTRL (years): 2.9 (60/40) … 6.6 (BL(0.1,S), GMV(S))
    assert full["MinTRL_yrs"].idxmin() == "60/40" and shown(full["MinTRL_yrs"].min(), 2.9, 1)
    assert shown(full["MinTRL_yrs"].max(), 6.6, 1)
    assert {r for r in full.index if shown(full.loc[r, "MinTRL_yrs"], 6.6, 1)} == {"BL(0.1,S)", "GMV(S)"}
    assert shown(full.loc["BMV(0.3)", "MinTRL_yrs"], 3.1, 1)


def test_d2_power_ranges_and_years_to_80(tabs):
    d2 = tabs["d2"]
    pw = lambda window, sr1: (100 * d2[(d2["window"] == window) & (d2["SR1_ann"] == sr1)]["power"]).round().astype(int)
    assert (pw("full", 0.25).min(), pw("full", 0.25).max()) == (27, 30)
    assert (pw("full", 0.5).min(), pw("full", 0.5).max()) == (66, 73)
    assert (pw("test (> split)", 0.5).min(), pw("test (> split)", 0.5).max()) == (23, 25)
    assert (pw("test (> split)", 0.75).min(), pw("test (> split)", 0.75).max()) == (38, 43)
    yrs = lambda sr1: d2[(d2["window"] == "full") & (d2["SR1_ann"] == sr1)]["T80_yrs"]
    assert (round(yrs(0.5).min()), round(yrs(0.5).max())) == (21, 26)
    assert (round(yrs(0.75).min()), round(yrs(0.75).max())) == (9, 11)
    assert (round(yrs(0.25).min()), round(yrs(0.25).max())) == (84, 102)


def test_d3_power_against_ew(tabs):
    d3 = tabs["d3"]
    se = d3.groupby("method")["se"].first()
    assert se.idxmin() == "BL(0.1,S)" and shown(se.min(), 0.10, 2)
    assert se.idxmax() == "MaxSharpe(S)" and shown(se.max(), 0.26, 2)
    g = lambda delta: d3[d3["Delta"] == delta].set_index("method")
    y02 = g(0.2)["Y80_unadj_yrs"]
    assert y02.idxmin() == "BL(0.1,S)" and round(y02.min()) == 34
    assert y02.idxmax() == "MaxSharpe(S)" and round(y02.max()) == 220
    y01 = g(0.1)["Y80_unadj_yrs"]
    assert (round(y01.min()), round(y01.max())) == (137, 879)
    p02 = (100 * g(0.2)["power_unadj"]).round().astype(int)
    assert (p02.min(), p02.max()) == (12, 51)
    b02 = (100 * g(0.2)["power_bonf"]).round().astype(int)
    assert (b02.min(), b02.max()) == (3, 24)
    assert round(100 * g(0.3).loc["BL(0.1,S)", "power_unadj"]) == 85
