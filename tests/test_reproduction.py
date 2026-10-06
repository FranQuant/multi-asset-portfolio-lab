"""Re-runs every registered backtest on the real price cache and checks round(Sharpe, 2)
against registrations/reproduction.toml in the full/train/test windows, plus the stored
notebook 08 OLS α t-statistics and notebook 09 forecast-bias q values.
Skipped without data/cache/prices.parquet. Slow part: `pytest -m repro`."""
import sys
import tomllib
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
    """Runs built exactly as notebook 08 does (n08-data, then h.build_runs / h.build_benchmarks)."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb08 as h8
        from helpers import nb07 as h7
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
    return {
        "h8": h8, "h7": h7, "panel": panel, "simple_returns": simple_returns, "rf_daily": rf_daily,
        "MKT_simple": (simple_returns["SPY"] - rf_daily).dropna(), "split_ts": split_ts, "rdates": rdates,
        "results": h8.build_runs(simple_returns, panel),
        "bench": h8.build_benchmarks(simple_returns, panel),
    }


def test_sharpe_matches_toml(nb):
    h8 = nb["h8"]
    with open(ROOT / "registrations" / "reproduction.toml", "rb") as f:
        keys = set(tomllib.load(f))
    assert set(h8.REPRO_KEYS.values()) | set(h8.BENCH_REPRO_KEYS.values()) == keys

    summary = h8.summary_table(nb["results"], h8.ALPHA_FAMILY, nb["split_ts"], nb["rf_daily"])
    ml.inference.check_reproduction(summary, h8.REPRO_KEYS)
    bench_summary = h8.summary_table(nb["bench"], h8.BENCH, nb["split_ts"], nb["rf_daily"])
    ml.inference.check_reproduction(bench_summary, h8.BENCH_REPRO_KEYS)


def test_alpha_tstats(nb):
    # mirrors notebook 08's gate
    expected = {"GMV(S)": 1.66, "MaxSharpe(S)": 2.08, "MDP(S)": 2.02, "HRP(S)": 2.02,
                "HRP(LW)": 1.99, "HRP[positional]": 1.92, "HRP[ward]": 1.93}
    capm = ml.inference.capm_table(nb["results"], nb["h8"].ALPHA_FAMILY, nb["MKT_simple"], nb["rf_daily"])
    for name, exp in expected.items():
        got = round(float(capm.loc[name, "t_alpha"]), 2)
        assert got == exp, f"{name}: t_alpha {got} != {exp}"


def test_forecast_bias_q(nb):
    # mirrors notebook 09
    expected = {"HRP(S)": -0.0999, "GMV(S)": -0.0356, "MDP(S)": -0.0518, "ERC(S)": -0.1049, "EW": -0.1885}
    q_table, _, _ = nb["h7"].forecast_bias_table(
        nb["panel"], nb["rdates"], nb["results"], nb["simple_returns"], nb["h8"].CORE)
    q_mean = q_table.groupby("method")["q"].mean()
    for name, exp in expected.items():
        got = round(float(q_mean[name]), 4)
        assert got == exp, f"{name}: mean q {got} != {exp}"
