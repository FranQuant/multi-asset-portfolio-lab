"""Notebooks 01-03 checks run on the real price cache: each notebook's runs, rebuilt exactly as its §5 cells do,
reproduce registrations/reproduction.toml (2 dp, full/train/test), and the solver fallback / retry / slack /
infeasible / concentration counts the notebooks report hold."""
from pathlib import Path

import pandas as pd
import pytest

import maplab as ml
from maplab import Panel, BetaTargetMinVar, BlackLitterman, GMV, MaxSharpe, EqualWeight, FixedWeight, backtest, summary

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]

BL01_CONCENTRATED = [
    "2013-12-31", "2014-01-31", "2014-02-28", "2014-03-31", "2014-04-30", "2015-12-31", "2016-01-31",
    "2018-07-31", "2018-10-31", "2018-11-30", "2018-12-31", "2022-05-31", "2023-04-30", "2023-05-31",
    "2023-07-31", "2023-11-30",
]


@pytest.fixture(scope="module")
def nb():
    """Data loaded exactly as the n01-data / n02-data / n03-data cells do."""
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    simple_returns = ml.to_simple_returns(prices)
    rf = ml.load_rf_returns()
    return {"simple_returns": simple_returns, "rf_daily": rf["BIL"],
            "panel": Panel({"prices": prices, "returns": log_returns, "rf": rf}),
            "split_ts": pd.Timestamp(ml.TRAIN_TEST_SPLIT)}


def test_nb01_runs(nb):
    simple_returns, panel, rf_daily, split_ts = nb["simple_returns"], nb["panel"], nb["rf_daily"], nb["split_ts"]

    # cell 685c5aed
    strategies = {
        "GMV": GMV(),
        "MaxSharpe": MaxSharpe(),
        "EqualWeight": EqualWeight(),
        "60/40": FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag}

    # cell n01-summary
    rows = []
    for name, r in results.items():
        net = r["net"]
        turnover = r["diag"]["turnover_per_rebalance"]
        windows = {
            "full": (net.index >= net.index.min(), turnover.index >= turnover.index.min()),
            "train (<= split)": (net.index <= split_ts, turnover.index <= split_ts),
            "test (> split)": (net.index > split_ts, turnover.index > split_ts),
        }
        for window_name, (net_sel, t_sel) in windows.items():
            s = summary(net.loc[net_sel], turnover.loc[t_sel], rf=rf_daily)
            s["strategy"] = name
            s["window"] = window_name
            rows.append(s)

    summary_table = pd.DataFrame(rows).set_index(["strategy", "window"])
    summary_table = summary_table[["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate", "ann_turnover"]]

    ml.inference.check_reproduction(summary_table, {
        "GMV": "GMV_sample", "MaxSharpe": "MaxSharpe_sample",
        "EqualWeight": "EW", "60/40": "SixtyForty_SPY_IEF",
    })
    msr = strategies["MaxSharpe"]
    assert len(msr.fallback_dates) == 0
    assert len(msr.retry_dates) == 0


def test_nb02_runs(nb):
    simple_returns, panel, rf_daily, split_ts = nb["simple_returns"], nb["panel"], nb["rf_daily"], nb["split_ts"]

    # cell bb150f16
    strategies = {
        "BetaMinVar(0.3)": BetaTargetMinVar(beta_target=0.3),
        "BetaMinVar(0.5)": BetaTargetMinVar(beta_target=0.5),
        "GMV": GMV(),
        "EqualWeight": EqualWeight(),
        "60/40": FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    scoring_start = results["GMV"]["diag"]["scoring_start"]
    for beta_target in (0.3, 0.5):
        bench_ret = (beta_target * simple_returns["SPY"] + (1 - beta_target) * rf_daily).loc[scoring_start:]
        results[f"β-match({beta_target})"] = {"net": bench_ret, "wlog": None, "diag": None, "strat": None}

    # cell n02-summary
    rows = []
    for name, r in results.items():
        net = r["net"]
        diag = r["diag"]
        turnover = diag["turnover_per_rebalance"] if diag is not None else None
        win_defs = {
            "full": (net.index >= net.index.min(), None if turnover is None else turnover.index >= turnover.index.min()),
            "train (<= split)": (net.index <= split_ts, None if turnover is None else turnover.index <= split_ts),
            "test (> split)": (net.index > split_ts, None if turnover is None else turnover.index > split_ts),
        }
        for window_name, (net_sel, t_sel) in win_defs.items():
            t_window = turnover.loc[t_sel] if (turnover is not None and t_sel is not None) else None
            s = summary(net.loc[net_sel], t_window, rf=rf_daily)
            s["strategy"] = name
            s["window"] = window_name
            rows.append(s)

    summary_table = pd.DataFrame(rows).set_index(["strategy", "window"])
    summary_table = summary_table[["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate", "ann_turnover"]]

    ml.inference.check_reproduction(summary_table, {
        "BetaMinVar(0.3)": "BetaMinVar_b0.3_sample", "BetaMinVar(0.5)": "BetaMinVar_b0.5_sample",
        "GMV": "GMV_sample", "EqualWeight": "EW", "60/40": "SixtyForty_SPY_IEF",
    })
    for name in ("BetaMinVar(0.3)", "BetaMinVar(0.5)"):
        s = strategies[name]
        assert len(s.slack_dates) == 0, name
        assert len(s.infeasible_dates) == 0, name
        assert len(s.retry_dates) == 0, name
        assert results[name]["diag"]["n_rebalances"] == 208, name


def test_nb03_runs(nb):
    simple_returns, panel, rf_daily, split_ts = nb["simple_returns"], nb["panel"], nb["rf_daily"], nb["split_ts"]

    # cell b07873e6
    strategies = {
        "BL(0.1)": BlackLitterman(k=0.1),
        "BL(0.2)": BlackLitterman(k=0.2),
        "EqualWeight": EqualWeight(),
        "60/40": FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40"),
    }

    results = {}
    for name, strat in strategies.items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}

    # cell n03-summary
    rows = []
    for name, r in results.items():
        net = r["net"]
        turnover = r["diag"]["turnover_per_rebalance"]
        win_defs = {
            "full": (net.index >= net.index.min(), turnover.index >= turnover.index.min()),
            "train (<= split)": (net.index <= split_ts, turnover.index <= split_ts),
            "test (> split)": (net.index > split_ts, turnover.index > split_ts),
        }
        for window_name, (net_sel, t_sel) in win_defs.items():
            s = summary(net.loc[net_sel], turnover.loc[t_sel], rf=rf_daily)
            s["strategy"] = name
            s["window"] = window_name
            rows.append(s)

    summary_table = pd.DataFrame(rows).set_index(["strategy", "window"])
    summary_table = summary_table[["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate", "ann_turnover"]]

    ml.inference.check_reproduction(summary_table, {
        "BL(0.1)": "BL_k0.1_sample", "BL(0.2)": "BL_k0.2_sample",
        "EqualWeight": "EW", "60/40": "SixtyForty_SPY_IEF",
    })
    counts = {n: (len(strategies[n].fallback_dates), len(strategies[n].retry_dates),
                  len(strategies[n].concentration_dates)) for n in ("BL(0.1)", "BL(0.2)")}
    assert counts == {"BL(0.1)": (0, 0, 16), "BL(0.2)": (0, 0, 66)}
    assert results["BL(0.1)"]["diag"]["n_rebalances"] == 208
    assert [str(d.date()) for d in strategies["BL(0.1)"].concentration_dates] == BL01_CONCENTRATED
