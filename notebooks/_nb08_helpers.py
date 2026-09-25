"""Notebook-08-only run construction and table assembly.

The statistics come from maplab.robust / maplab.inference; this module only
builds the Phase 1 runs, the summary table, the excess-return frame and the
drawdown dates that notebook 08 prints, asserts and plots.
"""
import pandas as pd

import maplab as ml
from maplab import (
    GMV, MaxSharpe, BlackLitterman, EqualWeight, BetaTargetMinVar, MostDiversified,
    EqualRiskContribution, HierarchicalRiskParity, backtest, summary,
)
from maplab import FixedWeight

# Display label -> registrations/reproduction.toml key, in α-family order.
REPRO_KEYS = {
    "GMV(S)": "GMV_sample",
    "GMV(LW)": "GMV_ledoit_wolf",
    "GMV(OAS)": "GMV_oas",
    "MaxSharpe(S)": "MaxSharpe_sample",
    "MaxSharpe(LW)": "MaxSharpe_ledoit_wolf",
    "MaxSharpe(OAS)": "MaxSharpe_oas",
    "BL(0.1,S)": "BL_k0.1_sample",
    "BL(0.1,LW)": "BL_k0.1_ledoit_wolf",
    "BL(0.1,OAS)": "BL_k0.1_oas",
    "BL(0.2,S)": "BL_k0.2_sample",
    "EW": "EW",
    "BMV(0.3)": "BetaMinVar_b0.3_sample",
    "BMV(0.5)": "BetaMinVar_b0.5_sample",
    "MDP(S)": "MDP_sample",
    "MDP(LW)": "MDP_ledoit_wolf",
    "IV": "IV_sample",
    "ERC(S)": "ERC_sample",
    "ERC(LW)": "ERC_ledoit_wolf",
    "HRP(S)": "HRP_sample",
    "HRP(LW)": "HRP_ledoit_wolf",
    "HRP[positional]": "HRP_positional_sample",
    "HRP[ward]": "HRP_ward_sample",
    "IVP": "IVP_sample",
}

ALPHA_FAMILY = list(REPRO_KEYS)
CORE = ["EW", "GMV(S)", "MaxSharpe(S)", "BMV(0.3)", "BL(0.1,S)", "MDP(S)", "ERC(S)", "HRP(S)"]
REF = ["IV", "IVP"]
WINDOWS = ("full", "train (<= split)", "test (> split)")


def make_strategies() -> dict:
    """Fresh instances, constructed exactly as in notebooks 01-07."""
    S, LW, OAS = ml.sample_cov, ml.ledoit_wolf_cov, ml.oas_cov
    return {
        "GMV(S)": GMV(cov_estimator=S),
        "GMV(LW)": GMV(cov_estimator=LW),
        "GMV(OAS)": GMV(cov_estimator=OAS),
        "MaxSharpe(S)": MaxSharpe(cov_estimator=S),
        "MaxSharpe(LW)": MaxSharpe(cov_estimator=LW),
        "MaxSharpe(OAS)": MaxSharpe(cov_estimator=OAS),
        "BL(0.1,S)": BlackLitterman(cov_estimator=S, k=0.1),
        "BL(0.1,LW)": BlackLitterman(cov_estimator=LW, k=0.1),
        "BL(0.1,OAS)": BlackLitterman(cov_estimator=OAS, k=0.1),
        "BL(0.2,S)": BlackLitterman(k=0.2),
        "EW": EqualWeight(),
        "BMV(0.3)": BetaTargetMinVar(beta_target=0.3),
        "BMV(0.5)": BetaTargetMinVar(beta_target=0.5),
        "MDP(S)": MostDiversified(cov_estimator=S),
        "MDP(LW)": MostDiversified(cov_estimator=LW),
        "IV": ml.InverseVol(),
        "ERC(S)": EqualRiskContribution(cov_estimator=S),
        "ERC(LW)": EqualRiskContribution(cov_estimator=LW),
        "HRP(S)": HierarchicalRiskParity(cov_estimator=S),
        "HRP(LW)": HierarchicalRiskParity(cov_estimator=LW),
        "HRP[positional]": HierarchicalRiskParity(bisection="positional"),
        "HRP[ward]": HierarchicalRiskParity(linkage="ward"),
        "IVP": ml.InverseVariance(),
    }


def build_runs(simple_returns, panel) -> dict:
    """Backtest every α-family strategy: label -> {net, wlog, diag, strat}."""
    results = {}
    for name, strat in make_strategies().items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}
    return results


def summary_table(results: dict, names, split_ts, rf_daily) -> pd.DataFrame:
    """Phase 1 summary table (nb07 §4 windows and columns) for `names`."""
    rows = []
    for name in names:
        net = results[name]["net"]
        turnover = results[name]["diag"]["turnover_per_rebalance"]
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
    tbl = pd.DataFrame(rows).set_index(["strategy", "window"])
    tbl = tbl[["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate", "ann_turnover"]]
    tbl["cost_drag"] = tbl["ann_turnover"] * ml.COST_BPS / 1e4
    return tbl


def excess_frame(results: dict, names, rf_daily) -> pd.DataFrame:
    """Daily net − rf for `names` on the common scoring index (no NaN)."""
    idx = results[names[0]]["net"].index
    for name in names:
        if not results[name]["net"].index.equals(idx):
            raise ValueError(f"excess_frame: {name} net index differs from {names[0]}")
    net = pd.DataFrame({name: results[name]["net"] for name in names})
    X = net.sub(rf_daily.reindex(idx), axis=0)
    if X.isna().any().any():
        raise ValueError("excess_frame: NaN after subtracting rf")
    return X


def window_frames(X: pd.DataFrame, split_ts) -> dict:
    """The summary-table windows: full, train (≤ split), test (> split)."""
    return {
        "full": X,
        "train (<= split)": X.loc[X.index <= split_ts],
        "test (> split)": X.loc[X.index > split_ts],
    }


def drawdown_dates(results: dict, names) -> pd.DataFrame:
    """Max drawdown with its peak, trough and recovery dates (first day the
    wealth regains the peak; NaT if never) per run, full window."""
    rows = []
    for name in names:
        wealth = (1.0 + results[name]["net"]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        trough = dd.idxmin()
        peak = wealth.loc[:trough].idxmax()
        after = wealth.loc[trough:]
        rec = after.index[after >= wealth.loc[peak]]
        rows.append({"strategy": name, "max_dd": float(dd.min()), "peak": peak.date(),
                     "trough": trough.date(), "recovery": rec[0].date() if len(rec) else pd.NaT})
    return pd.DataFrame(rows).set_index("strategy")


# ── 60/40 benchmark and active statistics (review pass) ─────────────────────

BENCH = ["60/40"]
BENCH_REPRO_KEYS = {"60/40": "SixtyForty_SPY_IEF"}


def make_benchmarks() -> dict:
    """60/40 (SPY/IEF), constructed as in notebooks 01-07."""
    return {"60/40": FixedWeight({"SPY": 0.6, "IEF": 0.4}, name="60/40")}


def build_benchmarks(simple_returns, panel) -> dict:
    """Backtest the benchmark runs: label -> {net, wlog, diag, strat}."""
    results = {}
    for name, strat in make_benchmarks().items():
        net, wlog, diag = backtest(strat, simple_returns, panel)
        results[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strat}
    return results


def active_stats(runs: dict, names, benches, split_ts) -> pd.DataFrame:
    """Per benchmark b and run m (m != b): annualized active return (mean of
    daily net_m - net_b x TRADING_DAYS) and tracking error (sd, ddof=1, x
    sqrt(TRADING_DAYS)) on the full and test (> split) windows, and the
    full-window correlation of daily net returns. Descriptive only."""
    td = ml.TRADING_DAYS
    rows = []
    for b in benches:
        rb = runs[b]["net"]
        for m in names:
            if m == b:
                continue
            rm = runs[m]["net"]
            if not rm.index.equals(rb.index):
                raise ValueError(f"active_stats: {m} and {b} net indices differ")
            d = rm - rb
            dt = d.loc[d.index > split_ts]
            rows.append({"benchmark": b, "strategy": m,
                         "active_ret": float(d.mean() * td), "TE": float(d.std(ddof=1) * td ** 0.5),
                         "corr": float(rm.corr(rb)),
                         "active_ret_test": float(dt.mean() * td), "TE_test": float(dt.std(ddof=1) * td ** 0.5)})
    return pd.DataFrame(rows).set_index(["benchmark", "strategy"])
