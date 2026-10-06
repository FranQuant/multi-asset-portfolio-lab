"""Code shared by notebooks 01–07."""
import pandas as pd

import maplab as ml
from maplab import summary

SUMMARY_COLS = ["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate", "ann_turnover"]
WINDOWS = {
    "full": lambda idx, split_ts: idx >= idx.min(),
    "train (<= split)": lambda idx, split_ts: idx <= split_ts,
    "test (> split)": lambda idx, split_ts: idx > split_ts,
}


def summary_by_window(results, split_ts, rf, *, cost_drag=False):
    """One row per (strategy, window) with maplab.summary on net returns and, when a run
    has diagnostics, its per-rebalance turnover; windows are full / train (<= split) /
    test (> split). A run with diag None gets NaN turnover. cost_drag adds
    ann_turnover * ml.COST_BPS / 1e4. Returns a (strategy, window)-indexed DataFrame."""
    rows = []
    for name, r in results.items():
        net = r["net"]
        diag = r["diag"]
        turnover = diag["turnover_per_rebalance"] if diag is not None else None
        for window_name, sel in WINDOWS.items():
            t_window = None if turnover is None else turnover.loc[sel(turnover.index, split_ts)]
            s = summary(net.loc[sel(net.index, split_ts)], t_window, rf=rf)
            s["strategy"] = name
            s["window"] = window_name
            rows.append(s)
    table = pd.DataFrame(rows).set_index(["strategy", "window"])[SUMMARY_COLS]
    if cost_drag:
        table["cost_drag"] = table["ann_turnover"] * ml.COST_BPS / 1e4
    return table


def overview_table(summary_table, *, rename=None, order=None):
    """Sharpe (full / train / test), return, vol, max DD and turnover, one row per run.
    rename maps run names (EqualWeight -> EW); order is the list of rows to keep, in order."""
    full = summary_table.xs("full", level="window")
    train = summary_table.xs("train (<= split)", level="window")
    test = summary_table.xs("test (> split)", level="window")
    overview = pd.DataFrame({
        "Sharpe full": full["sharpe"], "Sharpe train": train["sharpe"], "Sharpe test": test["sharpe"],
        "return (%)": full["ann_return"], "vol (%)": full["ann_vol"],
        "max DD (%)": full["max_dd"], "turnover (%/yr)": full["ann_turnover"],
    })
    if rename:
        overview = overview.rename(index=rename)
    if order is not None:
        overview = overview.loc[order]
    return overview.rename_axis(None)


def show_overview(overview, *, int_turnover=True):
    """ml.reporting.show_table of an overview_table. int_turnover=False leaves turnover
    at pct_dp decimals (for a table with NaN turnover, which cannot be cast to int)."""
    pct = ["return (%)", "vol (%)", "max DD (%)", "turnover (%/yr)"]
    return ml.reporting.show_table(overview, pct=pct, pct_dp=1, dp=2,
                                   int_cols=["turnover (%/yr)"] if int_turnover else [])
