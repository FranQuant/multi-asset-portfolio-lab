"""Notebook-09-only run construction, descriptive tables and figures.

The statistics come from maplab.robust / maplab.inference; the Phase 1 run
construction, summary table and excess-return windows are notebook 08's
(_nb08_helpers). This module builds the lookback grid, the H3 descriptive
table and figures F6/F7 that notebook 09 prints, asserts and plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import maplab as ml
from maplab import GMV, MostDiversified, EqualRiskContribution, HierarchicalRiskParity, backtest

import _nb08_helpers as h8

CORE = h8.CORE
LOOKBACKS = (252, 504, 756)
GRID_CLASSES = {
    "GMV": GMV,
    "MDP": MostDiversified,
    "ERC": EqualRiskContribution,
    "HRP": HierarchicalRiskParity,
}
GRID_METHODS = list(GRID_CLASSES)
P1_LABEL = {m: f"{m}(S)" for m in GRID_METHODS}


def grid_label(method: str, lookback: int) -> str:
    return f"{method}(S) {lookback}d"


GRID = [grid_label(m, L) for m in GRID_METHODS for L in LOOKBACKS]


def _run(strat, simple_returns, panel, start=None) -> dict:
    net, wlog, diag = backtest(strat, simple_returns, panel, start=start)
    return {"net": net, "wlog": wlog, "diag": diag, "strat": strat}


def build_core_runs(simple_returns, panel) -> dict:
    """The 8 notebook-08 core runs, same constructors, default start."""
    strats = h8.make_strategies()
    return {name: _run(strats[name], simple_returns, panel) for name in CORE}


def common_start(returns_index: pd.DatetimeIndex, min_rows: int) -> pd.Timestamp:
    """First rebalance label with at least `min_rows` return rows strictly before it."""
    for d in ml.rebalance_dates(returns_index):
        if int((returns_index < d).sum()) >= min_rows:
            return d
    raise ValueError(f"common_start: no rebalance label with {min_rows} prior rows")


def build_grid_runs(simple_returns, panel, start) -> dict:
    """GMV/MDP/ERC/HRP (sample Σ) x lookback, fresh instances, all from `start`."""
    results = {}
    for m, cls in GRID_CLASSES.items():
        for L in LOOKBACKS:
            strat = cls(cov_estimator=ml.sample_cov, lookback=L)
            results[grid_label(m, L)] = _run(strat, simple_returns, panel, start=start)
    return results


def weight_crosscheck(grid: dict, core: dict) -> pd.Series:
    """max |Δw| between grid 252d and default-start target weights on the grid dates."""
    out = {}
    for m in GRID_METHODS:
        wg = grid[grid_label(m, 252)]["wlog"]
        wc = core[P1_LABEL[m]]["wlog"].loc[wg.index, wg.columns]
        out[m] = float((wg - wc).abs().to_numpy().max())
    return pd.Series(out, name="max_abs_dw")


def h3_desc_table(grid: dict, summary_tbl: pd.DataFrame, at_date) -> pd.DataFrame:
    """Per (method, lookback): full-window annual turnover, median effN and
    median UUP target weight over the rebalances, UUP target weight at `at_date`."""
    at_date = pd.Timestamp(at_date)
    rows = []
    for m in GRID_METHODS:
        for L in LOOKBACKS:
            lab = grid_label(m, L)
            wlog = grid[lab]["wlog"]
            if at_date not in wlog.index:
                raise KeyError(f"h3_desc_table: {at_date.date()} is not a rebalance label of {lab}")
            effn = wlog.apply(lambda w: ml.diagnostics.effective_n(w.to_numpy()), axis=1)
            rows.append({
                "method": m, "lookback": L,
                "ann_turnover": float(summary_tbl.loc[(lab, "full"), "ann_turnover"]),
                "median_effN": float(effn.median()),
                "median_w_UUP": float(wlog["UUP"].median()),
                f"w_UUP_{at_date.date()}": float(wlog.loc[at_date, "UUP"]),
            })
    return pd.DataFrame(rows).set_index(["method", "lookback"])


def lookback_panels(desc: pd.DataFrame, sharpe_full: pd.Series, figsize=(13, 4)):
    """Figure F6: full-window Sharpe, annual turnover and median effN by lookback,
    one line per method. `sharpe_full` is indexed by grid label."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [("Sharpe (full, vs BIL)", None), ("annual turnover", "ann_turnover"), ("median effN", "median_effN")]
    for ax, (title, col) in zip(axes, panels):
        for m in GRID_METHODS:
            if col is None:
                y = [float(sharpe_full[grid_label(m, L)]) for L in LOOKBACKS]
            else:
                y = [float(desc.loc[(m, L), col]) for L in LOOKBACKS]
            ax.plot(LOOKBACKS, y, marker="o", lw=1.3, color=ml.plotting.METHOD_COLORS.get(m), label=P1_LABEL[m])
        ax.set_xticks(LOOKBACKS)
        ax.set_xlabel("lookback (days)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def uup_paths(grid: dict, split_ts, figsize=(12, 7.5)):
    """Figure F7: UUP target weight at each rebalance for the three lookbacks,
    one panel per method, shared y, train/test split marked."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    styles = {252: dict(lw=1.3, ls="-"), 504: dict(lw=1.1, ls="--"), 756: dict(lw=1.1, ls=":")}
    for ax, m in zip(axes.ravel(), GRID_METHODS):
        color = ml.plotting.METHOD_COLORS.get(m)
        for L in LOOKBACKS:
            w = grid[grid_label(m, L)]["wlog"]["UUP"]
            ax.plot(w.index, w.to_numpy(), color=color, label=f"{L}d", **styles[L])
        ax.axvline(split_ts, color="black", ls=":", lw=1)
        ax.set_title(P1_LABEL[m], fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("UUP target weight")
    for ax in axes[1, :]:
        ax.set_xlabel("rebalance date")
    fig.tight_layout()
    return fig


def bias_decomposition_bars(decomp_full: pd.DataFrame, figsize=(10, 4.5)):
    """Figure F8b: per method, grouped bars of q̄, ½·log v̄ and −J (full window).
    `decomp_full` is indexed by method with columns q_bar, half_log_vbar, J."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(decomp_full))
    width = 0.27
    bars = [("q_bar", decomp_full["q_bar"], "#2c3e50"),
            ("½·log v̄", decomp_full["half_log_vbar"], "#1f77b4"),
            ("−J", -decomp_full["J"], "#c0392b")]
    for k, (lab, vals, color) in enumerate(bars):
        ax.bar(x + (k - 1) * width, vals.to_numpy(), width, label=lab, color=color)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(decomp_full.index), rotation=30, ha="right")
    ax.set_title("Forecast-bias decomposition, full window: q̄ = ½·log v̄ − J")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def train_test_arrows(summary_tbl: pd.DataFrame, names, figsize=(8.5, 6)):
    """Figure F9: one arrow per run from its train (ann_vol, ann_return) point
    (hollow) to its test point (filled), labelled at the test end. Colours from
    METHOD_COLORS by base name; a missing or already-used colour falls back to
    the next unused colour of the matplotlib cycle."""
    fig, ax = plt.subplots(figsize=figsize)
    cycle = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    used, xs, ys = set(), [], []
    for name in names:
        tr = summary_tbl.loc[(name, "train (<= split)")]
        te = summary_tbl.loc[(name, "test (> split)")]
        train = (float(tr["ann_vol"]), float(tr["ann_return"]))
        test = (float(te["ann_vol"]), float(te["ann_return"]))
        c = ml.plotting._color_for(name)
        if c is None or c in used:
            c = next(cc for cc in cycle if cc not in used)
        used.add(c)
        ax.plot(*train, "o", ms=6, mfc="none", mec=c, mew=1.2)
        ax.plot(*test, "o", ms=6, color=c)
        ax.annotate("", xy=test, xytext=train, arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        ax.annotate(name, test, textcoords="offset points", xytext=(5, 3), fontsize=8, color=c)
        xs += [train[0], test[0]]
        ys += [train[1], test[1]]
    for vals, setter in ((xs, ax.set_xlim), (ys, ax.set_ylim)):
        lo, hi = min(vals), max(vals)
        pad = 0.08 * (hi - lo)
        setter(lo - pad, hi + pad)
    ax.set_xlabel("annualized vol")
    ax.set_ylabel("annualized return")
    ax.set_title("Train (hollow) → test (filled), core runs")
    fig.tight_layout()
    return fig


def variance_ratio_profile(q_table: pd.DataFrame, methods, spike_asof="2020-02-29") -> pd.DataFrame:
    """Per method, full window, v = (realized_vol / exante_vol)² over segments
    sorted by asof: median v, share of v < 1, the two largest segments, the
    five largest segments' share of Σv, and v̄ without the `spike_asof` segment."""
    spike_asof = pd.Timestamp(spike_asof)
    rows = []
    for m in methods:
        seg = q_table[q_table["method"] == m].sort_values("asof").reset_index(drop=True)
        v = (seg["realized_vol"] / seg["exante_vol"]) ** 2
        if int((seg["asof"] == spike_asof).sum()) != 1:
            raise KeyError(f"variance_ratio_profile: {m} has no single segment with asof {spike_asof.date()}")
        top = v.sort_values(ascending=False).index
        rows.append({
            "method": m,
            "median_v": float(v.median()),
            "share_v_lt_1": float((v < 1).mean()),
            "largest_asof": seg.loc[top[0], "asof"].date(),
            "largest_v": float(v[top[0]]),
            "second_asof": seg.loc[top[1], "asof"].date(),
            "second_v": float(v[top[1]]),
            "top5_share_sum_v": float(v[top[:5]].sum() / v.sum()),
            "vbar_ex_spike": float(v[seg["asof"] != spike_asof].mean()),
        })
    return pd.DataFrame(rows).set_index("method")
