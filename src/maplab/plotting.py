"""Shared plotting style and family colors.

Import `apply_style()` at the top of every notebook so all charts match.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

FAMILY_COLORS = {
    "Return-based":  "#1f4e79",   # MV, MSR, BL
    "Risk-based":    "#2e7d32",   # GMV, MDP, RP, HRP
    "Signal-based":  "#c0392b",   # TSMOM, factor tilts
    "Benchmark":     "#555555",   # EW
    "Overlay":       "#8e44ad",   # VMP
}


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ── Shared notebook figures ─────────────────────────────────────────────────
# Lifted from the figure cells of notebooks 03/05/06/07. Each function builds
# and returns the Figure; the notebook shows it.

import re

import numpy as np

from .contract import ASSET_GROUPS, GROUP_OF
from .diagnostics import long_only_frontier, port_vol, weights_by_group

# Snapshot-map markers (nb07 F1).
METHOD_STYLES = {
    "HRP":         {"marker": "o", "s": 170, "color": "#8e44ad", "label": "HRP"},
    "GMV":         {"marker": "*", "s": 200, "color": FAMILY_COLORS["Risk-based"], "label": "GMV"},
    "MDP":         {"marker": "P", "s": 140, "color": FAMILY_COLORS["Risk-based"], "label": "MDP"},
    "ERC":         {"marker": "X", "s": 160, "color": "#d4a017", "label": "ERC"},
    "MaxSharpe":   {"marker": "*", "s": 200, "color": FAMILY_COLORS["Return-based"], "label": "MaxSharpe"},
    "EqualWeight": {"marker": "D", "s": 90, "color": FAMILY_COLORS["Benchmark"], "label": "EqualWeight"},
    "IVP":         {"marker": "^", "s": 110, "color": "#555555", "label": "IVP"},
    "IV":          {"marker": "^", "s": 110, "color": "#555555", "label": "IV"},
}

# Time-series line colors (nb07 F4).
METHOD_COLORS = {
    "HRP": "#8e44ad",
    "ERC": "#d4a017",
    "MDP": FAMILY_COLORS["Risk-based"],
    "GMV": "#555555",
    "EW": "#999999",
}


def _base_name(label: str) -> str:
    return re.sub(r"\([^()]*\)$", "", label)


def _style_for(label: str) -> dict:
    """METHOD_STYLES entry for a run label, ignoring a trailing "(S)",
    "(LW)", "(sample)", ... estimator tag."""
    return METHOD_STYLES[_base_name(label)]


def _color_for(label: str):
    return METHOD_COLORS.get(_base_name(label))


def snapshot_map(Sigma, mu, weights: dict, title=None, highlight="UUP", figsize=(10, 6),
                 legend_outside=True, n_random=20000, dirichlet_alpha=0.3, seed=7, styles=None):
    """Snapshot risk/return map (nb07 F1): random long-only cloud, assets
    colored by group, long-only frontier (dashed below GMV), one marker per
    method in `weights` (name -> weight vector in Sigma's column order)."""
    from .models import _min_variance_long_only

    Sigma_np = Sigma.to_numpy()
    mu_np = mu.to_numpy()
    n = Sigma_np.shape[0]
    tickers = list(Sigma.columns)
    style_map = {**METHOD_STYLES, **(styles or {})}

    rng_mc = np.random.default_rng(seed)
    mc_weights = rng_mc.dirichlet(np.full(n, dirichlet_alpha), size=n_random)
    mc_ret = mc_weights @ mu_np
    mc_vol = np.sqrt(np.einsum("ij,jk,ik->i", mc_weights, Sigma_np, mc_weights))

    frontier_vol, frontier_ret = long_only_frontier(Sigma_np, mu_np)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(mc_vol, mc_ret, s=2, alpha=0.15, color="#bbbbbb", zorder=0,
               label=f"random long-only portfolios (Dirichlet a={dirichlet_alpha:g})")

    asset_vol = np.sqrt(np.diag(Sigma_np))
    group_order = list(ASSET_GROUPS.keys())
    group_colors = dict(zip(group_order, plt.cm.tab10(np.linspace(0, 1, len(group_order)))))
    for i, tkr in enumerate(tickers):
        grp = GROUP_OF[tkr]
        is_hl = tkr == highlight
        ax.scatter(asset_vol[i], mu_np[i], s=30, color=group_colors[grp], zorder=5 if is_hl else 2)
        ax.annotate(tkr, (asset_vol[i], mu_np[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points",
                    color="#c0392b" if is_hl else "#666666",
                    fontweight="bold" if is_hl else "normal",
                    zorder=5 if is_hl else 2)

    w_gmv = np.asarray(weights["GMV"]) if "GMV" in weights else _min_variance_long_only(Sigma)
    gmv_ret = float(w_gmv @ mu_np)

    efficient = frontier_ret >= gmv_ret
    ax.plot(frontier_vol[~efficient], frontier_ret[~efficient], color="#999999", ls="--", lw=1.2, zorder=3)
    ax.plot(frontier_vol[efficient], frontier_ret[efficient], color="#999999", lw=1.8, zorder=3, label="long-only frontier")

    for name, w in weights.items():
        w = np.asarray(w)
        st = style_map[name] if name in style_map else style_map[_base_name(name)]
        ret = float(w @ mu_np); vol = port_vol(w, Sigma_np)
        ax.scatter([vol], [ret], marker=st["marker"], s=st["s"], color=st["color"], zorder=4,
                   label=st.get("label", name))

    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized arithmetic expected return (mu)")
    if title is not None:
        ax.set_title(title)
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    else:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def group_stackplot(results: dict, names: list, figsize=None,
                    title_fmt="{name} -- target weights by asset group", split_ts=None):
    """Target weights by asset group, one stacked panel per run (nb06/nb07 F3)."""
    if figsize is None:
        figsize = (9, 13) if len(names) == 4 else (9, 10)
    group_order = list(ASSET_GROUPS.keys())
    group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_order)))

    fig, axes = plt.subplots(len(names), 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, name in zip(axes, names):
        grouped = weights_by_group(results[name]["wlog"])
        ax.stackplot(grouped.index, grouped.T.to_numpy(), labels=group_order, colors=group_colors)
        if split_ts is not None:
            ax.axvline(split_ts, color="black", ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(title_fmt.format(name=name))
    axes[-1].set_xlabel("rebalance date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    return fig


def concentration_panel(results: dict, names: list, rows=("effN", "max_w", "highlight"), ref="EW",
                        highlight="UUP", split_ts=None, figsize=(9.5, 11), suptitle=None, colors=None,
                        titles=None, percent=True, effn_floor=None, split_label=None,
                        xlabel="rebalance date"):
    """effN / max weight / highlight-asset weight over rebalances, one row
    each (nb07 F4; also nb03 cell 15 and nb05 cells 19/22 via rows, figsize,
    titles, percent, effn_floor, split_label, xlabel). Optional row "turnover":
    trailing 12-rebalance sum of diag["turnover_per_rebalance"], first entry
    (initial allocation from cash) dropped."""
    scale = 100 if percent else 1
    unit = " (%)" if percent else ""
    ylabels = {"effN": "effective N", "max_w": f"max weight{unit}", "highlight": f"{highlight} weight{unit}",
               "turnover": f"trailing 12-rebalance one-way turnover{' (%/yr)' if percent else ' (/yr)'}"}

    def series(name, row):
        wlog = results[name]["wlog"]
        if row == "effN":
            return 1.0 / (wlog ** 2).sum(axis=1)
        if row == "max_w":
            return wlog.max(axis=1) * scale
        if row == "highlight":
            return wlog[highlight] * scale
        if row == "turnover":
            tlog = results[name]["diag"]["turnover_per_rebalance"].iloc[1:]
            return tlog.rolling(12, min_periods=12).sum() * scale
        raise ValueError(f"concentration_panel: unknown row {row!r}")

    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]
    for k, (ax, row) in enumerate(zip(axes, rows)):
        for name in names:
            ts = series(name, row)
            color = colors[name] if colors and name in colors else _color_for(name)
            ax.plot(ts.index, ts.to_numpy(), label=name, color=color, lw=1.2)
        if row in ("effN", "turnover") and ref is not None:
            ts = series(ref, row)
            ax.plot(ts.index, ts.to_numpy(), label=ref, color="#999999", lw=0.8, ls="--")
        if row == "effN" and effn_floor is not None:
            ax.axhline(effn_floor, color="#888888", ls="--", lw=1, label=f"effN floor = {effn_floor:g}")
        if split_ts is not None:
            ax.axvline(split_ts, color="black", ls=":", lw=1, label=split_label)
        ax.set_ylabel(ylabels[row])
        if titles is not None:
            ax.set_title(titles[k])
        ax.legend(fontsize=8)
    if xlabel is not None:
        axes[-1].set_xlabel(xlabel)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def capital_vs_risk(weights: dict, Sigma, highlight="UUP", title=None, figsize=None):
    """Capital weight vs risk-contribution share per asset, one panel per
    method in `weights` (name -> pd.Series of weights indexed by ticker);
    risk shares = risk_contributions(w, Sigma) normalized to sum 1."""
    from .models import risk_contributions

    tickers = list(Sigma.columns)
    n = len(tickers)
    x = np.arange(n)
    if figsize is None:
        figsize = (3.2 * len(weights), 3.8)

    fig, axes = plt.subplots(1, len(weights), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (name, w) in zip(axes, weights.items()):
        if set(w.index) != set(tickers):
            raise ValueError(f"capital_vs_risk: weights[{name!r}] index does not match Sigma.columns")
        w = w.reindex(tickers)
        rc = risk_contributions(w, Sigma)
        share = rc / rc.sum()
        cap_colors = ["#c0392b" if t == highlight else "#1f4e79" for t in tickers]
        risk_colors = ["#c0392b" if t == highlight else "#2e7d32" for t in tickers]
        ax.bar(x - 0.2, w.to_numpy(), width=0.4, color=cap_colors, label="capital weight")
        ax.bar(x + 0.2, share.to_numpy(), width=0.4, color=risk_colors, alpha=0.6, label="risk share")
        ax.axhline(1.0 / n, color="black", ls="--", lw=0.8, label="1/N")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=90, fontsize=6)
        ax.set_title(name)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def cumulative_paired(paired_series: dict, pairs, split_ts, beta: dict | None = None, mkt_excess=None,
                      ax=None, title=None, ylabel="cumulative diff (%)", legend_kw=None):
    """Cumulative paired daily differences in % (nb06 F5 / nb07 F5). With
    `beta` ("a - b" -> beta_hat) and `mkt_excess`: solid = beta-adjusted,
    dashed = total; otherwise solid totals only."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9.5, 5))

    if beta is not None:
        for a_name, b_name in pairs:
            pair_key = f"{a_name} - {b_name}"
            diff = paired_series[(a_name, b_name)]
            beta_hat = float(beta[pair_key])
            idx = diff.index.intersection(mkt_excess.index)
            adj = (diff.loc[idx] - beta_hat * mkt_excess.loc[idx]).cumsum() * 100
            total = diff.cumsum() * 100
            line, = ax.plot(adj.index, adj.to_numpy(), lw=1.1, label=pair_key)
            ax.plot(total.index, total.to_numpy(), lw=0.8, ls="--", color=line.get_color())
        ax.axvline(split_ts, color="black", ls=":", lw=1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.legend(**(legend_kw if legend_kw is not None else {"fontsize": 7, "ncol": 2}))
    else:
        for a_name, b_name in pairs:
            diff = paired_series[(a_name, b_name)]
            cum = diff.cumsum() * 100
            ax.plot(cum.index, cum.to_numpy(), lw=1.1, label=f"{a_name} - {b_name}")
        ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("date")
        if title is not None:
            ax.set_title(title)
        ax.legend(**(legend_kw if legend_kw is not None else {"fontsize": 7}))

    if own_fig:
        fig.tight_layout()
    return ax.figure


# ── Phase 2 comparison figures ──────────────────────────────────────────────

def _line_style(label: str, styles: dict | None) -> dict:
    """Line color for a run label: `styles` override, else METHOD_COLORS /
    METHOD_STYLES by base name, else matplotlib's cycle (None)."""
    if styles and label in styles:
        return dict(styles[label])
    base = _base_name(label)
    color = METHOD_COLORS.get(base) or METHOD_STYLES.get(base, {}).get("color")
    return {"color": color} if color is not None else {}


def wealth_drawdown(net: dict, rf, split_ts, styles=None, figsize=(10, 8), title=None):
    """Log-scale wealth (1 = start) with BIL wealth from `rf` (grey dashed)
    on top, underwater drawdown below; `net` is label -> daily net Series."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, (ax_w, ax_d) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    start = end = None
    ymin, ymax = np.inf, -np.inf
    for label, r in net.items():
        wealth = (1.0 + r).cumprod()
        ymin, ymax = min(ymin, float(wealth.min())), max(ymax, float(wealth.max()))
        dd = wealth / wealth.cummax() - 1.0
        st = _line_style(label, styles)
        ax_w.plot(wealth.index, wealth.to_numpy(), lw=1.1, label=label, **st)
        ax_d.plot(dd.index, 100 * dd.to_numpy(), lw=0.9, label=label, **st)
        start = r.index.min() if start is None else min(start, r.index.min())
        end = r.index.max() if end is None else max(end, r.index.max())
    rf_w = (1.0 + rf.loc[start:end]).cumprod()
    ax_w.plot(rf_w.index, rf_w.to_numpy(), color="#999999", ls="--", lw=1.0, label="BIL (rf)")
    ymin, ymax = min(ymin, float(rf_w.min())), max(ymax, float(rf_w.max()))
    ax_w.set_yscale("log")
    lo = max(0.5, np.floor(2 * ymin) / 2)
    hi = np.ceil(2 * ymax) / 2
    ax_w.yaxis.set_major_locator(FixedLocator(np.arange(lo, hi + 1e-9, 0.5)))
    ax_w.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_w.yaxis.set_minor_locator(NullLocator())
    ax_w.set_ylabel("wealth (1 = start, log scale)")
    ax_d.set_ylabel("drawdown (%)")
    ax_d.set_xlabel("date")
    for ax in (ax_w, ax_d):
        ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax_w.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    if title is not None:
        ax_w.set_title(title)
    fig.tight_layout()
    return fig


def forest_plot(df, label_col, point, lo, hi, group=None, ref=None, figsize=None, title=None,
                xlabel=None):
    """Horizontal point + [lo, hi] interval per label; with `group` (e.g. the
    window column) the groups are dodged around each label's row; optional
    reference vertical line at `ref`."""
    labels = list(dict.fromkeys(df[label_col]))
    groups = list(dict.fromkeys(df[group])) if group is not None else [None]
    if figsize is None:
        figsize = (8, 0.45 * len(labels) * max(1, len(groups) * 0.6) + 1.2)
    fig, ax = plt.subplots(figsize=figsize)
    ypos = {lab: len(labels) - 1 - i for i, lab in enumerate(labels)}
    width = 0.6
    offsets = np.linspace(width / 2, -width / 2, len(groups)) if len(groups) > 1 else [0.0]
    for g, off in zip(groups, offsets):
        sub = df if g is None else df[df[group] == g]
        y = np.array([ypos[lab] for lab in sub[label_col]]) + off
        x = sub[point].to_numpy(dtype=float)
        err = np.vstack([x - sub[lo].to_numpy(dtype=float), sub[hi].to_numpy(dtype=float) - x])
        ax.errorbar(x, y, xerr=err, fmt="o", ms=4, capsize=2, lw=1, label=None if g is None else str(g))
    if ref is not None:
        ax.axvline(ref, color="black", ls="--", lw=0.8)
    ax.set_yticks([ypos[lab] for lab in labels])
    ax.set_yticklabels(list(labels))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if group is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    fig.tight_layout()
    return fig


def pair_matrix(M, labels, title, fmt="{:.2f}", vmax=None, figsize=None, cmap="RdBu_r"):
    """Annotated diverging heatmap of an antisymmetric pair matrix (e.g.
    ΔSR t, row − column), diagonal blanked (NaN)."""
    A = np.array(M.to_numpy() if hasattr(M, "to_numpy") else M, dtype=float)
    np.fill_diagonal(A, np.nan)
    labels = list(labels)
    k = len(labels)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 1.0
    if figsize is None:
        figsize = (0.6 * k + 2.5, 0.6 * k + 1.5)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax)
    for i in range(k):
        for j in range(k):
            if np.isfinite(A[i, j]):
                color = "white" if abs(A[i, j]) > 0.6 * vmax else "black"
                ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
