"""Notebook-03-only code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures/values; notebook 03 prints,
asserts and shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import maplab as ml
from maplab import BlackLitterman, tangency_closed_form


def mechanics_table(post):
    """§3 mechanics table at the split snapshot: Pi, sigma, 12-1 momentum,
    view sign, Q and mu_BL (all in %)."""
    mech_table = pd.DataFrame({
        "Pi_%": 100 * post["pi"],
        "sigma_%": 100 * post["sigma"],
        "mom12-1_%": 100 * post["momentum"],
        "sign": post["signs"],
        "Q_%": 100 * post["q"],
        "mu_BL_%": 100 * post["mu_bl"],
    })
    return mech_table


def design_checks(panel, asof_split, post):
    """§3 checks: k=0 -> EW, tau invariance, closed-form identity, and the
    unconstrained BL(k=0.1) tangency weights."""
    # check 1: k=0 -> long-only weights == EW, mu_bl == pi
    bl_k0 = BlackLitterman(k=0.0)
    w_k0 = bl_k0(panel, asof_split)
    w_ew_series = pd.Series(1.0 / len(ml.UNIVERSE), index=ml.UNIVERSE)
    k0_dev = (w_k0 - w_ew_series).abs().max()

    # check 2: tau invariance
    mu_bl_tau_lo = BlackLitterman(k=0.1, tau=0.01).posterior(panel, asof_split)["mu_bl"]
    mu_bl_tau_hi = BlackLitterman(k=0.1, tau=1.0).posterior(panel, asof_split)["mu_bl"]
    tau_dev = (mu_bl_tau_lo - mu_bl_tau_hi).abs().max()

    # check 3: closed-form identity
    Sigma_np = post["Sigma"].to_numpy()
    D = np.diag(np.diag(Sigma_np))
    lhs = np.linalg.solve(Sigma_np, post["mu_bl"].to_numpy())
    rhs = post["delta"] * post["w_ew"].to_numpy() + 0.1 * np.linalg.solve(
        Sigma_np + D, (post["signs"] * post["sigma"]).to_numpy()
    )
    identity_dev = np.abs(lhs - rhs).max()

    # unconstrained BL tangency
    w_unc = tangency_closed_form(post["mu_bl"], post["Sigma"], 0.0)
    return {"k0_dev": k0_dev, "tau_dev": tau_dev, "identity_dev": identity_dev, "w_unc": w_unc}


def k_sensitivity(panel, asof_split):
    """§4 cautionary snapshot: effN / max weight / unconstrained gross
    leverage per k, and the long-only weights per k."""
    rows = []
    wtab = {}
    for k in (0.1, 0.2, 0.4):
        bl_k = BlackLitterman(k=k)
        w = bl_k(panel, asof_split)
        post_k = bl_k.posterior(panel, asof_split)
        effN = 1.0 / float((w ** 2).sum())
        w_unc_k = tangency_closed_form(post_k["mu_bl"], post_k["Sigma"], 0.0)
        rows.append({
            "k": k, "effN": effN, "max_w": w.max(), "argmax": w.idxmax(),
            "unconstrained_gross_leverage": w_unc_k.abs().sum(),
        })
        wtab[f"k={k}"] = w

    return pd.DataFrame(rows).set_index("k"), pd.DataFrame(wtab)


def group_weights_figure(results, split_ts):
    """§6 figure: BL(k=0.1) target weights by asset group."""
    group_order = list(ml.ASSET_GROUPS.keys())
    group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_order)))

    bl01_group = ml.diagnostics.weights_by_group(results["BL(0.1)"]["wlog"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(bl01_group.index, bl01_group.T.to_numpy(), labels=group_order, colors=group_colors)
    ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title("BL(k=0.1) — target weights by asset group")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    return fig


def active_return_table(results, summary_table, split_ts):
    """§7 BL(k) - EW active return per window: annualized active return,
    tracking error, IR, t, and cost drags from ann_turnover x COST_BPS."""
    ew_net = results["EqualWeight"]["net"]

    active_rows = []
    active_series = {}
    for name in ["BL(0.1)", "BL(0.2)"]:
        net = results[name]["net"]
        idx = net.index.intersection(ew_net.index)
        active = net.loc[idx] - ew_net.loc[idx]
        active_series[name] = active
        win_defs = {
            "full": active.index >= active.index.min(),
            "train (<= split)": active.index <= split_ts,
            "test (> split)": active.index > split_ts,
        }
        for window_name, sel in win_defs.items():
            a = active.loc[sel]
            ann_active = float(a.mean() * ml.TRADING_DAYS)
            te = float(a.std(ddof=1) * np.sqrt(ml.TRADING_DAYS))
            ir = ann_active / te if te > 0 else np.nan
            t_active = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))
            cost_drag_bl = float(summary_table.loc[(name, window_name), "ann_turnover"]) * ml.COST_BPS / 1e4
            cost_drag_ew = float(summary_table.loc[("EqualWeight", window_name), "ann_turnover"]) * ml.COST_BPS / 1e4
            active_gross_of_costs = ann_active + (cost_drag_bl - cost_drag_ew)
            active_rows.append({
                "strategy": name, "window": window_name,
                "ann_active_return": ann_active, "tracking_error": te,
                "information_ratio": ir, "t_active": t_active,
                "cost_drag_bl": cost_drag_bl, "cost_drag_ew": cost_drag_ew,
                "active_gross_of_costs": active_gross_of_costs,
            })

    return pd.DataFrame(active_rows).set_index(["strategy", "window"]), active_series


def prior_posterior_figure(panel, asof):
    """Method figure: EW-implied prior Pi vs posterior mu_BL for k=0.1 and
    k=0.2 at `asof`, from BlackLitterman.posterior (in %)."""
    post01 = BlackLitterman(k=0.1).posterior(panel, asof)
    post02 = BlackLitterman(k=0.2).posterior(panel, asof)
    tickers = list(post01["pi"].index)
    x = np.arange(len(tickers))
    series = [
        ("Π (EW-implied prior)", post01["pi"], "#999999"),
        ("μ_BL (k=0.1)", post01["mu_bl"], ml.FAMILY_COLORS[BlackLitterman.family]),
        ("μ_BL (k=0.2)", post02["mu_bl"], "#c0392b"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (label, s, color) in enumerate(series):
        ax.bar(x + (j - 1) * 0.27, 100 * s.reindex(tickers).to_numpy(), width=0.27, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=90)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in ("UUP", "DBC"):
            lbl.set_color("#c0392b")
            lbl.set_fontweight("bold")
    ax.set_ylabel("annualized expected return (%)")
    ax.set_title(f"Prior Π vs posterior μ_BL, {asof.date()}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def ew_vs_bl_weights(wtab):
    """Method figure: EW (1/N) vs BL long-only weights per k at the
    snapshot, from k_sensitivity's wtab (in %)."""
    tickers = list(wtab.index)
    n = len(tickers)
    x = np.arange(n)
    series = [("EW (1/13)", pd.Series(1.0 / n, index=tickers), "#999999"),
              ("BL(0.1)", wtab["k=0.1"], ml.FAMILY_COLORS[BlackLitterman.family]),
              ("BL(0.2)", wtab["k=0.2"], "#c0392b")]
    if "k=0.4" in wtab.columns:
        series.append(("k=0.4, snapshot only", wtab["k=0.4"], "#555555"))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (label, w, color) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * width, 100 * w.to_numpy(), width=width, color=color, label=label)
    ax.axhline(100.0 / n, color="black", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=90)
    ax.set_ylabel("long-only weight (%)")
    ax.set_title("EW vs BL weights, 2022-12-31")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
