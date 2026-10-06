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
    """§4.2 mechanics table at the split snapshot: Pi, sigma, 12-1 momentum,
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
    """§4.2 checks: k=0 -> EW, tau invariance, closed-form identity, and the
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
    """§4.4 cautionary snapshot: effN / max weight / unconstrained gross
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


def prior_posterior_figure(panel, asof):
    """Method figure: EW-implied prior Pi vs posterior mu_BL for k=0.1 and
    k=0.2 at `asof`, from BlackLitterman.posterior (in %)."""
    post01 = BlackLitterman(k=0.1).posterior(panel, asof)
    post02 = BlackLitterman(k=0.2).posterior(panel, asof)
    tickers = list(post01["pi"].index)
    x = np.arange(len(tickers))
    series = [
        ("Π (EW-implied prior)", post01["pi"], "#999999", None),
        ("μ_BL (k=0.1)", post01["mu_bl"], ml.plotting.RUN_COLORS["BL"], None),
        ("μ_BL (k=0.2)", post02["mu_bl"], ml.plotting.RUN_COLORS["BL"], "//"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (label, s, color, hatch) in enumerate(series):
        ax.bar(x + (j - 1) * 0.27, 100 * s.reindex(tickers).to_numpy(), width=0.27, color=color, label=label,
               hatch=hatch, edgecolor="white" if hatch else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=90)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in ("UUP", "DBC"):
            if lbl.get_text() == "UUP":
                lbl.set_color(ml.plotting.HIGHLIGHT)
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
    bl_color = ml.plotting.RUN_COLORS["BL"]
    series = [("EW (1/13)", pd.Series(1.0 / n, index=tickers), ml.plotting.RUN_COLORS["EW"], None),
              ("BL(0.1)", wtab["k=0.1"], bl_color, None),
              ("BL(0.2)", wtab["k=0.2"], bl_color, "//")]
    if "k=0.4" in wtab.columns:
        series.append(("k=0.4, snapshot only", wtab["k=0.4"], bl_color, ".."))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (label, w, color, hatch) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * width, 100 * w.to_numpy(), width=width, color=color, label=label,
               hatch=hatch, edgecolor="white" if hatch else None)
    ax.axhline(100.0 / n, color="black", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=90)
    ax.set_ylabel("long-only weight (%)")
    ax.set_title("EW vs BL weights, 2022-12-31")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
