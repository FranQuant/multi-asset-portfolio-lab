"""Notebook-00-only descriptive figures.

Each function returns (Figure, data); notebook 00 prints and shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import maplab as ml
from maplab.models import _hrp_long_only


def clustered_corr_figure(log_returns):
    """Full-sample correlation heatmap in the HRP leaf order of notebook 07 (single
    linkage, distance of distances), values annotated, UUP row/column boxed. Also returns the root split as two sorted ticker lists."""
    Sigma = ml.sample_cov(log_returns[ml.UNIVERSE])
    _, info = _hrp_long_only(Sigma, "nb00", pd.Timestamp("2026-04-30"))
    order_tickers = [Sigma.columns[i] for i in info["order"]]

    sigma = np.sqrt(np.diag(Sigma.to_numpy()))
    C = pd.DataFrame(Sigma.to_numpy() / np.outer(sigma, sigma),
                     index=Sigma.index, columns=Sigma.columns)
    C_ord = C.loc[order_tickers, order_tickers].to_numpy()
    n = len(order_tickers)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = C_ord[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    k = order_tickers.index("UUP")
    ax.add_patch(Rectangle((-0.5, k - 0.5), n, 1, fill=False, edgecolor="black", lw=1.5))
    ax.add_patch(Rectangle((k - 0.5, -0.5), 1, n, fill=False, edgecolor="black", lw=1.5))
    ax.set_xticks(range(n)); ax.set_xticklabels(order_tickers, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(order_tickers, fontsize=7)
    ax.set_title(f"Full-sample correlation of daily log returns, "
                 f"{log_returns.index.min().date()} to {log_returns.index.max().date()}")
    fig.colorbar(im, ax=ax, shrink=0.7, label="corr")
    fig.tight_layout()

    cols = list(Sigma.columns)

    def _names(side):
        return sorted(cols[i] if isinstance(i, (int, np.integer)) else i for i in side)

    root = next((_names(L), _names(R)) for L, R, _ in info["splits"] if len(L) + len(R) == n)
    return fig, order_tickers, root


def uup_rolling_figure(panel, rdates, split_ts):
    """At each rebalance label, on the model's own trailing window: UUP's mean
    pairwise correlation with the other 12, sigma_UUP (annualized), and
    rho_bar_rest (mean off-diagonal correlation among the other 12)."""
    rest = [t for t in ml.UNIVERSE if t != "UUP"]
    n_rest = len(rest)
    rows = []
    for asof in rdates:
        rets = panel.slice(asof, "returns", ml.COV_LOOKBACK)[ml.UNIVERSE]
        corr = rets.corr()
        sub_C = corr.loc[rest, rest].to_numpy()
        rows.append({
            "asof": asof,
            "uup_mean_corr": float(corr.loc[rest, "UUP"].mean()),
            "sigma_UUP": float(rets["UUP"].std() * np.sqrt(ml.TRADING_DAYS)),
            "rho_bar_rest": float((sub_C.sum() - n_rest) / (n_rest * (n_rest - 1))),
        })
    table = pd.DataFrame(rows).set_index("asof")

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    specs = [
        ("uup_mean_corr", ml.plotting.HIGHLIGHT, "mean corr of UUP\nwith the other 12", r"(a) $\bar\rho_U$: UUP mean correlation with the other 12"),
        ("sigma_UUP", ml.plotting.HIGHLIGHT, "annualized vol (%)", r"(b) $\sigma_U$: UUP annualized volatility"),
        ("rho_bar_rest", "#2c3e50", "mean pairwise corr\n(12 non-UUP)", r"(c) $\bar\rho_{\mathrm{rest}}$: mean correlation among the other 12"),
    ]
    for ax, (col, color, ylabel, title) in zip(axes, specs):
        ax.plot(table.index, table[col] * (100 if col == "sigma_UUP" else 1), color=color, lw=1.1)
        if col != "sigma_UUP":
            ax.axhline(0, color="black", lw=0.5)
        ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("rebalance date")
    fig.tight_layout()
    return fig, table
