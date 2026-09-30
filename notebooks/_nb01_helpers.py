"""Notebook-01-only code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures/values; notebook 01 prints,
asserts and shows them.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import maplab as ml
from maplab import GMV, MaxSharpe
from maplab.models import gmv_closed_form, tangency_closed_form


def effective_n_stats(n_assets: int, alpha: float, size: int = 20_000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(n_assets, alpha), size=size)
    eff_n = 1.0 / np.sum(w ** 2, axis=1)
    return {"alpha": alpha, "median_eff_n": float(np.median(eff_n)), "median_max_w": float(np.median(w.max(axis=1)))}


def frontier_figure(Sigma, mu, w_gmv, w_msr, mc_alpha, rf_ann, split_rebal):
    """Long-only efficient frontier at the snapshot: Monte Carlo backdrop,
    assets, efficient/inefficient branch, EW/GMV/MaxSharpe markers, CML."""
    n = len(ml.UNIVERSE)
    Sigma_np = Sigma.to_numpy()
    mu_np = mu.to_numpy()

    # Monte Carlo backdrop: 20,000 random long-only portfolios, Dirichlet(mc_alpha)
    rng_mc = np.random.default_rng(7)
    mc_weights = rng_mc.dirichlet(np.full(n, mc_alpha), size=20_000)
    mc_ret = mc_weights @ mu_np
    mc_vol = np.sqrt(np.einsum("ij,jk,ik->i", mc_weights, Sigma_np, mc_weights))

    frontier_vol, frontier_ret = ml.diagnostics.long_only_frontier(Sigma, mu, n_points=60)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(mc_vol, mc_ret, s=2, alpha=0.08, color="#cccccc", zorder=0,
               label=f"random long-only portfolios (Dirichlet α={mc_alpha})")

    asset_vol = np.sqrt(np.diag(Sigma_np))
    ax.scatter(asset_vol, mu_np, s=18, color="black", zorder=2)
    for i, tkr in enumerate(ml.UNIVERSE):
        ax.annotate(tkr, (asset_vol[i], mu_np[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="black")

    gmv_ret = float(w_gmv.to_numpy() @ mu_np)
    gmv_vol = float(np.sqrt(w_gmv.to_numpy() @ Sigma_np @ w_gmv.to_numpy()))
    msr_ret = float(w_msr.to_numpy() @ mu_np)
    msr_vol = float(np.sqrt(w_msr.to_numpy() @ Sigma_np @ w_msr.to_numpy()))

    # Efficient (return >= GMV's) vs inefficient branch of the swept curve
    efficient = frontier_ret >= gmv_ret
    ax.plot(frontier_vol[~efficient], frontier_ret[~efficient], color="#999999",
            ls="--", lw=1.5, zorder=3, label="Inefficient branch")
    ax.plot(frontier_vol[efficient], frontier_ret[efficient], color="#2e7d32",
            lw=2, zorder=3, label="Efficient frontier (long-only)")

    ew_w = np.full(n, 1.0 / n)
    ew_ret = float(ew_w @ mu_np)
    ew_vol = float(np.sqrt(ew_w @ Sigma_np @ ew_w))
    ax.scatter([ew_vol], [ew_ret], marker="D", s=130, edgecolors="black", linewidths=1.0,
               color=ml.FAMILY_COLORS["Benchmark"], zorder=4, label="EW")

    w_6040 = pd.Series(0.0, index=Sigma.columns)
    w_6040[["SPY", "IEF"]] = [0.6, 0.4]
    w_6040 = w_6040.to_numpy()
    ret_6040 = float(w_6040 @ mu_np)
    vol_6040 = float(np.sqrt(w_6040 @ Sigma_np @ w_6040))
    ax.scatter([vol_6040], [ret_6040], marker="D", s=130, edgecolors="black", linewidths=1.0,
               color=ml.plotting.METHOD_COLORS["60/40"], zorder=4, label="60/40")

    ax.scatter([gmv_vol], [gmv_ret], marker="*", s=380, edgecolors="black", linewidths=1.0,
               color=ml.FAMILY_COLORS[GMV.family], zorder=4, label="GMV")
    ax.scatter([msr_vol], [msr_ret], marker="*", s=380, edgecolors="black", linewidths=1.0,
               color=ml.FAMILY_COLORS[MaxSharpe.family], zorder=4, label="MaxSharpe")

    cml_x = np.linspace(0, max(asset_vol.max(), msr_vol, mc_vol.max()) * 1.05, 20)
    cml_slope = (msr_ret - rf_ann) / msr_vol
    ax.plot(cml_x, rf_ann + cml_slope * cml_x, color="#1f4e79", ls="--", lw=1.5,
            zorder=1, label=f"CML (rf={rf_ann:.2%} → MaxSharpe)")

    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized arithmetic expected return (μ)")
    ax.set_title(f"Long-only efficient frontier — snapshot {split_rebal.date()}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def unconstrained_vs_long_only(Sigma, mu, rf_ann, w_gmv_lo, w_msr_lo):
    """Unconstrained (closed-form) vs long-only GMV and tangency weights at
    the snapshot, one panel each, grouped bars per ticker, UUP highlighted.
    Returns (fig, table): gross leverage, sum of negative weights, max |w|."""
    tickers = list(Sigma.columns)
    w_gmv_unc = gmv_closed_form(Sigma)
    w_tan_unc = tangency_closed_form(mu, Sigma, rf=rf_ann)
    pairs = {"GMV": (w_gmv_unc, w_gmv_lo.reindex(tickers)),
             "MaxSharpe": (w_tan_unc, w_msr_lo.reindex(tickers))}

    x = np.arange(len(tickers))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, (w_unc, w_lo)) in zip(axes, pairs.items()):
        unc_colors = ["#c0392b" if t == "UUP" else "#1f4e79" for t in tickers]
        lo_colors = ["#c0392b" if t == "UUP" else "#2e7d32" for t in tickers]
        ax.bar(x - 0.2, w_unc.to_numpy(), width=0.4, color=unc_colors, label="unconstrained (closed form)")
        ax.bar(x + 0.2, w_lo.to_numpy(), width=0.4, color=lo_colors, alpha=0.5, label="long-only (backtest)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=90, fontsize=7)
        ax.set_title("GMV" if name == "GMV" else "MSR (unconstrained = tangency)")
        ax.set_ylabel("weight")
    axes[0].legend(fontsize=8)
    fig.suptitle("Unconstrained vs long-only weights at the snapshot — independent y-axes "
                 f"(max |w|: tangency {w_tan_unc.abs().max():.2f}, GMV {w_gmv_unc.abs().max():.2f})")
    fig.tight_layout()

    cols = {"GMV unconstrained": w_gmv_unc, "GMV long-only": pairs["GMV"][1],
            "MSR unconstrained": w_tan_unc, "MSR long-only": pairs["MaxSharpe"][1]}
    table = pd.DataFrame({name: {"gross leverage Σ|w|": float(w.abs().sum()),
                                 "sum of negative weights": float(w[w < 0].sum()),
                                 "max |w|": float(w.abs().max())} for name, w in cols.items()})
    return fig, table
