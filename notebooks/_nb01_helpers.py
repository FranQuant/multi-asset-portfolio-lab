"""Notebook-01-only code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures/values; notebook 01 prints,
asserts and shows them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

import maplab as ml
from maplab import GMV, MaxSharpe


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

    ax.scatter(mc_vol, mc_ret, s=2, alpha=0.15, color="#bbbbbb", zorder=0,
               label=f"random long-only portfolios (Dirichlet α={mc_alpha})")

    asset_vol = np.sqrt(np.diag(Sigma_np))
    ax.scatter(asset_vol, mu_np, s=18, color="#999999", zorder=2)
    for i, tkr in enumerate(ml.UNIVERSE):
        ax.annotate(tkr, (asset_vol[i], mu_np[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="#666666")

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
    ax.scatter([ew_vol], [ew_ret], marker="D", s=90,
               color=ml.FAMILY_COLORS["Benchmark"], zorder=4, label="EqualWeight")

    ax.scatter([gmv_vol], [gmv_ret], marker="*", s=220,
               color=ml.FAMILY_COLORS[GMV.family], zorder=4, label="GMV")
    ax.scatter([msr_vol], [msr_ret], marker="*", s=220,
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


def wealth_figure(strategies, results, split_ts):
    """GMV vs MaxSharpe vs EqualWeight — cumulative wealth (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, strat in strategies.items():
        net = results[name]["net"]
        wealth = (1.0 + net).cumprod()
        color = ml.FAMILY_COLORS[strat.family]
        ax.plot(wealth.index, wealth.to_numpy(), label=strat.label, color=color, lw=1.6)

    ax.axvline(split_ts, color="#888888", ls=":", lw=1, label="train/test split")
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    candidate_ticks = [1, 2, 3, 4, 6, 8]
    yticks = [t for t in candidate_ticks if ymin <= t <= ymax]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("cumulative wealth (log scale, net of costs)")
    ax.set_title("GMV vs MaxSharpe vs EqualWeight — cumulative wealth")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig
