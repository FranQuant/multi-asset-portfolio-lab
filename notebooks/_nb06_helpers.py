"""Notebook-06-only mechanism code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures; notebook 06 prints, asserts and
shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import maplab as ml
from maplab import GMV, MaxSharpe
from maplab.models import _min_variance_long_only, _mdp_long_only, _erc_long_only, _tangency_long_only


def snapshot_tables(panel, dates):
    """§3 snapshot table (GMV/MDP/ERC/MaxSharpe/EW/IV at each date), UUP
    context table, and GMV first-order check {date: max|beta_i,p - 1|}."""
    snap_rows = []
    gmv_beta_dev = {}
    uup_context_rows = []
    for asof in dates:
        mu, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        Sigma_np = Sigma.to_numpy()
        sigma_np = np.sqrt(np.diag(Sigma_np))
        iU = list(Sigma.columns).index("UUP")
        n = Sigma_np.shape[0]

        w_gmv = _min_variance_long_only(Sigma)
        w_mdp, _ = _mdp_long_only(Sigma, "snap", asof)
        w_erc, _ = _erc_long_only(Sigma, "snap", asof)

        rf_ann = MaxSharpe()._rf_ann(panel, asof)
        excess = mu - rf_ann
        if excess.max() > 0:
            w_msr, _ = _tangency_long_only(excess.to_numpy(), Sigma_np, "snap", asof)
        else:
            w_msr = _min_variance_long_only(Sigma)

        w_ew = np.full(n, 1.0 / n)
        w_iv = (1.0 / sigma_np) / np.sum(1.0 / sigma_np)

        def row(name, w):
            w = np.asarray(w)
            var = float(w @ Sigma_np @ w)
            beta_p = (Sigma_np @ w) / var
            rc = (w * (Sigma_np @ w)) / np.sqrt(var)
            return {
                "asof": asof.date(), "strategy": name,
                "exante_vol": ml.diagnostics.port_vol(w, Sigma_np), "effN": ml.diagnostics.effective_n(w),
                "max_w": float(w.max()), "max_ticker": Sigma.columns[int(np.argmax(w))],
                "w_UUP": float(w[iU]), "UUP_rc_share": float(rc[iU] / rc.sum()),
                "UUP_beta_p": float(beta_p[iU]),
                "DR": ml.diagnostics.diversification_ratio(w, Sigma_np),
            }

        for name, w in [("GMV", w_gmv), ("MDP", w_mdp), ("ERC", w_erc),
                         ("MaxSharpe", w_msr), ("EW", w_ew), ("IV", w_iv)]:
            snap_rows.append(row(name, w))

        beta_gmv = (Sigma_np @ w_gmv) / float(w_gmv @ Sigma_np @ w_gmv)
        held = w_gmv > 1e-8
        gmv_beta_dev[asof.date()] = float(np.max(np.abs(beta_gmv[held] - 1.0)))

        D_inv = np.diag(1.0 / sigma_np)
        C = D_inv @ Sigma_np @ D_inv
        uup_mean_corr = float((C[iU].sum() - 1.0) / (n - 1))
        uup_vol_rank = int((sigma_np <= sigma_np[iU]).sum())
        rho_bar = float((C.sum() - n) / (n * (n - 1)))
        uup_context_rows.append({
            "asof": asof.date(), "uup_mean_pairwise_corr": uup_mean_corr,
            "uup_vol_rank": uup_vol_rank, "rho_bar": rho_bar,
        })

    snap_table = pd.DataFrame(snap_rows).set_index(["asof", "strategy"])
    uup_context_table = pd.DataFrame(uup_context_rows).set_index("asof")
    return snap_table, uup_context_table, gmv_beta_dev


def mechanism_grid(panel, dates):
    """Figure F2: cash weight vs. risk-contribution share, GMV/MDP/ERC/EW x
    dates. Returns (fig, grid_table)."""
    methods = ["GMV", "MDP", "ERC", "EW"]
    tickers = ml.UNIVERSE
    iU = tickers.index("UUP")
    one_over_n = 1.0 / len(tickers)

    grid_rows = []
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    for r, asof in enumerate(dates):
        _, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        Sigma = Sigma.reindex(index=tickers, columns=tickers)
        Sigma_np = Sigma.to_numpy()
        n = Sigma_np.shape[0]

        w_gmv = _min_variance_long_only(Sigma)
        w_mdp, _ = _mdp_long_only(Sigma, "F2", asof)
        w_erc, _ = _erc_long_only(Sigma, "F2", asof)
        w_ew = np.full(n, 1.0 / n)
        weights = {"GMV": w_gmv, "MDP": w_mdp, "ERC": w_erc, "EW": w_ew}

        for c, name in enumerate(methods):
            w = weights[name]
            var = float(w @ Sigma_np @ w)
            rc = (w * (Sigma_np @ w)) / np.sqrt(var)
            rc_share = rc / rc.sum()
            beta_p = (Sigma_np @ w) / var

            ax = axes[r, c]
            x = np.arange(n)
            colors_cash = ["#c0392b" if t == "UUP" else "#1f4e79" for t in tickers]
            colors_rc = ["#c0392b" if t == "UUP" else "#2e7d32" for t in tickers]
            ax.bar(x - 0.2, w, width=0.4, color=colors_cash, alpha=0.9,
                   label="cash weight" if r == 0 and c == 0 else None)
            ax.bar(x + 0.2, rc_share, width=0.4, color=colors_rc, alpha=0.6,
                   label="risk share" if r == 0 and c == 0 else None)
            ax.axhline(one_over_n, color="black", ls="--", lw=0.8)
            ax.set_title(f"{asof.date()} - {name}", fontsize=8)
            if r == 2:
                ax.set_xticks(x); ax.set_xticklabels(tickers, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])

            grid_rows.append({
                "asof": asof.date(), "method": name,
                "w_UUP": float(w[iU]), "UUP_rc_share": float(rc_share[iU]),
                "UUP_beta_p": float(beta_p[iU]), "effN": ml.diagnostics.effective_n(w),
            })

    fig.suptitle("Cash weight vs. risk-contribution share (dashed = 1/13)")
    plt.tight_layout()

    grid_table = pd.DataFrame(grid_rows).set_index(["asof", "method"])
    return fig, grid_table


def mechanism_checks(panel, rdates):
    """§5 mechanism checks over all rebalances x {sample, Ledoit-Wolf},
    recomputed fresh. Returns mech5_table (one row per estimator)."""
    mech5_rows = []
    for est_name, est_fn in [("sample", ml.sample_cov), ("lw", ml.ledoit_wolf_cov)]:
        max_rc_dev = 0.0
        max_euler = 0.0
        max_identA = 0.0
        max_identB = 0.0
        max_rho_bar_dev = 0.0
        min_vol_margin_gmv = np.inf   # sigma_ERC - sigma_GMV, want >= 0
        min_vol_margin_ew = np.inf    # sigma_EW - sigma_ERC, want >= 0
        min_w = np.inf
        sweeps = []
        for asof in rdates:
            _, Sigma = GMV(cov_estimator=est_fn)._estimate(panel, asof)
            Sigma_np = Sigma.to_numpy()
            sigma_np = np.sqrt(np.diag(Sigma_np))
            n = Sigma_np.shape[0]

            w_erc, n_sweeps = _erc_long_only(Sigma, "mech5", asof)
            sweeps.append(n_sweeps)
            var = float(w_erc @ Sigma_np @ w_erc)
            sigma_p = np.sqrt(var)
            rc = (w_erc * (Sigma_np @ w_erc)) / sigma_p
            rc_share = rc / rc.sum()
            max_rc_dev = max(max_rc_dev, float(np.max(np.abs(rc_share - 1.0 / n))))
            max_euler = max(max_euler, abs(float(rc.sum()) - sigma_p))

            beta_p = (Sigma_np @ w_erc) / var
            max_identA = max(max_identA, float(np.max(np.abs(w_erc * beta_p - 1.0 / n))))

            D_inv = np.diag(1.0 / sigma_np)
            C = pd.DataFrame(D_inv @ Sigma_np @ D_inv, index=Sigma.columns, columns=Sigma.columns)
            w_erc_c, _ = _erc_long_only(C, "mech5", asof)
            x = (w_erc * sigma_np) / float(w_erc @ sigma_np)
            max_identB = max(max_identB, float(np.max(np.abs(x - w_erc_c))))

            C_np = C.to_numpy()
            rho_bar = (C_np.sum() - n) / (n * (n - 1))
            C_bar = np.full((n, n), rho_bar); np.fill_diagonal(C_bar, 1.0)
            Sigma_bar = np.diag(sigma_np) @ C_bar @ np.diag(sigma_np)
            Sigma_bar_df = pd.DataFrame(Sigma_bar, index=Sigma.columns, columns=Sigma.columns)
            w_erc_bar, _ = _erc_long_only(Sigma_bar_df, "mech5", asof)
            iv = (1.0 / sigma_np) / np.sum(1.0 / sigma_np)
            max_rho_bar_dev = max(max_rho_bar_dev, float(np.max(np.abs(w_erc_bar - iv))))

            w_gmv = _min_variance_long_only(Sigma)
            w_ew = np.full(n, 1.0 / n)

            def pv(w):
                return float(np.sqrt(w @ Sigma_np @ w))

            min_vol_margin_gmv = min(min_vol_margin_gmv, pv(w_erc) - pv(w_gmv))
            min_vol_margin_ew = min(min_vol_margin_ew, pv(w_ew) - pv(w_erc))
            min_w = min(min_w, float(w_erc.min()))

        mech5_rows.append({
            "estimator": est_name, "max_rc_share_dev": max_rc_dev, "max_euler": max_euler,
            "max_identA": max_identA, "max_identB": max_identB,
            "max_rho_bar_dev": max_rho_bar_dev,
            "min_vol_margin_gmv": min_vol_margin_gmv, "min_vol_margin_ew": min_vol_margin_ew,
            "min_w": min_w, "median_sweeps": float(np.median(sweeps)), "max_sweeps": int(np.max(sweeps)),
        })

    return pd.DataFrame(mech5_rows).set_index("estimator")


def run_group(name):
    if name == "ERC(S)":
        return "registered"
    if name == "ERC(LW)":
        return "sensitivity"
    if name == "IV":
        return "reference"
    if name in ("EW", "60/40"):
        return "benchmark"
    return "reproduction gate"
