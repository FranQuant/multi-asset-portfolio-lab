"""Notebook-04-only code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures/values; notebook 04 prints,
asserts and shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS

import maplab as ml
from maplab import GMV, MaxSharpe, BlackLitterman

plot_colors = {"GMV": ml.FAMILY_COLORS["Risk-based"], "MaxSharpe": ml.FAMILY_COLORS["Return-based"], "BL(0.1)": "#8e44ad"}


def dose_table(panel, rdates):
    """§3 shrinkage dose: sklearn LW/OAS shrinkage weights and condition
    numbers of Sigma_sample vs Sigma_LW at every rebalance label."""
    dose_rows = []
    for asof in rdates:
        rets = panel.slice(asof, "returns", ml.COV_LOOKBACK)[ml.UNIVERSE]
        assert len(rets) == ml.COV_LOOKBACK and not rets.isna().any().any()
        X = rets.to_numpy()
        d_lw = LedoitWolf().fit(X).shrinkage_
        d_oas = OAS().fit(X).shrinkage_
        Sigma_sample = ml.sample_cov(rets).to_numpy()
        Sigma_lw = ml.ledoit_wolf_cov(rets).to_numpy()
        dose_rows.append({
            "asof": asof, "delta_lw": d_lw, "delta_oas": d_oas,
            "cond_sample": np.linalg.cond(Sigma_sample), "cond_lw": np.linalg.cond(Sigma_lw),
        })

    return pd.DataFrame(dose_rows).set_index("asof")


def dose_figure(dose, split_ts):
    """§3 figure: shrinkage dose (top) and condition numbers (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)

    axes[0].plot(dose.index, dose["delta_lw"], label="delta_LW", color=ml.FAMILY_COLORS["Risk-based"], lw=1.1)
    axes[0].plot(dose.index, dose["delta_oas"], label="delta_OAS", color="#c0392b", lw=1.1)
    axes[0].axvline(split_ts, color="black", ls=":", lw=1)
    axes[0].set_ylabel("shrinkage weight")
    axes[0].set_title("Shrinkage dose through time")
    axes[0].legend(fontsize=8)

    axes[1].plot(dose.index, dose["cond_sample"], label="cond(Sigma_sample)", color="#555555", lw=1.1)
    axes[1].plot(dose.index, dose["cond_lw"], label="cond(Sigma_LW)", color=ml.FAMILY_COLORS["Risk-based"], lw=1.1)
    axes[1].set_yscale("log")
    axes[1].axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
    axes[1].set_ylabel("condition number (log)")
    axes[1].set_title("Conditioning: sample vs Ledoit-Wolf")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def diag_share_tables(panel, dates):
    """§4: diag/off-diag share of Sigma_LW - Sigma_sample, and
    {date: annualized vol table (%), sample vs LW}."""
    diag_share_rows = []
    vol_tables = {}
    for asof in dates:
        rets = panel.slice(asof, "returns", ml.COV_LOOKBACK)[ml.UNIVERSE]
        assert len(rets) == ml.COV_LOOKBACK
        Sigma_sample = ml.sample_cov(rets)
        Sigma_lw = ml.ledoit_wolf_cov(rets)
        Delta = Sigma_lw.to_numpy() - Sigma_sample.to_numpy()
        diag_share = float(np.linalg.norm(np.diag(Delta)) ** 2 / np.linalg.norm(Delta, "fro") ** 2)
        diag_share_rows.append({"asof": asof.date(), "diag_share": diag_share, "off_diag_share": 1 - diag_share})
        vol_tables[asof.date()] = pd.DataFrame({
            "sample_%": 100 * np.sqrt(np.diag(Sigma_sample.to_numpy())),
            "LW_%": 100 * np.sqrt(np.diag(Sigma_lw.to_numpy())),
        }, index=ml.UNIVERSE)

    return pd.DataFrame(diag_share_rows).set_index("asof"), vol_tables


# Local, snapshot-only variant: Ledoit-Wolf applied to the correlation
# structure only, with sample vols preserved exactly. Never backtested.
def lw_corr_cov(returns):
    X = returns.to_numpy(); s = X.std(axis=0, ddof=1)
    Z = (X - X.mean(axis=0)) / s
    C = LedoitWolf().fit(Z).covariance_
    d = np.sqrt(np.diag(C)); R = C / np.outer(d, d)
    return pd.DataFrame(np.outer(s, s) * R * ml.TRADING_DAYS, index=returns.columns, columns=returns.columns)


def lw_corr_table(panel, dates):
    """§4 lw_corr snapshot: half-L1 vs sample of LW and lw_corr weights."""
    lw_corr_rows = []
    for asof in dates:
        for strat_name, cls, kwargs in [("GMV", GMV, {}), ("MaxSharpe", MaxSharpe, {}), ("BL(0.1)", BlackLitterman, {"k": 0.1})]:
            w_sample = cls(cov_estimator=ml.sample_cov, **kwargs)(panel, asof)
            w_lw = cls(cov_estimator=ml.ledoit_wolf_cov, **kwargs)(panel, asof)
            w_lwcorr = cls(cov_estimator=lw_corr_cov, **kwargs)(panel, asof)
            lw_corr_rows.append({
                "asof": asof.date(), "strategy": strat_name,
                "halfL1_LW_vs_sample_%": float(0.5 * (w_lw - w_sample).abs().sum() * 100),
                "halfL1_lwcorr_vs_sample_%": float(0.5 * (w_lwcorr - w_sample).abs().sum() * 100),
            })

    return pd.DataFrame(lw_corr_rows).set_index(["asof", "strategy"])


class MaxSharpeItoFix(MaxSharpe):
    def _estimate(self, panel, asof):
        mu, Sigma = super()._estimate(panel, asof)
        rets = panel.slice(asof, "returns", self.lookback)[self.universe]
        var_s = pd.Series(np.diag(ml.sample_cov(rets).to_numpy()), index=rets.columns)
        return rets.mean() * ml.TRADING_DAYS + 0.5 * var_s, Sigma


def ito_table(panel, dates):
    """§5b Itô-term check: max|dmu| (bp/yr) and half-L1 weight change of
    MaxSharpeItoFix vs MaxSharpe under LW and OAS."""
    ito_rows = []
    for asof in dates:
        for est_name, est_fn in [("ledoit_wolf", ml.ledoit_wolf_cov), ("oas", ml.oas_cov)]:
            base = MaxSharpe(cov_estimator=est_fn)
            fixed = MaxSharpeItoFix(cov_estimator=est_fn)
            mu_base, _ = base._estimate(panel, asof)
            mu_fixed, _ = fixed._estimate(panel, asof)
            dmu_bp = float((mu_fixed - mu_base).abs().max() * 1e4)
            w_base = base(panel, asof)
            w_fixed = fixed(panel, asof)
            halfL1 = float(0.5 * (w_fixed - w_base).abs().sum() * 100)
            ito_rows.append({"asof": asof.date(), "estimator": est_name, "max_abs_dmu_bp": dmu_bp, "halfL1_dw_%": halfL1})

    return pd.DataFrame(ito_rows).set_index(["asof", "estimator"])


def weight_tables(panel, dates):
    """§5c: weight metrics table (effN, max weight, half-L1 vs sample) and
    {(date, strategy): weights (%) by estimator}."""
    weight_rows = []
    weight_frames = {}
    for asof in dates:
        for strat_name, cls, kwargs in [("GMV", GMV, {}), ("MaxSharpe", MaxSharpe, {}), ("BL(0.1)", BlackLitterman, {"k": 0.1})]:
            w_by_est = {
                est_name: cls(cov_estimator=est_fn, **kwargs)(panel, asof)
                for est_name, est_fn in [("sample", ml.sample_cov), ("ledoit_wolf", ml.ledoit_wolf_cov), ("oas", ml.oas_cov)]
            }
            w_sample = w_by_est["sample"]
            weight_frames[(asof.date(), strat_name)] = pd.DataFrame({k: v * 100 for k, v in w_by_est.items()})
            for est_name, w in w_by_est.items():
                effN = float(1.0 / (w ** 2).sum())
                weight_rows.append({
                    "asof": asof.date(), "strategy": strat_name, "estimator": est_name,
                    "effN": effN, "max_w_%": float(w.max() * 100), "argmax": w.idxmax(),
                    "halfL1_vs_sample_%": float(0.5 * (w - w_sample).abs().sum() * 100),
                })

    return pd.DataFrame(weight_rows).set_index(["asof", "strategy", "estimator"]), weight_frames


def run_group(name):
    if "(OAS)" in name:
        return "sensitivity"
    if name == "EW":
        return "benchmark"
    return "registered"


def cumulative_lw_figure(paired_series, split_ts):
    """§8 figure: cumulative LW - sample net-return difference per family."""
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for est_name, base_name, label in [("GMV(LW)", "GMV(S)", "GMV"), ("MS(LW)", "MS(S)", "MaxSharpe"), ("BL(LW)", "BL(S)", "BL(0.1)")]:
        cum = paired_series[(est_name, base_name)].cumsum()
        ax.plot(cum.index, cum.to_numpy() * 100, label=label, color=plot_colors[label], lw=1.3)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
    ax.set_ylabel("cumulative LW - sample net-return diff (%, simple sum)")
    ax.set_title("LW - sample: cumulative net-return difference")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def halfl1_lw_figure(results, split_ts):
    """§8 figure: half-L1 allocation distance LW vs sample through time."""
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for est_name, base_name, label in [("GMV(LW)", "GMV(S)", "GMV"), ("MS(LW)", "MS(S)", "MaxSharpe"), ("BL(LW)", "BL(S)", "BL(0.1)")]:
        wlog_lw = results[est_name]["wlog"]
        wlog_s = results[base_name]["wlog"]
        idx = wlog_lw.index.intersection(wlog_s.index)
        halfL1_ts = 0.5 * (wlog_lw.loc[idx] - wlog_s.loc[idx]).abs().sum(axis=1)
        ax.plot(halfL1_ts.index, halfL1_ts.to_numpy() * 100, label=label, color=plot_colors[label], lw=1.1)
    ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
    ax.set_ylabel("half-L1(w_LW - w_S) (%)")
    ax.set_title("Allocation distance: LW vs sample, through time")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _corr(Sigma):
    Sigma = np.asarray(Sigma)
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


def corr_heatmaps(Sigma_S, Sigma_LW):
    """§4 method figure: sample-implied, LW-implied (each from its own
    diagonal) and LW - sample correlation, in ml.UNIVERSE order."""
    R_S, R_LW = _corr(Sigma_S), _corr(Sigma_LW)
    R_D = R_LW - R_S
    n = len(ml.UNIVERSE)
    bounds = np.cumsum([len(g) for g in ml.ASSET_GROUPS.values()])[:-1] - 0.5
    d_lim = float(np.abs(R_D).max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), layout="constrained")
    panels = [(R_S, "sample-implied correlation", -1, 1),
              (R_LW, "LW-implied correlation", -1, 1),
              (R_D, "LW − sample correlation", -d_lim, d_lim)]
    ims = []
    for ax, (R, title, lo, hi) in zip(axes, panels):
        im = ax.imshow(R, vmin=lo, vmax=hi, cmap="RdBu_r")
        ims.append(im)
        for b in bounds:
            ax.axhline(b, color="black", lw=0.6)
            ax.axvline(b, color="black", lw=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(ml.UNIVERSE, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ml.UNIVERSE, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    fig.colorbar(ims[0], ax=axes[:2].tolist(), shrink=0.8, label="correlation")
    fig.colorbar(ims[2], ax=axes[2], shrink=0.8, label="Δ correlation")
    fig.suptitle("Correlation at 2022-12-31: sample vs Ledoit-Wolf")
    return fig


def eigen_scree(S_emp, delta, m):
    """§3 method figure: eigenvalues (descending, annualized, log) of the
    sample Σ and of Σ_LW = (1-δ)S + δmI, condition numbers in the legend."""
    S_emp = np.asarray(S_emp)
    lam_S = np.sort(np.linalg.eigvalsh(S_emp))[::-1]
    lam_LW = np.sort(np.linalg.eigvalsh((1 - delta) * S_emp + delta * m * np.eye(len(S_emp))))[::-1]
    k = np.arange(1, len(lam_S) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(k, lam_S * ml.TRADING_DAYS, marker="o", lw=1.1, color="#555555",
            label=f"sample Σ (cond = {lam_S[0] / lam_S[-1]:.1f})")
    ax.plot(k, lam_LW * ml.TRADING_DAYS, marker="s", lw=1.1, color=ml.FAMILY_COLORS["Risk-based"],
            label=f"Σ_LW, δ={delta:.4f} (cond = {lam_LW[0] / lam_LW[-1]:.1f})")
    ax.axhline(m * ml.TRADING_DAYS, color="#cccccc", lw=0.8, ls="--", label="m (mean eigenvalue)")
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (annualized, log)")
    ax.set_title("Eigenvalue scree at 2022-12-31: sample vs Ledoit-Wolf")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
