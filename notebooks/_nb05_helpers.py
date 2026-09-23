"""Notebook-05-only mechanism code, moved verbatim from the notebook cells.

Each function returns DataFrames/values; notebook 05 prints and asserts them.
"""
import numpy as np
import pandas as pd

import maplab as ml
from maplab import GMV
from maplab.models import _min_variance_long_only, _mdp_long_only


def mechanism_checks(panel, rdates):
    """§6 mechanism table over all rebalances, sample and LW estimators:
    max risk-weight identity diff (x_MDP vs long-only GMV on C) and min DR
    margin of MDP over GMV/EW/IV."""
    mech_rows = []
    for est_name, est_fn in [("sample", ml.sample_cov), ("lw", ml.ledoit_wolf_cov)]:
        max_identity_diff = 0.0
        min_dr_margin = np.inf
        for asof in rdates:
            _, Sigma = GMV(cov_estimator=est_fn)._estimate(panel, asof)
            Sigma_np = Sigma.to_numpy()
            sigma_np = np.sqrt(np.diag(Sigma_np))
            D_inv = np.diag(1.0 / sigma_np)
            C = pd.DataFrame(D_inv @ Sigma_np @ D_inv, index=Sigma.columns, columns=Sigma.columns)

            w_mdp, _ = _mdp_long_only(Sigma, "MDP", asof)
            w_gmv_c = _min_variance_long_only(C)
            x_mdp = (w_mdp * sigma_np) / float(w_mdp @ sigma_np)
            max_identity_diff = max(max_identity_diff, float(np.max(np.abs(x_mdp - w_gmv_c))))

            w_gmv = _min_variance_long_only(Sigma)
            w_ew = np.full(len(sigma_np), 1.0 / len(sigma_np))
            w_iv = (1.0 / sigma_np) / np.sum(1.0 / sigma_np)

            def dr(w):
                return float(w @ sigma_np) / np.sqrt(float(w @ Sigma_np @ w))

            dr_margin = dr(w_mdp) - max(dr(w_gmv), dr(w_ew), dr(w_iv))
            min_dr_margin = min(min_dr_margin, dr_margin)

        mech_rows.append({"estimator": est_name, "max_identity_diff": max_identity_diff, "min_dr_margin": min_dr_margin})

    return pd.DataFrame(mech_rows).set_index("estimator")


def snapshot_tables(panel, results, dates):
    """§7 snapshot: {date: GMV/MDP/IV/EW weight table}, metrics table
    (DR, ex-ante vol, effN, max weight), and the UUP decomposition table."""
    snapshot_rows = []
    weight_tables = {}
    uup_rows = []
    for asof in dates:
        _, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        Sigma_np = Sigma.to_numpy()
        sigma = pd.Series(np.sqrt(np.diag(Sigma_np)), index=Sigma.columns)
        sigma_np = sigma.to_numpy()

        w_gmv = results["GMV(S)"]["wlog"].loc[asof]
        w_mdp = results["MDP(S)"]["wlog"].loc[asof]
        w_iv = results["IV"]["wlog"].loc[asof]
        w_ew = results["EW"]["wlog"].loc[asof]

        D_inv = np.diag(1.0 / sigma_np)
        C = pd.DataFrame(D_inv @ Sigma_np @ D_inv, index=Sigma.columns, columns=Sigma.columns)
        w_gmv_c = pd.Series(_min_variance_long_only(C), index=Sigma.columns)

        def dr(w):
            w_np = w.to_numpy()
            return float(w_np @ sigma_np) / np.sqrt(float(w_np @ Sigma_np @ w_np))

        def row_metrics(name, w):
            w_np = w.to_numpy()
            return {
                "asof": asof.date(), "strategy": name,
                "DR": dr(w), "exante_vol": float(np.sqrt(w_np @ Sigma_np @ w_np)),
                "effN": ml.diagnostics.effective_n(w), "max_w": float(w.max()), "max_ticker": w.idxmax(),
            }

        for name, w in [("GMV", w_gmv), ("MDP", w_mdp), ("IV", w_iv), ("EW", w_ew)]:
            snapshot_rows.append(row_metrics(name, w))

        weight_tables[asof.date()] = pd.DataFrame({"GMV": w_gmv, "MDP": w_mdp, "IV": w_iv, "EW": w_ew}).round(4)

        g = float(w_gmv["UUP"]); m = float(w_mdp["UUP"]); r_uup = float(w_gmv_c["UUP"])
        uup_vol = float(sigma["UUP"])
        uup_rank = int((sigma <= uup_vol).sum())
        rets_win = panel.slice(asof, "returns", ml.COV_LOOKBACK)[ml.UNIVERSE]
        corr = rets_win.corr()
        uup_mean_corr = float(corr["UUP"].drop("UUP").mean())
        uup_rows.append({
            "asof": asof.date(), "g": g, "r": r_uup, "m": m,
            "uup_vol": uup_vol, "uup_rank": uup_rank, "uup_mean_corr": uup_mean_corr,
        })

    snapshot_metrics = pd.DataFrame(snapshot_rows).set_index(["asof", "strategy"])
    uup_table = pd.DataFrame(uup_rows).set_index("asof")
    return weight_tables, snapshot_metrics, uup_table
