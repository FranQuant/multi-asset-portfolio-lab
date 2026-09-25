"""Notebook-07-only mechanism code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures; notebook 07 prints, asserts and
shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage, is_monotonic, cophenet
from scipy.spatial.distance import pdist

import maplab as ml
from maplab import GMV
from maplab.models import _min_variance_long_only, _mdp_long_only, _erc_long_only, _hrp_long_only


def ivp_var_independent(Sigma_np, idx) -> float:
    # Variance of the inverse-variance sub-portfolio on idx, written out here
    # (not via _hrp_cluster_var) so mechanism checks are an independent check.
    idx = list(idx)
    sub = Sigma_np[np.ix_(idx, idx)]
    p = 1.0 / np.diag(sub)
    p = p / p.sum()
    return float(p @ sub @ p)


def sigma_cf_zero_corr(Sigma_np, i):
    # Sigma with asset i's off-diagonal correlations set to 0 (its own variance kept).
    sigma = np.sqrt(np.diag(Sigma_np))
    C = Sigma_np / np.outer(sigma, sigma)
    C = 0.5 * (C + C.T)
    C[i, :] = 0.0
    C[:, i] = 0.0
    C[i, i] = 1.0
    return np.outer(sigma, sigma) * C


def root_singleton(Z, i) -> bool:
    return i in (int(Z[-1, 0]), int(Z[-1, 1]))


def comparators_at(Sigma, asof, tag):
    w_gmv = _min_variance_long_only(Sigma)
    w_mdp, _ = _mdp_long_only(Sigma, tag, asof)
    w_erc, _ = _erc_long_only(Sigma, tag, asof)
    return {"GMV": np.asarray(w_gmv, float), "MDP": np.asarray(w_mdp, float), "ERC": np.asarray(w_erc, float)}


def snapshot(panel, dates):
    """§3 snapshot at each date (sample Σ): weights, Σ_cf counterfactual,
    tree info, recipe alternatives and the sqrt(1-rho) sanity rebuild."""
    snap = {}
    for asof in dates:
        _, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        cols = list(Sigma.columns)
        Sigma_np = Sigma.to_numpy()
        n = len(cols)
        iU = cols.index("UUP")
        sigma_np = np.sqrt(np.diag(Sigma_np))
        C = Sigma_np / np.outer(sigma_np, sigma_np)

        w_hrp, info = _hrp_long_only(Sigma, "snap", asof, bisection="tree")
        w_pos, info_pos = _hrp_long_only(Sigma, "snap", asof, bisection="positional")
        w_ivp = (1.0 / np.diag(Sigma_np)) / np.sum(1.0 / np.diag(Sigma_np))
        comps = comparators_at(Sigma, asof, "snap")
        w_ew = np.full(n, 1.0 / n)

        W = {"HRP": w_hrp, "HRP[pos]": w_pos, "IVP": w_ivp, **comps, "EW": w_ew}

        Sigma_cf_np = sigma_cf_zero_corr(Sigma_np, iU)
        Sigma_cf = pd.DataFrame(Sigma_cf_np, index=cols, columns=cols)
        w_hrp_cf, info_cf = _hrp_long_only(Sigma_cf, "snapcf", asof, bisection="tree")
        comps_cf = comparators_at(Sigma_cf, asof, "snapcf")

        Z = info["Z"]
        # Rebuild D/y explicitly (the registered recipe: half-formula, distance of distances)
        # to compute the cophenetic correlation and to quasi-diagonalize the heatmap in F2.
        C_half = np.clip(0.5 * (C + C.T), -1.0, 1.0)
        np.fill_diagonal(C_half, 1.0)
        D = np.sqrt(0.5 * (1.0 - C_half))
        np.fill_diagonal(D, 0.0)
        y = pdist(D, metric="euclidean")

        alts = {
            "average": _hrp_long_only(Sigma, "snap", asof, linkage="average")[0],
            "ward": _hrp_long_only(Sigma, "snap", asof, linkage="ward")[0],
            "direct": _hrp_long_only(Sigma, "snap", asof, dist_of_dist=False)[0],
        }

        # sqrt(1-rho) sanity: rebuild d = sqrt(1-rho) directly and re-cluster/bisect in-notebook.
        C_clip = np.clip(0.5 * (C + C.T), -1.0, 1.0)
        np.fill_diagonal(C_clip, 1.0)
        D_full = np.sqrt(1.0 - C_clip)
        np.fill_diagonal(D_full, 0.0)
        y_full = pdist(D_full, metric="euclidean")
        Z_full = sch.linkage(y_full, method="single")
        w_sanity = np.ones(n)
        stack = [sch.to_tree(Z_full)]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                continue
            left, right = node.get_left(), node.get_right()
            L, R = left.pre_order(), right.pre_order()
            vL, vR = ivp_var_independent(Sigma_np, L), ivp_var_independent(Sigma_np, R)
            a = vR / (vL + vR)
            w_sanity[L] *= a
            w_sanity[R] *= 1.0 - a
            stack += [left, right]

        snap[asof] = dict(
            cols=cols, Sigma_np=Sigma_np, sigma_np=sigma_np, C=C, iU=iU, n=n,
            W=W, comps=comps, w_ivp=w_ivp, w_ew=w_ew,
            info=info, info_pos=info_pos, info_cf=info_cf, D=D, y=y,
            Sigma_cf_np=Sigma_cf_np, comps_cf=comps_cf, w_hrp_cf=w_hrp_cf,
            alts=alts, w_sanity=w_sanity,
        )
    return snap


def tree_path(Sigma_np, Z, i):
    node = sch.to_tree(Z)
    path = []
    while not node.is_leaf():
        L = node.get_left().pre_order(); R = node.get_right().pre_order()
        vL, vR = ivp_var_independent(Sigma_np, L), ivp_var_independent(Sigma_np, R)
        aL = vR / (vL + vR)
        if i in L:
            path.append((len(L), len(R), aL))
            node = node.get_left()
        else:
            path.append((len(R), len(L), 1.0 - aL))
            node = node.get_right()
    return path


def pos_path(Sigma_np, order, i):
    it = list(order)
    path = []
    while len(it) > 1:
        h = len(it) // 2
        L, R = it[:h], it[h:]
        vL, vR = ivp_var_independent(Sigma_np, L), ivp_var_independent(Sigma_np, R)
        aL = vR / (vL + vR)
        if i in L:
            path.append((len(L), len(R), aL))
            it = L
        else:
            path.append((len(R), len(L), 1.0 - aL))
            it = R
    return path


def mechanism_grid(snap, dates):
    """Figure F2: dendrogram / quasi-diagonalized correlation / UUP weight
    bars, one row per snapshot date. Each row's dendrogram nodes are annotated
    with that date's own tree splits. Returns (fig, [(asof, annotated, n_links)])."""
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    uup_methods = ["IVP", "HRP", "HRP[pos]", "ERC", "GMV", "MDP", "EW"]
    cf_methods = ["HRP", "GMV", "MDP", "ERC"]
    annotated_counts = []

    for r, asof in enumerate(dates):
        s = snap[asof]
        cols, iU, Z, order = s["cols"], s["iU"], s["info"]["Z"], s["info"]["order"]

        # (a) dendrogram -- capital cascade: leaf labels show final HRP weight,
        # internal nodes show the L | R share of HRP's tree split at that node.
        ax = axes[r, 0]
        labels = np.array(cols)
        dend = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=6,
                           color_threshold=0, above_threshold_color="#555555")
        w_hrp_map = {c: s["W"]["HRP"][ci] for ci, c in enumerate(cols)}
        new_labels = [f"{t}\n{w_hrp_map[t]:.1%}" for t in dend["ivl"]]
        ax.set_xticklabels(new_labels, rotation=90, fontsize=6)
        for lbl in ax.get_xticklabels():
            if lbl.get_text().startswith("UUP"):
                lbl.set_color("#c0392b"); lbl.set_fontweight("bold")
        root_node, nodelist = sch.to_tree(Z, rd=True)
        root_L = root_node.get_left().pre_order()
        root_R = root_node.get_right().pre_order()
        v_l = ivp_var_independent(s["Sigma_np"], root_L)
        v_r = ivp_var_independent(s["Sigma_np"], root_R)
        a_uup_side = v_r / (v_l + v_r) if iU in root_L else v_l / (v_l + v_r)
        ax.set_title(f"{asof.date()} dendrogram (root share to UUP side={a_uup_side:.3f})", fontsize=8)

        leaves_order = dend["leaves"]
        leaf_pos = {leaf: pos for pos, leaf in enumerate(leaves_order)}
        icoord = np.array(dend["icoord"])
        dcoord = np.array(dend["dcoord"])
        n_links = Z.shape[0]
        annotated = 0
        for k in range(icoord.shape[0]):
            h = dcoord[k, 1]
            x_pos = float(np.mean(icoord[k, 1:3]))
            matches = np.where(np.abs(Z[:, 2] - h) <= 1e-12)[0]
            if len(matches) != 1:
                continue  # ambiguous tied-height link, skip
            j = int(matches[0])
            left_id, right_id = int(Z[j, 0]), int(Z[j, 1])
            left_leaves = nodelist[left_id].pre_order()
            right_leaves = nodelist[right_id].pre_order()
            left_positions = [leaf_pos[l] for l in left_leaves]
            right_positions = [leaf_pos[l] for l in right_leaves]
            assert max(left_positions) < min(right_positions), (
                f"{asof.date()}: link {k} (row {j}) -- plotted left child is not Z[j,0]"
            )
            left_set = set(left_leaves)
            split_shares = None
            for (L, R, a) in s["info"]["splits"]:
                if set(L) == left_set:
                    split_shares = (a, 1.0 - a)
                    break
                if set(R) == left_set:
                    split_shares = (1.0 - a, a)
                    break
            if split_shares is None:
                continue
            a_left, a_right = split_shares
            ax.annotate(f"{a_left:.2f} | {a_right:.2f}", (x_pos, h), fontsize=5,
                        ha="center", va="bottom", color="#2c3e50",
                        xytext=(0, 2 + 6 * (k % 2)), textcoords="offset points")
            annotated += 1
        annotated_counts.append((asof, annotated, n_links))

        # (b) quasi-diagonalized correlation heatmap
        ax = axes[r, 1]
        C_ord = s["C"][np.ix_(order, order)]
        im = ax.imshow(C_ord, vmin=-1, vmax=1, cmap="RdBu_r")
        ordered_labels = [cols[j] for j in order]
        pos_uup = ordered_labels.index("UUP")
        ax.axhline(pos_uup - 0.5, color="black", lw=0.6)
        ax.axhline(pos_uup + 0.5, color="black", lw=0.6)
        ax.axvline(pos_uup - 0.5, color="black", lw=0.6)
        ax.axvline(pos_uup + 0.5, color="black", lw=0.6)
        boundary = len(root_L) - 0.5
        ax.axhline(boundary, color="#c0392b", lw=1.2, ls="--")
        ax.axvline(boundary, color="#c0392b", lw=1.2, ls="--")
        ax.set_xticks(range(s["n"])); ax.set_xticklabels(ordered_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(s["n"])); ax.set_yticklabels(ordered_labels, fontsize=5)
        ax.set_title(f"{asof.date()} correlation (leaf order)", fontsize=8)

        # (c) UUP weight bars, hollow = counterfactual
        ax = axes[r, 2]
        x = np.arange(len(uup_methods))
        vals = [s["W"][m][iU] for m in uup_methods]
        ax.bar(x, vals, color=["#555555" if m != "HRP" else "#8e44ad" for m in uup_methods])
        for m in cf_methods:
            xi = uup_methods.index(m)
            cf_val = s["w_hrp_cf"][iU] if m == "HRP" else s["comps_cf"][m][iU]
            ax.bar(xi, cf_val, facecolor="none", edgecolor="#c0392b", lw=1.5, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(uup_methods, rotation=45, fontsize=7, ha="right")
        ax.set_title(f"{asof.date()} UUP weight (hollow = Sigma_cf)", fontsize=8)

    fig.tight_layout()
    return fig, annotated_counts


def quasi_diag_check(Z):
    n = Z.shape[0] + 1
    order = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(order) >= n:
        new = []
        for c in order:
            if c >= n:
                new += [int(Z[c - n, 0]), int(Z[c - n, 1])]
            else:
                new.append(c)
        order = new
    return np.array(order)


def split_identity_dev(Sigma_np, w, splits):
    dev = 0.0
    for L, R, a in splits:
        vL = ivp_var_independent(Sigma_np, L)
        vR = ivp_var_independent(Sigma_np, R)
        a_expected = vR / (vL + vR)
        dev = max(dev, abs(a - a_expected))
        mL, mR = w[list(L)].sum(), w[list(R)].sum()
        dev = max(dev, abs(mL / (mL + mR) - a_expected))
    return dev


def perm_spread(Sigma_np, bisection, K, seed):
    w0, _ = _hrp_long_only(pd.DataFrame(Sigma_np), "mech5", pd.Timestamp("2020-01-01"), bisection=bisection)
    rng = np.random.default_rng(seed)
    n = Sigma_np.shape[0]
    worst = 0.0
    for _ in range(K):
        p = rng.permutation(n)
        wp, _ = _hrp_long_only(pd.DataFrame(Sigma_np[np.ix_(p, p)]), "mech5", pd.Timestamp("2020-01-01"), bisection=bisection)
        wb = np.empty(n)
        wb[p] = wp
        worst = max(worst, 0.5 * float(np.abs(wb - w0).sum()))
    return worst


def mechanism_checks(panel, rdates):
    """§5: all rebalances x {sample, Ledoit-Wolf} x {tree, positional}.
    Returns (H, pos_spreads_208, cophs, root_sides)."""
    ESTS = {"sample": ml.sample_cov, "lw": ml.ledoit_wolf_cov}
    hrows = []
    pos_spreads_208 = []
    cophs = []
    root_sides = []

    for est_name, est_fn in ESTS.items():
        a = dict(sum_dev=0.0, min_w=np.inf, split_tree=0.0, split_pos=0.0,
                 diag_ivp_tree=0.0, diag_ivp_pos=0.0, scale_tree=0.0, scale_pos=0.0,
                 perm_tree=0.0, perm_valid=0, link_ok=0, qd_match=0, dist_ok=0, n=0)
        for k, asof in enumerate(rdates):
            _, Sigma = GMV(cov_estimator=est_fn)._estimate(panel, asof)
            Sigma_np = Sigma.to_numpy()
            n_assets = Sigma_np.shape[0]

            w_t, info_t = _hrp_long_only(Sigma, "mech5", asof, bisection="tree")
            w_p, info_p = _hrp_long_only(Sigma, "mech5", asof, bisection="positional")

            a["sum_dev"] = max(a["sum_dev"], abs(w_t.sum() - 1.0), abs(w_p.sum() - 1.0))
            a["min_w"] = min(a["min_w"], float(w_t.min()), float(w_p.min()))
            a["split_tree"] = max(a["split_tree"], split_identity_dev(Sigma_np, w_t, info_t["splits"]))
            a["split_pos"] = max(a["split_pos"], split_identity_dev(Sigma_np, w_p, info_p["splits"]))

            Sigma_diag = pd.DataFrame(np.diag(np.diag(Sigma_np)))
            ivp = (1.0 / np.diag(Sigma_np)) / np.sum(1.0 / np.diag(Sigma_np))
            w_diag_t, _ = _hrp_long_only(Sigma_diag, "mech5", asof, bisection="tree")
            w_diag_p, _ = _hrp_long_only(Sigma_diag, "mech5", asof, bisection="positional")
            a["diag_ivp_tree"] = max(a["diag_ivp_tree"], float(np.max(np.abs(w_diag_t - ivp))))
            a["diag_ivp_pos"] = max(a["diag_ivp_pos"], float(np.max(np.abs(w_diag_p - ivp))))

            w_scale_t, _ = _hrp_long_only(Sigma * 7.3, "mech5", asof, bisection="tree")
            w_scale_p, _ = _hrp_long_only(Sigma * 7.3, "mech5", asof, bisection="positional")
            a["scale_tree"] = max(a["scale_tree"], float(np.max(np.abs(w_scale_t - w_t))))
            a["scale_pos"] = max(a["scale_pos"], float(np.max(np.abs(w_scale_p - w_p))))

            a["perm_tree"] = max(a["perm_tree"], perm_spread(Sigma_np, "tree", 5, seed=k))
            a["perm_valid"] += int(np.array_equal(np.sort(info_t["order"]), np.arange(n_assets)))

            Z = info_t["Z"]
            a["link_ok"] += int(is_valid_linkage(Z) and is_monotonic(Z))
            a["qd_match"] += int(np.array_equal(quasi_diag_check(Z), info_t["order"]))

            sigma_np = np.sqrt(np.diag(Sigma_np))
            C = np.clip(0.5 * ((Sigma_np / np.outer(sigma_np, sigma_np)) + (Sigma_np / np.outer(sigma_np, sigma_np)).T), -1.0, 1.0)
            np.fill_diagonal(C, 1.0)
            D = np.sqrt(0.5 * (1.0 - C))
            np.fill_diagonal(D, 0.0)
            a["dist_ok"] += int(np.array_equal(D, D.T) and np.all(np.diag(D) == 0.0) and D.min() >= 0.0 and D.max() <= 1.0)
            a["n"] += 1

            if est_name == "sample":
                pos_spreads_208.append(perm_spread(Sigma_np, "positional", 20, seed=1000 + k))
                y = pdist(D, metric="euclidean")
                cophs.append(float(cophenet(Z, y)[0]))
                root_node = sch.to_tree(Z)
                root_L = frozenset(root_node.get_left().pre_order())
                iU_here = list(Sigma.columns).index("UUP")
                uup_side = root_L if iU_here in root_L else frozenset(root_node.get_right().pre_order())
                root_sides.append(uup_side)

        hrows.append(pd.Series(a, name=est_name))

    H = pd.DataFrame(hrows)
    return H, pos_spreads_208, cophs, root_sides


def h2_counterfactual_table(panel, rdates):
    """H2a/H2b per rebalance (sample Σ): root-singleton flags under Σ and Σ_cf
    and |w_UUP(Σ) - w_UUP(Σ_cf)| for HRP / GMV / MDP / ERC."""
    h2_rows = []
    for asof in rdates:
        _, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        cols = list(Sigma.columns)
        Sigma_np = Sigma.to_numpy()
        iU = cols.index("UUP")

        w_hrp, info_hrp = _hrp_long_only(Sigma, "h2", asof, bisection="tree")
        w_gmv = _min_variance_long_only(Sigma)
        w_mdp, _ = _mdp_long_only(Sigma, "h2", asof)
        w_erc, _ = _erc_long_only(Sigma, "h2", asof)

        Sigma_cf_np = sigma_cf_zero_corr(Sigma_np, iU)
        Sigma_cf = pd.DataFrame(Sigma_cf_np, index=cols, columns=cols)
        w_hrp_cf, info_hrp_cf = _hrp_long_only(Sigma_cf, "h2cf", asof, bisection="tree")
        w_gmv_cf = _min_variance_long_only(Sigma_cf)
        w_mdp_cf, _ = _mdp_long_only(Sigma_cf, "h2cf", asof)
        w_erc_cf, _ = _erc_long_only(Sigma_cf, "h2cf", asof)

        root_s = root_singleton(info_hrp["Z"], iU)
        root_s_cf = root_singleton(info_hrp_cf["Z"], iU)

        h2_rows.append({
            "asof": asof,
            "root_singleton": root_s, "root_singleton_cf": root_s_cf,
            "dw_HRP": abs(w_hrp[iU] - w_hrp_cf[iU]),
            "dw_GMV": abs(w_gmv[iU] - w_gmv_cf[iU]),
            "dw_MDP": abs(w_mdp[iU] - w_mdp_cf[iU]),
            "dw_ERC": abs(w_erc[iU] - w_erc_cf[iU]),
        })

    return pd.DataFrame(h2_rows).set_index("asof")


def forecast_bias_table(panel, rdates, results, simple_returns, h7_methods):
    """H7: q = log(realized holding-segment vol / ex-ante vol) per rebalance
    and method. Returns (q_table, n_dropped, sigma_cache)."""
    sigma_cache = {}
    for asof in rdates:
        _, Sigma = GMV(cov_estimator=ml.sample_cov)._estimate(panel, asof)
        sigma_cache[asof] = Sigma

    boundaries = list(rdates) + [simple_returns.index.max() + pd.Timedelta(days=1)]
    q_rows = []
    n_dropped = 0
    for k, asof in enumerate(rdates):
        start, end = boundaries[k], boundaries[k + 1]
        Sigma = sigma_cache[asof]
        Sigma_np = Sigma.to_numpy()
        for m in h7_methods:
            net = results[m]["net"]
            seg_idx = net.index[(net.index >= start) & (net.index < end)]
            if len(seg_idx) <= 1:
                n_dropped += 1
                continue
            seg_idx_excl_first = seg_idx[1:]
            if len(seg_idx_excl_first) < 10:
                n_dropped += 1
                continue
            realized_vol = float(net.loc[seg_idx_excl_first].std(ddof=1) * np.sqrt(ml.TRADING_DAYS))

            w = results[m]["wlog"].loc[asof].reindex(Sigma.columns).to_numpy()
            exante_vol = ml.diagnostics.port_vol(w, Sigma_np)
            q = float(np.log(realized_vol / exante_vol))
            q_rows.append({"asof": asof, "method": m, "q": q, "exante_vol": exante_vol, "realized_vol": realized_vol})

    q_table = pd.DataFrame(q_rows)
    return q_table, n_dropped, sigma_cache


def forecast_bias_figure(q_pivot, split_ts):
    """Figure F6: q_HRP and q_GMV per holding segment."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    q_hrp = q_pivot["HRP(S)"].dropna()
    q_gmv = q_pivot["GMV(S)"].dropna()
    ax.plot(q_hrp.index, q_hrp.to_numpy(), label="q_HRP", color="#8e44ad", lw=1.1)
    ax.plot(q_gmv.index, q_gmv.to_numpy(), label="q_GMV", color="#555555", lw=1.1)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
    ax.set_ylabel("q = log(realized vol / ex-ante vol)")
    ax.set_xlabel("rebalance date")
    ax.set_title("Forecast bias per holding segment: HRP vs GMV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def root_split_table(sigma_cache, h2_table, results, rdates):
    """F7 table: sigma_UUP, sqrt(V_rest), rho_bar_rest, the root-singleton
    identity and w_UUP (HRP/GMV/ERC) per rebalance."""
    f7_rows = []
    for asof in rdates:
        Sigma = sigma_cache[asof]
        cols = list(Sigma.columns)
        iU = cols.index("UUP")
        Sigma_np = Sigma.to_numpy()
        rest_idx = [j for j in range(len(cols)) if j != iU]

        sigma_uup = float(np.sqrt(Sigma_np[iU, iU]))
        V_rest = ivp_var_independent(Sigma_np, rest_idx)
        sqrt_V_rest = float(np.sqrt(V_rest))
        identity = V_rest / (sigma_uup ** 2 + V_rest)

        sub = Sigma_np[np.ix_(rest_idx, rest_idx)]
        sub_sigma = np.sqrt(np.diag(sub))
        sub_C = sub / np.outer(sub_sigma, sub_sigma)
        n_rest = len(rest_idx)
        rho_bar_rest = float((sub_C.sum() - n_rest) / (n_rest * (n_rest - 1)))

        f7_rows.append({
            "asof": asof,
            "root_singleton": bool(h2_table.loc[asof, "root_singleton"]),
            "sigma_UUP": sigma_uup,
            "sqrt_V_rest": sqrt_V_rest,
            "rho_bar_rest": rho_bar_rest,
            "identity": identity,
            "w_UUP_HRP": float(results["HRP(S)"]["wlog"].loc[asof, "UUP"]),
            "w_UUP_GMV": float(results["GMV(S)"]["wlog"].loc[asof, "UUP"]),
            "w_UUP_ERC": float(results["ERC(S)"]["wlog"].loc[asof, "UUP"]),
        })
    return pd.DataFrame(f7_rows).set_index("asof")


def root_split_figure(f7_table, split_ts):
    """Figure F7(a-c), non-root-singleton rebalances shaded."""
    dates_idx = f7_table.index
    non_singleton_dates = dates_idx[~f7_table["root_singleton"].to_numpy()]
    diffs = dates_idx.to_series().diff().dropna()
    half_width = diffs.median() / 2 if len(diffs) else pd.Timedelta(days=15)

    def shade_non_singleton(ax):
        for d in non_singleton_dates:
            ax.axvspan(d - half_width, d + half_width, color="#dddddd", alpha=0.6, lw=0, zorder=0)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11), sharex=True)

    ax = axes[0]
    shade_non_singleton(ax)
    ax.plot(f7_table.index, f7_table["sigma_UUP"], label="sigma_UUP", color="#c0392b", lw=1.1)
    ax.plot(f7_table.index, f7_table["sqrt_V_rest"], label="sqrt(V_rest)", color="#555555", lw=1.1)
    ax.axvline(split_ts, color="black", ls=":", lw=1, label="train/test split")
    ax.set_ylabel("annualized vol")
    ax.set_title("F7(a): sigma_UUP vs sqrt(V_rest)", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    shade_non_singleton(ax)
    ax.plot(f7_table.index, f7_table["w_UUP_HRP"], label="w_UUP HRP(S)", color="#8e44ad", lw=1.3)
    ax.plot(f7_table.index, f7_table["identity"], label="identity V_rest/(sigma_UUP^2+V_rest)",
            color="#8e44ad", lw=1.0, ls="--")
    ax.plot(f7_table.index, f7_table["w_UUP_GMV"], label="w_UUP GMV(S)",
            color=ml.FAMILY_COLORS["Risk-based"], lw=0.7)
    ax.plot(f7_table.index, f7_table["w_UUP_ERC"], label="w_UUP ERC(S)", color="#d4a017", lw=0.7)
    ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax.set_ylabel("w_UUP")
    ax.set_title("F7(b): HRP w_UUP vs the root-singleton identity", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    shade_non_singleton(ax)
    ax.plot(f7_table.index, f7_table["rho_bar_rest"], color="#2c3e50", lw=1.1)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax.set_ylabel("mean pairwise corr\n(12 non-UUP)")
    ax.set_xlabel("rebalance date")
    ax.set_title("F7(c): rho_bar_rest", fontsize=9)

    fig.tight_layout()
    return fig


def f7_summary_row(f7_table, mask_or_date):
    if isinstance(mask_or_date, pd.Timestamp):
        row = f7_table.loc[mask_or_date]
        return {
            "median_sigma_UUP": float(row["sigma_UUP"]),
            "median_sqrt_V_rest": float(row["sqrt_V_rest"]),
            "median_ratio": float(row["sqrt_V_rest"] / row["sigma_UUP"]),
            "median_rho_bar_rest": float(row["rho_bar_rest"]),
            "median_w_UUP_HRP": float(row["w_UUP_HRP"]),
            "root_singleton_share": float(row["root_singleton"]),
        }
    sub = f7_table.loc[mask_or_date]
    ratio = sub["sqrt_V_rest"] / sub["sigma_UUP"]
    return {
        "median_sigma_UUP": float(sub["sigma_UUP"].median()),
        "median_sqrt_V_rest": float(sub["sqrt_V_rest"].median()),
        "median_ratio": float(ratio.median()),
        "median_rho_bar_rest": float(sub["rho_bar_rest"].median()),
        "median_w_UUP_HRP": float(sub["w_UUP_HRP"].median()),
        "root_singleton_share": float(sub["root_singleton"].mean()),
    }


def run_group(name):
    if name == "HRP(S)":
        return "registered"
    if name in ("HRP(LW)", "HRP[pos](S)", "HRP[ward](S)"):
        return "sensitivity"
    if name == "IVP":
        return "reference"
    if name in ("EW", "60/40"):
        return "benchmark"
    return "reproduction gate"
