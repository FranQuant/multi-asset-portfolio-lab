"""Notebook-10-only overlay runs, gates, tables and statistics.

Base runs, the summary table and excess-return frames are notebook
08's (_nb08_helpers); the overlay is maplab.vol_overlay; the statistics come
from maplab.robust / maplab.inference. Figures are added in a later step.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import maplab as ml
import _nb08_helpers as h8

M = ml.metrics

CORE = h8.CORE
BENCH = h8.BENCH
BASES = CORE + BENCH
WINDOWS = h8.WINDOWS
LIVE = pd.Timestamp("2010-02-02")
B, BLOCK, SEED = 10000, 21, 20260923

REGISTERED = dict(window=21, min_history=252, cap=1.0, scaling="inverse_vol",
                  target="expanding_mean", update="daily")
SENS = {
    "window 63": dict(REGISTERED, window=63),
    "inverse variance": dict(REGISTERED, scaling="inverse_variance"),
    "cap 1.5": dict(REGISTERED, cap=1.5),
    "month-end": dict(REGISTERED, update="month_end"),
}
EPISODES = {"2020-02/04": ("2020-02-01", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}
KEYS = {"EW": "EW", "GMV(S)": "GMV", "MaxSharpe(S)": "MaxSharpe", "BMV(0.3)": "BMV",
        "BL(0.1,S)": "BL", "MDP(S)": "MDP", "ERC(S)": "ERC", "HRP(S)": "HRP"}
assert list(KEYS) == CORE


def vmp(name: str) -> str:
    return f"VMP({name})"


VMPS = [vmp(n) for n in BASES]
H1_PAIRS = [(vmp(m), m) for m in CORE]
H1_IDS = [f"H1_{KEYS[m]}" for m in CORE]
H2_PAIRS = [(vmp(m), "60/40") for m in CORE]
H2_IDS = [f"H2_{KEYS[m]}" for m in CORE]
D_PAIR = (vmp("60/40"), "60/40")


def _years(idx) -> float:
    return (idx[-1] - idx[0]).days / 365.25


def _win_mask(idx, w, split_ts):
    if w == "full":
        return np.ones(len(idx), dtype=bool)
    if w == "train (<= split)":
        return np.asarray(idx <= split_ts)
    if w == "test (> split)":
        return np.asarray(idx > split_ts)
    raise ValueError(f"unknown window {w!r}")


# ---------------------------------------------------------------- runs
def build_base_runs(simple_returns, panel) -> dict:
    """The 8 nb08 core runs and the 60/40 benchmark, same constructors."""
    strats = h8.make_strategies()
    runs = {}
    for name in CORE:
        net, wlog, diag = ml.backtest(strats[name], simple_returns, panel)
        runs[name] = {"net": net, "wlog": wlog, "diag": diag, "strat": strats[name]}
    runs.update(h8.build_benchmarks(simple_returns, panel))
    return runs


def build_overlays(runs, rf_daily, params=REGISTERED, names=BASES) -> dict:
    """label VMP(m) -> {net, h (held fraction), c (target), diag, base}."""
    out = {}
    for name in names:
        net, h, diag = ml.vol_overlay(runs[name]["net"], rf_daily, **params)
        out[vmp(name)] = {"net": net, "h": h, "c": diag["target_c"], "diag": diag, "base": name}
    return out


# ---------------------------------------------------------------- gates
def identity_table(runs, overlays, rf_daily) -> pd.DataFrame:
    """G2: c = 1 reproduces base bit-for-bit; net rebuilt from h, r, BIL, cost;
    live date; warm-up c = 1; range of h; indexes aligned."""
    rows = []
    for lab, o in overlays.items():
        base = runs[o["base"]]["net"]
        p = o["diag"]["params"]
        kw = {k: p[k] for k in REGISTERED}
        n1, _, d1 = ml.vol_overlay(base, rf_daily, **dict(kw, min_history=len(base)))
        r = base.to_numpy()
        f = rf_daily.reindex(base.index).to_numpy()
        h = o["h"].to_numpy()
        rebuilt = h * r + (1.0 - h) * f - o["diag"]["turnover"].to_numpy() * p["cost_bps"] / 1e4
        mh = p["min_history"]
        rows.append({
            "overlay": lab,
            "c1_identical": bool(np.array_equal(n1.to_numpy(), r)) and d1["total_cost"] == 0.0,
            "rebuild_max_abs": float(np.max(np.abs(rebuilt - o["net"].to_numpy()))),
            "live_date": base.index[mh],
            "warmup_c_all_one": bool((o["c"].iloc[:mh] == 1.0).all()),
            "h_min": float(h.min()),
            "h_max": float(h.max()),
            "index_equal": bool(o["net"].index.equals(base.index) and o["h"].index.equals(base.index)),
        })
    return pd.DataFrame(rows).set_index("overlay")


def shock_check(runs, rf_daily, name, at, bump=0.05, params=REGISTERED) -> dict:
    """G3: add `bump` to the base return on day `at`; c up to and including `at`
    must not move (it uses information through t-1); c the next day must."""
    base = runs[name]["net"]
    shocked = base.copy()
    shocked.loc[at] += bump
    _, _, d0 = ml.vol_overlay(base, rf_daily, **params)
    _, _, d1 = ml.vol_overlay(shocked, rf_daily, **params)
    c0, c1 = d0["target_c"].to_numpy(), d1["target_c"].to_numpy()
    i = base.index.get_loc(pd.Timestamp(at))
    return {"run": name, "at": pd.Timestamp(at),
            "max_abs_dc_through_at": float(np.max(np.abs(c1[: i + 1] - c0[: i + 1]))),
            "dc_next_day": float(c1[i + 1] - c0[i + 1])}


# ---------------------------------------------------------------- tables
def exposure_table(overlays, split_ts, start=LIVE) -> pd.DataFrame:
    """Per overlay and window (from `start`): mean target c, mean held h,
    share of days braking (h < 1), min c and its date, the overlay's own
    annual turnover sum|h_t - d_{t-1}| and its cost drag in bps/yr."""
    rows = []
    for lab, o in overlays.items():
        c, h, tau = o["c"], o["h"], o["diag"]["turnover"]
        live = np.asarray(c.index >= start)
        for w in WINDOWS:
            sel = live & _win_mask(c.index, w, split_ts)
            cw, hw, tw = c[sel], h[sel], tau[sel]
            yrs = _years(cw.index)
            rows.append({"overlay": lab, "window": w,
                         "mean_c": float(cw.mean()), "mean_h": float(hw.mean()),
                         "share_braking": float((hw < 1.0).mean()),
                         "min_c": float(cw.min()), "min_c_date": cw.idxmin(),
                         "ann_overlay_turnover": float(tw.sum() / yrs),
                         "cost_drag_bps": float(tw.sum() / yrs * ml.COST_BPS)})
    return pd.DataFrame(rows).set_index(["overlay", "window"])


def scoring_frames(all_runs, names, rf_daily, split_ts, start=LIVE) -> dict:
    """Excess returns (net - BIL) from `start`, split into the three windows."""
    X = h8.excess_frame(all_runs, names, rf_daily)
    X = X.loc[X.index >= start]
    XW = h8.window_frames(X, split_ts)
    assert tuple(XW) == tuple(WINDOWS), tuple(XW)
    return XW


def perf_table(all_runs, names, split_ts, rf_daily, start=LIVE) -> pd.DataFrame:
    """D_perf table from `start`. Base runs: rebalance turnover (labels in window,
    initial_build=False -- no window starts at the build). Overlays: the overlay's
    own risky <-> BIL turnover only (the base's trading is already in its net)."""
    rows = []
    for name in names:
        run = all_runs[name]
        net = run["net"].loc[run["net"].index >= start]
        for w in WINDOWS:
            r = net.loc[_win_mask(net.index, w, split_ts)]
            row = {"strategy": name, "window": w,
                   "ann_return": M.ann_return(r), "ann_vol": M.ann_vol(r),
                   "sharpe": M.ann_sharpe(r, rf_daily), "max_dd": M.max_drawdown(r),
                   "calmar": M.calmar(r), "hit_rate": M.hit_rate(r)}
            if "h" in run:
                tau = run["diag"]["turnover"].loc[r.index]
                row["ann_turnover"] = float(tau.sum() / _years(r.index))
                row["turnover_kind"] = "overlay"
            else:
                t = run["diag"]["turnover_per_rebalance"]
                t = t.loc[np.asarray(t.index >= start) & _win_mask(t.index, w, split_ts)]
                row["ann_turnover"] = M.ann_turnover(t, initial_build=False)
                row["turnover_kind"] = "rebalance"
            row["cost_drag"] = row["ann_turnover"] * ml.COST_BPS / 1e4
            rows.append(row)
    return pd.DataFrame(rows).set_index(["strategy", "window"])


def episode_table(all_runs, names) -> pd.DataFrame:
    """Registered episodes: cumulative return and in-window peak-to-trough
    drawdown; mean and min held fraction for overlays."""
    rows = []
    for name in names:
        run = all_runs[name]
        for ep, (a, b) in EPISODES.items():
            r = run["net"].loc[a:b]
            row = {"run": name, "episode": ep, "first": r.index[0], "last": r.index[-1],
                   "cum_return": float((1.0 + r).prod() - 1.0), "max_dd": float(M.max_drawdown(r))}
            if "h" in run:
                hh = run["h"].loc[a:b]
                row["mean_h"] = float(hh.mean())
                row["min_h"] = float(hh.min())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["run", "episode"])


# ---------------------------------------------------------------- statistics
def bootstrap_full(XW, cols):
    """One stationary-bootstrap index draw on the full window, shared by every
    series; (B, k) Sharpe draws (ddof 1) as a DataFrame, plus the index chunks."""
    X = XW["full"]
    idx = list(ml.robust.bootstrap_index_chunks(len(X), B, BLOCK, seed=SEED))
    draws = ml.robust.bootstrap_sharpe(X[cols], idx)
    assert draws.shape == (B, len(cols))
    return pd.DataFrame(draws, columns=cols), idx


def dsr_family(XW, pairs, ids, SR_star):
    """Registered ΔSR family: delta-method HAC ΔSR (full and test), bootstrap
    95% CI and p, Holm (descriptive), Romano-Wolf on t* = (ΔSR* - ΔŜR)/se_NW,
    centred on the bootstrap estimator (ddof 1). Returns (table, stats, t_star)."""
    X_full, X_test = XW["full"], XW["test (> split)"]
    sr = X_full.mean() / X_full.std(ddof=1) * np.sqrt(ml.TRADING_DAYS)
    rows, tstar = [], []
    for (a, b), hid in zip(pairs, ids):
        r = ml.robust.sharpe_diff_hac(X_full[a], X_full[b])
        rt = ml.robust.sharpe_diff_hac(X_test[a], X_test[b])
        d_hat = float(sr[a] - sr[b])
        d_star = (SR_star[a] - SR_star[b]).to_numpy()
        lo, hi = ml.robust.percentile_ci(d_star[:, None], 0.95)
        tstar.append((d_star - d_hat) / r["se"])
        rows.append({"id": hid, "a": a, "b": b,
                     "dSR_full": r["d_sr"], "se": r["se"], "t": r["t"], "p_raw": r["p"], "L": r["L"],
                     "dSR_test": rt["d_sr"], "L_test": rt["L"],
                     "same_sign_test": bool(np.sign(rt["d_sr"]) == np.sign(r["d_sr"])),
                     "d_hat_ddof1": d_hat,
                     "ci_lo": float(np.ravel(lo)[0]), "ci_hi": float(np.ravel(hi)[0]),
                     "boot_p": float(np.ravel(ml.robust.bootstrap_p(d_hat, d_star))[0])})
    tbl = pd.DataFrame(rows).set_index("id")
    t_star = np.column_stack(tstar)
    tbl["p_holm"] = ml.robust.holm(tbl["p_raw"].to_numpy())
    tbl["p_rw"] = ml.robust.romano_wolf(tbl["t"].to_numpy(), t_star)
    stats = {hid: dict(display=f"ΔSR {row['dSR_full']:+.2f}, t {row['t']:.2f}, RW p {row['p_rw']:.3f}",
                       p_adj=float(row["p_rw"]), same_sign_test=bool(row["same_sign_test"]))
             for hid, row in tbl.iterrows()}
    return tbl, stats, t_star


def sensitivity_table(runs, rf_daily, split_ts, registered, names=BASES) -> pd.DataFrame:
    """D_sens: each variant on every base; full and test windows from LIVE.
    Never tested. dSR_vs_registered compares with the registered overlay."""
    rows = []
    for var, params in {"registered": REGISTERED, **SENS}.items():
        ov = registered if var == "registered" else build_overlays(runs, rf_daily, params, names)
        XW = scoring_frames({**runs, **ov}, names + [vmp(n) for n in names], rf_daily, split_ts)
        for w in ("full", "test (> split)"):
            X = XW[w]
            sr = X.mean() / X.std(ddof=1) * np.sqrt(ml.TRADING_DAYS)
            for n in names:
                o = ov[vmp(n)]
                sel = X.index
                tau = o["diag"]["turnover"].loc[sel]
                rows.append({"variant": var, "base": n, "window": w,
                             "SR_vmp": float(sr[vmp(n)]), "dSR_vs_base": float(sr[vmp(n)] - sr[n]),
                             "ann_vol": float(M.ann_vol(o["net"].loc[sel])),
                             "max_dd": float(M.max_drawdown(o["net"].loc[sel])),
                             "mean_h": float(o["h"].loc[sel].mean()),
                             "cost_drag_bps": float(tau.sum() / _years(sel) * ml.COST_BPS)})
    t = pd.DataFrame(rows)
    ref = t[t["variant"] == "registered"].set_index(["base", "window"])["SR_vmp"]
    t["dSR_vs_registered"] = t["SR_vmp"].to_numpy() - ref.loc[list(zip(t["base"], t["window"]))].to_numpy()
    return t.set_index(["variant", "base", "window"])


# ---------------------------------------------------------------- figures
def exposure_by_year(overlays, core) -> pd.DataFrame:
    """Mean held fraction h per calendar year from LIVE, per core overlay, plus
    the median across them."""
    H = pd.DataFrame({m: overlays[vmp(m)]["h"] for m in core})
    H = H.loc[H.index >= LIVE]
    yr = H.groupby(H.index.year).mean()
    yr["core median"] = yr.median(axis=1)
    yr.index.name = "year"
    return yr


def exposure_figure(overlays, core, bench="60/40", figsize=(10, 4)):
    """Held fraction h from LIVE -- range and median across the core
    overlays, with the benchmark overlay on top."""
    H = pd.DataFrame({m: overlays[vmp(m)]["h"] for m in core})
    H = H.loc[H.index >= LIVE]
    hb = overlays[vmp(bench)]["h"]
    hb = hb.loc[hb.index >= LIVE]
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(H.index, H.min(axis=1), H.max(axis=1), color="0.82", lw=0, label="core range")
    ax.plot(H.index, H.median(axis=1), color="0.15", lw=0.9, label="core median")
    ax.plot(hb.index, hb, color=ml.plotting.METHOD_COLORS["60/40"], lw=0.7, label=vmp(bench))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held fraction h")
    ax.set_title("Overlay exposure from 2010-02-02", pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3)
    fig.tight_layout()
    return fig


def arrows_figure(perf, names, window="full", figsize=(8.5, 6), tol=(0.12, 0.04)):
    """One arrow per run from the base (hollow, labelled) to its overlay
    (filled) in (ann_vol, ann_return) for `window`. A label within `tol` (share
    of the x and y data ranges) of one already placed in the same slot moves
    one slot (11 points) down."""
    fig, ax = plt.subplots(figsize=figsize)
    pts = []
    for n in names:
        b, v = perf.loc[(n, window)], perf.loc[(vmp(n), window)]
        p0 = (float(b["ann_vol"]), float(b["ann_return"]))
        p1 = (float(v["ann_vol"]), float(v["ann_return"]))
        pts.append((n, p0, p1, ml.plotting.run_color(n)))
    xs = [p[k][0] for p in pts for k in (1, 2)]
    ys = [p[k][1] for p in pts for k in (1, 2)]
    xr, yr = max(xs) - min(xs), max(ys) - min(ys)
    placed = []
    for n, p0, p1, c in pts:
        ax.plot(*p0, "o", ms=6, mfc="none", mec=c, mew=1.2)
        ax.plot(*p1, "o", ms=6, color=c)
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        slot = 0
        while any(s == slot and abs(p0[0] - x) <= tol[0] * xr and abs(p0[1] - y) <= tol[1] * yr
                  for x, y, s in placed):
            slot += 1
        placed.append((p0[0], p0[1], slot))
        ax.annotate(n, p0, textcoords="offset points", xytext=(6, 2 - 11 * slot), fontsize=8, color=c)
    ax.set_xlim(min(xs) - 0.05 * xr, max(xs) + 0.15 * xr)
    ax.set_ylim(min(ys) - 0.08 * yr, max(ys) + 0.08 * yr)
    ax.set_xlabel("annualized volatility (%)")
    ax.set_ylabel("annualized return (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Base (hollow, labelled) to VMP (filled), {window} window from 2010-02-02")
    fig.tight_layout()
    return fig


def _episode_dd(r: pd.Series) -> pd.Series:
    """Drawdown path within an episode; wealth starts at 1 on its first day."""
    w = (1.0 + r).cumprod()
    return w / w.cummax().clip(lower=1.0) - 1.0


def episode_figure(all_runs, core, bench="60/40", figsize=(11, 6.5)):
    """Per registered episode, drawdown (median across the core runs, base
    solid / VMP dashed; benchmark in its colour) and held fraction h (core range,
    core median, benchmark overlay)."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter
    col = ml.plotting.METHOD_COLORS[bench]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="row")
    for i, (ep, (a, b)) in enumerate(EPISODES.items()):
        axd, axh = axes[i, 0], axes[i, 1]
        DDb = pd.DataFrame({m: _episode_dd(all_runs[m]["net"].loc[a:b]) for m in core})
        DDv = pd.DataFrame({m: _episode_dd(all_runs[vmp(m)]["net"].loc[a:b]) for m in core})
        axd.plot(DDb.index, DDb.median(axis=1), color="0.15", lw=1.2, label="core median, base")
        axd.plot(DDv.index, DDv.median(axis=1), color="0.15", lw=1.2, ls="--", label="core median, VMP")
        db = _episode_dd(all_runs[bench]["net"].loc[a:b])
        dv = _episode_dd(all_runs[vmp(bench)]["net"].loc[a:b])
        axd.plot(db.index, db, color=col, lw=1.2, label=bench)
        axd.plot(dv.index, dv, color=col, lw=1.2, ls="--", label=vmp(bench))
        axd.yaxis.set_major_formatter(PercentFormatter(1.0))
        axd.set_title(f"{ep}: drawdown")
        H = pd.DataFrame({m: all_runs[vmp(m)]["h"].loc[a:b] for m in core})
        hb = all_runs[vmp(bench)]["h"].loc[a:b]
        axh.fill_between(H.index, H.min(axis=1), H.max(axis=1), color="0.82", lw=0, label="core range")
        axh.plot(H.index, H.median(axis=1), color="0.15", lw=1.0, label="core median")
        axh.plot(hb.index, hb, color=col, lw=1.0, label=vmp(bench))
        axh.set_ylim(0, 1.05)
        axh.set_title(f"{ep}: held fraction h")
        for ax in (axd, axh):
            loc = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    handles, labels = [], []
    for a in (axes[0, 0], axes[0, 1]):
        for hd, lb in zip(*a.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Registered episodes")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def sensitivity_heatmap(sens, names=BASES, window="full", figsize=(7.5, 5)):
    """Sharpe of each sensitivity variant minus the registered overlay's,
    per base run, for `window`."""
    t = sens.xs(window, level="window")["dSR_vs_registered"].unstack("variant")
    t = t.loc[list(names), list(SENS)]
    v = t.to_numpy(dtype=float)
    lim = float(np.abs(v).max())
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(v.shape[1]), list(t.columns))
    ax.set_yticks(range(v.shape[0]), list(t.index))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v[i, j]) > 0.6 * lim else "0.1")
    fig.colorbar(im, ax=ax, label="ΔSR vs registered overlay")
    ax.set_title(f"Sensitivities: Sharpe minus registered overlay's, {window} window")
    fig.tight_layout()
    return fig
