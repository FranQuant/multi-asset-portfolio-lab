"""Notebook-11-only data, tables and figures for Sharpe inference.

Runs, excess-return frames and windows are notebook 08's (_nb08_helpers); all
statistics come from maplab.sharpe and maplab.robust. Every Sharpe ratio here
is per observation (daily) unless a column says "ann"; annualising (x sqrt(252))
is for labels only.
"""
import tomllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import maplab as ml
import _nb08_helpers as h8
from maplab.plotting import FAMILY_COLORS

CORE = h8.CORE
BENCH = h8.BENCH
RUNS = CORE + BENCH
ALPHA = h8.ALPHA_FAMILY
WINDOWS = h8.WINDOWS
ALPHA_LEVEL = 0.05
SEED = 20261002
SR1_ANNUAL = (0.25, 0.5, 0.75)
DELTAS = (0.1, 0.2, 0.3)
N_H1 = 7                      # nb08 H1 tests (core methods vs EW)
TD = ml.TRADING_DAYS
ANN = float(np.sqrt(TD))

_KEYS = ("full", "train", "test")


# ---------------------------------------------------------------- data
def build_data() -> dict:
    """Excess-return frame (23 α-family runs + 60/40), its three windows, split."""
    prices = ml.load_prices()
    simple_returns = ml.to_simple_returns(prices)
    rf_daily = ml.load_rf_returns()["BIL"]
    panel = ml.Panel({"prices": prices, "returns": ml.to_log_returns(prices),
                      "rf": ml.load_rf_returns()})
    split_ts = pd.Timestamp(ml.TRAIN_TEST_SPLIT)
    results = h8.build_runs(simple_returns, panel)
    bench = h8.build_benchmarks(simple_returns, panel)
    X = h8.excess_frame({**results, **bench}, ALPHA + BENCH, rf_daily)
    assert len(ALPHA) == 23 and set(ALPHA + BENCH) <= set(X.columns)
    return {"X": X, "XW": h8.window_frames(X, split_ts), "split_ts": split_ts}


def repro_gate(XW: dict) -> pd.DataFrame:
    """Annualised Sharpe to 2 dp of the 23 runs and 60/40 in the three windows
    vs registrations/reproduction.toml; raises on any mismatch."""
    path = ml.find_repo_root() / "registrations" / "reproduction.toml"
    with open(path, "rb") as f:
        expected = tomllib.load(f)
    rows = []
    for label, key in {**h8.REPRO_KEYS, **h8.BENCH_REPRO_KEYS}.items():
        for w, k in zip(WINDOWS, _KEYS):
            x = XW[w][label]
            got = round(float(x.mean() / x.std(ddof=1) * ANN), 2)
            rows.append({"run": label, "window": w, "sharpe_2dp": got, "expected": expected[key][k]})
    out = pd.DataFrame(rows)
    out["ok"] = out["sharpe_2dp"] == out["expected"]
    if not out["ok"].all():
        bad = out.loc[~out["ok"]]
        raise AssertionError(f"REPRODUCTION GATE FAILED:\n{bad.to_string()}")
    return out


# ---------------------------------------------------------------- D1
def moments_table(XW: dict, runs=RUNS) -> pd.DataFrame:
    """T, daily SR, annualised SR, skew, Pearson kurtosis, lag-1 ρ per run × window."""
    rows = []
    for run in runs:
        for w in WINDOWS:
            m = ml.sharpe.sample_moments(XW[w][run].to_numpy())
            rows.append({"run": run, "window": w, "T": m["T"], "SR_d": m["sr"], "SR_ann": m["sr"] * ANN,
                         "skew": m["skew"], "kurt": m["kurt"], "rho": m["rho"]})
    return pd.DataFrame(rows)


def sd_newey_west(x) -> float:
    """Delta-method Newey–West sd of the daily SR: y_t = (x_t, x_t²), lag
    nw_lag(T), gradient of m/√(g − m²) with ddof-0 moments."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    m, g = x.mean(), (x ** 2).mean()
    s = np.sqrt(g - m ** 2)
    grad = np.array([1 / s + m ** 2 / s ** 3, -m / (2 * s ** 3)])
    psi = ml.robust.newey_west_lrv(np.column_stack([x, x ** 2]), ml.robust.nw_lag(T))
    return float(np.sqrt(grad @ psi @ grad / T))


def d1_table(XW: dict, runs=RUNS) -> pd.DataFrame:
    """Generalized / iid / Newey–West sd of the SR, PSR vs 0, MinTRL.
    `test_long_enough` (full-window rows only): test-window T >= MinTRL."""
    t_test = len(XW["test (> split)"])
    rows = []
    for run in runs:
        for w in WINDOWS:
            x = XW[w][run].to_numpy()
            m = ml.sharpe.sample_moments(x)
            kw = dict(skew=m["skew"], kurt=m["kurt"], rho=m["rho"])
            sd_gen = float(np.sqrt(ml.sharpe.sharpe_variance(m["sr"], m["T"], **kw)))
            sd_iid = float(np.sqrt(ml.sharpe.sharpe_variance(m["sr"], m["T"])))
            sd_nw = sd_newey_west(x)
            mt = ml.sharpe.min_trl(m["sr"], 0.0, alpha=ALPHA_LEVEL, **kw)
            rows.append({
                "run": run, "window": w, "T": m["T"],
                "sd_gen": sd_gen, "sd_iid": sd_iid, "gen_over_iid": sd_gen / sd_iid,
                "sd_NW": sd_nw, "gen_over_NW": sd_gen / sd_nw,
                "PSR0": ml.sharpe.psr(m["sr"], 0.0, m["T"], **kw),
                "PSR0_iid": ml.sharpe.psr(m["sr"], 0.0, m["T"]),
                "MinTRL_days": mt, "MinTRL_yrs": mt / TD,
                "test_long_enough": bool(t_test >= mt) if (w == "full" and np.isfinite(mt)) else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D2
def d2_table(XW: dict, runs=RUNS, sr1_annual=SR1_ANNUAL) -> pd.DataFrame:
    """Paper-native power of H0: SR = 0 at the window's T and the smallest
    T (days, years) with 80% power, full and test windows, that window's moments."""
    rows = []
    for run in runs:
        for w in (WINDOWS[0], WINDOWS[2]):
            m = ml.sharpe.sample_moments(XW[w][run].to_numpy())
            kw = dict(skew=m["skew"], kurt=m["kurt"], rho=m["rho"])
            for s1 in sr1_annual:
                s1d = s1 / ANN
                t80 = ml.sharpe.t_for_power(0.0, s1d, target=0.8, alpha=ALPHA_LEVEL, **kw)
                rows.append({"run": run, "window": w, "T": m["T"], "SR1_ann": s1,
                             "power": ml.sharpe.power(0.0, s1d, m["T"], alpha=ALPHA_LEVEL, **kw),
                             "T80_days": t80, "T80_yrs": t80 / TD})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D3
def d3_table(XW: dict, methods=None, deltas=DELTAS) -> pd.DataFrame:
    """Power (our adaptation) of nb08's H1 tests (method vs EW, full window,
    Newey–West se) at ΔSR in `deltas`, unadjusted and Bonferroni α/7, and years
    to 80% power with se ∝ 1/√years."""
    methods = [m for m in CORE if m != "EW"] if methods is None else list(methods)
    Xf = XW["full"]
    years = len(Xf) / TD
    sr = Xf[CORE].mean() / Xf[CORE].std(ddof=1) * ANN
    rows = []
    for m in methods:
        r = ml.robust.sharpe_diff_hac(Xf[m], Xf["EW"])
        for d in deltas:
            rows.append({
                "method": m, "Delta": d, "d_sr_hac": r["d_sr"], "d_hat_ddof1": float(sr[m] - sr["EW"]),
                "se": r["se"],
                "power_unadj": ml.sharpe.sharpe_diff_power(d, r["se"], alpha=ALPHA_LEVEL),
                "power_bonf": ml.sharpe.sharpe_diff_power(d, r["se"], alpha=ALPHA_LEVEL, n_tests=N_H1),
                "Y80_unadj_yrs": ml.sharpe.years_for_power(d, r["se"], years, target=0.8, alpha=ALPHA_LEVEL),
                "Y80_bonf_yrs": ml.sharpe.years_for_power(d, r["se"], years, target=0.8,
                                                          alpha=ALPHA_LEVEL / N_H1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D4
def d4_search(XW: dict) -> dict:
    """DSR of the best of the 23 runs (paper route, eq. 28–30) for K = 23
    and the effective K from trial clustering and effective rank, plus the
    sharp-null variance sensitivity."""
    X23 = XW["full"][ALPHA]
    T = len(X23)
    srk = X23.mean() / X23.std(ddof=1)
    best = str(srk.idxmax())
    sr_star = float(srk.max())
    C = np.clip(np.corrcoef(X23.to_numpy().T), -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    C = pd.DataFrame(C, index=ALPHA, columns=ALPHA)
    nK, labels = ml.sharpe.cluster_trials(C.to_numpy(), max_k=len(ALPHA) - 1, seed=SEED)
    labels = pd.Series(labels, index=ALPHA, name="cluster")
    k_er = ml.sharpe.effective_rank(C.to_numpy())
    var_all = float(srk.var(ddof=1))
    mom = [ml.sharpe.sample_moments(X23[c].to_numpy()) for c in ALPHA]
    v0 = float(np.mean([ml.sharpe.sharpe_variance(0.0, T, m["skew"], m["kurt"], m["rho"]) for m in mom]))
    cases = [("K=23, V_all", len(ALPHA), var_all), ("K=nK, V_all", nK, var_all),
             ("K=round(K_er), V_all", int(round(k_er)), var_all),
             ("sharp-null K=23, V0", len(ALPHA), v0), ("sharp-null K=nK, V0", nK, v0)]
    rows = [{"case": name, **ml.sharpe.dsr(sr_star, srk.to_numpy(), K=K, var=V)} for name, K, V in cases]
    return {"srk": srk, "srk_ann": srk * ANN, "best": best, "sr_star": sr_star, "C": C, "nK": int(nK),
            "labels": labels, "k_er": k_er, "var_all": var_all, "v0": v0, "dsr": pd.DataFrame(rows)}


# ---------------------------------------------------------------- figures
_RUN_FAMILY = {"EW": "Benchmark", "60/40": "Benchmark", "GMV(S)": "Risk-based", "MDP(S)": "Risk-based",
               "ERC(S)": "Risk-based", "HRP(S)": "Risk-based", "MaxSharpe(S)": "Return-based",
               "BL(0.1,S)": "Return-based", "BMV(0.3)": "Risk-based"}


def fig_mintrl(d1: pd.DataFrame):
    """MinTRL (years, full-window parameters) per run vs the test and full spans."""
    f = d1[d1["window"] == "full"].sort_values("MinTRL_yrs")
    t_test, t_full = d1.loc[d1["window"] == "test (> split)", "T"].iloc[0], f["T"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(f["run"], f["MinTRL_yrs"], color=[FAMILY_COLORS[_RUN_FAMILY[r]] for r in f["run"]])
    for y, v in enumerate(f["MinTRL_yrs"]):
        ax.text(v + 0.1, y, f"{v:.1f}", va="center", fontsize=8)
    for x, lab in ((t_test / TD, f"test window {t_test / TD:.1f} y"), (t_full / TD, f"full window {t_full / TD:.1f} y")):
        ax.axvline(x, color="black", ls=":", lw=1)
        ax.text(x, 1.01, lab, transform=ax.get_xaxis_transform(), va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("MinTRL vs SR = 0 (years, α = 0.05, full-window skew / kurtosis / ρ)")
    ax.set_title("Minimum track record length", pad=22)
    used = list(dict.fromkeys(_RUN_FAMILY[r] for r in f["run"]))
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[k]) for k in used], used,
              loc="center right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_power_cash(XW: dict, runs=RUNS, sr1_annual=SR1_ANNUAL, max_years=40):
    """Power of H0: SR = 0 vs years of data, median full-window moments of
    `runs` (line) and min–max across the runs' own moments (band)."""
    mom = [ml.sharpe.sample_moments(XW["full"][r].to_numpy()) for r in runs]
    med = {k: float(np.median([m[k] for m in mom])) for k in ("skew", "kurt", "rho")}
    years = np.linspace(0.5, max_years, 80)
    Ts = np.round(years * TD).astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s1, color in zip(sr1_annual, ("#9ecae1", "#4292c6", "#08519c")):
        s1d = s1 / ANN
        curves = np.array([[ml.sharpe.power(0.0, s1d, int(T), m["skew"], m["kurt"], m["rho"], ALPHA_LEVEL)
                            for T in Ts] for m in mom])
        mid = [ml.sharpe.power(0.0, s1d, int(T), alpha=ALPHA_LEVEL, **med) for T in Ts]
        ax.fill_between(years, curves.min(0), curves.max(0), color=color, alpha=0.2)
        ax.plot(years, mid, color=color, label=rf"$\mathrm{{SR}}_1$ = {s1:g}")
    ax.axhline(0.8, color="#888888", lw=0.8, ls="--")
    for yv in (len(XW["test (> split)"]) / TD, len(XW["full"]) / TD):
        ax.axvline(yv, color="black", ls=":", lw=1)
        ax.text(yv, 0.02, f" {yv:.1f} y", fontsize=8)
    ax.set_xlabel("years of daily data")
    ax.set_ylabel("power of $H_0$: SR = 0 (α = 0.05)")
    ax.set_ylim(0, 1)
    ax.set_title("Power against cash (band: min–max over the 9 runs)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# GMV(S) is dashed (drawn last) because it nearly coincides with HRP(S), whose
# SE of ΔSR vs EW is almost identical.
_LINE_LABEL = {"GMV(S)": "GMV(S) (≈ HRP(S))"}


def fig_power_ew(XW: dict, methods=None, deltas=None):
    """Years to 80% power (unadjusted, two-sided) of the H1 Sharpe-difference
    tests vs EW as a function of ΔSR, one line per method, log y."""
    methods = [m for m in CORE if m != "EW"] if methods is None else list(methods)
    deltas = np.linspace(0.05, 0.5, 40) if deltas is None else np.asarray(deltas)
    Xf = XW["full"]
    years = len(Xf) / TD
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(methods, key=lambda m: m == "GMV(S)"):   # GMV(S) last: dashed line over HRP(S)
        se = ml.robust.sharpe_diff_hac(Xf[m], Xf["EW"])["se"]
        y = [ml.sharpe.years_for_power(d, se, years, target=0.8, alpha=ALPHA_LEVEL) for d in deltas]
        style = dict(ls="--", lw=1.4, marker="o", markevery=5, ms=3) if m == "GMV(S)" else dict(ls="-", lw=2.0)
        ax.plot(deltas, y, color=ml.plotting.run_color(m), label=_LINE_LABEL.get(m, m), **style)
    ax.axhline(years, color="black", ls=":", lw=1)
    ax.text(deltas[0], years, f" full window {years:.1f} y", ha="left", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("ΔSR vs EW (annualized)")
    ax.set_ylabel("years for 80% power (α = 0.05, two-sided)")
    ax.set_title("Years needed to detect a Sharpe difference vs EW")
    h_, l_ = ax.get_legend_handles_labels()
    order = sorted(range(len(l_)), key=lambda i: (_RUN_FAMILY[[k for k in methods if _LINE_LABEL.get(k, k) == l_[i]][0]], l_[i]))
    ax.legend([h_[i] for i in order], [l_[i] for i in order], fontsize=8, ncol=2, handlelength=3)
    fig.tight_layout()
    return fig


def block_names(labels: pd.Series) -> dict:
    """Cluster id -> name from the members: the block holding EW is "Balanced /
    EW-like", the block of only MaxSharpe runs is "MaxSharpe", the other is
    "Risk-based". Raises if the blocks do not resolve that way."""
    members = {c: list(labels.index[labels == c]) for c in sorted(labels.unique())}
    names = {}
    for c, mem in members.items():
        if "EW" in mem:
            names[c] = "Balanced / EW-like"
        elif all(r.startswith("MaxSharpe") for r in mem):
            names[c] = "MaxSharpe"
    rest = [c for c in members if c not in names]
    if len(members) != 3 or len(rest) != 1 or len(names) != 2:
        raise ValueError(f"block_names: clusters do not resolve to the three named blocks: {members}")
    names[rest[0]] = "Risk-based"
    return names


def fig_search(d4: dict):
    """Correlation heatmap of the 23 runs ordered by trial cluster."""
    lab = d4["labels"]
    order = list(lab.sort_values(kind="stable").index)
    C = d4["C"].loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    offdiag = C.to_numpy()[~np.eye(len(C), dtype=bool)]
    im = ax.imshow(C.to_numpy(), cmap="Blues", vmin=float(offdiag.min()), vmax=1.0)
    ax.grid(False)
    n = len(order)
    ax.set_xticks(range(n), order, rotation=90, fontsize=7)
    ax.set_yticks(range(n), order, fontsize=7)
    edges = np.flatnonzero(np.diff(lab.loc[order].to_numpy())) + 0.5
    for e in edges:
        ax.axhline(e, color="black", lw=1)
        ax.axvline(e, color="black", lw=1)
    names = block_names(lab)
    for c, name in names.items():
        pos = np.flatnonzero(lab.loc[order].to_numpy() == c)
        ax.text(n - 0.3, pos.mean(), f"{name.replace(' / ', ' /' + chr(10))} ({len(pos)})", rotation=-90, ha="left",
                va="center", fontsize=8, fontweight="bold", clip_on=False)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.1, label="correlation of daily excess returns")
    ax.set_title(f"The 23 runs by trial cluster ({d4['nK']} trial clusters, effective rank = {d4['k_er']:.2f})")
    fig.tight_layout()
    return fig
