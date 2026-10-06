"""Notebook 08 checks run on the real price cache: the quoted bootstrap p and active returns, the
cross-checks between the notebook's own computations (Sharpe, ddof, Newey–West t, CAPM α, plot titles)."""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import maplab as ml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not (ROOT / "data" / "cache" / "prices.parquet").exists(),
                       reason="data/cache/prices.parquet not found (run scripts/build_panel.py)"),
]


@pytest.fixture(scope="module")
def nb():
    """Data, runs, summaries and excess-return frames built exactly as notebook 08's data, §3 and §4 cells do."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        from helpers import nb08 as h
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    prices = ml.load_prices()
    log_returns = ml.to_log_returns(prices)
    simple_returns = ml.to_simple_returns(prices)
    rf_daily = ml.load_rf_returns()["BIL"]
    panel = ml.Panel({"prices": prices, "returns": log_returns, "rf": ml.load_rf_returns()})
    split_ts = pd.Timestamp(ml.TRAIN_TEST_SPLIT)
    MKT_simple = (simple_returns["SPY"] - rf_daily).dropna()

    CORE, REF, ALPHA = h.CORE, h.REF, h.ALPHA_FAMILY
    BENCH = h.BENCH
    results = h.build_runs(simple_returns, panel)
    bench = h.build_benchmarks(simple_returns, panel)
    all_runs = {**results, **bench}

    summary_table = h.summary_table(results, ALPHA, split_ts, rf_daily)
    bench_summary = h.summary_table(bench, BENCH, split_ts, rf_daily)
    capm_ols = ml.inference.capm_table(results, ALPHA, MKT_simple, rf_daily)

    X = h.excess_frame(results, ALPHA, rf_daily)
    XW = h.window_frames(X, split_ts)
    return {"h": h, "rf_daily": rf_daily, "split_ts": split_ts, "MKT_simple": MKT_simple,
            "CORE": CORE, "REF": REF, "ALPHA": ALPHA, "BENCH": BENCH,
            "results": results, "all_runs": all_runs, "summary_table": summary_table,
            "bench_summary": bench_summary, "capm_ols": capm_ols, "XW": XW,
            "X_full": XW["full"], "X_test": XW["test (> split)"]}


def test_drawdown_panel_titles_match_the_performance_table(nb):
    h, CORE, BENCH = nb["h"], nb["CORE"], nb["BENCH"]
    perf = pd.concat([nb["summary_table"].loc[CORE + nb["REF"]], nb["bench_summary"]])
    pf = perf.xs("full", level="window")
    perf_shown = pd.DataFrame({"max DD (%)": pf["max_dd"]})

    WD = CORE + BENCH
    fig = ml.plotting.wealth_drawdown({m: nb["all_runs"][m]["net"] for m in WD}, nb["rf_daily"], nb["split_ts"])
    dd_tab = h.drawdown_dates(nb["all_runs"], WD)["max_dd"]
    for ax, m in zip(fig.axes[1:], WD):
        want = f"{m} — max DD −{abs(round(100 * dd_tab[m], 1)):.1f}%"
        assert ax.get_title() == want, (ax.get_title(), want)
        assert round(100 * dd_tab[m], 1) == round(100 * perf_shown.loc[m, "max DD (%)"], 1)


def test_active_returns_vs_ew_match_the_quoted_values(nb):
    active = nb["h"].active_stats(nb["all_runs"], nb["CORE"], ["EW"] + nb["BENCH"], nb["split_ts"])

    NB_ACTIVE_VS_EW = {"BL(0.1,S)": -1.36, "MDP(S)": -3.17, "ERC(S)": -2.78, "HRP(S)": -3.30}
    for m, exp in NB_ACTIVE_VS_EW.items():
        got = round(100 * float(active.loc[("EW", m), "active_ret"]), 2)
        assert got == exp, f"{m} vs EW {got} != {exp} %/yr"


def test_bootstrap_point_sharpe_equals_summary_sharpe(nb):
    CR = nb["CORE"] + nb["REF"]
    sr_gap = {}
    for w, Xw in nb["XW"].items():
        sr_point = Xw[CR].mean() / Xw[CR].std(ddof=1) * np.sqrt(ml.TRADING_DAYS)
        sr_gap[w] = (sr_point - nb["summary_table"].xs(w, level="window").loc[CR, "sharpe"]).abs()
    sr_gap = pd.DataFrame(sr_gap)
    assert float(sr_gap.max().max()) <= 1e-10


@pytest.fixture(scope="module")
def h1(nb):
    """§7 H1 table (ΔSR of each method vs EW), built as notebook 08's §4 and §7 cells do."""
    CORE, X_full, X_test = nb["CORE"], nb["X_full"], nb["X_test"]
    B, BLOCK, SEED = 10000, 21, 20260923
    CR = CORE + nb["REF"]
    idx = list(ml.robust.bootstrap_index_chunks(len(X_full), B, BLOCK, seed=SEED))
    draws = ml.robust.bootstrap_sharpe(X_full[CR], idx)
    SR_full_star = pd.DataFrame(draws, columns=CR)

    H1_METHODS = [m for m in CORE if m != "EW"]
    sr_full = X_full[CORE].mean() / X_full[CORE].std(ddof=1) * np.sqrt(ml.TRADING_DAYS)

    rows = []
    for m in H1_METHODS:
        r = ml.robust.sharpe_diff_hac(X_full[m], X_full["EW"])
        rt = ml.robust.sharpe_diff_hac(X_test[m], X_test["EW"])
        d_hat = float(sr_full[m] - sr_full["EW"])
        d_star = (SR_full_star[m] - SR_full_star["EW"]).to_numpy()
        rows.append({"method": m, "dSR_full": r["d_sr"], "se": r["se"], "t": r["t"], "p_raw": r["p"],
                     "dSR_test": rt["d_sr"], "same_sign_test": bool(np.sign(rt["d_sr"]) == np.sign(r["d_sr"])),
                     "boot_p": float(ml.robust.bootstrap_p(d_hat, d_star)), "d_hat_ddof1": d_hat})
    return pd.DataFrame(rows).set_index("method")


def test_h1_hac_dsr_matches_ddof1_dsr(h1):
    ddof_gap = float((h1["dSR_full"] - h1["d_hat_ddof1"]).abs().max())
    assert ddof_gap < 2e-5


def test_h1_smallest_bootstrap_p_is_the_quoted_value(h1):
    assert round(float(h1["boot_p"].min()), 2) == 0.32


def test_ew_column_of_pairwise_t_matrix_equals_h1_t(nb, h1):
    CORE, X_full = nb["CORE"], nb["X_full"]
    H1_METHODS = [m for m in CORE if m != "EW"]
    T_mat = pd.DataFrame(np.nan, index=CORE, columns=CORE)
    for a, b in combinations(CORE, 2):
        r = ml.robust.sharpe_diff_hac(X_full[a], X_full[b])
        T_mat.loc[a, b], T_mat.loc[b, a] = r["t"], -r["t"]

    ew_col_diff = float((T_mat.loc[H1_METHODS, "EW"] - h1["t"]).abs().max())
    assert ew_col_diff <= 1e-12


def test_nw_alpha_equals_ols_alpha_and_interval_excludes_zero_iff_t_ge_196(nb):
    ALPHA, results, rf_daily, MKT_simple = nb["ALPHA"], nb["results"], nb["rf_daily"], nb["MKT_simple"]
    Y_cols, x_ref = {}, None
    for m in ALPHA:
        net = results[m]["net"]
        idx_m = net.index.intersection(rf_daily.index).intersection(MKT_simple.index)
        Y_cols[m] = net.loc[idx_m] - rf_daily.loc[idx_m]
        x_m = MKT_simple.loc[idx_m]
        if x_ref is None:
            x_ref = x_m
    Y = pd.DataFrame(Y_cols)[ALPHA]
    x_df = x_ref.to_frame("MKT")

    alpha_hat, se_nw = pd.Series(np.nan, index=ALPHA), pd.Series(np.nan, index=ALPHA)
    for m in ALPHA:
        fit = ml.robust.ols_hac(Y[m], x_df, L=None)
        alpha_hat[m] = float(fit["coef"]["alpha"]) * ml.TRADING_DAYS
        se_nw[m] = float(fit["se"]["alpha"]) * ml.TRADING_DAYS
    t_nw = alpha_hat / se_nw

    alpha_gap = float((alpha_hat - nb["capm_ols"].loc[ALPHA, "alpha_ann"]).abs().max())
    assert alpha_gap <= 1e-12

    f5 = pd.DataFrame({
        "lo": 100 * (alpha_hat - 1.96 * se_nw).to_numpy(),
        "hi": 100 * (alpha_hat + 1.96 * se_nw).to_numpy(),
    })
    excl = ((f5["lo"] > 0) | (f5["hi"] < 0)).to_numpy()
    assert np.all(excl == (t_nw.abs() >= 1.96).to_numpy())
