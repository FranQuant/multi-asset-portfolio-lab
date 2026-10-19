"""Unit tests for maplab.roster (paper roster, batch 1) — synthetic data only, no real cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

import maplab as ml
from maplab.backtest import backtest
from maplab.contract import COV_LOOKBACK
from maplab.data import Panel
from maplab.models import EqualWeight, FixedWeight, GMV, InverseVol
from maplab.roster import (
    Ensemble, ExcludeAssets, MaxEntropy, MinCVaR, MinimumCorrelation, TECap, TailRiskParity, VolTarget,
    _max_entropy_sharpe_floor, _min_corr_weights, _min_cvar_long_only, _es_parity_long_only,
    es_contributions, es_parity_kkt_residual, historical_cvar, on_universe, tracking_error,
)

# Demo B labels columns by the paper's asset class, not by ticker: two of the paper's
# classes trade as tickers the locked harness reserves (BIL = RF_TICKER, IWM = a factor ticker).
U18 = ["US_LC", "US_SC", "US_VAL", "US_GRO", "INTL_DEV", "EM",
       "UST_SHORT", "UST_INT", "UST_LONG", "IG_CORP", "HY_CORP", "INTL_SOV", "INTL_CORP", "EM_DEBT",
       "REIT", "GOLD", "COMMOD", "CASH"]
B6040 = {"US_LC": 0.6, "US_AGG": 0.4}             # AGG is not one of the 18 classes
B18 = {"US_LC": 0.6, "UST_INT": 0.4}            # an in-universe stand-in benchmark for synthetic tests


def make_panel(n_days=400, seed=0, universe=U18, df=4):
    """Fat-tailed, correlated synthetic log returns; CASH has near-zero vol."""
    rng = np.random.default_rng(seed)
    n = len(universe)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    L = rng.normal(size=(n, 3))
    vols = np.linspace(0.004, 0.02, n)
    z = rng.standard_t(df, size=(n_days, n)) / np.sqrt(df / (df - 2))
    f = rng.standard_t(df, size=(n_days, 3)) / np.sqrt(df / (df - 2))
    X = 0.0003 + vols * (0.6 * z + 0.4 * (f @ L.T) / np.sqrt((L ** 2).sum(1)))
    logr = pd.DataFrame(X, index=idx, columns=universe)
    if "CASH" in universe:
        logr["CASH"] = 0.00008 + 0.00002 * rng.normal(size=n_days)
    rf = pd.DataFrame(0.00008 + 0.00001 * rng.normal(size=n_days), index=idx, columns=["BIL"])
    return Panel({"returns": logr, "rf": rf}), logr


def check_valid(w, universe=U18):
    assert list(w.index) == list(universe)
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert (w >= -1e-12).all()


# ── on_universe ─────────────────────────────────────────────────────────────

def test_on_universe_retargets_existing_classes_without_touching_them():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    w = on_universe(EqualWeight, U18)()(panel, asof)
    check_valid(w)
    assert np.allclose(w, 1 / 18)
    fw = on_universe(FixedWeight, U18)(B18, name="60/40")(panel, asof)
    assert fw["US_LC"] == 0.6 and fw["UST_INT"] == 0.4 and fw.drop(["US_LC", "UST_INT"]).eq(0).all()
    assert EqualWeight.universe == ml.UNIVERSE            # original class unchanged
    check_valid(on_universe(GMV, U18)()(panel, asof))


# ── Minimum correlation ─────────────────────────────────────────────────────

def test_min_corr_matches_varadi_spreadsheet_example():
    # Varadi's MinCorr spreadsheet example, as encoded in SIT min.corr.excel.portfolio.test:
    # vols 14/18/22%, correlations .90/.85/.70 -> 0.21059807 0.30878663 0.4806153.
    # On this example SIT's min.corr gives the same weights (worked by hand in the PR notes).
    risk = np.array([0.14, 0.18, 0.22])
    C = np.array([[1, 0.90, 0.85], [0.90, 1, 0.70], [0.85, 0.70, 1]])
    Sigma = pd.DataFrame(C * np.outer(risk, risk))
    w = _min_corr_weights(Sigma)
    assert np.allclose(w, [0.21059807, 0.30878663, 0.4806153], atol=1e-8)


def test_min_corr_special_cases_and_invariances():
    risk = np.array([0.1, 0.3])
    S2 = pd.DataFrame(np.array([[1, 0.5], [0.5, 1]]) * np.outer(risk, risk))
    assert np.allclose(_min_corr_weights(S2), [0.75, 0.25])          # n=2: inverse vol
    risk4 = np.array([0.1, 0.2, 0.3, 0.4])
    C = np.full((4, 4), 0.3); np.fill_diagonal(C, 1)
    w = _min_corr_weights(pd.DataFrame(C * np.outer(risk4, risk4)))
    assert np.allclose(w, (1 / risk4) / (1 / risk4).sum())          # equal correlations
    panel, logr = make_panel()
    S = ml.sample_cov(logr.iloc[:252])
    w = _min_corr_weights(S)
    assert np.allclose(_min_corr_weights(S * 7.0), w)                # scale invariance
    p = np.random.default_rng(1).permutation(len(S))
    assert np.allclose(_min_corr_weights(S.iloc[p, p]), w[p])        # permutation equivariance


def test_min_corr_strategy_valid():
    panel, logr = make_panel()
    check_valid(MinimumCorrelation(universe=U18)(panel, logr.index[COV_LOOKBACK]))


# ── CVaR ────────────────────────────────────────────────────────────────────

def test_min_cvar_beats_alternatives_and_matches_its_own_cvar():
    panel, logr = make_panel()
    R = np.expm1(logr.iloc[:252].to_numpy())
    w, obj = _min_cvar_long_only(R, 0.95, "t", None)
    assert np.isclose(obj, historical_cvar(R, w, 0.95), rtol=1e-9)
    rng = np.random.default_rng(3)
    for _ in range(200):
        x = rng.dirichlet(np.ones(R.shape[1]))
        assert historical_cvar(R, x, 0.95) >= obj - 1e-12
    for j in range(R.shape[1]):
        assert historical_cvar(R, np.eye(R.shape[1])[j], 0.95) >= obj - 1e-12


def test_min_cvar_uses_simple_returns_strictly_before_asof():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    w = MinCVaR(universe=U18)(panel, asof)
    check_valid(w)
    R = np.expm1(logr.loc[logr.index < asof].iloc[-COV_LOOKBACK:].to_numpy())
    w_ref, _ = _min_cvar_long_only(R, 0.95, "t", None)
    assert np.allclose(w.to_numpy(), w_ref, atol=1e-12)


# ── Tail-risk parity ────────────────────────────────────────────────────────

def test_tail_rp_kkt_certificate_and_cash_guard():
    panel, logr = make_panel()
    R = np.expm1(logr.drop(columns="CASH").iloc[:252].to_numpy())
    w = _es_parity_long_only(R, 0.95, "t", None)
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert np.isclose(es_contributions(R, w, 0.95).sum(), historical_cvar(R, w, 0.95), rtol=1e-9)
    assert es_parity_kkt_residual(R, w, 0.95) < 1e-6                 # equal budgets via a subgradient
    w_iv = (1 / R.std(0, ddof=1)) / (1 / R.std(0, ddof=1)).sum()
    assert es_parity_kkt_residual(R, w_iv, 0.95) > 1e-3               # the certificate discriminates
    with pytest.raises(ValueError, match="never lose"):
        _es_parity_long_only(np.expm1(logr.iloc[:252].to_numpy()), 0.95, "t", None)


def test_tail_rp_strategy_excludes_cash():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    w = TailRiskParity(universe=U18, exclude=["CASH"])(panel, asof)
    check_valid(w)
    assert w["CASH"] == 0
    iv = on_universe(InverseVol, U18)()(panel, asof)
    assert np.abs(w - iv).max() > 1e-4
    with pytest.raises(ValueError):
        TailRiskParity(universe=U18, exclude=["BIL"])


# ── Max entropy ─────────────────────────────────────────────────────────────

def test_max_entropy_inactive_floor_returns_equal_weights():
    excess = np.array([0.05, 0.04, 0.06])
    S = np.diag([0.04, 0.04, 0.04])
    assert np.allclose(_max_entropy_sharpe_floor(excess, S, 0.1, "t", None), 1 / 3)


def test_max_entropy_binding_floor():
    excess = np.array([0.08, 0.01, 0.00, -0.01])
    S = np.diag([0.04, 0.01, 0.0025, 0.09])
    floor = 0.30
    w = _max_entropy_sharpe_floor(excess, S, floor, "t", None)
    sr = w @ excess / np.sqrt(w @ S @ w)
    assert np.isclose(sr, floor, atol=1e-7)                            # floor binds
    ent = lambda x: -np.sum(x * np.log(np.clip(x, 1e-300, None)))
    rng = np.random.default_rng(0)
    for _ in range(2000):                                              # no feasible point has more entropy
        x = rng.dirichlet(np.ones(4) * 2)
        if x @ excess / np.sqrt(x @ S @ x) >= floor:
            assert ent(x) <= ent(w) + 1e-9


def test_max_entropy_degenerate_excess_falls_back_to_ew_and_records():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    me = MaxEntropy(universe=U18, rf=10.0)                             # rf so high no asset beats it
    w = me(panel, asof)
    assert np.allclose(w, 1 / 18) and me.fallback_dates == [asof]


def test_max_entropy_strategy_meets_floor():
    panel, logr = make_panel(seed=5)
    asof = logr.index[COV_LOOKBACK]
    me = MaxEntropy(universe=U18, rf=0.0)
    w = me(panel, asof).to_numpy()
    check_valid(pd.Series(w, index=U18))
    mu, S = me._estimate(panel, asof)
    from maplab.models import _tangency_long_only
    wt, _ = _tangency_long_only(mu.to_numpy(), S.to_numpy(), "t", asof)
    sr = lambda x: x @ mu.to_numpy() / np.sqrt(x @ S.to_numpy() @ x)
    assert sr(w) >= 0.75 * sr(wt) - 1e-8


# ── Vol targeting ───────────────────────────────────────────────────────────

def test_vol_target_moreira_muir_inverse_variance():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    vt = VolTarget(universe=U18, cash="CASH", target_vol=0.02)
    w = vt(panel, asof)
    check_valid(w)
    risky = [t for t in U18 if t != "CASH"]
    S = ml.sample_cov(logr.loc[logr.index < asof].iloc[-22:])        # previous month, 22 days
    b = pd.Series(1 / 17, index=risky)
    e = min(1.0, 0.02 ** 2 / (b @ S.loc[risky, risky] @ b))
    assert e < 1 and np.isclose(vt.exposure[asof], e)
    assert np.allclose(w[risky], e / 17) and np.isclose(w["CASH"], 1 - e)
    e_iv = min(1.0, 0.02 / np.sqrt(b @ S.loc[risky, risky] @ b))
    w_iv = VolTarget(universe=U18, cash="CASH", target_vol=0.02, scaling="inverse_vol")(panel, asof)
    assert np.isclose(w_iv["CASH"], 1 - e_iv)
    w_hi = VolTarget(universe=U18, cash="CASH", target_vol=5.0)(panel, asof)
    assert np.allclose(w_hi[risky], 1 / 17) and w_hi["CASH"] == 0      # capped at 1, no leverage
    with pytest.raises(ValueError):
        VolTarget(universe=U18, cash="BIL")


def test_vol_target_custom_base_drops_cash_and_renormalises():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    base = {"US_LC": 0.5, "UST_INT": 0.3, "CASH": 0.2}
    vt = VolTarget(universe=U18, cash="CASH", base=base, target_vol=5.0)
    w = vt(panel, asof)
    assert np.isclose(w["US_LC"], 0.625) and np.isclose(w["UST_INT"], 0.375) and w["CASH"] == 0
    with pytest.raises(ValueError):
        VolTarget(universe=U18, cash="CASH", base={"CASH": 1.0})


def test_exclude_assets_zero_fills_and_keeps_label():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    risky = [t for t in U18 if t != "CASH"]
    base = on_universe(GMV, risky)()
    ex = ExcludeAssets(base, universe=U18)
    w = ex(panel, asof)
    check_valid(w)
    assert w["CASH"] == 0 and ex.excluded == ["CASH"] and ex.label == base.label
    assert np.allclose(w[risky], on_universe(GMV, risky)()(panel, asof))
    full = on_universe(GMV, U18)()(panel, asof)
    assert full["CASH"] > 0.5                                         # why cash is excluded
    with pytest.raises(ValueError):
        ExcludeAssets(on_universe(GMV, U18 + ["X"])(), universe=U18)


# ── TE cap ──────────────────────────────────────────────────────────────────

def test_te_cap_meets_cap_and_leaves_compliant_portfolios_alone():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    base_cls = on_universe(EqualWeight, U18)
    _, S = on_universe(GMV, U18)()._estimate(panel, asof)
    b = pd.Series(B18).reindex(U18, fill_value=0).to_numpy()
    te_ew = tracking_error(np.full(18, 1 / 18), b, S.to_numpy())
    capped = TECap(base_cls(), B18, cap=te_ew / 2)
    w = capped(panel, asof)
    check_valid(w)
    assert np.isclose(tracking_error(w.to_numpy(), b, S.to_numpy()), te_ew / 2, rtol=1e-12)
    assert np.isclose(capped.lam[asof], 0.5) and np.isclose(capped.te_before[asof], te_ew)
    loose = TECap(base_cls(), B18, cap=1.0)
    assert np.allclose(loose(panel, asof), 1 / 18) and loose.lam[asof] == 1.0
    with pytest.raises(ValueError):
        TECap(base_cls(), B6040)                                       # US_AGG not in U18


# ── Ensembles ───────────────────────────────────────────────────────────────

def members():
    return [on_universe(EqualWeight, U18)(), MinimumCorrelation(universe=U18),
            MinCVaR(universe=U18), on_universe(FixedWeight, U18)(B18, name="60/40")]


def test_ensemble_average_and_trimmed_mean():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    avg = Ensemble(members(), "average")
    w = avg(panel, asof)
    W = avg.member_weights[asof]
    assert np.allclose(w, W.mean()) and np.allclose(avg.mix[asof], 0.25)
    tm = Ensemble(members(), "trimmed_mean", trim=0.25)
    wt = tm(panel, asof)
    ref = trim_mean(tm.member_weights[asof].to_numpy(), 0.25, axis=0)
    assert np.allclose(wt, ref / ref.sum())
    check_valid(wt)


@pytest.mark.parametrize("method", ["inverse_te_benchmark", "inverse_te_centroid"])
def test_ensemble_inverse_te(method):
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    ms = members()[:3]                                                # exclude the benchmark itself
    ens = Ensemble(ms, method, benchmark=B18)
    w = ens(panel, asof)
    check_valid(w)
    W = ens.member_weights[asof].to_numpy()
    _, S = ens._estimate(panel, asof)
    ref = pd.Series(B18).reindex(U18, fill_value=0).to_numpy() if method.endswith("benchmark") else W.mean(0)
    te = np.array([tracking_error(W[k], ref, S.to_numpy()) for k in range(3)])
    mix = (1 / te) / (1 / te).sum()
    assert np.allclose(ens.mix[asof], mix) and np.allclose(w, mix @ W)
    assert ens.fallback_dates == []


def test_ensemble_inverse_te_floor_records_member_on_benchmark():
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    ens = Ensemble(members(), "inverse_te_benchmark", benchmark=B18)   # 60/40 member: TE = 0
    ens(panel, asof)
    assert ens.fallback_dates == [asof] and ens.mix[asof]["60/40"] > 0.9


def test_ensemble_of_capped_members_is_compliant():
    # TE is a norm, so any convex combination of members with TE <= cap also has TE <= cap.
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK]
    cap = 0.01
    b = pd.Series(B18).reindex(U18, fill_value=0).to_numpy()
    for method in ("average", "inverse_te_benchmark", "inverse_te_centroid"):
        ens = Ensemble([TECap(m, B18, cap=cap) for m in members()[:3]], method, benchmark=B18)
        w = ens(panel, asof)
        _, S = ens._estimate(panel, asof)
        assert tracking_error(w.to_numpy(), b, S.to_numpy()) <= cap + 1e-12, method


def test_ensemble_rejects_mixed_universes_and_bad_method():
    with pytest.raises(ValueError):
        Ensemble([on_universe(EqualWeight, U18)(), EqualWeight()], "average")
    with pytest.raises(ValueError):
        Ensemble(members(), "median")
    with pytest.raises(ValueError):
        Ensemble(members(), "inverse_te_benchmark")


# ── No look-ahead and the harness ───────────────────────────────────────────

def all_strategies():
    ms = members()
    return [MinimumCorrelation(universe=U18), MinCVaR(universe=U18), TailRiskParity(universe=U18, exclude=["CASH"]),
            MaxEntropy(universe=U18), VolTarget(universe=U18, cash="CASH"),
            TECap(MinimumCorrelation(universe=U18), B18, cap=0.02),
            Ensemble(ms, "inverse_te_centroid")]


@pytest.mark.parametrize("idx", range(7))
def test_no_look_ahead(idx):
    panel, logr = make_panel()
    asof = logr.index[COV_LOOKBACK + 5]
    w1 = all_strategies()[idx](panel, asof)
    future = logr.copy()
    future.loc[future.index >= asof] *= -3.0                         # rewrite the future
    panel2 = Panel({"returns": future, "rf": panel.slice(logr.index[-1] + pd.Timedelta(days=1), "rf")})
    w2 = all_strategies()[idx](panel2, asof)
    assert np.allclose(w1, w2, atol=1e-12)


def test_roster_runs_through_the_locked_harness():
    panel, logr = make_panel(n_days=330)
    simple = np.expm1(logr.loc[:"2021-03-31"])                        # harness segments end at month-ends
    for s in [MinimumCorrelation(universe=U18), TailRiskParity(universe=U18, exclude=["CASH"]),
              Ensemble(members(), "average")]:
        net, wlog, diag = backtest(s, simple, panel)
        assert net.notna().all() and len(wlog) == diag["n_rebalances"] >= 2
        assert np.allclose(wlog.sum(axis=1), 1.0)


def test_families_have_colours():
    for cls in (MinimumCorrelation, MinCVaR, TailRiskParity, MaxEntropy, VolTarget, Ensemble):
        assert cls.family in ml.FAMILY_COLORS
