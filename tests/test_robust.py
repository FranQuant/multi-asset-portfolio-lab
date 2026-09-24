"""Unit tests for maplab.robust — synthetic data only, no real cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maplab.inference import ols
from maplab.robust import (
    nw_lag,
    newey_west_lrv,
    hac_mean_t,
    ols_hac,
    sharpe_diff_hac,
    bootstrap_index_chunks,
    bootstrap_sharpe,
    bootstrap_capm_alpha,
    percentile_ci,
    bootstrap_p,
    romano_wolf,
    holm,
    bh,
    forecast_bias_decomposition,
    segment_block_bootstrap_mean,
)


def test_nw_lag_values():
    assert nw_lag(4338) == 9
    assert nw_lag(3504) == 8
    assert nw_lag(834) == 6


def test_newey_west_lrv_lag0_and_bartlett():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(300, 3))
    Ud = U - U.mean(axis=0)
    assert np.allclose(newey_west_lrv(U, 0), Ud.T @ Ud / 300, rtol=0, atol=1e-14)

    u = np.array([1.0, 2.0, 3.0])
    # L=1: Γ0 = 14/3, Γ1 = 8/3, weight 1/2 -> 14/3 + 8/3
    assert newey_west_lrv(u, 1, demean=False).shape == (1, 1)
    assert np.isclose(newey_west_lrv(u, 1, demean=False)[0, 0], 22 / 3, rtol=0, atol=1e-14)
    # L=2: + weights 2/3, 1/3 on Γ1 = 8/3, Γ2 = 1
    assert np.isclose(newey_west_lrv(u, 2, demean=False)[0, 0], 80 / 9, rtol=0, atol=1e-14)


def test_newey_west_lrv_ar1():
    phi, T = 0.2, 200_000
    rng = np.random.default_rng(42)
    e = rng.standard_normal(T + 1000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    lrv = newey_west_lrv(x[1000:], 50)[0, 0]
    assert abs(lrv / (1 / (1 - phi) ** 2) - 1) <= 0.03


def test_hac_mean_t_lag0_is_iid_se():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, size=500)
    r = hac_mean_t(x, L=0)
    assert np.isclose(r["se"], x.std(ddof=0) / np.sqrt(500), rtol=0, atol=1e-14)
    assert r["L"] == 0 and r["n"] == 500
    assert hac_mean_t(x)["L"] == nw_lag(500)


def test_ols_hac_coef_and_hc0():
    rng = np.random.default_rng(2)
    n = 800
    X = pd.DataFrame({"MKT": rng.normal(size=n)})
    y = pd.Series(0.2 + 0.7 * X["MKT"].to_numpy() + rng.normal(size=n) * (1 + np.abs(X["MKT"].to_numpy())))
    fit = ols_hac(y, X, L=0)
    ref = ols(y, X)
    assert list(fit["coef"].index) == ["alpha", "MKT"]
    assert np.max(np.abs(fit["coef"].to_numpy() - ref["coef"].to_numpy())) <= 1e-12

    Xd = np.column_stack([np.ones(n), X.to_numpy()])
    e = y.to_numpy() - Xd @ ref["coef"].to_numpy()
    A = np.linalg.inv(Xd.T @ Xd)
    V = A @ (Xd.T @ (Xd * (e ** 2)[:, None])) @ A
    assert np.allclose(fit["se"].to_numpy(), np.sqrt(np.diag(V)), rtol=1e-12, atol=0)
    assert ols_hac(y, X)["L"] == nw_lag(n)


def test_sharpe_diff_hac_symmetry_and_zero():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0004, 0.01, size=2000)
    b = rng.normal(0.0002, 0.008, size=2000)
    assert sharpe_diff_hac(a, a)["d_sr"] == 0.0
    ab, ba = sharpe_diff_hac(a, b), sharpe_diff_hac(b, a)
    assert np.isclose(ab["d_sr"], -ba["d_sr"], rtol=0, atol=1e-14)
    assert np.isclose(ab["se"], ba["se"], rtol=1e-12, atol=0)
    assert np.isclose(ab["p"], ba["p"], rtol=1e-12, atol=0)


def test_sharpe_diff_hac_iid_closed_form():
    # Independent iid normal series: Var(SR_a - SR_b) = (2 + (SR_a^2 + SR_b^2)/2) / T
    # with SR at the sampling (daily) frequency; annualized, SR^2/2 enters as
    # SR_ann^2 / (2 * 252). (The Lo formula with ANNUAL SR applies to annual
    # observations only.)
    T, td = 200_000, 252
    sr_a, sr_b = 0.7, 0.6
    rng = np.random.default_rng(4)
    a = rng.normal(sr_a / np.sqrt(td) * 0.01, 0.01, size=T)
    b = rng.normal(sr_b / np.sqrt(td) * 0.02, 0.02, size=T)
    r = sharpe_diff_hac(a, b)
    T_years = T / td
    closed = np.sqrt((2 + (sr_a ** 2 + sr_b ** 2) / (2 * td)) / T_years)
    assert abs(r["se"] / closed - 1) <= 0.05


def test_sharpe_diff_hac_input_checks():
    a = np.zeros(10) + np.arange(10)
    with pytest.raises(ValueError):
        sharpe_diff_hac(a, a[:-1])
    b = a.copy(); b[3] = np.nan
    with pytest.raises(ValueError):
        sharpe_diff_hac(a, b)
    idx = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError):
        sharpe_diff_hac(pd.Series(a, index=idx), pd.Series(a, index=idx + pd.Timedelta(days=1)))


def test_bootstrap_index_chunks():
    T, B = 500, 250
    chunks = list(bootstrap_index_chunks(T, B, 21, seed=7, chunk=100))
    assert [c.shape for c in chunks] == [(100, T), (100, T), (50, T)]
    allidx = np.concatenate(chunks)
    assert allidx.min() >= 0 and allidx.max() < T
    again = np.concatenate(list(bootstrap_index_chunks(T, B, 21, seed=7, chunk=100)))
    assert np.array_equal(allidx, again)
    other = np.concatenate(list(bootstrap_index_chunks(T, B, 21, seed=8, chunk=100)))
    assert not np.array_equal(allidx, other)


def test_bootstrap_index_mean_run_length():
    T, B, mb = 20_000, 200, 21
    n_starts = 0
    for idx in bootstrap_index_chunks(T, B, mb, seed=11):
        cont = idx[:, 1:] == (idx[:, :-1] + 1) % T
        n_starts += idx.shape[0] + int((~cont).sum())
    assert abs(T * B / n_starts / mb - 1) <= 0.10


def _capm_alpha_lstsq(Y, x, td=252):
    Xd = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(Xd, Y, rcond=None)[0][0] * td


def test_bootstrap_capm_alpha_identity_index():
    rng = np.random.default_rng(5)
    T, k = 1500, 23
    x = rng.normal(0.0004, 0.01, size=T)
    Y = 0.0001 + x[:, None] * rng.uniform(0.2, 1.0, size=k) + rng.normal(0, 0.005, size=(T, k))
    got = bootstrap_capm_alpha(Y, x, [np.arange(T)[None]])
    assert got.shape == (1, k)
    assert np.max(np.abs(got[0] - _capm_alpha_lstsq(Y, x))) <= 1e-12
    got2 = bootstrap_capm_alpha(Y, x, bootstrap_index_chunks(T, 30, 21, seed=1, chunk=7))
    assert got2.shape == (30, k)


def test_bootstrap_sharpe_identity_index():
    rng = np.random.default_rng(6)
    X = rng.normal(0.0003, 0.01, size=(1000, 10))
    got = bootstrap_sharpe(X, [np.arange(1000)[None]])
    ref = X.mean(0) / X.std(0, ddof=1) * np.sqrt(252)
    assert got.shape == (1, 10)
    assert np.max(np.abs(got[0] - ref)) <= 1e-12


def test_percentile_ci_and_bootstrap_p():
    d = np.arange(1, 1001, dtype=float)[:, None] * np.array([1.0, -1.0])
    lo, hi = percentile_ci(d, 0.9)
    assert np.allclose(lo, np.percentile(d, 5, axis=0)) and np.allclose(hi, np.percentile(d, 95, axis=0))
    star = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert np.isclose(bootstrap_p(np.array([1.0]), star)[0], 0.75)


def test_holm_bh_hand_examples():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)
    assert np.allclose(bh(p), [0.02, 0.04, 0.04, 0.02], rtol=0, atol=1e-15)
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_romano_wolf_k1_and_monotone():
    rng = np.random.default_rng(9)
    t_star = rng.standard_normal((5000, 1))
    t_hat = np.array([1.7])
    assert np.isclose(romano_wolf(t_hat, t_star)[0], np.mean(np.abs(t_star[:, 0]) >= 1.7), rtol=0, atol=0)

    k = 6
    t_star = rng.standard_normal((4000, k))
    t_hat = np.array([0.5, 2.8, -1.9, 1.1, -3.2, 0.1])
    p_adj = romano_wolf(t_hat, t_star)
    order = np.argsort(-np.abs(t_hat))
    assert np.all(np.diff(p_adj[order]) >= 0)
    p_single = (np.abs(t_star) >= np.abs(t_hat)).mean(axis=0)
    assert np.all(p_adj >= p_single)


def test_forecast_bias_decomposition():
    rng = np.random.default_rng(10)
    sig = rng.uniform(0.05, 0.15, size=200)
    s = sig * np.exp(rng.normal(-0.05, 0.3, size=200))
    d = forecast_bias_decomposition(s, sig)
    assert abs(d["q_bar"] - (d["half_log_vbar"] - d["J"])) <= 1e-12
    assert d["J"] >= 0
    z = forecast_bias_decomposition(sig, sig)
    assert z["q_bar"] == 0 and z["half_log_vbar"] == 0 and z["J"] == 0 and z["vbar"] == 1


def test_segment_block_bootstrap_mean():
    rng = np.random.default_rng(12)
    v = rng.normal(-0.06, 0.2, size=208)
    a = segment_block_bootstrap_mean(v, block=12, B=2000, seed=3)
    b = segment_block_bootstrap_mean(v, block=12, B=2000, seed=3)
    assert a.shape == (2000,)
    assert np.array_equal(a, b)
    assert abs(a.mean() - v.mean()) <= 0.01
