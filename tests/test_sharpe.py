"""maplab.sharpe against the worked example and Exhibit 4 of López de Prado,
Lipton & Zoonekynd (2026), "How to Use the Sharpe Ratio". Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

import maplab.sharpe as sh

MU, SIGMA, T, G3, G4, RHO = .036, .079, 24, -2.448, 10.164, .2
SR = MU / SIGMA
KW = dict(skew=G3, kurt=G4, rho=RHO)
TOL = 1.5e-3

EXHIBIT4_SD = [1.00000, 0.82565, 0.74798, 0.70122, 0.66898,
               0.64492, 0.62603, 0.61065, 0.59779, 0.58681]


def test_sharpe_sd_paper_example():
    assert math.sqrt(sh.sharpe_variance(SR, T, **KW)) == pytest.approx(.379, abs=TOL)
    assert math.sqrt(sh.sharpe_variance(SR, T)) == pytest.approx(.214, abs=TOL)


def test_psr_paper_example():
    assert sh.psr(SR, 0, T, **KW) == pytest.approx(.966, abs=TOL)
    assert sh.psr(SR, .1, T, **KW) == pytest.approx(.900, abs=TOL)


def test_min_trl_paper_example():
    assert sh.min_trl(SR, 0, **KW) == pytest.approx(19.543, abs=2e-3)
    assert sh.min_trl(SR, .1, **KW) == pytest.approx(39.369, abs=2e-3)


def test_power_paper_example():
    assert 1 - sh.power(0, .5, T, **KW) == pytest.approx(.411, abs=TOL)
    assert 1 - sh.power(0, .5, T) == pytest.approx(.224, abs=TOL)


def test_dsr_paper_example():
    r = sh.dsr(SR, [], K=10, var=.1)
    assert r["K"] == 10 and r["var"] == .1
    assert r["sr0K"] == pytest.approx(.498, abs=TOL)
    assert r["s0K"] == pytest.approx(.186, abs=TOL)
    assert r["dsr"] == pytest.approx(.410, abs=TOL)


def test_sd_max_std_normal_exhibit4():
    for K, expected in enumerate(EXHIBIT4_SD, start=1):
        assert sh.sd_max_std_normal(K) == pytest.approx(expected, abs=1e-5)


def test_dsr_defaults_from_trials():
    trials = np.array([.01, .02, .03, .05])
    r = sh.dsr(.05, trials)
    assert r["K"] == 4 and r["var"] == pytest.approx(trials.var(ddof=1))
    assert r["s0K"] == pytest.approx(math.sqrt(r["var"]) * sh.sd_max_std_normal(4))


def test_identities():
    assert sh.psr(.3, .3, 100, **KW) == pytest.approx(.5)
    for sr0 in (0., .1):
        t_star = sh.min_trl(SR, sr0, **KW)
        assert sh.psr(SR, sr0, t_star, **KW) == pytest.approx(.95)
    assert sh.power(.1, .1, 50, **KW) == pytest.approx(.05)
    for sr in (0., .05, .3):
        assert sh.sharpe_variance(sr, 100) == pytest.approx((1 + sr ** 2 / 2) / 100)


def test_min_trl_nan_when_not_above_null():
    assert math.isnan(sh.min_trl(.0, .0))
    assert math.isnan(sh.min_trl(-.1, 0.))


def test_t_for_power_is_smallest_integer():
    t = sh.t_for_power(0, .5 / math.sqrt(252), **KW)
    s1 = .5 / math.sqrt(252)
    assert sh.power(0, s1, t, **KW) >= .8 > sh.power(0, s1, t - 1, **KW)


def test_sample_moments_matches_scipy():
    x = np.random.default_rng(1).standard_t(5, size=500) * .01 + .0004
    m = sh.sample_moments(x)
    assert m["T"] == 500
    assert m["sr"] == pytest.approx(x.mean() / x.std(ddof=1))
    assert m["skew"] == pytest.approx(stats.skew(x, bias=False))
    assert m["kurt"] == pytest.approx(stats.kurtosis(x, fisher=False, bias=False))
    assert m["rho"] == pytest.approx(np.corrcoef(x[1:], x[:-1])[0, 1])


def _block_corr():
    C = np.zeros((12, 12))
    for b in range(3):
        C[4 * b:4 * b + 4, 4 * b:4 * b + 4] = .6
    np.fill_diagonal(C, 1.0)
    return C


def test_cluster_trials_recovers_blocks_and_is_reproducible():
    C = _block_corr()
    k1, l1 = sh.cluster_trials(C, max_k=6)
    k2, l2 = sh.cluster_trials(C, max_k=6)
    assert k1 == k2 == 3
    assert np.array_equal(l1, l2)
    assert len({tuple(sorted(set(l1[4 * b:4 * b + 4]))) for b in range(3)}) == 3
    assert all(len(set(l1[4 * b:4 * b + 4])) == 1 for b in range(3))


def test_silhouette_quality_zero_spread_is_never_selected():
    assert sh._silhouette_quality(np.full(10, .7)) == -np.inf
    assert sh._silhouette_quality(np.array([.5, .5, np.nan])) == -np.inf
    assert sh._silhouette_quality(np.array([.2, .4, .6])) == pytest.approx(.4 / np.std([.2, .4, .6]))


def test_cluster_trials_raises_when_no_k_qualifies(monkeypatch):
    monkeypatch.setattr(sh, "_silhouette_quality", lambda sil: -np.inf)
    with pytest.raises(ValueError, match="no k"):
        sh.cluster_trials(_block_corr(), max_k=6)


def test_effective_rank_bounds():
    assert sh.effective_rank(np.eye(5)) == pytest.approx(5.)
    assert sh.effective_rank(np.ones((5, 5))) == pytest.approx(1.)
    assert 1 < sh.effective_rank(_block_corr()) < 12


def test_sharpe_diff_power_and_years_roundtrip():
    assert sh.sharpe_diff_power(0, .2) == pytest.approx(.05)
    assert sh.sharpe_diff_power(0, .2, n_tests=7) == pytest.approx(.05 / 7)
    assert sh.sharpe_diff_power(.2, .1) > sh.sharpe_diff_power(.2, .1, n_tests=7)
    assert sh.sharpe_diff_power(.2, .1) == sh.sharpe_diff_power(-.2, .1)
    y = sh.years_for_power(.2, .22, years=17.21)
    assert sh.sharpe_diff_power(.2, .22 * math.sqrt(17.21 / y)) == pytest.approx(.8, abs=1e-9)
    assert y == pytest.approx(17.21 * (.22 * (1.96 + .8416) / .2) ** 2, rel=.02)


def test_monte_carlo_sr_variance_iid_normal():
    rng = np.random.default_rng(20261002)
    n, reps, sr = 250, 10_000, .05
    x = rng.normal(sr, 1.0, size=(reps, n))
    hat = x.mean(axis=1) / x.std(axis=1, ddof=1)
    assert hat.var(ddof=1) == pytest.approx(sh.sharpe_variance(sr, n), rel=.05)
