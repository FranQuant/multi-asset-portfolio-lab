"""Unit tests for maplab.diagnostics — synthetic data only, no real cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.contract import COV_LOOKBACK
from maplab.data import Panel
from maplab.diagnostics import (
    effective_n,
    half_l1,
    port_vol,
    diversification_ratio,
    n_uncorrelated,
    max_rc_share,
    long_only_frontier,
    diversification_table,
)
from maplab.models import GMV, _erc_long_only, _min_variance_long_only


UNIVERSE = ml.UNIVERSE
N_ASSETS = len(UNIVERSE)


def make_panel(n_days=300, mean=0.0005, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(loc=mean, scale=vol, size=(n_days, N_ASSETS))
    log_returns = pd.DataFrame(data, index=idx, columns=UNIVERSE)
    return Panel({"returns": log_returns}), log_returns


def sample_sigma(seed=0):
    panel, log_returns = make_panel(seed=seed)
    asof = log_returns.index[COV_LOOKBACK]
    mu, Sigma = GMV()._estimate(panel, asof)
    return mu, Sigma, asof


def test_effective_n_ew_and_unit():
    assert abs(effective_n(np.full(13, 1.0 / 13)) - 13.0) <= 1e-12
    e = np.zeros(13)
    e[4] = 1.0
    assert effective_n(e) == 1.0


def test_half_l1_identical_and_unit_vectors():
    w = np.array([0.2, 0.3, 0.5])
    assert half_l1(w, w) == 0.0
    assert half_l1(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])) == 1.0


def test_half_l1_raises_on_misaligned_series():
    a = pd.Series([0.5, 0.5], index=["A", "B"])
    b = pd.Series([0.5, 0.5], index=["B", "A"])
    with pytest.raises(ValueError):
        half_l1(a, b)


def test_port_vol_and_dr_two_asset_hand_values():
    # vols 0.2 / 0.3, rho 0.1 -> cov 0.006; w = (0.5, 0.5)
    Sigma = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = np.array([0.5, 0.5])
    var = 0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.006
    assert abs(port_vol(w, Sigma) - np.sqrt(var)) <= 1e-15
    assert abs(port_vol(w, Sigma.to_numpy()) - np.sqrt(var)) <= 1e-15
    assert abs(diversification_ratio(w, Sigma) - 0.25 / np.sqrt(var)) <= 1e-14


def test_n_uncorrelated_diagonal_inverse_vol_and_single_asset():
    rng = np.random.default_rng(3)
    vols = rng.uniform(0.05, 0.30, size=N_ASSETS)
    Sigma = np.diag(vols ** 2)
    w_iv = (1.0 / vols) / np.sum(1.0 / vols)
    assert abs(n_uncorrelated(w_iv, Sigma) - N_ASSETS) <= 1e-12

    e = np.zeros(N_ASSETS)
    e[2] = 1.0
    assert abs(n_uncorrelated(e, Sigma) - 1.0) <= 1e-12


def test_max_rc_share_erc_is_one_over_n():
    _, Sigma, asof = sample_sigma(seed=5)
    w, _ = _erc_long_only(Sigma, "test", asof)
    assert abs(max_rc_share(w, Sigma) - 1.0 / N_ASSETS) <= 1e-11


def test_long_only_frontier_endpoints():
    mu, Sigma, _ = sample_sigma(seed=6)
    vols, rets = long_only_frontier(Sigma, mu)
    assert len(vols) == len(rets) > 0
    # the frontier starts at the lowest-mu asset alone and ends at the highest-mu asset alone
    i_min, i_max = int(np.argmin(mu.to_numpy())), int(np.argmax(mu.to_numpy()))
    sigma = np.sqrt(np.diag(Sigma.to_numpy()))
    assert abs(rets[0] - mu.min()) <= 1e-6
    assert abs(vols[0] - sigma[i_min]) <= 1e-6
    assert abs(rets[-1] - mu.max()) <= 1e-6
    assert abs(vols[-1] - sigma[i_max]) <= 1e-6
    # no frontier point beats the global minimum-variance portfolio
    gmv_vol = port_vol(_min_variance_long_only(Sigma), Sigma)
    assert vols.min() >= gmv_vol - 1e-9


def test_diversification_table_columns():
    _, Sigma, asof = sample_sigma(seed=7)
    w_erc, _ = _erc_long_only(Sigma, "test", asof)
    weights = {"EW": np.full(N_ASSETS, 1.0 / N_ASSETS), "ERC": w_erc}
    tbl = diversification_table(weights, Sigma)
    assert list(tbl.columns) == ["w_UUP", "effN", "n_uncorr", "max_RC_share", "DR", "exante_vol"]
    assert list(tbl.index) == ["EW", "ERC"]
    iU = list(Sigma.columns).index("UUP")
    assert tbl.loc["ERC", "w_UUP"] == w_erc[iU]
