"""Smoke tests for the shared figures in maplab.plotting — synthetic data only."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import maplab as ml  # noqa: E402
from maplab.plotting import (  # noqa: E402
    METHOD_STYLES,
    _style_for,
    snapshot_map,
    group_stackplot,
    concentration_panel,
    cumulative_paired,
)

UNIVERSE = ml.UNIVERSE
N_ASSETS = len(UNIVERSE)


def make_sigma_mu(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.05, size=(N_ASSETS, N_ASSETS))
    Sigma = A @ A.T + np.diag(rng.uniform(0.01, 0.04, size=N_ASSETS))
    mu = rng.uniform(0.01, 0.10, size=N_ASSETS)
    return (pd.DataFrame(Sigma, index=UNIVERSE, columns=UNIVERSE),
            pd.Series(mu, index=UNIVERSE))


def make_results(seed=1):
    rng = np.random.default_rng(seed)
    rdates = pd.date_range("2020-01-31", periods=30, freq="ME")
    days = pd.bdate_range(rdates[0], rdates[-1])
    results = {}
    for name in ["A(S)", "B(S)", "EW"]:
        if name == "EW":
            w = np.full((len(rdates), N_ASSETS), 1.0 / N_ASSETS)
        else:
            w = rng.dirichlet(np.ones(N_ASSETS), size=len(rdates))
        results[name] = {
            "wlog": pd.DataFrame(w, index=rdates, columns=UNIVERSE),
            "net": pd.Series(rng.normal(0.0003, 0.008, size=len(days)), index=days),
        }
    split_ts = rdates[20]
    return results, split_ts


def test_style_for_strips_estimator_tag():
    assert _style_for("HRP(S)") == METHOD_STYLES["HRP"]
    assert _style_for("GMV(sample)") == METHOD_STYLES["GMV"]
    assert _style_for("IVP") == METHOD_STYLES["IVP"]


def test_snapshot_map_one_label_per_method():
    Sigma, mu = make_sigma_mu()
    n = N_ASSETS
    weights = {
        "GMV": np.eye(n)[0] * 0.5 + np.full(n, 0.5 / n),
        "ERC": np.full(n, 1.0 / n) * 0.9 + np.eye(n)[1] * 0.1,
        "EqualWeight": np.full(n, 1.0 / n),
    }
    for outside in (True, False):
        fig = snapshot_map(Sigma, mu, weights, title="t", legend_outside=outside, n_random=500)
        assert isinstance(fig, Figure)
        _, labels = fig.axes[0].get_legend_handles_labels()
        for name in weights:
            assert labels.count(name) == 1
        assert labels.count("long-only frontier") == 1
        plt.close(fig)


def test_group_stackplot_axes_count():
    results, split_ts = make_results()
    for names in (["A(S)", "B(S)", "EW"], ["A(S)"]):
        fig = group_stackplot(results, names, split_ts=split_ts)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == len(names)
        plt.close(fig)


def test_concentration_panel_axes_count():
    results, split_ts = make_results()
    for rows, figsize in [(("effN", "max_w", "highlight"), (9.5, 11)), (("effN",), (9.5, 4.5))]:
        fig = concentration_panel(results, ["A(S)", "B(S)"], rows=rows, split_ts=split_ts, figsize=figsize)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == len(rows)
        plt.close(fig)


def test_cumulative_paired_lines_with_and_without_beta():
    results, split_ts = make_results()
    pairs = [("A(S)", "EW"), ("B(S)", "EW")]
    paired_series = {(a, b): results[a]["net"] - results[b]["net"] for a, b in pairs}
    mkt = pd.Series(np.random.default_rng(2).normal(0.0004, 0.01, size=len(results["EW"]["net"])),
                    index=results["EW"]["net"].index)
    beta = {"A(S) - EW": 0.1, "B(S) - EW": -0.2}

    # + 2 for the split line and the zero line
    fig = cumulative_paired(paired_series, pairs, split_ts, beta=beta, mkt_excess=mkt)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2 * len(pairs) + 2
    plt.close(fig)

    fig = cumulative_paired(paired_series, pairs, split_ts)
    assert len(fig.axes[0].lines) == len(pairs) + 2
    plt.close(fig)

    fig0, axes = plt.subplots(2, 1)
    fig = cumulative_paired(paired_series, pairs, split_ts, ax=axes[1])
    assert fig is fig0
    assert len(axes[1].lines) == len(pairs) + 2
    assert len(axes[0].lines) == 0
    plt.close(fig0)
