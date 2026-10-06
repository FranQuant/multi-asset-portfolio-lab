"""Smoke tests for the shared figures in maplab.plotting — synthetic data only."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import maplab as ml  # noqa: E402
from maplab.plotting import (  # noqa: E402
    METHOD_STYLES,
    _style_for,
    snapshot_map,
    group_stackplot,
    concentration_panel,
    capital_vs_risk,
    cumulative_paired,
    wealth_drawdown,
    forest_plot,
    pair_matrix,
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
    rng_t = np.random.default_rng(seed + 100)  # separate stream: existing draws unchanged
    for name in ["A(S)", "B(S)", "EW"]:
        if name == "EW":
            w = np.full((len(rdates), N_ASSETS), 1.0 / N_ASSETS)
        else:
            w = rng.dirichlet(np.ones(N_ASSETS), size=len(rdates))
        results[name] = {
            "wlog": pd.DataFrame(w, index=rdates, columns=UNIVERSE),
            "net": pd.Series(rng.normal(0.0003, 0.008, size=len(days)), index=days),
            "diag": {"turnover_per_rebalance": pd.Series(
                rng_t.uniform(0.0, 0.2, size=len(rdates)),
                index=rdates, name="one_way_turnover")},
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
            assert labels.count(METHOD_STYLES.get(name, {}).get("label", name)) == 1
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


def test_concentration_panel_turnover_row():
    results, split_ts = make_results()
    rows = ("effN", "max_w", "highlight", "turnover")
    fig = concentration_panel(results, ["A(S)", "B(S)"], rows=rows, split_ts=split_ts)
    assert len(fig.axes) == 4
    ax = fig.axes[3]
    line = next(ln for ln in ax.lines if ln.get_label() == "A(S)")
    y = np.asarray(line.get_ydata(), dtype=float)
    first = y[~np.isnan(y)][0]
    tlog = results["A(S)"]["diag"]["turnover_per_rebalance"]
    assert np.isclose(first, 100 * tlog.iloc[1:13].sum(), rtol=0, atol=1e-12)
    assert "EW" in [ln.get_label() for ln in ax.lines]
    plt.close(fig)


def test_capital_vs_risk_bars_and_alignment():
    rng = np.random.default_rng(3)
    tickers = ["A", "B", "UUP", "D", "E"]
    M = rng.normal(size=(5, 5))
    Sigma = pd.DataFrame(M @ M.T + 0.1 * np.eye(5), index=tickers, columns=tickers)
    weights = {f"m{k}": pd.Series(rng.dirichlet(np.ones(5)), index=tickers) for k in range(3)}
    weights["m1"] = weights["m1"].iloc[::-1]  # different order, same index set
    fig = capital_vs_risk(weights, Sigma, title="t")
    assert len(fig.axes) == 3
    n = len(tickers)
    for ax in fig.axes:
        assert len(ax.patches) == 2 * n
        heights = np.array([pch.get_height() for pch in ax.patches])
        assert np.isclose(heights[:n].sum(), 1.0, rtol=0, atol=1e-12)
        assert np.isclose(heights[n:].sum(), 1.0, rtol=0, atol=1e-12)
    plt.close(fig)

    bad = dict(weights, m0=weights["m0"].drop("D"))
    with pytest.raises(ValueError):
        capital_vs_risk(bad, Sigma)
    plt.close("all")


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


def test_wealth_drawdown_smoke():
    results, split_ts = make_results()
    net = {name: r["net"] for name, r in results.items()}
    rf = pd.Series(0.00008, index=results["EW"]["net"].index)
    fig = wealth_drawdown(net, rf, split_ts)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert "BIL (rf)" in labels and all(name in labels for name in net)
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_forest_plot_smoke():
    df = pd.DataFrame({
        "label": ["A(S)", "B(S)", "EW"] * 2,
        "window": ["full"] * 3 + ["test"] * 3,
        "d": [0.1, -0.2, 0.0, 0.3, -0.1, 0.05],
    })
    df["lo"], df["hi"] = df["d"] - 0.2, df["d"] + 0.2
    fig = forest_plot(df, "label", "d", "lo", "hi", group="window", ref=0.0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["A(S)", "B(S)", "EW"]
    plt.close(fig)
    fig = forest_plot(df[df["window"] == "full"], "label", "d", "lo", "hi")
    assert len(fig.axes) == 1
    plt.close(fig)


def test_pair_matrix_smoke():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3, 3))
    M = A - A.T
    labels = ["A(S)", "B(S)", "EW"]
    fig = pair_matrix(M, labels, "t")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # heatmap + colorbar
    assert len(fig.axes[0].texts) == 6  # diagonal blanked
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == labels
    plt.close(fig)


def test_wealth_drawdown_plain_tick_labels():
    results, split_ts = make_results()
    net = {name: r["net"] for name, r in results.items()}
    rf = pd.Series(0.00008, index=results["EW"]["net"].index)
    fig = wealth_drawdown(net, rf, split_ts)
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert texts and all(t for t in texts)
    assert not any("×" in t or "10^" in t or "$" in t for t in texts)
    plt.close(fig)


def test_forest_plot_first_group_on_top():
    df = pd.DataFrame({
        "label": ["A(S)", "B(S)", "EW"] * 2,
        "window": ["full"] * 3 + ["test"] * 3,
        "d": [0.1, -0.2, 0.0, 0.3, -0.1, 0.05],
    })
    df["lo"], df["hi"] = df["d"] - 0.2, df["d"] + 0.2
    fig = forest_plot(df, "label", "d", "lo", "hi", group="window", ref=0.0)
    ax = fig.axes[0]
    y_first = np.asarray(ax.containers[0].lines[0].get_ydata())
    y_last = np.asarray(ax.containers[-1].lines[0].get_ydata())
    assert np.all(y_first > y_last)
    plt.close(fig)


def test_pair_matrix_text_white_at_vmax():
    M = np.array([[0.0, 2.0, 0.5], [-2.0, 0.0, 1.0], [-0.5, -1.0, 0.0]])
    fig = pair_matrix(M, ["A(S)", "B(S)", "EW"], "t")
    colors = {t.get_text(): t.get_color() for t in fig.axes[0].texts}
    assert colors["2.00"] == "white" and colors["-2.00"] == "white"
    assert colors["0.50"] == "black"
    plt.close(fig)


def test_group_stackplot_labels_and_percent_axis():
    from maplab.plotting import GROUP_LABELS
    results, split_ts = make_results()
    fig = group_stackplot(results, ["A(S)", "B(S)"], split_ts=split_ts)
    for ax in fig.axes:
        assert ax.get_ylabel() == "weight (%)"
        assert ax.yaxis.get_major_formatter().format_pct(0.5, 1.0) == "50%"
        assert " — " in ax.get_title() and "--" not in ax.get_title()
    legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "inflation-linked" in legend_labels and "real assets" in legend_labels
    assert not any("_" in t for t in legend_labels)
    assert GROUP_LABELS["inflation_linked"] == "inflation-linked"
    plt.close(fig)


def test_run_colors_highlight_reserved_for_uup():
    from maplab.plotting import FAMILY_COLORS, HIGHLIGHT, METHOD_COLORS, RUN_COLORS, run_color
    assert HIGHLIGHT not in RUN_COLORS.values()
    assert HIGHLIGHT not in FAMILY_COLORS.values()
    assert all(METHOD_COLORS[k] == v for k, v in RUN_COLORS.items())
    assert run_color("BL(0.1,S)") == run_color("BL(0.2)") == RUN_COLORS["BL"]
    assert run_color("EqualWeight") == RUN_COLORS["EW"]
    with pytest.raises(KeyError):
        run_color("Nope(S)")
