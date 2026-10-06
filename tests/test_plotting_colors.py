"""cumulative_paired `colors` (cosmetics batch A)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from maplab.plotting import cumulative_paired


def test_cumulative_paired_colors():
    idx = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(0)
    series = {("A", "EW"): pd.Series(rng.normal(0, 1e-3, 60), index=idx),
              ("B", "EW"): pd.Series(rng.normal(0, 1e-3, 60), index=idx)}
    pairs = [("A", "EW"), ("B", "EW")]
    mkt = pd.Series(rng.normal(0, 1e-2, 60), index=idx)
    beta = {"A - EW": 0.1, "B - EW": -0.2}
    for kw in ({}, {"beta": beta, "mkt_excess": mkt}):
        fig = cumulative_paired(series, pairs, idx[30], colors={"B - EW": "#c0392b"}, **kw)
        lines = {ln.get_label(): ln for ln in fig.axes[0].lines}
        assert mcolors.to_hex(lines["B − EW"].get_color()) == "#c0392b"
        assert mcolors.to_hex(lines["A − EW"].get_color()) != "#c0392b"
        plt.close(fig)


def test_cumulative_paired_default_cycle_and_minus_labels():
    from maplab.plotting import PAIR_CYCLE
    idx = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(1)
    series = {("A", "EW"): pd.Series(rng.normal(0, 1e-3, 60), index=idx),
              ("B", "EW"): pd.Series(rng.normal(0, 1e-3, 60), index=idx)}
    pairs = [("A", "EW"), ("B", "EW")]
    mkt = pd.Series(rng.normal(0, 1e-2, 60), index=idx)
    for kw in ({}, {"beta": {"A - EW": 0.1, "B - EW": -0.2}, "mkt_excess": mkt}):
        fig = cumulative_paired(series, pairs, idx[30], **kw)
        lines = {ln.get_label(): ln for ln in fig.axes[0].lines}
        assert mcolors.to_hex(lines["A − EW"].get_color()) == PAIR_CYCLE[0]
        assert mcolors.to_hex(lines["B − EW"].get_color()) == PAIR_CYCLE[1]
        assert "train/test split" in lines
        plt.close(fig)


def test_cumulative_paired_no_colour_repeats_and_too_many_pairs():
    import pytest
    from maplab.plotting import PAIR_CYCLE
    assert len(set(PAIR_CYCLE)) == len(PAIR_CYCLE) == 8
    idx = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(2)
    n = len(PAIR_CYCLE)
    pairs = [(f"R{i}", "EW") for i in range(n)]
    series = {p: pd.Series(rng.normal(0, 1e-3, 60), index=idx) for p in pairs}
    mkt = pd.Series(rng.normal(0, 1e-2, 60), index=idx)
    beta = {f"{a} - {b}": 0.1 for a, b in pairs}
    for kw in ({}, {"beta": beta, "mkt_excess": mkt}):
        fig = cumulative_paired(series, pairs, idx[30], **kw)
        cols = [mcolors.to_hex(ln.get_color()) for ln in fig.axes[0].lines
                if ln.get_label().endswith("− EW")]
        assert len(cols) == n and len(set(cols)) == n
        plt.close(fig)
    pairs9 = pairs + [("R8", "EW")]
    series9 = {**series, ("R8", "EW"): pd.Series(rng.normal(0, 1e-3, 60), index=idx)}
    with pytest.raises(ValueError):
        cumulative_paired(series9, pairs9, idx[30])
