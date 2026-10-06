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
