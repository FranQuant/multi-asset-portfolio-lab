"""Shared plotting style and family colors.

Import `apply_style()` at the top of every notebook so all charts match.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

FAMILY_COLORS = {
    "Return-based":  "#1f4e79",   # MV, MSR, BL
    "Risk-based":    "#2e7d32",   # GMV, MDP, RP, HRP
    "Signal-based":  "#c0392b",   # TSMOM, factor tilts
    "Benchmark":     "#555555",   # EW
    "Overlay":       "#8e44ad",   # VMP
}


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
