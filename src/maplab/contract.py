"""Data contract: the single, locked source of truth for the whole repo.

Every notebook imports the universe, dates, and conventions from here so the
comparison is fair by construction. Change a number here and it changes
everywhere — that is the point.
"""
from __future__ import annotations

# ── Universe: 13 instruments across 6 asset groups, one per risk premium ───
# Grouping is explicit so models can declare which sub-universe they are valid
# on (e.g. equity-factor models should not allocate over commodities/FX).
ASSET_GROUPS: dict[str, list[str]] = {
    "equity":           ["SPY", "EFA", "EEM"],
    "rates":            ["IEF", "TLT"],
    "inflation_linked": ["TIP"],
    "credit":           ["LQD", "HYG", "EMB"],
    "real_assets":      ["VNQ", "DBC", "GLD"],
    "fx":               ["UUP"],
}

UNIVERSE: list[str] = [t for group in ASSET_GROUPS.values() for t in group]

# Sub-universe an equity cross-sectional model (factor tilts, CAPM) may use.
# For now this is just the equity group; notebook 02 will redefine its own
# cross-section as needed.
EQUITY_SLEEVE: list[str] = ASSET_GROUPS["equity"]

# BIL (1-3m T-bills) is the risk-free numeraire, deliberately NOT in UNIVERSE,
# so no model allocates to cash. It rides along in the cached panel only so a
# later step can build the RF series from it.
RF_TICKER = "BIL"
PANEL_TICKERS: list[str] = UNIVERSE + [RF_TICKER]

# ── Time & rebalancing conventions ──────────────────────────────────────────
START = "2008-01-01"
END   = "2026-04-30"
TRAIN_TEST_SPLIT = "2022-12-31"   # single split; 2023+ is the held-out window

TRADING_DAYS = 252
COV_LOOKBACK = 252                # days used to estimate the covariance matrix
REBALANCE    = "ME"               # month-end rebalancing (pandas offset alias)
# RF is only the Sharpe hurdle, not an earned return; a fixed 3% penalised
# low-vol methods (GMV ~1.8% ann. return) pre-2022 and flipped rankings.
# RF=0 puts every method on the same bar, is reproducible, and needs no
# external series.
RF_ANNUAL    = 0.0                # risk-free rate for Sharpe / tangency
COST_BPS     = 10.0               # one-way transaction cost, basis points

# ── Returns convention ──────────────────────────────────────────────────────
# Estimate moments (mu, Sigma) on LOG returns (additive, well-behaved), but
# COMPOUND the backtest on SIMPLE returns — portfolio return is the weighted
# sum of simple returns, never log returns. Mixing these up is a silent bug.
# The harness enforces: estimation -> log, compounding -> simple.

# ── Constraint regimes (per-method attribute, not a global rule) ────────────
# Each method declares its natural regime; the harness does NOT force one.
LONG_ONLY = "long_only"        # weights >= 0, sum to 1 (HRP, RP, MDP, GMV-lo, EW)
LONG_SHORT = "long_short"      # weights may be negative (unconstrained tangency, BL)
LEVERAGE_OK = "leverage_ok"    # |weights| may sum > 1 (some overlays)

# ── Missing-data policy ─────────────────────────────────────────────────────
# Daily EOD: forward-fill prices (standard). No back-fill (would be look-ahead).
# Documented here so the loader's behavior is a stated policy, not an accident.
FILL_POLICY = "ffill"

# ── Warm-up ─────────────────────────────────────────────────────────────────
# No strategy can score until COV_LOOKBACK days of history exist. The harness
# starts every strategy's scored window on the SAME first-eligible rebalance so
# methods with different internal lookbacks still start on a common date.
WARMUP_DAYS = COV_LOOKBACK

# ── Cost model (honesty note) ───────────────────────────────────────────────
# Flat symmetric COST_BPS one-way on realized turnover. This is a SIMPLIFICATION:
# real costs are asset-specific, size-dependent, and asymmetric. Adequate for a
# liquid ETF/large-cap universe; not to be mistaken for precision.

# ── Survivorship / continuity policy ────────────────────────────────────────
# Instruments must be present for the FULL window. One instrument per distinct
# risk premium, all live before 2008-01-01 (latest starts: EMB 2007-12-19,
# HYG 2007-04, BIL 2007-05). Single stocks, sector ETFs, and near-duplicates
# were removed: FXI (subset of EEM), USO (subset of DBC), SLV (near-duplicate
# of GLD), AGG (a blend of the other fixed-income sleeves), SHY (replaced by
# BIL as the RF numeraire), EURUSD (replaced by UUP). BTC-USD is deliberately
# excluded: it does not exist back to 2008, so including it would inject a
# survivorship/availability bias into every cross-sectional comparison.
EXCLUDED_FOR_CONTINUITY: list[str] = ["BTC-USD"]

GROUP_OF: dict[str, str] = {
    t: g for g, tickers in ASSET_GROUPS.items() for t in tickers
}
