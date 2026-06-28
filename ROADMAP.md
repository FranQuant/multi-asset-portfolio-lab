# ROADMAP — multi-asset-portfolio-lab

**Thesis.** No single deterministic allocation method dominates. Each family
wins in the regime it is built for; the durable edge is *rotating method by
regime*, plus a volatility-managed overlay that helps everything. This repo
proves that on one locked harness, one model per notebook.

**Scope of this repo:** deterministic families only. ML, DL, RL, and LLM-driven
allocation are deliberately split into **separate sequel repos** so each stays
self-contained and readable.

---

## Architecture (decided)

**Hybrid.** The *harness* lives in `src/maplab/` and is identical for every
method (data contract, Panel no-look-ahead gatekeeper, covariance estimators,
metrics, the one backtest loop, plotting). The *model logic* — the weight
computation that teaches the idea — lives **inside each notebook**, visible, not
imported from a black box. The comparison notebook imports each notebook's
`weight_fn` and runs them all through the single shared backtest. This gives
teaching transparency *and* fairness at once.

---

## The spine

### Part 0 — Foundation
- **00 — Data Contract & Panel** ✅ locks universe/dates/freq/split; builds the
  no-look-ahead Panel; survivorship + universe-applicability policy.

### Part 1 — One model per notebook
*Return-based (must forecast returns — inherit the forecast error):*
- 01 — Mean-Variance (GMV + tangency, the Markowitz core)
- 02 — Maximum-Sharpe / MSR (+ constrained variants)
- 03 — Black-Litterman (equilibrium prior + views)

*Risk-based (lean only on vol & correlation — sidestep the return forecast):*
- 04 — Global Minimum Variance in depth + covariance shrinkage
- 05 — Most-Diversified Portfolio (MDP)
- 06 — Risk Parity (RP)
- 07 — Hierarchical Risk Parity (HRP)

*Signal-based (comparators, equity-sleeve aware):*
- 08 — Time-Series Momentum (TSMOM)
- 09 — Factor tilts (FF-style, **equity sleeve only**)

### Part 2 — The comparison (the payoff)
- 10 — Monte Carlo cloud + efficient frontier, all methods together
- 11 — Allocation fingerprints (what each method actually holds)
- 12 — Master metrics table + **bootstrap CIs** (show the rankings overlap)
- 13 — Turnover & rebalancing cost analysis

### Part 3 — Dynamics & overlays (the thesis lands)
- 14 — Regime detection (rule-based, deterministic engine)
- 15 — Regime-conditional Sharpe + the SWITCH rule (train-only → held-out)
- 16 — Volatility-Managed Portfolio (VMP) overlay

### Audit
- 99 — Reproducibility guard: every headline reproduces from cache.

---

## Principles
1. **One contract, imported everywhere** — fairness by construction.
2. **No look-ahead by design** — the Panel refuses future data.
3. **Universe applicability is explicit** — factor models use the equity sleeve.
4. **Honest about uncertainty** — bootstrap CIs and regime swings, not a
   leaderboard.
5. **Reproducible from cache** — raw data enters once, via `scripts/`.

## Locked harness conventions (decided at foundation, never repatched)
- **Returns:** log for estimation (μ, Σ), **simple for compounding** the backtest.
- **Turnover:** drifted-weight tracking — cost charged on the real trade
  (new target − drifted actual), one-way, annualized exactly from the calendar.
- **Costs:** flat 10 bps one-way, symmetric (a stated simplification).
- **Constraints:** per-method attribute (natural weights + declared
  long-only / long-short), no forced global constraint.
- **Warm-up:** common scoring start after the 252-day covariance lookback.
- **Missing data:** forward-fill prices only (no back-fill).

## Open / incremental
- Drop real EODHD panel via `scripts/build_panel.py` (replaces synthetic).
- Internal reference reports land in `docs/references/` (gitignored) to ground
  each family's modeling choices.
