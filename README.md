# multi-asset-portfolio-lab

A fair, reproducible comparison of **deterministic** multi-asset portfolio
construction methods on a single locked harness — one model per notebook.

**Thesis:** no single method dominates. Each family wins in the regime it is
built for; the durable edge is rotating method by regime, plus a
volatility-managed overlay that improves every base method. See
[`ROADMAP.md`](ROADMAP.md) for the full plan.

> Scope is deterministic families only. ML / DL / RL / LLM allocation are
> separate sequel repos by design.

## The harness

Every method is scored on the same terms — that is what makes the comparison
fair:

- **29 instruments**, 6 asset groups, 2003–2026
- daily prices → **monthly** rebalance, 252-day covariance lookback
- single train/test split at end-2022
- 10 bps one-way transaction costs, 3% fixed risk-free

The shared logic lives in [`src/maplab/`](src/maplab); each model's *weight
function* lives in its own notebook for teaching transparency. The comparison
notebook runs them all through the one backtest loop.

## Layout

```
src/maplab/        shared harness: contract, Panel, covariance, metrics, backtest
notebooks/         00 data contract, then one model per notebook, then comparison
scripts/           build_panel.py — the only place raw data enters
data/cache/        cached panels (gitignored; regenerate from scripts/)
docs/references/   internal grounding docs (gitignored)
```

## Quick start

```bash
pip install -e .
# drop a raw EODHD CSV at data/raw/eod_prices.csv, then:
python scripts/build_panel.py
# open notebooks/00_data_contract.ipynb
```

Notebook 00 runs on a **synthetic fallback** if no real cache is present, so the
repo is reviewable before data is wired in. Synthetic runs are clearly flagged
and never reported as results.

## Acknowledgements

Methodology builds on Yves Hilpisch's *Python and AI for Asset Management* (The
Python Quants / CPF program) and the empirical framing in the working paper
*A Practitioner's Map of Portfolio Construction* (J. F. Salazar, 2026).
Reference materials are retained internally for grounding only.
