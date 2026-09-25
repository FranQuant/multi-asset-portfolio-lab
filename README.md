# multi-asset-portfolio-lab

Research repo comparing deterministic portfolio-construction methods on a
shared backtest harness. It provides an overview of portfolio construction 
frameworks, their historical performance across different market regimes, 
and practical considerations for implementation.

 Work in progress.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The notebooks read a cached price panel, `data/cache/prices.parquet` (gitignored),
built by `scripts/build_panel.py` from a local EODHD archive. Without it,
`00_data_contract.ipynb` falls back to a synthetic panel for plumbing checks only;
notebooks 01–09 require the real cache.
