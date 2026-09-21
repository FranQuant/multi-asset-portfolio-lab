"""Build the cached price panel from a raw EODHD-style CSV.

Usage: place a long CSV at data/raw/eod_prices.csv with columns
[Date, Ticker, AdjClose] (or wide with Date + one column per ticker), then run:

    python scripts/build_panel.py

Produces data/cache/prices.parquet restricted to the locked universe.
This script is the ONLY place raw data enters the repo.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maplab.contract import UNIVERSE, START, END

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eod_prices.csv"
OUT = ROOT / "data" / "cache" / "prices.parquet"


def main():
    if not RAW.exists():
        raise SystemExit(f"Place raw prices at {RAW} first.")
    df = pd.read_csv(RAW, parse_dates=["Date"])
    if {"Ticker", "AdjClose"}.issubset(df.columns):       # long format
        wide = df.pivot(index="Date", columns="Ticker", values="AdjClose")
    else:                                                  # already wide
        wide = df.set_index("Date")
    missing = [t for t in UNIVERSE if t not in wide.columns]
    if missing:
        raise SystemExit(f"Raw data missing tickers: {missing}")
    wide = wide.sort_index().loc[START:END, UNIVERSE]
    # Restrict to the US equity trading calendar (SPY's own listed dates).
    # FX (EURUSD) trades on weekends/holidays equities don't; keeping those
    # extra dates would union in non-trading days that ffill then papers
    # over with stale equity closes.
    trading_days = wide.index[wide["SPY"].notna()]
    wide = wide.loc[trading_days].ffill()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT)
    print(f"Wrote {OUT}  shape={wide.shape}  {wide.index.min().date()}→{wide.index.max().date()}")


if __name__ == "__main__":
    main()
