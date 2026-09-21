"""One-off reshape of the local EODHD archive into this repo's raw input.

Reads per-symbol daily CSVs from ~/Projects/research-data-eodhd (located via
its results/eodhd_archive_symbol_date_ranges.csv manifest), picks the
widest-coverage file per ticker in the locked universe, and writes long-format
Date,Ticker,AdjClose to data/raw/eod_prices.csv for scripts/build_panel.py.

Usage: python scripts/reshape_eodhd_archive.py
"""
from pathlib import Path
import csv
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maplab.contract import UNIVERSE

ARCHIVE = Path.home() / "Projects" / "research-data-eodhd"
MANIFEST = ARCHIVE / "results" / "eodhd_archive_symbol_date_ranges.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "eod_prices.csv"


def main():
    manifest_rows = list(csv.DictReader(open(MANIFEST)))

    frames = []
    for ticker in UNIVERSE:
        matches = [r for r in manifest_rows if r["symbol"].split(".")[0] == ticker]
        if not matches:
            raise SystemExit(f"{ticker} not found in {MANIFEST}")
        best = min(matches, key=lambda r: r["eod_start"])
        df = pd.read_csv(ARCHIVE / best["csv_path"], usecols=["date", best["price_col"]])
        df = df.rename(columns={"date": "Date", best["price_col"]: "AdjClose"})
        df["Ticker"] = ticker
        frames.append(df[["Date", "Ticker", "AdjClose"]])

    long = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  rows={len(long)}  tickers={long['Ticker'].nunique()}")


if __name__ == "__main__":
    main()
