"""Display helpers for notebook tables."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def show_table(df, pct=(), dp=2, pct_dp=1, int_cols=()):
    """Return a copy of df for display: columns in `pct` ×100 rounded to pct_dp, columns in
    `int_cols` as int, other numeric columns rounded to dp; non-numeric columns untouched.
    A column in both `pct` and `int_cols` is scaled ×100, then rounded to an int."""
    pct, int_cols = list(pct), list(int_cols)
    missing = [c for c in pct + int_cols if c not in df.columns]
    if missing:
        raise KeyError(f"show_table: unknown columns {missing}")
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if not is_numeric_dtype(col) or is_bool_dtype(col):
            continue
        if c in pct:
            col = col * 100
        if c in int_cols:
            out[c] = col.round().astype(int)
        elif c in pct:
            out[c] = col.round(pct_dp)
        else:
            out[c] = col.round(dp)
    return out
