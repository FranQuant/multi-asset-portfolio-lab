"""Display helpers for notebook tables."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_NEG_ZERO = re.compile(r"-0(\.0+)?")
_LABEL_SUBS = ((" - ", " − "), ("<=", "≤"), (">=", "≥"))


def _label(x):
    if isinstance(x, str):
        for a, b in _LABEL_SUBS:
            x = x.replace(a, b)
    return x


def _relabel(idx):
    if isinstance(idx, pd.MultiIndex):
        return idx.set_levels([_relabel(lv) for lv in idx.levels])
    return idx.map(_label) if pd.api.types.is_string_dtype(idx) or idx.dtype == object else idx


def _fmt_number(v, d):
    if pd.isna(v):
        return "–"
    if not np.isfinite(v):
        return str(v)
    s = f"{v:.{d}f}"
    if _NEG_ZERO.fullmatch(s):
        s = s[1:]
    return s.replace("-", "−")


def _fmt_sign(v):
    if pd.isna(v):
        return "–"
    return "+" if v > 0 else "−" if v < 0 else "0"


class ShownTable(pd.DataFrame):
    """Numeric DataFrame (rounded) that renders with fixed decimals: `_fmt` maps a
    column to its number of decimals (int) or "sign". The numeric values stay
    usable; only the HTML / text display is formatted (U+2212 minus, no "-0.00",
    NaN as "–", " - " / "<=" / ">=" in labels as " − " / "≤" / "≥")."""

    _metadata = ["_fmt"]

    @property
    def _constructor(self):
        return ShownTable

    def __finalize__(self, other, method=None, **kwargs):
        res = super().__finalize__(other, method=method, **kwargs)
        if method == "concat":
            fmt = {}
            for o in getattr(other, "objs", ()):
                fmt.update(getattr(o, "_fmt", None) or {})
            res._fmt = fmt
        return res

    def formatted(self) -> pd.DataFrame:
        """String copy of the table as displayed."""
        fmt = getattr(self, "_fmt", None) or {}
        out = pd.DataFrame(self).astype(object)
        for c, d in fmt.items():
            if c not in out.columns or not is_numeric_dtype(self[c]) or is_bool_dtype(self[c]):
                continue
            f = _fmt_sign if d == "sign" else (lambda v, d=d: _fmt_number(v, d))
            out[c] = [f(v) for v in self[c]]
        out.index = _relabel(out.index)
        out.columns = _relabel(out.columns)
        return out

    def _repr_html_(self):
        return self.formatted()._repr_html_()

    def __repr__(self):
        return self.formatted().__repr__()


def show_table(df, pct=(), dp=2, pct_dp=1, int_cols=(), sign_cols=(), col_dp=None):
    """Return a copy of df for display: columns in `pct` ×100 rounded to pct_dp, columns in
    `int_cols` as int, other numeric columns rounded to dp; non-numeric columns untouched.
    A column in both `pct` and `int_cols` is scaled ×100, then rounded to an int.
    `col_dp` (column -> decimals) overrides the decimals of a column. `sign_cols` hold
    ±1 and display as "+" / "−". The returned frame is numeric; when displayed every
    numeric column prints with exactly its decimals."""
    pct, int_cols, sign_cols = list(pct), list(int_cols), list(sign_cols)
    col_dp = dict(col_dp or {})
    missing = [c for c in pct + int_cols + sign_cols + list(col_dp) if c not in df.columns]
    if missing:
        raise KeyError(f"show_table: unknown columns {missing}")
    out = ShownTable(df.copy())
    fmt = {}
    for c in out.columns:
        col = out[c]
        if not is_numeric_dtype(col) or is_bool_dtype(col):
            continue
        if c in sign_cols:
            fmt[c] = "sign"
            continue
        if c in pct:
            col = col * 100
        if c in int_cols:
            out[c] = col.round().astype(int)
            fmt[c] = 0
        else:
            d = col_dp.get(c, pct_dp if c in pct else dp)
            out[c] = col.round(d) + 0.0
            fmt[c] = d
    out._fmt = fmt
    return out
