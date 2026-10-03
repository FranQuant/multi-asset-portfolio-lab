"""maplab.reporting.show_table: scaling, rounding, passthrough, no mutation."""
import numpy as np
import pandas as pd
import pytest

import maplab as ml
from maplab.reporting import show_table


def _df():
    return pd.DataFrame({
        "w": [0.58123, 0.1],
        "n": [3.0, 2.0],
        "x": [1.23456, 2.0],
        "label": ["a", "b"],
    }, index=["r1", "r2"])


def test_exposed_as_module():
    assert ml.reporting.show_table is show_table


def test_pct_scales_and_rounds():
    out = show_table(_df(), pct=["w"], pct_dp=1)
    assert out["w"].tolist() == [58.1, 10.0]
    assert show_table(_df(), pct=["w"], pct_dp=2)["w"].tolist() == [58.12, 10.0]


def test_int_cols():
    out = show_table(_df(), int_cols=["n"])
    assert out["n"].tolist() == [3, 2]
    assert pd.api.types.is_integer_dtype(out["n"])


def test_pct_and_int_cols_scales_then_rounds_to_int():
    out = show_table(_df(), pct=["w"], int_cols=["w"])
    assert out["w"].tolist() == [58, 10]
    assert pd.api.types.is_integer_dtype(out["w"])


def test_other_numeric_use_dp():
    out = show_table(_df(), dp=2)
    assert out["x"].tolist() == [1.23, 2.0]
    assert show_table(_df(), dp=3)["x"].tolist() == [1.235, 2.0]
    assert out["w"].tolist() == [0.58, 0.1]


def test_non_numeric_untouched_and_index_kept():
    out = show_table(_df(), pct=["w"])
    assert out["label"].tolist() == ["a", "b"]
    assert out.index.tolist() == ["r1", "r2"]
    assert list(out.columns) == list(_df().columns)


def test_input_not_mutated():
    df = _df()
    before = df.copy()
    show_table(df, pct=["w"], int_cols=["n"])
    pd.testing.assert_frame_equal(df, before)


def test_unknown_column_raises():
    with pytest.raises(KeyError):
        show_table(_df(), pct=["nope"])
