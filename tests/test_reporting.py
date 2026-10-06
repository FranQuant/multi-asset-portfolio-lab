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


def test_returns_numeric_frame():
    out = show_table(_df(), pct=["w"], int_cols=["n"])
    assert pd.api.types.is_float_dtype(out["w"]) and pd.api.types.is_integer_dtype(out["n"])


def test_fixed_decimals_in_display():
    df = pd.DataFrame({"p": [1.0, 1.0], "se": [0.1, 0.25]})
    shown = show_table(df, dp=2, col_dp={"p": 3}).formatted()
    assert shown["p"].tolist() == ["1.000", "1.000"]
    assert shown["se"].tolist() == ["0.10", "0.25"]
    assert show_table(df, dp=2, col_dp={"p": 3})["p"].tolist() == [1.0, 1.0]
    with pytest.raises(KeyError):
        show_table(df, col_dp={"nope": 3})


def test_negative_zero_and_unicode_minus():
    df = pd.DataFrame({"x": [-0.001, -1.234, 0.001, np.nan]})
    shown = show_table(df, dp=2).formatted()
    assert shown["x"].tolist() == ["0.00", "−1.23", "0.00", "–"]
    assert "-" not in "".join(shown["x"])
    html = show_table(df, dp=2)._repr_html_()
    assert "−1.23" in html and "-0.00" not in html


def test_sign_cols():
    df = pd.DataFrame({"s": [1, -1, 1], "v": [0.5, 0.25, 0.75]})
    out = show_table(df, sign_cols=["s"])
    assert out.formatted()["s"].tolist() == ["+", "−", "+"]
    assert out["s"].tolist() == [1, -1, 1]


def test_label_replacement_display_only():
    df = pd.DataFrame({"a <= b": [1.0]}, index=["ERC(LW) - ERC(S)"])
    out = show_table(df)
    assert out.index[0] == "ERC(LW) - ERC(S)"
    f = out.formatted()
    assert f.index[0] == "ERC(LW) − ERC(S)" and f.columns[0] == "a ≤ b"
    f2 = show_table(pd.DataFrame({"a >= b": [1.0]})).formatted()
    assert f2.columns[0] == "a ≥ b"


def test_concat_keeps_each_columns_decimals():
    a = show_table(pd.DataFrame({"p": [1.0, 1.0], "z": [-0.001, 0.5]}), dp=2, col_dp={"p": 3})
    b = show_table(pd.DataFrame({"t": [-0.2, 1.0]}), dp=1)
    both = pd.concat([a, b], axis=1)
    assert isinstance(both, type(a))
    f = both.formatted()
    assert f["p"].tolist() == ["1.000", "1.000"]
    assert f["z"].tolist() == ["0.00", "0.50"]
    assert f["t"].tolist() == ["−0.2", "1.0"]
    html = both._repr_html_()
    assert "−0.2" in html and "-0.00" not in html and "-0.2" not in html
