"""Unit tests for maplab.inference — synthetic data only, no real cache."""
from __future__ import annotations

import itertools
import tomllib

import numpy as np
import pandas as pd
import pytest

from maplab.data import find_repo_root
from maplab.inference import (
    ols,
    paired_table,
    capm_table,
    diff_regression,
    beta_decomposition,
    format_beta_decomposition,
    count_rule,
    null_verdict,
    load_registration,
    evaluate,
    check_reproduction,
)


def test_ols_coef_matches_lstsq():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x1", "x2"])
    y = pd.Series(0.3 + X.to_numpy() @ np.array([1.5, -0.7]) + rng.normal(scale=0.5, size=n))
    fit = ols(y, X)
    Xd = np.column_stack([np.ones(n), X.to_numpy()])
    ref, _, _, _ = np.linalg.lstsq(Xd, y.to_numpy(), rcond=None)
    assert list(fit["coef"].index) == ["alpha", "x1", "x2"]
    assert np.max(np.abs(fit["coef"].to_numpy() - ref)) <= 1e-12
    assert fit["n"] == n


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    y = pd.Series(2.0 + 3.0 * x + rng.normal(scale=0.5, size=n))
    fit = ols(y, pd.DataFrame({"x": x}))
    assert abs(fit["coef"]["alpha"] - 2.0) <= 0.1
    assert abs(fit["coef"]["x"] - 3.0) <= 0.1


def test_beta_decomposition_identity():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=750)
    mkt = pd.Series(rng.normal(0.0004, 0.01, size=len(idx)), index=idx)
    rf = pd.Series(rng.normal(0.00008, 0.00001, size=len(idx)), index=idx)
    results = {
        "A": {"net": rf + 0.0001 + 0.6 * mkt + pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)},
        "B": {"net": rf + 0.0002 + 0.9 * mkt + pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)},
    }
    pairs = [("A", "B")]
    split_ts = idx[500]

    paired, paired_series = paired_table(results, pairs, split_ts)
    assert "cost_drag_diff" not in paired.columns
    capm = capm_table(results, ["A", "B"], mkt, rf)
    diff_capm = diff_regression(paired_series, pairs, mkt)
    decomp = beta_decomposition(capm, paired, diff_capm, pairs, results, rf, mkt)

    d = decomp["A - B"]
    realized = float(paired.loc[("A - B", "full"), "ann_mean_diff"])
    assert abs(realized - (d["d_alpha"] + d["beta_part"])) <= 1e-12

    text = format_beta_decomposition(decomp, capm, diff_capm)
    assert text.startswith("=== A - B (full window) ===\n")
    assert text.endswith("\n\n")


def test_count_rule_truth_table():
    for holds, full_ok, test_ok in itertools.product([True, False], repeat=3):
        v = count_rule(holds, 0.6 if full_ok else 0.4, 0.6 if test_ok else 0.4)
        if holds and full_ok and test_ok:
            assert v == "SUPPORTED"
        elif holds and full_ok:
            assert v == "NOT ROBUST"
        else:
            assert v == "NOT SUPPORTED"


def test_null_verdict_boundaries():
    assert null_verdict(1.4999) == "NULL HOLDS"
    assert null_verdict(1.5) == "NULL HOLDS NARROWLY"
    assert null_verdict(1.9999) == "NULL HOLDS NARROWLY"
    assert null_verdict(2.0) == "NULL REJECTED"


def test_load_registration_raises_on_missing_key(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        'notebook = "x"\nprovenance = "y"\n\n[[hypothesis]]\nid = "H1"\nlabel = "L"\nkind = "null"\nrule_text = "r"\n'
    )
    with pytest.raises(ValueError, match="'text'"):
        load_registration(p)

    p2 = tmp_path / "bad_hyp.toml"
    p2.write_text(
        'notebook = "x"\nprovenance = "y"\ntext = "t"\n\n[[hypothesis]]\nid = "H1"\nkind = "null"\nrule_text = "r"\n'
    )
    with pytest.raises(ValueError, match="'label'"):
        load_registration(p2)


# Statistics from the executed nb07 (§6.7 verdict table).
NB07_STATS = {
    "H1a": dict(display="h1a", holds_2022=True, frac_full=159 / 208, frac_test=14 / 40),
    "H2b_GMV": dict(display="h2b gmv", holds_2022=True, frac_full=1.0, frac_test=1.0),
    "H2b_MDP": dict(display="h2b mdp", holds_2022=True, frac_full=1.0, frac_test=1.0),
    "H2b_ERC": dict(display="h2b erc", holds_2022=True, frac_full=204 / 208, frac_test=1.0),
    "H3_GMV": dict(display="h3 gmv", holds_2022=True, frac_full=197 / 208, frac_test=36 / 40),
    "H3_MDP": dict(display="h3 mdp", holds_2022=True, frac_full=1.0, frac_test=1.0),
    "H3_ERC": dict(display="h3 erc", holds_2022=True, frac_full=1.0, frac_test=1.0),
    "H4_GMV": dict(display="t=1.53", t=1.5274),
    "H4_MDP": dict(display="t=0.54", t=0.5420),
    "H4_ERC": dict(display="t=1.59", t=1.5897),
    "H5": dict(display="h5", t_full=-1.9395, mean_test=-0.0357, beta_part=-0.045988, d_alpha=0.012989),
    "H6_LW": dict(display="lw", t_full=0.2660, t_test=-0.3090),
    "H6_IVP": dict(display="t=1.01", t=1.0068),
    "H6_pos": dict(display="t=-0.33", t=-0.3294),
    "H6_ward": dict(display="t=-0.51", t=-0.5051),
    "H7": dict(display="h7", t_full=-6.18, mean_test=-0.0788),
}

NB07_VERDICTS = [
    "NOT ROBUST",
    "SUPPORTED", "SUPPORTED", "SUPPORTED",
    "SUPPORTED", "SUPPORTED", "SUPPORTED",
    "NULL HOLDS NARROWLY", "NULL HOLDS", "NULL HOLDS NARROWLY",
    "NOT SUPPORTED",
    "NULL HOLDS", "NULL HOLDS", "NULL HOLDS", "NULL HOLDS",
    "SUPPORTED",
]


def test_evaluate_reproduces_nb07_verdicts():
    reg = load_registration("nb07")
    table = evaluate(reg, NB07_STATS)
    assert list(table.columns) == ["item", "statistic", "rule", "verdict"]
    assert list(table["verdict"]) == NB07_VERDICTS

    registered = [h for h in reg["hypothesis"] if h["kind"] != "descriptive"]
    assert list(table["item"]) == [h["label"] for h in registered]
    assert list(table["rule"]) == [h["rule_text"] for h in registered]
    assert list(table["statistic"]) == [NB07_STATS[h["id"]]["display"] for h in registered]


def test_evaluate_directional_t_cases(tmp_path):
    p = tmp_path / "dir.toml"
    p.write_text(
        'notebook = "x"\nprovenance = "y"\ntext = "t"\n\n[[hypothesis]]\nid = "H"\nlabel = "L"\n'
        'kind = "directional_t"\ndirection = "<"\nbeta_condition = true\nallow_not_robust = true\n'
        'rule_text = "r"\n'
    )
    reg = load_registration(p)
    cases = [
        (-2.5, -0.01, -0.05, 0.01, "SUPPORTED"),
        (-2.5, 0.01, -0.05, 0.01, "NOT ROBUST"),
        (-2.5, -0.01, -0.01, 0.05, "NOT SUPPORTED"),  # beta condition fails
        (-1.9, -0.01, -0.05, 0.01, "NOT SUPPORTED"),
    ]
    for t_full, mean_test, beta_part, d_alpha, expected in cases:
        stats = {"H": dict(display="d", t_full=t_full, mean_test=mean_test,
                           beta_part=beta_part, d_alpha=d_alpha)}
        assert evaluate(reg, stats)["verdict"].iloc[0] == expected


def _write_reg(tmp_path, body, name="r.toml"):
    p = tmp_path / name
    p.write_text('notebook = "x"\nprovenance = "y"\ntext = "t"\n\n[[hypothesis]]\nid = "H"\nlabel = "L"\n'
                 + body + 'rule_text = "r"\n')
    return p


def test_evaluate_count_no_2022_ignores_holds_2022(tmp_path):
    reg = load_registration(_write_reg(tmp_path, 'kind = "count"\nrule = "no_2022"\n'))
    for full_ok, test_ok in itertools.product([True, False], repeat=2):
        stats = {"H": dict(display="d", frac_full=0.6 if full_ok else 0.4, frac_test=0.6 if test_ok else 0.4)}
        v = evaluate(reg, stats)["verdict"].iloc[0]
        if full_ok and test_ok:
            assert v == "SUPPORTED"
        elif full_ok:
            assert v == "NOT ROBUST"
        else:
            assert v == "NOT SUPPORTED"
        stats["H"]["holds_2022"] = False  # ignored under no_2022
        assert evaluate(reg, stats)["verdict"].iloc[0] == v


def test_evaluate_directional_t_full_t_only(tmp_path):
    for direction, sign in (("<", -1), (">", 1)):
        reg = load_registration(_write_reg(
            tmp_path, f'kind = "directional_t"\ndirection = "{direction}"\nrule = "full_t_only"\n'))
        # mean_test absent from stats; a test-window mean of the wrong sign would not matter
        assert evaluate(reg, {"H": dict(display="d", t_full=sign * 2.5)})["verdict"].iloc[0] == "SUPPORTED"
        assert evaluate(reg, {"H": dict(display="d", t_full=sign * 1.9)})["verdict"].iloc[0] == "NOT SUPPORTED"
        wrong = {"H": dict(display="d", t_full=sign * 2.5, mean_test=-sign * 0.01)}
        assert evaluate(reg, wrong)["verdict"].iloc[0] == "SUPPORTED"


def test_evaluate_alpha_null_gates_verdict(tmp_path):
    reg = load_registration(_write_reg(
        tmp_path, 'kind = "directional_t"\ndirection = "<"\nbeta_condition = true\n'
                  'allow_not_robust = true\nalpha_null = true\n'))
    base = dict(display="d", t_full=-2.5, mean_test=-0.01, beta_part=-0.05, d_alpha=0.01)
    assert evaluate(reg, {"H": dict(base, t_alpha=1.99)})["verdict"].iloc[0] == "SUPPORTED"
    assert evaluate(reg, {"H": dict(base, t_alpha=2.0)})["verdict"].iloc[0] == "NOT SUPPORTED"
    assert evaluate(reg, {"H": dict(base, t_alpha=-2.0)})["verdict"].iloc[0] == "NOT SUPPORTED"
    not_robust = dict(base, mean_test=0.01)
    assert evaluate(reg, {"H": dict(not_robust, t_alpha=1.99)})["verdict"].iloc[0] == "NOT ROBUST"
    assert evaluate(reg, {"H": dict(not_robust, t_alpha=2.5)})["verdict"].iloc[0] == "NOT SUPPORTED"


def test_evaluate_applied_in_rule_column(tmp_path):
    stats = {"H": dict(display="d", t=0.5)}
    reg = load_registration(_write_reg(tmp_path, 'kind = "null"\napplied = "how"\n'))
    assert evaluate(reg, stats)["rule"].iloc[0] == "r [applied: how]"
    reg = load_registration(_write_reg(tmp_path, 'kind = "null"\n', name="r2.toml"))
    assert evaluate(reg, stats)["rule"].iloc[0] == "r"


def test_load_registration_rejects_bad_rule_fields(tmp_path):
    with pytest.raises(ValueError, match="rule"):
        load_registration(_write_reg(tmp_path, 'kind = "count"\nrule = "bogus"\n', name="a.toml"))
    with pytest.raises(ValueError, match="rule"):
        load_registration(_write_reg(tmp_path, 'kind = "null"\nrule = "standard"\n', name="b.toml"))
    with pytest.raises(ValueError, match="alpha_null"):
        load_registration(_write_reg(tmp_path, 'kind = "count"\nalpha_null = true\n', name="c.toml"))


@pytest.mark.parametrize("name", ["nb04", "nb05", "nb06"])
def test_load_registration_nb04_nb05_nb06(name):
    reg = load_registration(name)
    assert reg["hypothesis"]

def _synthetic_summary_table(expected, mapping, windows):
    rows = []
    for label, canonical in mapping.items():
        for key, window in zip(("full", "train", "test"), windows):
            rows.append({"strategy": label, "window": window, "sharpe": expected[canonical][key]})
    return pd.DataFrame(rows).set_index(["strategy", "window"])


def test_check_reproduction_passes_and_raises():
    with open(find_repo_root() / "registrations" / "reproduction.toml", "rb") as f:
        expected = tomllib.load(f)
    mapping = {
        "GMV(S)": "GMV_sample", "MaxSharpe(S)": "MaxSharpe_sample", "EW": "EW",
        "BL(S)": "BL_k0.1_sample", "MDP(S)": "MDP_sample", "ERC(S)": "ERC_sample",
    }
    windows = ("full", "train (<= split)", "test (> split)")
    st = _synthetic_summary_table(expected, mapping, windows)
    check_reproduction(st, mapping)

    st.loc[("EW", "test (> split)"), "sharpe"] += 0.01
    with pytest.raises(AssertionError, match=r"REPRODUCTION GATE FAILED: EW test Sharpe 0.92 != 0.91"):
        check_reproduction(st, mapping)


def test_evaluate_adjusted_null_bands(tmp_path):
    reg = load_registration(_write_reg(tmp_path, 'kind = "adjusted_null"\n'))
    cases = [
        (0.0499, None, "NULL REJECTED"),
        (0.01, True, "NULL REJECTED"),
        (0.01, False, "NULL REJECTED, NOT ROBUST"),
        (0.05, None, "NULL HOLDS NARROWLY"),
        (0.05, False, "NULL HOLDS NARROWLY"),
        (0.0999, None, "NULL HOLDS NARROWLY"),
        (0.10, None, "NULL HOLDS"),
        (0.80, False, "NULL HOLDS"),
    ]
    for p_adj, same_sign, expected in cases:
        s = dict(display="d", p_adj=p_adj)
        if same_sign is not None:
            s["same_sign_test"] = same_sign
        assert evaluate(reg, {"H": s})["verdict"].iloc[0] == expected


def test_load_registration_rejects_rule_on_adjusted_null(tmp_path):
    with pytest.raises(ValueError, match="rule"):
        load_registration(_write_reg(tmp_path, 'kind = "adjusted_null"\nrule = "standard"\n'))
