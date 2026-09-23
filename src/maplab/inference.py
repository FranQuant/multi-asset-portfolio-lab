"""Inference helpers and pre-registration machinery shared by the notebooks.

The statistics (OLS, paired differences, CAPM / diff-regression, β
decomposition, verdict rules) are lifted verbatim from notebook 07 §6.4–§6.7
and its helper cell; `sharpe_se` from notebook 06. Registrations live in
<repo>/registrations/*.toml and are applied mechanically by `evaluate`.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import TRADING_DAYS
from .data import find_repo_root

HYPOTHESIS_KINDS = ("count", "null", "directional_t", "descriptive")


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS y ~ 1 + X via lstsq (no HAC). Returns coef/tstat Series indexed
    ["alpha", *X.columns], R^2, and n."""
    y_np = y.to_numpy()
    X_np = X.to_numpy()
    n = len(y_np)
    Xd = np.column_stack([np.ones(n), X_np])
    coefs, _, _, _ = np.linalg.lstsq(Xd, y_np, rcond=None)
    fitted = Xd @ coefs
    resid = y_np - fitted
    dof = n - Xd.shape[1]
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tstats = coefs / se
    ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    names = ["alpha"] + list(X.columns)
    return {
        "coef": pd.Series(coefs, index=names),
        "tstat": pd.Series(tstats, index=names),
        "r2": r2,
        "n": n,
    }


def sharpe_se(sr: float, years: float) -> float:
    """Sharpe standard error ~ sqrt((1 + SR^2/2) / T), T in years."""
    return float(np.sqrt((1 + sr ** 2 / 2) / years))


def paired_table(results: dict, pairs, split_ts, summary_table: pd.DataFrame | None = None,
                 trading_days: int = TRADING_DAYS) -> tuple[pd.DataFrame, dict]:
    """Paired daily net-return differences a − b per window (full / train /
    test): annualized mean, TE, t = mean/sd·√n. With `summary_table`, also the
    cost-drag difference and the gross-of-costs mean difference."""
    paired_rows = []
    paired_series = {}
    for a_name, b_name in pairs:
        net_a = results[a_name]["net"]
        net_b = results[b_name]["net"]
        idx = net_a.index.intersection(net_b.index)
        diff = net_a.loc[idx] - net_b.loc[idx]
        paired_series[(a_name, b_name)] = diff
        win_defs = {
            "full": diff.index >= diff.index.min(),
            "train (<= split)": diff.index <= split_ts,
            "test (> split)": diff.index > split_ts,
        }
        for window_name, sel in win_defs.items():
            d = diff.loc[sel]
            ann_mean = float(d.mean() * trading_days)
            te = float(d.std(ddof=1) * np.sqrt(trading_days))
            t_stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            row = {
                "pair": f"{a_name} - {b_name}", "window": window_name,
                "ann_mean_diff": ann_mean, "TE": te, "t": t_stat,
            }
            if summary_table is not None:
                cost_drag_a = float(summary_table.loc[(a_name, window_name), "cost_drag"])
                cost_drag_b = float(summary_table.loc[(b_name, window_name), "cost_drag"])
                cost_drag_diff = cost_drag_a - cost_drag_b
                gross_diff = ann_mean + cost_drag_diff
                row["cost_drag_diff"] = cost_drag_diff
                row["gross_of_costs_mean_diff"] = gross_diff
            paired_rows.append(row)

    return pd.DataFrame(paired_rows).set_index(["pair", "window"]), paired_series


def capm_table(results: dict, names, mkt_excess: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Single-strategy CAPM: (net − rf) ~ 1 + MKT excess, per strategy."""
    capm_rows = []
    for name in names:
        net = results[name]["net"]
        idx = net.index.intersection(rf.index).intersection(mkt_excess.index)
        strat_excess = net.loc[idx] - rf.loc[idx]
        mkt_aligned = mkt_excess.loc[idx]
        fit = ols(strat_excess, mkt_aligned.to_frame("MKT"))
        capm_rows.append({
            "strategy": name,
            "alpha_ann": float(fit["coef"]["alpha"] * TRADING_DAYS),
            "t_alpha": float(fit["tstat"]["alpha"]),
            "beta": float(fit["coef"]["MKT"]),
            "r2": fit["r2"], "n": fit["n"],
        })
    return pd.DataFrame(capm_rows).set_index("strategy")


def diff_regression(paired_series: dict, pairs, mkt_excess: pd.Series) -> pd.DataFrame:
    """Direct regression of each pair's daily diff series on MKT excess:
    Δα (annualized) with its own t, and Δβ."""
    diff_capm_rows = []
    for a_name, b_name in pairs:
        diff = paired_series[(a_name, b_name)]
        idx = diff.index.intersection(mkt_excess.index)
        fit = ols(diff.loc[idx], mkt_excess.loc[idx].to_frame("MKT"))
        diff_capm_rows.append({
            "pair": f"{a_name} - {b_name}",
            "alpha_diff_ann": float(fit["coef"]["alpha"] * TRADING_DAYS),
            "t_alpha_diff": float(fit["tstat"]["alpha"]),
            "beta_diff": float(fit["coef"]["MKT"]),
            "n": fit["n"],
        })
    return pd.DataFrame(diff_capm_rows).set_index("pair")


def beta_decomposition(capm: pd.DataFrame, paired: pd.DataFrame, diff_capm: pd.DataFrame, pairs,
                       results: dict, rf: pd.Series, mkt_excess: pd.Series) -> dict:
    """Full-window decomposition realized diff = Δα + Δβ·mean(MKT) per pair.
    Returns {"a - b": dict(d_alpha, beta_part, d_beta, mkt_ann_mean,
    realized_diff_full, decomposed, identity_check, t_alpha_diff, a_name, b_name)}."""
    decomp_results = {}
    for a_name, b_name in pairs:
        alpha_a = float(capm.loc[a_name, "alpha_ann"]); beta_a = float(capm.loc[a_name, "beta"])
        alpha_b = float(capm.loc[b_name, "alpha_ann"]); beta_b = float(capm.loc[b_name, "beta"])

        net_a = results[a_name]["net"]
        idx = net_a.index.intersection(rf.index).intersection(mkt_excess.index)
        mkt_ann_mean = float(mkt_excess.loc[idx].mean() * TRADING_DAYS)

        d_alpha = alpha_a - alpha_b
        d_beta = beta_a - beta_b
        beta_part = d_beta * mkt_ann_mean
        decomposed = d_alpha + beta_part

        realized_diff_full = float(paired.loc[(f"{a_name} - {b_name}", "full"), "ann_mean_diff"])
        identity_check = realized_diff_full - decomposed

        pair_key = f"{a_name} - {b_name}"
        t_alpha_diff = float(diff_capm.loc[pair_key, "t_alpha_diff"])

        decomp_results[pair_key] = dict(d_alpha=d_alpha, beta_part=beta_part, d_beta=d_beta,
                                        mkt_ann_mean=mkt_ann_mean, realized_diff_full=realized_diff_full,
                                        decomposed=decomposed, identity_check=identity_check,
                                        t_alpha_diff=t_alpha_diff, a_name=a_name, b_name=b_name)
    return decomp_results


def format_beta_decomposition(decomp: dict, capm: pd.DataFrame, diff_capm: pd.DataFrame) -> str:
    """The nb07 §6.5 β-decomposition printout, as one string."""
    lines = []
    for pair_key, r in decomp.items():
        a_name, b_name = r["a_name"], r["b_name"]
        d_alpha, d_beta, beta_part = r["d_alpha"], r["d_beta"], r["beta_part"]
        mkt_ann_mean, decomposed = r["mkt_ann_mean"], r["decomposed"]
        realized_diff_full, identity_check = r["realized_diff_full"], r["identity_check"]
        t_alpha_diff = r["t_alpha_diff"]

        lines.append(f"=== {pair_key} (full window) ===")
        lines.append(f"  t_alpha[{a_name}]={float(capm.loc[a_name, 't_alpha']):.2f}  t_alpha[{b_name}]={float(capm.loc[b_name, 't_alpha']):.2f}")
        lines.append(f"  d_alpha (ann)         = {d_alpha:.4%}   (direct diff-regression: {float(diff_capm.loc[pair_key, 'alpha_diff_ann']):.4%}, t={t_alpha_diff:.2f})")
        lines.append(f"  d_beta                = {d_beta:.4f}")
        lines.append(f"  ann mean MKT excess   = {mkt_ann_mean:.4%}")
        lines.append(f"  beta_part = d_beta * mkt_ann_mean = {beta_part:.4%}")
        lines.append(f"  d_alpha + beta_part   = {decomposed:.4%}")
        lines.append(f"  realized ann_mean_diff (full) = {realized_diff_full:.4%}")
        lines.append(f"  identity check (0 by OLS construction) = {identity_check:.4%}")
        lines.append("")
    return "\n".join(lines) + "\n" if lines else ""


def count_rule(holds_2022, frac_full, frac_test) -> str:
    if holds_2022 and frac_full > 0.5 and frac_test > 0.5:
        return "SUPPORTED"
    if holds_2022 and frac_full > 0.5:
        return "NOT ROBUST"
    return "NOT SUPPORTED"


def null_verdict(t_abs, narrow: float = 1.5, reject: float = 2.0) -> str:
    if t_abs < narrow:
        return "NULL HOLDS"
    if t_abs < reject:
        return "NULL HOLDS NARROWLY"
    return "NULL REJECTED"


def load_registration(name_or_path) -> dict:
    """Load and validate a registration TOML. A bare name ("nb07") resolves
    to <repo>/registrations/<name>.toml."""
    p = Path(name_or_path)
    if p.suffix != ".toml":
        p = find_repo_root(Path(__file__).resolve().parent) / "registrations" / f"{name_or_path}.toml"
    with open(p, "rb") as f:
        reg = tomllib.load(f)

    for key in ("notebook", "provenance", "text", "hypothesis"):
        if key not in reg:
            raise ValueError(f"{p.name}: missing required key {key!r}")
    if not isinstance(reg["hypothesis"], list):
        raise ValueError(f"{p.name}: 'hypothesis' must be a list of [[hypothesis]] tables")
    for i, h in enumerate(reg["hypothesis"]):
        for key in ("id", "label", "kind", "rule_text"):
            if key not in h:
                raise ValueError(f"{p.name}: hypothesis #{i} missing required key {key!r}")
        if h["kind"] not in HYPOTHESIS_KINDS:
            raise ValueError(f"{p.name}: hypothesis {h['id']!r} has unknown kind {h['kind']!r}")
    return reg


def render_registration(reg: dict):
    from IPython.display import Markdown
    return Markdown(reg["text"])


def evaluate(reg: dict, stats: dict) -> pd.DataFrame:
    """Apply each non-descriptive registered rule to stats[id]; one row per
    hypothesis in file order: item, statistic, rule, verdict."""
    rows = []
    for h in reg["hypothesis"]:
        kind = h["kind"]
        if kind == "descriptive":
            continue
        s = stats[h["id"]]
        if kind == "count":
            verdict = count_rule(s["holds_2022"], s["frac_full"], s["frac_test"])
        elif kind == "null":
            if "t" in s:
                t_abs = abs(s["t"])
            else:
                t_abs = max(abs(s["t_full"]), abs(s["t_test"]))
            verdict = null_verdict(t_abs)
        elif kind == "directional_t":
            direction = h.get("direction", s.get("direction"))
            if direction == "<":
                pass_t = s["t_full"] <= -2
                test_ok = s["mean_test"] < 0
            elif direction == ">":
                pass_t = s["t_full"] >= 2
                test_ok = s["mean_test"] > 0
            else:
                raise ValueError(f"evaluate: hypothesis {h['id']!r} has bad direction {direction!r}")
            beta_ok = (abs(s["beta_part"]) > abs(s["d_alpha"])) if h.get("beta_condition", False) else True
            if pass_t and test_ok and beta_ok:
                verdict = "SUPPORTED"
            elif pass_t and beta_ok and not test_ok and h.get("allow_not_robust", False):
                verdict = "NOT ROBUST"
            else:
                verdict = "NOT SUPPORTED"
        rows.append({"item": h["label"], "statistic": s["display"], "rule": h["rule_text"], "verdict": verdict})
    return pd.DataFrame(rows, columns=["item", "statistic", "rule", "verdict"])


def check_reproduction(summary_table: pd.DataFrame, mapping: dict,
                       windows=("full", "train (<= split)", "test (> split)")) -> None:
    """Reproduction gate: assert round(sharpe, 2) matches
    registrations/reproduction.toml for every label in `mapping`
    (label -> canonical entry), in every window."""
    p = find_repo_root(Path(__file__).resolve().parent) / "registrations" / "reproduction.toml"
    with open(p, "rb") as f:
        expected = tomllib.load(f)
    for key, window in zip(("full", "train", "test"), windows):
        for name, canonical in mapping.items():
            exp = expected[canonical][key]
            got = round(float(summary_table.loc[(name, window), "sharpe"]), 2)
            assert got == exp, f"REPRODUCTION GATE FAILED: {name} {key} Sharpe {got} != {exp}"
