"""Notebook-02-only code, moved verbatim from the notebook cells.

Each function returns DataFrames/Figures/values; notebook 02 prints,
asserts and shows them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

import maplab as ml
from maplab import GMV, MaxSharpe, BetaTargetMinVar


def factor_betas(returns_df: pd.DataFrame, fac_df: pd.DataFrame) -> pd.DataFrame:
    # OLS slopes (intercept discarded) of each column of returns_df on
    # fac_df's columns, one regression per asset. Returns (assets x factors).
    rows = {}
    for col in returns_df.columns:
        fit = ml.inference.ols(returns_df[col], fac_df)
        rows[col] = fit["coef"].drop("alpha")
    return pd.DataFrame(rows).T


def sml_figure(capm_table):
    """Security market line: empirical fit vs the CAPM line through (0,0)
    and (1, SPY). Returns (fig, slope, intercept, spy_excess_ann)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(capm_table["beta"], capm_table["ann_excess_return"], s=40, color="#2e7d32", zorder=3)
    for tkr, row in capm_table.iterrows():
        ax.annotate(tkr, (row["beta"], row["ann_excess_return"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")

    slope, intercept = np.polyfit(capm_table["beta"], capm_table["ann_excess_return"], 1)
    beta_grid = np.linspace(min(0, capm_table["beta"].min()) - 0.1, capm_table["beta"].max() * 1.1, 20)
    ax.plot(beta_grid, intercept + slope * beta_grid, color="#1f4e79", ls="--", lw=1.5,
            label=f"empirical SML fit (slope={slope:.3f}, intercept={intercept:.3f})")

    spy_excess_ann = float(capm_table.loc["SPY", "ann_excess_return"])
    ax.plot(beta_grid, spy_excess_ann * beta_grid, color="#b33", ls=":", lw=1.5,
            label=f"CAPM line through (0,0) & (1, SPY={spy_excess_ann:.2%})")

    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel(r"$\beta$ (full window)")
    ax.set_ylabel("annualized excess return")
    ax.set_title("Security market line")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, slope, intercept, spy_excess_ann


def rolling_beta(asset_excess: pd.Series, mkt_excess: pd.Series, window: int = 252) -> pd.Series:
    cov = asset_excess.rolling(window).cov(mkt_excess)
    var = mkt_excess.rolling(window).var()
    return cov / var


def rolling_beta_figure(excess, MKT):
    """Rolling 252d beta vs MKT for TLT/UUP/GLD/HYG."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tkr in ["TLT", "UUP", "GLD", "HYG"]:
        rb = rolling_beta(excess[tkr], MKT)
        ax.plot(rb.index, rb.to_numpy(), label=tkr, lw=1.2)
    ax.axhline(1.0, color="#999999", ls=":", lw=1, label="SPY (=1 by construction)")
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_ylabel(r"rolling 252d $\beta$ vs MKT")
    ax.set_title("Rolling beta is not stable")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    return fig


def last_trading_day_on_or_before(idx: pd.DatetimeIndex, date_str: str) -> pd.Timestamp:
    ts = pd.Timestamp(date_str)
    return idx[idx <= ts].max()


def window_before(idx: pd.DatetimeIndex, asof: pd.Timestamp, lookback: int) -> pd.DatetimeIndex:
    past = idx[idx < asof]
    return past[-lookback:]


def unconstrained_tangency(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(Sigma, mu)
    return raw / raw.sum()


def longonly_maxsharpe(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    n = len(mu)
    i0 = int(np.argmax(mu))
    y0 = np.zeros(n)
    y0[i0] = 1.0 / mu[i0]
    cons = [{"type": "eq", "fun": lambda y: mu @ y - 1.0, "jac": lambda y: mu}]
    bounds = [(0.0, None)] * n
    res = opt.minimize(lambda y: y @ Sigma @ y, y0, jac=lambda y: 2 * Sigma @ y, method="SLSQP",
                        bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
    y = res.x
    return y / y.sum()


def capm_collapse(log_returns, excess, asof_dates):
    """CAPM-collapse check per as-of date: A1 (CAPM) and A2 (MKT/TERM/CREDIT)
    tangency / long-only max-Sharpe weights, plus the algebra check
    Σx⁻¹Bλ = W(W'ΣxW)⁻¹λ. Returns [{asof, designs, rel1, rel2}]."""
    TICKERS = ml.UNIVERSE
    N = len(TICKERS)
    TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}

    out = []
    for date_str in asof_dates:
        asof = last_trading_day_on_or_before(log_returns.index, date_str)
        win = window_before(log_returns.index, asof, ml.COV_LOOKBACK)
        X_win = excess.loc[win, TICKERS]
        Sigma_x = X_win.cov().to_numpy() * ml.TRADING_DAYS

        fac_x = pd.DataFrame({
            "MKT": X_win["SPY"],
            "TERM": X_win["TLT"],
            "CREDIT": X_win["HYG"] - X_win["IEF"],
        })

        W_A1 = np.zeros((N, 1))
        W_A1[TICKER_IDX["SPY"], 0] = 1.0
        W_A2 = np.zeros((N, 3))
        W_A2[TICKER_IDX["SPY"], 0] = 1.0
        W_A2[TICKER_IDX["TLT"], 1] = 1.0
        W_A2[TICKER_IDX["HYG"], 2] = 1.0
        W_A2[TICKER_IDX["IEF"], 2] = -1.0

        lam_A1 = np.array([0.05])
        lam_A2 = np.array([0.05, 0.015, 0.01])

        B_capm = factor_betas(X_win[TICKERS], fac_x[["MKT"]]).to_numpy()
        B_a2 = factor_betas(X_win[TICKERS], fac_x[["MKT", "TERM", "CREDIT"]]).to_numpy()

        mu_A1 = B_capm @ lam_A1
        mu_A2 = B_a2 @ lam_A2

        designs = {
            "A1 CAPM tangency (unconstrained)": unconstrained_tangency(Sigma_x, mu_A1),
            "A1 CAPM long-only max-Sharpe": longonly_maxsharpe(Sigma_x, mu_A1),
            "A2 MKT/TERM/CREDIT tangency (unconstrained)": unconstrained_tangency(Sigma_x, mu_A2),
            "A2 MKT/TERM/CREDIT long-only max-Sharpe": longonly_maxsharpe(Sigma_x, mu_A2),
        }

        lhs1 = np.linalg.solve(Sigma_x, mu_A1)
        rhs1 = W_A1 @ np.linalg.solve(W_A1.T @ Sigma_x @ W_A1, lam_A1)
        rel1 = np.linalg.norm(lhs1 - rhs1) / np.linalg.norm(lhs1)

        lhs2 = np.linalg.solve(Sigma_x, mu_A2)
        rhs2 = W_A2 @ np.linalg.solve(W_A2.T @ Sigma_x @ W_A2, lam_A2)
        rel2 = np.linalg.norm(lhs2 - rhs2) / np.linalg.norm(lhs2)

        out.append({"asof": asof, "designs": designs, "rel1": rel1, "rel2": rel2})
    return out


def minvar_with_linear_constraints(Sigma: np.ndarray, A: np.ndarray, b: np.ndarray):
    n = Sigma.shape[0]
    x0 = np.full(n, 1.0 / n)
    cons = []
    for i in range(A.shape[0]):
        a_row = A[i]
        target = b[i]
        cons.append({"type": "eq", "fun": (lambda w, a=a_row, t=target: a @ w - t),
                      "jac": (lambda w, a=a_row: a)})
    bounds = [(0.0, 1.0)] * n
    res = opt.minimize(lambda w: w @ Sigma @ w, x0, jac=lambda w: 2 * Sigma @ w, method="SLSQP",
                        bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
    return res.x, res.success, res.message


def design_d3(log_returns, excess, factors3, asof_dates):
    """Design D3 per as-of date: long-only min-variance subject to 1'w = 1 and
    B'w = [0.5, 0.1, 0.1] on MKT/SIZE/VALUE betas. Returns [(asof, w, ok, msg)]."""
    TICKERS = ml.UNIVERSE
    N = len(TICKERS)

    out = []
    for date_str in asof_dates:
        asof = last_trading_day_on_or_before(log_returns.index, date_str)
        win = window_before(log_returns.index, asof, ml.COV_LOOKBACK)
        rets_win = log_returns.loc[win, TICKERS]
        Sigma_raw = rets_win.cov().to_numpy() * ml.TRADING_DAYS

        fac3_win = factors3.reindex(win).dropna()
        excess3_win = excess.loc[fac3_win.index, TICKERS]
        B3 = factor_betas(excess3_win, fac3_win).to_numpy()

        A = np.vstack([np.ones(N), B3.T])
        b = np.array([1.0, 0.5, 0.1, 0.1])
        w, ok, msg = minvar_with_linear_constraints(Sigma_raw, A, b)
        out.append((asof, w, ok, msg))
    return out


def group_weights_figure(bmv03_group, split_ts):
    """BetaMinVar(β ≥ 0.3) target weights by asset group (single panel)."""
    group_order = list(ml.ASSET_GROUPS.keys())
    group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_order)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(bmv03_group.index, bmv03_group.T.to_numpy(), labels=group_order, colors=group_colors)
    ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title(r"BetaMinVar($\beta \geq 0.3$) — target weights by asset group")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    return fig


def ex_ante_beta_figure(ex_ante_beta, split_ts):
    """Ex-ante portfolio beta of BetaMinVar(β ≥ 0.3) at each rebalance."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(ex_ante_beta.index, ex_ante_beta.to_numpy(), color="#1f4e79", lw=1.2)
    ax.axhline(0.3, color="#b33", ls="--", lw=1, label=r"$\beta$ target = 0.3")
    ax.axvline(split_ts, color="black", ls=":", lw=1)
    ax.set_ylabel(r"ex-ante portfolio $\beta$")
    ax.set_title(r"BetaMinVar($\beta \geq 0.3$) — achieved $\beta$ at each rebalance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


color_map = {
    "GMV": ml.FAMILY_COLORS[GMV.family],
    "MaxSharpe": ml.FAMILY_COLORS[MaxSharpe.family],
    "EqualWeight": ml.FAMILY_COLORS["Benchmark"],
    "BetaMinVar(0.3)": ml.FAMILY_COLORS[BetaTargetMinVar.family],
    "BetaMinVar(0.5)": ml.FAMILY_COLORS[BetaTargetMinVar.family],
}
ls_map = {"BetaMinVar(0.5)": "--", "β-match(0.3)": ":", "β-match(0.5)": "-."}


def wealth_figure(results, split_ts):
    """Strategy comparison — cumulative wealth (log scale)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, r in results.items():
        net = r["net"]
        wealth = (1.0 + net).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=name,
                color=color_map.get(name, "#888888"), lw=1.4, ls=ls_map.get(name, "-"))
    ax.axvline(split_ts, color="#888888", ls=":", lw=1, label="train/test split")
    ax.set_yscale("log")
    ax.set_ylabel("cumulative wealth (log scale, costs as noted)")
    ax.set_title("Strategy comparison — cumulative wealth")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    return fig


def drawdown_figure(results, split_ts):
    """Strategy comparison — drawdown. Returns (fig, dd_dates) with each
    strategy's max drawdown and its peak/trough dates."""
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    dd_dates = []
    for name, r in results.items():
        net = r["net"]
        wealth = (1.0 + net).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        ax.plot(dd.index, dd.to_numpy() * 100, label=name,
                color=color_map.get(name, "#888888"), lw=1.2, ls=ls_map.get(name, "-"))
        trough = dd.idxmin()
        peak = wealth.loc[:trough].idxmax()
        dd_dates.append({"strategy": name, "max_dd": float(dd.min()), "peak": peak.date(), "trough": trough.date()})
    ax.axvline(split_ts, color="#888888", ls=":", lw=1)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Strategy comparison — drawdown")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    plt.tight_layout()
    return fig, dd_dates
