# multi-asset-portfolio-lab

Classical portfolio-construction methods, implemented from scratch and tested on 13 ETFs
(2009–2026) with pre-registered hypotheses, multiple-testing control and explicit power analysis.

## Findings

- **No method beats equal weight.** Seven optimised allocations were tested against EW on
  Sharpe ratio (Holm-adjusted) and 23 Phase 1 strategies on CAPM α (Romano–Wolf): 30 of 30
  null hypotheses hold ([nb08](notebooks/08_method_comparison.ipynb)).
- **The tests could not have told them apart.** Detecting a true Sharpe gap of 0.2 against EW
  with 80% power would need 34 to 220 years of daily data; 17 years are available
  ([nb11](notebooks/11_sharpe_inference.ipynb)). The null verdicts mean "indistinguishable",
  not "equal".
- **Every method beats cash.** Over the full window the probabilistic Sharpe ratio against
  zero is at least 0.996 for each of the nine runs in the table below, and the best of the 23-run search survives deflation
  ([nb11](notebooks/11_sharpe_inference.ipynb)).
- **A volatility overlay controls drawdowns, not Sharpe.** It cuts volatility to 73–87% of each
  base method and the 2020 drawdown to 42–57%, with no significant Sharpe change: 16 of 16 nulls
  hold ([nb10](notebooks/10_vol_overlay.ipynb)).
- **The search was smaller than it looks.** The 23 Phase 1 runs behave like about three
  independent strategies: risk-based, balanced/EW-like and maximum-Sharpe
  ([nb11](notebooks/11_sharpe_inference.ipynb)).

## Data

Daily adjusted prices for 13 ETFs, one per risk premium, 2008-01-02 to 2026-04-30. BIL
(T-bills) is the risk-free rate and is not allocatable.

| Sleeve | ETFs |
|---|---|
| Equity | SPY, EFA, EEM |
| Rates | IEF, TLT |
| Inflation-linked | TIP |
| Credit | LQD, HYG, EMB |
| Real assets | VNQ, DBC, GLD |
| FX | UUP |

The price data come from a licensed vendor archive and are not distributed.
`scripts/build_panel.py` builds the panel from a CSV with columns Date, Ticker, AdjClose
(see notebook 00).

## Methods

| Notebook | Method | Idea |
|---|---|---|
| [00](notebooks/00_data_contract.ipynb) | Data contract | Panel construction, return conventions, checks |
| [01](notebooks/01_mean_variance.ipynb) | Mean-variance (GMV, MaxSharpe) | Minimum variance; maximum Sharpe on sample moments |
| [02](notebooks/02_capm_beta_target.ipynb) | Beta-targeted min-variance | Minimum variance subject to a market-beta floor |
| [03](notebooks/03_black_litterman.ipynb) | Black–Litterman | EW-implied prior updated with momentum views |
| [04](notebooks/04_covariance_shrinkage.ipynb) | Covariance shrinkage | Ledoit–Wolf and OAS estimators across methods |
| [05](notebooks/05_most_diversified.ipynb) | Most diversified portfolio (MDP) | Maximise the diversification ratio |
| [06](notebooks/06_risk_parity_erc.ipynb) | Equal risk contribution (ERC) | Equalise each asset's risk contribution |
| [07](notebooks/07_hierarchical_risk_parity.ipynb) | Hierarchical risk parity (HRP) | Cluster, then allocate by inverse variance down the tree |
| [08](notebooks/08_method_comparison.ipynb) | Comparison | Sharpe tests vs EW, CAPM α with Romano–Wolf |
| [09](notebooks/09_robustness.ipynb) | Robustness | Lookback, ex-ante volatility bias, rank stability |
| [10](notebooks/10_vol_overlay.ipynb) | Volatility overlay | Scale exposure to a volatility target, rest in T-bills |
| [11](notebooks/11_sharpe_inference.ipynb) | Sharpe inference | PSR, minimum track record, power, deflated Sharpe ratio |

## How methods are judged

- **Backtest.** Long-only, month-end rebalancing, 252-day estimation window, 10 bp per unit
  of turnover, drifted weights between rebalances. Log returns for estimation, simple
  returns for compounding. Scored from 2009-02-02; train window to 2022-12-31, test window
  2023-01-03 to 2026-04-30.
- **Benchmarks.** Equal weight (EW) and 60/40 (SPY/IEF).
- **Inference.** Hypotheses are written to `registrations/*.toml` and rendered in each
  notebook; notebooks 04–10 registered them before computing the tested statistics, notebook
  11's registration is descriptive and was written after an exploratory run. Notebooks gate
  their inputs against `registrations/reproduction.toml`. Newey–West standard errors,
  stationary bootstrap, Holm and Romano–Wolf adjustments.
- **Reference checks.** The GMV, maximum-Sharpe, MDP and HRP solvers are tested against
  PyPortfolioOpt 1.6 (cvxpy/Clarabel) in `tests/test_reference.py` (`pip install -e '.[ref]'`).

## Results

Annualised, net of costs. Return, volatility and drawdown are of the net return; Sharpe is in
excess of T-bills (BIL). Core configuration of each method (sample covariance).

| Method | Return | Volatility | Sharpe (full) | Sharpe (test) | Max drawdown |
|---|---|---|---|---|---|
| EW | 7.0% | 7.9% | 0.74 | 0.91 | −17.8% |
| GMV | 3.2% | 3.2% | 0.62 | 0.76 | −6.7% |
| MaxSharpe | 5.8% | 6.3% | 0.72 | 1.13 | −10.8% |
| Beta-targeted MV (0.3) | 6.2% | 5.8% | 0.86 | 1.13 | −13.9% |
| Black–Litterman (0.1) | 5.7% | 6.9% | 0.64 | 0.70 | −15.7% |
| MDP | 3.8% | 3.5% | 0.76 | 0.98 | −6.4% |
| ERC | 4.2% | 3.9% | 0.76 | 0.94 | −11.0% |
| HRP | 3.7% | 3.9% | 0.64 | 0.82 | −9.3% |
| 60/40 | 10.4% | 10.1% | 0.91 | 1.00 | −21.3% |

Full window 2009-02-02 to 2026-04-30 (17.2 years); test window 3.3 years.

## What the data can and cannot tell us

- **Test window.** At their full-window Sharpe ratios, seven of the nine runs need more than
  the 3.3-year test window (2.9 to 6.6 years) to show a positive Sharpe ratio. Test-window
  results are consistency checks, not evidence.
- **Power.** Against a true annual Sharpe ratio of 0.5, 17 years of data give 66–73% power.
  Against EW, power to detect a gap of 0.2 is 12–51% before multiple-testing adjustment.
- **Scope.** One universe, one 17-year sample, flat transaction costs, monthly rebalancing,
  long-only. The results describe these methods on this data, not portfolio construction in
  general.

## Repository

| Path | Contents |
|---|---|
| `src/maplab/` | Data, backtest engine, models, covariance estimators, overlay, inference (`robust`, `sharpe`) |
| `notebooks/` | One notebook per method or study, with `_nbXX_helpers.py` sidecars |
| `registrations/` | Hypothesis registrations (TOML), rendered into the notebooks |
| `tests/` | Unit and reproduction tests |
| `scripts/` | Data-panel construction |

## Reproduce

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]" jupyter
    pytest
    cd notebooks
    jupyter nbconvert --to notebook --execute --inplace 00_data_contract.ipynb   # then 01 … 11 in order

Requires the price panel described under Data. Python 3.11 or later (developed on 3.12).
