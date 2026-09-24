"""maplab — multi-asset portfolio lab shared harness.

Public API kept small on purpose: the data contract, the Panel gatekeeper,
covariance estimators, metrics, and the one backtest loop. Model classes
(GMV, MaxSharpe, EqualWeight, ...) live in `models.py` so every notebook that
uses a given strategy name means exactly the same thing; notebooks are the
teaching layer that instantiates them and narrates the results.
"""
from . import contract
from .contract import (
    UNIVERSE, ASSET_GROUPS, EQUITY_SLEEVE, GROUP_OF,
    START, END, TRAIN_TEST_SPLIT, TRADING_DAYS, COV_LOOKBACK,
    REBALANCE, COST_BPS, RF_TICKER, FACTOR_TICKERS, PANEL_TICKERS,
)
from .data import (
    Panel, load_prices, load_rf_returns, load_factor_returns,
    to_log_returns, to_simple_returns, find_repo_root,
)
from .covariance import sample_cov, ledoit_wolf_cov, oas_cov, COV_ESTIMATORS
from .metrics import (
    ann_return, ann_vol, ann_sharpe, max_drawdown, calmar, hit_rate,
    ann_turnover, summary,
)
from .backtest import backtest, rebalance_dates, first_eligible_rebalance
from .models import (
    Strategy, GMV, MaxSharpe, BetaTargetMinVar, BlackLitterman, EqualWeight,
    gmv_closed_form, tangency_closed_form,
)
from .models import MostDiversified, mdp_closed_form
from .models import EqualRiskContribution, risk_contributions
from .models import HierarchicalRiskParity
from .models import InverseVol, InverseVariance
from .models import FixedWeight
from . import diagnostics, inference
from . import robust
from .plotting import apply_style, FAMILY_COLORS

__all__ = [
    "contract", "UNIVERSE", "ASSET_GROUPS", "EQUITY_SLEEVE", "GROUP_OF",
    "START", "END", "TRAIN_TEST_SPLIT", "TRADING_DAYS", "COV_LOOKBACK",
    "REBALANCE", "COST_BPS", "RF_TICKER", "FACTOR_TICKERS", "PANEL_TICKERS",
    "Panel", "load_prices", "load_rf_returns", "load_factor_returns",
    "to_log_returns", "to_simple_returns", "find_repo_root",
    "sample_cov", "ledoit_wolf_cov", "oas_cov", "COV_ESTIMATORS",
    "ann_return", "ann_vol", "ann_sharpe", "max_drawdown", "calmar",
    "hit_rate", "ann_turnover", "summary",
    "backtest", "rebalance_dates", "first_eligible_rebalance",
    "Strategy", "GMV", "MaxSharpe", "BetaTargetMinVar", "BlackLitterman", "EqualWeight",
    "gmv_closed_form", "tangency_closed_form",
    "apply_style", "FAMILY_COLORS",
    "MostDiversified", "mdp_closed_form",
    "EqualRiskContribution", "risk_contributions",
    "HierarchicalRiskParity",
    "InverseVol", "InverseVariance",
    "FixedWeight",
]
