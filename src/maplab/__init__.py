"""maplab — multi-asset portfolio lab shared harness.

Public API kept small on purpose: the data contract, the Panel gatekeeper,
covariance estimators, metrics, and the one backtest loop. Model logic lives
in the notebooks for teaching transparency.
"""
from . import contract
from .contract import (
    UNIVERSE, ASSET_GROUPS, EQUITY_SLEEVE, GROUP_OF,
    START, END, TRAIN_TEST_SPLIT, TRADING_DAYS, COV_LOOKBACK,
    REBALANCE, RF_ANNUAL, COST_BPS,
)
from .data import Panel, load_prices, to_log_returns, to_simple_returns, find_repo_root
from .covariance import sample_cov, ledoit_wolf_cov, oas_cov, COV_ESTIMATORS
from .metrics import (
    ann_return, ann_vol, ann_sharpe, max_drawdown, calmar, hit_rate,
    ann_turnover, summary,
)
from .backtest import backtest, rebalance_dates, first_eligible_rebalance
from .plotting import apply_style, FAMILY_COLORS

__all__ = [
    "contract", "UNIVERSE", "ASSET_GROUPS", "EQUITY_SLEEVE", "GROUP_OF",
    "START", "END", "TRAIN_TEST_SPLIT", "TRADING_DAYS", "COV_LOOKBACK",
    "REBALANCE", "RF_ANNUAL", "COST_BPS",
    "Panel", "load_prices", "to_log_returns", "to_simple_returns", "find_repo_root",
    "sample_cov", "ledoit_wolf_cov", "oas_cov", "COV_ESTIMATORS",
    "ann_return", "ann_vol", "ann_sharpe", "max_drawdown", "calmar",
    "hit_rate", "ann_turnover", "summary",
    "backtest", "rebalance_dates", "first_eligible_rebalance",
    "apply_style", "FAMILY_COLORS",
]
