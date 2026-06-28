"""Covariance estimators — the one modeling choice shared across MV methods.

Kept here (not in notebooks) so that 'GMV(LW)' means exactly the same thing in
every notebook. Each returns an ANNUALIZED covariance matrix as a DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

from .contract import TRADING_DAYS


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov = returns.cov().to_numpy() * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    X = returns.dropna().to_numpy()
    cov = LedoitWolf().fit(X).covariance_ * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def oas_cov(returns: pd.DataFrame) -> pd.DataFrame:
    X = returns.dropna().to_numpy()
    cov = OAS().fit(X).covariance_ * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


COV_ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "oas": oas_cov,
}
