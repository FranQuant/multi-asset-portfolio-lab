"""N1: test-window turnover keeps its first rebalance; full/train drop the initial build."""
import numpy as np
import pandas as pd
import maplab as ml


def _turnover():
    idx = pd.date_range("2020-01-31", periods=25, freq="ME")
    return pd.Series([0.5] + [0.1] * 24, index=idx)


def test_ann_turnover_initial_build_flag():
    t = _turnover()
    years = (t.index[-1] - t.index[0]).days / 365.25
    assert np.isclose(ml.ann_turnover(t), 0.1 * 24 / years)
    assert ml.ann_turnover(t) == ml.ann_turnover(t, initial_build=True)
    t_test = t.iloc[12:]
    y2 = (t_test.index[-1] - t_test.index[0]).days / 365.25
    assert np.isclose(ml.ann_turnover(t_test, initial_build=False), 0.1 * 12 / y2)


def test_summary_detects_initial_build_from_dates():
    t = _turnover()
    days = pd.bdate_range("2020-02-03", "2022-01-31")
    r = pd.Series(0.0001, index=days)
    full = ml.summary(r, t, rf=0.0)["ann_turnover"]
    assert full == ml.ann_turnover(t, initial_build=True)
    split = pd.Timestamp("2020-12-31")
    r_test, t_test = r.loc[r.index > split], t.loc[t.index > split]
    assert t_test.index[0] > r_test.index[0]
    test = ml.summary(r_test, t_test, rf=0.0)["ann_turnover"]
    assert test == ml.ann_turnover(t_test, initial_build=False)
    assert test != ml.ann_turnover(t_test, initial_build=True) or np.isclose(t_test.iloc[0], t_test.iloc[1:].mean())
