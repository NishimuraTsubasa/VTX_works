import numpy as np
import pandas as pd

from stock_index_model.config import DEFAULT_CONFIG
from stock_index_model.risk import futures_risk


def test_risk_output():
    dates = pd.date_range("2020-01-31", periods=40, freq="ME")
    df = pd.DataFrame({
        "date": dates,
        "IDX1": np.random.default_rng(1).normal(0, 0.04, 40),
        "IDX2": np.random.default_rng(2).normal(0, 0.05, 40),
    })
    latest, history, corr = futures_risk(df, DEFAULT_CONFIG)
    assert set(latest["index_name"]) == {"IDX1", "IDX2"}
    assert corr.shape == (2, 2)
    assert not history.empty
