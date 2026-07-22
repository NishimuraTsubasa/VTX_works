from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.layer2_factor_return_weighting import (
    aggregate_raw_factor_scores,
    build_factor_return_history,
)
from stock_scoring_model.master import FactorMeta


def _config() -> dict:
    return {
        "columns": {"date": "date", "country": "country", "sector": "sector"},
        "layer2": {
            "factor_return_quantiles": 5,
            "factor_return_scope": "within_country",
            "minimum_stocks_per_factor_return_cell": 10,
            "country_factor_return_aggregation": "equal_country",
            "factor_return_lookback_periods": 12,
            "factor_return_minimum_periods": 3,
            "correlation_shrinkage": 0.3,
            "equal_weight_blend": 0.5,
            "maximum_factor_weight": 0.8,
            "weight_smoothing": 0.0,
            "effectiveness_filter_enabled": False,
        },
    }


def _sample() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FactorMeta]]:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    rows = []
    score_rows = []
    for date in dates:
        for country in ["JP", "US"]:
            for i in range(30):
                a = rng.normal()
                b = 0.7 * a + rng.normal(scale=0.7)
                ret = 0.01 * a + rng.normal(scale=0.05)
                rows.append({"date": date, "country": country, "sector": "S", "NextMonthReturn": ret})
                score_rows.append({"FA1": a, "FA2": b})
    return pd.DataFrame(rows), pd.DataFrame(score_rows), {
        "FA1": FactorMeta("FA1", "Value", 1, 1.0),
        "FA2": FactorMeta("FA2", "Value", 1, 1.0),
    }


def test_factor_return_history_and_weights() -> None:
    data, scores, metas = _sample()
    result = aggregate_raw_factor_scores(data, scores, metas, _config())
    assert not result.factor_return_history.empty
    assert not result.weight_history.empty
    sums = result.weight_history.groupby(["Date", "FactorGroup"])["Weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert result.factor_scores["Value"].std() > 0


def test_no_current_factor_return_used_in_weight() -> None:
    data, scores, metas = _sample()
    config = _config()
    history = build_factor_return_history(data, scores, metas, config)
    result = aggregate_raw_factor_scores(data, scores, metas, config)
    first_weight_date = result.weight_history["Date"].min()
    first_rows = result.weight_history[result.weight_history["Date"].eq(first_weight_date)]
    assert set(first_rows["Reason"]) == {"fallback_equal_weight_insufficient_history"}
    assert history["Date"].min() == first_weight_date
