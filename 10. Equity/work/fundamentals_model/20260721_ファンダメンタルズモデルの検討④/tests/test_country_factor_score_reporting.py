from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.country_factor_score_reporting import build_country_factor_score_diagnostics


def test_country_factor_score_history_and_top_factor():
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    rows = []
    for date in dates:
        for country in ["US", "JP"]:
            for idx in range(5):
                rows.append({"date": date, "country": country, "market_cap": idx + 1})
    data = pd.DataFrame(rows)
    factor_scores = pd.DataFrame({
        "Value": np.linspace(-1, 1, len(data)),
        "Momentum": np.linspace(1, -1, len(data)),
        "Quality": np.sin(np.arange(len(data)) / 5),
    })
    config = {
        "columns": {"date": "date", "country": "country", "market_cap": "market_cap"},
        "country_factor_score_diagnostics": {"weighting": "equal", "trailing_z_periods": 12, "minimum_z_periods": 6},
    }
    result = build_country_factor_score_diagnostics(data, factor_scores, config)
    assert not result.history.empty
    assert {"TrailingZScore", "CrossCountryZScore", "FactorRankWithinCountry"}.issubset(result.history.columns)
    assert not result.latest.empty
    assert not result.top_factor_history.empty
