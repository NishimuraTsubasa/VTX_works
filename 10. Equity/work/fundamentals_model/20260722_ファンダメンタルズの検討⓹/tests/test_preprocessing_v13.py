from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.master import FactorMeta
from stock_scoring_model.preprocessing import build_factor_scores


def test_centered_percentile_direction() -> None:
    data = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-31")] * 5,
        "country": ["JP"] * 5,
        "sector": ["A"] * 5,
        "market_cap": [1, 2, 3, 4, 5],
        "FA": [1, 2, 3, 4, 5],
    })
    config = {
        "columns": {"date": "date", "country": "country", "sector": "sector", "market_cap": "market_cap"},
        "preprocessing": {"winsorize_lower": 0.01, "winsorize_upper": 0.99, "gaussian_clip": 3.0, "minimum_cross_section": 2},
    }
    scores = build_factor_scores(data, config, {"FA": FactorMeta("FA", "Value", -1, 1.0)}, winsorize=False, neutralize=False, rank_transform="centered_percentile")
    assert scores.loc[0, "FA"] > scores.loc[4, "FA"]
    assert np.isclose(scores["FA"].mean(), 0.0)
