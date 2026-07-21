from __future__ import annotations

from typing import Any

import pandas as pd

from .layer3_cross_sectional import rolling_cross_sectional_coefficient_average
from .layer3_pooled import Layer3Prediction, rolling_pooled_prediction


def fit_regional_models(
    data: pd.DataFrame,
    X: pd.DataFrame,
    penalties,
    region: pd.Series,
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
    eligible_rows: pd.Series | None = None,
) -> Layer3Prediction:
    if config["layer3"].get("training_mode", "rolling_pooled") == "cross_sectional_coefficient_average":
        return rolling_cross_sectional_coefficient_average(data, X, penalties, config, region, "regional_pooling", target_col, eligible_rows)
    return rolling_pooled_prediction(data, X, penalties, config, region, "regional_pooling", target_col, eligible_rows)
