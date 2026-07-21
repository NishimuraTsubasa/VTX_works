from __future__ import annotations

from typing import Any

import pandas as pd

from .layer3_pooled import Layer3Prediction, rolling_pooled_prediction


def fit_partial_pooling_models(
    data: pd.DataFrame,
    X: pd.DataFrame,
    penalties,
    region: pd.Series,
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
    eligible_rows: pd.Series | None = None,
) -> Layer3Prediction:
    # 地域単位で地域共通係数と国別補正を同時推定する。
    return rolling_pooled_prediction(data, X, penalties, config, region, "hierarchical_partial_pooling", target_col, eligible_rows)
