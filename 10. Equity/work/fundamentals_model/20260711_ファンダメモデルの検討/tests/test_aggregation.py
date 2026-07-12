from copy import deepcopy

import pandas as pd

from stock_index_model.aggregation import aggregate_stock_scores
from stock_index_model.config import DEFAULT_CONFIG


def test_selected_universe_weighted_aggregation():
    scores = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31"] * 4),
        "ISIN": ["A", "B", "C", "D"],
        "stock_alpha": [0.1, 0.3, -0.2, 0.2],
        "stock_score": [0.2, 0.8, -0.8, 0.4],
        "confidence_score": [1.0, 2.0, -1.0, 1.0],
    })
    selection = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31"] * 4),
        "index_name": ["IDX"] * 4,
        "ISIN": ["A", "B", "C", "D"],
        "sector": ["S1", "S1", "S2", "S2"],
        "selection_weight": [0.50, 0.25, 0.15, 0.10],
        "target_sector_weight": [0.75, 0.75, 0.25, 0.25],
        "original_sector_weight": [0.75, 0.75, 0.25, 0.25],
        "is_actual_constituent": [True] * 4,
        "selection_key": ["X"] * 4,
        "target_count": [4] * 4,
        "rebalanced": [True] * 4,
    })
    cfg = deepcopy(DEFAULT_CONFIG)
    idx, sector = aggregate_stock_scores(scores, pd.DataFrame(), pd.DataFrame(), cfg, selection)
    expected = 0.5 * 0.1 + 0.25 * 0.3 + 0.15 * -0.2 + 0.10 * 0.2
    assert abs(idx.loc[0, "index_alpha"] - expected) < 1e-12
    assert abs(idx.loc[0, "index_breadth_count_based"] - 0.75) < 1e-12
    assert abs(idx.loc[0, "index_breadth_weighted"] - 0.85) < 1e-12
    assert sector["selected_count"].sum() == 4
