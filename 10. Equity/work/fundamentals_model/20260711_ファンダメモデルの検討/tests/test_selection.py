import pandas as pd

from stock_index_model.selection import allocate_sector_counts


def test_sector_allocation_matches_target_and_capacity():
    weights = pd.Series({"Tech": 0.50, "Fin": 0.30, "Ind": 0.20})
    available = pd.Series({"Tech": 20, "Fin": 20, "Ind": 2})
    result = allocate_sector_counts(10, weights, available)
    assert sum(result.values()) == 10
    assert result["Ind"] <= 2
    assert all(v >= 0 for v in result.values())
