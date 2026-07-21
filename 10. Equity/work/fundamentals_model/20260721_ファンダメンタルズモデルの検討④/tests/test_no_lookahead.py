import pandas as pd

from stock_scoring_model.feature_engineering import add_forward_return


def test_forward_return_alignment():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
        "ISIN": ["A", "A", "A"],
        "stock_return": [0.01, 0.02, 0.03],
    })
    cfg = {
        "columns": {"date": "date", "isin": "ISIN", "stock_return": "stock_return"},
        "target": {"stock_return_alignment": "contemporaneous_to_forward", "stock_horizon_periods": 1},
    }
    out = add_forward_return(data, cfg)
    assert out.loc[0, "NextMonthReturn"] == 0.02
    assert pd.isna(out.loc[2, "NextMonthReturn"])
