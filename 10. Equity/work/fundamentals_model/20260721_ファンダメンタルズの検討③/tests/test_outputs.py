from __future__ import annotations

import pandas as pd

from stock_scoring_model.scenarios import _stock_frame


def test_stock_frame_has_required_compact_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"] * 10),
            "ISIN": [f"X{i}" for i in range(10)],
            "currency": ["USD"] * 10,
            "market_cap": range(1, 11),
            "NextMonthReturn": [0.01] * 10,
        }
    )
    cfg = {
        "columns": {
            "date": "date", "isin": "ISIN", "currency": "currency", "market_cap": "market_cap"
        },
        "evaluation": {"quintiles": 5},
    }
    out = _stock_frame(data, pd.Series(range(10), dtype=float), cfg)
    required = {"Date", "ISIN", "Currency", "MarketCap", "Prediction", "NextMonthReturn", "TotalScore", "Quintile"}
    assert required.issubset(out.columns)
