import numpy as np
import pandas as pd

from stock_scoring_model.config import DEFAULT_CONFIG
from stock_scoring_model.preprocessing import preprocess_panel


def test_winsor_and_rank_transform():
    dates = pd.to_datetime(["2020-01-31"] * 40 + ["2020-02-29"] * 40)
    df = pd.DataFrame({
        "date": dates,
        "ISIN": [f"X{i:03d}" for i in range(40)] * 2,
        "stock_return": np.random.default_rng(1).normal(0, 0.02, 80),
        "market_cap": np.linspace(100, 10000, 80),
        "sector": ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20,
        "country": ["X"] * 80,
        "value": list(np.linspace(-1, 1, 39)) + [100] + list(np.linspace(-1, 1, 40)),
    })
    cfg = DEFAULT_CONFIG.copy()
    cfg["target"]["stock_return_alignment"] = "already_forward"
    result = preprocess_panel(df, ["value"], cfg)
    assert "value__z" in result.panel
    assert result.panel["value__z"].abs().max() <= 3.0 + 1e-12
