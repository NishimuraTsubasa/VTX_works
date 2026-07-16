from __future__ import annotations

import pandas as pd

from stock_scoring_model.feature_engineering import add_forward_return, generate_derived_features
from stock_scoring_model.master import FactorMeta


def _config() -> dict:
    return {
        "columns": {
            "date": "date",
            "isin": "ISIN",
            "stock_return": "stock_return",
            "market_cap": "market_cap",
            "currency": "currency",
            "country": "country",
            "sector": "sector",
        },
        "target": {"stock_return_alignment": "contemporaneous_to_forward", "stock_horizon_periods": 1},
    }


def test_forward_return_is_shifted_by_one_period() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "ISIN": ["A"] * 3,
            "stock_return": [0.01, 0.02, 0.03],
        }
    )
    out = add_forward_return(data, _config())
    assert out["NextMonthReturn"].iloc[0] == 0.02
    assert out["NextMonthReturn"].iloc[1] == 0.03
    assert pd.isna(out["NextMonthReturn"].iloc[2])


def test_derived_difference_respects_source_lag() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=4, freq="ME"),
            "ISIN": ["A"] * 4,
            "FA0101": [1.0, 2.0, 4.0, 7.0],
        }
    )
    control = pd.DataFrame(
        [["factor", "FA0101", 1, "selected", 1]],
        columns=["Scope_Type", "Scope_Value", "Enabled", "Generation_Mode", "Include_Raw"],
    )
    rules = pd.DataFrame(
        [["R1", "factor", "FA0101", "difference", 1, 0, 1, 1, 1, 1, 1, "inherit", 1]],
        columns=[
            "Rule_ID", "Scope_Type", "Scope_Value", "Feature_Type", "Difference_Periods",
            "Window_Periods", "Min_Periods", "Source_Lag_Periods",
            "Exclude_Source_From_Baseline", "Enabled", "Selected", "Direction_Mode", "Custom_Direction",
        ],
    )
    metas = {"FA0101": FactorMeta("FA0101", "Value", 1, 1.0)}
    cfg = _config()
    out, all_metas, lineage = generate_derived_features(data, cfg, metas, control, rules)
    code = "FA0101__DIFF_P1_L1"
    assert code in all_metas
    # t=3の値はx_{t-1}-x_{t-2}=4-2=2
    assert out[code].iloc[3] == 2.0
    assert int(lineage.loc[0, "Effective_Source_to_Target_Gap"]) == 2
