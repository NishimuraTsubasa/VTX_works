import pandas as pd

from stock_scoring_model.feature_engineering import build_engineered_factor_panel


def test_lagged_difference_and_moving_average_alignment():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    stocks = pd.DataFrame({
        "date": dates,
        "ISIN": ["A"] * 6,
        "FA0101": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
    })
    fm = pd.DataFrame([{
        "Factor_Code": "FA0101", "Factor_Name_JP": "益回り", "Factor_Name_EN": "EY",
        "Factor_Group": "Value", "Enabled": 1, "Direction": 1, "Base_Weight": 1.0,
        "Transform": "none", "Winsorize": "default", "Neutralize": 1,
        "Rank_Normalize": 1, "Min_Coverage": 0.0, "Description": "",
    }])
    control = pd.DataFrame([{
        "Scope_Type": "group", "Scope_Value": "Value", "Enabled": 1,
        "Generation_Mode": "all", "Include_Raw": 1, "Notes": "",
    }])
    rules = pd.DataFrame([
        {"Rule_ID": "D1", "Scope_Type": "group", "Scope_Value": "Value", "Feature_Type": "difference",
         "Difference_Periods": 1, "Window_Periods": 0, "Min_Periods": 1, "Source_Lag_Periods": 1,
         "Exclude_Source_From_Baseline": 1, "Enabled": 1, "Selected": 1,
         "Direction_Mode": "inherit", "Custom_Direction": 1, "Description": ""},
        {"Rule_ID": "M2", "Scope_Type": "group", "Scope_Value": "Value", "Feature_Type": "rolling_mean_deviation",
         "Difference_Periods": 0, "Window_Periods": 2, "Min_Periods": 2, "Source_Lag_Periods": 1,
         "Exclude_Source_From_Baseline": 1, "Enabled": 1, "Selected": 1,
         "Direction_Mode": "inherit", "Custom_Direction": 1, "Description": ""},
    ])
    cfg = {
        "columns": {"date": "date", "isin": "ISIN"},
        "target": {"stock_horizon_periods": 1},
        "feature_engineering": {"enabled": True, "defaults": {"enabled": False, "generation_mode": "selected", "include_raw": True}},
    }
    result = build_engineered_factor_panel(stocks, ["FA0101"], fm, control, rules, cfg)
    diff = result.panel["FA0101__DIFF_P1_L1"]
    # At scoring date t=3 (index 3), use x[t-1]-x[t-2] = 4-2 = 2.
    assert diff.iloc[3] == 2.0
    madev = result.panel["FA0101__MADEV_W2_L1"]
    # At index 4: source x[3]=7, baseline x[2],x[1] = (4+2)/2 = 3, deviation=4.
    assert madev.iloc[4] == 4.0
    lineage = result.lineage.set_index("Factor_Code")
    assert lineage.loc["FA0101__DIFF_P1_L1", "Effective_Target_Gap_Periods"] == 2
