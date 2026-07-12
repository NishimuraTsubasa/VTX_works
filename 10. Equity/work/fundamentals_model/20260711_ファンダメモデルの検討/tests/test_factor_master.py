from __future__ import annotations

import pandas as pd
import pytest

from stock_index_model.factor_master import validate_factor_settings


def _valid_tables():
    master = pd.DataFrame({
        "Factor_Code": ["FA0101", "FA1001"],
        "Factor_Name_JP": ["益回り", "モメンタム"],
        "Factor_Name_EN": ["Earnings Yield", "Momentum"],
        "Factor_Group": ["Value", "Momentum"],
        "Enabled": [1, 1],
        "Direction": [1, 1],
        "Base_Weight": [1.0, 1.0],
        "Transform": ["none", "none"],
        "Winsorize": ["default", "default"],
        "Neutralize": ["default", "default"],
        "Rank_Normalize": ["default", "default"],
        "Min_Coverage": [0.6, 0.6],
        "Description": ["", ""],
    })
    groups = pd.DataFrame({
        "Factor_Group": ["Value", "Momentum"],
        "Display_Name": ["バリュー", "モメンタム"],
        "Enabled": [1, 1],
        "Aggregation_Method": ["equal_weight", "ic_adjusted"],
        "Lookback_Periods": [36, 36],
        "Min_Periods": [18, 18],
        "Max_Weight": [0.6, 0.6],
        "Weight_Smoothing": [0.5, 0.5],
        "Fallback_Method": ["equal_weight", "equal_weight"],
    })
    return master, groups


def test_factor_master_validation_passes():
    master, groups = _valid_tables()
    result = validate_factor_settings(master, groups, ["FA0101", "FA1001"])
    assert not (result["severity"] == "ERROR").any()


def test_duplicate_factor_is_error():
    master, groups = _valid_tables()
    master = pd.concat([master, master.iloc[[0]]], ignore_index=True)
    result = validate_factor_settings(master, groups, ["FA0101", "FA1001"])
    assert (result["severity"] == "ERROR").any()
