from pathlib import Path

import pandas as pd

from stock_scoring_model.scenario_scoring import ScenarioResult, SCENARIO_DESCRIPTIONS
from stock_scoring_model.reporting.scenario_excel import create_scenario_workbook


def test_scenario_definitions_are_complete():
    expected = {
        "S00_Current_Direct_EW",
        "S01_Missing_Adjusted_EW",
        "S02_Winsorized_Direct_EW",
        "S03_Neutralized_Direct_EW",
        "S04_Hierarchical_Equal_Weight",
        "S05_Correlation_Adjusted_IC",
        "S06_Selected_Factor_Models",
        "S07_Full_OOF_Ridge",
    }
    assert set(SCENARIO_DESCRIPTIONS) == expected


def test_scenario_workbook_is_created(tmp_path: Path):
    scores = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
        "ISIN": ["A", "B"],
        "stock_score_0_1": [0.8, 0.2],
        "score_quintile": [5, 1],
    })
    factors = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
        "ISIN": ["A", "B"],
        "FA0101": [0.9, 0.1],
    })
    fm = pd.DataFrame({
        "Factor_Code": ["FA0101"],
        "Factor_Name_JP": ["益回り"],
        "Factor_Name_EN": ["Earnings Yield"],
        "Factor_Group": ["Value"],
        "Direction": [1],
        "Enabled": [1],
    })
    result = ScenarioResult(
        "S00_Current_Direct_EW", "現行", "説明", scores, factors,
        pd.DataFrame(), pd.DataFrame(),
    )
    config = {
        "report": {
            "scenario_excel": {
                "max_rows_per_sheet": 500000,
                "date_scope": "all",
                "include_factor_values": True,
                "include_group_scores": True,
                "include_factor_weights": True,
            }
        }
    }
    output = tmp_path / "scenario.xlsx"
    create_scenario_workbook(output, result, fm, config)
    assert output.exists()
    assert output.stat().st_size > 0
