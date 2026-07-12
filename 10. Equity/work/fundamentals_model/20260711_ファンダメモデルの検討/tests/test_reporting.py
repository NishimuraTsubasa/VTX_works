from pathlib import Path

import pandas as pd

from stock_index_model.config import DEFAULT_CONFIG
from stock_index_model.reporting.excel_report import create_history_workbook


def test_history_workbook_splits_rows(tmp_path: Path):
    config = {**DEFAULT_CONFIG}
    config["report"] = {**DEFAULT_CONFIG["report"]}
    config["report"]["history_excel"] = {
        **DEFAULT_CONFIG["report"]["history_excel"],
        "max_rows_per_sheet": 2,
    }
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "value": range(5)})
    path = tmp_path / "history.xlsx"
    create_history_workbook(path, "Test_History", df, "test", config)
    xls = pd.ExcelFile(path)
    assert xls.sheet_names == ["README", "Data_001", "Data_002", "Data_003"]
    assert sum(len(pd.read_excel(path, sheet_name=s)) for s in xls.sheet_names[1:]) == 5
