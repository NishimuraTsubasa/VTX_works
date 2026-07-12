from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json
import pandas as pd


HEADER = {"bold": True, "font_color": "white", "bg_color": "#17365D", "align": "center", "valign": "vcenter"}


def write_frame(path: Path, sheet_name: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        fmt = writer.book.add_format(HEADER)
        ws.set_row(0, 24, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
        for idx, col in enumerate(frame.columns):
            width = min(max(len(str(col)) + 2, 12), 28)
            ws.set_column(idx, idx, width)


def write_multi_sheet(path: Path, sheets: Mapping[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        fmt = writer.book.add_format(HEADER)
        for raw_name, frame in sheets.items():
            name = raw_name[:31]
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.set_row(0, 24, fmt)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            for idx, col in enumerate(frame.columns):
                width = min(max(len(str(col)) + 2, 12), 34)
                ws.set_column(idx, idx, width)
