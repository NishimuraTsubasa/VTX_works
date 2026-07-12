from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json
import pandas as pd


def read_market_inputs(root: Path) -> pd.DataFrame:
    files = sorted((root / "data" / "input" / "market").glob("Market_*.xlsx"))
    if not files:
        raise FileNotFoundError("No Market_*.xlsx files found under data/input/market")
    frames = [pd.read_excel(path, sheet_name="Market_Data", parse_dates=["date"]) for path in files]
    return pd.concat(frames, ignore_index=True)


def read_cross_asset_input(root: Path) -> pd.DataFrame:
    return pd.read_excel(root / "data" / "input" / "Cross_Asset_Data.xlsx", sheet_name="Cross_Asset_Data", parse_dates=["date"])


def read_universe(root: Path) -> pd.DataFrame:
    return pd.read_excel(root / "data" / "input" / "Universe_Master.xlsx", sheet_name="Universe")


def write_single_sheet(path: Path, sheet_name: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
        ws.set_row(0, 22, writer.book.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D", "align": "center"}))


def write_market_inputs(root: Path, market: pd.DataFrame) -> None:
    out_dir = root / "data" / "input" / "market"
    out_dir.mkdir(parents=True, exist_ok=True)
    for asset_id, frame in market.groupby("asset_id", sort=True):
        write_single_sheet(out_dir / f"Market_{asset_id}.xlsx", "Market_Data", frame.sort_values("date"))


def write_model_outputs(path: Path, sheets: Mapping[str, pd.DataFrame], manifest: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        header = writer.book.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D", "align": "center"})
        for sheet_name, frame in sheets.items():
            safe = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            ws.set_row(0, 22, header)
        manifest_df = pd.DataFrame({"Key": list(manifest.keys()), "Value": [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v) for v in manifest.values()]})
        manifest_df.to_excel(writer, sheet_name="Run_Manifest", index=False)
        writer.sheets["Run_Manifest"].set_row(0, 22, header)
