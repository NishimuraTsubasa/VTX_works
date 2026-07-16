from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from ..scenario_scoring import ScenarioResult
from ..utils import flatten_dict


def _safe_sheet(name: str) -> str:
    return name[:31]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col]).dt.tz_localize(None)
    return out


def _formats(wb) -> dict[str, Any]:
    return {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#17365D"}),
        "header": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1, "align": "center"}),
        "wrap": wb.add_format({"text_wrap": True, "valign": "top"}),
        "date": wb.add_format({"num_format": "yyyy-mm-dd"}),
        "number": wb.add_format({"num_format": "0.0000;[Red](0.0000);-"}),
        "percent": wb.add_format({"num_format": "0.00%;[Red](0.00%);-"}),
        "integer": wb.add_format({"num_format": "#,##0;[Red](#,##0);-"}),
        "input": wb.add_format({"font_color": "#0000FF"}),
    }


def _write_df(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, fmts: dict[str, Any]) -> None:
    out = _clean(df)
    sheet_name = _safe_sheet(sheet_name)
    out.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 2 if len(out.columns) >= 2 else 0)
    if len(out.columns):
        ws.autofilter(0, 0, max(len(out), 1), len(out.columns) - 1)
        ws.set_row(0, 24, fmts["header"])
        for idx, col in enumerate(out.columns):
            ws.write(0, idx, col, fmts["header"])
            lower = str(col).lower()
            sample = out[col].astype(str).replace("nan", "").head(500) if len(out) else pd.Series(dtype=str)
            width = min(max(len(str(col)) + 2, int(sample.map(len).quantile(0.9)) + 2 if len(sample) else 10), 28)
            fmt = None
            if lower == "date" or "date" in lower:
                fmt = fmts["date"]
            elif "return" in lower:
                fmt = fmts["percent"]
            elif any(t in lower for t in ["score", "prediction", "marketcap"]):
                fmt = fmts["number"]
            elif "quintile" in lower:
                fmt = fmts["integer"]
            ws.set_column(idx, idx, width, fmt)


def _write_split(writer: pd.ExcelWriter, prefix: str, df: pd.DataFrame, max_rows: int, fmts: dict[str, Any]) -> None:
    out = _clean(df)
    parts = max(1, math.ceil(len(out) / max_rows))
    if len(out) == 0:
        _write_df(writer, f"{prefix}_001", out, fmts)
        return
    for part in range(parts):
        start, end = part * max_rows, min((part + 1) * max_rows, len(out))
        _write_df(writer, f"{prefix}_{part + 1:03d}", out.iloc[start:end], fmts)



def _apply_scope(df: pd.DataFrame, settings: dict[str, Any], date_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.copy()
    scope = str(settings.get("date_scope", "latest")).lower()
    if scope == "latest":
        return df[df[date_col].eq(pd.to_datetime(df[date_col]).max())].copy()
    if scope == "selected":
        dates = pd.to_datetime(settings.get("selected_dates", []), errors="coerce")
        return df[pd.to_datetime(df[date_col]).isin(dates)].copy()
    return df.copy()

def _stock_score_output(result: ScenarioResult, config: dict[str, Any]) -> pd.DataFrame:
    cols = {
        "date": "date", "isin": "ISIN", "currency": "currency", "market_cap": "market_cap",
        **config.get("columns", {}),
    }
    source = result.stock_scores.copy()
    prediction = None
    for candidate in ["stock_alpha", "score_before_ranking", "stock_score_minus1_1", "stock_score_0_1"]:
        if candidate in source.columns:
            prediction = candidate
            break
    mapping = {
        cols["date"]: "Date",
        cols["isin"]: "ISIN",
        cols.get("currency", "currency"): "Currency",
        cols.get("market_cap", "market_cap"): "MarketCap",
        "stock_score_0_1": "TotalScore",
        "forward_return": "NextMonthReturn",
        "score_quintile": "Quintile",
    }
    keep = [c for c in mapping if c in source.columns]
    out = source[keep].rename(columns=mapping)
    out["Prediction"] = pd.to_numeric(source[prediction], errors="coerce") if prediction else pd.NA
    order = ["Date", "ISIN", "Currency", "MarketCap", "TotalScore", "Prediction", "NextMonthReturn", "Quintile"]
    return out[[c for c in order if c in out.columns]]


def _sub_score_output(result: ScenarioResult, config: dict[str, Any]) -> pd.DataFrame:
    columns = {"date": "date", "isin": "ISIN", **config.get("columns", {})}
    date_col, isin_col = columns["date"], columns["isin"]
    frame = result.group_scores.copy()
    if frame.empty:
        return pd.DataFrame(columns=["Date", "ISIN", "SubScore", "SubScoreValue"])
    if "group" in frame.columns and "group_prediction" in frame.columns:
        out = frame[[date_col, isin_col, "group", "group_prediction"]].rename(
            columns={date_col: "Date", isin_col: "ISIN", "group": "SubScore", "group_prediction": "SubScoreValue"}
        )
        return out
    id_cols = [c for c in [date_col, isin_col] if c in frame.columns]
    value_cols = [c for c in frame.columns if c not in id_cols]
    out = frame[id_cols + value_cols].melt(id_vars=id_cols, var_name="SubScore", value_name="SubScoreValue")
    return out.rename(columns={date_col: "Date", isin_col: "ISIN"})


def _factor_score_output(result: ScenarioResult, factor_master: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = {"date": "date", "isin": "ISIN", **config.get("columns", {})}
    date_col, isin_col = columns["date"], columns["isin"]
    frame = result.factor_values.copy()
    factor_codes = [
        f for f in factor_master.loc[factor_master["Enabled"].astype(int).eq(1), "Factor_Code"].astype(str)
        if f in frame.columns
    ]
    if frame.empty or not factor_codes:
        return pd.DataFrame(columns=["Date", "ISIN", "FactorCode", "FactorScore"])
    out = frame[[date_col, isin_col] + factor_codes].melt(
        id_vars=[date_col, isin_col], var_name="FactorCode", value_name="FactorScore"
    )
    return out.rename(columns={date_col: "Date", isin_col: "ISIN"})


def create_scenario_workbook(
    path: str | Path,
    result: ScenarioResult,
    factor_master: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = config["report"].get("scenario_excel", {})
    max_rows = max(1, min(int(settings.get("max_rows_per_sheet", 500000)), 1_048_000))
    tmp_dir = Path(tempfile.mkdtemp(prefix="stock_score_scenario_"))
    tmp = tmp_dir / path.name

    date_col = config.get("columns", {}).get("date", "date")
    scoped = ScenarioResult(
        result.scenario_id, result.title, result.description,
        _apply_scope(result.stock_scores, settings, date_col),
        _apply_scope(result.factor_values, settings, date_col),
        _apply_scope(result.group_scores, settings, date_col),
        _apply_scope(result.factor_weights, settings, date_col),
    )
    stock_out = _stock_score_output(scoped, config)
    sub_out = _sub_score_output(scoped, config)
    factor_out = _factor_score_output(scoped, factor_master, config)

    with pd.ExcelWriter(tmp, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        fmts = _formats(writer.book)
        wb = writer.book
        ws = wb.add_worksheet("README")
        writer.sheets["README"] = ws
        ws.hide_gridlines(2)
        ws.merge_range("A1:D1", f"個別銘柄スコア比較: {result.scenario_id}", fmts["title"])
        rows = [
            ("シナリオ名", result.title),
            ("処理内容", result.description),
            ("StockScore行数", len(stock_out)),
            ("SubScore行数", len(sub_out)),
            ("FactorScore行数", len(factor_out)),
            ("出力期間", settings.get("date_scope", "all")),
            ("TotalScore", "時点別0-1個別銘柄スコア。1に近いほど高評価。"),
            ("Prediction", "順位化前の予測シグナル。S07では期待リターン予測、他シナリオでは合成前スコア。"),
            ("NextMonthReturn", "スコア計算時点の翌期個別銘柄リターン。"),
            ("SubScore", "Value・Momentum・Quality等のグループ別スコアを縦持ちで出力。"),
            ("FactorScore", "FA0101等のファクター別スコアを縦持ちで出力。"),
            ("分位", "Quintile: 1=最低20%、5=最高20%。"),
        ]
        ws.write("A3", "項目", fmts["header"])
        ws.write("B3", "内容", fmts["header"])
        for r, (k, v) in enumerate(rows, start=3):
            ws.write(r, 0, k)
            ws.write(r, 1, v, fmts["wrap"])
        ws.set_column("A:A", 28)
        ws.set_column("B:B", 105)

        if settings.get("include_factor_map", True):
            factor_map = factor_master[[
                c for c in ["Factor_Code", "Base_Factor_Code", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group",
                            "Feature_Type", "Rule_ID", "Source_Lag_Periods", "Effective_Target_Gap_Periods",
                            "Direction", "Enabled"]
                if c in factor_master.columns
            ]].copy()
            _write_df(writer, "Factor_Map", factor_map, fmts)
        _write_split(writer, "StockScore", stock_out, max_rows, fmts)
        if settings.get("include_sub_scores", True):
            _write_split(writer, "SubScore", sub_out, max_rows, fmts)
        if settings.get("include_factor_scores", True):
            _write_split(writer, "FactorScore", factor_out, max_rows, fmts)
        if settings.get("include_scenario_config", True):
            scenario_config = pd.DataFrame(flatten_dict({"scenario_excel": settings}), columns=["config_key", "value"])
            _write_df(writer, "Scenario_Config", scenario_config, fmts)
            writer.sheets["Scenario_Config"].set_column(1, 1, 90, fmts["input"])

    shutil.copy2(tmp, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def create_file_inventory_workbook(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="file_inventory_"))
    tmp = tmp_dir / path.name
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
        fmts = _formats(writer.book)
        wb = writer.book
        ws = wb.add_worksheet("README")
        writer.sheets["README"] = ws
        ws.hide_gridlines(2)
        ws.merge_range("A1:D1", "個別銘柄スコアリングモデル ファイル一覧", fmts["title"])
        ws.write("A3", "目的", fmts["header"])
        ws.merge_range("B3:D5", "個別銘柄スコアリングモデルで使用する入力、設定、文書、出力を一覧化しています。", fmts["wrap"])
        ws.set_column("A:A", 22)
        ws.set_column("B:D", 32)
        _write_df(writer, "File_List", df, fmts)
    shutil.copy2(tmp, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
