from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import math
import re
import shutil
import tempfile

import pandas as pd

from ..utils import flatten_dict

LOGGER = logging.getLogger(__name__)


SUMMARY_SHEET_DESCRIPTIONS = [
    ("Output_Manifest", "生成対象、出力可否、行数、出力ファイルをまとめた一覧です。", "最重要"),
    ("Executive_Summary", "最新の指数予測、代表ユニバース品質、Breadth、先物リスクをまとめた最重要結果です。", "最重要"),
    ("Representative_Universe", "最新時点で指数を表現するために選定した銘柄、セクター、選定ウェイト、選定スコアです。", "最重要"),
    ("Universe_Selection_Quality", "代表銘柄数、候補数、セクターカバレッジ、指数追随相関・誤差の要約です。", "最重要"),
    ("Index_Scores_Latest", "最新の指数期待リターン、順位、信頼度、銘柄数・ウェイトベースBreadthです。", "最重要"),
    ("Model_Accuracy_Summary", "最終指数モデルのRankIC、方向正解率、上位指数Hit Rate、RMSE等の要約です。", "最重要"),
    ("Futures_Risk_Latest", "最新の先物ボラティリティ、下方リスク、VaR、ES、最大ドローダウンです。", "重要"),
    ("Index_Factor_Exposure", "最新の指数別Value、Quality、Momentum等の代表ユニバース加重エクスポージャーです。", "重要"),
    ("Index_Group_Contribution", "最新のファクターグループ別予測寄与です。", "重要"),
    ("Index_Breadth", "指数・セクター別の選定銘柄数、カバレッジ、プラス予測比率です。", "重要"),
    ("Factor_Model_Selection", "各ファクターの採用モデル、最良モデル、1-SE閾値、採用理由です。", "最重要"),
    ("Factor_Model_Candidate_Summary", "4候補モデルすべてのOOS平均RankIC、標準誤差、1-SE判定、複雑度です。", "最重要"),
    ("Factor_Model_Methodology", "モデル選択の判定フローと各列の読み方です。", "重要"),
    ("Factor_Bin_Summary", "各ファクターの分位別平均翌期リターン、標準誤差、正符号率です。", "重要"),
    ("Factor_Bin_Factor_Summary", "ファクターごとのTop-Bottomスプレッド、単調性、安定性の要約です。", "重要"),
    ("Factor_Master_Used", "今回使用したファクターコード、名称、グループ、方向、個別設定です。", "重要"),
    ("Group_Settings_Used", "今回使用したValue等のグループ統合方法と主要設定です。", "重要"),
    ("Group_Weight_Latest", "最新時点のグループ内ファクターウェイト、IC、PCAローディングです。", "重要"),
    ("Group_Scores_Latest", "最新時点の個別銘柄Value、Momentum等のグループスコアです。", "補足"),
    ("Stock_Scores_Latest", "最新の個別銘柄alpha、順位スコア、信頼度スコアです。", "補足"),
    ("Data_Quality", "ファクターカバレッジ、Winsorize率、入力データ検証結果です。", "品質管理"),
    ("Input_Index_Coverage", "指数名が構成銘柄・セクターウェイト・先物リターンに存在するかを確認します。", "品質管理"),
    ("Resolved_Factor_Settings", "defaultをPython設定へ解決した後の最終ファクター設定です。", "設定"),
    ("Config_Validation", "factor_master.xlsxの重複、方向、列名、グループ設定の検証結果です。", "品質管理"),
    ("Config", "今回使用したPython Config辞書をフラット化した一覧です。", "設定"),
]

HISTORY_DESCRIPTIONS = {
    "Universe_Selection_Quality_History": "代表銘柄数、候補数、セクターカバレッジ、指数追随相関・誤差の全時系列です。",
    "Futures_Risk_History": "指数先物のローリング・ボラティリティ、下方リスク、VaR、ES、最大ドローダウンです。",
    "Index_Score_History": "指数スコア、Breadth、代表ユニバースカバレッジの全時系列です。",
    "Index_Factor_History": "指数別ファクター傾向の全時系列です。",
    "Model_Accuracy_History": "月次または週次のRankIC、方向正解率、上位・下位スプレッドと移動平均です。",
    "Model_Accuracy_Detail": "指数・時点単位の予測値、実現翌期リターン、方向正解フラグです。",
    "Universe_Sector_Allocation": "各指数・時点・セクターの目標ウェイト、利用可能銘柄数、目標・実選定数です。",
    "Universe_Selection_History": "代表ユニバースの全時点における構成銘柄と選定ウェイトです。",
    "Factor_Bin_By_Date": "時点・ファクター・ビン別の翌期リターン詳細です。",
    "Factor_Performance": "ファクター・候補モデル・時点別のOOS RankIC、IC、分位スプレッドです。",
    "Factor_Coefficients": "Walk-forward推定した単一ファクターモデルの係数履歴です。",
    "Factor_IC_History": "採用した単一ファクターモデルのOOS RankIC履歴です。",
    "Group_Weight_History": "グループ内統合ウェイトの全履歴です。",
    "PCA_Loading_History": "PCAを指定したグループのローディング・使用ウェイト履歴です。",
    "Group_Score_History": "個別銘柄単位のValue、Momentum等のグループスコア履歴です。",
    "Index_Group_History": "指数別のValue、Momentum等のグループ予測寄与履歴です。",
    "Composite_Coefficients": "グループ間Ridge統合の係数履歴です。",
    "Stock_Score_History": "個別銘柄alpha、順位スコア、信頼度スコアの全履歴です。",
}

def _snake_case(name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()
    return re.sub(r"_+", "_", result).strip("_")


HISTORY_FILENAMES = {
    name: _snake_case(name) + ".xlsx"
    for name in HISTORY_DESCRIPTIONS
}


def _safe_sheet_name(name: str) -> str:
    return name[:31]


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col]).dt.tz_localize(None)
    return out


def _build_formats(wb) -> dict[str, Any]:
    return {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#17365D", "align": "left", "valign": "vcenter"}),
        "section": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1, "align": "left"}),
        "header": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1, "align": "center", "valign": "vcenter"}),
        "input": wb.add_format({"font_color": "#0000FF"}),
        "percent": wb.add_format({"num_format": "0.00%;[Red](0.00%);-"}),
        "number": wb.add_format({"num_format": "0.000;[Red](0.000);-"}),
        "integer": wb.add_format({"num_format": "#,##0;[Red](#,##0);-"}),
        "date": wb.add_format({"num_format": "yyyy-mm-dd"}),
        "wrap": wb.add_format({"text_wrap": True, "valign": "top"}),
        "link": wb.add_format({"font_color": "#0000FF", "underline": 1}),
        "green": wb.add_format({"font_color": "#008000"}),
        "yellow": wb.add_format({"bg_color": "#FFF2CC"}),
    }


def _write_df(writer: pd.ExcelWriter, name: str, df: pd.DataFrame, formats: dict[str, Any]) -> None:
    sheet = _safe_sheet_name(name)
    out = _clean_dataframe(df)
    out.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    ws.freeze_panes(1, 0)
    ws.hide_gridlines(2)
    if len(out.columns) > 0:
        ws.autofilter(0, 0, max(len(out), 1), len(out.columns) - 1)
        ws.set_row(0, 24, formats["header"])
        for c, col in enumerate(out.columns):
            ws.write(0, c, col, formats["header"])
            sample = out[col].astype(str).replace("nan", "").head(500) if len(out) else pd.Series(dtype=str)
            width = min(max(len(str(col)) + 2, int(sample.map(len).quantile(0.90)) + 2 if len(sample) else 10), 42)
            lower = str(col).lower()
            cell_format = None
            if any(token in lower for token in [
                "return", "alpha", "volatility", "var", "shortfall", "drawdown", "weight", "coverage", "breadth", "rate", "accuracy", "spread", "rmse", "error"
            ]):
                cell_format = formats["percent"]
            elif any(token in lower for token in [
                "score", "exposure", "contribution", "rank_ic", "pearson_ic", "coef", "skew", "kurtosis", "correlation", "monotonicity", "ir", "metric", "threshold", "delta"
            ]):
                cell_format = formats["number"]
            elif "date" in lower or lower in {"train_end", "test_date"}:
                cell_format = formats["date"]
            elif any(token in lower for token in ["count", "observations", "periods", "complexity", "rank"]):
                cell_format = formats["integer"]
            ws.set_column(c, c, width, cell_format)


def _write_readme_sheet(writer: pd.ExcelWriter, tables: dict[str, pd.DataFrame], formats: dict[str, Any]) -> None:
    wb = writer.book
    readme = wb.add_worksheet("README")
    writer.sheets["README"] = readme
    readme.hide_gridlines(2)
    readme.set_row(0, 28)
    readme.merge_range("A1:D1", "株式ファクター・代表ユニバース・指数先物分析サマリー", formats["title"])
    readme.write("A3", "シート名", formats["header"])
    readme.write("B3", "内容", formats["header"])
    readme.write("C3", "重要度", formats["header"])
    readme.write("D3", "リンク", formats["header"])
    row = 3
    for sheet, desc, importance in SUMMARY_SHEET_DESCRIPTIONS:
        if sheet not in tables:
            continue
        readme.write(row, 0, sheet)
        readme.write(row, 1, desc, formats["wrap"])
        readme.write(row, 2, importance)
        safe = _safe_sheet_name(sheet).replace("'", "''")
        readme.write_url(row, 3, f"internal:'{safe}'!A1", formats["link"], "開く")
        row += 1

    row += 1
    readme.merge_range(row, 0, row, 3, "履歴データの出力方針", formats["section"])
    row += 1
    text = (
        "2500銘柄規模の履歴をサマリーブックへ格納すると容量が急増するため、履歴はoutputs/history配下へ種類ごとに1ファイルずつ出力します。"
        "各履歴ファイルはREADMEとData_001以降のシートを持ち、1シートの設定上限を超える場合は同一ファイル内で自動分割します。"
        "出力可否はconfig/model_config.pyのreport.history_excel.tablesで個別に切り替えます。"
    )
    readme.merge_range(row, 0, row + 2, 3, text, formats["wrap"])
    readme.set_row(row, 52)

    row += 4
    readme.merge_range(row, 0, row, 3, "モデル選択の読み方", formats["section"])
    row += 1
    model_text = (
        "4候補をWalk-forward OOS平均RankICで比較し、最大値をbest_raw_modelとします。"
        "one-SE ruleを使用する場合、bestの平均RankICからbestの標準誤差を差し引いた値を閾値とし、その閾値以上の候補を『ほぼ同等』とみなします。"
        "その中から最も単純なモデルを採用するため、非線形モデルの改善が推定誤差の範囲内であればLinearが選択されます。"
        "詳細はFactor_Model_Selection、Factor_Model_Candidate_Summary、factor_model_selection_report.pdfを参照してください。"
    )
    readme.merge_range(row, 0, row + 3, 3, model_text, formats["wrap"])
    readme.set_row(row, 64)

    readme.set_column("A:A", 34)
    readme.set_column("B:B", 96)
    readme.set_column("C:C", 14)
    readme.set_column("D:D", 12)


def create_excel_report(path: str | Path, tables: dict[str, pd.DataFrame], config: dict[str, Any]) -> None:
    """Create the lightweight summary workbook; history tables must be excluded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="stock_index_summary_excel_"))
    tmp_path = tmp_dir / path.name
    with pd.ExcelWriter(tmp_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        formats = _build_formats(writer.book)
        _write_readme_sheet(writer, tables, formats)

        ordered_names = [x[0] for x in SUMMARY_SHEET_DESCRIPTIONS if x[0] in tables and x[0] != "Config"]
        for name in ordered_names:
            df = tables[name]
            LOGGER.info("Writing summary sheet %s shape=%s", name, getattr(df, "shape", None))
            _write_df(writer, name, df, formats)
            ws = writer.sheets[_safe_sheet_name(name)]
            if name in {
                "Executive_Summary", "Model_Accuracy_Summary", "Futures_Risk_Latest",
                "Factor_Model_Selection", "Factor_Model_Candidate_Summary",
                "Factor_Bin_Factor_Summary", "Universe_Selection_Quality",
            } and len(df) > 0 and len(df.columns) > 0:
                ws.conditional_format(1, 0, len(df), len(df.columns) - 1, {
                    "type": "3_color_scale",
                    "min_color": "#F4CCCC",
                    "mid_color": "#FFF2CC",
                    "max_color": "#D9EAD3",
                })

        config_df = pd.DataFrame(flatten_dict(config), columns=["config_key", "value"])
        _write_df(writer, "Config", config_df, formats)
        writer.sheets["Config"].set_column(1, 1, 90, formats["input"])

    shutil.copy2(tmp_path, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def create_history_workbook(
    path: str | Path,
    table_name: str,
    df: pd.DataFrame,
    description: str,
    config: dict[str, Any],
) -> None:
    """Write one history type per workbook, splitting rows across Data_NNN sheets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = int(config["report"]["history_excel"].get("max_rows_per_sheet", 800000))
    max_rows = max(1, min(max_rows, 1_048_000))
    out = _clean_dataframe(df)
    total_parts = max(1, math.ceil(len(out) / max_rows))

    tmp_dir = Path(tempfile.mkdtemp(prefix="stock_index_history_excel_"))
    tmp_path = tmp_dir / path.name
    with pd.ExcelWriter(tmp_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        formats = _build_formats(writer.book)
        wb = writer.book
        ws = wb.add_worksheet("README")
        writer.sheets["README"] = ws
        ws.hide_gridlines(2)
        ws.merge_range("A1:D1", f"履歴データ: {table_name}", formats["title"])
        rows = [
            ("データ名", table_name),
            ("内容", description),
            ("総行数", len(out)),
            ("総列数", len(out.columns)),
            ("データシート数", total_parts),
            ("最大行数/シート", max_rows),
            ("出力制御", f'report.history_excel.tables["{table_name}"]'),
        ]
        ws.write("A3", "項目", formats["header"])
        ws.write("B3", "値", formats["header"])
        for r, (key, value) in enumerate(rows, start=3):
            ws.write(r, 0, key)
            ws.write(r, 1, value, formats["wrap"])
        ws.set_column("A:A", 28)
        ws.set_column("B:B", 100)

        if len(out) == 0:
            _write_df(writer, "Data_001", out, formats)
        else:
            for part in range(total_parts):
                start = part * max_rows
                stop = min((part + 1) * max_rows, len(out))
                _write_df(writer, f"Data_{part + 1:03d}", out.iloc[start:stop], formats)

    shutil.copy2(tmp_path, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
