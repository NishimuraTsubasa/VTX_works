from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .binscatter import run_binscatter_analysis, write_binscatter_pdf
from .config_loader import load_config
from .feature_engineering import add_forward_return, generate_derived_features
from .font_support import setup_japanese_matplotlib_from_config
from .io import ensure_output_dirs, read_inputs
from .master import parse_master, validate_data_columns


def write_binscatter_summary_excel(
    output_path: Path,
    summary: pd.DataFrame,
    points: pd.DataFrame,
    lineage: pd.DataFrame,
) -> None:
    """Binscatter診断専用のExcelを出力する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        readme = pd.DataFrame(
            {
                "Sheet": ["Regression_Summary", "Bin_Points", "Feature_Lineage"],
                "Content": [
                    "Scope・Factor別の相関、Linear/Quadratic/Broken-stickのR2・係数・knot",
                    "Time-average後の各ビン座標、標準誤差、95%信頼区間",
                    "元FAコードと派生ファクターの対応および時点ラグ",
                ],
            }
        )
        readme.to_excel(writer, sheet_name="README", index=False)
        (summary if summary is not None and not summary.empty else pd.DataFrame({"Message": ["No data"]})).to_excel(
            writer, sheet_name="Regression_Summary", index=False
        )
        (points if points is not None and not points.empty else pd.DataFrame({"Message": ["No data"]})).to_excel(
            writer, sheet_name="Bin_Points", index=False
        )
        (lineage if lineage is not None and not lineage.empty else pd.DataFrame({"Message": ["No derived features"]})).to_excel(
            writer, sheet_name="Feature_Lineage", index=False
        )
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
            worksheet.set_column(0, max(0, worksheet.dim_colmax), 18)
            worksheet.autofilter(0, 0, max(0, worksheet.dim_rowmax), max(0, worksheet.dim_colmax))


def run_binscatter_pipeline(config_path: str | Path) -> dict[str, Any]:
    config, root = load_config(config_path)
    setup_japanese_matplotlib_from_config(config, root)
    data, sheets, _ = read_inputs(config, root)
    parsed = parse_master(sheets)
    validate_data_columns(data, config["columns"], parsed["metas"])
    data = add_forward_return(data, config)
    data, all_metas, lineage = generate_derived_features(
        data,
        config,
        parsed["metas"],
        parsed["feature_control"],
        parsed["derived_rules"],
    )
    output_dirs = ensure_output_dirs(config, root)
    records, summary, points = run_binscatter_analysis(data, all_metas, config)

    pdf_cfg = config["outputs"].get("pdf", {})
    jobs = [
        ("binscatter_all_universe", "all_universe", "binscatter_all_universe.pdf", "All Universe | Time-Averaged Binscatter"),
        ("binscatter_by_country", "by_country", "binscatter_by_country.pdf", "Country | Time-Averaged Binscatter"),
        (
            "binscatter_by_country_sector",
            "by_country_sector",
            "binscatter_by_country_sector.pdf",
            "Country x Sector | Time-Averaged Binscatter",
        ),
    ]
    for flag, scope, filename, title in jobs:
        if pdf_cfg.get(flag, True):
            write_binscatter_pdf(records.get(scope, []), output_dirs["diagnostics"] / filename, title, config)

    write_binscatter_summary_excel(
        output_dirs["diagnostics"] / "binscatter_regression_summary.xlsx",
        summary,
        points,
        lineage,
    )
    return {
        "output_dir": output_dirs["diagnostics"],
        "summary_rows": 0 if summary is None else len(summary),
        "point_rows": 0 if points is None else len(points),
    }
