from __future__ import annotations

import logging
from pathlib import Path
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from .binning import compute_factor_bins
from .config import load_config
from .factor_master import resolved_factor_settings
from .io import infer_factor_columns, load_all
from .modeling import (
    build_composite_oof,
    build_group_oof,
    fit_latest_stock_scores,
    selected_oof_predictions,
    walk_forward_single_factors,
)
from .preprocessing import preprocess_panel
from .reporting import (
    HISTORY_DESCRIPTIONS,
    HISTORY_FILENAMES,
    create_excel_report,
    create_history_workbook,
    create_scenario_workbook,
    create_file_inventory_workbook,
)
from .scenario_evaluation import evaluate_stock_scoring_scenarios
from .scenario_scoring import build_stock_scoring_scenarios
from .utils import ensure_dir, setup_logging
from .validation import validate_stock_panel

LOGGER = logging.getLogger(__name__)

PDF_REPORTS = {
    "factor_scatter": (
        "factor_scatter_pdf", "factor_scatter_diagnostics.pdf", "create_factor_scatter_pdf"
    ),
    "factor_bin": (
        "factor_bin_pdf", "factor_bin_diagnostics.pdf", "create_factor_bin_pdf"
    ),
    "factor_model_selection": (
        "factor_model_selection_pdf", "factor_model_selection_report.pdf", "create_factor_model_selection_pdf"
    ),
    "factor_model_performance": (
        "factor_model_performance_pdf", "factor_model_performance.pdf", "create_factor_performance_pdf"
    ),
    "scenario_quintile_cumulative": (
        "scenario_quintile_pdf", "quintile_cumulative_returns.pdf", "create_scenario_quintile_cumulative_pdf"
    ),
    "scenario_comparison": (
        "scenario_comparison_pdf", "stock_scoring_scenario_comparison.pdf", "create_scenario_comparison_pdf"
    ),
}

PDF_DESCRIPTIONS = {
    "factor_scatter": "個別銘柄リターンと単一ファクターの散布図・候補回帰線",
    "factor_bin": "単一ファクターの5分位別翌期リターン",
    "factor_model_selection": "4候補モデルのOOS RankIC・1-SE閾値・採用根拠",
    "factor_model_performance": "4候補モデルのローリングOOS RankIC推移",
    "scenario_quintile_cumulative": "各個別銘柄スコアリングパターンのQ1-Q5累積リターン",
    "scenario_comparison": "各パターンのQ5-Q1累積、RankIC推移、主要指標比較",
}


def _latest(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df:
        return df.copy()
    return df[df[date_col].eq(df[date_col].max())].copy()


def _summary_sheet_enabled(config: dict, name: str) -> bool:
    settings = config["report"].get("summary_excel", {})
    return bool(settings.get("sheets", {}).get(name, True))


def _model_selection_methodology(config: dict) -> pd.DataFrame:
    s = config["model"].get("model_selection", {})
    metric = s.get("primary_metric", "mean_rank_ic")
    multiplier = float(s.get("one_se_multiplier", 1.0))
    return pd.DataFrame([
        {"Step": 1, "判定": "Walk-forward OOS評価", "内容": "各テスト時点で過去データだけを使い、4候補モデルの個別銘柄予測を作成する。"},
        {"Step": 2, "判定": "最良モデル", "内容": f"{metric}が最大の候補をbest_raw_modelとする。"},
        {"Step": 3, "判定": "1-SE閾値", "内容": f"閾値 = bestの平均RankIC - {multiplier:.1f} × bestのRankIC標準誤差。"},
        {"Step": 4, "判定": "複雑度", "内容": "閾値以上の候補の中で最も単純なモデルを採用する。"},
        {"Step": 5, "判定": "採用ゲート", "内容": "平均RankIC、正符号率、評価期間数の最低基準を確認する。"},
        {"Step": 6, "判定": "Linearが多い理由", "内容": "非線形モデルの改善が推定誤差を上回らない場合、安定性を優先してLinearを選ぶため。"},
    ])


def _run_pdf_job(
    base_dir: Path,
    output_dir: Path,
    config: dict,
    filename: str,
    function_name: str,
    args: tuple,
) -> Path:
    """Create a PDF directly in-process.

    The stock-only reports are compact enough that process isolation is unnecessary and
    direct execution avoids repeated font-cache/process startup overhead.
    """
    from . import diagnostics

    path = output_dir / filename
    func = getattr(diagnostics, function_name)
    func(*args, path)
    return path


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    setup_logging(config["project"]["log_level"])
    base_dir = config_path.parent.parent
    output_dir = ensure_dir(base_dir / config["project"]["output_dir"])
    date_col = config["columns"]["date"]
    isin_col = config["columns"]["isin"]

    LOGGER.info("Loading individual-stock inputs")
    data = load_all(config, config_path)
    stocks = data.stocks
    factor_master = data.factor_settings.factor_master
    group_settings = data.factor_settings.group_settings
    method_params = data.factor_settings.method_params
    factors = infer_factor_columns(stocks, data.factor_settings, config)
    factor_map = {f: f"{f}__z" for f in factors}
    config["runtime"] = {
        "factor_labels": factor_master.set_index("Factor_Code")["Factor_Name_JP"].to_dict(),
        "factor_groups": factor_master.set_index("Factor_Code")["Factor_Group"].to_dict(),
        "group_labels": group_settings.set_index("Factor_Group")["Display_Name"].to_dict(),
    }

    validation = validate_stock_panel(stocks, factors, config)

    LOGGER.info("Preprocessing individual-stock factor panel")
    prep = preprocess_panel(stocks, factors, config, factor_master)
    bins = compute_factor_bins(prep.panel, factor_map, config)

    LOGGER.info("Running walk-forward single-factor models")
    run = walk_forward_single_factors(prep.panel, factor_map, config)
    factor_oof = selected_oof_predictions(run, config)
    group_oof, group_weights = build_group_oof(
        factor_oof, factor_master, group_settings, method_params, config
    )
    composite_oof, composite_coefs = build_composite_oof(group_oof, config)

    LOGGER.info("Fitting latest individual-stock scores")
    latest_stock, latest_factor, latest_group, latest_group_weights = fit_latest_stock_scores(
        prep.panel, factor_map, run, factor_oof, group_oof,
        factor_master, group_settings, method_params, config
    )
    group_weight_all = pd.concat([group_weights, latest_group_weights], ignore_index=True, sort=False).drop_duplicates(
        ["date", "group", "factor"], keep="last"
    )

    LOGGER.info("Building stock-scoring comparison scenarios")
    scenarios = build_stock_scoring_scenarios(
        prep.panel, factors, factor_master, group_settings, method_params,
        factor_oof, latest_factor, group_oof, latest_group,
        composite_oof, latest_stock, group_weight_all, config,
    )
    evaluation = evaluate_stock_scoring_scenarios(scenarios, config)

    # Human-readable metadata for model diagnostics.
    factor_meta = factor_master[["Factor_Code", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group"]].rename(
        columns={"Factor_Code": "factor"}
    )
    run.selection = run.selection.merge(factor_meta, on="factor", how="left")
    run.selection_detail = run.selection_detail.merge(factor_meta, on="factor", how="left")
    run.metrics_by_date = run.metrics_by_date.merge(factor_meta, on="factor", how="left")
    run.metrics_summary = run.metrics_summary.merge(factor_meta, on="factor", how="left")
    run.coefficients = run.coefficients.merge(factor_meta, on="factor", how="left")
    bins.summary = bins.summary.merge(factor_meta, on="factor", how="left")
    bins.factor_summary = bins.factor_summary.merge(factor_meta, on="factor", how="left")
    bins.by_date = bins.by_date.merge(factor_meta, on="factor", how="left")
    selected_models = run.selection[["factor", "selected_model"]].drop_duplicates()
    factor_ic_history = run.metrics_by_date.merge(selected_models, on="factor", how="left")
    factor_ic_history = factor_ic_history[factor_ic_history["model"].eq(factor_ic_history["selected_model"])].copy()

    selection_cols = [
        "factor", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group",
        "selected_model", "best_raw_model", "selection_reason_code", "selection_reason_jp",
        "selected_primary_metric", "best_primary_metric", "best_standard_error",
        "one_se_threshold", "selected_delta_from_best", "selected_complexity",
        "positive_rate", "rank_ic_ir", "mean_top_bottom_spread", "count_periods", "adopted",
    ]
    selection_view = run.selection[[c for c in selection_cols if c in run.selection.columns]].copy()
    candidate_cols = [
        "factor", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group", "model",
        "mean_rank_ic", "rank_ic_se", "rank_ic_ir", "positive_rate", "mean_pearson_ic",
        "mean_top_bottom_spread", "count_periods", "primary_metric_rank", "complexity",
        "best_raw_model", "one_se_threshold", "within_one_se", "selected", "selected_model",
        "selection_reason_jp", "adopted",
    ]
    candidate_view = run.selection_detail[[c for c in candidate_cols if c in run.selection_detail.columns]].copy()
    candidate_view = candidate_view.sort_values(["factor", "primary_metric_rank"])

    latest_stock_small = _latest(latest_stock, date_col)
    keep_latest = [
        date_col, isin_col, config["columns"].get("currency"), config["columns"].get("market_cap"),
        "stock_alpha", "stock_score", "confidence_score",
    ]
    latest_stock_small = latest_stock_small[[c for c in keep_latest if c and c in latest_stock_small.columns]]

    summary_tables = {
        "Scenario_Comparison": evaluation.summary,
        "Scenario_Quintile_Summary": evaluation.quintile_summary,
        "Factor_Model_Selection": selection_view,
        "Factor_Model_Candidate_Summary": candidate_view,
        "Factor_Model_Methodology": _model_selection_methodology(config),
        "Factor_Bin_Factor_Summary": bins.factor_summary,
        "Group_Weight_Latest": _latest(group_weight_all, "date"),
        "Stock_Scores_Latest": latest_stock_small,
        "Data_Quality": pd.concat([prep.quality, validation], ignore_index=True, sort=False),
        "Factor_Master_Used": factor_master,
        "Group_Settings_Used": group_settings,
        "Resolved_Factor_Settings": resolved_factor_settings(factor_master, config),
        "Config_Validation": data.factor_settings.validation,
        "Config": pd.DataFrame(),
    }
    summary_tables = {
        name: df for name, df in summary_tables.items()
        if name == "Config" or _summary_sheet_enabled(config, name)
    }

    history_tables = {
        "Scenario_RankIC_History": evaluation.rank_ic_history,
        "Scenario_Quintile_Return_History": evaluation.quintile_return_history,
        "Scenario_LongShort_History": evaluation.long_short_history,
        "Factor_Bin_By_Date": bins.by_date,
        "Factor_Performance": run.metrics_by_date,
        "Factor_Coefficients": run.coefficients,
        "Factor_IC_History": factor_ic_history,
        "Group_Weight_History": group_weight_all,
        "PCA_Loading_History": group_weight_all[group_weight_all.get("method", pd.Series(dtype=str)).eq("pca")].copy(),
        "Group_Score_History": pd.concat([group_oof, latest_group], ignore_index=True, sort=False),
        "Composite_Coefficients": composite_coefs,
        "Stock_Score_History": pd.concat([composite_oof, latest_stock], ignore_index=True, sort=False),
    }

    outputs: dict[str, Path] = {}
    output_rows: list[dict] = []

    history_cfg = config["report"].get("history_excel", {})
    history_dir = ensure_dir(output_dir / history_cfg.get("output_subdir", "history"))
    for table_name, df in history_tables.items():
        enabled = bool(history_cfg.get("enabled", True) and history_cfg.get("tables", {}).get(table_name, False))
        filename = HISTORY_FILENAMES.get(table_name, f"{table_name.lower()}.xlsx")
        path = history_dir / filename
        if enabled:
            create_history_workbook(path, table_name, df, HISTORY_DESCRIPTIONS.get(table_name, table_name), config)
            outputs[f"history_{table_name}"] = path
        elif path.exists():
            path.unlink()
        output_rows.append({
            "output_id": table_name, "category": "history_excel", "enabled": enabled,
            "generated": bool(enabled and path.exists()), "rows": int(len(df)), "columns": int(len(df.columns)),
            "relative_path": str(path.relative_to(output_dir)),
            "description": HISTORY_DESCRIPTIONS.get(table_name, table_name),
        })

    scenario_cfg = config["report"].get("scenario_excel", {})
    scenario_dir = ensure_dir(output_dir / scenario_cfg.get("output_subdir", "stock_score_patterns"))
    for scenario_id, result in scenarios.items():
        enabled = bool(scenario_cfg.get("enabled", True) and scenario_cfg.get("scenarios", {}).get(scenario_id, True))
        path = scenario_dir / f"{scenario_id}.xlsx"
        if enabled:
            create_scenario_workbook(path, result, factor_master, config)
            outputs[f"scenario_{scenario_id}"] = path
        elif path.exists():
            path.unlink()
        output_rows.append({
            "output_id": scenario_id, "category": "stock_score_pattern_excel", "enabled": enabled,
            "generated": bool(enabled and path.exists()), "rows": int(len(result.stock_scores)), "columns": 8,
            "relative_path": str(path.relative_to(output_dir)), "description": result.description,
        })

    pdf_args = {
        "factor_scatter": (prep.panel, factor_map, config),
        "factor_bin": (bins.summary, bins.factor_summary, config),
        "factor_model_selection": (run.selection_detail, run.selection, config),
        "factor_model_performance": (run.metrics_by_date, run.selection, config),
        "scenario_quintile_cumulative": (evaluation.quintile_return_history, evaluation.summary, config),
        "scenario_comparison": (evaluation.summary, evaluation.rank_ic_history, evaluation.long_short_history, config),
    }
    pdf_cfg = config["report"].get("pdf", {})
    for report_id, (_, filename, function_name) in PDF_REPORTS.items():
        enabled = bool(pdf_cfg.get("enabled", True) and pdf_cfg.get("reports", {}).get(report_id, True))
        path = output_dir / filename
        if enabled:
            path = _run_pdf_job(base_dir, output_dir, config, filename, function_name, pdf_args[report_id])
            outputs[report_id] = path
        elif path.exists():
            path.unlink()
        output_rows.append({
            "output_id": report_id, "category": "pdf", "enabled": enabled,
            "generated": bool(enabled and path.exists()), "rows": None, "columns": None,
            "relative_path": path.name, "description": PDF_DESCRIPTIONS[report_id],
        })

    inventory_cfg = config["report"].get("file_inventory", {})
    inventory_path = output_dir / inventory_cfg.get("filename", "file_inventory.xlsx")
    inventory_rows = [
        {"stage": "Input", "requirement": "Required", "file_path": config["data"]["factors_file"], "file_type": "xlsx", "purpose": "個別銘柄の時点別ファクター値、リターン、時価総額、通貨、属性"},
        {"stage": "Input", "requirement": "Required", "file_path": config["data"]["factor_master_file"], "file_type": "xlsx", "purpose": "FAコード、名称、グループ、方向、統合方法"},
        {"stage": "Config", "requirement": "Required", "file_path": str(config_path.relative_to(base_dir)), "file_type": "py", "purpose": "前処理、モデル、評価、出力可否"},
        {"stage": "Documentation", "requirement": "Reference", "file_path": "README.md", "file_type": "md", "purpose": "実行方法と個別銘柄分析フロー"},
        {"stage": "Documentation", "requirement": "Reference", "file_path": "docs/file_inventory.md", "file_type": "md", "purpose": "入力から出力までのファイル一覧"},
    ]
    for row in output_rows:
        inventory_rows.append({
            "stage": "Output", "requirement": "Generated" if row["enabled"] else "Disabled",
            "file_path": row["relative_path"], "file_type": Path(str(row["relative_path"])).suffix.lstrip("."),
            "purpose": row["description"],
        })
    if inventory_cfg.get("enabled", True):
        create_file_inventory_workbook(inventory_path, inventory_rows)
        outputs["file_inventory"] = inventory_path

    summary_cfg = config["report"].get("summary_excel", {})
    summary_path = output_dir / summary_cfg.get("filename", "analysis_summary.xlsx")
    output_rows.append({
        "output_id": "summary_excel", "category": "summary_excel", "enabled": summary_cfg.get("enabled", True),
        "generated": summary_cfg.get("enabled", True), "rows": sum(len(v) for v in summary_tables.values()),
        "columns": None, "relative_path": summary_path.name,
        "description": "個別銘柄スコアリングモデルの比較・最新結果・設定サマリー",
    })
    summary_tables["Output_Manifest"] = pd.DataFrame(output_rows)
    if summary_cfg.get("enabled", True):
        create_excel_report(summary_path, summary_tables, config)
        outputs["summary_excel"] = summary_path

    LOGGER.info("Stock-only pipeline completed. Outputs: %s", outputs)
    return outputs
