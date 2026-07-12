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

from .aggregation import aggregate_factor_exposures, aggregate_group_contributions, aggregate_stock_scores
from .binning import compute_factor_bins
from .config import load_config
from .io import attach_security_attributes, infer_factor_columns, load_all
from .factor_master import resolved_factor_settings
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
)
from .risk import evaluate_index_scores, futures_risk, summarize_model_accuracy
from .selection import build_representative_universe
from .utils import ensure_dir, setup_logging
from .validation import validate_index_names, validate_stock_panel

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
        "factor_performance_pdf", "factor_model_performance.pdf", "create_factor_performance_pdf"
    ),
    "index_factor_exposure": (
        "index_exposure_pdf", "index_factor_exposure.pdf", "create_index_exposure_pdf"
    ),
    "index_factor_trends": (
        "index_factor_trends_pdf", "index_factor_trends.pdf", "create_index_factor_trends_pdf"
    ),
    "model_accuracy": (
        "model_accuracy_pdf", "model_accuracy_report.pdf", "create_model_accuracy_pdf"
    ),
    "universe_selection": (
        "universe_selection_pdf", "universe_selection_report.pdf", "create_universe_selection_pdf"
    ),
    "futures_risk": (
        "futures_risk_pdf", "futures_risk_report.pdf", "create_futures_risk_pdf"
    ),
}

PDF_DESCRIPTIONS = {
    "factor_scatter": "個別銘柄散布図と線形・区分線形・二次・Combined Ridge回帰線",
    "factor_bin": "ファクター分位別の翌期リターン",
    "factor_model_selection": "4候補モデルの平均OOS RankIC、1-SE閾値、採用根拠",
    "factor_model_performance": "4候補モデルのローリングOOS RankIC推移",
    "index_factor_exposure": "最新指数ファクター傾向とグループ予測寄与",
    "index_factor_trends": "指数別ファクター傾向の時系列",
    "model_accuracy": "最終指数モデルのRankIC・方向正解率・スプレッド推移",
    "universe_selection": "代表銘柄選定数、セクターカバレッジ、指数追随品質",
    "futures_risk": "指数先物の最新・ローリングリスク",
}


def _latest(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df:
        return df.copy()
    return df[df[date_col] == df[date_col].max()].copy()


def _summary_sheet_enabled(config: dict, name: str) -> bool:
    settings = config["report"].get("summary_excel", {})
    return bool(settings.get("sheets", {}).get(name, True))


def _model_selection_methodology(config: dict) -> pd.DataFrame:
    s = config["model"].get("model_selection", {})
    metric = s.get("primary_metric", "mean_rank_ic")
    multiplier = float(s.get("one_se_multiplier", 1.0))
    return pd.DataFrame([
        {
            "Step": 1,
            "判定": "Walk-forward OOS評価",
            "内容": "各テスト時点で過去データだけを使い、Linear・Piecewise・Quadratic・Combined Ridgeの予測を作成する。",
        },
        {
            "Step": 2,
            "判定": "最良モデルを特定",
            "内容": f"{metric}が最大の候補をbest_raw_modelとする。既定値は月次OOS RankICの平均。",
        },
        {
            "Step": 3,
            "判定": "1-SE閾値",
            "内容": f"閾値 = bestの平均RankIC - {multiplier:.1f} × bestのRankIC標準誤差。閾値以上を『ほぼ同等』とみなす。",
        },
        {
            "Step": 4,
            "判定": "複雑度で選択",
            "内容": "ほぼ同等候補の中で最も単純なモデルを採用する。Linear=1、Piecewise=2、Quadratic=2、Combined Ridge=3。",
        },
        {
            "Step": 5,
            "判定": "採用ゲート",
            "内容": "選択後に平均RankIC、正符号率、評価期間数の最低基準を確認する。",
        },
        {
            "Step": 6,
            "判定": "Linearが多い理由",
            "内容": "非線形候補の改善が1標準誤差を超えない場合、将来安定性と説明可能性を優先してLinearを選ぶため。",
        },
    ])


def _run_pdf_job(
    base_dir: Path,
    output_dir: Path,
    config: dict,
    key: str,
    filename: str,
    function_name: str,
    args: tuple,
) -> Path:
    path = output_dir / filename
    tmp_dir = Path(tempfile.mkdtemp(prefix="stock_index_pdf_"))
    tmp_path = tmp_dir / filename
    payload_path = tmp_dir / "payload.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump({
            "function": function_name,
            "args": args,
            "output_path": str(tmp_path),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Creating PDF %s in isolated process", filename)
    env = os.environ.copy()
    src_dir = str(base_dir / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run(
        [sys.executable, "-m", "stock_index_model.report_worker", str(payload_path)],
        check=True,
        env=env,
        timeout=int(config["report"].get("pdf_timeout_seconds", 180)),
    )
    shutil.copy2(tmp_path, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    LOGGER.info("Finished PDF %s", filename)
    return path


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    setup_logging(config["project"]["log_level"])
    base_dir = config_path.parent.parent
    output_dir = ensure_dir(base_dir / config["project"]["output_dir"])
    date_col = config["columns"]["date"]
    isin_col = config["columns"]["isin"]

    LOGGER.info("Loading input workbooks")
    data = load_all(config, config_path)
    stocks, attribute_issues = attach_security_attributes(data.stocks, data.constituents, config)
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
    LOGGER.info("Detected factors: %s", factors)

    validation_stock = validate_stock_panel(stocks, factors, config)
    validation_index = validate_index_names(data.constituents, data.sector_weights, data.futures_returns, config)

    LOGGER.info("Building rolling representative stock universes")
    selection = build_representative_universe(
        stocks, factors, data.constituents, data.sector_weights, data.futures_returns, config
    )

    LOGGER.info("Preprocessing stock panel")
    prep = preprocess_panel(stocks, factors, config, factor_master)

    LOGGER.info("Computing factor-bin diagnostics")
    bins = compute_factor_bins(prep.panel, factor_map, config)

    LOGGER.info("Running walk-forward single-factor models")
    run = walk_forward_single_factors(prep.panel, factor_map, config)
    factor_oof = selected_oof_predictions(run, config)
    group_oof, group_weights = build_group_oof(
        factor_oof, factor_master, group_settings, method_params, config
    )
    composite_oof, composite_coefs = build_composite_oof(group_oof, config)

    LOGGER.info("Fitting latest stock scores")
    latest_stock, latest_factor, latest_group, latest_group_weights = fit_latest_stock_scores(
        prep.panel, factor_map, run, factor_oof, group_oof,
        factor_master, group_settings, method_params, config
    )
    history_cols = [date_col, isin_col, "stock_alpha", "stock_score", "confidence_score"]
    stock_all = pd.concat([
        composite_oof[history_cols],
        latest_stock[history_cols],
    ], ignore_index=True).drop_duplicates([date_col, isin_col], keep="last")

    LOGGER.info("Aggregating selected-stock signals to indices")
    index_scores, sector_detail = aggregate_stock_scores(
        stock_all, data.constituents, data.sector_weights, config, selection.history
    )
    exposures = aggregate_factor_exposures(
        prep.panel, factor_map, data.constituents, data.sector_weights, config, selection.history
    )
    group_all = pd.concat([group_oof, latest_group], ignore_index=True, sort=False)
    group_weight_all = pd.concat([group_weights, latest_group_weights], ignore_index=True, sort=False).drop_duplicates(
        ["date", "group", "factor"], keep="last"
    )
    contributions = aggregate_group_contributions(
        group_all, data.constituents, data.sector_weights, config, selection.history
    )

    LOGGER.info("Calculating futures risk and model accuracy")
    risk_latest, risk_history, corr = futures_risk(data.futures_returns, config)
    accuracy_detail, accuracy_history = evaluate_index_scores(index_scores, data.futures_returns, config)
    accuracy_overall, accuracy_by_index = summarize_model_accuracy(accuracy_detail, accuracy_history, config)

    # Attach human-readable factor metadata to diagnostic tables.
    factor_meta = factor_master[[
        "Factor_Code", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group"
    ]].rename(columns={"Factor_Code": "factor"})
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

    latest_index = _latest(index_scores, date_col).sort_values("index_alpha", ascending=False) if not index_scores.empty else pd.DataFrame()
    executive_cols = [
        "index_name", "index_alpha", "index_score", "index_confidence",
        "index_breadth_count_based", "index_breadth_weighted",
        "selection_weight_coverage", "selected_constituent_count", "usable",
    ]
    executive = latest_index[[c for c in executive_cols if c in latest_index]].copy()
    if not selection.quality.empty:
        latest_selection_q = _latest(selection.quality, date_col)
        qcols = [
            "index_name", "selection_key", "target_count", "selected_count", "candidate_count",
            "sector_weight_coverage", "tracking_correlation", "tracking_rmse", "actual_constituent_share",
        ]
        executive = executive.merge(
            latest_selection_q[[c for c in qcols if c in latest_selection_q]],
            on="index_name", how="left"
        )
    if not risk_latest.empty:
        executive = executive.merge(
            risk_latest[["index_name", "annualized_volatility", "historical_var", "max_drawdown"]],
            on="index_name", how="left",
        )

    quality = pd.concat([
        prep.quality.assign(issue_type="preprocessing_quality"),
        validation_stock,
        attribute_issues,
    ], ignore_index=True, sort=False)

    latest_selection = _latest(selection.history, date_col)
    latest_selection_quality = _latest(selection.quality, date_col)
    latest_exp = _latest(exposures, date_col)
    latest_contrib = _latest(contributions, date_col)
    latest_sector = _latest(sector_detail, date_col)

    selection_summary_cols = [
        "factor", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group",
        "selected_model", "best_raw_model", "selection_reason_code", "selection_reason_jp",
        "selected_primary_metric", "best_primary_metric", "best_standard_error",
        "one_se_threshold", "selected_delta_from_best", "selected_complexity",
        "positive_rate", "rank_ic_ir", "mean_top_bottom_spread", "count_periods",
        "adoption_mean_rank_ic_pass", "adoption_positive_rate_pass",
        "adoption_periods_pass", "adopted",
    ]
    selection_summary_view = run.selection[[
        c for c in selection_summary_cols if c in run.selection.columns
    ]].copy()

    candidate_summary_cols = [
        "factor", "Factor_Name_JP", "Factor_Name_EN", "Factor_Group",
        "model", "mean_rank_ic", "rank_ic_se", "rank_ic_ir", "positive_rate",
        "mean_pearson_ic", "mean_top_bottom_spread", "count_periods",
        "primary_metric_rank", "complexity", "best_raw_model", "best_primary_metric",
        "one_se_threshold", "delta_from_best", "within_one_se", "selected",
        "selected_model", "selection_reason_jp", "evaluation_periods_pass", "adopted",
    ]
    candidate_summary_view = run.selection_detail[[
        c for c in candidate_summary_cols if c in run.selection_detail.columns
    ]].sort_values(["factor", "primary_metric_rank"]).copy()

    summary_tables = {
        "Executive_Summary": executive,
        "Representative_Universe": latest_selection.sort_values(
            ["index_name", "sector", "selection_weight"], ascending=[True, True, False]
        ) if not latest_selection.empty else latest_selection,
        "Universe_Selection_Quality": latest_selection_quality,
        "Index_Scores_Latest": latest_index,
        "Model_Accuracy_Summary": pd.concat([
            accuracy_overall.assign(index_name="ALL"),
            accuracy_by_index.assign(scope="per_index"),
        ], ignore_index=True, sort=False),
        "Futures_Risk_Latest": risk_latest,
        "Index_Factor_Exposure": latest_exp,
        "Index_Group_Contribution": latest_contrib,
        "Index_Breadth": latest_sector,
        "Factor_Model_Selection": selection_summary_view,
        "Factor_Model_Candidate_Summary": candidate_summary_view,
        "Factor_Model_Methodology": _model_selection_methodology(config),
        "Factor_Bin_Summary": bins.summary,
        "Factor_Bin_Factor_Summary": bins.factor_summary,
        "Group_Weight_Latest": _latest(group_weight_all, "date"),
        "Group_Scores_Latest": _latest(group_all, date_col),
        "Stock_Scores_Latest": latest_stock,
        "Data_Quality": quality,
        "Input_Index_Coverage": validation_index,
        "Factor_Master_Used": factor_master,
        "Group_Settings_Used": group_settings,
        "Resolved_Factor_Settings": resolved_factor_settings(factor_master, config),
        "Config_Validation": data.factor_settings.validation,
        "Config": pd.DataFrame(),
    }

    history_tables = {
        "Universe_Selection_Quality_History": selection.quality,
        "Futures_Risk_History": risk_history,
        "Index_Score_History": index_scores,
        "Index_Factor_History": exposures,
        "Model_Accuracy_History": accuracy_history,
        "Model_Accuracy_Detail": accuracy_detail,
        "Universe_Sector_Allocation": selection.sector_allocation,
        "Universe_Selection_History": selection.history,
        "Factor_Bin_By_Date": bins.by_date,
        "Factor_Performance": run.metrics_by_date,
        "Factor_Coefficients": run.coefficients,
        "Factor_IC_History": factor_ic_history,
        "Group_Weight_History": group_weight_all,
        "PCA_Loading_History": group_weight_all[
            group_weight_all.get("method", pd.Series(dtype=str)).eq("pca")
        ].copy(),
        "Group_Score_History": group_all,
        "Index_Group_History": contributions,
        "Composite_Coefficients": composite_coefs,
        "Stock_Score_History": stock_all,
    }

    # Apply per-summary-sheet switches before writing.
    summary_tables = {
        name: df for name, df in summary_tables.items()
        if name == "Config" or _summary_sheet_enabled(config, name)
    }

    outputs: dict[str, Path] = {}
    output_rows: list[dict] = []

    # History Excel files: one data type per workbook.
    history_cfg = config["report"].get("history_excel", {})
    history_enabled = bool(history_cfg.get("enabled", True))
    history_dir = ensure_dir(output_dir / history_cfg.get("output_subdir", "history"))
    table_switches = history_cfg.get("tables", {})
    for table_name, df in history_tables.items():
        enabled = bool(history_enabled and table_switches.get(table_name, False))
        filename = HISTORY_FILENAMES.get(table_name, f"{table_name.lower()}.xlsx")
        path = history_dir / filename
        if enabled:
            create_history_workbook(
                path,
                table_name,
                df,
                HISTORY_DESCRIPTIONS.get(table_name, table_name),
                config,
            )
            outputs[f"history_{table_name}"] = path
        elif path.exists():
            path.unlink()
        output_rows.append({
            "output_id": table_name,
            "category": "history_excel",
            "enabled": enabled,
            "generated": bool(enabled and path.exists()),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "relative_path": str(path.relative_to(output_dir)),
            "description": HISTORY_DESCRIPTIONS.get(table_name, table_name),
        })

    # PDFs with report-level switches.
    pdf_cfg = config["report"].get("pdf", {})
    pdf_enabled = bool(pdf_cfg.get("enabled", config["report"].get("create_pdf", True)))
    pdf_switches = pdf_cfg.get("reports", {})
    pdf_args = {
        "factor_scatter": (prep.panel, factor_map, config),
        "factor_bin": (bins.summary, bins.factor_summary, config),
        "factor_model_selection": (run.selection_detail, run.selection, config),
        "factor_model_performance": (run.metrics_by_date, run.selection, config),
        "index_factor_exposure": (index_scores, exposures, contributions, config),
        "index_factor_trends": (exposures, contributions, config),
        "model_accuracy": (accuracy_history, accuracy_detail, accuracy_by_index, config),
        "universe_selection": (selection.quality, selection.sector_allocation, config),
        "futures_risk": (risk_latest, risk_history, corr, config),
    }
    for report_id, (output_key, filename, function_name) in PDF_REPORTS.items():
        enabled = bool(pdf_enabled and pdf_switches.get(report_id, True))
        path = output_dir / filename
        if enabled:
            path = _run_pdf_job(
                base_dir, output_dir, config, output_key, filename, function_name, pdf_args[report_id]
            )
            outputs[output_key] = path
        elif path.exists():
            path.unlink()
        output_rows.append({
            "output_id": report_id,
            "category": "pdf",
            "enabled": enabled,
            "generated": bool(enabled and path.exists()),
            "rows": None,
            "columns": None,
            "relative_path": path.name,
            "description": PDF_DESCRIPTIONS[report_id],
        })

    # Lightweight summary workbook is generated last so its Output_Manifest reflects actual files.
    summary_cfg = config["report"].get("summary_excel", {})
    summary_enabled = bool(summary_cfg.get("enabled", config["report"].get("create_excel", True)))
    summary_path = output_dir / summary_cfg.get("filename", "analysis_summary.xlsx")
    output_rows.append({
        "output_id": "summary_excel",
        "category": "summary_excel",
        "enabled": summary_enabled,
        "generated": summary_enabled,
        "rows": sum(len(v) for v in summary_tables.values()),
        "columns": None,
        "relative_path": summary_path.name,
        "description": "最新値・要約・設定だけを格納する軽量サマリーブック",
    })
    summary_tables["Output_Manifest"] = pd.DataFrame(output_rows)
    if summary_enabled:
        create_excel_report(summary_path, summary_tables, config)
        outputs["summary_excel"] = summary_path
    elif summary_path.exists():
        summary_path.unlink()

    validation_index.to_csv(output_dir / "index_input_coverage.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Pipeline completed. Outputs: %s", outputs)
    return outputs
