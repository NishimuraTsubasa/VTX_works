from __future__ import annotations

from pathlib import Path
from typing import Any
from time import perf_counter

import pandas as pd

from .config_loader import load_config
from .evaluation import evaluate_scenarios
from .feature_engineering import add_forward_return, generate_derived_features
from .io import ensure_output_dirs, read_inputs
from .master import apply_layer3_excel_settings, parse_master, validate_data_columns
from .layer3_country_reporting import (
    write_country_diagnostics_excel,
    write_country_diagnostics_pdf,
)
from .model_fit_reporting import (
    write_model_fit_diagnostics_excel,
    write_model_fit_diagnostics_pdf,
)
from .country_factor_score_reporting import (
    write_country_factor_score_excel,
    write_country_factor_score_pdf,
)
from .factor_score_performance_reporting import (
    write_factor_score_performance_excel,
    write_factor_score_performance_pdf,
)
from .reporting import (
    write_analysis_summary,
    write_coefficient_stability_pdf,
    write_layer3_country_diagnostics_pdf,
    write_layer3_diagnostics_excel,
    write_layer3_history_files,
    write_layer3_scope_comparison_pdf,
    write_quintile_pdf,
    write_scenario_comparison_pdf,
    write_scenario_excels,
    write_s07_estimator_comparison_pdf,
    write_s07_estimator_comparison_excel,
    write_sector_factor_interactions_pdf,
)
from .scenarios import build_scenarios


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """3層モデルとS00-S07（OLS/Ridge分岐）の比較を実行する。"""
    pipeline_started = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - pipeline_started
        print(f"[v0.12.6 | {elapsed:8.1f}s] {message}", flush=True)

    progress("パイプライン開始")
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    progress(f"Config読込完了: {config_path}")
    data, sheets, _ = read_inputs(config, root)
    progress(f"入力読込完了: rows={len(data):,}")
    parsed = parse_master(sheets)
    config = apply_layer3_excel_settings(config, parsed.get("layer3_settings", {}))
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
    progress(f"前処理・派生特徴量作成完了: features={len(all_metas):,}")

    progress("S00-S07シナリオ構築開始（最も時間がかかる工程です）")
    scenarios, layer3, diagnostics = build_scenarios(
        data,
        config,
        parsed["metas"],
        all_metas,
        parsed["group_methods"],
        parsed["country_region_map"],
        parsed["sector_group_map"],
        parsed["sector_factor_interaction"],
    )
    progress(f"シナリオ構築完了: scenarios={len(scenarios)}")
    summary, quintiles, rank_ic, common_quintiles, common_rank_ic = evaluate_scenarios(scenarios, config)
    progress("共通OOS評価完了")
    pdf_cfg = config["outputs"].get("pdf", {})
    progress("PDF・Excelレポート生成開始")

    if pdf_cfg.get("quintile_cumulative_returns", True):
        write_quintile_pdf(quintiles, output_dirs["root"] / "quintile_cumulative_returns.pdf", config)
    if pdf_cfg.get("scenario_comparison", True):
        write_scenario_comparison_pdf(summary, quintiles, rank_ic, common_quintiles, common_rank_ic, output_dirs["root"] / "stock_scoring_scenario_comparison.pdf", config)
    if pdf_cfg.get("layer3_scope_comparison", True):
        write_layer3_scope_comparison_pdf(data, layer3, output_dirs["root"] / "layer3_scope_comparison.pdf", config)
    if pdf_cfg.get("layer3_country_diagnostics", True):
        write_layer3_country_diagnostics_pdf(data, layer3, output_dirs["root"] / "layer3_country_diagnostics.pdf", config)
    if pdf_cfg.get("coefficient_stability", True):
        write_coefficient_stability_pdf(layer3, output_dirs["root"] / "coefficient_stability.pdf", config)
    if pdf_cfg.get("sector_factor_interactions", True):
        write_sector_factor_interactions_pdf(layer3, output_dirs["root"] / "sector_factor_interactions.pdf", config)
    if pdf_cfg.get("s07_estimator_comparison", True):
        write_s07_estimator_comparison_pdf(data, diagnostics.get("S07Variants", {}), output_dirs["root"] / "s07_ols_ridge_comparison.pdf", config)
    if pdf_cfg.get("s07_country_diagnostics", True):
        write_country_diagnostics_pdf(
            data,
            diagnostics.get("S07Variants", {}),
            output_dirs["root"] / "s07_country_diagnostics.pdf",
            config,
        )
    if pdf_cfg.get("s06_s07_model_fit_diagnostics", True):
        write_model_fit_diagnostics_pdf(
            data,
            scenarios,
            diagnostics,
            output_dirs["root"] / "s06_s07_model_fit_diagnostics.pdf",
            config,
        )
    if pdf_cfg.get("country_factor_score_trends", True):
        write_country_factor_score_pdf(
            data,
            diagnostics.get("Layer2FactorScores", pd.DataFrame()),
            output_dirs["root"] / "country_factor_score_trends.pdf",
            config,
        )
    if pdf_cfg.get("factor_score_performance_diagnostics", True):
        write_factor_score_performance_pdf(
            data,
            scenarios,
            diagnostics,
            all_metas,
            output_dirs["root"] / "factor_score_performance_diagnostics.pdf",
            config,
        )

    if config["outputs"].get("analysis_summary_xlsx", True):
        write_analysis_summary(
            output_dirs["root"] / "analysis_summary.xlsx",
            summary,
            quintiles,
            rank_ic,
            common_quintiles,
            common_rank_ic,
            diagnostics.get("Layer1Selection", pd.DataFrame()),
            diagnostics.get("Layer2Weights", pd.DataFrame()),
            lineage,
        )
    if config["outputs"].get("layer3_diagnostics_xlsx", True):
        write_layer3_diagnostics_excel(output_dirs["root"] / "layer3_diagnostics.xlsx", data, layer3, config)
    if config["outputs"].get("s07_estimator_comparison_xlsx", True):
        write_s07_estimator_comparison_excel(data, diagnostics.get("S07Variants", {}), output_dirs["root"] / "s07_ols_ridge_comparison.xlsx", config)
    if config["outputs"].get("s07_country_diagnostics_xlsx", True):
        write_country_diagnostics_excel(
            data,
            diagnostics.get("S07Variants", {}),
            output_dirs["root"] / "s07_country_diagnostics.xlsx",
            config,
        )
    if config["outputs"].get("s06_s07_model_fit_diagnostics_xlsx", True):
        write_model_fit_diagnostics_excel(
            data,
            scenarios,
            diagnostics,
            output_dirs["root"] / "s06_s07_model_fit_diagnostics.xlsx",
            config,
        )
    if config["outputs"].get("country_factor_score_trends_xlsx", True):
        write_country_factor_score_excel(
            data,
            diagnostics.get("Layer2FactorScores", pd.DataFrame()),
            output_dirs["root"] / "country_factor_score_trends.xlsx",
            config,
        )
    if config["outputs"].get("factor_score_performance_diagnostics_xlsx", True):
        write_factor_score_performance_excel(
            data,
            scenarios,
            diagnostics,
            all_metas,
            output_dirs["root"] / "factor_score_performance_diagnostics.xlsx",
            config,
        )
    write_scenario_excels(scenarios, output_dirs["patterns"], config)
    write_layer3_history_files(output_dirs["history"], layer3, diagnostics, config)
    progress(f"全処理完了: output={output_dirs['root']}")

    return {
        "root": root,
        "output_dirs": output_dirs,
        "summary": summary,
        "common_oos_start": summary["CommonStartDate"].dropna().min() if not summary.empty else pd.NaT,
        "common_oos_end": summary["CommonEndDate"].dropna().max() if not summary.empty else pd.NaT,
        "scenario_count": len(scenarios),
        "layer3_scopes": list(layer3),
        "primary_scope": config["layer3"].get("primary_scope"),
    }
