from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .config_loader import load_config
from .evaluation import evaluate_scenarios
from .font_support import setup_japanese_matplotlib_from_config
from .feature_engineering import add_forward_return, generate_derived_features
from .io import ensure_output_dirs, read_inputs
from .master import (
    apply_layer2_excel_settings,
    apply_layer3_excel_settings,
    parse_master,
    validate_data_columns,
)
from .core_reporting import write_quintile_pdf, write_scenario_comparison_pdf
from .scenarios import build_scenarios
from .v13_reporting import (
    write_aggregate_factor_diagnostics_excel,
    write_aggregate_factor_diagnostics_pdf,
    write_analysis_summary_v13,
    write_country_factor_score_trends_excel,
    write_country_factor_score_trends_pdf,
    write_factor_return_weight_diagnostics_excel,
    write_factor_return_weight_diagnostics_pdf,
    write_layer3_model_diagnostics_excel,
    write_layer3_model_diagnostics_pdf,
    write_model_parameter_summary_excel,
    write_scenario_excels_v13,
)


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Raw Factor中心のv0.13パイプラインを実行する。"""
    started = perf_counter()

    def progress(message: str) -> None:
        print(f"[v0.13.1 | {perf_counter() - started:8.1f}s] {message}", flush=True)

    progress("パイプライン開始")
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    font_family, font_path = setup_japanese_matplotlib_from_config(config, root)
    progress(f"日本語PDFフォント確認完了: {font_family} ({font_path})")
    data, sheets, _ = read_inputs(config, root)
    parsed = parse_master(sheets)
    config = apply_layer2_excel_settings(config, parsed.get("layer2_settings", {}))
    config = apply_layer3_excel_settings(config, parsed.get("layer3_settings", {}))
    validate_data_columns(data, config["columns"], parsed["metas"])
    progress(f"入力読込完了: rows={len(data):,}")

    data = add_forward_return(data, config)
    data, all_metas, lineage = generate_derived_features(
        data,
        config,
        parsed["metas"],
        parsed.get("feature_control"),
        parsed.get("derived_rules"),
    )
    output_dirs = ensure_output_dirs(config, root)
    progress(f"前処理準備完了: raw factors={len(parsed['metas']):,}, all factors={len(all_metas):,}")

    scenarios, layer3, diagnostics = build_scenarios(
        data,
        config,
        parsed["metas"],
        all_metas,
        parsed.get("group_methods", {}),
        parsed.get("country_region_map"),
        parsed.get("sector_group_map"),
        parsed.get("sector_factor_interaction"),
    )
    progress(f"N00-N07シナリオ構築完了: scenarios={len(scenarios)}")

    summary, quintiles, rank_ic, common_quintiles, common_rank_ic = evaluate_scenarios(scenarios, config)
    progress("共通OOS評価完了")

    root_out = output_dirs["root"]
    pdf_cfg = config["outputs"].get("pdf", {})
    if pdf_cfg.get("quintile_cumulative_returns", True):
        write_quintile_pdf(quintiles, root_out / "quintile_cumulative_returns.pdf", config)
    if pdf_cfg.get("scenario_comparison", True):
        write_scenario_comparison_pdf(
            summary,
            quintiles,
            rank_ic,
            common_quintiles,
            common_rank_ic,
            root_out / "scenario_comparison.pdf",
            config,
        )
    if pdf_cfg.get("factor_return_weight_diagnostics", True):
        write_factor_return_weight_diagnostics_pdf(root_out / "factor_return_weight_diagnostics.pdf", diagnostics)
    if pdf_cfg.get("aggregate_factor_diagnostics", True):
        write_aggregate_factor_diagnostics_pdf(root_out / "aggregate_factor_diagnostics.pdf", data, diagnostics, config)
    if pdf_cfg.get("layer3_model_diagnostics", True):
        write_layer3_model_diagnostics_pdf(root_out / "layer3_model_diagnostics.pdf", data, diagnostics, summary, config)
    if pdf_cfg.get("country_factor_score_trends", True):
        write_country_factor_score_trends_pdf(root_out / "country_factor_score_trends.pdf", data, diagnostics, config)

    if config["outputs"].get("analysis_summary_xlsx", True):
        write_analysis_summary_v13(
            root_out / "analysis_summary.xlsx",
            summary,
            quintiles,
            rank_ic,
            common_quintiles,
            common_rank_ic,
            diagnostics,
            config,
        )
    if config["outputs"].get("factor_return_weight_diagnostics_xlsx", True):
        write_factor_return_weight_diagnostics_excel(root_out / "factor_return_weight_diagnostics.xlsx", diagnostics)
    if config["outputs"].get("aggregate_factor_diagnostics_xlsx", True):
        write_aggregate_factor_diagnostics_excel(root_out / "aggregate_factor_diagnostics.xlsx", data, diagnostics, config)
    if config["outputs"].get("layer3_model_diagnostics_xlsx", True):
        write_layer3_model_diagnostics_excel(root_out / "layer3_model_diagnostics.xlsx", data, scenarios, diagnostics, summary, config)
    if config["outputs"].get("country_factor_score_trends_xlsx", True):
        write_country_factor_score_trends_excel(root_out / "country_factor_score_trends.xlsx", data, diagnostics, config)
    if config["outputs"].get("model_parameter_summary_xlsx", True):
        write_model_parameter_summary_excel(root_out / "model_parameter_summary.xlsx", diagnostics, parsed["metas"])

    write_scenario_excels_v13(scenarios, output_dirs["patterns"], config)
    progress(f"全処理完了: output={root_out}")

    return {
        "root": root,
        "output_dirs": output_dirs,
        "summary": summary,
        "scenario_count": len(scenarios),
        "layer3_scopes": list(layer3),
        "primary_scope": config["layer3"].get("primary_scope"),
        "feature_lineage": lineage,
    }
