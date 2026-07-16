from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import load_config
from .evaluation import evaluate_scenarios
from .feature_engineering import add_forward_return, generate_derived_features
from .io import ensure_output_dirs, read_inputs
from .master import parse_master, validate_data_columns
from .reporting import (
    write_analysis_summary,
    write_quintile_pdf,
    write_scenario_comparison_pdf,
    write_scenario_excels,
)
from .scenarios import build_scenarios


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """個別銘柄スコアのシナリオ比較・5分位評価を実行する。

    Time-averaged binscatterはscripts/run_binscatter.pyで別プロセスとして実行する。
    """
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    data, sheets = read_inputs(config, root)
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

    scenarios = build_scenarios(data, config, parsed["metas"], all_metas)
    summary, quintiles, rank_ic = evaluate_scenarios(scenarios, config)

    pdf_cfg = config["outputs"].get("pdf", {})
    if pdf_cfg.get("quintile_cumulative_returns", True):
        write_quintile_pdf(quintiles, output_dirs["root"] / "quintile_cumulative_returns.pdf", config)
    if pdf_cfg.get("scenario_comparison", True):
        write_scenario_comparison_pdf(
            summary,
            quintiles,
            rank_ic,
            output_dirs["root"] / "stock_scoring_scenario_comparison.pdf",
            config,
        )

    if config["outputs"].get("analysis_summary_xlsx", True):
        write_analysis_summary(
            output_dirs["root"] / "analysis_summary.xlsx",
            summary,
            quintiles,
            rank_ic,
            pd.DataFrame(),
            pd.DataFrame(),
            lineage,
        )
    write_scenario_excels(scenarios, output_dirs["patterns"], config)

    return {
        "root": root,
        "output_dirs": output_dirs,
        "summary": summary,
        "scenario_count": len(scenarios),
    }
