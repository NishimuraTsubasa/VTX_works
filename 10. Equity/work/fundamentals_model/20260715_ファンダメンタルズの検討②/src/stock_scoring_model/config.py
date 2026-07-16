from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "output_dir": "outputs",
        "random_seed": 42,
        "log_level": "INFO",
    },
    "data": {
        "factors_file": "data/input/factors_and_returns.xlsx",
        "factors_sheet": "data",
        "factor_master_file": "data/input/factor_master.xlsx",
        "factor_master_sheet_map": {
            "factor_master": "Factor_Master",
            "group_settings": "Group_Settings",
            "method_params": "Group_Method_Params",
            "feature_control": "Feature_Engineering_Control",
            "derived_rules": "Derived_Feature_Rules",
        },
        "frequency": "monthly",  # monthly / weekly
    },
    "columns": {
        "date": "date",
        "isin": "ISIN",
        "stock_return": "stock_return",
        "sector": "sector",
        "country": "country",
        "currency": "currency",
        "market_cap": "market_cap",
    },
    "target": {
        # stock_return が時点tのリターンなら、内部で1期先の forward_return を作ります。
        "stock_return_alignment": "contemporaneous_to_forward",
        "stock_horizon_periods": 1,
    },
    "factors": {
        "reject_unknown_factor_columns": False,
        "require_all_configured_factors": True,
    },
    "feature_engineering": {
        "enabled": True,
        # ExcelのFeature_Engineering_Controlに指定がない場合の共通値。
        "defaults": {
            "enabled": False,
            "generation_mode": "selected",  # all / selected
            "include_raw": True,
        },
        # 派生特徴量はスコア時点tの行に格納します。Source_Lag=1なら、
        # t-1までの情報だけを使い、t+1リターンと対応するため、実質2時点の間隔です。
        "strict_lag_alignment": True,
    },
    "preprocessing": {
        "winsorize": {
            "enabled": True,
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "minimum_observations": 20,
        },
        "neutralization": {
            "enabled": True,
            "categorical": ["sector", "country"],
            "numeric": ["log_market_cap"],
            "minimum_observations": 30,
            "ridge_alpha": 1e-6,
        },
        "rank_transform": "gaussian",
        "gaussian_clip": 3.0,
        "fill_missing_value": 0.0,
        "add_missing_indicators": False,
    },
    "diagnostics": {
        "quantile_bins": 5,
        "minimum_bin_observations": 8,
        "piecewise_knot": 0.0,
        "scatter_max_points": 1200,
        "scatter_plots_per_page": 4,
        "bin_plots_per_page": 4,
        "scenario_plots_per_page": 4,
        "rolling_rank_ic_window": {"monthly": 12, "weekly": 26},
        "rolling_accuracy_window": {"monthly": 12, "weekly": 26},
        "annualization": {"monthly": 12, "weekly": 52},
    },
    "model": {
        "training_window_periods": 24,
        "minimum_train_periods": 12,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "candidate_models": ["linear", "piecewise", "quadratic", "combined_ridge"],
        "model_selection": {
            "primary_metric": "mean_rank_ic",
            "standard_error_column": "rank_ic_se",
            "one_se_rule": True,
            "one_se_multiplier": 1.0,
            "minimum_evaluation_periods": 12,
            "complexity_order": {
                "linear": 1,
                "piecewise": 2,
                "quadratic": 2,
                "combined_ridge": 3,
            },
            "tie_breaker": "higher_primary_metric",
        },
        "factor_adoption": {
            "minimum_mean_rank_ic": -1.0,
            "minimum_positive_rate": 0.0,
        },
        "group_method_defaults": {
            "ic_type": "spearman",
            "ic_mean_method": "ewm",
            "ewm_halflife": 12,
            "positive_ic_only": True,
            "correlation_shrinkage": 0.20,
            "pca_sign_alignment": "group_average",
        },
        "composite_method": "ridge",
        "composite_minimum_train_periods": 12,
    },
    "report": {
        "pdf_timeout_seconds": 180,
        "line_plots_per_page": 4,
        "summary_excel": {
            "enabled": True,
            "filename": "analysis_summary.xlsx",
            "sheets": {
                "Scenario_Comparison": True,
                "Scenario_Quintile_Summary": True,
                "Factor_Model_Selection": True,
                "Factor_Model_Candidate_Summary": True,
                "Factor_Model_Methodology": True,
                "Factor_Bin_Factor_Summary": True,
                "Group_Weight_Latest": True,
                "Stock_Scores_Latest": True,
                "Data_Quality": True,
                "Feature_Lineage": True,
                "Feature_Engineering_Control": True,
                "Derived_Feature_Rules": True,
            },
        },
        "history_excel": {
            "enabled": True,
            "output_subdir": "history",
            "max_rows_per_sheet": 800000,
            "tables": {
                "Scenario_RankIC_History": True,
                "Scenario_Quintile_Return_History": True,
                "Scenario_LongShort_History": True,
                "Factor_Bin_By_Date": True,
                "Factor_Performance": True,
                "Factor_Coefficients": True,
                "Factor_IC_History": True,
                "Group_Weight_History": True,
                "PCA_Loading_History": True,
                "Composite_Coefficients": True,
                "Group_Score_History": False,
                "Stock_Score_History": False,
            },
        },
        # 1パターンにつき1つのExcel。列数を抑えるためSubScoreとFactorScoreは縦持ちです。
        "scenario_excel": {
            "enabled": True,
            "output_subdir": "stock_score_patterns",
            "date_scope": "latest",  # all / latest / selected
            "selected_dates": [],
            "max_rows_per_sheet": 500000,
            "include_sub_scores": True,
            "include_factor_scores": True,
            "include_factor_map": True,
            "include_scenario_config": True,
            "scenarios": {
                "S00_Current_Direct_EW": True,
                "S01_Missing_Adjusted_EW": True,
                "S02_Winsorized_Direct_EW": True,
                "S03_Neutralized_Direct_EW": True,
                "S04_Hierarchical_Equal_Weight": True,
                "S05_Correlation_Adjusted_IC": True,
                "S06_Selected_Factor_Models": True,
                "S07_Full_OOF_Ridge": True,
            },
        },
        "file_inventory": {
            "enabled": True,
            "filename": "file_inventory.xlsx",
        },
        "pdf": {
            "enabled": True,
            "reports": {
                "factor_scatter": True,
                "factor_bin": True,
                "factor_model_selection": True,
                "factor_model_performance": True,
                "scenario_quintile_cumulative": True,
                "scenario_comparison": True,
            },
        },
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_python_dict(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("stock_score_user_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user = getattr(module, "CONFIG", None)
    if not isinstance(user, dict):
        raise ValueError(f"Python config must define a dict named CONFIG: {path}")
    return user


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path.suffix.lower() != ".py":
        raise ValueError("Config must be a .py file defining CONFIG = {...}.")
    config = deepcopy(DEFAULT_CONFIG)
    user = _load_python_dict(config_path)
    return _deep_update(config, user)
