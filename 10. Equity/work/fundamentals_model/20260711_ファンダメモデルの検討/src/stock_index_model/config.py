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
        },
        "constituents_file": "data/input/index_constituents.xlsx",
        "sector_weights_file": "data/input/index_sector_weights.xlsx",
        "sector_weights_sheet": "sector_weights",
        "futures_file": "data/input/futures_returns.xlsx",
        "frequency": "monthly",
        "futures_sheet_map": {
            "monthly": "monthly_returns",
            "weekly": "weekly_returns",
        },
        "ignore_constituent_sheets": ["README", "Readme", "readme"],
        "index_country_map": {},
    },
    "columns": {
        "date": "date",
        "isin": "ISIN",
        "stock_return": "stock_return",
        "sector": "sector",
        "country": "country",
        "market_cap": "market_cap",
        "constituent_weight": "constituent_weight",
        "effective_date": "effective_date",
        "sector_weights_sector": "sector",
    },
    "target": {
        "stock_return_alignment": "contemporaneous_to_forward",
        "stock_horizon_periods": 1,
        "futures_return_alignment": "contemporaneous_to_forward",
        "futures_horizon_periods": 1,
    },
    "factors": {
        "reject_unknown_factor_columns": False,
        "require_all_configured_factors": True,
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
        "minimum_bin_observations": 10,
        "piecewise_knot": 0.0,
        "scatter_max_points": 4000,
        "scatter_plots_per_page": 4,
        "bin_plots_per_page": 4,
        "index_trends_per_page": 4,
        "index_trend_max_factors": 6,
        "rolling_accuracy_window": {"monthly": 12, "weekly": 26},
    },
    "model": {
        "candidate_models": ["linear", "piecewise", "quadratic", "combined_ridge"],
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "training_window_periods": 60,
        "minimum_train_periods": 24,
        "date_equal_weighting": True,
        "model_selection": {
            # まずOOS平均RankICが最大のモデルをbest_raw_modelとします。
            # one_se_rule=Trueの場合、best - one_se_multiplier × bestの標準誤差以上を
            # 「統計的にほぼ同等」とみなし、その中から最も単純なモデルを選びます。
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
        # 後方互換用。model_selection.one_se_ruleが優先されます。
        "one_se_rule": True,
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
    "universe_selection": {
        "enabled": True,
        "key_column": "country",
        "stock_key_value_map": {},
        "index_key_map": {},
        "target_count_by_key": {},
        "target_count_by_index": {},
        "candidate_mode": "constituent_then_country_fallback",
        "lookback_periods": {"monthly": 36, "weekly": 52},
        "minimum_history_periods": {"monthly": 12, "weekly": 26},
        "rebalance_every_periods": 3,
        "exclude_current_period_from_selection": True,
        "minimum_factor_coverage": 0.50,
        "minimum_return_coverage": 0.60,
        "constituent_bonus": 0.10,
        "selection_score": {
            "correlation_weight": 0.55,
            "coverage_weight": 0.20,
            "market_cap_weight": 0.15,
            "constituent_weight": 0.10,
            "redundancy_penalty": 0.20,
        },
        "weight_optimization": {
            "enabled": True,
            "ridge_penalty": 1e-4,
            "min_stock_weight": 1e-6,
            "max_stock_weight": 0.08,
            "solver_maxiter": 500,
            "solver_ftol": 1e-10,
        },
    },
    "aggregation": {
        "method": "selected_universe_weighted",
        "normalize_sector_weights": True,
        "minimum_index_weight_coverage": 0.75,
        "minimum_constituent_count": 5,
        "positive_alpha_threshold": 0.0,
    },
    "risk": {
        "window_periods": {"monthly": 36, "weekly": 52},
        "annualization": {"monthly": 12, "weekly": 52},
        "var_confidence": 0.95,
    },
    "report": {
        "pdf_timeout_seconds": 180,
        "line_plots_per_page": 4,
        "summary_excel": {
            "enabled": True,
            "filename": "analysis_summary.xlsx",
            # Falseにしたシートはサマリーブックに出力しません。
            "sheets": {},
        },
        "history_excel": {
            "enabled": True,
            "output_subdir": "history",
            # Excelの1シート上限より余裕を持たせ、超過時はData_001等へ分割します。
            "max_rows_per_sheet": 800000,
            "tables": {
                "Universe_Selection_Quality_History": True,
                "Futures_Risk_History": True,
                "Index_Score_History": True,
                "Index_Factor_History": True,
                "Model_Accuracy_History": True,
                "Model_Accuracy_Detail": True,
                "Universe_Sector_Allocation": True,
                "Universe_Selection_History": False,
                "Factor_Bin_By_Date": True,
                "Factor_Performance": True,
                "Factor_Coefficients": True,
                "Factor_IC_History": True,
                "Group_Weight_History": True,
                "PCA_Loading_History": True,
                "Group_Score_History": False,
                "Index_Group_History": True,
                "Composite_Coefficients": True,
                "Stock_Score_History": False,
            },
        },
        "pdf": {
            "enabled": True,
            "reports": {
                "factor_scatter": True,
                "factor_bin": True,
                "factor_model_selection": True,
                "factor_model_performance": True,
                "index_factor_exposure": True,
                "index_factor_trends": True,
                "model_accuracy": True,
                "universe_selection": True,
                "futures_risk": True,
            },
        },
        # 旧設定との後方互換用。新しいsummary_excel/pdf設定が優先されます。
        "create_pdf": True,
        "create_excel": True,
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
    spec = importlib.util.spec_from_file_location("stock_index_user_config", path)
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
