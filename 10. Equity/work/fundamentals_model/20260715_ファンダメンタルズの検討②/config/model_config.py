"""個別銘柄スコアリングモデルの分析設定。

この CONFIG 辞書と data/input/factor_master.xlsx を編集して条件を変更します。
"""

CONFIG = {
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
