"""分析設定。

このファイルの CONFIG 辞書だけを編集して分析条件を変更します。
例：USD の代表銘柄数を 200 にする場合は target_count_by_key["USD"] = 200。
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
        "constituents_file": "data/input/index_constituents.xlsx",
        "sector_weights_file": "data/input/index_sector_weights.xlsx",
        "sector_weights_sheet": "sector_weights",
        "futures_file": "data/input/futures_returns.xlsx",
        "frequency": "monthly",  # "monthly" or "weekly"
        "futures_sheet_map": {
            "monthly": "monthly_returns",
            "weekly": "weekly_returns",
        },
        "index_country_map": {
            "JP_INDEX": "Japan",
            "US_INDEX": "United States",
            "EU_INDEX": "Europe",
        },
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
    "factors": {
        "reject_unknown_factor_columns": False,
        "require_all_configured_factors": True,
    },
    "diagnostics": {
        "quantile_bins": 5,
        "minimum_bin_observations": 8,
        "scatter_max_points": 1200,
        "scatter_plots_per_page": 4,
        "bin_plots_per_page": 4,
        "index_trends_per_page": 4,
        "index_trend_max_factors": 6,
        "rolling_accuracy_window": {"monthly": 12, "weekly": 26},
    },
    "model": {
        "training_window_periods": 36,
        "minimum_train_periods": 18,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "candidate_models": ["linear", "piecewise", "quadratic", "combined_ridge"],
        "model_selection": {
            # 1) OOS平均RankICが最大のモデルをbest_raw_modelとする。
            # 2) best - 1標準誤差以上の候補を「ほぼ同等」と判定する。
            # 3) その中で最も単純なモデルを採用する。
            # Linearが多く選ばれる場合、非線形モデルが統計的に明確な改善を示していないことを意味します。
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
        "group_method_defaults": {
            "ic_type": "spearman",
            "ic_mean_method": "ewm",
            "ewm_halflife": 12,
            "positive_ic_only": True,
            "correlation_shrinkage": 0.20,
            "pca_sign_alignment": "group_average",
        },
        "composite_method": "ridge",
    },
    "universe_selection": {
        "enabled": True,
        # key_column の値を stock_key_value_map で任意キーへ変換できます。
        # 例えば country="United States" を "USD" として管理します。
        "key_column": "country",
        "stock_key_value_map": {
            "Japan": "JPY",
            "United States": "USD",
            "Europe": "EUR",
        },
        "index_key_map": {
            "JP_INDEX": "JPY",
            "US_INDEX": "USD",
            "EU_INDEX": "EUR",
        },
        # 実運用では例："USD": 200 と指定できます。
        "target_count_by_key": {
            "JPY": 12,
            "USD": 12,
            "EUR": 12,
        },
        "target_count_by_index": {},
        # constituent_only / country_universe / constituent_then_country_fallback
        "candidate_mode": "constituent_then_country_fallback",
        "lookback_periods": {"monthly": 24, "weekly": 52},
        "minimum_history_periods": {"monthly": 12, "weekly": 26},
        "rebalance_every_periods": 6,
        "exclude_current_period_from_selection": True,
        "minimum_factor_coverage": 0.50,
        "minimum_return_coverage": 0.60,
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
        "minimum_index_weight_coverage": 0.75,
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

        # 最新値・要約・設定だけを格納する軽量なサマリーブックです。
        "summary_excel": {
            "enabled": True,
            "filename": "analysis_summary.xlsx",
            # 個別シートを止めたい場合だけFalseを設定します。未指定はTrueです。
            "sheets": {
                "Executive_Summary": True,
                "Representative_Universe": True,
                "Universe_Selection_Quality": True,
                "Index_Scores_Latest": True,
                "Model_Accuracy_Summary": True,
                "Futures_Risk_Latest": True,
                "Index_Factor_Exposure": True,
                "Index_Group_Contribution": True,
                "Index_Breadth": True,
                "Factor_Model_Selection": True,
                "Factor_Model_Candidate_Summary": True,
                "Factor_Model_Methodology": True,
                "Factor_Bin_Factor_Summary": True,
                "Group_Weight_Latest": True,
                "Stock_Scores_Latest": True,
            },
        },

        # 履歴は種類ごとに1ファイルずつ出力します。
        # 2500銘柄×長期履歴になり得るため、重い個別銘柄履歴は初期値Falseです。
        "history_excel": {
            "enabled": True,
            "output_subdir": "history",
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

        # PDFも種類ごとに出力可否を切り替えられます。
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
    },
}
