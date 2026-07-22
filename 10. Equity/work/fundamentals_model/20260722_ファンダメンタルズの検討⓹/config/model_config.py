"""個別銘柄スコアリングモデル v0.13.1 ユーザー設定。

v0.13では第1層の単一FAリターン回帰を本番スコア生成から削除し、
前処理済みRaw Factor ScoreをQ5-Q1 Factor Return相関で集約する。
"""

CONFIG = {
    "data": {
        "factors_file": "data/input/factors_and_returns.xlsx",
        "factors_sheet": "data",
        "factor_master_file": "data/input/factor_master.xlsx",
        "frequency": "monthly",
    },
    "columns": {
        "date": "date",
        "isin": "ISIN",
        "stock_return": "stock_return",
        "market_cap": "market_cap",
        "currency": "currency",
        "country": "country",
        "sector": "sector",
    },
    "target": {
        "stock_return_alignment": "contemporaneous_to_forward",
        "stock_horizon_periods": 1,
    },
    "preprocessing": {
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "minimum_cross_section": 20,
        "gaussian_clip": 3.0,
        "neutralization_mode": "country_sector_and_size",
        "neutralization_ridge_alpha": 1e-6,
    },
    "layer2": {
        # raw_only / raw_and_derived。初期本線はRaw Factorのみ。
        "factor_universe": "raw_only",
        "winsorize": True,
        "use_neutralized_scores": True,
        # centered_percentile / gaussian / zscore / uniform_0_1
        "raw_score_transform": "centered_percentile",
        # Q5-Q1 Factor Returnの作成単位
        # global / within_country / within_country_sector
        "factor_return_scope": "within_country",
        "country_factor_return_aggregation": "equal_country",
        "factor_return_quantiles": 5,
        "minimum_stocks_per_factor_return_cell": 20,
        # 時点tのウェイトはDate<tのFactor Returnだけで計算する。
        "factor_return_lookback_periods": 36,
        "factor_return_minimum_periods": 12,
        "correlation_shrinkage": 0.30,
        # 1.0=完全EW、0.0=相関最小分散ウェイトのみ
        "equal_weight_blend": 0.50,
        "maximum_factor_weight": 0.50,
        "weight_smoothing": 0.50,
        # 独立だが無効なFAへの過大配分を防ぐ任意フィルター。
        "effectiveness_filter_enabled": False,
        "minimum_mean_factor_return": 0.0,
        "minimum_factor_return_positive_rate": 0.45,
    },
    "layer3": {
        "primary_scope": "country_independent",
        "comparison_scopes": ["country_independent", "regional_pooling"],
        "training_mode": "rolling_pooled",
        "lookback_periods": 36,
        "minimum_train_periods": 12,
        "minimum_stocks_per_country_date": 10,
        "minimum_stocks_per_sector_group": 5,
        "minimum_training_observations": 150,
        "minimum_factor_score_coverage": 0.50,
        "ridge_validation_periods": 6,
        "ridge_alphas": [0.1, 1.0, 10.0],
        "piecewise_knot": 0.0,
        "interaction_mode": "selected_interactions",
        "include_country_controls_in_regional": True,
        "demean_target_by_date": True,
        "standardize_continuous_features": True,
        "standardize_feature_types": ["factor_basis", "sector_factor_interaction", "country_deviation"],
        "country_deviation_penalty_multiplier": 10.0,
        "country_intercept_penalty_multiplier": 2.0,
        "fallback_scope": "regional_pooling",
        "final_score_rank_scope": "country",
        # 第3層は単純モデルから順に追加効果を確認する。
        "variants": {
            "N05_L3_OLS_MainEffects": {
                "enabled": True,
                "estimator": "ols",
                "nonlinear_basis": ["linear"],
                "include_sector_group_dummy": False,
                "include_sector_factor_interactions": False,
            },
            "N06_L3_Ridge_MainEffects": {
                "enabled": True,
                "estimator": "ridge",
                "nonlinear_basis": ["linear"],
                "include_sector_group_dummy": False,
                "include_sector_factor_interactions": False,
            },
            "N07_L3_Ridge_SelectedInteractions": {
                "enabled": True,
                "estimator": "ridge",
                "nonlinear_basis": ["linear"],
                # 階段状予測を避けるためSector Dummy主効果は既定で入れない。
                "include_sector_group_dummy": False,
                "include_sector_factor_interactions": True,
            },
        },
    },
    "scenarios": {
        "N00_Direct_RawScore_EW": True,
        "N01_Hierarchical_FactorCount_EW": True,
        "N02_Hierarchical_Group_EW": True,
        "N03_FactorReturn_Correlation": True,
        "N04_FactorReturn_Correlation_ShrunkEW": True,
        "N05_L3_OLS_MainEffects": True,
        "N06_L3_Ridge_MainEffects": True,
        "N07_L3_Ridge_SelectedInteractions": True,
    },
    "evaluation": {
        "quintiles": 5,
        "rolling_rank_ic_periods": 12,
        "annualization": 12,
        "transaction_cost_bps_one_way": 0.0,
        "common_oos": {
            "enabled": True,
            "universe_mode": "stock_date_intersection",
            "rerank_on_common_universe": True,
            "benchmark_scenario": "N00_Direct_RawScore_EW",
            "minimum_stocks_per_date": 30,
            "minimum_periods_warning": 24,
        },
    },
    "diagnostics": {
        "factor_score_quantiles": 5,
        "rolling_rank_ic_periods": 12,
        "calibration_bins": 10,
        "country_factor_weighting": "equal",
        "country_factor_trailing_z_periods": 36,
        "country_factor_minimum_z_periods": 12,
    },
    "binscatter": {
        "enabled": True,
        "factor_codes": ["FA0101", "FA0102", "FA1001", "FA2001"],
        "weighting": "equal",
        "x_stat": "mean",
        "scopes": {"all_universe": True, "by_country": True, "by_country_sector": True},
        "scope_filters": {"countries": ["US", "Japan"], "sectors": [], "max_country_sector_scopes": 4},
        "n_bins": {"all_universe": 20, "by_country": 20, "by_country_sector": 5},
        "minimum_observations_per_period": {"all_universe": 80, "by_country": 18, "by_country_sector": 6},
        "minimum_periods": 12,
        "error_bar": "standard_error",
        "regressions": {"linear": True, "quadratic": True, "broken_stick": True, "broken_stick_knot": "auto"},
        "plots_per_page": 6,
        "show_r_squared": True,
        "show_adjusted_r_squared": False,
        "show_correlations": True,
        "show_top_bottom_spread": True,
    },
    "reporting": {
        # 空欄ならWindows/macOS/Linuxの日本語フォントを自動検出する。
        # 自動検出に失敗する場合のみ、実行PC上のフォント絶対パスを指定する。
        # 例: r"C:\\Windows\\Fonts\\YuGothM.ttc"
        "japanese_font_path": "",
    },
    "outputs": {
        "output_dir": "outputs",
        "analysis_summary_xlsx": True,
        "factor_return_weight_diagnostics_xlsx": True,
        "aggregate_factor_diagnostics_xlsx": True,
        "layer3_model_diagnostics_xlsx": True,
        "country_factor_score_trends_xlsx": True,
        "model_parameter_summary_xlsx": True,
        "scenario_excel": {"enabled": True, "date_scope": "latest", "include_raw_factor_scores": False, "include_factor_scores": True},
        "pdf": {
            "quintile_cumulative_returns": True,
            "scenario_comparison": True,
            "factor_return_weight_diagnostics": True,
            "aggregate_factor_diagnostics": True,
            "layer3_model_diagnostics": True,
            "country_factor_score_trends": True,
            "binscatter_all_universe": True,
            "binscatter_by_country": True,
            "binscatter_by_country_sector": True,
        },
    },
}
