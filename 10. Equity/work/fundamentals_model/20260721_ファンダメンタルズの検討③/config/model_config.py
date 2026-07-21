"""個別銘柄スコアリングモデル v0.12.2 のユーザー設定。

個別FAの分類・方向・派生特徴量、国-地域対応、セクターグループ、
交差項の利用可否は data/input/factor_master.xlsx で管理します。
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
        # contemporaneous_to_forward: 行の日付のリターンを1期先へシフト
        # already_forward: 入力済みリターンをそのまま翌期リターンとして使用
        "stock_return_alignment": "contemporaneous_to_forward",
        "stock_horizon_periods": 1,
    },
    "preprocessing": {
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "minimum_cross_section": 20,
        "rank_transform": "gaussian",
        "gaussian_clip": 3.0,
        # S03比較用の従来中立化
        "neutralization_mode": "country_sector_and_size",
        "neutralization_ridge_alpha": 1e-6,
    },
    "layer1": {
        # グローバル単一FAモデル
        "candidate_models": ["linear", "piecewise", "quadratic"],
        "training_window_periods": 36,
        # データ開始が2018年のため、初回OOFを過度に遅らせない18か月を既定値とする。
        "minimum_train_periods": 18,
        "validation_periods": 6,
        "minimum_fit_observations": 200,
        "minimum_validation_observations": 100,
        "piecewise_knot": 0.0,
        "one_se_rule": True,
        "ridge_alpha": 1e-8,
    },
    "layer2": {
        "ic_lookback_periods": 18,
        "ic_minimum_periods": 8,
        "correlation_shrinkage": 0.20,
        "maximum_factor_weight": 0.60,
        "weight_smoothing": 0.50,
        "pca_lookback_periods": 36,
        "pca_minimum_periods": 12,
    },
    "layer3": {
        # country_independent / regional_pooling / hierarchical_partial_pooling
        "primary_scope": "country_independent",
        "comparison_scopes": [
            "country_independent",
            "regional_pooling",
            # 必要時に追加: "hierarchical_partial_pooling"
        ],
        # rolling_pooled / cross_sectional_coefficient_average
        "training_mode": "rolling_pooled",
        "lookback_periods": 36,
        # Layer1 OOF開始後、12か月分のFactorScoreが蓄積したら第3層OOSを開始。
        "minimum_train_periods": 12,
        "minimum_stocks_per_country_date": 10,
        "minimum_stocks_per_sector_group": 5,
        "minimum_training_observations": 150,
        "minimum_factor_score_coverage": 0.50,
        # ridge / ols。S07の比較はs07_variantsで個別指定する。
        "estimator": "ridge",
        "ridge_validation_periods": 6,
        "include_nonlinear_basis": True,
        "nonlinear_basis": ["linear", "piecewise", "quadratic"],
        "piecewise_knot": 0.0,
        "include_sector_group_dummy": True,
        "include_sector_factor_interactions": True,
        # all_interactions / selected_interactions
        "interaction_mode": "selected_interactions",
        "include_country_controls_in_regional": True,
        "demean_target_by_date": True,
        "ridge_alphas": [0.1, 1.0, 10.0],
        # 部分プーリング：国固有補正を地域共通係数より強く縮小
        "country_deviation_penalty_multiplier": 10.0,
        "country_intercept_penalty_multiplier": 2.0,
        "fallback_scope": "regional_pooling",
        # country / global。国別モデルの比較可能性を考え既定はcountry
        "final_score_rank_scope": "country",
        # S07内の比較。OLSとRidgeは同じ線形基底・同じOOS開始条件で比較する。
        "s07_variants": {
            "S07_OLS_Linear": {
                "enabled": True,
                "estimator": "ols",
                "nonlinear_basis": ["linear"],
            },
            "S07_Ridge_Linear": {
                "enabled": True,
                "estimator": "ridge",
                "nonlinear_basis": ["linear"],
            },
            # 現行の柔軟な3基底Ridgeは補助比較。必要時のみ有効化する。
            "S07_Ridge_Flexible": {
                "enabled": False,
                "estimator": "ridge",
                "nonlinear_basis": ["linear", "piecewise", "quadratic"],
            },
        },
    },
    "scenarios": {
        "S00_Current_Direct_EW": True,
        "S01_Missing_Adjusted_EW": True,
        "S02_Winsorized_Direct_EW": True,
        "S03_Neutralized_Direct_EW": True,
        "S04_Hierarchical_Equal_Weight": True,
        "S05_Correlation_Adjusted_IC": True,
        # 第1層 + 第2層までの予測
        "S06_Selected_Factor_Models": True,
        # 第3層はlayer3.s07_variantsでOLS/Ridgeを個別に有効化。
        "S07_OLS_Linear": True,
        "S07_Ridge_Linear": True,
        "S07_Ridge_Flexible": False,
    },
    "binscatter": {
        "enabled": True,
        "factor_codes": ["FA0101", "FA0102", "FA1001", "FA2001"],
        "weighting": "equal",
        "x_stat": "mean",
        "scopes": {
            "all_universe": True,
            "by_country": True,
            "by_country_sector": True,
        },
        "scope_filters": {
            "countries": ["US", "Japan"],
            "sectors": [],
            "max_country_sector_scopes": 4,
        },
        "n_bins": {"all_universe": 20, "by_country": 20, "by_country_sector": 5},
        "minimum_observations_per_period": {
            "all_universe": 80,
            "by_country": 18,
            "by_country_sector": 6,
        },
        "minimum_periods": 12,
        "error_bar": "standard_error",
        "regressions": {"linear": True, "quadratic": True, "broken_stick": True, "broken_stick_knot": "auto"},
        "plots_per_page": 6,
        "show_r_squared": True,
        "show_adjusted_r_squared": False,
        "show_correlations": True,
        "show_top_bottom_spread": True,
    },
    "evaluation": {
        "quintiles": 5,
        "rolling_rank_ic_periods": 12,
        "annualization": 12,
        "transaction_cost_bps_one_way": 0.0,
        # 全シナリオでDate×ISINが共通する純粋OOSサンプルを主比較に使用。
        "common_oos": {
            "enabled": True,
            "universe_mode": "stock_date_intersection",
            "rerank_on_common_universe": True,
            "benchmark_scenario": "S03_Neutralized_Direct_EW",
            "minimum_stocks_per_date": 30,
            "minimum_periods_warning": 24,
        },
    },
    "outputs": {
        "output_dir": "outputs",
        "analysis_summary_xlsx": True,
        "layer3_diagnostics_xlsx": True,
        "s07_estimator_comparison_xlsx": True,
        "scenario_excel": {
            "enabled": True,
            "date_scope": "latest",
            "include_sub_scores": True,
            "include_factor_scores": True,
        },
        "history_excel": {
            "enabled": True,
            "layer1_model_selection": True,
            "layer1_subscore": False,
            "layer2_factor_score": False,
            "layer3_prediction": True,
            "layer3_coefficients": True,
            "layer3_scope_performance": True,
            "sector_interactions": True,
        },
        "pdf": {
            "binscatter_all_universe": True,
            "binscatter_by_country": True,
            "binscatter_by_country_sector": True,
            "quintile_cumulative_returns": True,
            "scenario_comparison": True,
            "layer3_scope_comparison": True,
            "layer3_country_diagnostics": True,
            "coefficient_stability": True,
            "sector_factor_interactions": True,
            "s07_estimator_comparison": True,
        },
    },
}
