"""ユーザーが変更する設定。

ファクター固有の分類・方向・派生特徴量は factor_master.xlsx で管理します。
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
        "neutralize_columns": ["country", "sector", "log_market_cap"],
        "minimum_cross_section": 20,
        "rank_transform": "uniform_0_1",  # uniform_0_1 / gaussian
        "gaussian_clip": 3.0,
    },
    "model": {
        "training_window_periods": 24,
        "minimum_train_periods": 12,
        "ic_lookback_periods": 24,
        "ic_minimum_periods": 8,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "candidate_models": ["linear", "piecewise", "quadratic", "combined_ridge"],
        "one_se_rule": True,
    },
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
    "binscatter": {
        "enabled": True,
        # 空リストならEnabledな全ファクター。派生コードも指定可能。
        "factor_codes": ["FA0101", "FA0102", "FA1001", "FA2001"],
        "weighting": "equal",  # equal / market_cap
        "x_stat": "mean",      # mean / median
        "scopes": {
            "all_universe": True,
            "by_country": True,
            "by_country_sector": True,
        },
        "scope_filters": {
            # 空リストなら全件。サンプルPDFではページ数を抑えるため上限を利用。
            # サンプルPDFはUS/Japanのみ。空リストにすると全対象。
            "countries": ["US", "Japan"],
            "sectors": [],
            "max_country_sector_scopes": 4,
        },
        "n_bins": {
            "all_universe": 20,
            "by_country": 20,
            "by_country_sector": 10,
        },
        "minimum_observations_per_period": {
            "all_universe": 100,
            "by_country": 60,
            "by_country_sector": 15,
        },
        "minimum_periods": 12,
        "error_bar": "standard_error",  # standard_error / ci95 / none
        "regressions": {
            "linear": True,
            "quadratic": True,
            "broken_stick": True,
            "broken_stick_knot": "auto",  # auto / zero / median
        },
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
    },
    "outputs": {
        "output_dir": "outputs",
        "analysis_summary_xlsx": True,
        "scenario_excel": {
            "enabled": True,
            "date_scope": "latest",  # latest / all
            "include_sub_scores": True,
            "include_factor_scores": True,
        },
        "pdf": {
            "binscatter_all_universe": True,
            "binscatter_by_country": True,
            "binscatter_by_country_sector": True,
            "quintile_cumulative_returns": True,
            "scenario_comparison": True,
        },
    },
}
