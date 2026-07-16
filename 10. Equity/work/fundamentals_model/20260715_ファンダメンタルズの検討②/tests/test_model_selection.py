import pandas as pd

from stock_scoring_model.config import DEFAULT_CONFIG
from stock_scoring_model.modeling import select_models


def test_one_se_rule_prefers_simpler_model_within_threshold():
    summary = pd.DataFrame([
        {"factor": "FA0001", "model": "linear", "mean_rank_ic": 0.040, "rank_ic_se": 0.010, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0001", "model": "piecewise", "mean_rank_ic": 0.045, "rank_ic_se": 0.010, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0001", "model": "quadratic", "mean_rank_ic": 0.050, "rank_ic_se": 0.015, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0001", "model": "combined_ridge", "mean_rank_ic": 0.048, "rank_ic_se": 0.012, "count_periods": 24, "positive_rate": 0.60},
    ])
    config = DEFAULT_CONFIG.copy()
    selection, detail = select_models(summary, config)
    row = selection.iloc[0]
    assert row["best_raw_model"] == "quadratic"
    assert row["selected_model"] == "linear"
    assert row["selection_reason_code"] == "ONE_SE_SIMPLER_MODEL"
    assert detail.loc[detail["model"].eq("linear"), "within_one_se"].iloc[0]


def test_non_linear_selected_when_linear_below_one_se_threshold():
    summary = pd.DataFrame([
        {"factor": "FA0002", "model": "linear", "mean_rank_ic": 0.010, "rank_ic_se": 0.005, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0002", "model": "piecewise", "mean_rank_ic": 0.049, "rank_ic_se": 0.005, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0002", "model": "quadratic", "mean_rank_ic": 0.050, "rank_ic_se": 0.005, "count_periods": 24, "positive_rate": 0.60},
        {"factor": "FA0002", "model": "combined_ridge", "mean_rank_ic": 0.052, "rank_ic_se": 0.005, "count_periods": 24, "positive_rate": 0.60},
    ])
    config = DEFAULT_CONFIG.copy()
    selection, _ = select_models(summary, config)
    # Piecewise and quadratic are within one SE of Combined Ridge. Linear is outside
    # the threshold, so the higher-RankIC candidate among complexity-2 models is selected.
    assert selection.iloc[0]["selected_model"] == "quadratic"
