from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.layer1_oof import generate_layer1_oof_subscores
from stock_scoring_model.master import FactorMeta


def test_layer1_returns_coefficients_and_fit_history():
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    rows = []
    for date_idx, date in enumerate(dates):
        for stock in range(30):
            x = (stock - 15) / 10 + 0.05 * date_idx
            rows.append({"date": date, "x": x, "NextMonthReturn": 0.01 * x + 0.001 * np.sin(stock)})
    data = pd.DataFrame(rows)
    scores = pd.DataFrame({"FA0101": data["x"]}, index=data.index)
    config = {
        "columns": {"date": "date"},
        "layer1": {
            "training_window_periods": 8,
            "minimum_train_periods": 5,
            "validation_periods": 2,
            "minimum_fit_observations": 60,
            "minimum_validation_observations": 30,
            "candidate_models": ["linear", "piecewise", "quadratic"],
            "piecewise_knot": 0.0,
            "ridge_alpha": 1e-8,
            "one_se_rule": True,
        },
    }
    metas = {"FA0101": FactorMeta(code="FA0101", group="Value", direction=1, base_weight=1.0)}
    pred, selection, coef, fit = generate_layer1_oof_subscores(data, scores, metas, config)
    assert pred["FA0101"].notna().any()
    assert not selection.empty
    assert {"Term", "Coefficient", "Intercept"}.issubset(coef.columns)
    assert {"TrainR2", "ValidationR2", "ValidationMeanRankIC"}.issubset(fit.columns)
