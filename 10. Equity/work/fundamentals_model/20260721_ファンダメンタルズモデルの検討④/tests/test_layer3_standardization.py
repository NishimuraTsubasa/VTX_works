from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.regularization import coefficient_frame, fit_ols, fit_penalized_ridge


def test_standardized_model_predicts_raw_feature_values() -> None:
    x = pd.DataFrame(
        {
            "factor": np.linspace(-20.0, 30.0, 100),
            "sector_dummy": np.tile([0.0, 1.0], 50),
        }
    )
    y = pd.Series(0.03 * x["factor"] + 0.20 * x["sector_dummy"] + 0.5)
    mask = np.array([True, False])

    model = fit_ols(x, y, standardize_mask=mask)
    pred = model.predict(x)

    assert np.max(np.abs(pred - y.to_numpy())) < 1.0e-10
    assert model.standardized_mask_.tolist() == [True, False]
    assert model.feature_scales_[0] > 1.0
    assert model.feature_scales_[1] == 1.0

    coef = coefficient_frame(model)
    assert {"StandardizedCoefficient", "RawCoefficient", "FeatureScale", "RawIntercept"}.issubset(coef.columns)
    raw_factor_coef = coef.loc[coef["Feature"].eq("factor"), "RawCoefficient"].iloc[0]
    assert abs(raw_factor_coef - 0.03) < 1.0e-10


def test_ridge_standardization_is_scale_robust() -> None:
    base = np.linspace(-2.0, 2.0, 120)
    x1 = pd.DataFrame({"factor": base, "dummy": np.tile([0.0, 1.0], 60)})
    x2 = x1.copy()
    x2["factor"] *= 1000.0
    y = pd.Series(0.5 * base + 0.1 * x1["dummy"])
    mask = np.array([True, False])

    m1 = fit_penalized_ridge(x1, y, alpha=1.0, penalty_multipliers=np.ones(2), standardize_mask=mask)
    m2 = fit_penalized_ridge(x2, y, alpha=1.0, penalty_multipliers=np.ones(2), standardize_mask=mask)

    assert np.max(np.abs(m1.predict(x1) - m2.predict(x2))) < 1.0e-10
