from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.regularization import fit_ols, fit_penalized_ridge_cv


def test_ols_and_ridge_are_both_available() -> None:
    x = pd.DataFrame({"x1": np.linspace(-1, 1, 40), "x2": np.linspace(-1, 1, 40) + 0.01})
    y = pd.Series(0.5 * x["x1"] + 0.2 * x["x2"])
    ols = fit_ols(x, y)
    ridge = fit_penalized_ridge_cv(x.iloc[:30], y.iloc[:30], x.iloc[30:], y.iloc[30:], [0.1, 1.0], np.ones(2))
    assert ols.estimator_name == "ols"
    assert ridge.estimator_name == "ridge"
    assert ols.alpha == 0.0
    assert ridge.alpha in {0.1, 1.0}
    assert np.isfinite(ols.predict(x)).all()
    assert np.isfinite(ridge.predict(x)).all()
