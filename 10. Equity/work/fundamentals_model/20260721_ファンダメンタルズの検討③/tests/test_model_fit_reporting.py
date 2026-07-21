from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from stock_scoring_model.model_fit_reporting import build_model_fit_diagnostics


def _config():
    return {
        "columns": {
            "date": "date",
            "isin": "isin",
            "country": "country",
            "market_cap": "market_cap",
        },
        "model_fit_diagnostics": {"calibration_bins": 5},
    }


def test_build_model_fit_diagnostics_contains_s06_s07():
    dates = pd.to_datetime(["2021-01-31"] * 20 + ["2021-02-28"] * 20)
    data = pd.DataFrame({
        "date": dates,
        "isin": [f"X{i:03d}" for i in range(40)],
        "country": ["US"] * 10 + ["JP"] * 10 + ["US"] * 10 + ["JP"] * 10,
        "market_cap": 1.0,
        "NextMonthReturn": np.linspace(-0.05, 0.05, 40),
    })
    s06_pred = pd.Series(np.linspace(-0.02, 0.02, 40), index=data.index)
    s06_stock = pd.DataFrame({"Prediction": s06_pred}, index=data.index)
    scenarios = {"S06_Selected_Factor_Models": SimpleNamespace(stock_scores=s06_stock)}
    variants = {
        "S07_OLS_Linear": {
            "Prediction": pd.Series(np.linspace(-0.03, 0.03, 40), index=data.index),
            "CoefficientHistory": pd.DataFrame({"Date": [dates.max()], "Scope": ["country_independent"], "ScopeLabel": ["US"], "Feature": ["Value__LIN"], "Coefficient": [0.2]}),
            "ModelHistory": pd.DataFrame({"Date": [dates.max()], "Scope": ["country_independent"], "ScopeLabel": ["US"], "TrainR2": [0.1], "ValidationR2": [0.01]}),
        }
    }
    diagnostics = {
        "S07Variants": variants,
        "Layer1Coefficients": pd.DataFrame({"FactorCode": ["FA0101"], "Coefficient": [0.1]}),
        "Layer1FitHistory": pd.DataFrame({"FactorCode": ["FA0101"], "ValidationR2": [0.01]}),
        "Layer2Weights": pd.DataFrame({"Date": [dates.max()], "Factor_Group": ["Value"], "FactorCode": ["FA0101"], "Weight": [1.0]}),
    }
    result = build_model_fit_diagnostics(data, scenarios, diagnostics, _config())
    assert set(result.summary["Scenario"]) == {"S06_Selected_Factor_Models", "S07_OLS_Linear"}
    assert "R2" in result.summary.columns
    assert not result.calibration_bins.empty
    assert "FinalEffectiveWeight" in result.s06_effective_weights.columns
