from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.layer3_country_reporting import build_country_diagnostics


def _payload(index: pd.Index, prediction: pd.Series, estimator: str, alpha: float) -> dict[str, pd.DataFrame | pd.Series]:
    dates = sorted(pd.to_datetime(index.get_level_values("Date").unique()))
    countries = sorted(index.get_level_values("Country").unique())
    coef_rows = []
    model_rows = []
    for date in dates:
        for country in countries:
            coef_rows += [
                {
                    "Date": date,
                    "Scope": "country_independent",
                    "ScopeLabel": country,
                    "Feature": "Value__LIN",
                    "Coefficient": 0.2,
                    "StandardizedCoefficient": 0.2,
                    "RawCoefficient": 0.1,
                    "Estimator": estimator,
                    "Alpha": alpha,
                },
                {
                    "Date": date,
                    "Scope": "country_independent",
                    "ScopeLabel": country,
                    "Feature": "INT__Banks__Value__LIN",
                    "Coefficient": 0.1,
                    "StandardizedCoefficient": 0.1,
                    "RawCoefficient": 0.05,
                    "Estimator": estimator,
                    "Alpha": alpha,
                },
            ]
            model_rows.append(
                {
                    "Date": date,
                    "Scope": "country_independent",
                    "ScopeLabel": country,
                    "Estimator": estimator,
                    "Alpha": alpha,
                }
            )
    return {
        "Prediction": prediction.reset_index(drop=True),
        "CoefficientHistory": pd.DataFrame(coef_rows),
        "ModelHistory": pd.DataFrame(model_rows),
    }


def test_country_diagnostics_contains_country_performance_and_coefficients() -> None:
    dates = pd.date_range("2022-01-31", periods=6, freq="ME")
    countries = ["JP", "US"]
    rows = []
    for date in dates:
        for country in countries:
            for n in range(20):
                signal = (n - 9.5) / 10.0
                rows.append(
                    {
                        "Date": date,
                        "Country": country,
                        "ISIN": f"{country}{n:03d}",
                        "Sector": "Banks" if n < 10 else "Other",
                        "Return": 0.02 * signal,
                        "Signal": signal,
                    }
                )
    frame = pd.DataFrame(rows)
    data = pd.DataFrame(
        {
            "date": frame["Date"],
            "ISIN": frame["ISIN"],
            "country": frame["Country"],
            "sector": frame["Sector"],
            "NextMonthReturn": frame["Return"],
        }
    )
    index = pd.MultiIndex.from_frame(frame[["Date", "Country"]])
    variants = {
        "S07_OLS_Linear": _payload(index, frame["Signal"], "ols", 0.0),
        "S07_Ridge_Linear": _payload(index, frame["Signal"] * 0.9, "ridge", 1.0),
    }
    config = {
        "columns": {"date": "date", "isin": "ISIN", "country": "country", "sector": "sector"},
        "evaluation": {"quintiles": 5, "annualization": 12},
        "layer3": {"country_diagnostics_minimum_stocks": 10},
    }

    diagnostics = build_country_diagnostics(data, variants, config)

    assert len(diagnostics.summary) == 4
    assert set(diagnostics.summary["Country"]) == {"JP", "US"}
    assert diagnostics.summary["MeanRankIC"].min() > 0.99
    assert not diagnostics.latest_coefficients.empty
    assert not diagnostics.coefficient_stability.empty
    assert not diagnostics.effective_sector_slopes.empty
    assert np.allclose(diagnostics.effective_sector_slopes["EffectiveSectorSlope"], 0.3)
