from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from stock_scoring_model.factor_score_performance_reporting import (
    build_factor_score_performance_diagnostics,
    write_factor_score_performance_excel,
    write_factor_score_performance_pdf,
)
from stock_scoring_model.master import FactorMeta


def _fixture():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    rows = []
    value = []
    momentum = []
    raw1 = []
    raw2 = []
    sub1 = []
    sub2 = []
    for date in dates:
        for country in ["JP", "US"]:
            for idx in range(30):
                v = rng.normal()
                m = rng.normal()
                rows.append({
                    "date": date,
                    "ISIN": f"{country}_{idx:03d}",
                    "country": country,
                    "sector": "Banks" if idx < 15 else "Industrials",
                    "market_cap": 100 + idx,
                    "NextMonthReturn": 0.02 * v + 0.01 * m + rng.normal(scale=0.03),
                })
                value.append(v)
                momentum.append(m)
                raw1.append(v + rng.normal(scale=0.2))
                raw2.append(m + rng.normal(scale=0.2))
                sub1.append(0.02 * v)
                sub2.append(0.01 * m)
    data = pd.DataFrame(rows)
    layer2 = pd.DataFrame({"Value": value, "Momentum": momentum}, index=data.index)
    layer1_input = pd.DataFrame({"FA0101": raw1, "FA1001": raw2}, index=data.index)
    subscores = pd.DataFrame({"FA0101": sub1, "FA1001": sub2}, index=data.index)
    stock = pd.DataFrame({
        "Date": data["date"], "ISIN": data["ISIN"], "Prediction": np.asarray(value) + np.asarray(momentum),
        "NextMonthReturn": data["NextMonthReturn"],
    }, index=data.index)
    scenarios = {
        "S06_Selected_Factor_Models": SimpleNamespace(stock_scores=stock),
        "S07_OLS_Linear": SimpleNamespace(stock_scores=stock),
        "S07_Ridge_Linear": SimpleNamespace(stock_scores=stock),
    }
    diagnostics = {
        "Layer1InputScores": layer1_input,
        "Layer1Subscores": subscores,
        "Layer2FactorScores": layer2,
    }
    metas = {
        "FA0101": FactorMeta("FA0101", "Value", 1, 1.0),
        "FA1001": FactorMeta("FA1001", "Momentum", 1, 1.0),
    }
    config = {
        "columns": {"date": "date", "isin": "ISIN", "country": "country", "sector": "sector"},
        "evaluation": {"quintiles": 5, "annualization": 12},
        "factor_score_performance_diagnostics": {
            "common_oos_scenarios": list(scenarios),
            "minimum_stocks_per_date": 30,
            "minimum_stocks_per_group": 10,
            "minimum_stocks_per_country_sector": 8,
            "minimum_country_sector_periods": 3,
            "quantiles": 5,
            "calibration_bins": 5,
            "rolling_rank_ic_periods": 3,
            "subscore_top_n_pdf": 4,
        },
    }
    return data, scenarios, diagnostics, metas, config


def test_factor_score_performance_diagnostics_and_outputs(tmp_path):
    data, scenarios, diagnostics, metas, config = _fixture()
    result = build_factor_score_performance_diagnostics(data, scenarios, diagnostics, metas, config)
    assert not result.factor_group_summary.empty
    assert set(result.factor_group_summary["Signal"]) == {"Value", "Momentum"}
    assert not result.subscore_summary.empty
    assert not result.raw_vs_subscore.empty
    assert not result.leave_one_group_out.empty
    xlsx = tmp_path / "diagnostics.xlsx"
    pdf = tmp_path / "diagnostics.pdf"
    write_factor_score_performance_excel(data, scenarios, diagnostics, metas, xlsx, config)
    write_factor_score_performance_pdf(data, scenarios, diagnostics, metas, pdf, config)
    assert xlsx.exists() and xlsx.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
