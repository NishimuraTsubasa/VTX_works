from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.evaluation import evaluate_scenarios
from stock_scoring_model.scenarios import ScenarioResult


def _result(name: str, dates: list[str], isins: list[str], offset: float = 0.0) -> ScenarioResult:
    frame = pd.DataFrame({
        "Scenario": name,
        "Date": pd.to_datetime(dates),
        "ISIN": isins,
        "Country": ["US"] * len(dates),
        "Prediction": np.arange(len(dates), dtype=float) + offset,
        "NextMonthReturn": np.arange(len(dates), dtype=float) / 100,
        "RankScope": ["global"] * len(dates),
    })
    frame["TotalScore"] = frame.groupby("Date")["Prediction"].rank(pct=True)
    frame["Quintile"] = 1
    return ScenarioResult(frame, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_common_oos_uses_stock_date_intersection() -> None:
    dates_a = ["2020-01-31"] * 10 + ["2020-02-29"] * 10
    isins_a = [f"X{i}" for i in range(10)] * 2
    dates_b = ["2020-02-29"] * 10
    isins_b = [f"X{i}" for i in range(10)]
    results = {
        "A": _result("A", dates_a, isins_a),
        "B": _result("B", dates_b, isins_b, 0.1),
    }
    config = {
        "evaluation": {
            "annualization": 12,
            "quintiles": 5,
            "common_oos": {
                "minimum_stocks_per_date": 10,
                "rerank_on_common_universe": True,
                "benchmark_scenario": "A",
            },
        }
    }
    summary, _, _, common_q, common_ic = evaluate_scenarios(results, config)
    assert set(pd.to_datetime(common_ic["Date"])) == {pd.Timestamp("2020-02-29")}
    assert set(common_q["Scenario"]) == {"A", "B"}
    assert set(summary["CommonEvaluationPeriods"]) == {1}
