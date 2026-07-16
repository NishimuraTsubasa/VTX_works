from __future__ import annotations

import pandas as pd


def validate_stock_panel(stocks: pd.DataFrame, factor_columns: list[str], config: dict) -> pd.DataFrame:
    cols = config["columns"]
    issues: list[dict] = []
    key = [cols["date"], cols["isin"]]
    dup = stocks.duplicated(key, keep=False)
    if dup.any():
        issues.append({
            "issue_type": "duplicate_stock_key",
            "count": int(dup.sum()),
            "detail": f"Duplicate rows by {key}",
        })
    for factor in factor_columns:
        coverage = stocks.groupby(cols["date"])[factor].apply(lambda x: x.notna().mean())
        for date, value in coverage.items():
            if value < 0.60:
                issues.append({
                    "issue_type": "low_factor_coverage",
                    "date": date,
                    "factor": factor,
                    "value": float(value),
                    "detail": "Coverage below 60%",
                })
    return pd.DataFrame(issues)

