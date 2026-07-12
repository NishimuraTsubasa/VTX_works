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


def validate_index_names(constituents: pd.DataFrame, sector_weights: pd.DataFrame, futures_returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    date_col = config["columns"]["date"]
    c = set(constituents["index_name"].unique())
    w = set(sector_weights["index_name"].unique())
    f = set(futures_returns.columns) - {date_col}
    all_names = sorted(c | w | f)
    rows = []
    for name in all_names:
        rows.append({
            "index_name": name,
            "in_constituents": name in c,
            "in_sector_weights": name in w,
            "in_futures_returns": name in f,
            "usable_full_pipeline": name in c and name in w and name in f,
        })
    return pd.DataFrame(rows)
