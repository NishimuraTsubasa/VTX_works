from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(g: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = g[value_col].notna() & g[weight_col].notna() & (g[weight_col] > 0)
    denom = g.loc[valid, weight_col].sum()
    if denom <= 0:
        return np.nan
    return float(np.dot(g.loc[valid, value_col], g.loc[valid, weight_col]) / denom)


def _selected_base(selection_history: pd.DataFrame, config: dict) -> pd.DataFrame:
    if selection_history.empty:
        return selection_history.copy()
    cols = config["columns"]
    keep = [
        cols["date"], "index_name", cols["isin"], cols["sector"],
        "selection_weight", "target_sector_weight", "original_sector_weight",
        "is_actual_constituent", "selection_key", "target_count", "rebalanced",
    ]
    return selection_history[[c for c in keep if c in selection_history.columns]].copy()


def aggregate_stock_scores(
    stock_scores: pd.DataFrame,
    constituents: pd.DataFrame,
    sector_weights: pd.DataFrame,
    config: dict,
    selection_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    date_col, isin_col, sector_col = cols["date"], cols["isin"], cols["sector"]
    use_selection = (
        config["aggregation"].get("method") == "selected_universe_weighted"
        and selection_history is not None
        and not selection_history.empty
    )

    if use_selection:
        base = _selected_base(selection_history, config)
        merged = base.merge(stock_scores, on=[date_col, isin_col], how="left")
        weight_col = "selection_weight"
    else:
        base = constituents[["index_name", isin_col, sector_col]].drop_duplicates()
        dates = stock_scores[[date_col]].drop_duplicates()
        base["_key"] = 1
        dates["_key"] = 1
        merged = base.merge(dates, on="_key").drop(columns="_key").merge(
            stock_scores, on=[date_col, isin_col], how="left"
        )
        merged = merged.merge(sector_weights, on=["index_name", sector_col], how="left")
        merged["selection_weight"] = merged.groupby([date_col, "index_name", sector_col])[isin_col].transform(
            lambda x: 1.0 / len(x)
        ) * merged["sector_weight"].fillna(0.0)
        weight_col = "selection_weight"

    threshold = float(config["aggregation"].get("positive_alpha_threshold", 0.0))
    merged["positive_alpha"] = np.where(
        merged["stock_alpha"].notna(), (merged["stock_alpha"] > threshold).astype(float), np.nan
    )
    merged["covered_weight"] = np.where(merged["stock_alpha"].notna(), merged[weight_col], 0.0)

    index_rows: list[dict] = []
    sector_rows: list[dict] = []
    for (date, index_name), g in merged.groupby([date_col, "index_name"]):
        weight_coverage = float(g["covered_weight"].sum())
        row = {
            date_col: date,
            "index_name": index_name,
            "index_alpha": _weighted_mean(g, "stock_alpha", weight_col),
            "index_score": _weighted_mean(g, "stock_score", weight_col),
            "index_confidence": _weighted_mean(g, "confidence_score", weight_col),
            "index_breadth_count_based": float(g["positive_alpha"].dropna().mean()) if g["positive_alpha"].notna().any() else np.nan,
            "index_breadth_weighted": _weighted_mean(g, "positive_alpha", weight_col),
            "selection_weight_coverage": weight_coverage,
            "predicted_constituent_count": int(g["stock_alpha"].notna().sum()),
            "selected_constituent_count": int(g[isin_col].nunique()),
            "usable": weight_coverage >= float(config["aggregation"].get("minimum_index_weight_coverage", 0.75)),
        }
        if "selection_key" in g:
            row["selection_key"] = g["selection_key"].dropna().iloc[0] if g["selection_key"].notna().any() else ""
        index_rows.append(row)

        for sector, s in g.groupby(sector_col):
            sector_rows.append({
                date_col: date,
                "index_name": index_name,
                sector_col: sector,
                "selected_count": int(s[isin_col].nunique()),
                "covered_count": int(s["stock_alpha"].notna().sum()),
                "selection_weight": float(s[weight_col].sum()),
                "covered_weight": float(s["covered_weight"].sum()),
                "sector_alpha": _weighted_mean(s, "stock_alpha", weight_col),
                "sector_score": _weighted_mean(s, "stock_score", weight_col),
                "sector_confidence": _weighted_mean(s, "confidence_score", weight_col),
                "breadth_count": float(s["positive_alpha"].dropna().mean()) if s["positive_alpha"].notna().any() else np.nan,
                "breadth_weighted": _weighted_mean(s, "positive_alpha", weight_col),
            })
    return pd.DataFrame(index_rows), pd.DataFrame(sector_rows)


def aggregate_factor_exposures(
    panel: pd.DataFrame,
    factor_map: dict[str, str],
    constituents: pd.DataFrame,
    sector_weights: pd.DataFrame,
    config: dict,
    selection_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols = config["columns"]
    date_col, isin_col, sector_col = cols["date"], cols["isin"], cols["sector"]
    use_selection = selection_history is not None and not selection_history.empty
    if use_selection:
        base = _selected_base(selection_history, config)
    else:
        base = constituents[["index_name", isin_col, sector_col]].drop_duplicates()
        dates = panel[[date_col]].drop_duplicates()
        base["_key"] = 1
        dates["_key"] = 1
        base = base.merge(dates, on="_key").drop(columns="_key")
        base = base.merge(sector_weights, on=["index_name", sector_col], how="left")
        base["selection_weight"] = base.groupby([date_col, "index_name", sector_col])[isin_col].transform(
            lambda x: 1.0 / len(x)
        ) * base["sector_weight"].fillna(0.0)

    keep = [date_col, isin_col] + list(factor_map.values())
    merged = base.merge(panel[keep], on=[date_col, isin_col], how="left")
    rows: list[dict] = []
    for (date, index_name), g in merged.groupby([date_col, "index_name"]):
        row = {
            date_col: date,
            "index_name": index_name,
            "selected_constituent_count": int(g[isin_col].nunique()),
        }
        factor_coverages = []
        for factor, z_col in factor_map.items():
            row[f"exposure_{factor}"] = _weighted_mean(g, z_col, "selection_weight")
            valid = g[z_col].notna()
            coverage = float(g.loc[valid, "selection_weight"].sum())
            row[f"coverage_{factor}"] = coverage
            factor_coverages.append(coverage)
        row["mean_factor_weight_coverage"] = float(np.nanmean(factor_coverages)) if factor_coverages else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_group_contributions(
    group_predictions: pd.DataFrame,
    constituents: pd.DataFrame,
    sector_weights: pd.DataFrame,
    config: dict,
    selection_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols = config["columns"]
    date_col, isin_col, sector_col = cols["date"], cols["isin"], cols["sector"]
    pivot = group_predictions.pivot_table(
        index=[date_col, isin_col], columns="group", values="group_prediction"
    ).reset_index()
    group_cols = [c for c in pivot.columns if c not in [date_col, isin_col]]
    if selection_history is not None and not selection_history.empty:
        base = _selected_base(selection_history, config)
    else:
        base = constituents[["index_name", isin_col, sector_col]].drop_duplicates()
        dates = pivot[[date_col]].drop_duplicates()
        base["_key"] = 1
        dates["_key"] = 1
        base = base.merge(dates, on="_key").drop(columns="_key")
        base = base.merge(sector_weights, on=["index_name", sector_col], how="left")
        base["selection_weight"] = base.groupby([date_col, "index_name", sector_col])[isin_col].transform(
            lambda x: 1.0 / len(x)
        ) * base["sector_weight"].fillna(0.0)

    merged = base.merge(pivot, on=[date_col, isin_col], how="left")
    rows: list[dict] = []
    for (date, index_name), g in merged.groupby([date_col, "index_name"]):
        row = {date_col: date, "index_name": index_name}
        for group in group_cols:
            row[f"contribution_{group}"] = _weighted_mean(g, group, "selection_weight")
        rows.append(row)
    return pd.DataFrame(rows)
