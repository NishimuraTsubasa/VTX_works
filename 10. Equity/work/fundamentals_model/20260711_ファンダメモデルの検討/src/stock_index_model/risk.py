from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .utils import safe_spearman


def align_futures_forward(futures_returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    date_col = config["columns"]["date"]
    value_cols = [c for c in futures_returns.columns if c != date_col]
    out = futures_returns.copy().sort_values(date_col)
    alignment = config["target"]["futures_return_alignment"]
    horizon = int(config["target"].get("futures_horizon_periods", 1))
    if alignment == "already_forward":
        return out
    if alignment == "contemporaneous_to_forward":
        out[value_cols] = out[value_cols].shift(-horizon)
        return out
    raise ValueError(f"Unknown futures_return_alignment: {alignment}")


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def futures_risk(futures_returns: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_col = config["columns"]["date"]
    frequency = config["data"]["frequency"]
    window = int(config["risk"]["window_periods"][frequency])
    ann = float(config["risk"]["annualization"][frequency])
    confidence = float(config["risk"]["var_confidence"])
    value_cols = [c for c in futures_returns.columns if c != date_col]
    df = futures_returns.sort_values(date_col).set_index(date_col)[value_cols]
    history_rows: list[dict] = []

    for date in df.index:
        window_df = df.loc[:date].tail(window)
        for index_name in value_cols:
            s = window_df[index_name].dropna()
            if len(s) < max(6, window // 3):
                continue
            q = s.quantile(1.0 - confidence)
            downside = s[s < 0]
            history_rows.append({
                "date": date,
                "index_name": index_name,
                "current_return": float(df.loc[date, index_name]) if pd.notna(df.loc[date, index_name]) else np.nan,
                "annualized_volatility": float(s.std(ddof=1) * np.sqrt(ann)),
                "annualized_downside_volatility": float(downside.std(ddof=1) * np.sqrt(ann)) if len(downside) > 1 else np.nan,
                "historical_var": float(-q),
                "historical_expected_shortfall": float(-s[s <= q].mean()) if (s <= q).any() else np.nan,
                "max_drawdown": _max_drawdown(s),
                "skewness": float(skew(s, bias=False)) if len(s) > 2 else np.nan,
                "excess_kurtosis": float(kurtosis(s, fisher=True, bias=False)) if len(s) > 3 else np.nan,
                "observations": len(s),
            })
    history = pd.DataFrame(history_rows)
    latest = history[history["date"] == history["date"].max()].copy() if not history.empty else pd.DataFrame()
    corr = df.tail(window).corr()
    return latest, history, corr


def evaluate_index_scores(index_scores: pd.DataFrame, futures_returns: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate cross-sectional index forecasts and build rolling accuracy metrics."""
    date_col = config["columns"]["date"]
    forward = align_futures_forward(futures_returns, config)
    long = forward.melt(id_vars=[date_col], var_name="index_name", value_name="forward_futures_return")
    merged = index_scores.merge(long, on=[date_col, "index_name"], how="left")
    merged["direction_correct"] = np.where(
        merged["index_alpha"].notna() & merged["forward_futures_return"].notna(),
        (np.sign(merged["index_alpha"]) == np.sign(merged["forward_futures_return"])).astype(float),
        np.nan,
    )
    rows: list[dict] = []
    for date, g in merged.groupby(date_col):
        valid = g[["index_alpha", "forward_futures_return", "direction_correct"]].dropna(
            subset=["index_alpha", "forward_futures_return"]
        )
        if len(valid) < 3:
            continue
        rank_ic = safe_spearman(valid["index_alpha"], valid["forward_futures_return"])
        pearson_ic = float(valid["index_alpha"].corr(valid["forward_futures_return"]))
        ordered = g.dropna(subset=["index_alpha", "forward_futures_return"]).sort_values("index_alpha")
        n = min(3, max(1, len(ordered) // 3))
        top_ret = float(ordered.tail(n)["forward_futures_return"].mean())
        bottom_ret = float(ordered.head(n)["forward_futures_return"].mean())
        actual_top = set(ordered.nlargest(n, "forward_futures_return")["index_name"])
        predicted_top = set(ordered.tail(n)["index_name"])
        rows.append({
            "date": date,
            "index_rank_ic": rank_ic,
            "index_pearson_ic": pearson_ic,
            "top_bottom_spread": top_ret - bottom_ret,
            "top_bucket_return": top_ret,
            "bottom_bucket_return": bottom_ret,
            "directional_accuracy": float(valid["direction_correct"].mean()),
            "top_bucket_hit_rate": len(actual_top & predicted_top) / max(n, 1),
            "index_count": len(valid),
        })
    history = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    if not history.empty:
        frequency = config["data"]["frequency"]
        window = int(config["diagnostics"]["rolling_accuracy_window"][frequency])
        minp = max(3, window // 3)
        history["rolling_rank_ic"] = history["index_rank_ic"].rolling(window, min_periods=minp).mean()
        history["rolling_pearson_ic"] = history["index_pearson_ic"].rolling(window, min_periods=minp).mean()
        history["rolling_directional_accuracy"] = history["directional_accuracy"].rolling(window, min_periods=minp).mean()
        history["rolling_top_bucket_hit_rate"] = history["top_bucket_hit_rate"].rolling(window, min_periods=minp).mean()
        history["rolling_top_bottom_spread"] = history["top_bottom_spread"].rolling(window, min_periods=minp).mean()
        history["cumulative_top_bottom_spread"] = (1.0 + history["top_bottom_spread"].fillna(0.0)).cumprod() - 1.0
    return merged, history


def summarize_model_accuracy(detail: pd.DataFrame, history: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create overall and per-index accuracy summaries."""
    overall_rows: list[dict] = []
    if not history.empty:
        spread_std = history["top_bottom_spread"].std(ddof=1)
        overall_rows.append({
            "scope": "overall_cross_section",
            "mean_rank_ic": float(history["index_rank_ic"].mean()),
            "rank_ic_positive_rate": float((history["index_rank_ic"] > 0).mean()),
            "mean_pearson_ic": float(history["index_pearson_ic"].mean()),
            "mean_directional_accuracy": float(history["directional_accuracy"].mean()),
            "mean_top_bucket_hit_rate": float(history["top_bucket_hit_rate"].mean()),
            "mean_top_bottom_spread": float(history["top_bottom_spread"].mean()),
            "top_bottom_spread_ir": float(history["top_bottom_spread"].mean() / spread_std) if pd.notna(spread_std) and spread_std > 0 else np.nan,
            "periods": int(len(history)),
        })
    overall = pd.DataFrame(overall_rows)

    per_index_rows: list[dict] = []
    for index_name, g in detail.dropna(subset=["index_alpha", "forward_futures_return"]).groupby("index_name"):
        corr = float(g["index_alpha"].corr(g["forward_futures_return"])) if len(g) >= 3 else np.nan
        rank_corr = safe_spearman(g["index_alpha"], g["forward_futures_return"])
        per_index_rows.append({
            "index_name": index_name,
            "observations": int(len(g)),
            "time_series_correlation": corr,
            "time_series_rank_correlation": rank_corr,
            "directional_accuracy": float(g["direction_correct"].mean()),
            "mean_predicted_alpha": float(g["index_alpha"].mean()),
            "mean_realized_return": float(g["forward_futures_return"].mean()),
            "prediction_bias": float((g["index_alpha"] - g["forward_futures_return"]).mean()),
            "rmse": float(np.sqrt(np.mean((g["index_alpha"] - g["forward_futures_return"]) ** 2))),
        })
    return overall, pd.DataFrame(per_index_rows)
