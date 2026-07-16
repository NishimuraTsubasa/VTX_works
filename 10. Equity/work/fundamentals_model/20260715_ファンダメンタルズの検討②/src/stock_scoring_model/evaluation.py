from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .scenarios import ScenarioResult


def _evaluate_one(df: pd.DataFrame, annual: int, qn: int) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """1シナリオ・1評価サンプルのRankICと5分位指標を計算する。"""
    quintile_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    monthly_ic: list[float] = []

    for date, g in df.groupby("Date"):
        if len(g) < 10:
            continue
        ic = spearmanr(g["TotalScore"], g["NextMonthReturn"]).statistic
        monthly_ic.append(ic)
        ic_rows.append({"Date": date, "RankIC": ic})
        for q, qg in g.dropna(subset=["Quintile"]).groupby("Quintile"):
            quintile_rows.append({
                "Date": date,
                "Quintile": int(q),
                "Return": qg["NextMonthReturn"].mean(),
                "Count": len(qg),
            })

    qdf = pd.DataFrame(quintile_rows)
    if qdf.empty:
        return {}, qdf, pd.DataFrame(ic_rows)

    pivot = qdf.pivot(index="Date", columns="Quintile", values="Return").sort_index()
    ls = pivot.get(qn, pd.Series(index=pivot.index, dtype=float)) - pivot.get(1, pd.Series(index=pivot.index, dtype=float))
    mean_ls = ls.mean()
    vol_ls = ls.std(ddof=1)
    ann_ret = (1 + ls.fillna(0)).prod() ** (annual / max(len(ls), 1)) - 1 if len(ls) else np.nan
    sharpe = mean_ls / vol_ls * np.sqrt(annual) if vol_ls and np.isfinite(vol_ls) else np.nan
    cumulative = (1 + ls.fillna(0)).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    qmean = pivot.mean()
    monotonicity = spearmanr(qmean.index.astype(float), qmean.to_numpy()).statistic if len(qmean) >= 2 else np.nan
    adjacent_pairs = [
        qmean.get(i + 1, np.nan) > qmean.get(i, np.nan)
        for i in range(1, qn)
        if np.isfinite(qmean.get(i + 1, np.nan)) and np.isfinite(qmean.get(i, np.nan))
    ]
    adjacent = float(np.mean(adjacent_pairs)) if adjacent_pairs else np.nan
    ic_arr = np.asarray([x for x in monthly_ic if np.isfinite(x)], float)

    metrics = {
        "EvaluationPeriods": len(pivot),
        "MeanRankIC": np.nanmean(ic_arr) if len(ic_arr) else np.nan,
        "MedianRankIC": np.nanmedian(ic_arr) if len(ic_arr) else np.nan,
        "RankICIR": np.nanmean(ic_arr) / np.nanstd(ic_arr, ddof=1)
        if len(ic_arr) > 1 and np.nanstd(ic_arr, ddof=1) > 0 else np.nan,
        "RankICPositiveRate": np.mean(ic_arr > 0) if len(ic_arr) else np.nan,
        "Q5MinusQ1Mean": mean_ls,
        "Q5MinusQ1AnnualizedReturn": ann_ret,
        "Q5MinusQ1Sharpe": sharpe,
        "Q5MinusQ1MaxDrawdown": drawdown.min() if len(drawdown) else np.nan,
        "QuintileMonotonicity": monotonicity,
        "AdjacentMonotonicity": adjacent,
    }
    return metrics, qdf, pd.DataFrame(ic_rows)


def evaluate_scenarios(results: dict[str, ScenarioResult], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """8シナリオを利用可能期間と共通OOS期間の両方で評価する。"""
    annual = int(config["evaluation"].get("annualization", 12))
    qn = int(config["evaluation"].get("quintiles", 5))

    cleaned: dict[str, pd.DataFrame] = {
        scenario: result.stock_scores.dropna(subset=["NextMonthReturn", "TotalScore"]).copy()
        for scenario, result in results.items()
    }
    valid_date_sets = [set(pd.to_datetime(df["Date"].unique())) for df in cleaned.values() if not df.empty]
    common_dates = set.intersection(*valid_date_sets) if valid_date_sets else set()
    common_start = min(common_dates) if common_dates else pd.NaT
    common_end = max(common_dates) if common_dates else pd.NaT

    summary_rows: list[dict[str, Any]] = []
    quintile_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []

    for scenario, df in cleaned.items():
        metrics, qdf, icdf = _evaluate_one(df, annual, qn)
        if not metrics:
            continue
        qdf.insert(0, "Scenario", scenario)
        icdf.insert(0, "Scenario", scenario)
        quintile_frames.append(qdf)
        ic_frames.append(icdf)

        common_df = df[pd.to_datetime(df["Date"]).isin(common_dates)] if common_dates else df.iloc[0:0]
        common_metrics, _, _ = _evaluate_one(common_df, annual, qn)
        row: dict[str, Any] = {"Scenario": scenario, **metrics}
        row["CommonStartDate"] = common_start
        row["CommonEndDate"] = common_end
        for key, value in common_metrics.items():
            row[f"Common{key}"] = value
        summary_rows.append(row)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame(),
        pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame(),
    )


def cumulative_quintile_returns(quintiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for scenario, g in quintiles.groupby("Scenario"):
        pivot = g.pivot(index="Date", columns="Quintile", values="Return").sort_index()
        out[scenario] = (1 + pivot.fillna(0)).cumprod()
    return out


def cumulative_long_short(quintiles: pd.DataFrame, qn: int = 5) -> dict[str, pd.Series]:
    out = {}
    for scenario, g in quintiles.groupby("Scenario"):
        p = g.pivot(index="Date", columns="Quintile", values="Return").sort_index()
        ls = p.get(qn, 0) - p.get(1, 0)
        out[scenario] = (1 + ls.fillna(0)).cumprod()
    return out
