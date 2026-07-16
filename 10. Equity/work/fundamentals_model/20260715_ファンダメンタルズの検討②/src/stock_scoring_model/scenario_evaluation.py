from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .scenario_scoring import ScenarioResult


@dataclass
class ScenarioEvaluation:
    summary: pd.DataFrame
    quintile_summary: pd.DataFrame
    rank_ic_history: pd.DataFrame
    quintile_return_history: pd.DataFrame
    long_short_history: pd.DataFrame


def _max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _annualized_return(returns: pd.Series, annualization: int) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    total = float((1.0 + r).prod())
    if total <= 0:
        return np.nan
    return total ** (annualization / len(r)) - 1.0


def _safe_rank_ic(group: pd.DataFrame, score_col: str, return_col: str) -> float:
    valid = group[[score_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 5 or valid[score_col].nunique() < 2 or valid[return_col].nunique() < 2:
        return np.nan
    return float(valid[score_col].corr(valid[return_col], method="spearman"))


def _common_evaluation_dates(
    scenarios: dict[str, ScenarioResult],
    date_col: str,
    return_col: str,
) -> set[pd.Timestamp]:
    date_sets: list[set[pd.Timestamp]] = []
    for result in scenarios.values():
        df = result.stock_scores
        if df.empty or date_col not in df or return_col not in df:
            continue
        valid_dates = set(pd.to_datetime(df.loc[df[return_col].notna(), date_col]).dropna().unique())
        if valid_dates:
            date_sets.append(valid_dates)
    return set.intersection(*date_sets) if date_sets else set()


def evaluate_stock_scoring_scenarios(
    scenarios: dict[str, ScenarioResult],
    config: dict[str, Any],
) -> ScenarioEvaluation:
    date_col = config["columns"]["date"]
    return_col = "forward_return"
    score_col = "stock_score_0_1"
    quintile_col = "score_quintile"
    annualization = int(config["diagnostics"]["annualization"][config["data"]["frequency"]])
    common_dates = _common_evaluation_dates(scenarios, date_col, return_col)

    summary_rows: list[dict[str, Any]] = []
    quintile_summary_rows: list[dict[str, Any]] = []
    rank_ic_rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    long_short_rows: list[dict[str, Any]] = []

    for scenario_id, result in scenarios.items():
        df = result.stock_scores.copy()
        if df.empty:
            continue
        df[date_col] = pd.to_datetime(df[date_col])
        if common_dates:
            df = df[df[date_col].isin(common_dates)].copy()
        df = df[df[return_col].notna()].copy()

        for date, g in df.groupby(date_col, sort=True):
            rank_ic_rows.append({
                "date": date,
                "scenario_id": scenario_id,
                "scenario_name": result.title,
                "rank_ic": _safe_rank_ic(g, score_col, return_col),
                "stock_count": int(g[[score_col, return_col]].dropna().shape[0]),
            })
            q = (
                g.dropna(subset=[quintile_col, return_col])
                .groupby(quintile_col, as_index=False)[return_col]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={quintile_col: "quintile", "mean": "quintile_return", "count": "stock_count"})
            )
            for row in q.to_dict(orient="records"):
                quintile_rows.append({
                    "date": date,
                    "scenario_id": scenario_id,
                    "scenario_name": result.title,
                    **row,
                })
            qmap = dict(zip(q["quintile"].astype(int), q["quintile_return"])) if not q.empty else {}
            spread = qmap.get(5, np.nan) - qmap.get(1, np.nan)
            long_short_rows.append({
                "date": date,
                "scenario_id": scenario_id,
                "scenario_name": result.title,
                "q5_return": qmap.get(5, np.nan),
                "q1_return": qmap.get(1, np.nan),
                "q5_minus_q1": spread,
            })

        rank_df = pd.DataFrame([r for r in rank_ic_rows if r["scenario_id"] == scenario_id]).sort_values("date")
        quint_df = pd.DataFrame([r for r in quintile_rows if r["scenario_id"] == scenario_id]).sort_values(["date", "quintile"])
        ls_df = pd.DataFrame([r for r in long_short_rows if r["scenario_id"] == scenario_id]).sort_values("date")

        mean_returns = quint_df.groupby("quintile")["quintile_return"].mean() if not quint_df.empty else pd.Series(dtype=float)
        monotonicity = float(pd.Series(mean_returns.index).corr(pd.Series(mean_returns.values), method="spearman")) if len(mean_returns) >= 3 else np.nan
        adjacent = np.nan
        if all(q in mean_returns.index for q in range(1, 6)):
            adjacent = float(np.mean([mean_returns[q + 1] > mean_returns[q] for q in range(1, 5)]))
        for q, value in mean_returns.items():
            q_series = quint_df.loc[quint_df["quintile"].eq(q), "quintile_return"]
            quintile_summary_rows.append({
                "scenario_id": scenario_id,
                "scenario_name": result.title,
                "quintile": int(q),
                "mean_return": float(value),
                "annualized_return": _annualized_return(q_series, annualization),
                "annualized_volatility": float(q_series.std(ddof=1) * np.sqrt(annualization)) if len(q_series.dropna()) > 1 else np.nan,
                "positive_month_rate": float((q_series > 0).mean()) if len(q_series.dropna()) else np.nan,
                "period_count": int(q_series.notna().sum()),
            })

        ic = rank_df["rank_ic"].dropna() if not rank_df.empty else pd.Series(dtype=float)
        ls = ls_df["q5_minus_q1"].dropna() if not ls_df.empty else pd.Series(dtype=float)
        summary_rows.append({
            "scenario_id": scenario_id,
            "scenario_name": result.title,
            "evaluation_start": df[date_col].min() if not df.empty else pd.NaT,
            "evaluation_end": df[date_col].max() if not df.empty else pd.NaT,
            "evaluation_periods": int(df[date_col].nunique()),
            "mean_stock_count": float(df.groupby(date_col).size().mean()) if not df.empty else np.nan,
            "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "median_rank_ic": float(ic.median()) if len(ic) else np.nan,
            "rank_ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
            "rank_ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
            "rank_ic_positive_rate": float((ic > 0).mean()) if len(ic) else np.nan,
            "q5_minus_q1_mean": float(ls.mean()) if len(ls) else np.nan,
            "q5_minus_q1_annualized_return": _annualized_return(ls, annualization),
            "q5_minus_q1_annualized_volatility": float(ls.std(ddof=1) * np.sqrt(annualization)) if len(ls) > 1 else np.nan,
            "q5_minus_q1_sharpe": float(ls.mean() / ls.std(ddof=1) * np.sqrt(annualization)) if len(ls) > 1 and ls.std(ddof=1) > 0 else np.nan,
            "q5_minus_q1_positive_rate": float((ls > 0).mean()) if len(ls) else np.nan,
            "q5_minus_q1_max_drawdown": _max_drawdown(ls),
            "quintile_monotonicity": monotonicity,
            "adjacent_monotonicity": adjacent,
        })

    return ScenarioEvaluation(
        summary=pd.DataFrame(summary_rows),
        quintile_summary=pd.DataFrame(quintile_summary_rows),
        rank_ic_history=pd.DataFrame(rank_ic_rows),
        quintile_return_history=pd.DataFrame(quintile_rows),
        long_short_history=pd.DataFrame(long_short_rows),
    )
