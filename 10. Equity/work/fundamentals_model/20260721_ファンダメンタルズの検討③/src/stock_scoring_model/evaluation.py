from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .scenarios import ScenarioResult


def _assign_quintile(s: pd.Series, q: int) -> pd.Series:
    valid = s.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    if int(valid.sum()) >= q:
        ranks = s.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=q, labels=range(1, q + 1)).astype(int)
    return out


def _rerank_sample(df: pd.DataFrame, qn: int) -> pd.DataFrame:
    """共通銘柄集合上で各シナリオの順位・分位を再計算する。"""
    out = df.copy()
    rank_scope = str(out["RankScope"].dropna().iloc[0]) if "RankScope" in out and out["RankScope"].notna().any() else "global"
    group_cols = ["Date", "Country"] if rank_scope == "country" and "Country" in out.columns else ["Date"]
    out["TotalScore"] = out.groupby(group_cols)["Prediction"].rank(pct=True)
    qseries = out.groupby(group_cols, group_keys=False)["TotalScore"].apply(lambda s: _assign_quintile(s, qn))
    if isinstance(qseries.index, pd.MultiIndex):
        qseries = qseries.reset_index(level=list(range(qseries.index.nlevels - 1)), drop=True)
    out["Quintile"] = qseries.reindex(out.index)
    return out


def _evaluate_one(df: pd.DataFrame, annual: int, qn: int) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """1シナリオ・1評価サンプルのRankICと5分位指標を計算する。"""
    quintile_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    monthly_ic: list[float] = []

    for date, g in df.groupby("Date"):
        g = g.dropna(subset=["TotalScore", "NextMonthReturn"])
        if len(g) < 10:
            continue
        ic = spearmanr(g["TotalScore"], g["NextMonthReturn"]).statistic
        monthly_ic.append(ic)
        ic_rows.append({"Date": date, "RankIC": ic, "ObservationCount": len(g)})
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
        "ObservationCount": int(df[["Date", "ISIN"]].drop_duplicates().shape[0]) if "ISIN" in df else len(df),
        "MeanStocksPerPeriod": float(df.groupby("Date")["ISIN"].nunique().mean()) if "ISIN" in df and not df.empty else np.nan,
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


def _common_stock_date_keys(cleaned: dict[str, pd.DataFrame], min_stocks_per_date: int) -> pd.MultiIndex:
    key_sets: list[set[tuple[pd.Timestamp, str]]] = []
    for df in cleaned.values():
        keys = set(zip(pd.to_datetime(df["Date"]), df["ISIN"].astype(str)))
        key_sets.append(keys)
    if not key_sets:
        return pd.MultiIndex.from_arrays([[], []], names=["Date", "ISIN"])
    common = set.intersection(*key_sets)
    if not common:
        return pd.MultiIndex.from_arrays([[], []], names=["Date", "ISIN"])
    key_df = pd.DataFrame(common, columns=["Date", "ISIN"])
    counts = key_df.groupby("Date")["ISIN"].nunique()
    valid_dates = counts[counts >= min_stocks_per_date].index
    key_df = key_df[key_df["Date"].isin(valid_dates)]
    return pd.MultiIndex.from_frame(key_df[["Date", "ISIN"]].sort_values(["Date", "ISIN"]))


def rank_ic_delta_table(common_rank_ic: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if common_rank_ic.empty or benchmark not in set(common_rank_ic["Scenario"]):
        return pd.DataFrame()
    pivot = common_rank_ic.pivot(index="Date", columns="Scenario", values="RankIC").sort_index()
    rows = []
    for scenario in pivot.columns:
        if scenario == benchmark:
            continue
        pair = pivot[[scenario, benchmark]].dropna()
        if pair.empty:
            continue
        delta = pair[scenario] - pair[benchmark]
        rows.append({
            "Scenario": scenario,
            "Benchmark": benchmark,
            "CommonPeriods": len(delta),
            "MeanRankICDifference": delta.mean(),
            "MedianRankICDifference": delta.median(),
            "RankICWinRate": float((delta > 0).mean()),
            "RankICDifferenceStd": delta.std(ddof=1),
        })
    return pd.DataFrame(rows)


def evaluate_scenarios(
    results: dict[str, ScenarioResult],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """全利用可能期間と、Date×ISINを揃えた共通OOS期間の両方で評価する。"""
    annual = int(config["evaluation"].get("annualization", 12))
    qn = int(config["evaluation"].get("quintiles", 5))
    common_cfg = config["evaluation"].get("common_oos", {})
    min_stocks = int(common_cfg.get("minimum_stocks_per_date", 10))
    rerank_common = bool(common_cfg.get("rerank_on_common_universe", True))

    cleaned: dict[str, pd.DataFrame] = {
        scenario: result.stock_scores.dropna(subset=["NextMonthReturn", "Prediction"]).copy()
        for scenario, result in results.items()
    }
    common_keys = _common_stock_date_keys(cleaned, min_stocks)
    common_dates = pd.DatetimeIndex(common_keys.get_level_values("Date").unique()).sort_values() if len(common_keys) else pd.DatetimeIndex([])
    common_start = common_dates.min() if len(common_dates) else pd.NaT
    common_end = common_dates.max() if len(common_dates) else pd.NaT

    summary_rows: list[dict[str, Any]] = []
    quintile_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    common_quintile_frames: list[pd.DataFrame] = []
    common_ic_frames: list[pd.DataFrame] = []

    for scenario, df in cleaned.items():
        metrics, qdf, icdf = _evaluate_one(df, annual, qn)
        if not metrics:
            continue
        qdf.insert(0, "Scenario", scenario)
        icdf.insert(0, "Scenario", scenario)
        quintile_frames.append(qdf)
        ic_frames.append(icdf)

        if len(common_keys):
            indexed = df.assign(Date=pd.to_datetime(df["Date"]), ISIN=df["ISIN"].astype(str)).set_index(["Date", "ISIN"], drop=False)
            common_df = indexed.reindex(common_keys).dropna(subset=["Prediction", "NextMonthReturn"]).reset_index(drop=True)
            if rerank_common and not common_df.empty:
                common_df = _rerank_sample(common_df, qn)
        else:
            common_df = df.iloc[0:0].copy()
        common_metrics, cqdf, cicdf = _evaluate_one(common_df, annual, qn)
        if not cqdf.empty:
            cqdf.insert(0, "Scenario", scenario)
            common_quintile_frames.append(cqdf)
        if not cicdf.empty:
            cicdf.insert(0, "Scenario", scenario)
            common_ic_frames.append(cicdf)

        row: dict[str, Any] = {
            "Scenario": scenario,
            "AvailableStartDate": pd.to_datetime(df["Date"]).min(),
            "AvailableEndDate": pd.to_datetime(df["Date"]).max(),
            **metrics,
            "CommonStartDate": common_start,
            "CommonEndDate": common_end,
            "CommonUniverseMode": str(common_cfg.get("universe_mode", "stock_date_intersection")),
        }
        for key, value in common_metrics.items():
            row[f"Common{key}"] = value
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    full_q = pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame()
    full_ic = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    common_q = pd.concat(common_quintile_frames, ignore_index=True) if common_quintile_frames else pd.DataFrame()
    common_ic = pd.concat(common_ic_frames, ignore_index=True) if common_ic_frames else pd.DataFrame()

    benchmark = str(common_cfg.get("benchmark_scenario", "S03_Neutralized_Direct_EW"))
    deltas = rank_ic_delta_table(common_ic, benchmark)
    if not summary.empty and not deltas.empty:
        summary = summary.merge(
            deltas[["Scenario", "MeanRankICDifference", "MedianRankICDifference", "RankICWinRate"]],
            on="Scenario",
            how="left",
        )
        summary = summary.rename(columns={
            "MeanRankICDifference": f"CommonMeanRankICDiffVs_{benchmark}",
            "MedianRankICDifference": f"CommonMedianRankICDiffVs_{benchmark}",
            "RankICWinRate": f"CommonRankICWinRateVs_{benchmark}",
        })
        benchmark_mask = summary["Scenario"].eq(benchmark)
        summary.loc[benchmark_mask, f"CommonMeanRankICDiffVs_{benchmark}"] = 0.0
        summary.loc[benchmark_mask, f"CommonMedianRankICDiffVs_{benchmark}"] = 0.0
        summary.loc[benchmark_mask, f"CommonRankICWinRateVs_{benchmark}"] = np.nan

    warning_periods = int(common_cfg.get("minimum_periods_warning", 24))
    if not summary.empty:
        summary["CommonOOSWarning"] = np.where(
            summary.get("CommonEvaluationPeriods", 0).fillna(0) < warning_periods,
            f"Common OOS is shorter than {warning_periods} periods",
            "",
        )
    return summary, full_q, full_ic, common_q, common_ic


def cumulative_quintile_returns(quintiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    if quintiles.empty:
        return out
    for scenario, g in quintiles.groupby("Scenario"):
        pivot = g.pivot(index="Date", columns="Quintile", values="Return").sort_index()
        out[scenario] = (1 + pivot.fillna(0)).cumprod()
    return out


def cumulative_long_short(quintiles: pd.DataFrame, qn: int = 5) -> dict[str, pd.Series]:
    out = {}
    if quintiles.empty:
        return out
    for scenario, g in quintiles.groupby("Scenario"):
        p = g.pivot(index="Date", columns="Quintile", values="Return").sort_index()
        ls = p.get(qn, 0) - p.get(1, 0)
        out[scenario] = (1 + ls.fillna(0)).cumprod()
    return out
