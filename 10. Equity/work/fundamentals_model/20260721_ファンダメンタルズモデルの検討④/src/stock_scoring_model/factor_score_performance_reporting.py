from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from .master import FactorMeta
from .reporting import _setup_matplotlib
from .scenarios import ScenarioResult


@dataclass
class FactorScorePerformanceDiagnostics:
    factor_group_summary: pd.DataFrame
    factor_group_monthly_ic: pd.DataFrame
    factor_group_quintiles: pd.DataFrame
    factor_group_long_short: pd.DataFrame
    factor_group_calibration: pd.DataFrame
    subscore_summary: pd.DataFrame
    subscore_monthly_ic: pd.DataFrame
    subscore_quintiles: pd.DataFrame
    raw_vs_subscore: pd.DataFrame
    country_factor_group: pd.DataFrame
    country_sector_factor_group: pd.DataFrame
    factor_group_correlation: pd.DataFrame
    leave_one_group_out: pd.DataFrame
    coverage_dispersion: pd.DataFrame
    common_oos_keys: pd.DataFrame


def _safe_spearman(x: pd.Series, y: pd.Series, minimum: int) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < minimum or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(frame["x"], frame["y"]).statistic)


def _assign_quantile(score: pd.Series, q: int) -> pd.Series:
    valid = pd.to_numeric(score, errors="coerce").notna()
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    if int(valid.sum()) < q:
        return out
    ranked = pd.to_numeric(score.loc[valid], errors="coerce").rank(method="first")
    out.loc[valid] = pd.qcut(ranked, q=q, labels=range(1, q + 1)).astype(int)
    return out


def _summary_from_monthly_ic(monthly: pd.DataFrame) -> dict[str, float]:
    values = pd.to_numeric(monthly.get("RankIC", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return {
            "EvaluationPeriods": 0,
            "MeanRankIC": np.nan,
            "MedianRankIC": np.nan,
            "RankICStd": np.nan,
            "RankICIR": np.nan,
            "RankICPositiveRate": np.nan,
        }
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "EvaluationPeriods": int(len(values)),
        "MeanRankIC": float(values.mean()),
        "MedianRankIC": float(values.median()),
        "RankICStd": std,
        "RankICIR": float(values.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
        "RankICPositiveRate": float((values > 0).mean()),
    }


def _quintile_metrics(quintiles: pd.DataFrame, qn: int, annual: int) -> dict[str, float]:
    if quintiles.empty:
        return {
            "Q5MinusQ1Mean": np.nan,
            "Q5MinusQ1AnnualizedReturn": np.nan,
            "Q5MinusQ1Sharpe": np.nan,
            "Q5MinusQ1MaxDrawdown": np.nan,
            "QuintileMonotonicity": np.nan,
            "AdjacentMonotonicity": np.nan,
        }
    pivot = quintiles.pivot_table(index="Date", columns="Quintile", values="Return", aggfunc="mean").sort_index()
    if 1 not in pivot.columns or qn not in pivot.columns:
        return {
            "Q5MinusQ1Mean": np.nan,
            "Q5MinusQ1AnnualizedReturn": np.nan,
            "Q5MinusQ1Sharpe": np.nan,
            "Q5MinusQ1MaxDrawdown": np.nan,
            "QuintileMonotonicity": np.nan,
            "AdjacentMonotonicity": np.nan,
        }
    ls = (pivot[qn] - pivot[1]).dropna()
    if ls.empty:
        return {}
    wealth = (1.0 + ls).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    std = float(ls.std(ddof=1)) if len(ls) > 1 else np.nan
    qmean = pivot.mean().sort_index()
    monotonicity = _safe_spearman(pd.Series(qmean.index, index=qmean.index), qmean, 2)
    adjacent = [qmean.loc[q + 1] > qmean.loc[q] for q in range(1, qn) if q in qmean.index and q + 1 in qmean.index]
    return {
        "Q5MinusQ1Mean": float(ls.mean()),
        "Q5MinusQ1AnnualizedReturn": float((1.0 + ls).prod() ** (annual / len(ls)) - 1.0),
        "Q5MinusQ1Sharpe": float(ls.mean() / std * np.sqrt(annual)) if np.isfinite(std) and std > 0 else np.nan,
        "Q5MinusQ1MaxDrawdown": float(drawdown.min()),
        "QuintileMonotonicity": monotonicity,
        "AdjacentMonotonicity": float(np.mean(adjacent)) if adjacent else np.nan,
    }


def _pool_quintiles(quintiles: pd.DataFrame) -> pd.DataFrame:
    if quintiles.empty:
        return pd.DataFrame()
    rows = []
    for (date, quantile), group in quintiles.groupby(["Date", "Quintile"]):
        valid = group.dropna(subset=["Return", "Count"])
        if valid.empty or valid["Count"].sum() <= 0:
            continue
        rows.append({
            "Date": date,
            "Quintile": int(quantile),
            "Return": float(np.average(valid["Return"], weights=valid["Count"])),
            "MeanSignal": float(np.average(valid["MeanSignal"], weights=valid["Count"])),
            "Count": int(valid["Count"].sum()),
        })
    return pd.DataFrame(rows)


def _common_oos_mask(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    config: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    c = config["columns"]
    cfg = config.get("factor_score_performance_diagnostics", {})
    names = list(cfg.get("common_oos_scenarios", [
        "S06_Selected_Factor_Models", "S07_OLS_Linear", "S07_Ridge_Linear"
    ]))
    names = [name for name in names if name in scenarios]
    key_sets: list[set[tuple[pd.Timestamp, str]]] = []
    for name in names:
        frame = scenarios[name].stock_scores.dropna(subset=["Prediction", "NextMonthReturn"])
        key_sets.append(set(zip(pd.to_datetime(frame["Date"]), frame["ISIN"].astype(str))))
    base_keys = pd.DataFrame({
        "Date": pd.to_datetime(data[c["date"]]),
        "ISIN": data[c["isin"]].astype(str),
    }, index=data.index)
    if not key_sets:
        mask = data["NextMonthReturn"].notna()
        return mask, base_keys.loc[mask].drop_duplicates().reset_index(drop=True)
    common = set.intersection(*key_sets)
    if not common:
        return pd.Series(False, index=data.index), pd.DataFrame(columns=["Date", "ISIN"])
    common_keys = pd.DataFrame(common, columns=["Date", "ISIN"])
    min_stocks = int(cfg.get("minimum_stocks_per_date", 30))
    counts = common_keys.groupby("Date")["ISIN"].nunique()
    valid_dates = counts[counts >= min_stocks].index
    common_keys = common_keys[common_keys["Date"].isin(valid_dates)].sort_values(["Date", "ISIN"])
    key_index = pd.MultiIndex.from_frame(common_keys[["Date", "ISIN"]])
    base_index = pd.MultiIndex.from_frame(base_keys[["Date", "ISIN"]])
    return pd.Series(base_index.isin(key_index), index=data.index), common_keys.reset_index(drop=True)


def _evaluate_signal(
    base: pd.DataFrame,
    signal: pd.Series,
    *,
    signal_name: str,
    scope: str,
    group_columns: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config.get("factor_score_performance_diagnostics", {})
    qn = int(cfg.get("quantiles", config.get("evaluation", {}).get("quintiles", 5)))
    annual = int(config.get("evaluation", {}).get("annualization", 12))
    minimum = int(cfg.get("minimum_stocks_per_group", 15))
    frame = base.copy()
    frame["Signal"] = pd.to_numeric(signal, errors="coerce")
    frame = frame.dropna(subset=["Signal", "NextMonthReturn"])
    monthly_rows: list[dict[str, object]] = []
    quintile_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if len(group) < minimum:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_columns, keys))
        ic = _safe_spearman(group["Signal"], group["NextMonthReturn"], minimum)
        monthly_rows.append({"Signal": signal_name, "Scope": scope, **key_dict, "RankIC": ic, "ObservationCount": len(group)})
        group = group.copy()
        group["Quintile"] = _assign_quantile(group["Signal"], qn)
        for quantile, qframe in group.dropna(subset=["Quintile"]).groupby("Quintile"):
            quintile_rows.append({
                "Signal": signal_name,
                "Scope": scope,
                **key_dict,
                "Quintile": int(quantile),
                "Return": float(qframe["NextMonthReturn"].mean()),
                "MeanSignal": float(qframe["Signal"].mean()),
                "Count": int(len(qframe)),
            })
    monthly = pd.DataFrame(monthly_rows)
    quintiles = pd.DataFrame(quintile_rows)
    # Aggregate subgroup ICs to one value per month using observation counts.
    if scope != "global" and not monthly.empty:
        weighted_rows = []
        for date, group in monthly.groupby("Date"):
            valid = group.dropna(subset=["RankIC", "ObservationCount"])
            weighted = np.average(valid["RankIC"], weights=valid["ObservationCount"]) if not valid.empty else np.nan
            weighted_rows.append({"Signal": signal_name, "Scope": scope, "Date": date, "RankIC": weighted, "ObservationCount": valid["ObservationCount"].sum()})
        summary_monthly = pd.DataFrame(weighted_rows)
    else:
        summary_monthly = monthly.copy()
    summary = {"Signal": signal_name, "Scope": scope, **_summary_from_monthly_ic(summary_monthly)}
    # Pool subgroup quintile returns to date-level equal-security averages.
    q_for_metrics = _pool_quintiles(quintiles)
    summary.update(_quintile_metrics(q_for_metrics, qn, annual))
    long_short = pd.DataFrame()
    if not q_for_metrics.empty:
        pivot = q_for_metrics.pivot(index="Date", columns="Quintile", values="Return").sort_index()
        if 1 in pivot and qn in pivot:
            long_short = (pivot[qn] - pivot[1]).rename("LongShortReturn").reset_index()
            long_short.insert(0, "Scope", scope)
            long_short.insert(0, "Signal", signal_name)
            long_short["CumulativeWealth"] = (1.0 + long_short["LongShortReturn"].fillna(0)).cumprod()
    return summary, monthly, quintiles, long_short


def _calibration_table(base: pd.DataFrame, signal: pd.Series, signal_name: str, bins: int, minimum: int) -> pd.DataFrame:
    frame = base.copy()
    frame["SignalValue"] = pd.to_numeric(signal, errors="coerce")
    frame = frame.dropna(subset=["SignalValue", "NextMonthReturn"])
    rows = []
    for date, group in frame.groupby("Date"):
        if len(group) < max(minimum, bins * 2):
            continue
        group = group.copy()
        group["Bin"] = _assign_quantile(group["SignalValue"], bins)
        for bin_no, bframe in group.dropna(subset=["Bin"]).groupby("Bin"):
            rows.append({
                "Signal": signal_name,
                "Date": date,
                "Bin": int(bin_no),
                "MeanSignal": float(bframe["SignalValue"].mean()),
                "MeanNextMonthReturn": float(bframe["NextMonthReturn"].mean()),
                "ObservationCount": int(len(bframe)),
            })
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly
    return monthly.groupby(["Signal", "Bin"], as_index=False).agg(
        MeanSignal=("MeanSignal", "mean"),
        MeanNextMonthReturn=("MeanNextMonthReturn", "mean"),
        TimeSeriesStd=("MeanNextMonthReturn", "std"),
        Periods=("Date", "nunique"),
        ObservationCount=("ObservationCount", "sum"),
    )


def _monthly_correlation(factor_scores: pd.DataFrame, base: pd.DataFrame, minimum: int) -> pd.DataFrame:
    records = []
    columns = list(factor_scores.columns)
    matrices = []
    for date, indices in base.groupby("Date").groups.items():
        frame = factor_scores.loc[indices, columns]
        if len(frame) < minimum:
            continue
        corr = frame.corr(method="spearman")
        corr["__Date"] = date
        matrices.append(corr)
    if not matrices:
        return pd.DataFrame(columns=["FactorGroup1", "FactorGroup2", "MeanSpearmanCorrelation", "Periods"])
    for left in columns:
        for right in columns:
            values = [matrix.loc[left, right] for matrix in matrices if left in matrix.index and right in matrix.columns]
            records.append({
                "FactorGroup1": left,
                "FactorGroup2": right,
                "MeanSpearmanCorrelation": float(np.nanmean(values)) if values else np.nan,
                "Periods": int(np.isfinite(values).sum()),
            })
    return pd.DataFrame(records)


def build_factor_score_performance_diagnostics(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
) -> FactorScorePerformanceDiagnostics:
    c = config["columns"]
    cfg = config.get("factor_score_performance_diagnostics", {})
    minimum = int(cfg.get("minimum_stocks_per_group", 15))
    cs_minimum = int(cfg.get("minimum_stocks_per_country_sector", 8))
    bins = int(cfg.get("calibration_bins", 10))
    common_mask, common_keys = _common_oos_mask(data, scenarios, config)
    base = pd.DataFrame({
        "Date": pd.to_datetime(data[c["date"]]),
        "ISIN": data[c["isin"]].astype(str),
        "Country": data[c["country"]].astype(str),
        "Sector": data[c["sector"]].astype(str),
        "NextMonthReturn": pd.to_numeric(data["NextMonthReturn"], errors="coerce"),
    }, index=data.index)
    base = base.loc[common_mask].copy()
    layer1_input = diagnostics.get("Layer1InputScores", pd.DataFrame(index=data.index)).reindex(base.index)
    subscores = diagnostics.get("Layer1Subscores", pd.DataFrame(index=data.index)).reindex(base.index)
    factor_scores = diagnostics.get("Layer2FactorScores", pd.DataFrame(index=data.index)).reindex(base.index)

    group_summary_rows = []
    group_monthly_frames = []
    group_quintile_frames = []
    group_ls_frames = []
    calibration_frames = []
    country_rows = []
    country_sector_rows = []

    for factor in factor_scores.columns:
        signal = factor_scores[factor]
        for scope, columns in {
            "global": ["Date"],
            "within_country": ["Date", "Country"],
            "within_country_sector": ["Date", "Country", "Sector"],
        }.items():
            local_cfg = dict(config)
            local_cfg["factor_score_performance_diagnostics"] = dict(cfg)
            if scope == "within_country_sector":
                local_cfg["factor_score_performance_diagnostics"]["minimum_stocks_per_group"] = cs_minimum
            summary, monthly, quintiles, long_short = _evaluate_signal(
                base, signal, signal_name=factor, scope=scope, group_columns=columns, config=local_cfg
            )
            group_summary_rows.append(summary)
            if not monthly.empty:
                monthly.insert(0, "FactorGroup", factor)
                group_monthly_frames.append(monthly)
            if not quintiles.empty:
                quintiles.insert(0, "FactorGroup", factor)
                group_quintile_frames.append(quintiles)
            if not long_short.empty:
                long_short.insert(0, "FactorGroup", factor)
                group_ls_frames.append(long_short)
            if scope == "within_country" and not monthly.empty:
                for country, country_monthly in monthly.groupby("Country"):
                    country_quintiles = quintiles[quintiles["Country"].eq(country)] if not quintiles.empty else pd.DataFrame()
                    country_summary = _summary_from_monthly_ic(country_monthly)
                    country_summary.update(_quintile_metrics(_pool_quintiles(country_quintiles), int(cfg.get("quantiles", 5)), int(config.get("evaluation", {}).get("annualization", 12))))
                    country_rows.append({"Country": country, "FactorGroup": factor, **country_summary})
            if scope == "within_country_sector" and not monthly.empty:
                for (country, sector), cs_monthly in monthly.groupby(["Country", "Sector"]):
                    if len(cs_monthly.dropna(subset=["RankIC"])) < int(cfg.get("minimum_country_sector_periods", 6)):
                        continue
                    cs_quintiles = quintiles[quintiles["Country"].eq(country) & quintiles["Sector"].eq(sector)] if not quintiles.empty else pd.DataFrame()
                    cs_summary = _summary_from_monthly_ic(cs_monthly)
                    cs_summary.update(_quintile_metrics(_pool_quintiles(cs_quintiles), int(cfg.get("quantiles", 5)), int(config.get("evaluation", {}).get("annualization", 12))))
                    country_sector_rows.append({"Country": country, "Sector": sector, "FactorGroup": factor, **cs_summary})
        calibration = _calibration_table(base, signal, factor, bins, minimum)
        if not calibration.empty:
            calibration_frames.append(calibration)
    # SubScore diagnostics.
    sub_summary_rows = []
    sub_monthly_frames = []
    sub_quintile_frames = []
    raw_vs_rows = []
    for code in subscores.columns:
        meta = metas.get(code)
        group_name = meta.group if meta is not None else "Unknown"
        sub_summary, sub_monthly, sub_quintiles, _ = _evaluate_signal(
            base, subscores[code], signal_name=code, scope="global", group_columns=["Date"], config=config
        )
        sub_summary_rows.append({"FactorCode": code, "FactorGroup": group_name, **{k: v for k, v in sub_summary.items() if k not in {"Signal", "Scope"}}})
        if not sub_monthly.empty:
            sub_monthly.insert(0, "FactorGroup", group_name)
            sub_monthly.insert(0, "FactorCode", code)
            sub_monthly_frames.append(sub_monthly)
        if not sub_quintiles.empty:
            sub_quintiles.insert(0, "FactorGroup", group_name)
            sub_quintiles.insert(0, "FactorCode", code)
            sub_quintile_frames.append(sub_quintiles)
        if code in layer1_input.columns:
            raw_summary, _, _, _ = _evaluate_signal(
                base, layer1_input[code], signal_name=code, scope="raw_input", group_columns=["Date"], config=config
            )
            raw_vs_rows.append({
                "FactorCode": code,
                "FactorGroup": group_name,
                "RawMeanRankIC": raw_summary.get("MeanRankIC"),
                "SubScoreMeanRankIC": sub_summary.get("MeanRankIC"),
                "RankICChange": sub_summary.get("MeanRankIC", np.nan) - raw_summary.get("MeanRankIC", np.nan),
                "RawQ5MinusQ1Mean": raw_summary.get("Q5MinusQ1Mean"),
                "SubScoreQ5MinusQ1Mean": sub_summary.get("Q5MinusQ1Mean"),
                "Q5MinusQ1Change": sub_summary.get("Q5MinusQ1Mean", np.nan) - raw_summary.get("Q5MinusQ1Mean", np.nan),
                "RawPositiveRate": raw_summary.get("RankICPositiveRate"),
                "SubScorePositiveRate": sub_summary.get("RankICPositiveRate"),
            })

    # Leave-one-group-out contribution to S06.
    leave_rows = []
    if not factor_scores.empty:
        full_prediction = factor_scores.mean(axis=1, skipna=True)
        full_summary, _, _, _ = _evaluate_signal(base, full_prediction, signal_name="S06_AllGroups", scope="global", group_columns=["Date"], config=config)
        for excluded in factor_scores.columns:
            remaining = [column for column in factor_scores.columns if column != excluded]
            if not remaining:
                continue
            prediction = factor_scores[remaining].mean(axis=1, skipna=True)
            summary, _, _, _ = _evaluate_signal(base, prediction, signal_name=f"S06_without_{excluded}", scope="global", group_columns=["Date"], config=config)
            leave_rows.append({
                "ExcludedFactorGroup": excluded,
                "FullS06MeanRankIC": full_summary.get("MeanRankIC"),
                "WithoutGroupMeanRankIC": summary.get("MeanRankIC"),
                "IncrementalMeanRankIC": full_summary.get("MeanRankIC", np.nan) - summary.get("MeanRankIC", np.nan),
                "FullS06Q5MinusQ1Mean": full_summary.get("Q5MinusQ1Mean"),
                "WithoutGroupQ5MinusQ1Mean": summary.get("Q5MinusQ1Mean"),
                "IncrementalQ5MinusQ1Mean": full_summary.get("Q5MinusQ1Mean", np.nan) - summary.get("Q5MinusQ1Mean", np.nan),
            })

    # Coverage and dispersion by period and FactorGroup.
    coverage_rows = []
    for factor in factor_scores.columns:
        for date, idx in base.groupby("Date").groups.items():
            values = pd.to_numeric(factor_scores.loc[idx, factor], errors="coerce")
            coverage_rows.append({
                "Date": date,
                "FactorGroup": factor,
                "CoverageRate": float(values.notna().mean()),
                "ObservationCount": int(values.notna().sum()),
                "MeanScore": float(values.mean()),
                "MedianScore": float(values.median()),
                "ScoreStd": float(values.std(ddof=1)),
                "ScoreMin": float(values.min()),
                "ScoreMax": float(values.max()),
            })

    return FactorScorePerformanceDiagnostics(
        factor_group_summary=pd.DataFrame(group_summary_rows),
        factor_group_monthly_ic=pd.concat(group_monthly_frames, ignore_index=True) if group_monthly_frames else pd.DataFrame(),
        factor_group_quintiles=pd.concat(group_quintile_frames, ignore_index=True) if group_quintile_frames else pd.DataFrame(),
        factor_group_long_short=pd.concat(group_ls_frames, ignore_index=True) if group_ls_frames else pd.DataFrame(),
        factor_group_calibration=pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame(),
        subscore_summary=pd.DataFrame(sub_summary_rows),
        subscore_monthly_ic=pd.concat(sub_monthly_frames, ignore_index=True) if sub_monthly_frames else pd.DataFrame(),
        subscore_quintiles=pd.concat(sub_quintile_frames, ignore_index=True) if sub_quintile_frames else pd.DataFrame(),
        raw_vs_subscore=pd.DataFrame(raw_vs_rows),
        country_factor_group=pd.DataFrame(country_rows),
        country_sector_factor_group=pd.DataFrame(country_sector_rows),
        factor_group_correlation=_monthly_correlation(factor_scores, base, minimum),
        leave_one_group_out=pd.DataFrame(leave_rows),
        coverage_dispersion=pd.DataFrame(coverage_rows),
        common_oos_keys=common_keys,
    )


def _write_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    output = frame if frame is not None and not frame.empty else pd.DataFrame({"Message": ["該当データなし"]})
    output.to_excel(writer, sheet_name=name[:31], index=False)
    worksheet = writer.sheets[name[:31]]
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(1, len(output)), max(0, len(output.columns) - 1))
    worksheet.set_column(0, max(0, len(output.columns) - 1), 18)


def write_factor_score_performance_excel(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    metas: dict[str, FactorMeta],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_factor_score_performance_diagnostics(data, scenarios, diagnostics, metas, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        readme = pd.DataFrame({
            "Sheet": [
                "FactorGroup_Summary", "FactorGroup_MonthlyIC", "FactorGroup_Quintiles",
                "FactorGroup_LongShort", "FactorGroup_Calibration", "SubScore_Summary",
                "SubScore_MonthlyIC", "SubScore_Quintiles", "Raw_vs_SubScore",
                "Country_FactorGroup", "CountrySector_FactorGroup", "FactorGroup_Correlation",
                "LeaveOneGroupOut", "Coverage_Dispersion", "Common_OOS_Keys",
            ],
            "Content": [
                "集約FactorScoreのグローバル・国別内・国×セクター内の予測力",
                "FactorScoreの月次Rank IC", "FactorScore別Q1-Q5翌月リターン",
                "FactorScore別Q5-Q1時系列・累積値", "Time-averaged binsによるスコアと実現リターン",
                "単一FAのOOF SubScore予測力", "単一FAの月次Rank IC", "単一FAのQ1-Q5翌月リターン",
                "回帰前Input ScoreとOOF SubScoreの比較", "国別FactorScore予測力",
                "国×セクター別FactorScore予測力", "FactorScore間の月次Spearman相関平均",
                "S06からFactor Groupを1つ除外した増分分析", "FactorScoreのカバレッジ・分散推移",
                "S06・S07で共通するDate×ISIN評価集合",
            ],
        })
        _write_sheet(writer, "README", readme)
        for name, frame in {
            "FactorGroup_Summary": result.factor_group_summary,
            "FactorGroup_MonthlyIC": result.factor_group_monthly_ic,
            "FactorGroup_Quintiles": result.factor_group_quintiles,
            "FactorGroup_LongShort": result.factor_group_long_short,
            "FactorGroup_Calibration": result.factor_group_calibration,
            "SubScore_Summary": result.subscore_summary,
            "SubScore_MonthlyIC": result.subscore_monthly_ic,
            "SubScore_Quintiles": result.subscore_quintiles,
            "Raw_vs_SubScore": result.raw_vs_subscore,
            "Country_FactorGroup": result.country_factor_group,
            "CountrySector_FactorGroup": result.country_sector_factor_group,
            "FactorGroup_Correlation": result.factor_group_correlation,
            "LeaveOneGroupOut": result.leave_one_group_out,
            "Coverage_Dispersion": result.coverage_dispersion,
            "Common_OOS_Keys": result.common_oos_keys,
        }.items():
            _write_sheet(writer, name, frame)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        for worksheet in writer.sheets.values():
            worksheet.set_row(0, 22, header)


def _plot_heatmap(ax: plt.Axes, frame: pd.DataFrame, title: str, value_label: str, vmin: float | None = None, vmax: float | None = None) -> None:
    if frame.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "該当データなし", ha="center", va="center")
        return
    values = frame.to_numpy(float)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(image, ax=ax, label=value_label, fraction=0.025, pad=0.02)


def write_factor_score_performance_pdf(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    metas: dict[str, FactorMeta],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_factor_score_performance_diagnostics(data, scenarios, diagnostics, metas, config)
    if result.factor_group_summary.empty:
        return
    _setup_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.get("factor_score_performance_diagnostics", {})
    rolling = int(cfg.get("rolling_rank_ic_periods", 12))
    qn = int(cfg.get("quantiles", config.get("evaluation", {}).get("quintiles", 5)))
    top_n = int(cfg.get("subscore_top_n_pdf", 12))

    with PdfPages(output_path) as pdf:
        # 1: overview bars.
        global_summary = result.factor_group_summary[result.factor_group_summary["Scope"].eq("global")].copy()
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
        order = global_summary.sort_values("MeanRankIC")["Signal"]
        axes[0].barh(order, global_summary.set_index("Signal").loc[order, "MeanRankIC"])
        axes[0].axvline(0, linewidth=0.8)
        axes[0].set_title("集約FactorScore別 平均Rank IC", fontweight="bold")
        axes[0].set_xlabel("平均Rank IC")
        axes[0].grid(axis="x", alpha=0.2)
        order_ls = global_summary.sort_values("Q5MinusQ1Mean")["Signal"]
        axes[1].barh(order_ls, global_summary.set_index("Signal").loc[order_ls, "Q5MinusQ1Mean"])
        axes[1].axvline(0, linewidth=0.8)
        axes[1].set_title("集約FactorScore別 Q5-Q1平均リターン", fontweight="bold")
        axes[1].set_xlabel("翌月平均リターン")
        axes[1].grid(axis="x", alpha=0.2)
        fig.suptitle("集約FactorScoreのリターン予測力 - 共通OOS", fontsize=16, fontweight="bold")
        fig.text(0.02, 0.02, "高いFactorScoreの銘柄が翌月に高いリターンを得たかを、Rank ICと分位差で確認。", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2: rolling IC.
        monthly = result.factor_group_monthly_ic[result.factor_group_monthly_ic["Scope"].eq("global")]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for factor, frame in monthly.groupby("FactorGroup"):
            series = frame.groupby("Date")["RankIC"].mean().sort_index()
            ax.plot(series.index, series.rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=factor, linewidth=1.5)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"集約FactorScore別 ローリング{rolling}期間平均Rank IC", fontsize=15, fontweight="bold")
        ax.set_ylabel("Rank IC")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3+: quintile mean returns, up to 6 groups per page.
        quintiles = result.factor_group_quintiles[result.factor_group_quintiles["Scope"].eq("global")]
        factors = sorted(quintiles["FactorGroup"].unique()) if not quintiles.empty else []
        for start in range(0, len(factors), 6):
            subset = factors[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
            for ax, factor in zip(axes.ravel(), subset):
                frame = quintiles[quintiles["FactorGroup"].eq(factor)]
                mean_return = frame.groupby("Quintile")["Return"].mean().reindex(range(1, qn + 1))
                ax.bar([f"Q{q}" for q in range(1, qn + 1)], mean_return)
                ax.axhline(0, linewidth=0.7)
                ax.set_title(factor, fontweight="bold")
                ax.set_ylabel("翌月平均リターン")
                ax.grid(axis="y", alpha=0.2)
            for ax in axes.ravel()[len(subset):]:
                ax.axis("off")
            fig.suptitle("高いFactorScoreは高い翌月リターンにつながったか - 5分位比較", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Cumulative long-short.
        long_short = result.factor_group_long_short[result.factor_group_long_short["Scope"].eq("global")]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for factor, frame in long_short.groupby("FactorGroup"):
            frame = frame.sort_values("Date")
            ax.plot(frame["Date"], frame["CumulativeWealth"], label=factor, linewidth=1.5)
        ax.axhline(1, linewidth=0.8)
        ax.set_title("FactorScore別 Q5-Q1累積リターン", fontsize=15, fontweight="bold")
        ax.set_ylabel("累積資産価値")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Time-averaged calibration / binscatter.
        calibration = result.factor_group_calibration
        factors = sorted(calibration["Signal"].unique()) if not calibration.empty else []
        for start in range(0, len(factors), 6):
            subset = factors[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
            for ax, factor in zip(axes.ravel(), subset):
                frame = calibration[calibration["Signal"].eq(factor)].sort_values("Bin")
                ax.scatter(frame["MeanSignal"], frame["MeanNextMonthReturn"], s=35)
                if len(frame) >= 2:
                    coef = np.polyfit(frame["MeanSignal"], frame["MeanNextMonthReturn"], 1)
                    xline = np.linspace(frame["MeanSignal"].min(), frame["MeanSignal"].max(), 100)
                    ax.plot(xline, coef[0] * xline + coef[1], linewidth=1.2)
                    fitted = coef[0] * frame["MeanSignal"] + coef[1]
                    ss_res = float(((frame["MeanNextMonthReturn"] - fitted) ** 2).sum())
                    ss_tot = float(((frame["MeanNextMonthReturn"] - frame["MeanNextMonthReturn"].mean()) ** 2).sum())
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    ax.text(0.03, 0.95, f"傾き={coef[0]:.4f}\nR2={r2:.3f}", transform=ax.transAxes, va="top", fontsize=8)
                ax.set_title(factor, fontweight="bold")
                ax.set_xlabel("ビン平均FactorScore")
                ax.set_ylabel("ビン平均翌月リターン")
                ax.grid(alpha=0.2)
            for ax in axes.ravel()[len(subset):]:
                ax.axis("off")
            fig.suptitle("FactorScoreのTime-Averaged Binscatter", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Country x factor heatmap.
        country = result.country_factor_group.pivot(index="Country", columns="FactorGroup", values="MeanRankIC") if not result.country_factor_group.empty else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        _plot_heatmap(ax, country, "国別・FactorScore別 平均Rank IC", "平均Rank IC", vmin=-0.10, vmax=0.10)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Correlation heatmap.
        correlation = result.factor_group_correlation.pivot(index="FactorGroup1", columns="FactorGroup2", values="MeanSpearmanCorrelation") if not result.factor_group_correlation.empty else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_heatmap(ax, correlation, "FactorScore間の平均Spearman相関", "相関係数", vmin=-1, vmax=1)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Leave-one-out contribution.
        leave = result.leave_one_group_out.sort_values("IncrementalMeanRankIC")
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
        axes[0].barh(leave["ExcludedFactorGroup"], leave["IncrementalMeanRankIC"])
        axes[0].axvline(0, linewidth=0.8)
        axes[0].set_title("S06への増分Rank IC寄与", fontweight="bold")
        axes[0].set_xlabel("全グループIC - 除外後IC")
        axes[0].grid(axis="x", alpha=0.2)
        axes[1].barh(leave["ExcludedFactorGroup"], leave["IncrementalQ5MinusQ1Mean"])
        axes[1].axvline(0, linewidth=0.8)
        axes[1].set_title("S06への増分Q5-Q1寄与", fontweight="bold")
        axes[1].set_xlabel("全グループQ5-Q1 - 除外後Q5-Q1")
        axes[1].grid(axis="x", alpha=0.2)
        fig.suptitle("Factor Group除外分析 - 負値は除外した方が良かったことを示す", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Raw vs SubScore comparison.
        comparison = result.raw_vs_subscore.dropna(subset=["RankICChange"]).copy()
        if not comparison.empty:
            comparison = comparison.sort_values("RankICChange")
            selected = pd.concat([comparison.head(top_n // 2), comparison.tail(top_n - top_n // 2)]).drop_duplicates()
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.barh(selected["FactorCode"], selected["RankICChange"])
            ax.axvline(0, linewidth=0.8)
            ax.set_title("単一FA回帰によるRank IC変化 - SubScore対回帰前Input Score", fontsize=15, fontweight="bold")
            ax.set_xlabel("SubScore Mean Rank IC - Raw/Input Mean Rank IC")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top/Bottom SubScore predictive power.
        sub = result.subscore_summary.dropna(subset=["MeanRankIC"]).sort_values("MeanRankIC")
        if not sub.empty:
            selected = pd.concat([sub.head(top_n // 2), sub.tail(top_n - top_n // 2)]).drop_duplicates()
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.barh(selected["FactorCode"], selected["MeanRankIC"])
            ax.axvline(0, linewidth=0.8)
            ax.set_title("OOF SubScore別 平均Rank IC - 上位・下位", fontsize=15, fontweight="bold")
            ax.set_xlabel("平均Rank IC")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
