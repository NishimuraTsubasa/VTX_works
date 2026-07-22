from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from .evaluation import cumulative_long_short, cumulative_quintile_returns, rank_ic_delta_table
from .master import FactorMeta
from .core_reporting import _setup_matplotlib
from .scenarios import ScenarioResult


def _write_frame(writer: pd.ExcelWriter, frame: pd.DataFrame, sheet_name: str) -> None:
    if frame is None or frame.empty:
        pd.DataFrame({"Message": ["No data"]}).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    else:
        frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _safe_spearman(x: pd.Series, y: pd.Series, minimum: int = 10) -> float:
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(temp) < minimum or temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(temp["x"], temp["y"]).statistic)


def _quantile(score: pd.Series, q: int) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce")
    valid = x.notna()
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    if int(valid.sum()) >= q:
        out.loc[valid] = pd.qcut(x.loc[valid].rank(method="first"), q=q, labels=range(1, q + 1)).astype(int)
    return out


def write_analysis_summary_v13(
    output_path: Path,
    summary: pd.DataFrame,
    quintiles: pd.DataFrame,
    rank_ic: pd.DataFrame,
    common_quintiles: pd.DataFrame,
    common_rank_ic: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    benchmark = str(config["evaluation"].get("common_oos", {}).get("benchmark_scenario", "N00_Direct_RawScore_EW"))
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        readme = pd.DataFrame({
            "Sheet": [
                "Scenario_Comparison", "Quintile_Summary", "RankIC_History", "Common_Quintiles",
                "Common_RankIC", "Common_RankIC_Delta", "Latest_Layer2_Weights", "Factor_Return_Summary",
            ],
            "Content": [
                "全利用可能期間と共通Date×ISIN OOSの主要指標",
                "全期間の分位別リターン要約",
                "全期間の月次Rank IC",
                "共通OOSの分位別リターン",
                "共通OOSの月次Rank IC",
                f"共通OOSにおける{benchmark}との差",
                "最新時点のQ5-Q1 Factor Return相関調整ウェイト",
                "FA別Q5-Q1 Factor Returnの平均・標準偏差・正符号率",
            ],
        })
        readme.to_excel(writer, sheet_name="README", index=False)
        _write_frame(writer, summary, "Scenario_Comparison")
        qsum = (
            quintiles.groupby(["Scenario", "Quintile"]).agg(
                MeanReturn=("Return", "mean"), StdReturn=("Return", "std"), Periods=("Date", "nunique")
            ).reset_index()
            if not quintiles.empty else pd.DataFrame()
        )
        _write_frame(writer, qsum, "Quintile_Summary")
        _write_frame(writer, rank_ic, "RankIC_History")
        _write_frame(writer, common_quintiles, "Common_Quintiles")
        _write_frame(writer, common_rank_ic, "Common_RankIC")
        _write_frame(writer, rank_ic_delta_table(common_rank_ic, benchmark), "Common_RankIC_Delta")
        weights = diagnostics.get("Layer2Weights", pd.DataFrame())
        latest_weights = weights[weights["Date"].eq(weights["Date"].max())].copy() if isinstance(weights, pd.DataFrame) and not weights.empty else pd.DataFrame()
        _write_frame(writer, latest_weights, "Latest_Layer2_Weights")
        factor_returns = diagnostics.get("FactorReturnHistory", pd.DataFrame())
        fr_summary = (
            factor_returns.groupby(["FactorGroup", "FactorCode"]).agg(
                MeanFactorReturn=("FactorReturn", "mean"),
                FactorReturnStd=("FactorReturn", "std"),
                PositiveRate=("FactorReturn", lambda s: float((s > 0).mean())),
                Periods=("Date", "nunique"),
            ).reset_index()
            if isinstance(factor_returns, pd.DataFrame) and not factor_returns.empty else pd.DataFrame()
        )
        _write_frame(writer, fr_summary, "Factor_Return_Summary")
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_scenario_excels_v13(results: dict[str, ScenarioResult], output_dir: Path, config: dict[str, Any]) -> None:
    cfg = config["outputs"].get("scenario_excel", {})
    if not bool(cfg.get("enabled", True)):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    date_scope = str(cfg.get("date_scope", "latest"))
    for name, result in results.items():
        stock = result.stock_scores.copy()
        if date_scope == "latest" and not stock.empty:
            stock = stock[stock["Date"].eq(stock["Date"].max())]
        factor = result.factor_scores.copy()
        if date_scope == "latest" and not factor.empty:
            factor = factor[factor["Date"].eq(factor["Date"].max())]
        factor_wide = factor.pivot_table(index=["Date", "ISIN"], columns="FactorGroup", values="FactorScore", aggfunc="first") if not factor.empty else pd.DataFrame()
        if not factor_wide.empty:
            factor_wide.columns = [f"FactorScore__{col}" for col in factor_wide.columns]
            stock = stock.merge(factor_wide.reset_index(), on=["Date", "ISIN"], how="left")
        with pd.ExcelWriter(output_dir / f"{name}.xlsx", engine="xlsxwriter") as writer:
            pd.DataFrame({
                "Column": list(stock.columns),
                "Description": [
                    "シナリオ名" if c == "Scenario" else
                    "モデルの生予測値または集約スコア" if c == "Prediction" else
                    "順位化された0-1の最終スコア" if c == "TotalScore" else
                    "翌月実現リターン" if c == "NextMonthReturn" else
                    "Aggregate FactorScore" if c.startswith("FactorScore__") else ""
                    for c in stock.columns
                ],
            }).to_excel(writer, sheet_name="README", index=False)
            stock.to_excel(writer, sheet_name="Stock_Scores", index=False)
            for ws in writer.sheets.values():
                ws.freeze_panes(1, 0)
                ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_factor_return_weight_diagnostics_excel(
    output_path: Path,
    diagnostics: dict[str, Any],
) -> None:
    returns = diagnostics.get("FactorReturnHistory", pd.DataFrame())
    weights = diagnostics.get("Layer2Weights", pd.DataFrame())
    correlations = diagnostics.get("FactorReturnCorrelations", pd.DataFrame())
    latest_date = weights["Date"].max() if isinstance(weights, pd.DataFrame) and not weights.empty else pd.NaT
    latest_weights = weights[weights["Date"].eq(latest_date)] if pd.notna(latest_date) else pd.DataFrame()
    latest_corr = correlations[correlations["Date"].eq(correlations["Date"].max())] if isinstance(correlations, pd.DataFrame) and not correlations.empty else pd.DataFrame()
    concentration = (
        weights.groupby(["Date", "FactorGroup"]).agg(
            MaxWeight=("Weight", "max"),
            Herfindahl=("Weight", lambda s: float(np.square(s).sum())),
            EffectiveFactorCount=("Weight", lambda s: float(1.0 / np.square(s).sum()) if np.square(s).sum() > 0 else np.nan),
        ).reset_index()
        if isinstance(weights, pd.DataFrame) and not weights.empty else pd.DataFrame()
    )
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": ["Factor_Return_History", "Latest_Weights", "Weight_History", "Latest_Correlation", "Correlation_History", "Weight_Concentration"],
            "Content": [
                "FA別のQ5-Q1翌月リターン履歴",
                "最新時点の相関調整・EW縮小後ウェイト",
                "全時点のウェイトと採用理由",
                "最新時点で使用されたFactor Return相関",
                "相関行列履歴",
                "最大ウェイト・HHI・実効FA数",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        _write_frame(writer, returns, "Factor_Return_History")
        _write_frame(writer, latest_weights, "Latest_Weights")
        _write_frame(writer, weights, "Weight_History")
        _write_frame(writer, latest_corr, "Latest_Correlation")
        _write_frame(writer, correlations, "Correlation_History")
        _write_frame(writer, concentration, "Weight_Concentration")
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_factor_return_weight_diagnostics_pdf(
    output_path: Path,
    diagnostics: dict[str, Any],
) -> None:
    returns = diagnostics.get("FactorReturnHistory", pd.DataFrame())
    weights = diagnostics.get("Layer2Weights", pd.DataFrame())
    correlations = diagnostics.get("FactorReturnCorrelations", pd.DataFrame())
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return
    _setup_matplotlib()
    with PdfPages(output_path) as pdf:
        latest = weights[weights["Date"].eq(weights["Date"].max())]
        groups = sorted(latest["FactorGroup"].unique())
        for start in range(0, len(groups), 6):
            subset = groups[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
            for ax, group in zip(axes.ravel(), subset):
                frame = latest[latest["FactorGroup"].eq(group)].sort_values("Weight")
                ax.barh(frame["FactorCode"], frame["Weight"])
                ax.set_title(group, fontweight="bold")
                ax.set_xlabel("最新ウェイト")
                ax.grid(axis="x", alpha=0.2)
            for ax in axes.ravel()[len(subset):]:
                ax.axis("off")
            fig.suptitle("Q5-Q1 Factor Return相関に基づく最新グループ内ウェイト", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if isinstance(returns, pd.DataFrame) and not returns.empty:
            for group, frame in returns.groupby("FactorGroup"):
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                for code, series in frame.groupby("FactorCode"):
                    series = series.sort_values("Date")
                    wealth = (1.0 + series["FactorReturn"].fillna(0.0)).cumprod()
                    ax.plot(series["Date"], wealth, label=code, linewidth=1.4)
                ax.axhline(1.0, linewidth=0.8)
                ax.set_title(f"{group} | FA別Q5-Q1 Factor Return累積推移", fontsize=15, fontweight="bold")
                ax.set_ylabel("累積資産価値")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, ncol=3, loc="upper left")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for group, frame in weights.groupby("FactorGroup"):
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for code, series in frame.groupby("FactorCode"):
                series = series.sort_values("Date")
                ax.plot(series["Date"], series["Weight"], label=code, linewidth=1.3)
            ax.set_title(f"{group} | グループ内ウェイト推移", fontsize=15, fontweight="bold")
            ax.set_ylabel("Weight")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=3, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if isinstance(correlations, pd.DataFrame) and not correlations.empty:
            latest_corr = correlations[correlations["Date"].eq(correlations["Date"].max())]
            for group, frame in latest_corr.groupby("FactorGroup"):
                pivot = frame.pivot(index="FactorCode1", columns="FactorCode2", values="FactorReturnCorrelation")
                fig, ax = plt.subplots(figsize=(9, 7))
                image = ax.imshow(pivot.to_numpy(float), vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)), pivot.index)
                ax.set_title(f"{group} | 最新Q5-Q1 Factor Return相関", fontsize=14, fontweight="bold")
                fig.colorbar(image, ax=ax, label="相関係数")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _aggregate_factor_diagnostics(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    c = config["columns"]
    qn = int(config.get("diagnostics", {}).get("factor_score_quantiles", 5))
    base = pd.DataFrame({
        "Date": pd.to_datetime(data[c["date"]]),
        "ISIN": data[c["isin"]].astype(str),
        "Country": data[c["country"]].astype(str),
        "NextMonthReturn": data["NextMonthReturn"],
    })
    monthly_rows: list[dict[str, object]] = []
    quintile_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    for group in factor_scores.columns:
        signal = pd.to_numeric(factor_scores[group], errors="coerce")
        temp = base.assign(Signal=signal).dropna(subset=["Signal", "NextMonthReturn"])
        for date, frame in temp.groupby("Date"):
            monthly_rows.append({
                "Date": date, "FactorGroup": group,
                "RankIC": _safe_spearman(frame["Signal"], frame["NextMonthReturn"]),
                "ObservationCount": len(frame),
            })
            frame = frame.copy()
            frame["Quintile"] = _quantile(frame["Signal"], qn)
            for quantile, qframe in frame.dropna(subset=["Quintile"]).groupby("Quintile"):
                quintile_rows.append({
                    "Date": date, "FactorGroup": group, "Quintile": int(quantile),
                    "Return": float(qframe["NextMonthReturn"].mean()), "Count": len(qframe),
                    "MeanFactorScore": float(qframe["Signal"].mean()),
                })
        for (country, date), frame in temp.groupby(["Country", "Date"]):
            if len(frame) >= 15:
                country_rows.append({
                    "Country": country, "Date": date, "FactorGroup": group,
                    "RankIC": _safe_spearman(frame["Signal"], frame["NextMonthReturn"]),
                    "ObservationCount": len(frame),
                })
    monthly = pd.DataFrame(monthly_rows)
    quintiles = pd.DataFrame(quintile_rows)
    country_monthly = pd.DataFrame(country_rows)
    summary_rows: list[dict[str, object]] = []
    long_short_rows: list[dict[str, object]] = []
    for group in factor_scores.columns:
        ic = monthly[monthly["FactorGroup"].eq(group)]["RankIC"].dropna()
        qframe = quintiles[quintiles["FactorGroup"].eq(group)]
        pivot = qframe.pivot_table(index="Date", columns="Quintile", values="Return", aggfunc="mean").sort_index() if not qframe.empty else pd.DataFrame()
        ls = pivot[qn] - pivot[1] if 1 in pivot.columns and qn in pivot.columns else pd.Series(dtype=float)
        qmean = pivot.mean() if not pivot.empty else pd.Series(dtype=float)
        summary_rows.append({
            "FactorGroup": group,
            "Periods": int(len(ic)),
            "MeanRankIC": float(ic.mean()) if len(ic) else np.nan,
            "MedianRankIC": float(ic.median()) if len(ic) else np.nan,
            "RankICIR": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
            "RankICPositiveRate": float((ic > 0).mean()) if len(ic) else np.nan,
            "Q5MinusQ1Mean": float(ls.mean()) if len(ls) else np.nan,
            "QuintileMonotonicity": _safe_spearman(pd.Series(qmean.index, dtype=float), pd.Series(qmean.values), minimum=2) if len(qmean) else np.nan,
        })
        if len(ls):
            wealth = (1.0 + ls.fillna(0)).cumprod()
            for date, value in ls.items():
                long_short_rows.append({"Date": date, "FactorGroup": group, "LongShortReturn": value, "CumulativeWealth": wealth.loc[date]})
    country_summary = (
        country_monthly.groupby(["Country", "FactorGroup"]).agg(
            MeanRankIC=("RankIC", "mean"),
            RankICPositiveRate=("RankIC", lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan),
            Periods=("Date", "nunique"),
        ).reset_index()
        if not country_monthly.empty else pd.DataFrame()
    )
    corr_rows: list[dict[str, object]] = []
    for date, idx in data.groupby(c["date"]).groups.items():
        corr = factor_scores.loc[idx].corr(method="spearman")
        for a in corr.index:
            for b in corr.columns:
                corr_rows.append({"Date": pd.Timestamp(date), "FactorGroup1": a, "FactorGroup2": b, "Correlation": corr.loc[a, b]})
    corr_history = pd.DataFrame(corr_rows)
    corr_summary = corr_history.groupby(["FactorGroup1", "FactorGroup2"])["Correlation"].mean().reset_index(name="MeanSpearmanCorrelation") if not corr_history.empty else pd.DataFrame()

    # 全グループ等ウェイトに対するLeave-one-out。N04の説明用診断。
    leave_rows: list[dict[str, object]] = []
    all_score = factor_scores.mean(axis=1, skipna=True)
    all_ic = []
    for date, idx in data.groupby(c["date"]).groups.items():
        all_ic.append(_safe_spearman(all_score.loc[idx], data.loc[idx, "NextMonthReturn"]))
    all_mean_ic = float(pd.Series(all_ic).mean())
    for excluded in factor_scores.columns:
        remaining = [col for col in factor_scores.columns if col != excluded]
        score = factor_scores[remaining].mean(axis=1, skipna=True) if remaining else pd.Series(np.nan, index=factor_scores.index)
        values = []
        for date, idx in data.groupby(c["date"]).groups.items():
            values.append(_safe_spearman(score.loc[idx], data.loc[idx, "NextMonthReturn"]))
        excluded_ic = float(pd.Series(values).mean())
        leave_rows.append({
            "ExcludedFactorGroup": excluded,
            "AllGroupMeanRankIC": all_mean_ic,
            "ExcludedMeanRankIC": excluded_ic,
            "IncrementalMeanRankIC": all_mean_ic - excluded_ic,
        })
    return {
        "Summary": pd.DataFrame(summary_rows),
        "MonthlyIC": monthly,
        "Quintiles": quintiles,
        "LongShort": pd.DataFrame(long_short_rows),
        "CountryMonthlyIC": country_monthly,
        "CountrySummary": country_summary,
        "CorrelationHistory": corr_history,
        "CorrelationSummary": corr_summary,
        "LeaveOneOut": pd.DataFrame(leave_rows),
    }


def write_aggregate_factor_diagnostics_excel(
    output_path: Path,
    data: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    factor_scores = diagnostics.get("Layer2FactorScores", pd.DataFrame())
    result = _aggregate_factor_diagnostics(data, factor_scores, config)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": list(result),
            "Content": [
                "Aggregate FactorScore別の主要予測力",
                "月次Rank IC",
                "Q1-Q5翌月リターン",
                "Q5-Q1リターンと累積推移",
                "国別・月次Rank IC",
                "国別平均Rank IC",
                "FactorScore間相関履歴",
                "FactorScore間平均相関",
                "1系列ずつ除外した増分Rank IC",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        for name, frame in result.items():
            _write_frame(writer, frame, name)
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_aggregate_factor_diagnostics_pdf(
    output_path: Path,
    data: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    factor_scores = diagnostics.get("Layer2FactorScores", pd.DataFrame())
    if not isinstance(factor_scores, pd.DataFrame) or factor_scores.empty:
        return
    result = _aggregate_factor_diagnostics(data, factor_scores, config)
    _setup_matplotlib()
    rolling = int(config.get("diagnostics", {}).get("rolling_rank_ic_periods", 12))
    qn = int(config.get("diagnostics", {}).get("factor_score_quantiles", 5))
    with PdfPages(output_path) as pdf:
        summary = result["Summary"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
        axes[0].barh(summary["FactorGroup"], summary["MeanRankIC"])
        axes[0].axvline(0, linewidth=0.8)
        axes[0].set_title("Aggregate FactorScore別 平均Rank IC", fontweight="bold")
        axes[0].grid(axis="x", alpha=0.2)
        axes[1].barh(summary["FactorGroup"], summary["Q5MinusQ1Mean"])
        axes[1].axvline(0, linewidth=0.8)
        axes[1].set_title("Aggregate FactorScore別 Q5-Q1平均リターン", fontweight="bold")
        axes[1].grid(axis="x", alpha=0.2)
        fig.suptitle("Raw Factor集約後スコアの予測力", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        monthly = result["MonthlyIC"]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for group, frame in monthly.groupby("FactorGroup"):
            series = frame.groupby("Date")["RankIC"].mean().sort_index()
            ax.plot(series.index, series.rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=group, linewidth=1.5)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"Aggregate FactorScore別 ローリング{rolling}期間Rank IC", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        quintiles = result["Quintiles"]
        groups = sorted(quintiles["FactorGroup"].unique()) if not quintiles.empty else []
        for start in range(0, len(groups), 6):
            subset = groups[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
            for ax, group in zip(axes.ravel(), subset):
                frame = quintiles[quintiles["FactorGroup"].eq(group)]
                values = frame.groupby("Quintile")["Return"].mean().reindex(range(1, qn + 1))
                ax.bar([f"Q{i}" for i in range(1, qn + 1)], values)
                ax.axhline(0, linewidth=0.7)
                ax.set_title(group, fontweight="bold")
                ax.set_ylabel("翌月平均リターン")
                ax.grid(axis="y", alpha=0.2)
            for ax in axes.ravel()[len(subset):]:
                ax.axis("off")
            fig.suptitle("高いAggregate FactorScoreは高い翌月リターンにつながったか", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        long_short = result["LongShort"]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for group, frame in long_short.groupby("FactorGroup"):
            frame = frame.sort_values("Date")
            ax.plot(frame["Date"], frame["CumulativeWealth"], label=group, linewidth=1.5)
        ax.axhline(1.0, linewidth=0.8)
        ax.set_title("Aggregate FactorScore別 Q5-Q1累積リターン", fontsize=15, fontweight="bold")
        ax.set_ylabel("累積資産価値")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        country = result["CountrySummary"].pivot(index="Country", columns="FactorGroup", values="MeanRankIC") if not result["CountrySummary"].empty else pd.DataFrame()
        if not country.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            image = ax.imshow(country.to_numpy(float), vmin=-0.1, vmax=0.1, aspect="auto")
            ax.set_xticks(range(len(country.columns)), country.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(country.index)), country.index)
            ax.set_title("国別・Aggregate FactorScore別 平均Rank IC", fontsize=15, fontweight="bold")
            fig.colorbar(image, ax=ax, label="平均Rank IC")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        corr = result["CorrelationSummary"].pivot(index="FactorGroup1", columns="FactorGroup2", values="MeanSpearmanCorrelation") if not result["CorrelationSummary"].empty else pd.DataFrame()
        if not corr.empty:
            fig, ax = plt.subplots(figsize=(9, 7))
            image = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)), corr.index)
            ax.set_title("Aggregate FactorScore間の平均Spearman相関", fontsize=14, fontweight="bold")
            fig.colorbar(image, ax=ax, label="相関係数")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        leave = result["LeaveOneOut"].sort_values("IncrementalMeanRankIC")
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.barh(leave["ExcludedFactorGroup"], leave["IncrementalMeanRankIC"])
        ax.axvline(0, linewidth=0.8)
        ax.set_title("Factor Group除外分析 | 正値は当該系列が全体Rank ICへ寄与", fontsize=15, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _variant_frames(variants: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    for name, payload in variants.items():
        coef = payload.get("CoefficientHistory", pd.DataFrame())
        if isinstance(coef, pd.DataFrame) and not coef.empty:
            temp = coef.copy()
            temp.insert(0, "Variant", name)
            coef_frames.append(temp)
        model = payload.get("ModelHistory", pd.DataFrame())
        if isinstance(model, pd.DataFrame) and not model.empty:
            temp = model.copy()
            temp.insert(0, "Variant", name)
            model_frames.append(temp)
    return (
        pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame(),
        pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame(),
    )


def write_layer3_model_diagnostics_excel(
    output_path: Path,
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    c = config["columns"]
    variants = diagnostics.get("Layer3Variants", {})
    coefficients, models = _variant_frames(variants)
    prediction = data[[c["date"], c["isin"], c["country"], c["sector"], "NextMonthReturn"]].copy().rename(
        columns={c["date"]: "Date", c["isin"]: "ISIN", c["country"]: "Country", c["sector"]: "Sector"}
    )
    distribution_rows: list[dict[str, object]] = []
    for name, payload in variants.items():
        pred = pd.to_numeric(payload.get("Prediction", pd.Series(index=data.index, dtype=float)), errors="coerce")
        prediction[name] = pred
        for (date, country), idx in data.groupby([c["date"], c["country"]]).groups.items():
            p = pred.loc[idx].dropna()
            y = data.loc[idx, "NextMonthReturn"].dropna()
            aligned = pd.concat([pred.loc[idx], data.loc[idx, "NextMonthReturn"]], axis=1).dropna()
            distribution_rows.append({
                "Variant": name, "Date": date, "Country": country,
                "ObservationCount": len(aligned),
                "PredictionMean": p.mean(), "PredictionStd": p.std(ddof=1),
                "PredictionUniqueCount": int(p.nunique()),
                "TargetStd": y.std(ddof=1),
                "PredictionTargetStdRatio": p.std(ddof=1) / y.std(ddof=1) if y.std(ddof=1) and np.isfinite(y.std(ddof=1)) else np.nan,
                "OOSRankIC": _safe_spearman(aligned.iloc[:, 0], aligned.iloc[:, 1]),
            })
    layer3_summary = summary[summary["Scenario"].isin(list(variants))].copy() if not summary.empty else pd.DataFrame()
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": ["Layer3_Summary", "Predictions", "Prediction_Distribution", "Coefficients", "Model_Fit"],
            "Content": [
                "第3層variantの共通OOS比較",
                "銘柄別のOLS/Ridge予測値",
                "国別・月別の予測分散、ユニーク数、実現値分散比",
                "国別・時点別の標準化係数とRaw係数",
                "Train/Validation R2、RankIC、RMSE、Alpha、観測数",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        _write_frame(writer, layer3_summary, "Layer3_Summary")
        _write_frame(writer, prediction, "Predictions")
        _write_frame(writer, pd.DataFrame(distribution_rows), "Prediction_Distribution")
        _write_frame(writer, coefficients, "Coefficients")
        _write_frame(writer, models, "Model_Fit")
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_layer3_model_diagnostics_pdf(
    output_path: Path,
    data: pd.DataFrame,
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    variants = diagnostics.get("Layer3Variants", {})
    if not variants:
        return
    _, models = _variant_frames(variants)
    _setup_matplotlib()
    layer3_summary = summary[summary["Scenario"].isin(list(variants))].copy() if not summary.empty else pd.DataFrame()
    c = config["columns"]
    with PdfPages(output_path) as pdf:
        if not layer3_summary.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
            axes[0].barh(layer3_summary["Scenario"], layer3_summary.get("CommonMeanRankIC", np.nan))
            axes[0].axvline(0, linewidth=0.8)
            axes[0].set_title("第3層 共通OOS平均Rank IC", fontweight="bold")
            axes[0].grid(axis="x", alpha=0.2)
            axes[1].barh(layer3_summary["Scenario"], layer3_summary.get("CommonQ5MinusQ1Mean", np.nan))
            axes[1].axvline(0, linewidth=0.8)
            axes[1].set_title("第3層 共通OOS Q5-Q1平均リターン", fontweight="bold")
            axes[1].grid(axis="x", alpha=0.2)
            fig.suptitle("第3層モデル比較", fontsize=16, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not models.empty:
            latest = models.sort_values("Date").groupby(["Variant", "ScopeLabel"], as_index=False).tail(1)
            metrics = [col for col in ["TrainR2", "ValidationR2", "ValidationSpearman", "Alpha"] if col in latest.columns]
            for metric in metrics:
                pivot = latest.pivot(index="ScopeLabel", columns="Variant", values=metric)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                pivot.plot(kind="bar", ax=ax)
                ax.axhline(0, linewidth=0.8)
                ax.set_title(f"国別最新 {metric}", fontsize=15, fontweight="bold")
                ax.set_xlabel("Country")
                ax.grid(axis="y", alpha=0.2)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        fig, axes = plt.subplots(len(variants), 1, figsize=(11.69, max(4, 3.2 * len(variants))))
        axes = np.atleast_1d(axes)
        for ax, (name, payload) in zip(axes, variants.items()):
            pred = pd.to_numeric(payload["Prediction"], errors="coerce").dropna()
            target = pd.to_numeric(data.loc[pred.index, "NextMonthReturn"], errors="coerce").dropna()
            ax.hist(pred, bins=40, alpha=0.65, density=True, label="予測値")
            if len(target):
                ax.hist(target, bins=40, alpha=0.35, density=True, label="実現リターン")
            ax.set_title(f"{name} | 予測値と実現リターンの分布")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle("第3層予測値の分散・圧縮状態", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_country_factor_score_trends_excel(
    output_path: Path,
    data: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    c = config["columns"]
    scores = diagnostics.get("Layer2FactorScores", pd.DataFrame())
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return
    rows: list[dict[str, object]] = []
    market_cap = pd.to_numeric(data[c["market_cap"]], errors="coerce")
    for (date, country), idx in data.groupby([c["date"], c["country"]]).groups.items():
        for group in scores.columns:
            values = pd.to_numeric(scores.loc[idx, group], errors="coerce")
            valid = values.notna()
            if not valid.any():
                continue
            weights = market_cap.loc[idx][valid].clip(lower=0)
            rows.append({
                "Date": pd.Timestamp(date), "Country": country, "FactorGroup": group,
                "EqualWeightScore": float(values[valid].mean()),
                "MarketCapWeightScore": float(np.average(values[valid], weights=weights)) if weights.sum() > 0 else float(values[valid].mean()),
                "StockCount": int(valid.sum()),
            })
    history = pd.DataFrame(rows).sort_values(["Country", "FactorGroup", "Date"])
    periods = int(config.get("diagnostics", {}).get("country_factor_trailing_z_periods", 36))
    minimum = int(config.get("diagnostics", {}).get("country_factor_minimum_z_periods", 12))
    history["TrailingZ"] = history.groupby(["Country", "FactorGroup"])["EqualWeightScore"].transform(
        lambda s: (s - s.rolling(periods, min_periods=minimum).mean()) / s.rolling(periods, min_periods=minimum).std(ddof=0)
    )
    latest = history.sort_values("Date").groupby(["Country", "FactorGroup"], as_index=False).tail(1)
    top = history.loc[history.groupby(["Date", "Country"])["EqualWeightScore"].idxmax()].copy() if not history.empty else pd.DataFrame()
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": ["Country_Factor_History", "Latest_Country_Factors", "Top_Factor_History"],
            "Content": [
                "国別Aggregate FactorScoreの等ウェイト・時価総額加重・Trailing Z",
                "最新時点の国別FactorScore",
                "各国・各時点で最も高かったFactor Group",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        _write_frame(writer, history, "Country_Factor_History")
        _write_frame(writer, latest, "Latest_Country_Factors")
        _write_frame(writer, top, "Top_Factor_History")
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_country_factor_score_trends_pdf(
    output_path: Path,
    data: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    # Excel構築と同じ集計をインメモリで再現。
    c = config["columns"]
    scores = diagnostics.get("Layer2FactorScores", pd.DataFrame())
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return
    rows: list[dict[str, object]] = []
    for (date, country), idx in data.groupby([c["date"], c["country"]]).groups.items():
        for group in scores.columns:
            values = pd.to_numeric(scores.loc[idx, group], errors="coerce").dropna()
            if len(values):
                rows.append({"Date": pd.Timestamp(date), "Country": country, "FactorGroup": group, "Score": float(values.mean())})
    history = pd.DataFrame(rows)
    if history.empty:
        return
    periods = int(config.get("diagnostics", {}).get("country_factor_trailing_z_periods", 36))
    minimum = int(config.get("diagnostics", {}).get("country_factor_minimum_z_periods", 12))
    history = history.sort_values(["Country", "FactorGroup", "Date"])
    history["TrailingZ"] = history.groupby(["Country", "FactorGroup"])["Score"].transform(
        lambda s: (s - s.rolling(periods, min_periods=minimum).mean()) / s.rolling(periods, min_periods=minimum).std(ddof=0)
    )
    _setup_matplotlib()
    with PdfPages(output_path) as pdf:
        latest = history.sort_values("Date").groupby(["Country", "FactorGroup"], as_index=False).tail(1)
        pivot = latest.pivot(index="Country", columns="FactorGroup", values="TrailingZ")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            image = ax.imshow(pivot.to_numpy(float), vmin=-2.5, vmax=2.5, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_title("国別Aggregate FactorScore | 最新Trailing Z", fontsize=15, fontweight="bold")
            fig.colorbar(image, ax=ax, label="過去推移対比Zスコア")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for country, frame in history.groupby("Country"):
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for group, series in frame.groupby("FactorGroup"):
                series = series.sort_values("Date")
                ax.plot(series["Date"], series["TrailingZ"], label=group, linewidth=1.5)
            ax.axhline(0, linewidth=0.8)
            ax.set_title(f"{country} | Aggregate FactorScore推移（自国過去対比Z）", fontsize=15, fontweight="bold")
            ax.set_ylabel("Trailing Z")
            ax.grid(alpha=0.2)
            ax.legend(ncol=3, fontsize=8, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_model_parameter_summary_excel(
    output_path: Path,
    diagnostics: dict[str, Any],
    metas: dict[str, FactorMeta],
) -> None:
    weights = diagnostics.get("Layer2Weights", pd.DataFrame())
    variants = diagnostics.get("Layer3Variants", {})
    coefficients, models = _variant_frames(variants)
    latest_weights = weights[weights["Date"].eq(weights["Date"].max())].copy() if isinstance(weights, pd.DataFrame) and not weights.empty else pd.DataFrame()
    latest_coefficients = (
        coefficients.sort_values("Date").groupby(["Variant", "ScopeLabel", "Feature"], as_index=False).tail(1)
        if not coefficients.empty else pd.DataFrame()
    )
    map_rows: list[dict[str, object]] = []
    if not latest_weights.empty:
        for row in latest_weights.itertuples(index=False):
            base = {
                "FactorCode": row.FactorCode,
                "FactorGroup": row.FactorGroup,
                "Layer2Weight": row.Weight,
                "Layer2Reason": row.Reason,
                "MeanFactorReturn": row.MeanFactorReturn,
            }
            group_feature = f"{row.FactorGroup}__LIN"
            matches = latest_coefficients[latest_coefficients["Feature"].eq(group_feature)] if not latest_coefficients.empty else pd.DataFrame()
            if matches.empty:
                map_rows.append(base)
            else:
                for coef in matches.itertuples(index=False):
                    map_rows.append({
                        **base,
                        "Layer3Variant": coef.Variant,
                        "Country": coef.ScopeLabel,
                        "Layer3StandardizedCoefficient": coef.StandardizedCoefficient,
                        "Layer3RawCoefficient": coef.RawCoefficient,
                        "Layer3Alpha": coef.Alpha,
                    })
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": ["Latest_Layer2_Weights", "Layer2_Weight_History", "Latest_Layer3_Coefficients", "Layer3_Coefficients", "Layer3_Model_Fit", "EndToEnd_Parameter_Map"],
            "Content": [
                "最新時点のFA→Aggregate FactorScoreウェイト",
                "全時点のLayer2ウェイト",
                "最新国別Layer3係数",
                "全時点のLayer3係数",
                "Train/Validation指標とAlpha",
                "FactorCode→FactorGroup→国別Layer3係数の対応",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        _write_frame(writer, latest_weights, "Latest_Layer2_Weights")
        _write_frame(writer, weights, "Layer2_Weight_History")
        _write_frame(writer, latest_coefficients, "Latest_Layer3_Coefficients")
        _write_frame(writer, coefficients, "Layer3_Coefficients")
        _write_frame(writer, models, "Layer3_Model_Fit")
        _write_frame(writer, pd.DataFrame(map_rows), "EndToEnd_Parameter_Map")
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, ws.dim_colmax), 18)
