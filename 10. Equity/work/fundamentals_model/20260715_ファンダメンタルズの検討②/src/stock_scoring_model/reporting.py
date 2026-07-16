from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .evaluation import cumulative_long_short, cumulative_quintile_returns
from .scenarios import ScenarioResult


def _setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9


def write_quintile_pdf(quintiles: pd.DataFrame, output_path: Path, config: dict[str, Any]) -> None:
    _setup_matplotlib()
    curves = cumulative_quintile_returns(quintiles)
    with PdfPages(output_path) as pdf:
        for scenario, frame in curves.items():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for q in sorted(frame.columns):
                ax.plot(frame.index, frame[q], label=f"Q{int(q)}", linewidth=1.8)
            ax.set_title(f"{scenario} | スコア5分位ポートフォリオ累積リターン", fontsize=15, fontweight="bold")
            ax.set_ylabel("Cumulative Wealth")
            ax.set_xlabel("Date")
            ax.grid(alpha=0.2)
            ax.legend(ncol=5, loc="upper left")
            ax.text(0.01, 0.01, "Q1=最低スコア20%、Q5=最高スコア20%。各月等ウェイト、翌期リターンで評価。", transform=ax.transAxes, fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_scenario_comparison_pdf(
    summary: pd.DataFrame,
    quintiles: pd.DataFrame,
    rank_ic_history: pd.DataFrame,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    ls_curves = cumulative_long_short(quintiles, int(config["evaluation"].get("quintiles", 5)))
    rolling = int(config["evaluation"].get("rolling_rank_ic_periods", 12))
    with PdfPages(output_path) as pdf:
        # Page 1: 共通OOS期間のLS cumulative（シナリオ間の公平比較）
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        common_start = pd.to_datetime(summary["CommonStartDate"].dropna().iloc[0]) if not summary.empty and summary["CommonStartDate"].notna().any() else None
        common_end = pd.to_datetime(summary["CommonEndDate"].dropna().iloc[0]) if not summary.empty and summary["CommonEndDate"].notna().any() else None
        for scenario, series in ls_curves.items():
            s = series
            if common_start is not None and common_end is not None:
                s = s[(pd.to_datetime(s.index) >= common_start) & (pd.to_datetime(s.index) <= common_end)]
                if not s.empty:
                    s = s / s.iloc[0]
            ax.plot(s.index, s.values, label=scenario, linewidth=1.7)
        period_note = f"共通OOS期間: {common_start.date()} - {common_end.date()}" if common_start is not None and common_end is not None else "共通OOS期間なし"
        ax.set_title("シナリオ別 Q5-Q1 累積リターン（共通OOS期間）", fontsize=15, fontweight="bold")
        ax.text(0.01, 0.01, period_note, transform=ax.transAxes, fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Cumulative Wealth")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: 各シナリオの利用可能期間
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for scenario, s in ls_curves.items():
            ax.plot(s.index, s.values, label=scenario, linewidth=1.7)
        ax.set_title("シナリオ別 Q5-Q1 累積リターン（各モデル利用可能期間）", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Cumulative Wealth")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: rolling rank IC
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for scenario, g in rank_ic_history.groupby("Scenario"):
            g = g.sort_values("Date")
            ax.plot(g["Date"], g["RankIC"].rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=scenario, linewidth=1.5)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"シナリオ別 ローリング{rolling}期間平均RankIC", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: 共通OOS期間のsummary bars
        if not summary.empty:
            metrics = ["CommonMeanRankIC", "CommonQ5MinusQ1Sharpe", "CommonQuintileMonotonicity", "CommonQ5MinusQ1MaxDrawdown"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, metric in zip(axes.ravel(), metrics):
                plot_df = summary.sort_values(metric, ascending=metric == "Q5MinusQ1MaxDrawdown")
                ax.barh(plot_df["Scenario"], plot_df[metric])
                ax.set_title(metric)
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle("個別銘柄スコアリングモデルの主要比較指標（共通OOS期間）", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_dataframe(writer: pd.ExcelWriter, df: pd.DataFrame, sheet: str) -> None:
    if df is None or df.empty:
        pd.DataFrame({"Message": ["No data"]}).to_excel(writer, sheet_name=sheet, index=False)
    else:
        df.to_excel(writer, sheet_name=sheet[:31], index=False)


def write_analysis_summary(
    output_path: Path,
    summary: pd.DataFrame,
    quintiles: pd.DataFrame,
    rank_ic: pd.DataFrame,
    bins_summary: pd.DataFrame,
    bins_points: pd.DataFrame,
    feature_lineage: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        readme = pd.DataFrame({
            "Sheet": ["Scenario_Comparison", "Quintile_Summary", "RankIC_History", "Binscatter_Regression", "Binscatter_Bin_Points", "Feature_Lineage"],
            "Content": [
                "各スコアリングシナリオの主要評価指標",
                "シナリオ・分位別のリターン集計",
                "月次RankIC履歴",
                "Time-averaged binscatterの回帰・R2・相関結果",
                "Time-averaged後の各ビン座標と標準誤差",
                "元FAコードから派生特徴量への系譜",
            ],
        })
        readme.to_excel(writer, sheet_name="README", index=False)
        _write_dataframe(writer, summary, "Scenario_Comparison")
        qsum = quintiles.groupby(["Scenario", "Quintile"]).agg(MeanReturn=("Return", "mean"), StdReturn=("Return", "std"), Periods=("Date", "nunique")).reset_index() if not quintiles.empty else pd.DataFrame()
        _write_dataframe(writer, qsum, "Quintile_Summary")
        _write_dataframe(writer, rank_ic, "RankIC_History")
        _write_dataframe(writer, bins_summary, "Binscatter_Regression")
        _write_dataframe(writer, bins_points, "Binscatter_Bin_Points")
        _write_dataframe(writer, feature_lineage, "Feature_Lineage")
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        pct = workbook.add_format({"num_format": "0.00%"})
        dec = workbook.add_format({"num_format": "0.0000"})
        for name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
            worksheet.autofilter(0, 0, max(0, worksheet.dim_rowmax), max(0, worksheet.dim_colmax))
            worksheet.set_column(0, max(0, worksheet.dim_colmax), 18)
        if "Scenario_Comparison" in writer.sheets:
            writer.sheets["Scenario_Comparison"].set_column("C:K", 18, dec)
        if "Quintile_Summary" in writer.sheets:
            writer.sheets["Quintile_Summary"].set_column("C:D", 16, pct)


def write_scenario_excels(results: dict[str, ScenarioResult], output_dir: Path, config: dict[str, Any]) -> None:
    settings = config["outputs"].get("scenario_excel", {})
    if not settings.get("enabled", True):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    date_scope = settings.get("date_scope", "latest")
    for scenario, result in results.items():
        stock = result.stock_scores.copy()
        factors = result.factor_scores.copy()
        subs = result.sub_scores.copy()
        if date_scope == "latest" and not stock.empty:
            latest = stock["Date"].max()
            stock = stock[stock["Date"] == latest]
            factors = factors[factors["Date"] == latest]
            subs = subs[subs["Date"] == latest]
        path = output_dir / f"{scenario}.xlsx"
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            pd.DataFrame({
                "Item": ["Scenario", "DateScope", "StockScoreColumns"],
                "Value": [scenario, date_scope, "Date, ISIN, Currency, MarketCap, TotalScore, Prediction, NextMonthReturn, Quintile"],
            }).to_excel(writer, sheet_name="README", index=False)
            stock.to_excel(writer, sheet_name="StockScore_001", index=False)
            if settings.get("include_sub_scores", True):
                subs.to_excel(writer, sheet_name="SubScore_001", index=False)
            if settings.get("include_factor_scores", True):
                factors.to_excel(writer, sheet_name="FactorScore_001", index=False)
            if not result.weight_history.empty:
                result.weight_history.to_excel(writer, sheet_name="FactorWeights", index=False)
            if not result.model_selection.empty:
                result.model_selection.to_excel(writer, sheet_name="ModelSelection", index=False)
