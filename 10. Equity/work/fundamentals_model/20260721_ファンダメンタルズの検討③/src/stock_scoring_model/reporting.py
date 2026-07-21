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
    layer1_selection: pd.DataFrame,
    layer2_weights: pd.DataFrame,
    feature_lineage: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        readme = pd.DataFrame({
            "Sheet": ["Scenario_Comparison", "Quintile_Summary", "RankIC_History", "Layer1_Model_Selection", "Layer2_Weights", "Feature_Lineage"],
            "Content": [
                "S00-S07の主要評価指標",
                "シナリオ・分位別リターン集計",
                "月次RankIC履歴",
                "グローバル単一FAのOOFモデル選択履歴",
                "SubScoreからFactorScoreへ集約したウェイト履歴",
                "元FAコードから派生特徴量への系譜",
            ],
        })
        readme.to_excel(writer, sheet_name="README", index=False)
        _write_dataframe(writer, summary, "Scenario_Comparison")
        qsum = quintiles.groupby(["Scenario", "Quintile"]).agg(MeanReturn=("Return", "mean"), StdReturn=("Return", "std"), Periods=("Date", "nunique")).reset_index() if not quintiles.empty else pd.DataFrame()
        _write_dataframe(writer, qsum, "Quintile_Summary")
        _write_dataframe(writer, rank_ic, "RankIC_History")
        _write_dataframe(writer, layer1_selection, "Layer1_Model_Selection")
        _write_dataframe(writer, layer2_weights, "Layer2_Weights")
        _write_dataframe(writer, feature_lineage, "Feature_Lineage")
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        pct = workbook.add_format({"num_format": "0.00%"})
        dec = workbook.add_format({"num_format": "0.0000"})
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
            worksheet.autofilter(0, 0, max(0, worksheet.dim_rowmax), max(0, worksheet.dim_colmax))
            worksheet.set_column(0, max(0, worksheet.dim_colmax), 18)
        if "Scenario_Comparison" in writer.sheets:
            writer.sheets["Scenario_Comparison"].set_column("C:Z", 18, dec)
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


def _component_frame(
    data: pd.DataFrame,
    prediction: pd.Series,
    name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    c = config["columns"]
    out = pd.DataFrame(
        {
            "Scenario": name,
            "Date": data[c["date"]],
            "ISIN": data[c["isin"]],
            "Prediction": prediction,
            "NextMonthReturn": data["NextMonthReturn"],
        }
    )
    out["TotalScore"] = out.groupby("Date")["Prediction"].rank(pct=True)
    qn = int(config["evaluation"].get("quintiles", 5))
    def qcut_safe(s: pd.Series) -> pd.Series:
        valid = s.notna()
        result = pd.Series(pd.NA, index=s.index, dtype="Int64")
        if int(valid.sum()) >= qn:
            result.loc[valid] = pd.qcut(s.loc[valid].rank(method="first"), qn, labels=range(1, qn + 1)).astype(int)
        return result
    out["Quintile"] = out.groupby("Date", group_keys=False)["TotalScore"].apply(qcut_safe).reset_index(level=0, drop=True)
    return out


def _component_evaluation(
    data: pd.DataFrame,
    two_stage: dict[str, pd.DataFrame | pd.Series],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .evaluation import evaluate_scenarios
    from .scenarios import ScenarioResult

    components = {
        "CS_Only": two_stage["CountrySectorStockPrediction"],
        "Within_Only": two_stage["WithinStockPrediction"],
        "Simple_Sum": two_stage["SimpleSumPrediction"],
        "OOF_Ridge_Blend": two_stage["BlendedPrediction"],
    }
    results = {}
    for name, pred in components.items():
        stock = _component_frame(data, pred, name, config)
        results[name] = ScenarioResult(
            stock_scores=stock,
            factor_scores=pd.DataFrame(),
            sub_scores=pd.DataFrame(),
            weight_history=pd.DataFrame(),
            model_selection=pd.DataFrame(),
        )
    return evaluate_scenarios(results, config)


def write_two_stage_diagnostics_pdf(
    data: pd.DataFrame,
    two_stage: dict[str, pd.DataFrame | pd.Series],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    summary, quintiles, rank_ic = _component_evaluation(data, two_stage, config)
    ls_curves = cumulative_long_short(quintiles, int(config["evaluation"].get("quintiles", 5)))
    rolling = int(config["evaluation"].get("rolling_rank_ic_periods", 12))

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for name, series in ls_curves.items():
            ax.plot(series.index, series.values, label=name, linewidth=1.8)
        ax.set_title("二段階モデル構成要素 | Q5-Q1累積リターン", fontsize=15, fontweight="bold")
        ax.set_ylabel("Cumulative Wealth")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        ax.text(
            0.01,
            0.01,
            "CS Only=国×セクター予測、Within Only=銘柄固有予測、Simple Sum=単純加算、OOF Ridge Blend=学習済み合成。",
            transform=ax.transAxes,
            fontsize=9,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for name, g in rank_ic.groupby("Scenario"):
            g = g.sort_values("Date")
            ax.plot(
                g["Date"],
                g["RankIC"].rolling(rolling, min_periods=max(3, rolling // 2)).mean(),
                label=name,
                linewidth=1.6,
            )
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"二段階モデル構成要素 | ローリング{rolling}期間RankIC", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not summary.empty:
            metrics = ["CommonMeanRankIC", "CommonQ5MinusQ1Sharpe", "CommonQuintileMonotonicity", "CommonQ5MinusQ1MaxDrawdown"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, metric in zip(axes.ravel(), metrics):
                plot_df = summary.sort_values(metric)
                ax.barh(plot_df["Scenario"], plot_df[metric])
                ax.set_title(metric)
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle("二段階モデル構成要素の比較（共通OOS期間）", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        blend = two_stage.get("BlendHistory", pd.DataFrame())
        if isinstance(blend, pd.DataFrame) and not blend.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.plot(blend["Date"], blend["CountrySectorWeight"], label="Country-Sector Weight", linewidth=1.8)
            ax.plot(blend["Date"], blend["WithinWeight"], label="Within Weight", linewidth=1.8)
            ax.axhline(1, linewidth=0.8, linestyle="--")
            ax.axhline(0, linewidth=0.8)
            ax.set_title("OOF Ridge合成係数の推移", fontsize=15, fontweight="bold")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_two_stage_diagnostics_excel(
    output_path: Path,
    data: pd.DataFrame,
    two_stage: dict[str, pd.DataFrame | pd.Series],
    config: dict[str, Any],
) -> None:
    c = config["columns"]
    summary, quintiles, rank_ic = _component_evaluation(data, two_stage, config)
    stock = data[[c["date"], c["isin"], c["country"], c["sector"], "NextMonthReturn", "NextMonthCountrySectorReturn", "NextMonthWithinReturn"]].copy()
    stock = stock.rename(columns={c["date"]: "Date", c["isin"]: "ISIN", c["country"]: "Country", c["sector"]: "Sector"})
    stock["CountrySectorPrediction"] = two_stage["CountrySectorStockPrediction"]
    stock["WithinPrediction"] = two_stage["WithinStockPrediction"]
    stock["SimpleSumPrediction"] = two_stage["SimpleSumPrediction"]
    stock["BlendedPrediction"] = two_stage["BlendedPrediction"]

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame(
            {
                "Sheet": ["Component_Summary", "Component_Quintiles", "Component_RankIC", "CountrySector_Panel", "CountrySector_Predictions", "Stock_Predictions", "CS_Model_History", "Blend_History"],
                "Content": [
                    "CS/Within/単純合算/OOF合成の比較",
                    "構成要素別の5分位リターン",
                    "構成要素別の月次RankIC",
                    "国×セクター学習パネル",
                    "国×セクターの実現値と予測値",
                    "銘柄別の両成分・総予測",
                    "国×セクターRidgeの推定履歴",
                    "OOF Ridge合成係数履歴",
                ],
            }
        ).to_excel(writer, sheet_name="README", index=False)
        _write_dataframe(writer, summary, "Component_Summary")
        _write_dataframe(writer, quintiles, "Component_Quintiles")
        _write_dataframe(writer, rank_ic, "Component_RankIC")
        _write_dataframe(writer, two_stage.get("CountrySectorPanel", pd.DataFrame()), "CountrySector_Panel")
        _write_dataframe(writer, two_stage.get("CountrySectorPredictions", pd.DataFrame()), "CountrySector_Predictions")
        _write_dataframe(writer, stock, "Stock_Predictions")
        _write_dataframe(writer, two_stage.get("CountrySectorModelHistory", pd.DataFrame()), "CS_Model_History")
        _write_dataframe(writer, two_stage.get("BlendHistory", pd.DataFrame()), "Blend_History")
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
            worksheet.set_column(0, max(0, worksheet.dim_colmax), 18)


def _layer3_component_results(
    data: pd.DataFrame,
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    config: dict[str, Any],
) -> dict[str, ScenarioResult]:
    results: dict[str, ScenarioResult] = {}
    c = config["columns"]
    rank_scope = str(config["layer3"].get("final_score_rank_scope", "country"))
    for scope, payload in layer3.items():
        pred = payload["Prediction"]
        stock = pd.DataFrame({
            "Scenario": scope,
            "Date": data[c["date"]],
            "ISIN": data[c["isin"]],
            "Country": data[c["country"]],
            "Sector": data[c["sector"]],
            "Currency": data[c["currency"]],
            "MarketCap": data[c["market_cap"]],
            "Prediction": pred,
            "NextMonthReturn": data["NextMonthReturn"],
        })
        if rank_scope == "country":
            stock["TotalScore"] = stock.groupby(["Date", "Country"])["Prediction"].rank(pct=True)
            group_cols = ["Date", "Country"]
        else:
            stock["TotalScore"] = stock.groupby("Date")["Prediction"].rank(pct=True)
            group_cols = ["Date"]
        qn = int(config["evaluation"].get("quintiles", 5))
        def qcut_safe(s: pd.Series) -> pd.Series:
            valid = s.notna()
            out = pd.Series(pd.NA, index=s.index, dtype="Int64")
            if int(valid.sum()) >= qn:
                out.loc[valid] = pd.qcut(s.loc[valid].rank(method="first"), qn, labels=range(1, qn + 1)).astype(int)
            return out
        qseries = stock.groupby(group_cols, group_keys=False)["TotalScore"].apply(qcut_safe)
        if isinstance(qseries.index, pd.MultiIndex):
            qseries = qseries.reset_index(level=list(range(qseries.index.nlevels - 1)), drop=True)
        stock["Quintile"] = qseries.reindex(stock.index)
        results[scope] = ScenarioResult(stock, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    return results


def write_layer3_scope_comparison_pdf(
    data: pd.DataFrame,
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    from .evaluation import evaluate_scenarios
    _setup_matplotlib()
    results = _layer3_component_results(data, layer3, config)
    summary, quintiles, rank_ic = evaluate_scenarios(results, config)
    ls_curves = cumulative_long_short(quintiles, int(config["evaluation"].get("quintiles", 5)))
    rolling = int(config["evaluation"].get("rolling_rank_ic_periods", 12))
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for scope, series in ls_curves.items():
            ax.plot(series.index, series.values, label=scope, linewidth=1.8)
        ax.set_title("第3層推定範囲別 Q5-Q1累積リターン", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        ax.set_ylabel("Cumulative Wealth")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for scope, g in rank_ic.groupby("Scenario"):
            g = g.sort_values("Date")
            ax.plot(g["Date"], g["RankIC"].rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=scope)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"第3層推定範囲別 ローリング{rolling}期間RankIC", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2); ax.legend(loc="upper left")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not summary.empty:
            metrics = ["CommonMeanRankIC", "CommonQ5MinusQ1Sharpe", "CommonQuintileMonotonicity", "CommonQ5MinusQ1MaxDrawdown"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, metric in zip(axes.ravel(), metrics):
                frame = summary.sort_values(metric)
                ax.barh(frame["Scenario"], frame[metric])
                ax.set_title(metric); ax.grid(axis="x", alpha=0.2)
            fig.suptitle("第3層推定範囲の共通OOS比較", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_layer3_country_diagnostics_pdf(
    data: pd.DataFrame,
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    c = config["columns"]
    primary = str(config["layer3"].get("primary_scope", "country_independent"))
    pred = layer3[primary]["Prediction"]
    frame = pd.DataFrame({"Date": data[c["date"]], "Country": data[c["country"]], "Prediction": pred, "Return": data["NextMonthReturn"]})
    rows = []
    from scipy.stats import spearmanr
    for (country, date), g in frame.groupby(["Country", "Date"]):
        g = g.dropna()
        if len(g) >= 8:
            rows.append({"Country": country, "Date": date, "RankIC": spearmanr(g["Prediction"], g["Return"]).statistic})
    ic = pd.DataFrame(rows)
    with PdfPages(output_path) as pdf:
        if not ic.empty:
            mean_ic = ic.groupby("Country")["RankIC"].agg(["mean", "std", "count"]).sort_values("mean")
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.barh(mean_ic.index.astype(str), mean_ic["mean"])
            ax.axvline(0, linewidth=0.8); ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{primary} | 国別平均RankIC", fontsize=15, fontweight="bold")
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for country, g in ic.groupby("Country"):
                g = g.sort_values("Date")
                ax.plot(g["Date"], g["RankIC"].rolling(12, min_periods=6).mean(), label=str(country), linewidth=1.3)
            ax.axhline(0, linewidth=0.8); ax.grid(alpha=0.2); ax.legend(ncol=3, fontsize=8)
            ax.set_title(f"{primary} | 国別ローリングRankIC", fontsize=15, fontweight="bold")
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_coefficient_stability_pdf(
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    primary = str(config["layer3"].get("primary_scope", "country_independent"))
    coef = layer3[primary].get("CoefficientHistory", pd.DataFrame())
    if not isinstance(coef, pd.DataFrame) or coef.empty:
        return
    top = coef.groupby("Feature")["Coefficient"].apply(lambda s: s.abs().mean()).nlargest(12).index
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for feature in top:
            g = coef[coef["Feature"].eq(feature)].groupby("Date")["Coefficient"].mean().sort_index()
            ax.plot(g.index, g.values, label=str(feature), linewidth=1.1)
        ax.axhline(0, linewidth=0.8); ax.grid(alpha=0.2); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"{primary} | 主要係数の時系列安定性", fontsize=15, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_sector_factor_interactions_pdf(
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    primary = str(config["layer3"].get("primary_scope", "country_independent"))
    coef = layer3[primary].get("CoefficientHistory", pd.DataFrame())
    if not isinstance(coef, pd.DataFrame) or coef.empty:
        return
    interactions = coef[coef["Feature"].astype(str).str.startswith("INT__")].copy()
    if interactions.empty:
        return
    summary = interactions.groupby("Feature")["Coefficient"].agg(["mean", "std", "count"])
    summary["abs_mean"] = summary["mean"].abs()
    summary = summary.nlargest(20, "abs_mean").sort_values("mean")
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.barh(summary.index.astype(str), summary["mean"], xerr=summary["std"].fillna(0))
        ax.axvline(0, linewidth=0.8); ax.grid(axis="x", alpha=0.2)
        ax.set_title(f"{primary} | セクターグループ×FactorScore交差項", fontsize=15, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_layer3_diagnostics_excel(
    output_path: Path,
    data: pd.DataFrame,
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    config: dict[str, Any],
) -> None:
    from .evaluation import evaluate_scenarios
    results = _layer3_component_results(data, layer3, config)
    summary, quintiles, rank_ic = evaluate_scenarios(results, config)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame({"Sheet": ["Scope_Summary", "Scope_Quintiles", "Scope_RankIC", "Predictions", "Coefficients", "Model_History"], "Content": ["第3層範囲比較", "5分位履歴", "RankIC履歴", "銘柄別予測", "係数履歴", "学習履歴"]}).to_excel(writer, sheet_name="README", index=False)
        summary.to_excel(writer, sheet_name="Scope_Summary", index=False)
        quintiles.to_excel(writer, sheet_name="Scope_Quintiles", index=False)
        rank_ic.to_excel(writer, sheet_name="Scope_RankIC", index=False)
        c = config["columns"]
        pred = data[[c["date"], c["isin"], c["country"], c["sector"], "NextMonthReturn"]].copy().rename(columns={c["date"]: "Date", c["isin"]: "ISIN", c["country"]: "Country", c["sector"]: "Sector"})
        for scope, payload in layer3.items():
            pred[scope] = payload["Prediction"]
        pred.to_excel(writer, sheet_name="Predictions", index=False)
        coefficients = pd.concat([payload["CoefficientHistory"] for payload in layer3.values() if isinstance(payload.get("CoefficientHistory"), pd.DataFrame) and not payload["CoefficientHistory"].empty], ignore_index=True) if layer3 else pd.DataFrame()
        models = pd.concat([payload["ModelHistory"] for payload in layer3.values() if isinstance(payload.get("ModelHistory"), pd.DataFrame) and not payload["ModelHistory"].empty], ignore_index=True) if layer3 else pd.DataFrame()
        coefficients.to_excel(writer, sheet_name="Coefficients", index=False)
        models.to_excel(writer, sheet_name="Model_History", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0); ws.set_column(0, max(0, ws.dim_colmax), 18)


def write_layer3_history_files(
    output_dir: Path,
    layer3: dict[str, dict[str, pd.DataFrame | pd.Series]],
    diagnostics: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> None:
    cfg = config["outputs"].get("history_excel", {})
    if not cfg.get("enabled", True):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("layer1_model_selection", True) and not diagnostics.get("Layer1Selection", pd.DataFrame()).empty:
        diagnostics["Layer1Selection"].to_excel(output_dir / "layer1_model_selection_history.xlsx", index=False)
    if cfg.get("layer1_subscore", False):
        pass  # 大容量のためシナリオExcelまたはユーザー拡張で出力
    if cfg.get("layer2_factor_score", False):
        diagnostics.get("Layer2FactorScores", pd.DataFrame()).to_excel(output_dir / "layer2_factor_score_history.xlsx", index=False)
    if cfg.get("layer3_prediction", True):
        pred = pd.DataFrame({scope: payload["Prediction"] for scope, payload in layer3.items()})
        pred.to_excel(output_dir / "layer3_prediction_history.xlsx", index=True)
    if cfg.get("layer3_coefficients", True):
        frames = [payload["CoefficientHistory"] for payload in layer3.values() if isinstance(payload.get("CoefficientHistory"), pd.DataFrame) and not payload["CoefficientHistory"].empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(output_dir / "layer3_coefficient_history.xlsx", index=False)
    if cfg.get("sector_interactions", True):
        frames = [payload["CoefficientHistory"] for payload in layer3.values() if isinstance(payload.get("CoefficientHistory"), pd.DataFrame) and not payload["CoefficientHistory"].empty]
        if frames:
            interactions = pd.concat(frames, ignore_index=True)
            interactions = interactions[interactions["Feature"].astype(str).str.startswith("INT__")]
            interactions.to_excel(output_dir / "sector_interaction_history.xlsx", index=False)
