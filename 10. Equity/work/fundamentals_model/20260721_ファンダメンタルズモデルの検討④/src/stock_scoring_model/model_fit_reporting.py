from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .regression_metrics import regression_metrics
from .reporting import _setup_matplotlib
from .scenarios import ScenarioResult


@dataclass
class ModelFitDiagnostics:
    summary: pd.DataFrame
    monthly: pd.DataFrame
    distribution: pd.DataFrame
    calibration_bins: pd.DataFrame
    predictions: pd.DataFrame
    s06_coefficients: pd.DataFrame
    s06_fit_history: pd.DataFrame
    s06_effective_weights: pd.DataFrame
    s07_coefficients: pd.DataFrame
    s07_model_fit: pd.DataFrame


def _calibration_coefficients(actual: pd.Series, prediction: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"Actual": actual, "Prediction": prediction}).dropna()
    if len(frame) < 3 or frame["Prediction"].nunique() < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(frame)), frame["Prediction"].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X, frame["Actual"].to_numpy(float), rcond=None)
    return float(coef[0]), float(coef[1])


def _prediction_frame(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    s07_variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
    config: dict[str, Any],
) -> pd.DataFrame:
    c = config["columns"]
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(data[c["date"]]),
            "ISIN": data[c["isin"]].astype(str),
            "Country": data[c["country"]].astype(str),
            "NextMonthReturn": pd.to_numeric(data["NextMonthReturn"], errors="coerce"),
        },
        index=data.index,
    )
    predictions: dict[str, pd.Series] = {}
    s06 = scenarios.get("S06_Selected_Factor_Models")
    if s06 is not None and not s06.stock_scores.empty:
        predictions["S06_Selected_Factor_Models"] = pd.to_numeric(
            s06.stock_scores["Prediction"], errors="coerce"
        ).reindex(data.index)
    for name, payload in s07_variants.items():
        predictions[name] = pd.to_numeric(payload.get("Prediction"), errors="coerce").reindex(data.index)
    if not predictions:
        return pd.DataFrame()
    common = base["NextMonthReturn"].notna()
    for pred in predictions.values():
        common &= pred.notna()
    rows = []
    for name, pred in predictions.items():
        frame = base.loc[common].copy()
        frame["Scenario"] = name
        frame["Prediction"] = pred.loc[common]
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metric_row(frame: pd.DataFrame, scenario: str, scope: str, label: str) -> dict[str, object]:
    metrics = regression_metrics(frame["NextMonthReturn"], frame["Prediction"], feature_count=1)
    intercept, slope = _calibration_coefficients(frame["NextMonthReturn"], frame["Prediction"])
    return {
        "Scenario": scenario,
        "Scope": scope,
        "ScopeLabel": label,
        "StartDate": frame["Date"].min(),
        "EndDate": frame["Date"].max(),
        "EvaluationPeriods": int(frame["Date"].nunique()),
        "CalibrationIntercept": intercept,
        "CalibrationSlope": slope,
        **metrics,
    }


def _summary_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for scenario, frame in predictions.groupby("Scenario"):
        summary_rows.append(_metric_row(frame, scenario, "global", "ALL"))
        for country, group in frame.groupby("Country"):
            summary_rows.append(_metric_row(group, scenario, "country", str(country)))
        for date, group in frame.groupby("Date"):
            monthly_rows.append({
                **_metric_row(group, scenario, "global", "ALL"),
                "Date": date,
            })
        for (country, date), group in frame.groupby(["Country", "Date"]):
            monthly_rows.append({
                **_metric_row(group, scenario, "country", str(country)),
                "Date": date,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def _distribution_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    for (scenario, country), frame in predictions.groupby(["Scenario", "Country"]):
        for variable in ["Prediction", "NextMonthReturn"]:
            s = pd.to_numeric(frame[variable], errors="coerce").dropna()
            row: dict[str, object] = {
                "Scenario": scenario,
                "Country": country,
                "Variable": variable,
                "Count": int(len(s)),
                "Mean": s.mean(),
                "Std": s.std(ddof=1),
                "Min": s.min(),
                "Max": s.max(),
            }
            for q in quantiles:
                row[f"P{int(q * 100):02d}"] = s.quantile(q) if not s.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _calibration_bins(predictions: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (scenario, country), frame in predictions.groupby(["Scenario", "Country"]):
        valid = frame[["Prediction", "NextMonthReturn"]].dropna().copy()
        if len(valid) < n_bins * 3 or valid["Prediction"].nunique() < n_bins:
            continue
        valid["Bin"] = pd.qcut(valid["Prediction"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
        agg = valid.groupby("Bin", observed=True).agg(
            MeanPrediction=("Prediction", "mean"),
            MeanActual=("NextMonthReturn", "mean"),
            StdActual=("NextMonthReturn", "std"),
            Count=("NextMonthReturn", "size"),
        ).reset_index()
        agg.insert(0, "Country", country)
        agg.insert(0, "Scenario", scenario)
        rows.extend(agg.to_dict("records"))
    return pd.DataFrame(rows)


def _s06_effective_weights(layer2_weights: pd.DataFrame) -> pd.DataFrame:
    if layer2_weights.empty:
        return pd.DataFrame()
    frame = layer2_weights.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    group_count = frame.groupby("Date")["Factor_Group"].transform("nunique").replace(0, np.nan)
    frame["FinalEffectiveWeight"] = pd.to_numeric(frame["Weight"], errors="coerce") / group_count
    return frame


def _s07_tables(s07_variants: dict[str, dict[str, pd.DataFrame | pd.Series]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_frames = []
    fit_frames = []
    for name, payload in s07_variants.items():
        coef = payload.get("CoefficientHistory", pd.DataFrame())
        if isinstance(coef, pd.DataFrame) and not coef.empty:
            tmp = coef.copy()
            tmp.insert(0, "Scenario", name)
            coef_frames.append(tmp)
        hist = payload.get("ModelHistory", pd.DataFrame())
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            tmp = hist.copy()
            tmp.insert(0, "Scenario", name)
            fit_frames.append(tmp)
    return (
        pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame(),
        pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame(),
    )


def build_model_fit_diagnostics(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> ModelFitDiagnostics:
    s07_variants = diagnostics.get("S07Variants", {})
    predictions = _prediction_frame(data, scenarios, s07_variants, config)
    summary, monthly = _summary_tables(predictions)
    cfg = config.get("model_fit_diagnostics", {})
    distribution = _distribution_table(predictions)
    calibration = _calibration_bins(predictions, int(cfg.get("calibration_bins", 10)))
    s07_coefficients, s07_model_fit = _s07_tables(s07_variants)
    return ModelFitDiagnostics(
        summary=summary,
        monthly=monthly,
        distribution=distribution,
        calibration_bins=calibration,
        predictions=predictions,
        s06_coefficients=diagnostics.get("Layer1Coefficients", pd.DataFrame()),
        s06_fit_history=diagnostics.get("Layer1FitHistory", pd.DataFrame()),
        s06_effective_weights=_s06_effective_weights(diagnostics.get("Layer2Weights", pd.DataFrame())),
        s07_coefficients=s07_coefficients,
        s07_model_fit=s07_model_fit,
    )


def write_model_fit_diagnostics_excel(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_model_fit_diagnostics(data, scenarios, diagnostics, config)
    sheets = {
        "Model_Summary": result.summary,
        "OOS_Monthly": result.monthly,
        "Prediction_Distribution": result.distribution,
        "Calibration_Bins": result.calibration_bins,
        "Common_OOS_Predictions": result.predictions,
        "S06_L1_Coefficients": result.s06_coefficients,
        "S06_L1_Fit_History": result.s06_fit_history,
        "S06_Effective_Weights": result.s06_effective_weights,
        "S07_Coefficients": result.s07_coefficients,
        "S07_Model_Fit": result.s07_model_fit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        pd.DataFrame(
            {
                "Sheet": list(sheets),
                "Content": [
                    "S06/S07共通OOSの回帰診断。OOS R2は負になり得る",
                    "月別・国別のOOS R2、RMSE、Pearson、Spearman、Calibration slope",
                    "国別の予測値・実現リターン分布",
                    "予測値分位別の平均予測値と平均実現リターン",
                    "S06/S07が全て存在するDate×ISIN共通OOSデータ",
                    "S06を構成する第1層の最終選択モデル係数",
                    "S06第1層のTrain/Validation R2・誤差・相関",
                    "S06最終平均に対するFAの実効ウェイト",
                    "S07の国別・時点別最終係数",
                    "S07の国別Train/Validation fit指標とRidge alpha",
                ],
            }
        ).to_excel(writer, sheet_name="README", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.sheets[name[:31]]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))


def _plot_summary_table(pdf: PdfPages, summary: pd.DataFrame) -> None:
    global_summary = summary[summary["Scope"].eq("global")].copy()
    cols = ["Scenario", "EvaluationPeriods", "R2", "CalibrationSlope", "Pearson", "Spearman", "PredictionTargetStdRatio"]
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("S06/S07 モデルフィッティング診断（共通OOS）", fontsize=15, fontweight="bold")
    if global_summary.empty:
        ax.text(0.5, 0.5, "No common OOS observations", ha="center", va="center")
    else:
        display = global_summary[cols].copy()
        display["Scenario"] = display["Scenario"].replace({
            "S06_Selected_Factor_Models": "S06",
            "S07_OLS_Linear": "S07 OLS",
            "S07_Ridge_Linear": "S07 Ridge",
            "S07_Ridge_Flexible": "S07 Flexible",
        })
        display = display.rename(columns={
            "EvaluationPeriods": "Periods",
            "CalibrationSlope": "CalibSlope",
            "PredictionTargetStdRatio": "Pred/Actual Std",
        })
        for col in ["R2", "CalibSlope", "Pearson", "Spearman", "Pred/Actual Std"]:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
    ax.text(0.01, 0.03, "S06には最終回帰はなく、Layer2 FactorScoreの平均が最終予測です。S06の係数確認では、Layer1の採用回帰係数とLayer2の実効ウェイトを併用します。", transform=ax.transAxes, fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_model_fit_diagnostics_pdf(
    data: pd.DataFrame,
    scenarios: dict[str, ScenarioResult],
    diagnostics: dict[str, Any],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_model_fit_diagnostics(data, scenarios, diagnostics, config)
    if result.predictions.empty:
        return
    _setup_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_scatter = int(config.get("model_fit_diagnostics", {}).get("scatter_max_points", 5000))
    with PdfPages(output_path) as pdf:
        _plot_summary_table(pdf, result.summary)

        # Monthly global OOS R2 / Spearman.
        monthly = result.monthly[result.monthly["Scope"].eq("global")].sort_values("Date")
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        for scenario, frame in monthly.groupby("Scenario"):
            axes[0].plot(frame["Date"], frame["R2"], label=scenario)
            axes[1].plot(frame["Date"], frame["Spearman"], label=scenario)
        axes[0].axhline(0, linewidth=0.8)
        axes[1].axhline(0, linewidth=0.8)
        axes[0].set_title("Monthly OOS R-squared")
        axes[1].set_title("Monthly OOS rank correlation")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Prediction/actual distributions.
        scenarios_order = list(result.predictions["Scenario"].drop_duplicates())
        ncols = 2
        nrows = int(np.ceil(len(scenarios_order) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 4.0 * nrows), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios_order):
            frame = result.predictions[result.predictions["Scenario"].eq(scenario)]
            ax.hist(frame["NextMonthReturn"].dropna(), bins=40, density=True, alpha=0.55, label="Actual")
            ax.hist(frame["Prediction"].dropna(), bins=40, density=True, alpha=0.55, label="Prediction")
            ax.set_title(scenario)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(scenarios_order):]:
            ax.axis("off")
        fig.suptitle("Prediction and realized return distributions", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Scatter plots.
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 4.2 * nrows), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios_order):
            frame = result.predictions[result.predictions["Scenario"].eq(scenario)][["Prediction", "NextMonthReturn"]].dropna()
            if len(frame) > max_scatter:
                frame = frame.sample(max_scatter, random_state=1234)
            ax.scatter(frame["Prediction"], frame["NextMonthReturn"], s=7, alpha=0.20)
            intercept, slope = _calibration_coefficients(frame["NextMonthReturn"], frame["Prediction"])
            if np.isfinite(slope):
                x = np.linspace(frame["Prediction"].min(), frame["Prediction"].max(), 100)
                ax.plot(x, intercept + slope * x, linewidth=1.6, label=f"actual={intercept:.4f}+{slope:.2f}*pred")
            ax.set_title(scenario)
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Next-month return")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7)
        for ax in axes.flat[len(scenarios_order):]:
            ax.axis("off")
        fig.suptitle("Common-OOS prediction versus realized return", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Calibration bins.
        calib = result.calibration_bins.groupby(["Scenario", "Bin"], as_index=False).agg(
            MeanPrediction=("MeanPrediction", "mean"), MeanActual=("MeanActual", "mean")
        )
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for scenario, frame in calib.groupby("Scenario"):
            ax.plot(frame["Bin"].astype(int), frame["MeanActual"], marker="o", label=f"{scenario}: actual")
            ax.plot(frame["Bin"].astype(int), frame["MeanPrediction"], linestyle="--", label=f"{scenario}: prediction")
        ax.set_title("Calibration by prediction decile", fontsize=15, fontweight="bold")
        ax.set_xlabel("Prediction bin (low to high)")
        ax.set_ylabel("Mean return")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # S06 Layer1 fit diagnostics.
        if not result.s06_fit_history.empty:
            latest_date = pd.to_datetime(result.s06_fit_history["Date"]).max()
            latest = result.s06_fit_history[pd.to_datetime(result.s06_fit_history["Date"]).eq(latest_date)].copy()
            latest = latest.sort_values("ValidationMeanRankIC", ascending=False)
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
            axes[0].barh(latest["FactorCode"].astype(str), latest["ValidationR2"])
            axes[0].set_title(f"S06 Layer1 validation R2 ({latest_date.date()})")
            axes[0].axvline(0, linewidth=0.8)
            axes[1].barh(latest["FactorCode"].astype(str), latest["ValidationMeanRankIC"])
            axes[1].set_title("S06 Layer1 validation mean RankIC")
            axes[1].axvline(0, linewidth=0.8)
            for ax in axes:
                ax.grid(alpha=0.15)
                ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # S07 training/validation R2 by country.
        if not result.s07_model_fit.empty and "ValidationR2" in result.s07_model_fit.columns:
            fit = result.s07_model_fit.copy()
            fit["Country"] = fit["ScopeLabel"].astype(str)
            agg = fit.groupby(["Scenario", "Country"], as_index=False).agg(
                TrainR2=("TrainR2", "mean"), ValidationR2=("ValidationR2", "mean"), Alpha=("Alpha", "mean")
            )
            countries = sorted(agg["Country"].unique())
            variants = sorted(agg["Scenario"].unique())
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
            width = 0.8 / max(len(variants), 1)
            x = np.arange(len(countries))
            for idx, scenario in enumerate(variants):
                frame = agg[agg["Scenario"].eq(scenario)].set_index("Country").reindex(countries)
                axes[0].bar(x + idx * width, frame["TrainR2"], width=width, label=scenario)
                axes[1].bar(x + idx * width, frame["ValidationR2"], width=width, label=scenario)
            for ax, title in zip(axes, ["Mean training R2", "Mean validation R2"]):
                ax.set_xticks(x + width * (len(variants) - 1) / 2, countries, rotation=45, ha="right")
                ax.axhline(0, linewidth=0.8)
                ax.set_title(title)
                ax.grid(alpha=0.15)
                ax.legend(fontsize=7)
            fig.suptitle("S07 model fit by country", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
