from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from .reporting import _setup_matplotlib


@dataclass
class CountryDiagnostics:
    summary: pd.DataFrame
    rank_ic: pd.DataFrame
    quintiles: pd.DataFrame
    long_short: pd.DataFrame
    latest_coefficients: pd.DataFrame
    coefficient_stability: pd.DataFrame
    effective_sector_slopes: pd.DataFrame
    model_history: pd.DataFrame


def _safe_rank_ic(group: pd.DataFrame, minimum_stocks: int) -> float:
    valid = group[["Prediction", "NextMonthReturn"]].dropna()
    if len(valid) < minimum_stocks:
        return np.nan
    if valid["Prediction"].nunique() < 2 or valid["NextMonthReturn"].nunique() < 2:
        return np.nan
    return float(spearmanr(valid["Prediction"], valid["NextMonthReturn"]).statistic)


def _safe_quintile(scores: pd.Series, n_quantiles: int) -> pd.Series:
    valid = scores.notna()
    out = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    if int(valid.sum()) < n_quantiles:
        return out
    ranked = scores.loc[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranked, n_quantiles, labels=range(1, n_quantiles + 1)).astype(int)
    return out


def _max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _sign_flip_count(values: pd.Series) -> int:
    s = np.sign(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0


def _country_common_oos_frame(
    data: pd.DataFrame,
    variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
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
    common = base["NextMonthReturn"].notna()
    for payload in variants.values():
        pred = pd.to_numeric(payload["Prediction"], errors="coerce").reindex(data.index)
        common &= pred.notna()
    base = base.loc[common].copy()
    rows: list[pd.DataFrame] = []
    qn = int(config["evaluation"].get("quintiles", 5))
    for name, payload in variants.items():
        frame = base.copy()
        frame["Variant"] = name
        frame["Prediction"] = pd.to_numeric(payload["Prediction"], errors="coerce").reindex(frame.index)
        frame["TotalScore"] = frame.groupby(["Country", "Date"])["Prediction"].rank(pct=True)
        frame["Quintile"] = (
            frame.groupby(["Country", "Date"], group_keys=False)["TotalScore"]
            .apply(lambda s: _safe_quintile(s, qn))
            .reindex(frame.index)
        )
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _country_performance_tables(
    common: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if common.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    minimum_stocks = int(config["layer3"].get("country_diagnostics_minimum_stocks", 15))
    annualization = int(config["evaluation"].get("annualization", 12))
    qn = int(config["evaluation"].get("quintiles", 5))

    rank_rows: list[dict[str, object]] = []
    for (variant, country, date), group in common.groupby(["Variant", "Country", "Date"]):
        ic = _safe_rank_ic(group, minimum_stocks)
        if np.isfinite(ic):
            rank_rows.append(
                {
                    "Variant": variant,
                    "Country": country,
                    "Date": date,
                    "RankIC": ic,
                    "ObservationCount": int(group[["Prediction", "NextMonthReturn"]].dropna().shape[0]),
                }
            )
    rank_ic = pd.DataFrame(rank_rows)

    quintiles = (
        common.dropna(subset=["Quintile", "NextMonthReturn"])
        .groupby(["Variant", "Country", "Date", "Quintile"], as_index=False)
        .agg(Return=("NextMonthReturn", "mean"), Stocks=("ISIN", "nunique"))
    )
    if quintiles.empty:
        long_short = pd.DataFrame()
    else:
        pivot = quintiles.pivot_table(
            index=["Variant", "Country", "Date"], columns="Quintile", values="Return"
        )
        if 1 in pivot.columns and qn in pivot.columns:
            long_short = (pivot[qn] - pivot[1]).rename("Q5MinusQ1").reset_index()
        else:
            long_short = pd.DataFrame(columns=["Variant", "Country", "Date", "Q5MinusQ1"])

    rows: list[dict[str, object]] = []
    keys = sorted(common[["Variant", "Country"]].drop_duplicates().itertuples(index=False, name=None))
    for variant, country in keys:
        ric = rank_ic[(rank_ic["Variant"] == variant) & (rank_ic["Country"] == country)].sort_values("Date")
        ls = long_short[(long_short["Variant"] == variant) & (long_short["Country"] == country)].sort_values("Date")
        q = quintiles[(quintiles["Variant"] == variant) & (quintiles["Country"] == country)]
        sample = common[(common["Variant"] == variant) & (common["Country"] == country)]
        ic_mean = ric["RankIC"].mean() if not ric.empty else np.nan
        ic_std = ric["RankIC"].std(ddof=1) if len(ric) > 1 else np.nan
        ls_mean = ls["Q5MinusQ1"].mean() if not ls.empty else np.nan
        ls_std = ls["Q5MinusQ1"].std(ddof=1) if len(ls) > 1 else np.nan
        qmean = q.groupby("Quintile")["Return"].mean().sort_index() if not q.empty else pd.Series(dtype=float)
        monotonicity = (
            float(spearmanr(qmean.index.astype(float), qmean.values).statistic)
            if len(qmean) >= 3 and qmean.nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "Variant": variant,
                "Country": country,
                "CommonStartDate": sample["Date"].min(),
                "CommonEndDate": sample["Date"].max(),
                "CommonEvaluationPeriods": int(sample["Date"].nunique()),
                "CommonObservationCount": int(len(sample)),
                "CommonMeanStocksPerPeriod": float(sample.groupby("Date")["ISIN"].nunique().mean()),
                "MeanRankIC": ic_mean,
                "MedianRankIC": ric["RankIC"].median() if not ric.empty else np.nan,
                "RankICIR": ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan,
                "RankICPositiveRate": float((ric["RankIC"] > 0).mean()) if not ric.empty else np.nan,
                "Q5MinusQ1Mean": ls_mean,
                "Q5MinusQ1AnnualizedReturn": (1.0 + ls_mean) ** annualization - 1.0 if np.isfinite(ls_mean) else np.nan,
                "Q5MinusQ1Sharpe": ls_mean / ls_std * np.sqrt(annualization) if np.isfinite(ls_std) and ls_std > 0 else np.nan,
                "Q5MinusQ1MaxDrawdown": _max_drawdown(ls["Q5MinusQ1"]) if not ls.empty else np.nan,
                "QuintileMonotonicity": monotonicity,
            }
        )
    return pd.DataFrame(rows), rank_ic, quintiles, long_short


def _collect_coefficients(
    variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    for name, payload in variants.items():
        coef = payload.get("CoefficientHistory", pd.DataFrame())
        if isinstance(coef, pd.DataFrame) and not coef.empty:
            tmp = coef.copy()
            tmp.insert(0, "Variant", name)
            coef_frames.append(tmp)
        hist = payload.get("ModelHistory", pd.DataFrame())
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            tmp = hist.copy()
            tmp.insert(0, "Variant", name)
            model_frames.append(tmp)
    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    models = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    if not coefficients.empty:
        coefficients["Date"] = pd.to_datetime(coefficients["Date"])
        coefficients["Country"] = coefficients["ScopeLabel"].astype(str)
        coefficients = coefficients[coefficients["Scope"].astype(str).eq("country_independent")].copy()
    if not models.empty:
        models["Date"] = pd.to_datetime(models["Date"])
        models["Country"] = models["ScopeLabel"].astype(str)
        models = models[models["Scope"].astype(str).eq("country_independent")].copy()
    return coefficients, models


def _coefficient_tables(
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coefficients.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    coef_col = "StandardizedCoefficient" if "StandardizedCoefficient" in coefficients.columns else "Coefficient"
    raw_col = "RawCoefficient" if "RawCoefficient" in coefficients.columns else coef_col
    coefficients = coefficients.sort_values(["Variant", "Country", "Feature", "Date"]).copy()

    latest_date = coefficients.groupby(["Variant", "Country"])["Date"].transform("max")
    latest = coefficients[coefficients["Date"].eq(latest_date)].copy()
    latest["CoefficientForComparison"] = latest[coef_col]
    latest["RawCoefficientForPrediction"] = latest[raw_col]

    stability_rows: list[dict[str, object]] = []
    for (variant, country, feature), group in coefficients.groupby(["Variant", "Country", "Feature"]):
        group = group.sort_values("Date")
        values = pd.to_numeric(group[coef_col], errors="coerce").dropna()
        if values.empty:
            continue
        stability_rows.append(
            {
                "Variant": variant,
                "Country": country,
                "Feature": feature,
                "Periods": int(len(values)),
                "MeanCoefficient": float(values.mean()),
                "CoefficientStd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "LatestCoefficient": float(values.iloc[-1]),
                "PositiveRate": float((values > 0).mean()),
                "NegativeRate": float((values < 0).mean()),
                "SignFlipCount": _sign_flip_count(values),
                "FirstDate": group["Date"].min(),
                "LatestDate": group["Date"].max(),
            }
        )
    stability = pd.DataFrame(stability_rows)

    interactions = coefficients[coefficients["Feature"].astype(str).str.startswith("INT__")].copy()
    slope_rows: list[dict[str, object]] = []
    if not interactions.empty:
        main_lookup = coefficients.set_index(["Variant", "Country", "Date", "Feature"])
        for row in interactions.itertuples(index=False):
            parts = str(row.Feature).split("__")
            if len(parts) < 4:
                continue
            sector = parts[1]
            base_feature = "__".join(parts[2:])
            key = (row.Variant, row.Country, row.Date, base_feature)
            if key not in main_lookup.index:
                continue
            main_row = main_lookup.loc[key]
            if isinstance(main_row, pd.DataFrame):
                main_row = main_row.iloc[0]
            interaction_std = float(getattr(row, coef_col))
            interaction_raw = float(getattr(row, raw_col))
            main_std = float(main_row[coef_col])
            main_raw = float(main_row[raw_col])
            slope_rows.append(
                {
                    "Variant": row.Variant,
                    "Country": row.Country,
                    "Date": row.Date,
                    "SectorGroup": sector,
                    "FactorFeature": base_feature,
                    "MainCoefficient": main_std,
                    "InteractionCoefficient": interaction_std,
                    "EffectiveSectorSlope": main_std + interaction_std,
                    "RawMainCoefficient": main_raw,
                    "RawInteractionCoefficient": interaction_raw,
                    "RawEffectiveSectorSlope": main_raw + interaction_raw,
                }
            )
    return latest, stability, pd.DataFrame(slope_rows)


def build_country_diagnostics(
    data: pd.DataFrame,
    variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
    config: dict[str, Any],
) -> CountryDiagnostics:
    common = _country_common_oos_frame(data, variants, config)
    summary, rank_ic, quintiles, long_short = _country_performance_tables(common, config)
    coefficients, model_history = _collect_coefficients(variants)
    latest, stability, sector_slopes = _coefficient_tables(coefficients)
    return CountryDiagnostics(
        summary=summary,
        rank_ic=rank_ic,
        quintiles=quintiles,
        long_short=long_short,
        latest_coefficients=latest,
        coefficient_stability=stability,
        effective_sector_slopes=sector_slopes,
        model_history=model_history,
    )


def write_country_diagnostics_excel(
    data: pd.DataFrame,
    variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    if not variants:
        return
    diagnostics = build_country_diagnostics(data, variants, config)
    sheets = {
        "Country_Summary": diagnostics.summary,
        "Country_RankIC": diagnostics.rank_ic,
        "Country_Quintiles": diagnostics.quintiles,
        "Country_LongShort": diagnostics.long_short,
        "Latest_Coefficients": diagnostics.latest_coefficients,
        "Coefficient_Stability": diagnostics.coefficient_stability,
        "Effective_Sector_Slopes": diagnostics.effective_sector_slopes,
        "Model_History": diagnostics.model_history,
    }
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame(
            {
                "Sheet": list(sheets),
                "Content": [
                    "国別・S07推定方式別の共通OOS指標",
                    "国別月次RankIC（OLS/Ridge共通銘柄集合）",
                    "国別5分位リターン",
                    "国別Q5-Q1リターン",
                    "最新の国別係数。CoefficientForComparisonは標準化係数",
                    "国別係数の平均・標準偏差・正符号率・符号反転回数",
                    "主効果＋セクター交差項で計算したセクター内の合計傾き",
                    "国別Alpha・学習期間・観測数履歴",
                ],
            }
        ).to_excel(writer, sheet_name="README", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        decimal = workbook.add_format({"num_format": "0.0000"})
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_row(0, 22, header)
            ws.set_column(0, max(0, ws.dim_colmax), 18)
            if ws.dim_rowmax >= 0 and ws.dim_colmax >= 0:
                ws.autofilter(0, 0, ws.dim_rowmax, ws.dim_colmax)
        for name in ["Country_Summary", "Coefficient_Stability", "Effective_Sector_Slopes"]:
            if name in writer.sheets:
                writer.sheets[name].set_column(2, max(2, writer.sheets[name].dim_colmax), 18, decimal)


def _plot_heatmap(
    pdf: PdfPages,
    matrix: pd.DataFrame,
    title: str,
    value_format: str = ".3f",
) -> None:
    if matrix.empty:
        return
    width = max(11.69, 1.2 * len(matrix.columns) + 3.5)
    height = max(8.27, 0.45 * len(matrix.index) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    arr = matrix.to_numpy(float)
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    image = ax.imshow(arr, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title, fontsize=15, fontweight="bold")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if np.isfinite(arr[r, c]):
                ax.text(c, r, format(arr[r, c], value_format), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_country_diagnostics_pdf(
    data: pd.DataFrame,
    variants: dict[str, dict[str, pd.DataFrame | pd.Series]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    if not variants:
        return
    _setup_matplotlib()
    diagnostics = build_country_diagnostics(data, variants, config)
    rolling = int(config["evaluation"].get("rolling_rank_ic_periods", 12))

    with PdfPages(output_path) as pdf:
        if not diagnostics.summary.empty:
            pivot = diagnostics.summary.pivot(index="Country", columns="Variant", values="MeanRankIC")
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            pivot.plot(kind="barh", ax=ax)
            ax.axvline(0, linewidth=0.8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title("S07 国別Mean Rank IC（国別共通OOS）", fontsize=15, fontweight="bold")
            ax.set_xlabel("Mean Rank IC")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        countries = sorted(diagnostics.rank_ic["Country"].dropna().astype(str).unique()) if not diagnostics.rank_ic.empty else []
        for country in countries:
            fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
            ric = diagnostics.rank_ic[diagnostics.rank_ic["Country"].eq(country)]
            for variant, group in ric.groupby("Variant"):
                group = group.sort_values("Date")
                axes[0].plot(
                    group["Date"],
                    group["RankIC"].rolling(rolling, min_periods=max(3, rolling // 2)).mean(),
                    label=variant,
                    linewidth=1.6,
                )
            axes[0].axhline(0, linewidth=0.8)
            axes[0].grid(alpha=0.2)
            axes[0].legend(loc="upper left")
            axes[0].set_title(f"{country} | ローリング{rolling}期間RankIC")

            ls = diagnostics.long_short[diagnostics.long_short["Country"].eq(country)]
            for variant, group in ls.groupby("Variant"):
                group = group.sort_values("Date")
                wealth = (1.0 + group["Q5MinusQ1"].fillna(0.0)).cumprod()
                axes[1].plot(group["Date"], wealth, label=variant, linewidth=1.6)
            axes[1].grid(alpha=0.2)
            axes[1].legend(loc="upper left")
            axes[1].set_title(f"{country} | Q5-Q1累積リターン")
            axes[1].set_ylabel("Cumulative Wealth")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        latest = diagnostics.latest_coefficients.copy()
        if not latest.empty:
            main = latest[
                latest["Feature"].astype(str).str.endswith("__LIN")
                & ~latest["Feature"].astype(str).str.startswith(("INT__", "DEV__"))
            ].copy()
            for variant, group in main.groupby("Variant"):
                matrix = group.pivot_table(
                    index="Country", columns="Feature", values="CoefficientForComparison", aggfunc="last"
                )
                matrix.columns = [str(c).replace("__LIN", "") for c in matrix.columns]
                _plot_heatmap(pdf, matrix, f"{variant} | 最新の国別Factor係数（標準化係数）")

        stability = diagnostics.coefficient_stability.copy()
        if not stability.empty:
            main_stability = stability[
                stability["Feature"].astype(str).str.endswith("__LIN")
                & ~stability["Feature"].astype(str).str.startswith(("INT__", "DEV__"))
            ]
            for variant, group in main_stability.groupby("Variant"):
                matrix = group.pivot_table(index="Country", columns="Feature", values="PositiveRate", aggfunc="last")
                matrix.columns = [str(c).replace("__LIN", "") for c in matrix.columns]
                _plot_heatmap(pdf, matrix, f"{variant} | 国別Factor係数の正符号率", value_format=".0%")

        slopes = diagnostics.effective_sector_slopes.copy()
        if not slopes.empty:
            latest_dates = slopes.groupby(["Variant", "Country"])["Date"].transform("max")
            latest_slopes = slopes[slopes["Date"].eq(latest_dates)].copy()
            latest_slopes["Label"] = (
                latest_slopes["Country"].astype(str)
                + " | "
                + latest_slopes["SectorGroup"].astype(str)
                + " × "
                + latest_slopes["FactorFeature"].astype(str).str.replace("__LIN", "", regex=False)
            )
            for variant, group in latest_slopes.groupby("Variant"):
                show = group.assign(abs_slope=group["EffectiveSectorSlope"].abs()).nlargest(25, "abs_slope").sort_values("EffectiveSectorSlope")
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.barh(show["Label"], show["EffectiveSectorSlope"])
                ax.axvline(0, linewidth=0.8)
                ax.grid(axis="x", alpha=0.2)
                ax.set_title(f"{variant} | 最新のセクター別Factor合計傾き（上位25）", fontsize=15, fontweight="bold")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        history = diagnostics.model_history.copy()
        if not history.empty:
            ridge = history[history["Estimator"].astype(str).eq("ridge")]
            for variant, group in ridge.groupby("Variant"):
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                for country, country_group in group.groupby("Country"):
                    country_group = country_group.sort_values("Date")
                    ax.step(country_group["Date"], country_group["Alpha"], where="post", label=country, linewidth=1.3)
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
                ax.legend(ncol=3, fontsize=8)
                ax.set_title(f"{variant} | 国別Ridge Alpha推移", fontsize=15, fontweight="bold")
                ax.set_ylabel("Alpha (log scale)")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
