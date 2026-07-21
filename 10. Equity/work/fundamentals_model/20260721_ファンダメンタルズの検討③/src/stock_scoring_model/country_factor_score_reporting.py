from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .reporting import _setup_matplotlib


@dataclass
class CountryFactorScoreDiagnostics:
    history: pd.DataFrame
    latest: pd.DataFrame
    top_factor_history: pd.DataFrame
    top_factor_frequency: pd.DataFrame


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any() or w.loc[mask].sum() <= 0:
        return np.nan
    return float(np.average(v.loc[mask], weights=w.loc[mask]))


def build_country_factor_score_diagnostics(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    config: dict[str, Any],
) -> CountryFactorScoreDiagnostics:
    if factor_scores.empty:
        return CountryFactorScoreDiagnostics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    c = config["columns"]
    cfg = config.get("country_factor_score_diagnostics", {})
    window = int(cfg.get("trailing_z_periods", 36))
    min_periods = int(cfg.get("minimum_z_periods", 12))
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(data[c["date"]]),
            "Country": data[c["country"]].astype(str),
            "MarketCap": pd.to_numeric(data[c["market_cap"]], errors="coerce"),
        },
        index=data.index,
    )
    rows: list[dict[str, object]] = []
    for factor in factor_scores.columns:
        tmp = base.copy()
        tmp["Score"] = pd.to_numeric(factor_scores[factor], errors="coerce")
        for (date, country), group in tmp.groupby(["Date", "Country"]):
            score = group["Score"].dropna()
            if score.empty:
                continue
            rows.append(
                {
                    "Date": date,
                    "Country": country,
                    "FactorGroup": factor,
                    "EqualWeightedScore": score.mean(),
                    "MarketCapWeightedScore": _weighted_mean(group["Score"], group["MarketCap"]),
                    "MedianScore": score.median(),
                    "ScoreStd": score.std(ddof=1),
                    "StockCount": int(len(score)),
                }
            )
    history = pd.DataFrame(rows)
    if history.empty:
        return CountryFactorScoreDiagnostics(history, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    weighting = str(cfg.get("weighting", "equal")).lower()
    score_col = "MarketCapWeightedScore" if weighting in {"market_cap", "marketcap", "cap_weighted"} else "EqualWeightedScore"
    history["SelectedCountryScore"] = history[score_col]
    history = history.sort_values(["Country", "FactorGroup", "Date"])
    grouped = history.groupby(["Country", "FactorGroup"])["SelectedCountryScore"]
    rolling_mean = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    rolling_std = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=1))
    history["TrailingZScore"] = (history["SelectedCountryScore"] - rolling_mean) / rolling_std.replace(0, np.nan)
    history["CrossCountryZScore"] = history.groupby(["Date", "FactorGroup"])["SelectedCountryScore"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1) if s.notna().sum() > 1 and s.std(ddof=1) > 0 else np.nan
    )
    history["FactorRankWithinCountry"] = history.groupby(["Date", "Country"])["TrailingZScore"].rank(
        ascending=False, method="min"
    )
    history["CountryRankForFactor"] = history.groupby(["Date", "FactorGroup"])["SelectedCountryScore"].rank(
        ascending=False, method="min"
    )
    latest_dates = history.groupby("Country")["Date"].transform("max")
    latest = history[history["Date"].eq(latest_dates)].copy()
    top = history.sort_values(["Date", "Country", "FactorRankWithinCountry"]).groupby(["Date", "Country"], as_index=False).first()
    top = top.rename(columns={"FactorGroup": "TopFactorGroup", "TrailingZScore": "TopFactorTrailingZScore"})
    frequency = (
        top.groupby(["Country", "TopFactorGroup"], as_index=False)
        .size()
        .rename(columns={"size": "TopFactorPeriods"})
    )
    frequency["TopFactorFrequency"] = frequency["TopFactorPeriods"] / frequency.groupby("Country")["TopFactorPeriods"].transform("sum")
    return CountryFactorScoreDiagnostics(history, latest, top, frequency)


def write_country_factor_score_excel(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_country_factor_score_diagnostics(data, factor_scores, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        pd.DataFrame(
            {
                "Sheet": ["Country_Factor_History", "Latest_Country_Factors", "Top_Factor_History", "Top_Factor_Frequency"],
                "Content": [
                    "国別FactorScore推移。これはモデルシグナル/スタイル傾向であり、厳密な分散リスクではない",
                    "各国の最新FactorScore・Trailing Z-score・国間順位",
                    "各国・各時点でTrailing Z-scoreが最も高いFactor Group",
                    "各国で各Factor Groupがトップだった期間割合",
                ],
            }
        ).to_excel(writer, sheet_name="README", index=False)
        for name, frame in {
            "Country_Factor_History": result.history,
            "Latest_Country_Factors": result.latest,
            "Top_Factor_History": result.top_factor_history,
            "Top_Factor_Frequency": result.top_factor_frequency,
        }.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))


def write_country_factor_score_pdf(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    result = build_country_factor_score_diagnostics(data, factor_scores, config)
    if result.history.empty:
        return
    _setup_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        # Latest heatmap.
        latest = result.latest.pivot(index="Country", columns="FactorGroup", values="TrailingZScore")
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        image = ax.imshow(latest.fillna(0).to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(len(latest.columns)), latest.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(latest.index)), latest.index)
        ax.set_title("Latest country factor-score trends (trailing z-score)", fontsize=15, fontweight="bold")
        fig.colorbar(image, ax=ax, label="Trailing z-score")
        ax.text(0.01, -0.13, "Positive = the country's current model FactorScore is high versus its own trailing history. This is a signal/style indicator, not standalone volatility risk.", transform=ax.transAxes, fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Country pages.
        for country, frame in result.history.groupby("Country"):
            pivot = frame.pivot(index="Date", columns="FactorGroup", values="TrailingZScore").sort_index()
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for factor in pivot.columns:
                ax.plot(pivot.index, pivot[factor], label=factor, linewidth=1.5)
            ax.axhline(0, linewidth=0.8)
            ax.set_title(f"{country} | FactorScore trend", fontsize=15, fontweight="bold")
            ax.set_ylabel("Trailing z-score")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=3, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top factor frequency.
        freq = result.top_factor_frequency.copy()
        countries = sorted(freq["Country"].unique())
        factors = sorted(freq["TopFactorGroup"].unique())
        pivot = freq.pivot(index="Country", columns="TopFactorGroup", values="TopFactorFrequency").reindex(index=countries, columns=factors).fillna(0)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        bottom = np.zeros(len(pivot))
        for factor in pivot.columns:
            values = pivot[factor].to_numpy(float)
            ax.bar(pivot.index, values, bottom=bottom, label=factor)
            bottom += values
        ax.set_title("Factor most frequently ranked highest by country", fontsize=15, fontweight="bold")
        ax.set_ylabel("Share of evaluated periods")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
