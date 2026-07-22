from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .evaluation import cumulative_long_short, cumulative_quintile_returns
from .font_support import setup_japanese_matplotlib


def _setup_matplotlib(explicit_font_path: str | Path | None = None) -> str:
    """Configure a verified Japanese font for every PDF report.

    A concrete font file is registered before plotting, so the result does not
    depend on Matplotlib's potentially stale font cache.  Missing Japanese
    fonts raise a clear error instead of silently producing mojibake.
    """
    family, _ = setup_japanese_matplotlib(explicit_font_path)
    return family


def write_quintile_pdf(quintiles: pd.DataFrame, output_path: Path, config: dict[str, Any]) -> None:
    if quintiles.empty:
        return
    _setup_matplotlib()
    curves = cumulative_quintile_returns(quintiles)
    with PdfPages(output_path) as pdf:
        for scenario, frame in curves.items():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for q in sorted(frame.columns):
                ax.plot(frame.index, frame[q], label=f"Q{int(q)}", linewidth=1.7)
            ax.set_title(f"{scenario} | スコア5分位累積リターン", fontsize=15, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("累積資産価値")
            ax.grid(alpha=0.2)
            ax.legend(ncol=5, loc="upper left")
            ax.text(0.01, 0.01, "Q1=最低スコア20%、Q5=最高スコア20%。各月等ウェイトで翌月リターンを評価。", transform=ax.transAxes)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_scenario_comparison_pdf(
    summary: pd.DataFrame,
    quintiles: pd.DataFrame,
    rank_ic_history: pd.DataFrame,
    common_quintiles: pd.DataFrame,
    common_rank_ic_history: pd.DataFrame,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    _setup_matplotlib()
    qn = int(config["evaluation"].get("quintiles", 5))
    rolling = int(config["evaluation"].get("rolling_rank_ic_periods", 12))
    full_curves = cumulative_long_short(quintiles, qn)
    common_curves = cumulative_long_short(common_quintiles, qn)
    benchmark = str(config["evaluation"].get("common_oos", {}).get("benchmark_scenario", "N00_Direct_RawScore_EW"))
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for name, series in common_curves.items():
            if not series.empty:
                series = series / series.iloc[0]
            ax.plot(series.index, series.values, label=name, linewidth=1.6)
        ax.set_title("シナリオ別 Q5-Q1累積リターン（共通OOS）", fontsize=15, fontweight="bold")
        ax.set_ylabel("累積資産価値")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for name, series in full_curves.items():
            ax.plot(series.index, series.values, label=name, linewidth=1.5)
        ax.set_title("シナリオ別 Q5-Q1累積リターン（各モデル利用可能期間）", fontsize=15, fontweight="bold")
        ax.set_ylabel("累積資産価値")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for name, frame in common_rank_ic_history.groupby("Scenario"):
            frame = frame.sort_values("Date")
            ax.plot(frame["Date"], frame["RankIC"].rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=name, linewidth=1.5)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"シナリオ別 ローリング{rolling}期間Rank IC（共通OOS）", fontsize=15, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not summary.empty:
            metrics = ["CommonMeanRankIC", "CommonQ5MinusQ1Sharpe", "CommonQuintileMonotonicity", "CommonQ5MinusQ1MaxDrawdown"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, metric in zip(axes.ravel(), metrics):
                frame = summary.sort_values(metric)
                ax.barh(frame["Scenario"], frame[metric])
                ax.set_title(metric)
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle("個別銘柄スコアリングモデル主要比較（共通OOS）", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not common_rank_ic_history.empty and benchmark in set(common_rank_ic_history["Scenario"]):
            pivot = common_rank_ic_history.pivot(index="Date", columns="Scenario", values="RankIC").sort_index()
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for name in pivot.columns:
                if name == benchmark:
                    continue
                delta = (pivot[name] - pivot[benchmark]).dropna()
                ax.plot(delta.index, delta.rolling(rolling, min_periods=max(3, rolling // 2)).mean(), label=f"{name} - {benchmark}")
            ax.axhline(0, linewidth=0.8)
            ax.set_title(f"ローリング{rolling}期間Rank IC差（対{benchmark}）", fontsize=15, fontweight="bold")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
