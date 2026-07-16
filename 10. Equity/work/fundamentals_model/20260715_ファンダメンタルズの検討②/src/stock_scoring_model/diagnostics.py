from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge

from .modeling import _fit, _predict, design_matrix
from .utils import chunks, ensure_dir

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


def _factor_label(factor: str, config: dict) -> str:
    labels = config.get("runtime", {}).get("factor_labels", {})
    label = labels.get(factor, factor)
    return f"{factor} {label}" if label != factor else factor



def _empty_page(pdf: PdfPages, title: str, message: str = "No data") -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.5, 0.44, message, ha="center", va="center", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def create_factor_scatter_pdf(panel: pd.DataFrame, factor_map: dict[str, str], config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    per_page = int(config["diagnostics"]["scatter_plots_per_page"])
    max_points = int(config["diagnostics"]["scatter_max_points"])
    rng = np.random.default_rng(int(config["project"]["random_seed"]))
    candidates = [m for m in config["model"]["candidate_models"] if m in {"linear", "piecewise", "quadratic", "combined_ridge"}]

    with PdfPages(output_path) as pdf:
        if not factor_map:
            _empty_page(pdf, "単一ファクター散布図", "ファクターがありません")
            return
        for batch in chunks(list(factor_map.items()), per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, (factor, z_col) in zip(axes.ravel(), batch):
                ax.set_visible(True)
                data = panel[[z_col, "forward_return"]].dropna()
                if len(data) > max_points:
                    data = data.iloc[rng.choice(len(data), max_points, replace=False)]
                ax.scatter(data[z_col], data["forward_return"], s=8, alpha=0.18)
                if len(data) >= 30:
                    grid = np.linspace(data[z_col].quantile(0.01), data[z_col].quantile(0.99), 200)
                    for model_name in candidates:
                        if model_name == "combined_ridge":
                            X = design_matrix(data[z_col].to_numpy(), model_name, float(config["diagnostics"]["piecewise_knot"]))
                            model = Ridge(alpha=1.0, fit_intercept=True).fit(X, data["forward_return"].to_numpy())
                        else:
                            model = _fit(model_name, data[z_col].to_numpy(), data["forward_return"].to_numpy(), None, config)
                        ax.plot(grid, _predict(model, model_name, grid, config), linewidth=1.5, label=model_name)
                ax.axhline(0, linewidth=0.7)
                ax.set_title(_factor_label(factor, config))
                ax.set_xlabel("前処理後ファクター値")
                ax.set_ylabel("翌期個別銘柄リターン")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2)
            fig.suptitle("単一ファクター散布図と候補回帰線", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def create_factor_bin_pdf(bin_summary: pd.DataFrame, factor_summary: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    per_page = int(config["diagnostics"].get("bin_plots_per_page", 4))
    with PdfPages(output_path) as pdf:
        if bin_summary.empty:
            _empty_page(pdf, "ファクタービン分析", "ビン分析に必要な観測数がありません")
            return
        factors = sorted(bin_summary["factor"].unique())
        stats = factor_summary.set_index("factor") if not factor_summary.empty else pd.DataFrame()
        for batch in chunks(factors, per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, factor in zip(axes.ravel(), batch):
                ax.set_visible(True)
                g = bin_summary[bin_summary["factor"] == factor].sort_values("bin")
                ax.errorbar(
                    g["bin"], g["mean_forward_return"],
                    yerr=g["se_across_dates"], marker="o", capsize=3, linewidth=1.5,
                )
                ax.axhline(0, linewidth=0.8)
                ax.set_xlabel("ファクタービン（低 → 高）")
                ax.set_ylabel("翌期平均リターン")
                title = _factor_label(factor, config)
                if not stats.empty and factor in stats.index:
                    row = stats.loc[factor]
                    title += f"\nTop-Bottom={row['mean_top_bottom_spread']:.3%}, 単調性={row['monotonicity_score']:.2f}"
                ax.set_title(title)
                ax.grid(alpha=0.2)
                ax.set_xticks(g["bin"].astype(int))
            fig.suptitle("ファクター分位別の翌期リターン", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


def create_factor_performance_pdf(metrics_by_date: pd.DataFrame, selection: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    """Plot rolling OOS RankIC for every candidate model, emphasizing the selected one."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    per_page = int(config["report"]["line_plots_per_page"])
    with PdfPages(output_path) as pdf:
        if metrics_by_date.empty or selection.empty:
            _empty_page(pdf, "ファクターモデルOOS評価")
            return
        selected = selection.set_index("factor")["selected_model"].to_dict()
        factors = list(selected)
        rolling_window = int(config["diagnostics"]["rolling_accuracy_window"][config["data"]["frequency"]])
        model_order = list(config["model"].get("candidate_models", []))
        for batch in chunks(factors, per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, factor in zip(axes.ravel(), batch):
                ax.set_visible(True)
                fg = metrics_by_date[metrics_by_date["factor"] == factor].copy()
                for model_name in model_order:
                    g = fg[fg["model"] == model_name].sort_values("date").copy()
                    if g.empty:
                        continue
                    g["rolling_rank_ic"] = g["rank_ic"].rolling(
                        rolling_window, min_periods=max(3, rolling_window // 3)
                    ).mean()
                    is_selected = model_name == selected[factor]
                    ax.plot(
                        g["date"], g["rolling_rank_ic"],
                        linewidth=2.4 if is_selected else 0.9,
                        alpha=1.0 if is_selected else 0.55,
                        label=f"{model_name}{' [採用]' if is_selected else ''}",
                    )
                ax.axhline(0, linewidth=0.8)
                ax.set_title(f"{_factor_label(factor, config)} | 採用={selected[factor]}")
                ax.set_ylabel(f"{rolling_window}期移動平均 OOS RankIC")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, ncol=2)
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle("候補4モデルのOOS精度推移（太線が採用モデル）", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def create_factor_model_selection_pdf(
    selection_detail: pd.DataFrame,
    selection: pd.DataFrame,
    config: dict,
    output_path: str | Path,
) -> None:
    """Explain model selection using candidate mean OOS RankIC and the one-SE rule."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with PdfPages(output_path) as pdf:
        if selection_detail.empty or selection.empty:
            _empty_page(pdf, "単一ファクターモデル選択根拠")
            return

        # Methodology page.
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        settings = config["model"].get("model_selection", {})
        multiplier = float(settings.get("one_se_multiplier", 1.0))
        lines = [
            "単一ファクターモデルの選択ルール",
            "",
            "1. 各候補（Linear / Piecewise / Quadratic / Combined Ridge）をWalk-forwardで評価する。",
            "2. 主指標は各テスト月のSpearman RankICを平均したOOS平均RankIC。",
            "3. OOS平均RankICが最大のモデルを best_raw_model とする。",
            f"4. 閾値 = bestの平均RankIC - {multiplier:.1f} × bestのRankIC標準誤差。",
            "5. 閾値以上の候補は、最良モデルと統計的に明確な差がない『ほぼ同等』候補とみなす。",
            "6. ほぼ同等候補の中から、複雑度が最も低いモデルを採用する。",
            "   複雑度: Linear=1, Piecewise=2, Quadratic=2, Combined Ridge=3。",
            "7. その後、平均RankIC・正符号率・評価期間数の採用ゲートを確認する。",
            "",
            "Linearが多く選ばれる主な理由",
            "非線形モデルの平均RankICがわずかに高くても、その差が推定誤差の範囲内であれば、",
            "将来安定性と説明可能性を優先してLinearを選ぶためです。",
            "Linearが無条件に優先されるのではなく、非線形モデルが1標準誤差を超える改善を示せば、",
            "Piecewise / Quadratic / Combined Ridgeが選ばれます。",
        ]
        ax.text(0.05, 0.95, lines[0], va="top", fontsize=18, weight="bold")
        ax.text(0.06, 0.88, "\n".join(lines[2:]), va="top", fontsize=11, linespacing=1.55)
        pdf.savefig(fig)
        plt.close(fig)

        per_page = int(config["report"].get("line_plots_per_page", 4))
        factors = selection["factor"].tolist()
        for batch in chunks(factors, per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, factor in zip(axes.ravel(), batch):
                ax.set_visible(True)
                g = selection_detail[selection_detail["factor"] == factor].copy()
                if g.empty:
                    continue
                order = list(config["model"].get("candidate_models", g["model"].tolist()))
                g["model"] = pd.Categorical(g["model"], categories=order, ordered=True)
                g = g.sort_values("model")
                x = np.arange(len(g))
                y = g["mean_rank_ic"].to_numpy(dtype=float)
                se = g["rank_ic_se"].fillna(0.0).to_numpy(dtype=float)
                bars = ax.bar(x, y, yerr=se, capsize=3, alpha=0.75)
                for bar, selected_flag, eligible in zip(bars, g["selected"], g["within_one_se"]):
                    if selected_flag:
                        bar.set_linewidth(2.8)
                        bar.set_edgecolor("black")
                    elif not eligible:
                        bar.set_alpha(0.30)
                threshold = float(g["one_se_threshold"].iloc[0])
                ax.axhline(threshold, linestyle="--", linewidth=1.2, label="1-SE閾値")
                ax.axhline(0, linewidth=0.7)
                ax.set_xticks(x, [str(v) for v in g["model"]], rotation=20, ha="right")
                sel = selection[selection["factor"] == factor].iloc[0]
                ax.set_title(
                    f"{_factor_label(factor, config)}\n採用={sel['selected_model']} / 最良={sel['best_raw_model']}"
                )
                ax.set_ylabel("OOS平均RankIC（誤差棒=標準誤差）")
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=7)
                reason = str(sel.get("selection_reason_jp", ""))
                ax.text(0.02, -0.32, reason, transform=ax.transAxes, fontsize=8, va="top", wrap=True)
            fig.suptitle("候補モデル比較と採用根拠", fontsize=16)
            fig.tight_layout(rect=[0, 0.04, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

def create_scenario_quintile_cumulative_pdf(
    quintile_history: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    config: dict,
    output_path: str | Path,
) -> None:
    """シナリオ別にQ1-Q5累積リターンを複数パネルで表示する。"""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    per_page = int(config["diagnostics"].get("scenario_plots_per_page", 4))
    with PdfPages(output_path) as pdf:
        if quintile_history.empty:
            _empty_page(pdf, "スコア5分位ポートフォリオ累積リターン", "評価データがありません")
            return
        scenario_ids = list(quintile_history["scenario_id"].drop_duplicates())
        names = scenario_summary.set_index("scenario_id")["scenario_name"].to_dict() if not scenario_summary.empty else {}
        for batch in chunks(scenario_ids, per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, sid in zip(axes.ravel(), batch):
                ax.set_visible(True)
                g = quintile_history[quintile_history["scenario_id"].eq(sid)].copy()
                pivot = g.pivot_table(index="date", columns="quintile", values="quintile_return", aggfunc="mean").sort_index()
                for q in sorted(pivot.columns):
                    cumulative = (1.0 + pivot[q].fillna(0.0)).cumprod() - 1.0
                    ax.plot(cumulative.index, cumulative.values, linewidth=1.4, label=f"Q{int(q)}")
                ax.axhline(0, linewidth=0.7)
                ax.set_title(f"{sid}\n{names.get(sid, '')}")
                ax.set_ylabel("累積リターン")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, ncol=5)
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle("個別銘柄スコア5分位ポートフォリオの累積リターン", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


def create_scenario_comparison_pdf(
    scenario_summary: pd.DataFrame,
    rank_ic_history: pd.DataFrame,
    long_short_history: pd.DataFrame,
    config: dict,
    output_path: str | Path,
) -> None:
    """シナリオ間のQ5-Q1累積とRankICを比較する。"""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rolling = int(config["diagnostics"]["rolling_rank_ic_window"][config["data"]["frequency"]])
    with PdfPages(output_path) as pdf:
        if scenario_summary.empty:
            _empty_page(pdf, "個別銘柄スコアリングモデル比較", "評価データがありません")
            return

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for sid, g in long_short_history.groupby("scenario_id"):
            g = g.sort_values("date")
            cumulative = (1.0 + g["q5_minus_q1"].fillna(0.0)).cumprod() - 1.0
            ax.plot(g["date"], cumulative, linewidth=1.6, label=sid.split("_")[0])
        ax.axhline(0, linewidth=0.7)
        ax.set_title("シナリオ別 Q5-Q1 累積リターン")
        ax.set_ylabel("累積リターン")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for sid, g in rank_ic_history.groupby("scenario_id"):
            g = g.sort_values("date").copy()
            g["rolling_rank_ic"] = g["rank_ic"].rolling(rolling, min_periods=max(3, rolling // 3)).mean()
            ax.plot(g["date"], g["rolling_rank_ic"], linewidth=1.5, label=sid.split("_")[0])
        ax.axhline(0, linewidth=0.7)
        ax.set_title(f"シナリオ別 {rolling}期移動平均 RankIC")
        ax.set_ylabel("RankIC")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        metrics = [
            ("mean_rank_ic", "平均RankIC"),
            ("q5_minus_q1_sharpe", "Q5-Q1 Sharpe"),
            ("quintile_monotonicity", "5分位単調性"),
            ("q5_minus_q1_max_drawdown", "Q5-Q1最大ドローダウン"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        ordered = scenario_summary.sort_values("scenario_id")
        for ax, (col, title) in zip(axes.ravel(), metrics):
            ax.bar(ordered["scenario_id"].str.split("_").str[0], ordered[col])
            ax.axhline(0, linewidth=0.7)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle("個別銘柄スコアリングモデルの主要評価指標", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
