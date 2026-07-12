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


def _group_label(group: str, config: dict) -> str:
    labels = config.get("runtime", {}).get("group_labels", {})
    label = labels.get(group, group)
    return f"{group} ({label})" if label != group else group


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

def _heatmap(ax, matrix: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    if matrix.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_index_exposure_pdf(index_scores: pd.DataFrame, exposures: pd.DataFrame, contributions: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    date_col = config["columns"]["date"]
    with PdfPages(output_path) as pdf:
        if index_scores.empty:
            _empty_page(pdf, "指数ファクターエクスポージャー")
            return
        latest_date = index_scores[date_col].max()
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        latest_scores = index_scores[index_scores[date_col] == latest_date].set_index("index_name")
        score_cols = [c for c in ["index_alpha", "index_score", "index_breadth_count_based", "selection_weight_coverage"] if c in latest_scores]
        _heatmap(axes[0], latest_scores[score_cols], f"最新指数スコア・Breadth ({latest_date:%Y-%m-%d})")
        latest_exp = exposures[exposures[date_col] == exposures[date_col].max()].set_index("index_name") if not exposures.empty else pd.DataFrame()
        exp_cols = [c for c in latest_exp.columns if c.startswith("exposure_")]
        exp_matrix = latest_exp[exp_cols].copy()
        exp_matrix.columns = [c.replace("exposure_", "") for c in exp_cols]
        _heatmap(axes[1], exp_matrix, "最新の代表ユニバース加重ファクター傾向")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not contributions.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            latest_contrib = contributions[contributions[date_col] == contributions[date_col].max()].set_index("index_name")
            contrib_cols = [c for c in latest_contrib.columns if c.startswith("contribution_")]
            _heatmap(ax, latest_contrib[contrib_cols], "指数別ファクターグループ予測寄与")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def create_index_factor_trends_pdf(
    exposures: pd.DataFrame,
    contributions: pd.DataFrame,
    config: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    date_col = config["columns"]["date"]
    per_page = int(config["diagnostics"].get("index_trends_per_page", 4))
    max_factors = int(config["diagnostics"].get("index_trend_max_factors", 6))
    with PdfPages(output_path) as pdf:
        if exposures.empty and contributions.empty:
            _empty_page(pdf, "指数ファクター傾向時系列")
            return
        if not exposures.empty:
            indices = sorted(exposures["index_name"].unique())
            exp_cols_all = [c for c in exposures.columns if c.startswith("exposure_")]
            for batch in chunks(indices, per_page):
                ncols = 2
                nrows = int(np.ceil(len(batch) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
                for ax in axes.ravel():
                    ax.set_visible(False)
                for ax, index_name in zip(axes.ravel(), batch):
                    ax.set_visible(True)
                    g = exposures[exposures["index_name"] == index_name].sort_values(date_col)
                    if len(exp_cols_all) > max_factors:
                        importance = g[exp_cols_all].abs().mean().sort_values(ascending=False)
                        exp_cols = importance.head(max_factors).index.tolist()
                    else:
                        exp_cols = exp_cols_all
                    for col in exp_cols:
                        factor = col.replace("exposure_", "")
                        ax.plot(g[date_col], g[col], linewidth=1.2, label=factor)
                    ax.axhline(0, linewidth=0.7)
                    ax.set_title(index_name)
                    ax.set_ylabel("個別ファクター・エクスポージャー")
                    ax.grid(alpha=0.2)
                    ax.legend(fontsize=7, ncol=2)
                    ax.tick_params(axis="x", rotation=30)
                fig.suptitle("指数別・個別ファクター傾向の時系列", fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

        if not contributions.empty:
            indices = sorted(contributions["index_name"].unique())
            group_cols = [c for c in contributions.columns if c.startswith("contribution_")]
            for batch in chunks(indices, per_page):
                ncols = 2
                nrows = int(np.ceil(len(batch) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
                for ax in axes.ravel():
                    ax.set_visible(False)
                for ax, index_name in zip(axes.ravel(), batch):
                    ax.set_visible(True)
                    g = contributions[contributions["index_name"] == index_name].sort_values(date_col)
                    for col in group_cols:
                        group = col.replace("contribution_", "")
                        ax.plot(g[date_col], g[col], linewidth=1.3, label=_group_label(group, config))
                    ax.axhline(0, linewidth=0.7)
                    ax.set_title(index_name)
                    ax.set_ylabel("グループ予測寄与")
                    ax.grid(alpha=0.2)
                    ax.legend(fontsize=7, ncol=2)
                    ax.tick_params(axis="x", rotation=30)
                fig.suptitle("指数別・Value等のグループ傾向時系列", fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

def create_model_accuracy_pdf(history: pd.DataFrame, detail: pd.DataFrame, per_index_summary: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    date_col = config["columns"]["date"]
    with PdfPages(output_path) as pdf:
        if history.empty:
            _empty_page(pdf, "モデル正解率・予測精度推移")
            return
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        axes[0, 0].plot(history["date"], history["index_rank_ic"], alpha=0.4, label="期間RankIC")
        axes[0, 0].plot(history["date"], history["rolling_rank_ic"], linewidth=1.8, label="移動平均")
        axes[0, 0].axhline(0, linewidth=0.7)
        axes[0, 0].set_title("指数横断RankIC")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.2)

        axes[0, 1].plot(history["date"], history["directional_accuracy"], alpha=0.4, label="期間正解率")
        axes[0, 1].plot(history["date"], history["rolling_directional_accuracy"], linewidth=1.8, label="移動平均")
        axes[0, 1].axhline(0.5, linewidth=0.7)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title("方向予測の正解率")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.2)

        axes[1, 0].plot(history["date"], history["top_bottom_spread"], alpha=0.4, label="期間スプレッド")
        axes[1, 0].plot(history["date"], history["rolling_top_bottom_spread"], linewidth=1.8, label="移動平均")
        axes[1, 0].axhline(0, linewidth=0.7)
        axes[1, 0].set_title("上位指数 - 下位指数リターン")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.2)

        axes[1, 1].plot(history["date"], history["cumulative_top_bottom_spread"], linewidth=1.8)
        axes[1, 1].axhline(0, linewidth=0.7)
        axes[1, 1].set_title("上位 - 下位スプレッド累積")
        axes[1, 1].grid(alpha=0.2)
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("最終指数モデルのOOS精度推移", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        if not per_index_summary.empty:
            fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
            matrix = per_index_summary.set_index("index_name")[[
                "time_series_correlation", "time_series_rank_correlation", "directional_accuracy", "rmse"
            ]]
            _heatmap(axes[0], matrix, "指数別の時系列予測精度")
            latest = detail.dropna(subset=["index_alpha", "forward_futures_return"]).sort_values(date_col)
            if not latest.empty:
                latest = latest.groupby("index_name").tail(1).set_index("index_name")[["index_alpha", "forward_futures_return"]]
            _heatmap(axes[1], latest, "最新予測値と実現翌期リターン")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def create_universe_selection_pdf(quality: pd.DataFrame, allocation: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    date_col = config["columns"]["date"]
    sector_col = config["columns"]["sector"]
    with PdfPages(output_path) as pdf:
        if quality.empty:
            _empty_page(pdf, "代表銘柄ユニバース選定")
            return
        latest_date = quality[date_col].max()
        latest_q = quality[quality[date_col] == latest_date].set_index("index_name")
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        qcols = [c for c in ["target_count", "selected_count", "candidate_count", "sector_weight_coverage", "tracking_correlation", "tracking_rmse", "actual_constituent_share"] if c in latest_q]
        _heatmap(axes[0], latest_q[qcols], f"最新の代表ユニバース選定品質 ({latest_date:%Y-%m-%d})")
        latest_a = allocation[allocation[date_col] == allocation[date_col].max()].copy() if not allocation.empty else pd.DataFrame()
        if not latest_a.empty:
            pivot = latest_a.pivot_table(index=["index_name", sector_col], values=["target_sector_weight", "actual_selected_count", "available_count"], aggfunc="first")
        else:
            pivot = pd.DataFrame()
        _heatmap(axes[1], pivot, "セクター配分・選定数")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        indices = sorted(quality["index_name"].unique())
        per_page = int(config["report"].get("line_plots_per_page", 4))
        for batch in chunks(indices, per_page):
            ncols = 2
            nrows = int(np.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, index_name in zip(axes.ravel(), batch):
                ax.set_visible(True)
                g = quality[quality["index_name"] == index_name].sort_values(date_col)
                if "tracking_correlation" in g:
                    ax.plot(g[date_col], g["tracking_correlation"], label="Tracking Corr")
                if "sector_weight_coverage" in g:
                    ax.plot(g[date_col], g["sector_weight_coverage"], label="Sector Coverage")
                ax.set_ylim(-0.1, 1.05)
                ax.set_title(index_name)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle("代表ユニバース選定品質の推移", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def create_futures_risk_pdf(latest: pd.DataFrame, history: pd.DataFrame, corr: pd.DataFrame, config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        if not latest.empty:
            matrix = latest.set_index("index_name")[["annualized_volatility", "annualized_downside_volatility", "historical_var", "historical_expected_shortfall", "max_drawdown"]]
            _heatmap(axes[0], matrix, "最新の指数先物リスク")
        else:
            axes[0].axis("off")
        _heatmap(axes[1], corr, "直近指数先物リターン相関")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not history.empty:
            indices = sorted(history["index_name"].unique())
            per_page = int(config["report"]["line_plots_per_page"])
            for batch in chunks(indices, per_page):
                ncols = 2
                nrows = int(np.ceil(len(batch) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
                for ax in axes.ravel():
                    ax.set_visible(False)
                for ax, index_name in zip(axes.ravel(), batch):
                    ax.set_visible(True)
                    g = history[history["index_name"] == index_name].sort_values("date")
                    ax.plot(g["date"], g["annualized_volatility"], label="Volatility")
                    ax.plot(g["date"], g["annualized_downside_volatility"], label="Downside vol")
                    ax.set_title(index_name)
                    ax.grid(alpha=0.2)
                    ax.legend(fontsize=8)
                    ax.tick_params(axis="x", rotation=30)
                fig.suptitle("指数先物ローリングリスク", fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
