from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from .master import FactorMeta
from .core_reporting import _setup_matplotlib as _setup_japanese_matplotlib
from .preprocessing import make_diagnostic_score


@dataclass
class RegressionFit:
    model: str
    r2: float
    adjusted_r2: float
    coefficients: list[float]
    knot: float | None
    x_grid: np.ndarray
    y_grid: np.ndarray


def _r2(y: np.ndarray, pred: np.ndarray, p: int) -> tuple[float, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < p + 2:
        return np.nan, np.nan
    ssr = np.sum((y - pred) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - p - 1) if np.isfinite(r2) and len(y) > p + 1 else np.nan
    return float(r2), float(adj)


def fit_regressions(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> dict[str, RegressionFit]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], float)
    y = np.asarray(y[mask], float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 5:
        return {}
    grid = np.linspace(x.min(), x.max(), 200)
    settings = config["binscatter"]["regressions"]
    fits: dict[str, RegressionFit] = {}

    if settings.get("linear", True):
        X = x[:, None]
        m = LinearRegression().fit(X, y)
        pred = m.predict(X)
        r2, adj = _r2(y, pred, 1)
        fits["linear"] = RegressionFit("linear", r2, adj, [float(m.intercept_), float(m.coef_[0])], None, grid, m.predict(grid[:, None]))

    if settings.get("quadratic", True):
        X = np.column_stack([x, x**2])
        m = LinearRegression().fit(X, y)
        pred = m.predict(X)
        r2, adj = _r2(y, pred, 2)
        fits["quadratic"] = RegressionFit(
            "quadratic", r2, adj, [float(m.intercept_), float(m.coef_[0]), float(m.coef_[1])], None,
            grid, m.predict(np.column_stack([grid, grid**2]))
        )

    if settings.get("broken_stick", True):
        mode = str(settings.get("broken_stick_knot", "auto"))
        if mode == "zero":
            candidates = [0.0]
        elif mode == "median":
            candidates = [float(np.median(x))]
        else:
            unique = np.unique(x)
            candidates = [float(v) for v in unique[2:-2]] if len(unique) > 6 else [float(np.median(x))]
        best: tuple[float, float, LinearRegression] | None = None
        for knot in candidates:
            X = np.column_stack([x, np.maximum(x - knot, 0.0)])
            m = LinearRegression().fit(X, y)
            pred = m.predict(X)
            rr, _ = _r2(y, pred, 2)
            if best is None or (np.isfinite(rr) and rr > best[0]):
                best = (rr, knot, m)
        if best is not None:
            rr, knot, m = best
            X = np.column_stack([x, np.maximum(x - knot, 0.0)])
            pred = m.predict(X)
            r2, adj = _r2(y, pred, 2)
            G = np.column_stack([grid, np.maximum(grid - knot, 0.0)])
            fits["broken_stick"] = RegressionFit(
                "broken_stick", r2, adj,
                [float(m.intercept_), float(m.coef_[0]), float(m.coef_[1])], float(knot), grid, m.predict(G)
            )
    return fits


def _weighted_average(values: pd.Series, weights: pd.Series | None) -> float:
    mask = values.notna()
    if weights is None:
        return float(values[mask].mean()) if mask.any() else np.nan
    w = pd.to_numeric(weights[mask], errors="coerce").clip(lower=0)
    v = pd.to_numeric(values[mask], errors="coerce")
    valid = v.notna() & w.notna()
    if not valid.any() or w[valid].sum() <= 0:
        return float(v[valid].mean()) if valid.any() else np.nan
    return float(np.average(v[valid], weights=w[valid]))


def calculate_time_averaged_bins(
    scope_data: pd.DataFrame,
    factor_code: str,
    meta: FactorMeta,
    config: dict[str, Any],
    n_bins: int,
    scope_type: str,
    scope_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    c = config["columns"]
    df = scope_data[[c["date"], c["isin"], c["market_cap"], "NextMonthReturn", factor_code]].copy()
    df = df.dropna(subset=[factor_code, "NextMonthReturn"])
    min_obs = int(config["binscatter"]["minimum_observations_per_period"][scope_type])
    valid_dates = df.groupby(c["date"]).size()
    valid_dates = valid_dates[valid_dates >= max(min_obs, n_bins * 2)].index
    df = df[df[c["date"]].isin(valid_dates)].copy()
    if df.empty:
        return pd.DataFrame(), {"Status": "insufficient_observations"}

    df["FactorScore"] = make_diagnostic_score(
        df,
        factor_code,
        c["date"],
        meta.direction,
        lower=float(config["preprocessing"].get("winsorize_lower", 0.01)),
        upper=float(config["preprocessing"].get("winsorize_upper", 0.99)),
        clip=float(config["preprocessing"].get("gaussian_clip", 3.0)),
    )

    def assign_bin_series(s: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        valid = s.notna()
        if valid.sum() < n_bins * 2:
            return out
        ranks = s.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=n_bins, labels=range(1, n_bins + 1)).astype(int).to_numpy()
        return out

    # transformを使い、groupby後も日付列を保持する。
    df["Bin"] = df.groupby(c["date"])["FactorScore"].transform(assign_bin_series)
    df = df.dropna(subset=["Bin"])
    df["Bin"] = df["Bin"].astype(int)
    rows: list[dict[str, Any]] = []
    weighting = config["binscatter"].get("weighting", "equal")
    x_stat = config["binscatter"].get("x_stat", "mean")

    for (date, bin_id), g in df.groupby([c["date"], "Bin"]):
        weights = g[c["market_cap"]] if weighting == "market_cap" else None
        x_value = float(g["FactorScore"].median()) if x_stat == "median" else _weighted_average(g["FactorScore"], weights)
        y_value = _weighted_average(g["NextMonthReturn"], weights)
        rows.append({
            "Date": date,
            "Bin": bin_id,
            "FactorScore": x_value,
            "Return": y_value,
            "Count": len(g),
        })
    period_bins = pd.DataFrame(rows)
    if period_bins.empty:
        return pd.DataFrame(), {"Status": "no_valid_bins"}
    grouped = period_bins.groupby("Bin")
    avg = grouped.agg(
        FactorScore=("FactorScore", "mean"),
        Return=("Return", "mean"),
        ReturnStd=("Return", "std"),
        Periods=("Date", "nunique"),
        TotalObservations=("Count", "sum"),
        MeanObservations=("Count", "mean"),
    ).reset_index()
    avg["StandardError"] = avg["ReturnStd"] / np.sqrt(avg["Periods"].clip(lower=1))
    avg["CI95"] = 1.96 * avg["StandardError"]
    avg["ScopeType"] = scope_type
    avg["ScopeLabel"] = scope_label
    avg["FactorCode"] = factor_code

    minimum_periods = int(config["binscatter"].get("minimum_periods", 18))
    if avg["Periods"].min() < minimum_periods:
        return pd.DataFrame(), {"Status": "insufficient_periods", "MinPeriods": int(avg["Periods"].min())}

    x = avg["FactorScore"].to_numpy(float)
    y = avg["Return"].to_numpy(float)
    pearson = pearsonr(x, y).statistic if len(x) >= 3 else np.nan
    spearman = spearmanr(x, y).statistic if len(x) >= 3 else np.nan
    fits = fit_regressions(x, y, config)
    summary: dict[str, Any] = {
        "Status": "ok",
        "ScopeType": scope_type,
        "ScopeLabel": scope_label,
        "FactorCode": factor_code,
        "NBins": len(avg),
        "NPeriods": int(avg["Periods"].min()),
        "NObservations": int(avg["TotalObservations"].sum()),
        "PearsonBin": float(pearson),
        "SpearmanBin": float(spearman),
        "TopBottomSpread": float(avg.loc[avg["Bin"].idxmax(), "Return"] - avg.loc[avg["Bin"].idxmin(), "Return"]),
    }
    for name, fit in fits.items():
        prefix = {"linear": "Linear", "quadratic": "Quadratic", "broken_stick": "BrokenStick"}[name]
        summary[f"{prefix}R2"] = fit.r2
        summary[f"{prefix}AdjustedR2"] = fit.adjusted_r2
        summary[f"{prefix}Coefficients"] = ",".join(f"{v:.10g}" for v in fit.coefficients)
        if fit.knot is not None:
            summary["BrokenStickKnot"] = fit.knot
    summary["Fits"] = fits
    return avg, summary


def build_scope_definitions(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    c = config["columns"]
    scopes: dict[str, list[tuple[str, pd.DataFrame]]] = {"all_universe": [], "by_country": [], "by_country_sector": []}
    if config["binscatter"]["scopes"].get("all_universe", True):
        scopes["all_universe"].append(("All", data))
    filters = config["binscatter"].get("scope_filters", {})
    countries = filters.get("countries") or sorted(data[c["country"]].dropna().astype(str).unique())
    sectors_filter = set(filters.get("sectors") or [])
    if config["binscatter"]["scopes"].get("by_country", True):
        for country in countries:
            g = data[data[c["country"]].astype(str) == str(country)]
            if not g.empty:
                scopes["by_country"].append((str(country), g))
    if config["binscatter"]["scopes"].get("by_country_sector", True):
        combos = []
        for (country, sector), g in data.groupby([c["country"], c["sector"]]):
            if str(country) not in set(map(str, countries)):
                continue
            if sectors_filter and str(sector) not in sectors_filter:
                continue
            combos.append((len(g), f"{country} | {sector}", g))
        combos.sort(key=lambda z: z[0], reverse=True)
        max_scopes = int(filters.get("max_country_sector_scopes", 0) or 0)
        if max_scopes > 0:
            combos = combos[:max_scopes]
        scopes["by_country_sector"] = [(label, g) for _, label, g in combos]
    return scopes


def run_binscatter_analysis(
    data: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame]:
    requested = config["binscatter"].get("factor_codes") or [k for k in metas if k in data.columns]
    factors = [k for k in requested if k in metas and k in data.columns]
    scopes = build_scope_definitions(data, config)
    plot_records: dict[str, list[dict[str, Any]]] = {k: [] for k in scopes}
    summary_rows = []
    bin_rows = []
    for scope_type, definitions in scopes.items():
        n_bins = int(config["binscatter"]["n_bins"][scope_type])
        for scope_label, frame in definitions:
            for code in factors:
                avg, summary = calculate_time_averaged_bins(frame, code, metas[code], config, n_bins, scope_type, scope_label)
                summary_clean = {k: v for k, v in summary.items() if k != "Fits"}
                summary_rows.append(summary_clean)
                if summary.get("Status") == "ok":
                    plot_records[scope_type].append({"avg": avg, "summary": summary})
                    bin_rows.append(avg)
    return plot_records, pd.DataFrame(summary_rows), pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()


def _setup_matplotlib() -> None:
    _setup_japanese_matplotlib()


def _stats_text(summary: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        f"n_periods={summary.get('NPeriods', 0):,}",
        f"bins={summary.get('NBins', 0)}  n={summary.get('NObservations', 0):,}",
    ]
    if config["binscatter"].get("show_correlations", True):
        lines += [
            f"Pearson(bin)={summary.get('PearsonBin', np.nan):.3f}",
            f"Spearman(bin)={summary.get('SpearmanBin', np.nan):.3f}",
        ]
    if config["binscatter"].get("show_r_squared", True):
        lines += [
            f"R2 Linear={summary.get('LinearR2', np.nan):.3f}",
            f"R2 Quadratic={summary.get('QuadraticR2', np.nan):.3f}",
            f"R2 Broken={summary.get('BrokenStickR2', np.nan):.3f}",
        ]
    if config["binscatter"].get("show_top_bottom_spread", True):
        lines.append(f"Top-Bottom={summary.get('TopBottomSpread', np.nan):.3%}")
    return "\n".join(lines)


def write_binscatter_pdf(records: list[dict[str, Any]], output_path: Path, title_prefix: str, config: dict[str, Any]) -> None:
    _setup_matplotlib()
    plots_per_page = int(config["binscatter"].get("plots_per_page", 6))
    ncols = 3 if plots_per_page >= 6 else 2
    nrows = int(np.ceil(plots_per_page / ncols))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        if not records:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.text(0.5, 0.5, "条件を満たすBinscatter結果がありません。", ha="center", va="center", fontsize=16)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return
        for start in range(0, len(records), plots_per_page):
            subset = records[start:start + plots_per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9.5), squeeze=False)
            fig.suptitle(title_prefix, fontsize=15, fontweight="bold", y=0.995)
            for ax, rec in zip(axes.ravel(), subset):
                avg = rec["avg"]
                summary = rec["summary"]
                err_mode = config["binscatter"].get("error_bar", "standard_error")
                yerr = None if err_mode == "none" else avg["CI95"] if err_mode == "ci95" else avg["StandardError"]
                ax.errorbar(avg["FactorScore"], avg["Return"], yerr=yerr, fmt="o", markersize=6, capsize=2.5, alpha=0.9, label="Time-avg bin")
                fits: dict[str, RegressionFit] = summary.get("Fits", {})
                if "linear" in fits:
                    f = fits["linear"]
                    ax.plot(f.x_grid, f.y_grid, linewidth=1.8, label=f"Linear (R2={f.r2:.3f})")
                if "quadratic" in fits:
                    f = fits["quadratic"]
                    ax.plot(f.x_grid, f.y_grid, linestyle="--", linewidth=1.8, label=f"Quadratic (R2={f.r2:.3f})")
                if "broken_stick" in fits:
                    f = fits["broken_stick"]
                    ax.plot(f.x_grid, f.y_grid, linestyle="-.", linewidth=1.8, label=f"Broken (R2={f.r2:.3f}, k={f.knot:.2f})")
                    ax.axvline(f.knot, linewidth=0.7, alpha=0.35)
                ax.axvline(0, linewidth=0.7, alpha=0.25)
                ax.axhline(0, linewidth=0.7, alpha=0.25)
                ax.set_title(f"{summary['ScopeLabel']} | {summary['FactorCode']}  n={summary['NObservations']:,}", fontsize=10, fontweight="bold")
                ax.set_xlabel("FactorScore (std)")
                ax.set_ylabel("Forward Return")
                ax.grid(alpha=0.18)
                ax.text(0.02, 0.98, _stats_text(summary, config), transform=ax.transAxes, ha="left", va="top", fontsize=7.4,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.45"})
                ax.legend(loc="lower right", fontsize=6.8, framealpha=0.8)
            for ax in axes.ravel()[len(subset):]:
                ax.axis("off")
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
