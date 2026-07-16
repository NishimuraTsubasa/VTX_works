from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BinResult:
    by_date: pd.DataFrame
    summary: pd.DataFrame
    factor_summary: pd.DataFrame


def compute_factor_bins(panel: pd.DataFrame, factor_map: dict[str, str], config: dict) -> BinResult:
    date_col = config["columns"]["date"]
    q_requested = int(config["diagnostics"].get("quantile_bins", 5))
    min_obs = int(config["diagnostics"].get("minimum_bin_observations", 10))
    rows: list[dict] = []

    for factor, z_col in factor_map.items():
        for date, g in panel[[date_col, z_col, "forward_return"]].groupby(date_col):
            valid = g.dropna(subset=[z_col, "forward_return"]).copy()
            if len(valid) < max(q_requested * min_obs, q_requested + 2):
                continue
            try:
                bins = pd.qcut(valid[z_col], q=q_requested, labels=False, duplicates="drop")
            except ValueError:
                continue
            valid["bin"] = pd.to_numeric(bins, errors="coerce") + 1
            valid = valid.dropna(subset=["bin"])
            q_actual = int(valid["bin"].nunique())
            if q_actual < 3:
                continue
            for bin_no, b in valid.groupby("bin"):
                n = len(b)
                rows.append({
                    date_col: date,
                    "factor": factor,
                    "bin": int(bin_no),
                    "bin_count": n,
                    "factor_mean": float(b[z_col].mean()),
                    "factor_median": float(b[z_col].median()),
                    "forward_return_mean": float(b["forward_return"].mean()),
                    "forward_return_median": float(b["forward_return"].median()),
                    "forward_return_std": float(b["forward_return"].std(ddof=1)) if n > 1 else np.nan,
                    "forward_return_se": float(b["forward_return"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                    "actual_bin_count": q_actual,
                })

    by_date = pd.DataFrame(rows)
    if by_date.empty:
        return BinResult(by_date, pd.DataFrame(), pd.DataFrame())

    summary = (
        by_date.groupby(["factor", "bin"], as_index=False)
        .agg(
            mean_factor_value=("factor_mean", "mean"),
            mean_forward_return=("forward_return_mean", "mean"),
            median_forward_return=("forward_return_mean", "median"),
            std_across_dates=("forward_return_mean", "std"),
            positive_month_rate=("forward_return_mean", lambda x: float((x > 0).mean())),
            date_count=("date", "nunique"),
            total_observations=("bin_count", "sum"),
        )
    )
    summary["se_across_dates"] = summary["std_across_dates"] / np.sqrt(summary["date_count"].clip(lower=1))

    factor_rows: list[dict] = []
    for factor, g in by_date.groupby("factor"):
        pivot = g.pivot_table(index=date_col, columns="bin", values="forward_return_mean")
        if pivot.empty:
            continue
        low_col, high_col = pivot.columns.min(), pivot.columns.max()
        spread = pivot[high_col] - pivot[low_col]
        bin_means = summary[summary["factor"] == factor].sort_values("bin")["mean_forward_return"].to_numpy()
        monotonic_steps = np.diff(bin_means)
        monotonicity = float(np.mean(monotonic_steps >= 0)) if len(monotonic_steps) else np.nan
        factor_rows.append({
            "factor": factor,
            "mean_top_bottom_spread": float(spread.mean()),
            "spread_positive_rate": float((spread > 0).mean()),
            "spread_std": float(spread.std(ddof=1)) if spread.notna().sum() > 1 else np.nan,
            "spread_ir": float(spread.mean() / spread.std(ddof=1)) if spread.notna().sum() > 1 and spread.std(ddof=1) > 0 else np.nan,
            "monotonicity_score": monotonicity,
            "date_count": int(spread.notna().sum()),
        })
    return BinResult(by_date=by_date, summary=summary, factor_summary=pd.DataFrame(factor_rows))
