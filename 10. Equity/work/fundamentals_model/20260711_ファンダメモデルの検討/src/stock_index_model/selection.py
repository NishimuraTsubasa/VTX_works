from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOGGER = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    history: pd.DataFrame
    quality: pd.DataFrame
    sector_allocation: pd.DataFrame


def _rank01(s: pd.Series) -> pd.Series:
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    return s.rank(method="average", pct=True).fillna(0.0)


def _selection_key_series(df: pd.DataFrame, config: dict) -> pd.Series:
    sel = config["universe_selection"]
    key_column = sel.get("key_column", config["columns"]["country"])
    if key_column not in df:
        return pd.Series("", index=df.index, dtype=object)
    mapping = sel.get("stock_key_value_map", {})
    return df[key_column].astype(str).replace(mapping)


def _index_selection_key(index_name: str, config: dict) -> str:
    sel = config["universe_selection"]
    if index_name in sel.get("index_key_map", {}):
        return str(sel["index_key_map"][index_name])
    country = config["data"].get("index_country_map", {}).get(index_name, index_name)
    return str(sel.get("stock_key_value_map", {}).get(country, country))


def _target_count(index_name: str, selection_key: str, config: dict) -> int:
    sel = config["universe_selection"]
    if index_name in sel.get("target_count_by_index", {}):
        return int(sel["target_count_by_index"][index_name])
    if selection_key in sel.get("target_count_by_key", {}):
        return int(sel["target_count_by_key"][selection_key])
    return int(config["aggregation"].get("minimum_constituent_count", 5))


def allocate_sector_counts(
    target_count: int,
    sector_weights: pd.Series,
    available_counts: pd.Series,
) -> dict[str, int]:
    """Largest-remainder allocation with availability caps and exact total where feasible."""
    weights = sector_weights.astype(float).clip(lower=0.0)
    available = available_counts.reindex(weights.index).fillna(0).astype(int).clip(lower=0)
    feasible_total = min(int(target_count), int(available.sum()))
    if feasible_total <= 0 or weights.sum() <= 0:
        return {str(s): 0 for s in weights.index}

    active = (weights > 0) & (available > 0)
    weights = weights.where(active, 0.0)
    if weights.sum() <= 0:
        return {str(s): 0 for s in weights.index}
    weights = weights / weights.sum()

    raw = weights * feasible_total
    counts = np.floor(raw).astype(int)
    counts = np.minimum(counts, available)

    # Give each represented positive-weight sector one name where possible.
    active_sectors = list(weights[weights > 0].index)
    if feasible_total >= len(active_sectors):
        for sector in active_sectors:
            if counts.loc[sector] == 0 and available.loc[sector] > 0:
                counts.loc[sector] = 1

    # Reduce if minimum-one logic overshoots.
    while int(counts.sum()) > feasible_total:
        reducible = counts[counts > 0]
        if reducible.empty:
            break
        sector = min(reducible.index, key=lambda s: (raw.loc[s] - counts.loc[s], weights.loc[s]))
        counts.loc[sector] -= 1

    remainder = raw - np.floor(raw)
    while int(counts.sum()) < feasible_total:
        capacity = available - counts
        eligible = capacity[capacity > 0]
        if eligible.empty:
            break
        sector = max(
            eligible.index,
            key=lambda s: (remainder.loc[s], weights.loc[s], int(capacity.loc[s])),
        )
        counts.loc[sector] += 1
        # After the first largest-remainder pass, use proportional deficit.
        remainder.loc[sector] = raw.loc[sector] - counts.loc[sector]

    return {str(s): int(counts.loc[s]) for s in counts.index}


def _candidate_metrics(
    candidates: pd.DataFrame,
    stock_history: pd.DataFrame,
    index_history: pd.Series,
    selected_isins: list[str],
    config: dict,
) -> pd.DataFrame:
    cols = config["columns"]
    isin_col = cols["isin"]
    return_col = cols["stock_return"]
    score_cfg = config["universe_selection"]["selection_score"]

    pivot = stock_history.pivot_table(index=cols["date"], columns=isin_col, values=return_col)
    y = index_history.dropna()
    pivot = pivot.reindex(y.index)
    min_cov = float(config["universe_selection"].get("minimum_return_coverage", 0.6))

    rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        isin = getattr(row, isin_col)
        if isin in pivot:
            x = pivot[isin]
            valid = pd.concat([x, y], axis=1).dropna()
            coverage = len(valid) / max(len(y), 1)
            corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1])) if len(valid) >= 3 else np.nan
        else:
            coverage = 0.0
            corr = np.nan
        rows.append({isin_col: isin, "return_coverage": coverage, "corr_to_index": corr})
    metrics = candidates.merge(pd.DataFrame(rows), on=isin_col, how="left")
    metrics["corr_component"] = metrics["corr_to_index"].fillna(0.0).clip(lower=0.0)
    metrics["coverage_component"] = metrics["return_coverage"].fillna(0.0).clip(0.0, 1.0)
    market_cap_col = cols["market_cap"]
    if market_cap_col in metrics:
        metrics["market_cap_component"] = _rank01(pd.to_numeric(metrics[market_cap_col], errors="coerce"))
    else:
        metrics["market_cap_component"] = 0.0
    metrics["constituent_component"] = metrics.get("is_actual_constituent", False).astype(float)
    metrics["base_selection_score"] = (
        float(score_cfg.get("correlation_weight", 0.55)) * metrics["corr_component"]
        + float(score_cfg.get("coverage_weight", 0.20)) * metrics["coverage_component"]
        + float(score_cfg.get("market_cap_weight", 0.15)) * metrics["market_cap_component"]
        + float(score_cfg.get("constituent_weight", 0.10)) * metrics["constituent_component"]
    )
    # Candidates below minimum coverage are not discarded if the sector is sparse,
    # but are pushed to the bottom of the ranking.
    metrics.loc[metrics["return_coverage"] < min_cov, "base_selection_score"] -= 1.0

    if selected_isins and not pivot.empty:
        selected_cols = [c for c in selected_isins if c in pivot.columns]
        if selected_cols:
            corr_matrix = pivot.corr(min_periods=3)
            metrics["redundancy"] = metrics[isin_col].map(
                lambda x: float(corr_matrix.loc[x, selected_cols].abs().max())
                if x in corr_matrix.index and corr_matrix.loc[x, selected_cols].notna().any()
                else 0.0
            )
        else:
            metrics["redundancy"] = 0.0
    else:
        metrics["redundancy"] = 0.0
    metrics["selection_score"] = metrics["base_selection_score"] - float(
        score_cfg.get("redundancy_penalty", 0.20)
    ) * metrics["redundancy"]
    return metrics


def _select_sector_names(
    candidates: pd.DataFrame,
    n_select: int,
    stock_history: pd.DataFrame,
    index_history: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Select names with a vectorized score and a greedy redundancy penalty."""
    if n_select <= 0 or candidates.empty:
        return candidates.iloc[0:0].copy()
    cols = config["columns"]
    date_col, isin_col = cols["date"], cols["isin"]
    return_col = cols["stock_return"]
    score_cfg = config["universe_selection"]["selection_score"]
    names = candidates[isin_col].astype(str).drop_duplicates().tolist()

    pivot = stock_history[stock_history[isin_col].isin(names)].pivot_table(
        index=date_col, columns=isin_col, values=return_col
    )
    y = index_history.dropna()
    X = pivot.reindex(y.index).reindex(columns=names)
    if len(y) > 0:
        coverage = X.notna().sum(axis=0) / len(y)
        corr = X.corrwith(y, axis=0)
    else:
        coverage = pd.Series(0.0, index=names)
        corr = pd.Series(np.nan, index=names)

    metrics = candidates.drop_duplicates(isin_col).copy()
    metrics["return_coverage"] = metrics[isin_col].map(coverage).fillna(0.0)
    metrics["corr_to_index"] = metrics[isin_col].map(corr)
    metrics["corr_component"] = metrics["corr_to_index"].fillna(0.0).clip(lower=0.0)
    metrics["coverage_component"] = metrics["return_coverage"].clip(0.0, 1.0)
    market_cap_col = cols["market_cap"]
    metrics["market_cap_component"] = (
        _rank01(pd.to_numeric(metrics[market_cap_col], errors="coerce"))
        if market_cap_col in metrics else 0.0
    )
    metrics["constituent_component"] = metrics["is_actual_constituent"].astype(float)
    metrics["base_selection_score"] = (
        float(score_cfg.get("correlation_weight", 0.55)) * metrics["corr_component"]
        + float(score_cfg.get("coverage_weight", 0.20)) * metrics["coverage_component"]
        + float(score_cfg.get("market_cap_weight", 0.15)) * metrics["market_cap_component"]
        + float(score_cfg.get("constituent_weight", 0.10)) * metrics["constituent_component"]
    )
    min_cov = float(config["universe_selection"].get("minimum_return_coverage", 0.6))
    metrics.loc[metrics["return_coverage"] < min_cov, "base_selection_score"] -= 1.0

    corr_matrix = X.corr(min_periods=3).abs() if not X.empty else pd.DataFrame()
    remaining = metrics.set_index(isin_col, drop=False)
    chosen_rows: list[pd.Series] = []
    selected_names: list[str] = []
    penalty = float(score_cfg.get("redundancy_penalty", 0.20))
    for order in range(min(n_select, len(remaining))):
        if selected_names and not corr_matrix.empty:
            redundancy = pd.Series(0.0, index=remaining.index)
            common_selected = [x for x in selected_names if x in corr_matrix.columns]
            common_remaining = [x for x in remaining.index if x in corr_matrix.index]
            if common_selected and common_remaining:
                redundancy.loc[common_remaining] = corr_matrix.loc[common_remaining, common_selected].max(axis=1).fillna(0.0)
        else:
            redundancy = pd.Series(0.0, index=remaining.index)
        score = remaining["base_selection_score"] - penalty * redundancy
        chosen_name = score.sort_values(ascending=False).index[0]
        row = remaining.loc[chosen_name].copy()
        row["redundancy"] = float(redundancy.loc[chosen_name])
        row["selection_score"] = float(score.loc[chosen_name])
        row["selection_order_in_sector"] = order + 1
        chosen_rows.append(row)
        selected_names.append(str(chosen_name))
        remaining = remaining.drop(index=chosen_name)
        if remaining.empty:
            break
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def _optimize_tracking_weights(
    selected: pd.DataFrame,
    stock_history: pd.DataFrame,
    index_history: pd.Series,
    target_sector_weights: pd.Series,
    config: dict,
) -> tuple[np.ndarray, dict]:
    cols = config["columns"]
    date_col, isin_col, sector_col = cols["date"], cols["isin"], cols["sector"]
    n = len(selected)
    if n == 0:
        return np.array([]), {"optimizer_success": False, "optimizer_message": "no_selected_names"}

    sector_weights = target_sector_weights.reindex(selected[sector_col].unique()).fillna(0.0)
    sector_weights = sector_weights[sector_weights > 0]
    sector_weights = sector_weights / sector_weights.sum()
    x0 = np.zeros(n, dtype=float)
    for sector, target in sector_weights.items():
        mask = selected[sector_col].eq(sector).to_numpy()
        if mask.sum() > 0:
            x0[mask] = float(target) / mask.sum()

    cfg = config["universe_selection"]["weight_optimization"]
    if not cfg.get("enabled", True):
        return x0, {"optimizer_success": True, "optimizer_message": "equal_within_sector"}

    pivot = stock_history.pivot_table(index=date_col, columns=isin_col, values=cols["stock_return"])
    y = index_history.dropna()
    X = pivot.reindex(y.index).reindex(columns=selected[isin_col].astype(str).tolist())
    valid_rows = y.notna() & X.notna().any(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    min_hist = int(config["universe_selection"]["minimum_history_periods"][config["data"]["frequency"]])
    if len(y) < min_hist:
        return x0, {"optimizer_success": True, "optimizer_message": "insufficient_history_equal_weights"}

    X = X.apply(lambda c: c.fillna(c.mean())).fillna(0.0).to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    ridge = float(cfg.get("ridge_penalty", 1e-4))

    def objective(w: np.ndarray) -> float:
        error = yv - X @ w
        return float(np.mean(error * error) + ridge * np.dot(w, w))

    constraints = []
    for sector, target in sector_weights.items():
        mask = selected[sector_col].eq(sector).to_numpy(dtype=float)
        constraints.append({"type": "eq", "fun": lambda w, m=mask, t=float(target): float(np.dot(m, w) - t)})

    min_w_cfg = float(cfg.get("min_stock_weight", 0.0))
    max_w_cfg = float(cfg.get("max_stock_weight", 1.0))
    bounds: list[tuple[float, float]] = []
    for sector in selected[sector_col]:
        target = float(sector_weights.get(sector, 0.0))
        count = int(selected[sector_col].eq(sector).sum())
        equal = target / max(count, 1)
        lower = min(min_w_cfg, equal * 0.25)
        upper = max(max_w_cfg, equal * 1.25)
        upper = min(1.0, upper)
        bounds.append((lower, upper))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": int(cfg.get("solver_maxiter", 500)),
            "ftol": float(cfg.get("solver_ftol", 1e-10)),
            "disp": False,
        },
    )
    w = result.x if result.success and np.all(np.isfinite(result.x)) else x0
    error = yv - X @ w
    pred = X @ w
    corr = float(np.corrcoef(pred, yv)[0, 1]) if np.std(pred) > 0 and np.std(yv) > 0 else np.nan
    return w, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "tracking_rmse": float(np.sqrt(np.mean(error * error))),
        "tracking_correlation": corr,
        "history_observations": int(len(yv)),
    }


def _prepare_candidates(
    current: pd.DataFrame,
    constituents_index: pd.DataFrame,
    index_name: str,
    selection_key: str,
    factors: Iterable[str],
    config: dict,
) -> pd.DataFrame:
    cols = config["columns"]
    isin_col, sector_col = cols["isin"], cols["sector"]
    mode = config["universe_selection"].get("candidate_mode", "constituent_then_country_fallback")
    current = current.copy()
    current["_selection_key"] = _selection_key_series(current, config)
    current["is_actual_constituent"] = current[isin_col].isin(constituents_index[isin_col])
    factor_cols = [f for f in factors if f in current.columns]
    if factor_cols:
        current["factor_coverage"] = current[factor_cols].notna().mean(axis=1)
    else:
        current["factor_coverage"] = 1.0
    current = current[
        current[sector_col].notna()
        & (current["factor_coverage"] >= float(config["universe_selection"].get("minimum_factor_coverage", 0.5)))
    ]
    country_pool = current[current["_selection_key"].astype(str) == str(selection_key)]
    constituent_pool = current[current["is_actual_constituent"]]
    if mode == "constituent_only":
        pool = constituent_pool
    elif mode == "country_universe":
        pool = country_pool
    elif mode == "constituent_then_country_fallback":
        # Use the full country pool while rewarding actual constituent overlap.
        pool = country_pool if not country_pool.empty else constituent_pool
    else:
        raise ValueError(f"Unknown candidate_mode: {mode}")
    return pool.drop_duplicates(isin_col).copy()


def build_representative_universe(
    stocks: pd.DataFrame,
    factors: list[str],
    constituents: pd.DataFrame,
    sector_weights: pd.DataFrame,
    futures_returns: pd.DataFrame,
    config: dict,
) -> SelectionResult:
    """Build rolling representative universes with sector quotas and tracking weights."""
    cols = config["columns"]
    date_col, isin_col, sector_col = cols["date"], cols["isin"], cols["sector"]
    if not config["universe_selection"].get("enabled", True):
        return SelectionResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    dates = sorted(pd.to_datetime(stocks[date_col].dropna().unique()))
    index_names = [c for c in futures_returns.columns if c != date_col]
    futures = futures_returns.set_index(date_col).sort_index()
    frequency = config["data"]["frequency"]
    lookback = int(config["universe_selection"]["lookback_periods"][frequency])
    rebalance_every = max(1, int(config["universe_selection"].get("rebalance_every_periods", 1)))
    exclude_current = bool(config["universe_selection"].get("exclude_current_period_from_selection", True))

    history_rows: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    allocation_rows: list[dict] = []
    last_by_index: dict[str, pd.DataFrame] = {}
    last_quality_by_index: dict[str, dict] = {}

    for date_no, date in enumerate(dates):
        current = stocks[stocks[date_col] == date].copy()
        hist_end_mask = stocks[date_col] < date if exclude_current else stocks[date_col] <= date
        stock_history_all = stocks.loc[hist_end_mask].sort_values(date_col)
        history_dates = sorted(stock_history_all[date_col].dropna().unique())[-lookback:]
        stock_history_all = stock_history_all[stock_history_all[date_col].isin(history_dates)]

        for index_name in index_names:
            should_rebalance = index_name not in last_by_index or date_no % rebalance_every == 0
            if not should_rebalance:
                carried = last_by_index[index_name].copy()
                carried[date_col] = date
                carried["rebalanced"] = False
                history_rows.append(carried)
                previous_q = last_quality_by_index.get(index_name, {}).copy()
                previous_q.update({
                    date_col: date,
                    "index_name": index_name,
                    "rebalanced": False,
                    "optimizer_message": "carried_from_previous_rebalance",
                })
                quality_rows.append(previous_q)
                continue

            selection_key = _index_selection_key(index_name, config)
            target_count = _target_count(index_name, selection_key, config)
            const_idx = constituents[constituents["index_name"] == index_name].copy()
            candidates = _prepare_candidates(current, const_idx, index_name, selection_key, factors, config)
            sw = sector_weights[sector_weights["index_name"] == index_name].set_index(sector_col)["sector_weight"]
            available_counts = candidates.groupby(sector_col)[isin_col].nunique()
            allocation = allocate_sector_counts(target_count, sw, available_counts)

            hist_index = futures[index_name].dropna()
            hist_index = hist_index[hist_index.index < date] if exclude_current else hist_index[hist_index.index <= date]
            hist_index = hist_index.tail(lookback)
            stock_history = stock_history_all[stock_history_all[isin_col].isin(candidates[isin_col])]

            selected_parts: list[pd.DataFrame] = []
            for sector, count in allocation.items():
                sector_candidates = candidates[candidates[sector_col].astype(str) == str(sector)]
                chosen = _select_sector_names(sector_candidates, count, stock_history, hist_index, config)
                if not chosen.empty:
                    selected_parts.append(chosen)
                allocation_rows.append({
                    date_col: date,
                    "index_name": index_name,
                    "selection_key": selection_key,
                    sector_col: sector,
                    "target_sector_weight": float(sw.get(sector, 0.0)),
                    "available_count": int(available_counts.get(sector, 0)),
                    "target_selected_count": int(count),
                    "actual_selected_count": int(len(chosen)),
                })

            selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.iloc[0:0].copy()
            covered_sectors = selected[sector_col].dropna().unique()
            original_covered_weights = sw.reindex(covered_sectors).fillna(0.0)
            sector_weight_coverage = float(original_covered_weights.sum())
            normalized_sw = original_covered_weights / original_covered_weights.sum() if original_covered_weights.sum() > 0 else original_covered_weights
            weights, opt = _optimize_tracking_weights(selected, stock_history, hist_index, normalized_sw, config)

            if not selected.empty:
                selected = selected.copy()
                selected[date_col] = date
                selected["index_name"] = index_name
                selected["selection_key"] = selection_key
                selected["target_count"] = target_count
                selected["selection_weight"] = weights
                selected["target_sector_weight"] = selected[sector_col].map(normalized_sw).fillna(0.0)
                selected["original_sector_weight"] = selected[sector_col].map(sw).fillna(0.0)
                selected["rebalanced"] = True
                selected["optimizer_success"] = bool(opt.get("optimizer_success", False))
                selected["optimizer_message"] = str(opt.get("optimizer_message", ""))
                keep = [
                    date_col, "index_name", "selection_key", isin_col, sector_col,
                    "selection_weight", "target_sector_weight", "original_sector_weight",
                    "target_count", "is_actual_constituent", "factor_coverage",
                    "selection_score", "corr_to_index", "return_coverage",
                    "market_cap_component", "selection_order_in_sector",
                    "rebalanced", "optimizer_success", "optimizer_message",
                ]
                selected = selected[[c for c in keep if c in selected.columns]]
                last_by_index[index_name] = selected.copy()
                history_rows.append(selected)

            q_row = {
                date_col: date,
                "index_name": index_name,
                "rebalanced": True,
                "selection_key": selection_key,
                "target_count": target_count,
                "selected_count": int(len(selected)),
                "candidate_count": int(len(candidates)),
                "sector_weight_coverage": sector_weight_coverage,
                "actual_constituent_share": float(selected["is_actual_constituent"].mean()) if not selected.empty else np.nan,
                **opt,
            }
            quality_rows.append(q_row)
            last_quality_by_index[index_name] = q_row.copy()

    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    allocation_df = pd.DataFrame(allocation_rows)
    return SelectionResult(history=history, quality=quality, sector_allocation=allocation_df)
