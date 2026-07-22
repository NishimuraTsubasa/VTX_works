from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .master import FactorMeta


@dataclass
class Layer2AggregationResult:
    factor_scores: pd.DataFrame
    weight_history: pd.DataFrame
    factor_return_history: pd.DataFrame
    correlation_history: pd.DataFrame


def _assign_quantile(score: pd.Series, q: int) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce")
    valid = x.notna()
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    if int(valid.sum()) < q:
        return out
    ranked = x.loc[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranked, q=q, labels=range(1, q + 1)).astype(int)
    return out


def _cap_long_only_weights(weights: np.ndarray, maximum: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    if n == 0:
        return w
    w = np.clip(w, 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w /= w.sum()
    maximum = max(float(maximum), 1.0 / n)
    for _ in range(n + 2):
        over = w > maximum + 1e-12
        if not over.any():
            break
        excess = float((w[over] - maximum).sum())
        w[over] = maximum
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def _correlation_minimum_variance_weights(
    correlation: pd.DataFrame,
    shrinkage: float,
    maximum_weight: float,
) -> np.ndarray:
    n = len(correlation)
    if n <= 1:
        return np.ones(n, dtype=float)
    matrix = correlation.to_numpy(float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(n)
    raw = np.linalg.pinv(matrix) @ np.ones(n, dtype=float)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones(n, dtype=float)
    return _cap_long_only_weights(raw / raw.sum(), maximum_weight)


def _groups(metas: dict[str, FactorMeta], available: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in available:
            result.setdefault(meta.group, []).append(code)
    return result


def build_factor_return_history(
    data: pd.DataFrame,
    raw_scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
) -> pd.DataFrame:
    """各FAのQ5-Q1リターンを作る。

    Dateはスコア形成日。Date=tのFactorReturnはt+1リターンから計算されるため、
    Date=tのウェイト計算では必ずDate<tだけを参照する。
    """
    c = config["columns"]
    cfg = config["layer2"]
    qn = int(cfg.get("factor_return_quantiles", 5))
    scope = str(cfg.get("factor_return_scope", "within_country"))
    minimum = int(cfg.get("minimum_stocks_per_factor_return_cell", 20))
    country_aggregation = str(cfg.get("country_factor_return_aggregation", "equal_country"))
    codes = [code for code in metas if code in raw_scores.columns]
    rows: list[dict[str, object]] = []

    work = data[[c["date"], c["country"], c["sector"], target_col]].copy()
    work[c["date"]] = pd.to_datetime(work[c["date"]])

    for date, date_idx in work.groupby(c["date"]).groups.items():
        for code in codes:
            temp = work.loc[date_idx].copy()
            temp["Score"] = pd.to_numeric(raw_scores.loc[date_idx, code], errors="coerce")
            temp = temp.dropna(subset=["Score", target_col])
            cell_rows: list[dict[str, object]] = []
            if scope == "global":
                cell_groups = [("GLOBAL", temp)]
            elif scope == "within_country_sector":
                cell_groups = list(temp.groupby([c["country"], c["sector"]], dropna=False))
            else:
                cell_groups = list(temp.groupby(c["country"], dropna=False))

            for cell, frame in cell_groups:
                if len(frame) < max(minimum, qn * 2):
                    continue
                quantile = _assign_quantile(frame["Score"], qn)
                long_return = frame.loc[quantile.eq(qn), target_col].mean()
                short_return = frame.loc[quantile.eq(1), target_col].mean()
                long_count = int(quantile.eq(qn).sum())
                short_count = int(quantile.eq(1).sum())
                if not np.isfinite(long_return) or not np.isfinite(short_return):
                    continue
                cell_rows.append({
                    "Cell": str(cell),
                    "FactorReturn": float(long_return - short_return),
                    "LongReturn": float(long_return),
                    "ShortReturn": float(short_return),
                    "LongCount": long_count,
                    "ShortCount": short_count,
                    "Weight": long_count + short_count,
                })

            if not cell_rows:
                continue
            cell_df = pd.DataFrame(cell_rows)
            if country_aggregation == "stock_count_weighted":
                weights = cell_df["Weight"].to_numpy(float)
            else:
                weights = np.ones(len(cell_df), dtype=float)
            rows.append({
                "Date": pd.Timestamp(date),
                "FactorCode": code,
                "FactorGroup": metas[code].group,
                "FactorReturn": float(np.average(cell_df["FactorReturn"], weights=weights)),
                "LongReturn": float(np.average(cell_df["LongReturn"], weights=weights)),
                "ShortReturn": float(np.average(cell_df["ShortReturn"], weights=weights)),
                "CellCount": int(len(cell_df)),
                "LongCount": int(cell_df["LongCount"].sum()),
                "ShortCount": int(cell_df["ShortCount"].sum()),
                "Scope": scope,
            })
    return pd.DataFrame(rows).sort_values(["FactorGroup", "FactorCode", "Date"]).reset_index(drop=True) if rows else pd.DataFrame()


def rolling_factor_return_correlation_weights(
    dates: list[pd.Timestamp],
    factor_return_history: pd.DataFrame,
    groups: dict[str, list[str]],
    config: dict[str, Any],
    *,
    equal_weight_blend: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["layer2"]
    lookback = int(cfg.get("factor_return_lookback_periods", 36))
    minimum = int(cfg.get("factor_return_minimum_periods", 12))
    shrinkage = float(cfg.get("correlation_shrinkage", 0.30))
    maximum = float(cfg.get("maximum_factor_weight", 0.50))
    smoothing = float(cfg.get("weight_smoothing", 0.50))
    if equal_weight_blend is None:
        equal_weight_blend = float(cfg.get("equal_weight_blend", 0.50))
    equal_weight_blend = float(np.clip(equal_weight_blend, 0.0, 1.0))
    use_filter = bool(cfg.get("effectiveness_filter_enabled", False))
    min_mean_return = float(cfg.get("minimum_mean_factor_return", 0.0))
    min_positive_rate = float(cfg.get("minimum_factor_return_positive_rate", 0.45))

    weight_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    previous_by_group: dict[str, np.ndarray] = {}

    for group, codes in groups.items():
        n = len(codes)
        if n == 0:
            continue
        ew = np.ones(n, dtype=float) / n
        previous = ew.copy()
        for date in dates:
            history_dates = [d for d in dates if d < date][-lookback:]
            hist = factor_return_history[
                factor_return_history["Date"].isin(history_dates)
                & factor_return_history["FactorCode"].isin(codes)
            ]
            wide = hist.pivot_table(index="Date", columns="FactorCode", values="FactorReturn", aggfunc="mean").reindex(columns=codes)
            counts = wide.count().reindex(codes).fillna(0)
            means = wide.mean().reindex(codes)
            positive = wide.gt(0).sum().div(counts.replace(0, np.nan)).reindex(codes)
            eligible = counts.ge(minimum)
            if use_filter:
                eligible &= means.ge(min_mean_return) & positive.ge(min_positive_rate)
            eligible_codes = [code for code in codes if bool(eligible.get(code, False))]

            if len(eligible_codes) == 0:
                raw = ew.copy()
                reason = "fallback_equal_weight_insufficient_history"
                correlation = pd.DataFrame(np.eye(n), index=codes, columns=codes)
            elif len(eligible_codes) == 1:
                raw = np.zeros(n, dtype=float)
                raw[codes.index(eligible_codes[0])] = 1.0
                raw = _cap_long_only_weights(raw, maximum)
                reason = "single_eligible_factor"
                correlation = wide.corr(min_periods=minimum).reindex(index=codes, columns=codes)
            else:
                eligible_wide = wide[eligible_codes]
                eligible_corr = eligible_wide.corr(min_periods=minimum).fillna(0.0)
                np.fill_diagonal(eligible_corr.values, 1.0)
                eligible_weights = _correlation_minimum_variance_weights(eligible_corr, shrinkage, maximum)
                raw = np.zeros(n, dtype=float)
                for code, value in zip(eligible_codes, eligible_weights):
                    raw[codes.index(code)] = value
                correlation = wide.corr(min_periods=minimum).reindex(index=codes, columns=codes)
                reason = "factor_return_correlation_min_variance"

            blended = equal_weight_blend * ew + (1.0 - equal_weight_blend) * raw
            blended = _cap_long_only_weights(blended, maximum)
            final = smoothing * previous + (1.0 - smoothing) * blended
            final = _cap_long_only_weights(final, maximum)
            previous = final
            previous_by_group[group] = final

            for i, code in enumerate(codes):
                weight_rows.append({
                    "Date": pd.Timestamp(date),
                    "FactorGroup": group,
                    "FactorCode": code,
                    "Weight": float(final[i]),
                    "RawCorrelationWeight": float(raw[i]),
                    "EqualWeight": float(ew[i]),
                    "MeanFactorReturn": float(means.get(code, np.nan)),
                    "FactorReturnPositiveRate": float(positive.get(code, np.nan)),
                    "HistoryPeriods": int(counts.get(code, 0)),
                    "Eligible": bool(eligible.get(code, False)),
                    "EqualWeightBlend": equal_weight_blend,
                    "Reason": reason,
                })
            for code1 in codes:
                for code2 in codes:
                    corr_rows.append({
                        "Date": pd.Timestamp(date),
                        "FactorGroup": group,
                        "FactorCode1": code1,
                        "FactorCode2": code2,
                        "FactorReturnCorrelation": float(correlation.loc[code1, code2]) if code1 in correlation.index and code2 in correlation.columns and np.isfinite(correlation.loc[code1, code2]) else np.nan,
                        "HistoryPeriods": int(min(counts.get(code1, 0), counts.get(code2, 0))),
                    })
    return pd.DataFrame(weight_rows), pd.DataFrame(corr_rows)


def aggregate_with_weight_history(
    data: pd.DataFrame,
    raw_scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    weight_history: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    c = config["columns"]
    groups = _groups(metas, set(raw_scores.columns))
    result = pd.DataFrame(index=data.index)
    for group, codes in groups.items():
        values = pd.Series(np.nan, index=data.index, dtype=float)
        for date, idx in data.groupby(c["date"]).groups.items():
            wh = weight_history[
                weight_history["Date"].eq(pd.Timestamp(date))
                & weight_history["FactorGroup"].eq(group)
            ].set_index("FactorCode")
            if wh.empty:
                weights = np.ones(len(codes), dtype=float) / len(codes)
            else:
                weights = wh["Weight"].reindex(codes).fillna(0.0).to_numpy(float)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(codes), dtype=float) / len(codes)
            arr = raw_scores.loc[idx, codes].to_numpy(float)
            valid = np.isfinite(arr)
            numerator = np.nansum(arr * weights, axis=1)
            denominator = np.sum(valid * weights, axis=1)
            values.loc[idx] = np.divide(numerator, denominator, out=np.full(len(idx), np.nan), where=denominator > 0)
        result[group] = values
    return result


def aggregate_raw_factor_scores(
    data: pd.DataFrame,
    raw_scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
    *,
    equal_weight_blend: float | None = None,
) -> Layer2AggregationResult:
    groups = _groups(metas, set(raw_scores.columns))
    factor_returns = build_factor_return_history(data, raw_scores, metas, config)
    dates = sorted(pd.to_datetime(data[config["columns"]["date"]].dropna().unique()))
    weights, correlations = rolling_factor_return_correlation_weights(
        dates,
        factor_returns,
        groups,
        config,
        equal_weight_blend=equal_weight_blend,
    )
    factor_scores = aggregate_with_weight_history(data, raw_scores, metas, weights, config)
    return Layer2AggregationResult(factor_scores, weights, factor_returns, correlations)
