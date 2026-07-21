from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rolling_ic_weights(
    data: pd.DataFrame,
    subscores: pd.DataFrame,
    factor_codes: list[str],
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = config["columns"]
    cfg = config["layer2"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    lookback = int(cfg.get("ic_lookback_periods", 36))
    minp = int(cfg.get("ic_minimum_periods", 12))
    shrink = float(cfg.get("correlation_shrinkage", 0.2))
    max_weight = float(cfg.get("maximum_factor_weight", 0.6))
    smoothing = float(cfg.get("weight_smoothing", 0.5))

    ic_rows: list[dict[str, object]] = []
    for date, idx in data.groupby(c["date"]).groups.items():
        y = data.loc[idx, target_col]
        for code in factor_codes:
            x = subscores.loc[idx, code]
            mask = x.notna() & y.notna()
            ic = spearmanr(x[mask], y[mask]).statistic if mask.sum() >= 8 else np.nan
            ic_rows.append({"Date": date, "FactorCode": code, "RankIC": ic})
    ic_frame = pd.DataFrame(ic_rows)

    weight_rows: list[dict[str, object]] = []
    weight_by_date = pd.DataFrame(index=pd.Index(dates, name="Date"), columns=factor_codes, dtype=float)
    previous = np.ones(len(factor_codes), dtype=float) / max(len(factor_codes), 1)
    for date in dates:
        past = [d for d in dates if d < date][-lookback:]
        hist = ic_frame[ic_frame["Date"].isin(past)]
        mu = hist.groupby("FactorCode")["RankIC"].mean().reindex(factor_codes)
        count = hist.groupby("FactorCode")["RankIC"].count().reindex(factor_codes).fillna(0)
        if (count < minp).any() or mu.fillna(0).clip(lower=0).sum() <= 0:
            raw = np.ones(len(factor_codes), dtype=float) / len(factor_codes)
            reason = "fallback_equal_weight"
        else:
            current_idx = data.index[data[c["date"]].eq(date)]
            corr = subscores.loc[current_idx, factor_codes].corr().fillna(0).to_numpy(float)
            corr = (1 - shrink) * corr + shrink * np.eye(len(factor_codes))
            signal = mu.clip(lower=0).fillna(0).to_numpy(float)
            raw = np.clip(np.linalg.pinv(corr) @ signal, 0, None)
            raw = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
            raw = np.minimum(raw, max_weight)
            raw = raw / raw.sum()
            reason = "correlation_adjusted_ic"
        w = smoothing * previous + (1 - smoothing) * raw
        w = w / w.sum()
        previous = w
        weight_by_date.loc[date] = w
        for code, value in zip(factor_codes, w):
            weight_rows.append({"Date": date, "FactorCode": code, "Weight": value, "Reason": reason})
    return weight_by_date, pd.DataFrame(weight_rows)
