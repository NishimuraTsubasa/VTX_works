from __future__ import annotations

import numpy as np
import pandas as pd


def _make_weights(g: pd.DataFrame, score_col: str, max_abs: float, gross: float) -> pd.Series:
    score = g[score_col].astype(float)
    centered = score - score.mean()
    risk = g.rv20.astype(float).replace(0, np.nan).fillna(g.rv20.median())
    raw = centered / risk.clip(lower=0.05)
    if raw.abs().sum() == 0:
        return pd.Series(0.0, index=g.index)
    w = raw / raw.abs().sum() * gross
    # Cap and re-normalize while retaining dollar neutrality approximately.
    w = w.clip(-max_abs, max_abs)
    w = w - w.mean()
    if w.abs().sum() > 0:
        w = w / w.abs().sum() * gross
    return w


def build_portfolios(pred: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_abs = float(cfg["portfolio"]["max_abs_weight"])
    gross = float(cfg["portfolio"]["gross_exposure"])
    cost_rate = float(cfg["portfolio"]["transaction_cost_bps"]) / 10000.0
    rows, latest_weights = [], []
    prev_dyn = prev_static = None
    for dt, g0 in pred.groupby("date", sort=True):
        g = g0.copy()
        g["dynamic_weight"] = _make_weights(g, "dynamic_score", max_abs, gross)
        g["static_weight"] = _make_weights(g, "static_score", max_abs, gross)
        dyn_turnover = 0.5 * (g.set_index("asset_id").dynamic_weight - (prev_dyn if prev_dyn is not None else 0)).abs().sum()
        sta_turnover = 0.5 * (g.set_index("asset_id").static_weight - (prev_static if prev_static is not None else 0)).abs().sum()
        dyn_gross = float((g.dynamic_weight * g.target_return).sum())
        sta_gross = float((g.static_weight * g.target_return).sum())
        benchmark = float(g.target_return.mean())
        rows.append({
            "date": dt, "dynamic_return_gross": dyn_gross, "dynamic_turnover": dyn_turnover,
            "dynamic_cost": dyn_turnover * cost_rate, "dynamic_return_net": dyn_gross - dyn_turnover * cost_rate,
            "static_return_gross": sta_gross, "static_turnover": sta_turnover,
            "static_cost": sta_turnover * cost_rate, "static_return_net": sta_gross - sta_turnover * cost_rate,
            "benchmark_return": benchmark,
        })
        prev_dyn = g.set_index("asset_id").dynamic_weight
        prev_static = g.set_index("asset_id").static_weight
        if dt == pred.date.max():
            latest_weights = g[["date", "asset_id", "dynamic_score", "static_score", "dynamic_weight", "static_weight", "target_return"]].copy()
    bt = pd.DataFrame(rows).sort_values("date")
    for col in ["dynamic_return_net", "static_return_net", "benchmark_return"]:
        bt[f"cum_{col}"] = (1 + bt[col]).cumprod()
    bt["dynamic_drawdown"] = bt.cum_dynamic_return_net / bt.cum_dynamic_return_net.cummax() - 1
    bt["static_drawdown"] = bt.cum_static_return_net / bt.cum_static_return_net.cummax() - 1
    return bt, latest_weights


def performance_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [("Dynamic Ensemble", "dynamic_return_net"), ("Static Ensemble", "static_return_net"), ("Equal Weight Benchmark", "benchmark_return")]:
        r = bt[col].dropna()
        ann_ret = (1 + r).prod() ** (12 / max(len(r), 1)) - 1
        ann_vol = r.std(ddof=1) * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        curve = (1 + r).cumprod()
        mdd = float((curve / curve.cummax() - 1).min())
        hit = float((r > 0).mean())
        rows.append({"strategy": name, "annual_return": ann_ret, "annual_volatility": ann_vol, "sharpe": sharpe, "max_drawdown": mdd, "positive_month_ratio": hit})
    return pd.DataFrame(rows)
