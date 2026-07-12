from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_COLS = ["Rule", "Ridge", "RandomForest", "LightGBM", "Transformer"]
EVIDENCE_COLS = [
    "evidence_persistence", "evidence_correction", "evidence_volatility_support", "evidence_flow",
    "evidence_relative_strength", "evidence_intermarket", "evidence_macro_market",
]


def _rank_ic(group: pd.DataFrame, pred_col: str) -> float:
    x = group[pred_col].rank(pct=True)
    y = group["target_return"].rank(pct=True)
    return float(x.corr(y, method="pearson"))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.nanmax(x)
    e = np.exp(np.nan_to_num(z, nan=0.0))
    return e / e.sum() if e.sum() > 0 else np.ones_like(e) / len(e)


def _bounded_normalize(w: np.ndarray, lower: float, upper: float) -> np.ndarray:
    w = np.clip(w, lower, upper)
    # Iterative normalization with bounds.
    for _ in range(20):
        s = w.sum()
        if abs(s - 1) < 1e-9:
            break
        free = (w > lower + 1e-10) & (w < upper - 1e-10)
        if not free.any():
            w = w / w.sum()
            break
        w[free] += (1 - s) / free.sum()
        w = np.clip(w, lower, upper)
    return w / w.sum()


def dynamic_ensemble(pred: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pred.date.unique())
    prior_share = float(cfg["model"]["dynamic_weight_prior_share"])
    data_share = float(cfg["model"]["dynamic_weight_data_share"])
    window = int(cfg["model"]["reliability_window_months"])
    lower = float(cfg["model"]["model_weight_min"])
    upper = float(cfg["model"]["model_weight_max"])
    prior = np.repeat(1 / len(MODEL_COLS), len(MODEL_COLS))

    ic_rows = []
    for dt in dates:
        g = pred[pred.date == dt]
        row = {"date": dt}
        for m in MODEL_COLS:
            row[m] = _rank_ic(g, m)
        for e in EVIDENCE_COLS:
            row[e] = _rank_ic(g, e)
        ic_rows.append(row)
    ic = pd.DataFrame(ic_rows).sort_values("date").reset_index(drop=True)

    weight_rows = []
    enriched = []
    for idx, dt in enumerate(dates):
        hist = ic.iloc[max(0, idx - window):idx]
        if len(hist) < 4:
            w = prior.copy()
            metrics = {m: (np.nan, np.nan, np.nan, 0.5) for m in MODEL_COLS}
        else:
            quality = []
            metrics = {}
            for m in MODEL_COLS:
                s = hist[m].dropna()
                mean_ic = float(s.mean()) if len(s) else 0.0
                icir = float(mean_ic / (s.std(ddof=1) + 1e-6)) if len(s) > 1 else 0.0
                hit = float((s > 0).mean()) if len(s) else 0.5
                scaled_mean = np.clip((mean_ic + 0.25) / 0.50, 0, 1)
                scaled_icir = np.clip((icir + 1.0) / 2.0, 0, 1)
                q = 0.50 * scaled_mean + 0.30 * scaled_icir + 0.20 * hit
                quality.append(q)
                metrics[m] = (mean_ic, icir, hit, q)
            data_w = _softmax(np.asarray(quality))
            w = prior_share * prior + data_share * data_w
            w = _bounded_normalize(w, lower, upper)
        wr = {"date": dt}
        for j, m in enumerate(MODEL_COLS):
            wr[m] = w[j]
            wr[f"{m}_trailing_ic"] = metrics[m][0]
            wr[f"{m}_icir"] = metrics[m][1]
            wr[f"{m}_hit"] = metrics[m][2]
            wr[f"{m}_quality"] = metrics[m][3]
        weight_rows.append(wr)
        g = pred[pred.date == dt].copy()
        g["static_score"] = g[MODEL_COLS].mean(axis=1)
        g["dynamic_score"] = sum(w[j] * g[m] for j, m in enumerate(MODEL_COLS))
        enriched.append(g)
    enriched_df = pd.concat(enriched, ignore_index=True)
    weights = pd.DataFrame(weight_rows)

    # Evidence reliability weights: prior + 25% trailing IC adjustment, primarily for Rule interpretation.
    eprior = np.array([0.25, 0.10, 0.15, 0.15, 0.20, 0.10, 0.05])
    evidence_weight_rows = []
    for idx, dt in enumerate(dates):
        hist = ic.iloc[max(0, idx - window):idx]
        if len(hist) < 4:
            ew = eprior.copy()
        else:
            q = []
            for e in EVIDENCE_COLS:
                s = hist[e].dropna()
                mean_ic = float(s.mean()) if len(s) else 0.0
                q.append(np.clip((mean_ic + 0.25) / 0.50, 0, 1))
            data_w = _softmax(np.asarray(q))
            ew = 0.75 * eprior + 0.25 * data_w
            ew = ew / ew.sum()
        r = {"date": dt}
        r.update({e: ew[j] for j, e in enumerate(EVIDENCE_COLS)})
        evidence_weight_rows.append(r)
    evidence_weights = pd.DataFrame(evidence_weight_rows)
    return enriched_df, weights, ic, evidence_weights
