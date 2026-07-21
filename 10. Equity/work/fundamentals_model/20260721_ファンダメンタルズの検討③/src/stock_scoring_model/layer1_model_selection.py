from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .layer1_single_factor import fit_single_factor


COMPLEXITY = {"linear": 1, "piecewise": 2, "quadratic": 2}


def monthly_rank_ic(dates: pd.Series, prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"Date": dates.to_numpy(), "Prediction": prediction, "Target": target})
    values: list[float] = []
    for _, g in tmp.groupby("Date"):
        g = g.dropna()
        if len(g) >= 8:
            values.append(float(spearmanr(g["Prediction"], g["Target"]).statistic))
    arr = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan
    return float(arr.mean()), se, len(arr)


def select_candidate_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_dates: pd.Series,
    config: dict[str, Any],
) -> tuple[str, pd.DataFrame]:
    cfg = config["layer1"]
    rows = []
    for name in cfg.get("candidate_models", ["linear", "piecewise", "quadratic"]):
        model = fit_single_factor(
            x_fit,
            y_fit,
            name,
            knot=float(cfg.get("piecewise_knot", 0.0)),
            ridge_alpha=float(cfg.get("ridge_alpha", 1e-8)),
        )
        pred = model.predict(x_valid)
        mean_ic, se, n = monthly_rank_ic(valid_dates, pred, y_valid)
        rows.append({"CandidateModel": name, "MeanRankIC": mean_ic, "RankICSE": se, "EvaluationPeriods": n})
    result = pd.DataFrame(rows)
    valid = result[result["MeanRankIC"].notna()].copy()
    if valid.empty:
        return "linear", result
    best_row = valid.loc[valid["MeanRankIC"].idxmax()]
    threshold = float(best_row["MeanRankIC"])
    if bool(cfg.get("one_se_rule", True)) and np.isfinite(best_row["RankICSE"]):
        threshold -= float(best_row["RankICSE"])
    eligible = valid[valid["MeanRankIC"] >= threshold].copy()
    eligible["Complexity"] = eligible["CandidateModel"].map(COMPLEXITY)
    selected = eligible.sort_values(["Complexity", "MeanRankIC"], ascending=[True, False]).iloc[0]
    result["BestRawModel"] = str(best_row["CandidateModel"])
    result["OneSEThreshold"] = threshold
    result["SelectedModel"] = str(selected["CandidateModel"])
    return str(selected["CandidateModel"]), result
