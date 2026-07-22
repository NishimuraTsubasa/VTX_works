from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def regression_metrics(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    feature_count: int | None = None,
) -> dict[str, float | int]:
    """Return robust regression diagnostics for finite paired observations.

    R-squared is the usual 1-SSE/SST measure and may be negative for OOS data.
    """
    y = np.asarray(pd.to_numeric(pd.Series(actual), errors="coerce"), dtype=float)
    p = np.asarray(pd.to_numeric(pd.Series(prediction), errors="coerce"), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = int(len(y))
    if n == 0:
        return {
            "ObservationCount": 0,
            "R2": np.nan,
            "AdjustedR2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "PredictionMean": np.nan,
            "PredictionStd": np.nan,
            "TargetMean": np.nan,
            "TargetStd": np.nan,
            "PredictionTargetStdRatio": np.nan,
        }
    residual = y - p
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    k = int(feature_count or 0)
    adjusted = (
        1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        if np.isfinite(r2) and n > k + 1
        else np.nan
    )
    pearson = (
        float(pearsonr(p, y).statistic)
        if n >= 3 and np.std(p) > 0 and np.std(y) > 0
        else np.nan
    )
    spearman = (
        float(spearmanr(p, y).statistic)
        if n >= 3 and len(np.unique(p)) > 1 and len(np.unique(y)) > 1
        else np.nan
    )
    p_std = float(np.std(p, ddof=1)) if n > 1 else 0.0
    y_std = float(np.std(y, ddof=1)) if n > 1 else 0.0
    return {
        "ObservationCount": n,
        "R2": r2,
        "AdjustedR2": adjusted,
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Pearson": pearson,
        "Spearman": spearman,
        "PredictionMean": float(np.mean(p)),
        "PredictionStd": p_std,
        "TargetMean": float(np.mean(y)),
        "TargetStd": y_std,
        "PredictionTargetStdRatio": p_std / y_std if y_std > 0 else np.nan,
    }


def prefixed_metrics(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    prefix: str,
    feature_count: int | None = None,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in regression_metrics(actual, prediction, feature_count).items()
    }
