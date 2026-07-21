from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regularization import PenalizedRidgeModel, coefficient_frame


def _fit_month_ridge(X: pd.DataFrame, y: pd.Series, alpha: float, penalties: np.ndarray) -> tuple[np.ndarray, float] | None:
    arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(arr).all(axis=1) & np.isfinite(target)
    arr, target = arr[mask], target[mask]
    if len(target) <= len(X.columns) + 2:
        return None
    xm, ym = arr.mean(axis=0), target.mean()
    xc, yc = arr - xm, target - ym
    coef = np.linalg.pinv(xc.T @ xc + np.diag(alpha * penalties)) @ (xc.T @ yc)
    intercept = float(ym - xm @ coef)
    return coef, intercept


def rolling_cross_sectional_coefficient_average(
    data: pd.DataFrame,
    X: pd.DataFrame,
    penalty_multipliers: np.ndarray,
    config: dict[str, Any],
    scope_labels: pd.Series,
    scope_name: str,
    target_col: str = "NextMonthReturn",
):
    from .layer3_pooled import Layer3Prediction

    c = config["columns"]
    cfg = config["layer3"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("lookback_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 18))
    alpha = float(cfg.get("ridge_alphas", [1.0])[0])
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    coef_frames = []
    model_rows = []

    for label in sorted(scope_labels.dropna().astype(str).unique()):
        label_mask = scope_labels.astype(str).eq(label)
        for pos, date in enumerate(dates):
            train_dates = dates[max(0, pos - window):pos]
            if len(train_dates) < min_train:
                continue
            monthly = []
            for train_date in train_dates:
                idx = data.index[label_mask & data[c["date"]].eq(train_date)]
                y = data.loc[idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y = y - y.mean()
                fitted = _fit_month_ridge(X.loc[idx], y, alpha, penalty_multipliers)
                if fitted is not None:
                    monthly.append(fitted)
            if len(monthly) < min_train:
                continue
            coef = np.mean([m[0] for m in monthly], axis=0)
            intercept = float(np.mean([m[1] for m in monthly]))
            model = PenalizedRidgeModel(list(X.columns), coef, intercept, alpha, penalty_multipliers)
            test_idx = data.index[label_mask & data[c["date"]].eq(date)]
            prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_frames.append(coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label))
            model_rows.append({"Date": date, "Scope": scope_name, "ScopeLabel": label, "Alpha": alpha, "TrainingPeriods": len(monthly), "FeatureCount": len(X.columns)})
    return Layer3Prediction(prediction, pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame(), pd.DataFrame(model_rows))
