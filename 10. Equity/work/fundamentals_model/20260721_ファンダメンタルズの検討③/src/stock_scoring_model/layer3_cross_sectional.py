from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regularization import PenalizedRidgeModel, coefficient_frame


def _fit_month_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: str,
    alpha: float,
    penalties: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(arr).all(axis=1) & np.isfinite(target)
    arr, target = arr[mask], target[mask]
    if len(target) <= len(X.columns) + 2:
        return None
    xm, ym = arr.mean(axis=0), target.mean()
    xc, yc = arr - xm, target - ym
    if estimator == "ols":
        coef, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    else:
        lhs = xc.T @ xc + np.diag(alpha * penalties)
        rhs = xc.T @ yc
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(lhs) @ rhs
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
    eligible_rows: pd.Series | None = None,
):
    from .layer3_pooled import Layer3Prediction

    c = config["columns"]
    cfg = config["layer3"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("lookback_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 12))
    estimator = str(cfg.get("estimator", "ridge")).lower()
    alpha = 0.0 if estimator == "ols" else float(cfg.get("ridge_alphas", [1.0])[0])
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    coef_frames = []
    model_rows = []
    eligible = pd.Series(True, index=data.index) if eligible_rows is None else eligible_rows.reindex(data.index).fillna(False)

    for label in sorted(scope_labels.dropna().astype(str).unique()):
        label_mask = scope_labels.astype(str).eq(label)
        for pos, date in enumerate(dates):
            candidate_train_dates = dates[max(0, pos - window):pos]
            train_dates = [d for d in candidate_train_dates if bool((label_mask & data[c["date"]].eq(d) & eligible).any())]
            if len(train_dates) < min_train:
                continue
            monthly = []
            for train_date in train_dates:
                idx = data.index[label_mask & data[c["date"]].eq(train_date) & eligible]
                y = data.loc[idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y = y - y.mean()
                fitted = _fit_month_model(X.loc[idx], y, estimator, alpha, penalty_multipliers)
                if fitted is not None:
                    monthly.append(fitted)
            if len(monthly) < min_train:
                continue
            coef = np.mean([m[0] for m in monthly], axis=0)
            intercept = float(np.mean([m[1] for m in monthly]))
            model = PenalizedRidgeModel(
                list(X.columns), coef, intercept, alpha,
                np.zeros_like(penalty_multipliers) if estimator == "ols" else penalty_multipliers,
                estimator_name=estimator,
            )
            test_idx = data.index[label_mask & data[c["date"]].eq(date) & eligible]
            if len(test_idx):
                prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_frames.append(coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label))
            model_rows.append({
                "Date": date,
                "Scope": scope_name,
                "ScopeLabel": label,
                "Estimator": estimator,
                "Alpha": alpha,
                "TrainingPeriods": len(monthly),
                "ValidationPeriods": 0,
                "FeatureCount": len(X.columns),
            })
    return Layer3Prediction(
        prediction=prediction,
        coefficient_history=pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame(),
        model_history=pd.DataFrame(model_rows),
    )
