from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression_metrics import prefixed_metrics
from .regularization import (
    PenalizedRidgeModel,
    coefficient_frame,
    fit_ols,
    fit_penalized_ridge,
)


def _fit_month_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: str,
    alpha: float,
    penalties: np.ndarray,
    standardize_mask: np.ndarray,
) -> PenalizedRidgeModel | None:
    arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(arr).all(axis=1) & np.isfinite(target)
    if int(mask.sum()) <= len(X.columns) + 2:
        return None
    X_fit = X.loc[X.index[mask]]
    y_fit = y.loc[y.index[mask]]
    if estimator == "ols":
        return fit_ols(X_fit, y_fit, standardize_mask=standardize_mask)
    return fit_penalized_ridge(
        X_fit,
        y_fit,
        alpha,
        penalties,
        standardize_mask=standardize_mask,
    )


def rolling_cross_sectional_coefficient_average(
    data: pd.DataFrame,
    X: pd.DataFrame,
    penalty_multipliers: np.ndarray,
    standardize_mask: np.ndarray,
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
    coef_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    eligible = pd.Series(True, index=data.index) if eligible_rows is None else eligible_rows.reindex(data.index).fillna(False)

    for label in sorted(scope_labels.dropna().astype(str).unique()):
        label_mask = scope_labels.astype(str).eq(label)
        for pos, date in enumerate(dates):
            candidate_train_dates = dates[max(0, pos - window):pos]
            train_dates = [
                d for d in candidate_train_dates
                if bool((label_mask & data[c["date"]].eq(d) & eligible).any())
            ]
            if len(train_dates) < min_train:
                continue

            monthly_models: list[PenalizedRidgeModel] = []
            for train_date in train_dates:
                idx = data.index[label_mask & data[c["date"]].eq(train_date) & eligible]
                y = data.loc[idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y = y - y.mean()
                fitted = _fit_month_model(
                    X.loc[idx],
                    y,
                    estimator,
                    alpha,
                    penalty_multipliers,
                    standardize_mask,
                )
                if fitted is not None:
                    monthly_models.append(fitted)
            if len(monthly_models) < min_train:
                continue

            # 月ごとに標準化尺度が異なるため、予測可能な元スケール係数へ戻して平均する。
            raw_coef = np.mean([m.raw_coef_ for m in monthly_models], axis=0)
            raw_intercept = float(np.mean([m.raw_intercept_ for m in monthly_models]))
            model = PenalizedRidgeModel(
                list(X.columns),
                raw_coef,
                raw_intercept,
                alpha,
                np.zeros_like(penalty_multipliers) if estimator == "ols" else penalty_multipliers,
                estimator_name=estimator,
                feature_means_=np.zeros(X.shape[1], dtype=float),
                feature_scales_=np.ones(X.shape[1], dtype=float),
                standardized_mask_=np.zeros(X.shape[1], dtype=bool),
            )
            test_idx = data.index[label_mask & data[c["date"]].eq(date) & eligible]
            if len(test_idx):
                prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_frame = coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label)
            coef_frame["CoefficientAveragingScale"] = "raw"
            coef_frames.append(coef_frame)
            train_idx_all = data.index[label_mask & data[c["date"]].isin(train_dates) & eligible]
            y_train_all = data.loc[train_idx_all, target_col]
            if cfg.get("demean_target_by_date", True):
                y_train_all = y_train_all - y_train_all.groupby(data.loc[train_idx_all, c["date"]]).transform("mean")
            train_mask_all = y_train_all.notna() & np.isfinite(X.loc[train_idx_all]).all(axis=1)
            train_metrics = prefixed_metrics(
                y_train_all.loc[train_idx_all[train_mask_all]],
                model.predict(X.loc[train_idx_all[train_mask_all]]),
                "Train",
                feature_count=len(X.columns),
            )
            model_row = {
                "Date": date,
                "Scope": scope_name,
                "ScopeLabel": label,
                "Estimator": estimator,
                "Alpha": alpha,
                "TrainingPeriods": len(monthly_models),
                "ValidationPeriods": 0,
                "FeatureCount": len(X.columns),
                "StandardizedFeatureCount": int(np.asarray(standardize_mask, dtype=bool).sum()),
                "CoefficientAveragingScale": "raw",
            }
            model_row.update(train_metrics)
            model_rows.append(model_row)
    return Layer3Prediction(
        prediction=prediction,
        coefficient_history=pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame(),
        model_history=pd.DataFrame(model_rows),
    )
