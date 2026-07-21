from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .regularization import coefficient_frame, fit_penalized_ridge_cv


@dataclass
class Layer3Prediction:
    prediction: pd.Series
    coefficient_history: pd.DataFrame
    model_history: pd.DataFrame


def _demean_by_date(y: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "Date": dates})
    return frame["y"] - frame.groupby("Date")["y"].transform("mean")


def rolling_pooled_prediction(
    data: pd.DataFrame,
    X: pd.DataFrame,
    penalty_multipliers: np.ndarray,
    config: dict[str, Any],
    scope_labels: pd.Series,
    scope_name: str,
    target_col: str = "NextMonthReturn",
) -> Layer3Prediction:
    c = config["columns"]
    cfg = config["layer3"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("lookback_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 18))
    min_obs = int(cfg.get("minimum_training_observations", 250))
    alphas = list(cfg.get("ridge_alphas", [0.1, 1, 10]))
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    coef_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []

    for label in sorted(scope_labels.dropna().astype(str).unique()):
        label_mask = scope_labels.astype(str).eq(label)
        for pos, date in enumerate(dates):
            train_dates = dates[max(0, pos - window):pos]
            if len(train_dates) < min_train:
                continue
            valid_n = max(3, min(6, len(train_dates) // 4))
            fit_dates, valid_dates = train_dates[:-valid_n], train_dates[-valid_n:]
            fit_idx = data.index[label_mask & data[c["date"]].isin(fit_dates)]
            valid_idx = data.index[label_mask & data[c["date"]].isin(valid_dates)]
            test_idx = data.index[label_mask & data[c["date"]].eq(date)]
            if len(test_idx) == 0:
                continue
            y_fit = data.loc[fit_idx, target_col]
            y_valid = data.loc[valid_idx, target_col]
            if cfg.get("demean_target_by_date", True):
                y_fit = _demean_by_date(y_fit, data.loc[fit_idx, c["date"]])
                y_valid = _demean_by_date(y_valid, data.loc[valid_idx, c["date"]])
            fit_mask = y_fit.notna() & np.isfinite(X.loc[fit_idx]).all(axis=1)
            valid_mask = y_valid.notna() & np.isfinite(X.loc[valid_idx]).all(axis=1)
            if int(fit_mask.sum() + valid_mask.sum()) < min_obs or fit_mask.sum() < max(50, len(X.columns) + 5):
                continue
            model = fit_penalized_ridge_cv(
                X.loc[fit_idx[fit_mask]],
                y_fit.loc[fit_idx[fit_mask]],
                X.loc[valid_idx[valid_mask]],
                y_valid.loc[valid_idx[valid_mask]],
                alphas,
                penalty_multipliers,
            )
            prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_rows.append(coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label))
            model_rows.append({
                "Date": date,
                "Scope": scope_name,
                "ScopeLabel": label,
                "Alpha": model.alpha,
                "TrainingPeriods": len(train_dates),
                "TrainingObservations": int(fit_mask.sum() + valid_mask.sum()),
                "FeatureCount": len(model.columns),
            })
    return Layer3Prediction(
        prediction=prediction,
        coefficient_history=pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame(),
        model_history=pd.DataFrame(model_rows),
    )
