from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .regularization import coefficient_frame, fit_ols, fit_penalized_ridge_cv


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
    eligible_rows: pd.Series | None = None,
) -> Layer3Prediction:
    """過去期間だけで逐次推定し、各時点の純粋なOOS予測を生成する。"""
    c = config["columns"]
    cfg = config["layer3"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("lookback_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 12))
    min_obs = int(cfg.get("minimum_training_observations", 250))
    estimator = str(cfg.get("estimator", "ridge")).lower()
    alphas = list(cfg.get("ridge_alphas", [0.1, 1, 10]))
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    coef_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    eligible = pd.Series(True, index=data.index) if eligible_rows is None else eligible_rows.reindex(data.index).fillna(False)

    for label in sorted(scope_labels.dropna().astype(str).unique()):
        label_mask = scope_labels.astype(str).eq(label)
        for pos, date in enumerate(dates):
            candidate_train_dates = dates[max(0, pos - window):pos]
            # Layer2 FactorScoreが利用可能な過去時点だけを学習期間として数える。
            available_train_dates = [
                d for d in candidate_train_dates
                if bool((label_mask & data[c["date"]].eq(d) & eligible).any())
            ]
            if len(available_train_dates) < min_train:
                continue

            test_idx = data.index[label_mask & data[c["date"]].eq(date) & eligible]
            if len(test_idx) == 0:
                continue

            if estimator == "ols":
                train_idx = data.index[label_mask & data[c["date"]].isin(available_train_dates) & eligible]
                y_train = data.loc[train_idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y_train = _demean_by_date(y_train, data.loc[train_idx, c["date"]])
                train_mask = y_train.notna() & np.isfinite(X.loc[train_idx]).all(axis=1)
                required = max(min_obs, len(X.columns) + 5)
                if int(train_mask.sum()) < required:
                    continue
                model = fit_ols(X.loc[train_idx[train_mask]], y_train.loc[train_idx[train_mask]])
                training_observations = int(train_mask.sum())
                validation_periods = 0
            else:
                valid_n = max(3, min(int(cfg.get("ridge_validation_periods", 6)), max(3, len(available_train_dates) // 4)))
                if len(available_train_dates) <= valid_n:
                    continue
                fit_dates = available_train_dates[:-valid_n]
                valid_dates = available_train_dates[-valid_n:]
                fit_idx = data.index[label_mask & data[c["date"]].isin(fit_dates) & eligible]
                valid_idx = data.index[label_mask & data[c["date"]].isin(valid_dates) & eligible]
                y_fit = data.loc[fit_idx, target_col]
                y_valid = data.loc[valid_idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y_fit = _demean_by_date(y_fit, data.loc[fit_idx, c["date"]])
                    y_valid = _demean_by_date(y_valid, data.loc[valid_idx, c["date"]])
                fit_mask = y_fit.notna() & np.isfinite(X.loc[fit_idx]).all(axis=1)
                valid_mask = y_valid.notna() & np.isfinite(X.loc[valid_idx]).all(axis=1)
                required = max(min_obs, len(X.columns) + 5)
                if int(fit_mask.sum() + valid_mask.sum()) < required or fit_mask.sum() < max(50, len(X.columns) + 5):
                    continue
                model = fit_penalized_ridge_cv(
                    X.loc[fit_idx[fit_mask]],
                    y_fit.loc[fit_idx[fit_mask]],
                    X.loc[valid_idx[valid_mask]],
                    y_valid.loc[valid_idx[valid_mask]],
                    alphas,
                    penalty_multipliers,
                )
                training_observations = int(fit_mask.sum() + valid_mask.sum())
                validation_periods = len(valid_dates)

            prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_rows.append(coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label))
            model_rows.append({
                "Date": date,
                "Scope": scope_name,
                "ScopeLabel": label,
                "Estimator": estimator,
                "Alpha": model.alpha,
                "TrainingPeriods": len(available_train_dates),
                "ValidationPeriods": validation_periods,
                "TrainingObservations": training_observations,
                "FeatureCount": len(model.columns),
                "FirstAvailableLayer2Date": min(available_train_dates) if available_train_dates else pd.NaT,
            })
    return Layer3Prediction(
        prediction=prediction,
        coefficient_history=pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame(),
        model_history=pd.DataFrame(model_rows),
    )
