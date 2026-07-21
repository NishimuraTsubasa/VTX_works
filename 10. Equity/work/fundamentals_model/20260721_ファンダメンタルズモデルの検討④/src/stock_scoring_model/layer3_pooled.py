from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .regularization import (
    coefficient_frame,
    fit_ols,
    fit_penalized_ridge,
    fit_penalized_ridge_cv,
)
from .regression_metrics import prefixed_metrics


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
    standardize_mask: np.ndarray,
    config: dict[str, Any],
    scope_labels: pd.Series,
    scope_name: str,
    target_col: str = "NextMonthReturn",
    eligible_rows: pd.Series | None = None,
) -> Layer3Prediction:
    """Fit sequential pooled regressions and retain fit/validation diagnostics."""
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
            available_train_dates = [
                d for d in candidate_train_dates
                if bool((label_mask & data[c["date"]].eq(d) & eligible).any())
            ]
            if len(available_train_dates) < min_train:
                continue

            test_idx = data.index[label_mask & data[c["date"]].eq(date) & eligible]
            if len(test_idx) == 0:
                continue

            diagnostic_valid_n = max(
                3,
                min(
                    int(cfg.get("ridge_validation_periods", 6)),
                    max(3, len(available_train_dates) // 4),
                ),
            )
            fit_dates_diag = available_train_dates[:-diagnostic_valid_n]
            valid_dates_diag = available_train_dates[-diagnostic_valid_n:]
            validation_metrics: dict[str, object] = {}

            if estimator == "ols":
                train_idx = data.index[label_mask & data[c["date"]].isin(available_train_dates) & eligible]
                y_train = data.loc[train_idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y_train = _demean_by_date(y_train, data.loc[train_idx, c["date"]])
                train_mask = y_train.notna() & np.isfinite(X.loc[train_idx]).all(axis=1)
                required = max(min_obs, len(X.columns) + 5)
                if int(train_mask.sum()) < required:
                    continue
                model = fit_ols(
                    X.loc[train_idx[train_mask]],
                    y_train.loc[train_idx[train_mask]],
                    standardize_mask=standardize_mask,
                )
                training_observations = int(train_mask.sum())

                if len(fit_dates_diag) >= 3:
                    fit_idx = data.index[label_mask & data[c["date"]].isin(fit_dates_diag) & eligible]
                    valid_idx = data.index[label_mask & data[c["date"]].isin(valid_dates_diag) & eligible]
                    y_fit = data.loc[fit_idx, target_col]
                    y_valid = data.loc[valid_idx, target_col]
                    if cfg.get("demean_target_by_date", True):
                        y_fit = _demean_by_date(y_fit, data.loc[fit_idx, c["date"]])
                        y_valid = _demean_by_date(y_valid, data.loc[valid_idx, c["date"]])
                    fit_mask = y_fit.notna() & np.isfinite(X.loc[fit_idx]).all(axis=1)
                    valid_mask = y_valid.notna() & np.isfinite(X.loc[valid_idx]).all(axis=1)
                    if fit_mask.sum() > len(X.columns) + 5 and valid_mask.sum() > 0:
                        diagnostic_model = fit_ols(
                            X.loc[fit_idx[fit_mask]],
                            y_fit.loc[fit_idx[fit_mask]],
                            standardize_mask=standardize_mask,
                        )
                        validation_metrics = prefixed_metrics(
                            y_valid.loc[valid_idx[valid_mask]],
                            diagnostic_model.predict(X.loc[valid_idx[valid_mask]]),
                            "Validation",
                            feature_count=len(X.columns),
                        )
                validation_periods = diagnostic_valid_n if validation_metrics else 0
            else:
                valid_n = diagnostic_valid_n
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
                    standardize_mask=standardize_mask,
                )
                diagnostic_model = fit_penalized_ridge(
                    X.loc[fit_idx[fit_mask]],
                    y_fit.loc[fit_idx[fit_mask]],
                    model.alpha,
                    penalty_multipliers,
                    standardize_mask=standardize_mask,
                )
                validation_metrics = prefixed_metrics(
                    y_valid.loc[valid_idx[valid_mask]],
                    diagnostic_model.predict(X.loc[valid_idx[valid_mask]]),
                    "Validation",
                    feature_count=len(X.columns),
                )
                training_observations = int(fit_mask.sum() + valid_mask.sum())
                validation_periods = len(valid_dates)
                train_idx = data.index[label_mask & data[c["date"]].isin(available_train_dates) & eligible]
                y_train = data.loc[train_idx, target_col]
                if cfg.get("demean_target_by_date", True):
                    y_train = _demean_by_date(y_train, data.loc[train_idx, c["date"]])
                train_mask = y_train.notna() & np.isfinite(X.loc[train_idx]).all(axis=1)

            prediction.loc[test_idx] = model.predict(X.loc[test_idx])
            coef_rows.append(coefficient_frame(model, Date=date, Scope=scope_name, ScopeLabel=label))
            train_metrics = prefixed_metrics(
                y_train.loc[train_idx[train_mask]],
                model.predict(X.loc[train_idx[train_mask]]),
                "Train",
                feature_count=len(X.columns),
            )
            model_row: dict[str, object] = {
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
                "StandardizedFeatureCount": int(np.asarray(standardize_mask, dtype=bool).sum()),
            }
            model_row.update(train_metrics)
            model_row.update(validation_metrics)
            model_rows.append(model_row)
    return Layer3Prediction(
        prediction=prediction,
        coefficient_history=pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame(),
        model_history=pd.DataFrame(model_rows),
    )
