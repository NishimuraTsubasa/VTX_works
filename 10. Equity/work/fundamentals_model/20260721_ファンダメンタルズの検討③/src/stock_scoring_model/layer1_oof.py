from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .layer1_model_selection import select_candidate_model
from .layer1_single_factor import fit_single_factor
from .master import FactorMeta
from .regression_metrics import regression_metrics


def _coefficient_rows(
    model: object,
    *,
    date: pd.Timestamp,
    factor_code: str,
    training_periods: int,
    training_observations: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    rows = []
    for term, coef in zip(model.term_names, model.coef_):
        rows.append(
            {
                "Date": date,
                "FactorCode": factor_code,
                "SelectedModel": model.model_name,
                "Term": term,
                "Coefficient": float(coef),
                "Intercept": float(model.intercept_),
                "Knot": float(model.knot),
                "RidgeAlpha": float(ridge_alpha),
                "TrainingPeriods": int(training_periods),
                "TrainingObservations": int(training_observations),
            }
        )
    return pd.DataFrame(rows)


def generate_layer1_oof_subscores(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate global walk-forward OOF SubScores and model diagnostics.

    Returns
    -------
    predictions, selection_history, coefficient_history, fit_history
    """
    c = config["columns"]
    cfg = config["layer1"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("training_window_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 18))
    val_periods = int(cfg.get("validation_periods", 6))
    min_fit_obs = int(cfg.get("minimum_fit_observations", 200))
    min_val_obs = int(cfg.get("minimum_validation_observations", 100))
    ridge_alpha = float(cfg.get("ridge_alpha", 1e-8))
    predictions = pd.DataFrame(index=data.index)
    selection_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []

    for code in [k for k in metas if k in factor_scores.columns]:
        output = pd.Series(np.nan, index=data.index, dtype=float)
        for pos, date in enumerate(dates):
            train_dates = dates[max(0, pos - window):pos]
            if len(train_dates) < min_train or len(train_dates) <= val_periods:
                continue
            fit_dates = train_dates[:-val_periods]
            valid_dates = train_dates[-val_periods:]
            fit_idx = data.index[data[c["date"]].isin(fit_dates)]
            valid_idx = data.index[data[c["date"]].isin(valid_dates)]
            test_idx = data.index[data[c["date"]].eq(date)]
            fit_mask = factor_scores.loc[fit_idx, code].notna() & data.loc[fit_idx, target_col].notna()
            valid_mask = factor_scores.loc[valid_idx, code].notna() & data.loc[valid_idx, target_col].notna()
            if fit_mask.sum() < min_fit_obs or valid_mask.sum() < min_val_obs:
                continue
            selected, candidates = select_candidate_model(
                factor_scores.loc[fit_idx[fit_mask], code].to_numpy(float),
                data.loc[fit_idx[fit_mask], target_col].to_numpy(float),
                factor_scores.loc[valid_idx[valid_mask], code].to_numpy(float),
                data.loc[valid_idx[valid_mask], target_col].to_numpy(float),
                data.loc[valid_idx[valid_mask], c["date"]],
                config,
            )
            full_idx = data.index[data[c["date"]].isin(train_dates)]
            full_mask = factor_scores.loc[full_idx, code].notna() & data.loc[full_idx, target_col].notna()
            test_mask = factor_scores.loc[test_idx, code].notna()
            model = fit_single_factor(
                factor_scores.loc[full_idx[full_mask], code].to_numpy(float),
                data.loc[full_idx[full_mask], target_col].to_numpy(float),
                selected,
                knot=float(cfg.get("piecewise_knot", 0.0)),
                ridge_alpha=ridge_alpha,
            )
            if test_mask.any():
                output.loc[test_idx[test_mask]] = model.predict(
                    factor_scores.loc[test_idx[test_mask], code].to_numpy(float)
                )
            candidates.insert(0, "Date", date)
            candidates.insert(1, "FactorCode", code)
            selection_rows.append(candidates)
            coefficient_rows.append(
                _coefficient_rows(
                    model,
                    date=pd.Timestamp(date),
                    factor_code=code,
                    training_periods=len(train_dates),
                    training_observations=int(full_mask.sum()),
                    ridge_alpha=ridge_alpha,
                )
            )

            train_prediction = model.predict(
                factor_scores.loc[full_idx[full_mask], code].to_numpy(float)
            )
            train_metrics = regression_metrics(
                data.loc[full_idx[full_mask], target_col],
                train_prediction,
                feature_count=len(model.coef_),
            )
            selected_validation = candidates[candidates["CandidateModel"].eq(selected)]
            selected_row = selected_validation.iloc[0] if not selected_validation.empty else pd.Series(dtype=object)
            fit_rows.append(
                {
                    "Date": date,
                    "FactorCode": code,
                    "SelectedModel": selected,
                    "TrainingPeriods": len(train_dates),
                    "FitPeriods": len(fit_dates),
                    "ValidationPeriods": len(valid_dates),
                    "TrainingObservations": int(full_mask.sum()),
                    "TrainR2": train_metrics["R2"],
                    "TrainAdjustedR2": train_metrics["AdjustedR2"],
                    "TrainRMSE": train_metrics["RMSE"],
                    "TrainMAE": train_metrics["MAE"],
                    "TrainPearson": train_metrics["Pearson"],
                    "TrainSpearman": train_metrics["Spearman"],
                    "TrainPredictionStd": train_metrics["PredictionStd"],
                    "TrainTargetStd": train_metrics["TargetStd"],
                    "ValidationR2": selected_row.get("ValidationR2", np.nan),
                    "ValidationAdjustedR2": selected_row.get("ValidationAdjustedR2", np.nan),
                    "ValidationRMSE": selected_row.get("ValidationRMSE", np.nan),
                    "ValidationMAE": selected_row.get("ValidationMAE", np.nan),
                    "ValidationPearson": selected_row.get("ValidationPearson", np.nan),
                    "ValidationSpearman": selected_row.get("ValidationSpearman", np.nan),
                    "ValidationMeanRankIC": selected_row.get("MeanRankIC", np.nan),
                    "ValidationObservations": selected_row.get("ValidationObservations", np.nan),
                    "ValidationPredictionStd": selected_row.get("ValidationPredictionStd", np.nan),
                    "ValidationTargetStd": selected_row.get("ValidationTargetStd", np.nan),
                }
            )
        predictions[code] = output
    selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    coefficients = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()
    fit_history = pd.DataFrame(fit_rows)
    return predictions, selection, coefficients, fit_history
