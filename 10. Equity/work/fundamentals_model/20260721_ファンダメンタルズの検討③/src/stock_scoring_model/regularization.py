from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PenalizedRidgeModel:
    columns: list[str]
    coef_: np.ndarray
    intercept_: float
    alpha: float
    penalty_multipliers: np.ndarray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.reindex(columns=self.columns, fill_value=0.0).to_numpy(float)
        return self.intercept_ + arr @ self.coef_


def _fit_closed_form(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    penalty_multipliers: np.ndarray | None = None,
) -> PenalizedRidgeModel:
    cols = list(X.columns)
    arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(target) & np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    target = target[mask]
    if len(target) == 0:
        return PenalizedRidgeModel(cols, np.zeros(len(cols)), np.nan, alpha, np.ones(len(cols)))
    x_mean = arr.mean(axis=0)
    y_mean = target.mean()
    xc = arr - x_mean
    yc = target - y_mean
    multipliers = np.ones(len(cols)) if penalty_multipliers is None else np.asarray(penalty_multipliers, float)
    penalty = np.diag(alpha * multipliers)
    coef = np.linalg.pinv(xc.T @ xc + penalty) @ (xc.T @ yc)
    intercept = float(y_mean - x_mean @ coef)
    return PenalizedRidgeModel(cols, coef, intercept, alpha, multipliers)


def fit_penalized_ridge_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    alphas: list[float],
    penalty_multipliers: np.ndarray | None = None,
) -> PenalizedRidgeModel:
    best: PenalizedRidgeModel | None = None
    best_loss = np.inf
    for alpha in alphas:
        model = _fit_closed_form(X_train, y_train, float(alpha), penalty_multipliers)
        pred = model.predict(X_valid)
        actual = pd.to_numeric(y_valid, errors="coerce").to_numpy(float)
        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() == 0:
            continue
        loss = float(np.mean((pred[mask] - actual[mask]) ** 2))
        if loss < best_loss:
            best_loss, best = loss, model
    if best is None:
        best = _fit_closed_form(X_train, y_train, float(alphas[0]), penalty_multipliers)
    # 選択alphaで学習+検証を再推定
    X_all = pd.concat([X_train, X_valid], axis=0)
    y_all = pd.concat([y_train, y_valid], axis=0)
    return _fit_closed_form(X_all, y_all, best.alpha, penalty_multipliers)


def coefficient_frame(model: PenalizedRidgeModel, **metadata: object) -> pd.DataFrame:
    frame = pd.DataFrame({
        "Feature": model.columns,
        "Coefficient": model.coef_,
        "PenaltyMultiplier": model.penalty_multipliers,
    })
    frame["Intercept"] = model.intercept_
    frame["Alpha"] = model.alpha
    for key, value in metadata.items():
        frame[key] = value
    return frame
