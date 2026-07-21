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
    estimator_name: str = "ridge"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.reindex(columns=self.columns, fill_value=0.0).to_numpy(float)
        return self.intercept_ + arr @ self.coef_


def _fit_closed_form(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    penalty_multipliers: np.ndarray | None = None,
    estimator_name: str = "ridge",
) -> PenalizedRidgeModel:
    """切片を罰則対象外としてOLS/Ridgeを閉形式で推定する。"""
    cols = list(X.columns)
    arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(target) & np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    target = target[mask]
    multipliers = np.ones(len(cols)) if penalty_multipliers is None else np.asarray(penalty_multipliers, float)
    if len(target) == 0:
        return PenalizedRidgeModel(cols, np.zeros(len(cols)), np.nan, float(alpha), multipliers, estimator_name)

    x_mean = arr.mean(axis=0)
    y_mean = target.mean()
    xc = arr - x_mean
    yc = target - y_mean
    if float(alpha) == 0.0:
        # OLSは最小二乗法を直接使い、SVDによる大きな正方行列のpinvを避ける。
        coef, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    else:
        lhs = xc.T @ xc + np.diag(float(alpha) * multipliers)
        rhs = xc.T @ yc
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(lhs) @ rhs
    intercept = float(y_mean - x_mean @ coef)
    return PenalizedRidgeModel(cols, coef, intercept, float(alpha), multipliers, estimator_name)


def fit_ols(X: pd.DataFrame, y: pd.Series) -> PenalizedRidgeModel:
    """単純OLS。多重共線性時も擬似逆行列で推定を継続する。"""
    return _fit_closed_form(X, y, 0.0, np.zeros(X.shape[1], dtype=float), estimator_name="ols")


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
        model = _fit_closed_form(
            X_train,
            y_train,
            float(alpha),
            penalty_multipliers,
            estimator_name="ridge",
        )
        pred = model.predict(X_valid)
        actual = pd.to_numeric(y_valid, errors="coerce").to_numpy(float)
        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() == 0:
            continue
        loss = float(np.mean((pred[mask] - actual[mask]) ** 2))
        if loss < best_loss:
            best_loss, best = loss, model
    if best is None:
        best = _fit_closed_form(
            X_train,
            y_train,
            float(alphas[0]),
            penalty_multipliers,
            estimator_name="ridge",
        )
    # 選択alphaで学習+検証を再推定
    X_all = pd.concat([X_train, X_valid], axis=0)
    y_all = pd.concat([y_train, y_valid], axis=0)
    return _fit_closed_form(
        X_all,
        y_all,
        best.alpha,
        penalty_multipliers,
        estimator_name="ridge",
    )


def coefficient_frame(model: PenalizedRidgeModel, **metadata: object) -> pd.DataFrame:
    frame = pd.DataFrame({
        "Feature": model.columns,
        "Coefficient": model.coef_,
        "PenaltyMultiplier": model.penalty_multipliers,
    })
    frame["Intercept"] = model.intercept_
    frame["Estimator"] = model.estimator_name
    frame["Alpha"] = model.alpha
    for key, value in metadata.items():
        frame[key] = value
    return frame
