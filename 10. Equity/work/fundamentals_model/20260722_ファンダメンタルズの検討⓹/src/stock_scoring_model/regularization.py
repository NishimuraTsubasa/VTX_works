from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_EPS = 1.0e-12


@dataclass
class PenalizedRidgeModel:
    columns: list[str]
    coef_: np.ndarray
    intercept_: float
    alpha: float
    penalty_multipliers: np.ndarray
    estimator_name: str = "ridge"
    feature_means_: np.ndarray | None = None
    feature_scales_: np.ndarray | None = None
    standardized_mask_: np.ndarray | None = None

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.reindex(columns=self.columns, fill_value=0.0).to_numpy(float)
        means = np.zeros(len(self.columns), dtype=float) if self.feature_means_ is None else np.asarray(self.feature_means_, dtype=float)
        scales = np.ones(len(self.columns), dtype=float) if self.feature_scales_ is None else np.asarray(self.feature_scales_, dtype=float)
        mask = np.zeros(len(self.columns), dtype=bool) if self.standardized_mask_ is None else np.asarray(self.standardized_mask_, dtype=bool)
        safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) > _EPS), scales, 1.0)
        transformed = arr.copy()
        transformed[:, mask] = (transformed[:, mask] - means[mask]) / safe_scales[mask]
        return transformed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._transform(X)
        return self.intercept_ + arr @ self.coef_

    @property
    def raw_coef_(self) -> np.ndarray:
        scales = np.ones(len(self.columns), dtype=float) if self.feature_scales_ is None else np.asarray(self.feature_scales_, dtype=float)
        mask = np.zeros(len(self.columns), dtype=bool) if self.standardized_mask_ is None else np.asarray(self.standardized_mask_, dtype=bool)
        safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) > _EPS), scales, 1.0)
        raw = np.asarray(self.coef_, dtype=float).copy()
        raw[mask] = raw[mask] / safe_scales[mask]
        return raw

    @property
    def raw_intercept_(self) -> float:
        means = np.zeros(len(self.columns), dtype=float) if self.feature_means_ is None else np.asarray(self.feature_means_, dtype=float)
        scales = np.ones(len(self.columns), dtype=float) if self.feature_scales_ is None else np.asarray(self.feature_scales_, dtype=float)
        mask = np.zeros(len(self.columns), dtype=bool) if self.standardized_mask_ is None else np.asarray(self.standardized_mask_, dtype=bool)
        safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) > _EPS), scales, 1.0)
        adjustment = np.sum(np.asarray(self.coef_, dtype=float)[mask] * means[mask] / safe_scales[mask])
        return float(self.intercept_ - adjustment)


def _normalize_standardize_mask(
    n_features: int,
    standardize_mask: np.ndarray | pd.Series | list[bool] | None,
) -> np.ndarray:
    if standardize_mask is None:
        return np.zeros(n_features, dtype=bool)
    mask = np.asarray(standardize_mask, dtype=bool)
    if mask.shape != (n_features,):
        raise ValueError(
            f"standardize_mask must have shape ({n_features},), got {mask.shape}."
        )
    return mask


def _fit_closed_form(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    penalty_multipliers: np.ndarray | None = None,
    estimator_name: str = "ridge",
    standardize_mask: np.ndarray | pd.Series | list[bool] | None = None,
) -> PenalizedRidgeModel:
    """切片を罰則対象外としてOLS/Ridgeを閉形式で推定する。

    ``standardize_mask`` がTrueの列だけ、学習標本の平均・標準偏差で標準化する。
    セクターダミー等はFalseのまま残せるため、Ridge罰則の単位依存を抑えつつ、
    ダミー変数の0/1解釈を維持できる。
    """
    cols = list(X.columns)
    raw_arr = X.to_numpy(float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(float)
    finite = np.isfinite(target) & np.isfinite(raw_arr).all(axis=1)
    raw_arr = raw_arr[finite]
    target = target[finite]
    multipliers = np.ones(len(cols)) if penalty_multipliers is None else np.asarray(penalty_multipliers, float)
    std_mask = _normalize_standardize_mask(len(cols), standardize_mask)

    feature_means = np.zeros(len(cols), dtype=float)
    feature_scales = np.ones(len(cols), dtype=float)
    if len(target) == 0:
        return PenalizedRidgeModel(
            cols,
            np.zeros(len(cols)),
            np.nan,
            float(alpha),
            multipliers,
            estimator_name,
            feature_means,
            feature_scales,
            std_mask,
        )

    arr = raw_arr.copy()
    if std_mask.any():
        feature_means[std_mask] = np.nanmean(raw_arr[:, std_mask], axis=0)
        feature_scales[std_mask] = np.nanstd(raw_arr[:, std_mask], axis=0, ddof=0)
        feature_scales[std_mask] = np.where(
            np.isfinite(feature_scales[std_mask]) & (feature_scales[std_mask] > _EPS),
            feature_scales[std_mask],
            1.0,
        )
        arr[:, std_mask] = (
            arr[:, std_mask] - feature_means[std_mask]
        ) / feature_scales[std_mask]

    x_mean = arr.mean(axis=0)
    y_mean = target.mean()
    xc = arr - x_mean
    yc = target - y_mean
    if float(alpha) == 0.0:
        coef, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    else:
        lhs = xc.T @ xc + np.diag(float(alpha) * multipliers)
        rhs = xc.T @ yc
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(lhs) @ rhs
    intercept = float(y_mean - x_mean @ coef)
    return PenalizedRidgeModel(
        cols,
        coef,
        intercept,
        float(alpha),
        multipliers,
        estimator_name,
        feature_means,
        feature_scales,
        std_mask,
    )


def fit_ols(
    X: pd.DataFrame,
    y: pd.Series,
    standardize_mask: np.ndarray | pd.Series | list[bool] | None = None,
) -> PenalizedRidgeModel:
    """単純OLS。多重共線性時も最小二乗解で推定を継続する。"""
    return _fit_closed_form(
        X,
        y,
        0.0,
        np.zeros(X.shape[1], dtype=float),
        estimator_name="ols",
        standardize_mask=standardize_mask,
    )


def fit_penalized_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    penalty_multipliers: np.ndarray | None = None,
    standardize_mask: np.ndarray | pd.Series | list[bool] | None = None,
) -> PenalizedRidgeModel:
    return _fit_closed_form(
        X,
        y,
        float(alpha),
        penalty_multipliers,
        estimator_name="ridge",
        standardize_mask=standardize_mask,
    )


def fit_penalized_ridge_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    alphas: list[float],
    penalty_multipliers: np.ndarray | None = None,
    standardize_mask: np.ndarray | pd.Series | list[bool] | None = None,
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
            standardize_mask=standardize_mask,
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
            standardize_mask=standardize_mask,
        )
    # 選択alphaで学習+検証を再推定。標準化統計量も全学習標本で再計算する。
    X_all = pd.concat([X_train, X_valid], axis=0)
    y_all = pd.concat([y_train, y_valid], axis=0)
    return _fit_closed_form(
        X_all,
        y_all,
        best.alpha,
        penalty_multipliers,
        estimator_name="ridge",
        standardize_mask=standardize_mask,
    )


def coefficient_frame(model: PenalizedRidgeModel, **metadata: object) -> pd.DataFrame:
    mask = (
        np.zeros(len(model.columns), dtype=bool)
        if model.standardized_mask_ is None
        else np.asarray(model.standardized_mask_, dtype=bool)
    )
    means = (
        np.zeros(len(model.columns), dtype=float)
        if model.feature_means_ is None
        else np.asarray(model.feature_means_, dtype=float)
    )
    scales = (
        np.ones(len(model.columns), dtype=float)
        if model.feature_scales_ is None
        else np.asarray(model.feature_scales_, dtype=float)
    )
    frame = pd.DataFrame(
        {
            "Feature": model.columns,
            # 係数比較の主列は標準化後の係数。旧コードとの互換性のため列名を維持。
            "Coefficient": model.coef_,
            "StandardizedCoefficient": model.coef_,
            "RawCoefficient": model.raw_coef_,
            "FeatureMean": means,
            "FeatureScale": scales,
            "Standardized": mask.astype(int),
            "PenaltyMultiplier": model.penalty_multipliers,
        }
    )
    frame["Intercept"] = model.intercept_
    frame["RawIntercept"] = model.raw_intercept_
    frame["Estimator"] = model.estimator_name
    frame["Alpha"] = model.alpha
    for key, value in metadata.items():
        frame[key] = value
    return frame
