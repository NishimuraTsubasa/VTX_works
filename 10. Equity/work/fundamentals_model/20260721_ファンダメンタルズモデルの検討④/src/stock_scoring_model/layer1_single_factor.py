from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class SingleFactorModel:
    model_name: str
    estimator: object
    knot: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(design_matrix(x, self.model_name, self.knot))

    @property
    def intercept_(self) -> float:
        value = np.asarray(getattr(self.estimator, "intercept_", np.nan)).reshape(-1)
        return float(value[0]) if len(value) else np.nan

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(getattr(self.estimator, "coef_", []), dtype=float).reshape(-1)

    @property
    def term_names(self) -> list[str]:
        if self.model_name == "linear":
            return ["Linear"]
        if self.model_name == "piecewise":
            return ["Linear", "Hinge"]
        if self.model_name == "quadratic":
            return ["Linear", "Quadratic"]
        return [f"Term{idx + 1}" for idx in range(len(self.coef_))]


def design_matrix(x: np.ndarray, model_name: str, knot: float = 0.0) -> np.ndarray:
    z = np.asarray(x, dtype=float).reshape(-1)
    if model_name == "linear":
        return z[:, None]
    if model_name == "piecewise":
        return np.column_stack([z, np.maximum(z - knot, 0.0)])
    if model_name == "quadratic":
        return np.column_stack([z, z**2])
    raise ValueError(f"Unsupported layer1 model: {model_name}")


def fit_single_factor(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    knot: float = 0.0,
    ridge_alpha: float = 1e-8,
) -> SingleFactorModel:
    X = design_matrix(x, model_name, knot)
    if ridge_alpha > 0:
        estimator = Ridge(alpha=ridge_alpha)
    else:
        estimator = LinearRegression()
    estimator.fit(X, np.asarray(y, float))
    return SingleFactorModel(model_name=model_name, estimator=estimator, knot=knot)
