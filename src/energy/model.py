"""Конвейер scikit-learn и инкрементальный Ridge."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(alpha: float) -> Pipeline:
    """Предобработка (импутация + масштабирование) + MultiOutput Ridge."""
    pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    est = MultiOutputRegressor(Ridge(alpha=alpha))
    return Pipeline([("pre", pre), ("model", est)])


class IncrementalRidge:
    """Ridge с частичным обучением по чанкам."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def partial_fit(self, X, y) -> "IncrementalRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if not hasattr(self, "gram_"):
            d = X.shape[1]
            self.gram_ = np.zeros((d, d))
            self.xty_ = np.zeros((d, y.shape[1]))
            self.x_sum_ = np.zeros(d)
            self.y_sum_ = np.zeros(y.shape[1])
            self.n_ = 0
        self.gram_ += X.T @ X
        self.xty_ += X.T @ y
        self.x_sum_ += X.sum(axis=0)
        self.y_sum_ += y.sum(axis=0)
        self.n_ += len(X)
        return self

    def _finalize(self) -> None:
        n = self.n_
        mu_x = self.x_sum_ / n
        mu_y = self.y_sum_ / n
        gram_c = self.gram_ - np.outer(self.x_sum_, mu_x)
        xty_c = self.xty_ - np.outer(self.x_sum_, mu_y)
        eye = np.eye(self.gram_.shape[0])
        self.coef_ = np.linalg.solve(gram_c + self.alpha * eye, xty_c)
        self.intercept_ = mu_y - mu_x @ self.coef_

    def predict(self, X) -> np.ndarray:
        self._finalize()
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
