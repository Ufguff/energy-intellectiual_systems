"""Метрики регрессии с бутстрэп-интервалами."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def rmse(y_true, y_pred) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def r2(y_true, y_pred) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / ss_tot


METRICS = {"mae": mae, "rmse": rmse, "r2": r2}


def bootstrap(y_true, y_pred, metric, n_bootstrap: int = 2000, seed: int = 42):
    """Бутстрэп-интервал: медиана и 2,5/97,5 процентили по столбцам (целям)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty((n_bootstrap, y_true.shape[1]))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y_true[idx], y_pred[idx])
    return np.percentile(scores, [50.0, 2.5, 97.5], axis=0)
