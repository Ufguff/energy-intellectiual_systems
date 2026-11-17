"""Загрузка данных и разбиение по формам здания."""

from __future__ import annotations

import numpy as np
import pandas as pd

RENAME = {
    "X1": "compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_dist",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}
FEATURES = list(RENAME.values())[:8]
TARGETS = list(RENAME.values())[8:]
SHAPE_COLS = FEATURES[:4]


def load_data(path) -> pd.DataFrame:
    return pd.read_excel(path).rename(columns=RENAME)


def split_by_shape(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> dict:
    """Разбиение по группам-формам здания."""
    shape_ids = df.groupby(SHAPE_COLS, sort=False).ngroup()
    shapes = np.unique(shape_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(shapes)
    n = len(shapes)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    test_shapes = set(perm[:n_test])
    val_shapes = set(perm[n_test : n_test + n_val])
    train_shapes = set(perm[n_test + n_val :])

    train_mask = shape_ids.isin(train_shapes).to_numpy()
    val_mask = shape_ids.isin(val_shapes).to_numpy()
    test_mask = shape_ids.isin(test_shapes).to_numpy()

    return {
        "X_train": df.loc[train_mask, FEATURES],
        "y_train": df.loc[train_mask, TARGETS],
        "X_val": df.loc[val_mask, FEATURES],
        "y_val": df.loc[val_mask, TARGETS],
        "X_test": df.loc[test_mask, FEATURES],
        "y_test": df.loc[test_mask, TARGETS],
        "shapes": {
            "n": int(n),
            "n_train": len(train_shapes),
            "n_val": len(val_shapes),
            "n_test": len(test_shapes),
        },
    }
