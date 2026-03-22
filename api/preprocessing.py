from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas import SETTING_COLS, USEFUL_SENSORS, WINDOW_SIZE, CycleReading

# FD001 and FD003 are single-condition; no KMeans was fit for them.
_SINGLE_CONDITION = {"FD001", "FD003"}


def cycles_to_dataframe(cycles: list[CycleReading]) -> pd.DataFrame:
    """Convert 30 CycleReading objects into a (30, 17) DataFrame.
    Columns: setting1, setting2, setting3, s2 … s21 (useful only).
    Row order is preserved (oldest → most recent)."""
    return pd.DataFrame([c.model_dump() for c in cycles])


def assign_clusters(
    df: pd.DataFrame,
    subset: str,
    kmeans_models: dict,
) -> np.ndarray:
    """Return a (30,) integer array — one cluster id per cycle row.

    FD002/FD004: predict on all 30 rows at once using the saved KMeans.
    FD001/FD003: every row gets cluster 0 (stored as np.int64 in the scalers).
    The dtype intentionally matches what was used as dict keys during training:
      np.int32 for multi-condition subsets, np.int64 for single-condition.
    """
    if subset in _SINGLE_CONDITION:
        return np.zeros(WINDOW_SIZE, dtype=np.int64)

    km = kmeans_models[subset]
    return km.predict(df[SETTING_COLS].values).astype(np.int32)


def normalise_window(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    subset: str,
    scalers: dict,
) -> np.ndarray:
    """Apply per-cluster MinMaxScaler to the sensor columns.

    Each cycle row is scaled by the scaler that matches its own cluster id —
    replicating the row-wise assignment done during training in notebook 07.

    Returns a (30, 14) float64 array of normalised sensor values.
    """
    sensor_array = df[USEFUL_SENSORS].values.copy().astype(np.float64)

    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        scaler = scalers[subset][cluster_id]
        sensor_array[mask] = scaler.transform(sensor_array[mask])

    return sensor_array  # shape (30, 14)


def build_xgb_features(window: np.ndarray) -> np.ndarray:
    """Flatten (30, 14) → (1, 420) for XGBoost.
    Row-major, matching X.reshape(X.shape[0], -1) from notebook 08."""
    return window.reshape(1, -1)


def build_lstm_tensor(window: np.ndarray):
    """Reshape (30, 14) → (1, 30, 14) PyTorch tensor.
    Import is deferred so the module loads without PyTorch installed."""
    import torch  # noqa: PLC0415
    return torch.tensor(window, dtype=torch.float32).unsqueeze(0)


def preprocess(
    cycles: list[CycleReading],
    subset: str,
    scalers: dict,
    kmeans_models: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline.

    Returns
    -------
    xgb_features : np.ndarray, shape (1, 420)  — ready for XGBRegressor.predict
    window       : np.ndarray, shape (30, 14)   — raw normalised window for LSTM path
    """
    df = cycles_to_dataframe(cycles)
    cluster_ids = assign_clusters(df, subset, kmeans_models)
    window = normalise_window(df, cluster_ids, subset, scalers)
    xgb_features = build_xgb_features(window)
    return xgb_features, window
