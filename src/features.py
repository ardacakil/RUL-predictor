"""
features.py
-----------
Sliding-window construction utilities for the CMAPSS multi-condition pipeline.

Imported by preprocess.py — not intended to be run directly.
"""

import numpy as np


def create_train_windows(df, window_size: int, sensors: list[str]):
    """
    Slide a window of `window_size` cycles over every engine in `df`.

    Each window produces one training sample whose label is the RUL at the
    final cycle of the window.  Engines shorter than `window_size` are skipped
    (they are rare and contribute noise rather than signal).

    Parameters
    ----------
    df          : DataFrame with columns [engine_id, cycle, *sensors, RUL]
    window_size : number of cycles per sample (30 in the paper)
    sensors     : list of sensor column names to include

    Returns
    -------
    X : np.ndarray  shape (n_samples, window_size, n_sensors)
    y : np.ndarray  shape (n_samples,)
    """
    X, y = [], []

    for engine_id in df["engine_id"].unique():
        eng = df[df["engine_id"] == engine_id].sort_values("cycle")

        if len(eng) < window_size:
            continue  # engine too short — skip rather than pad train data

        sensor_vals = eng[sensors].values
        rul_vals    = eng["RUL"].values

        for i in range(len(eng) - window_size + 1):
            X.append(sensor_vals[i : i + window_size])
            y.append(rul_vals[i + window_size - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_test_windows(df, window_size: int, sensors: list[str]):
    """
    Extract the single prediction window for each test engine.

    Test files are cut off at an unknown point — we predict RUL from the
    last `window_size` cycles available.  Engines shorter than `window_size`
    are zero-padded at the front so every engine produces exactly one sample.

    Parameters
    ----------
    df          : DataFrame with columns [engine_id, cycle, *sensors]
    window_size : number of cycles per sample
    sensors     : list of sensor column names to include

    Returns
    -------
    X : np.ndarray  shape (n_engines, window_size, n_sensors)
    """
    X = []

    for engine_id in df["engine_id"].unique():
        eng = df[df["engine_id"] == engine_id].sort_values("cycle")
        vals = eng[sensors].values

        if len(eng) >= window_size:
            X.append(vals[-window_size:])
        else:
            # Zero-pad at the front — the model never saw healthy-only engines
            # during training so padding with zeros is the safest neutral choice
            pad = np.zeros((window_size - len(eng), len(sensors)), dtype=np.float32)
            X.append(np.vstack([pad, vals]))

    return np.array(X, dtype=np.float32)