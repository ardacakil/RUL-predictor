from __future__ import annotations

from typing import Callable

import pandas as pd
import requests

from data_utils import SETTING_COLS, USEFUL_SENSORS, build_degradation_windows

_DEFAULT_BASE_URL = "http://localhost:8000"


def predict_window(
    window_df: pd.DataFrame,
    subset: str,
    base_url: str = _DEFAULT_BASE_URL,
    endpoint: str = "/predict",
) -> float:
    """POST one 30-cycle window to /predict and return predicted_rul.

    Parameters
    ----------
    window_df : DataFrame with exactly 30 rows and columns:
                setting1, setting2, setting3, s2 … s21 (useful sensors)
    subset    : one of FD001–FD004
    base_url  : API base URL (no trailing slash)

    Raises
    ------
    RuntimeError if the API returns a non-200 response.
    """
    all_cols = SETTING_COLS + USEFUL_SENSORS
    cycles_payload = window_df[all_cols].to_dict(orient="records")

    payload = {"subset": subset, "cycles": cycles_payload}

    try:
        resp = requests.post(
            f"{base_url}{endpoint}",
            json=payload,
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach the API at {base_url}. "
            "Make sure the server is running."
        )

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(f"API error {resp.status_code}: {detail}")

    return float(resp.json()["predicted_rul"])


def predict_degradation_curve(
    engine_df: pd.DataFrame,
    subset: str,
    base_url: str = _DEFAULT_BASE_URL,
    endpoint: str = "/predict",
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[int], list[float]]:
    """Run predict_window for every sliding window across an engine's history.

    Parameters
    ----------
    engine_df   : full cycle history DataFrame (settings + useful sensors)
    subset      : one of FD001–FD004
    base_url    : API base URL
    on_progress : optional callback(current, total) for progress reporting

    Returns
    -------
    cycle_numbers  : list[int]   — x-axis (cycle index of window's last row)
    predicted_ruls : list[float] — y-axis (predicted RUL at that cycle)
    """
    windows = build_degradation_windows(engine_df)
    total = len(windows)

    cycle_numbers:  list[int]   = []
    predicted_ruls: list[float] = []

    for i, (cycle_num, window_df) in enumerate(windows):
        rul = predict_window(window_df, subset, base_url, endpoint)
        cycle_numbers.append(cycle_num)
        predicted_ruls.append(rul)
        if on_progress:
            on_progress(i + 1, total)

    return cycle_numbers, predicted_ruls