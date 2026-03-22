from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ALL_COLS = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
    "s20", "s21",
]
_N_COLS = len(_ALL_COLS)

USEFUL_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
SETTING_COLS = ["setting1", "setting2", "setting3"]
WINDOW_SIZE  = 30
_KEEP_COLS   = SETTING_COLS + USEFUL_SENSORS

DEMO_ENGINES = [
    {"label": "Engine 36", "tier": "CRITICAL", "subset": "FD001", "engine_id": 36, "true_rul": 19},
    {"label": "Engine 21", "tier": "WARNING",  "subset": "FD001", "engine_id": 21, "true_rul": 57},
    {"label": "Engine 9",  "tier": "HEALTHY",  "subset": "FD001", "engine_id":  9, "true_rul": 111},
]

TIER_COLOUR = {
    "CRITICAL": "#ef4444",
    "WARNING":  "#f59e0b",
    "HEALTHY":  "#10b981",
}


def _read_cmapss(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = _ALL_COLS[: len(df.columns)]
    return df


def load_test_engine(subset, engine_id, data_dir="data/raw"):
    path = Path(data_dir) / f"test_{subset}.txt"
    df = _read_cmapss(path)
    engine_df = (
        df[df["engine_id"] == engine_id]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    if engine_df.empty:
        raise ValueError(f"Engine {engine_id} not found in {subset}.")
    return engine_df[_KEEP_COLS]


def build_degradation_windows(engine_df):
    n = len(engine_df)
    if n < WINDOW_SIZE:
        raise ValueError(f"Engine has only {n} cycles — need at least {WINDOW_SIZE}.")
    return [
        (end, engine_df.iloc[end - WINDOW_SIZE : end].reset_index(drop=True))
        for end in range(WINDOW_SIZE, n + 1)
    ]


def parse_uploaded_csv(file):
    try:
        df = pd.read_csv(file, sep=r"\s+", header=None)
    except Exception as exc:
        raise ValueError(f"Could not parse file: {exc}") from exc

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < _N_COLS - 1:
        raise ValueError(
            f"Expected {_N_COLS} columns (raw CMAPSS format), got {df.shape[1]}."
        )

    df.columns = _ALL_COLS[: df.shape[1]]

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"File has only {len(df)} rows — need at least {WINDOW_SIZE} cycles.")

    engine_ids = df["engine_id"].unique()
    if len(engine_ids) > 1:
        df = df[df["engine_id"] == int(engine_ids[0])].copy()

    return df.sort_values("cycle").reset_index(drop=True)[_KEEP_COLS]


def get_demo_engines(data_dir="data/raw"):
    result = []
    for spec in DEMO_ENGINES:
        engine_df = load_test_engine(spec["subset"], spec["engine_id"], data_dir)
        result.append({**spec, "engine_df": engine_df})
    return result