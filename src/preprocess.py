"""
preprocess.py
-------------
Full multi-condition preprocessing pipeline for the NASA CMAPSS dataset.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Load raw train/test/RUL files for all four subsets (FD001–FD004)
2. Drop the seven flat sensors that carry no degradation signal
3. Label training data with capped RUL values
4. Assign each row to an operating-condition cluster
   - FD001 / FD003  (single condition)  → cluster 0 for every row, no KMeans
   - FD002 / FD004  (6 conditions)      → KMeans(k=6) fit on train settings
5. Fit one MinMaxScaler per cluster on training data only; transform train + test
6. Build sliding-window arrays for training and the last-window test arrays
7. Save .npy arrays to --out-dir and scalers/kmeans to --models-dir

Usage
~~~~~
    python src/preprocess.py --data-dir data/raw --out-dir data --models-dir models

All three flags have defaults so a bare `python src/preprocess.py` works if
the repo layout is unchanged.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from features import create_train_windows, create_test_windows

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_NAMES = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3",
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13",
    "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

SETTING_COLS   = ["setting1", "setting2", "setting3"]

# Sensors with zero variance across all operating conditions — drop on load
FLAT_SENSORS   = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Sensors that carry degradation signal (validated in notebook 01 / 06)
USEFUL_SENSORS = [
    "s2",  "s3",  "s4",  "s7",  "s8",  "s9",  "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

SUBSETS      = ["FD001", "FD002", "FD003", "FD004"]
MULTI_COND   = {"FD002", "FD004"}   # subsets that need per-cluster scaling
RUL_CAP      = 125
WINDOW_SIZE  = 30
N_CLUSTERS   = 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(data_dir: Path, subset: str):
    """Load train, test and RUL files for one subset."""
    kw = dict(sep=r"\s+", header=None, names=COL_NAMES, engine="python")

    train = pd.read_csv(data_dir / f"train_{subset}.txt", **kw)
    test  = pd.read_csv(data_dir / f"test_{subset}.txt",  **kw)
    rul   = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", header=None, names=["RUL"]
    )

    # Drop flat sensors immediately — they add nothing downstream
    train.drop(columns=FLAT_SENSORS, inplace=True)
    test.drop(columns=FLAT_SENSORS,  inplace=True)

    # Ensure sensor columns are float
    train[USEFUL_SENSORS] = train[USEFUL_SENSORS].astype(float)
    test[USEFUL_SENSORS]  = test[USEFUL_SENSORS].astype(float)

    return train, test, rul


# ---------------------------------------------------------------------------
# RUL labelling
# ---------------------------------------------------------------------------

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-engine RUL and cap at RUL_CAP.

    RUL at cycle t = (max cycle for that engine) - t
    Capped at 125 so the model focuses precision on the danger zone rather
    than distinguishing healthy-vs-very-healthy.
    """
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df = df.copy()
    df["RUL"] = df["engine_id"].map(max_cycles) - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

def assign_clusters(train: pd.DataFrame, test: pd.DataFrame, subset: str):
    """
    Assign each row to an operating-condition cluster.

    Single-condition subsets (FD001, FD003) always get cluster 0.
    Multi-condition subsets (FD002, FD004) use KMeans fit on train settings.

    Returns (train_with_cluster, test_with_cluster, kmeans_or_None)
    """
    train = train.copy()
    test  = test.copy()

    if subset not in MULTI_COND:
        train["cluster"] = 0
        test["cluster"]  = 0
        return train, test, None

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    train["cluster"] = km.fit_predict(train[SETTING_COLS])
    test["cluster"]  = km.predict(test[SETTING_COLS])
    return train, test, km


# ---------------------------------------------------------------------------
# Per-cluster normalisation
# ---------------------------------------------------------------------------

def fit_and_apply_scalers(train: pd.DataFrame, test: pd.DataFrame):
    """
    Fit one MinMaxScaler per cluster on train rows only; apply to train + test.

    Returns (scaled_train, scaled_test, scalers_dict)
    where scalers_dict maps cluster_id -> fitted MinMaxScaler.
    """
    train  = train.copy()
    test   = test.copy()
    scalers = {}

    for cluster_id in sorted(train["cluster"].unique()):
        train_mask = train["cluster"] == cluster_id
        test_mask  = test["cluster"]  == cluster_id

        scaler = MinMaxScaler()
        scaler.fit(train.loc[train_mask, USEFUL_SENSORS])

        train.loc[train_mask, USEFUL_SENSORS] = scaler.transform(
            train.loc[train_mask, USEFUL_SENSORS]
        )
        test.loc[test_mask, USEFUL_SENSORS] = scaler.transform(
            test.loc[test_mask, USEFUL_SENSORS]
        )

        scalers[cluster_id] = scaler

    return train, test, scalers


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_dir: Path, out_dir: Path, models_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_scalers = {}
    all_kmeans  = {}
    train_windows_per_subset = {}
    test_windows_per_subset  = {}

    for subset in SUBSETS:
        print(f"\n── {subset} ──────────────────────────────")

        # 1. Load
        train, test, rul = load_raw(data_dir, subset)
        print(f"  loaded   train {train.shape}  test {test.shape}")

        # 2. RUL labels (train only)
        train = add_rul(train)
        print(f"  RUL range: {train['RUL'].min():.0f} – {train['RUL'].max():.0f}")

        # 3. Cluster assignment
        train, test, km = assign_clusters(train, test, subset)
        n_clusters = train["cluster"].nunique()
        print(f"  clusters : {n_clusters}  ({'KMeans' if km else 'trivial'})")

        # 4. Per-cluster normalisation
        train, test, scalers = fit_and_apply_scalers(train, test)
        print(f"  normalised {len(scalers)} cluster(s)")

        all_scalers[subset] = scalers
        all_kmeans[subset]  = km

        # 5. Sliding windows
        X_train, y_train = create_train_windows(train, WINDOW_SIZE, USEFUL_SENSORS)
        X_test            = create_test_windows(test,  WINDOW_SIZE, USEFUL_SENSORS)
        y_test            = rul["RUL"].values.clip(max=RUL_CAP).astype(np.float32)

        train_windows_per_subset[subset] = (X_train, y_train)
        test_windows_per_subset[subset]  = (X_test,  y_test)

        print(f"  train windows : {X_train.shape}")
        print(f"  test  windows : {X_test.shape}")

        # 6. Save per-subset test arrays
        np.save(out_dir / f"X_test_{subset}.npy", X_test)
        np.save(out_dir / f"y_test_{subset}.npy", y_test)

    # 7. Combine training windows across all subsets
    X_train_all = np.concatenate([train_windows_per_subset[s][0] for s in SUBSETS])
    y_train_all = np.concatenate([train_windows_per_subset[s][1] for s in SUBSETS])

    # Subset origin label per sample (handy for debugging)
    subset_labels = np.concatenate([
        np.full(len(train_windows_per_subset[s][1]), i)
        for i, s in enumerate(SUBSETS)
    ])

    np.save(out_dir / "X_train_multi.npy", X_train_all)
    np.save(out_dir / "y_train_multi.npy", y_train_all)
    np.save(out_dir / "subset_labels.npy", subset_labels)

    print(f"\n── Combined ──────────────────────────────")
    print(f"  X_train : {X_train_all.shape}")
    print(f"  y_train : {y_train_all.shape}  RUL {y_train_all.min():.0f}–{y_train_all.max():.0f}")

    # 8. Save scalers + KMeans for the API / inference
    artifact = {"scalers": all_scalers, "kmeans": all_kmeans}
    with open(models_dir / "multi_scalers.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n  scalers  → {models_dir / 'multi_scalers.pkl'}")
    print(f"  arrays   → {out_dir}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess NASA CMAPSS data for the multi-condition RUL pipeline."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the raw train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt files. "
             "(default: data/raw)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory where .npy arrays will be written. (default: data)",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where multi_scalers.pkl will be saved. (default: models)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir   = args.data_dir,
        out_dir    = args.out_dir,
        models_dir = args.models_dir,
    )