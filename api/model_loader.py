from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("rul_api.loader")

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _models_dir() -> Path:
    env = os.getenv("MODELS_DIR")
    return Path(env) if env else _DEFAULT_MODELS_DIR


# ---------------------------------------------------------------------------
# LSTM architecture — must match notebook 09 exactly so state_dict loads clean
# ---------------------------------------------------------------------------

def _build_lstm_model():
    """Build an uninitialised RULPredictor.
    Class definition mirrors notebook 09 cell 3 verbatim:
      self.lstm      → nn.LSTM(14, 256, num_layers=3, batch_first=True, dropout=0.3)
      self.regressor → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,1)
    """
    import torch.nn as nn  # noqa: PLC0415

    class RULPredictor(nn.Module):
        def __init__(
            self,
            input_size: int = 14,
            hidden_size: int = 256,
            num_layers: int = 3,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_step = lstm_out[:, -1, :]          # final timestep only
            return self.regressor(last_step).squeeze(1)

    return RULPredictor()


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class ModelStore:
    """Holds all loaded artefacts. Instantiated once via get_model_store()."""

    def __init__(self) -> None:
        mdir = _models_dir()

        # --- scalers + kmeans -----------------------------------------------
        scalers_path = mdir / "multi_scalers.pkl"
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers not found: {scalers_path}")
        with open(scalers_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.scalers: dict = bundle["scalers"]
        self.kmeans: dict  = bundle["kmeans"]
        log.info("Scalers and KMeans loaded.")

        # --- XGBoost --------------------------------------------------------
        xgb_path = mdir / "xgb_multi.pkl"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
        with open(xgb_path, "rb") as fh:
            self.xgb = pickle.load(fh)
        log.info("XGBoost loaded (best_iteration=%d).", self.xgb.best_iteration)

        # --- LSTM (optional) ------------------------------------------------
        # Skipped by default — torch import hangs on some machines at startup.
        # Opt in explicitly: LOAD_LSTM=1 uvicorn api.main:app ...
        self.lstm = None
        if os.getenv("LOAD_LSTM") == "1":
            lstm_path = mdir / "lstm_multi_best.pt"
            try:
                import torch  # noqa: PLC0415
                model = _build_lstm_model()
                state_dict = torch.load(lstm_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.lstm = model
                log.info("LSTM loaded.")
            except Exception as e:
                log.warning("LSTM not loaded (%s) — LSTM endpoint disabled.", e)
        else:
            log.info("LSTM skipped (set LOAD_LSTM=1 to enable).")

    @property
    def loaded_models(self) -> list[str]:
        names = ["xgboost", "scalers", "kmeans"]
        if self.lstm is not None:
            names.append("lstm")
        return names


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """Return the singleton ModelStore, loading artefacts on first call."""
    return ModelStore()