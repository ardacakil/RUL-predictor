"""
RUL Predictor API
─────────────────
GET  /health          liveness check + loaded model list
POST /predict         XGBoost prediction (primary, no GPU required)
POST /predict/lstm    LSTM prediction (requires PyTorch + checkpoint)

Run locally:
  uvicorn api.main:app --reload --port 8000

Production:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from api.model_loader import get_model_store
from api.preprocessing import build_lstm_tensor, preprocess
from api.schemas import HealthResponse, PredictRequest, PredictResponse

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rul_api")


# ---------------------------------------------------------------------------
# Lifespan: warm-up model store before the first request
# ---------------------------------------------------------------------------

import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artefacts …")
    loop = asyncio.get_event_loop()
    store = await loop.run_in_executor(None, get_model_store)
    log.info("Ready. Loaded: %s", store.loaded_models)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RUL Predictor — NASA CMAPSS",
    description=(
        "Predicts Remaining Useful Life for turbofan engines "
        "using XGBoost (primary) or LSTM (alternative)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns which model artefacts are currently loaded."""
    store = get_model_store()
    return HealthResponse(models_loaded=store.loaded_models)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """XGBoost RUL prediction.

    Accepts the last 30 cycles of sensor readings for a single engine.
    Preprocessing mirrors the training pipeline exactly:
      1. Assign a per-cycle operating-regime cluster via saved KMeans.
      2. Normalise each cycle with its cluster's MinMaxScaler.
      3. Flatten the (30, 14) window to (1, 420) for XGBoost.
    """
    store = get_model_store()

    xgb_features, _ = preprocess(
        cycles=request.cycles,
        subset=request.subset,
        scalers=store.scalers,
        kmeans_models=store.kmeans,
    )

    raw_pred: float = float(store.xgb.predict(xgb_features)[0])
    predicted_rul = max(0.0, raw_pred)

    log.info("XGBoost | subset=%s | predicted_rul=%.2f", request.subset, predicted_rul)

    return PredictResponse(
        predicted_rul=predicted_rul,
        subset=request.subset,
        model_used="xgboost",
    )


@app.post("/predict/lstm", response_model=PredictResponse)
def predict_lstm(request: PredictRequest) -> PredictResponse:
    """LSTM RUL prediction (alternative endpoint).

    Same preprocessing as /predict. Raises HTTP 503 if the LSTM checkpoint
    was not loaded (PyTorch absent or .pt file missing).
    """
    store = get_model_store()

    if store.lstm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "LSTM model is not available. Either PyTorch is not installed "
                "or models/lstm_multi_best.pt was not found."
            ),
        )

    _, window = preprocess(
        cycles=request.cycles,
        subset=request.subset,
        scalers=store.scalers,
        kmeans_models=store.kmeans,
    )

    import torch  # noqa: PLC0415 — only reached when lstm is loaded, so torch exists

    tensor = build_lstm_tensor(window)
    with torch.no_grad():
        raw_pred = float(store.lstm(tensor).item())

    predicted_rul = max(0.0, raw_pred)

    log.info("LSTM | subset=%s | predicted_rul=%.2f", request.subset, predicted_rul)

    return PredictResponse(
        predicted_rul=predicted_rul,
        subset=request.subset,
        model_used="lstm",
    )
