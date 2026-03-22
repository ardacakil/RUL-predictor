from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

WINDOW_SIZE = 30
USEFUL_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
SETTING_COLS = ["setting1", "setting2", "setting3"]


class CycleReading(BaseModel):
    """One cycle: operational settings + the 14 useful sensors only.
    Flat/uninformative sensors (s1,s5,s6,s10,s16,s18,s19) are excluded —
    callers should not send them."""

    setting1: float
    setting2: float
    setting3: float

    s2:  float
    s3:  float
    s4:  float
    s7:  float
    s8:  float
    s9:  float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s17: float
    s20: float
    s21: float


class PredictRequest(BaseModel):
    """POST /predict  &  POST /predict/lstm

    `cycles` must be exactly WINDOW_SIZE (30) rows in chronological order,
    oldest first."""

    subset: Literal["FD001", "FD002", "FD003", "FD004"] = Field(
        ...,
        description="CMAPSS subset the engine belongs to. Controls which "
                    "KMeans model and scalers are applied.",
    )
    cycles: list[CycleReading] = Field(
        ...,
        description=f"Exactly {WINDOW_SIZE} consecutive sensor cycles, oldest first.",
    )

    @model_validator(mode="after")
    def _check_window_length(self) -> "PredictRequest":
        if len(self.cycles) != WINDOW_SIZE:
            raise ValueError(
                f"'cycles' must contain exactly {WINDOW_SIZE} rows, "
                f"got {len(self.cycles)}."
            )
        return self


class PredictResponse(BaseModel):
    predicted_rul: float = Field(
        ...,
        ge=0.0,
        description="Predicted Remaining Useful Life in cycles (clipped to ≥ 0).",
    )
    subset: str
    model_used: Literal["xgboost", "lstm"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    models_loaded: list[str]
