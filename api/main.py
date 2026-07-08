"""FastAPI serving layer — /predict and /health.

Loads the model trained by `python -m src.train`. Run from the project root:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import config
from src.features import build_feature_row

_MODEL = None
_MODEL_SOURCE = "none"


def _load_model():
    """Load the serving model from the MLflow registry alias; fall back to a local joblib file."""
    global _MODEL, _MODEL_SOURCE
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL}@{config.MODEL_ALIAS}"
    try:
        _MODEL = mlflow.sklearn.load_model(uri)
        _MODEL_SOURCE = f"registry ({uri})"
        print(f"Loaded model from {uri}")
        return
    except Exception as e:  # registry empty / unavailable — degrade gracefully
        print(f"Registry load failed ({type(e).__name__}); trying local joblib fallback.")
    path = config.MODELS_DIR / "model.joblib"
    if path.exists():
        _MODEL = joblib.load(path)
        _MODEL_SOURCE = "local joblib fallback"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Steam Player-Count Forecaster", version="0.2.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    recent_players: list[float] = Field(
        ..., description="Last >=7 daily peak concurrent players, oldest -> newest"
    )
    day_of_week: int = Field(..., ge=0, le=6, description="Weekday of the day being predicted (0=Mon..6=Sun)")
    days_since_release: int = Field(..., ge=0)
    genre: str = Field(..., description=f"one of: {', '.join(config.GENRES)}")


class PredictResponse(BaseModel):
    predicted_players: int
    model_version: str = "0.1.0"


@app.get("/", include_in_schema=False)
def root():
    """Redirect the base URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None, "model_source": _MODEL_SOURCE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run `python -m src.train` first.")
    try:
        X = build_feature_row(req.recent_players, req.day_of_week, req.days_since_release, req.genre)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    pred = float(_MODEL.predict(X)[0])
    return PredictResponse(predicted_players=max(0, round(pred)))
