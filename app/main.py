"""FastAPI service exposing the credit risk model."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from . import model as m
from .schemas import Applicant, BatchRequest, BatchResponse, ModelInfo, Prediction

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = m.load_or_train()
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    version=m.MODEL_VERSION,
    description=(
        "Scores loan applications for probability of default, assigns a risk band and "
        "recommendation, and explains the main risk drivers. Educational project trained on the "
        "public German Credit dataset — not for real lending decisions."
    ),
    lifespan=lifespan,
)


def _frame(applicants: list[Applicant]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in applicants])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    tm = state["model"]
    return ModelInfo(
        model_version=tm.version,
        algorithm="Logistic regression (class-balanced) with one-hot encoding and scaling",
        trained_on_rows=tm.rows,
        features=m.FEATURES,
        excluded_features=m.EXCLUDED,
        risk_band_thresholds=m.THRESHOLDS,
        **tm.metrics,
    )


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> Prediction:
    return Prediction(**m.score(state["model"], _frame([applicant]))[0])


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=[Prediction(**r) for r in m.score(state["model"], _frame(req.applicants))])
