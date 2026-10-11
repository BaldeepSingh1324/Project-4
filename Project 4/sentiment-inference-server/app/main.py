"""Sentiment-analysis inference server.

Wraps a pre-trained Hugging Face model in a FastAPI service. The model is
loaded once at start-up and shared across requests.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference-server")


def load_model():
    """Download (first run only) and load the model into memory."""
    log.info("Loading model %s", MODEL_NAME)
    return pipeline("sentiment-analysis", model=MODEL_NAME, truncation=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    log.info("Model ready")
    yield
    app.state.model = None


app = FastAPI(title="Sentiment Inference Server", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(Prediction):
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


def run_inference(texts: list[str]) -> tuple[list[Prediction], float]:
    """Run the model on a list of texts and time it."""
    start = time.perf_counter()
    results = app.state.model(texts)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return [Prediction(label=r["label"], score=round(r["score"], 4)) for r in results], elapsed_ms


@app.get("/health")
def health():
    """Liveness/readiness probe: reports whether the model is loaded."""
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds, ms = run_inference([req.text])
    return PredictResponse(**preds[0].model_dump(), latency_ms=round(ms, 2))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_BATCH_SIZE})")
    if any(not t.strip() for t in req.texts):
        raise HTTPException(status_code=422, detail="Texts must be non-empty")
    preds, ms = run_inference(req.texts)
    return BatchResponse(predictions=preds, latency_ms=round(ms, 2))
