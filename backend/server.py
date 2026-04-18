"""
Fake Review Detector — FastAPI Backend
Serves the trained TF-IDF + Logistic Regression model over HTTP.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent / "ML Project 2026"
MODEL_PATH = ROOT / "models" / "fake_review_detector.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

# ── Patterns ─────────────────────────────────────────────────────────────────
PROMO_PATTERN = re.compile(
    r"\b(best|amazing|perfect|must buy|buy now|highly recommend|life[- ]changing)\b",
    re.IGNORECASE,
)

# ── Model loading ────────────────────────────────────────────────────────────
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    yield
    model = None


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake Review Detector API",
    version="1.0.0",
    description="Classifies product reviews as genuine or computer-generated.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # browser extension can call from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper functions (ported from predict_review.py) ─────────────────────────
def confidence_band(probability: float) -> str:
    if probability >= 0.85 or probability <= 0.15:
        return "high"
    if probability >= 0.65 or probability <= 0.35:
        return "moderate"
    return "uncertain"


def human_label(probability: float) -> str:
    if probability >= 0.6:
        return "Likely computer-generated / fake"
    if probability <= 0.4:
        return "Likely original / genuine"
    return "Borderline / uncertain"


def explain_text(mdl, text: str, top_n: int = 8) -> list[str]:
    vectorizer = mdl.named_steps["tfidf"]
    classifier = mdl.named_steps["clf"]
    features = vectorizer.transform([text])
    coefs = classifier.coef_[0]
    nz = features.nonzero()[1]
    if len(nz) == 0:
        return []
    names = vectorizer.get_feature_names_out()
    scored = sorted(
        ((names[i], float(features[0, i] * coefs[i])) for i in nz),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [f"{term} ({value:+.3f})" for term, value in scored[:top_n]]


def suspicious_flags(text: str) -> list[str]:
    flags: list[str] = []
    words = text.split()
    upper_words = [w for w in words if len(w) > 2 and w.isupper()]
    if text.count("!") >= 3:
        flags.append("heavy exclamation use")
    if len(upper_words) >= 2:
        flags.append("multiple all-caps words")
    if PROMO_PATTERN.search(text):
        flags.append("contains promotional superlatives")
    if len(words) <= 4:
        flags.append("very short review")
    return flags


def predict_single(text: str) -> dict[str, Any]:
    probability = float(model.predict_proba(pd.Series([text]))[0, 1])
    return {
        "label": human_label(probability),
        "fake_probability": round(probability, 4),
        "confidence_band": confidence_band(probability),
        "top_terms": explain_text(model, text),
        "flags": suspicious_flags(text),
    }


# ── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str
    fake_probability: float
    confidence_band: str
    top_terms: list[str]
    flags: list[str]


class BatchPredictRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=50)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    accuracy = None
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as f:
            accuracy = json.load(f).get("test_accuracy")
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_accuracy=accuracy,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(503, "Model is not loaded yet.")
    result = predict_single(req.text)
    return PredictResponse(**result)


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if model is None:
        raise HTTPException(503, "Model is not loaded yet.")
    results = [PredictResponse(**predict_single(t)) for t in req.reviews]
    return BatchPredictResponse(results=results)


# ── Run with: uvicorn server:app --reload ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
