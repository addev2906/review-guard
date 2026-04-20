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
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# Load default .env file in the backend
load_dotenv()

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
model_artifact: dict[str, Any] | None = None


def unpack_model_artifact(raw: Any) -> dict[str, Any]:
    """Handle both legacy (raw pipeline) and v3 (dict) artifacts."""
    if isinstance(raw, dict) and "models" in raw:
        return raw
    return {
        "format_version": 1,
        "selected_model_name": "tfidf_logreg",
        "models": {"tfidf_logreg": raw},
        "ensembles": {},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    model_artifact = unpack_model_artifact(joblib.load(MODEL_PATH))
    print(f"✓ Model loaded from {MODEL_PATH}")
    yield
    model_artifact = None


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
    nz = features.nonzero()[1]
    if len(nz) == 0:
        return []
    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
    elif hasattr(classifier, "feature_importances_"):
        weights = classifier.feature_importances_
    else:
        return []
    names = vectorizer.get_feature_names_out()
    scored = sorted(
        ((names[i], float(features[0, i] * weights[i])) for i in nz),
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
    models = model_artifact["models"]
    selected = model_artifact.get("selected_model_name", "tfidf_logreg")
    ensembles = model_artifact.get("ensembles", {})
    ensemble_cfg = ensembles.get(selected, {})

    if ensemble_cfg.get("enabled"):
        weights = ensemble_cfg.get("weights", {})
        probability = sum(
            float(weights.get(name, 0)) * float(models[name].predict_proba(pd.Series([text]))[0, 1])
            for name in ensemble_cfg.get("base_models", []) if name in models
        )
        explanation_model = models.get("tfidf_logreg")
    else:
        pipeline = models.get(selected) or models["tfidf_logreg"]
        probability = float(pipeline.predict_proba(pd.Series([text]))[0, 1])
        explanation_model = pipeline

    return {
        "label": human_label(probability),
        "fake_probability": round(probability, 4),
        "confidence_band": confidence_band(probability),
        "top_terms": explain_text(explanation_model, text) if explanation_model else [],
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

class ExplainRequest(BaseModel):
    review_text: str
    verdict: str
    nvidia_key: str | None = None

class ExplainResponse(BaseModel):
    explanation: str


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
            metrics = json.load(f)
            if "test_accuracy" in metrics:
                accuracy = metrics["test_accuracy"]
            elif "models" in metrics:
                sel = metrics.get("selected_model", "tfidf_logreg")
                accuracy = metrics["models"].get(sel, {}).get("test", {}).get("accuracy")
    return HealthResponse(
        status="ok",
        model_loaded=model_artifact is not None,
        model_accuracy=accuracy,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model_artifact is None:
        raise HTTPException(503, "Model is not loaded yet.")
    result = predict_single(req.text)
    return PredictResponse(**result)


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if model_artifact is None:
        raise HTTPException(503, "Model is not loaded yet.")
    results = [PredictResponse(**predict_single(t)) for t in req.reviews]
    return BatchPredictResponse(results=results)

@app.post("/api/explain", response_model=ExplainResponse)
def explain_review(req: ExplainRequest) -> ExplainResponse:
    # First try token from the request (extension), then from .env
    token = req.nvidia_key or os.environ.get("NVIDIA_API_KEY")
    if not token:
        raise HTTPException(400, "NVIDIA API Key is missing. Add NVIDIA_API_KEY=... to your backend/.env file.")
        
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=token
        )

        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert fraud analyst. Provide a short, concise explanation on why the following review is considered {req.verdict.upper()}. If FAKE, mention lack of specifics, vague language, repetitive praise, etc. If GENUINE, mention authentic details, balanced tone, etc."
                },
                {
                    "role": "user",
                    "content": f"Our ML model has classified this review as: {req.verdict.upper()}\\n\\nReview text: '{req.review_text}'"
                }
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024},
            stream=True
        )

        full_response = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            
            # Extract content answer
            if getattr(chunk.choices[0].delta, "content", None) is not None:
                full_response += chunk.choices[0].delta.content

        return ExplainResponse(explanation=full_response.strip())

    except Exception as e:
        status_code = 500
        detail = f"NVIDIA API Error: {str(e)}"
        
        # Check if it resembles an unauthorized error
        if "401" in str(e):
            status_code = 401
            detail = "Invalid NVIDIA API Key."
            
        raise HTTPException(status_code, detail)



# ── Run with: uvicorn server:app --reload ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
