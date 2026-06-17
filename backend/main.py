"""
backend/main.py

FastAPI application for the Customer Review Intelligence System.

Endpoints:
    POST /predict        - score a single review
    POST /predict/batch   - score a list of reviews, return summary stats
    GET  /analytics       - return saved batch-run history
    GET  /health          - model/server status check

Also mounts the static frontend (frontend/) so the entire app is served
from a single origin — no CORS configuration needed.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Then open:
    http://localhost:8000/           -> frontend
    http://localhost:8000/docs       -> Swagger UI
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import model_loader
from .schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchSummary,
    PredictionResult,
    AnalyticsResponse,
    AnalyticsRun,
    HealthResponse,
)


# ── Paths ─────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_ANALYTICS_FILE = os.path.join(_DATA_DIR, "analytics_history.json")

MAX_HISTORY_RUNS = 100  # cap analytics_history.json size


# ── Lifespan: load model once at startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    yield
    # No teardown needed — model is released when the process exits


app = FastAPI(
    title="Customer Review Intelligence API",
    description=(
        "Serves a fine-tuned **DistilBERT** 3-class sentiment model "
        "(Positive / Neutral / Negative) with confidence scores, a full "
        "probability distribution, and an uncertainty flag for low-confidence "
        "predictions (default threshold: 0.65).\n\n"
        "**Model:** `distilbert-base-uncased` fine-tuned on product reviews  \n"
        "**Metric:** Macro F1 ≈ 0.89 on held-out test set  \n"
        "**Classes:** Positive · Neutral · Negative"
    ),
    version="1.0.0",
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)


# ── Analytics persistence helpers ───────────────────────────────────────

def _load_analytics_history() -> list[dict]:
    """Load analytics_history.json, returning [] if missing or invalid."""
    if not os.path.exists(_ANALYTICS_FILE):
        return []
    try:
        with open(_ANALYTICS_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_analytics_run(run: dict) -> None:
    """Append a run to analytics_history.json, capping total stored runs."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    history = _load_analytics_history()
    history.insert(0, run)  # most recent first
    history = history[:MAX_HISTORY_RUNS]
    with open(_ANALYTICS_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── Batch summary builder ───────────────────────────────────────────────

def _build_summary(results: list[dict]) -> dict:
    """
    Compute aggregated statistics from a list of prediction results
    (as returned by SentimentModel.predict_batch).
    """
    total = len(results)

    positive = [r for r in results if r["label"] == "Positive"]
    neutral  = [r for r in results if r["label"] == "Neutral"]
    negative = [r for r in results if r["label"] == "Negative"]
    uncertain = [r for r in results if r["uncertain"]]

    avg_confidence = sum(r["confidence"] for r in results) / total if total else 0.0

    most_positive = (
        max(positive, key=lambda r: r["scores"]["Positive"]) if positive else None
    )
    most_negative = (
        max(negative, key=lambda r: r["scores"]["Negative"]) if negative else None
    )
    # Most uncertain = lowest top-class confidence across ALL predictions
    most_uncertain = (
        min(results, key=lambda r: r["confidence"]) if results else None
    )

    return {
        "total": total,
        "positive_count": len(positive),
        "neutral_count": len(neutral),
        "negative_count": len(negative),
        "positive_pct": round(len(positive) / total * 100, 2) if total else 0.0,
        "neutral_pct": round(len(neutral) / total * 100, 2) if total else 0.0,
        "negative_pct": round(len(negative) / total * 100, 2) if total else 0.0,
        "uncertain_count": len(uncertain),
        "uncertain_pct": round(len(uncertain) / total * 100, 2) if total else 0.0,
        "avg_confidence": round(avg_confidence, 4),
        "most_positive": most_positive,
        "most_negative": most_negative,
        "most_uncertain": most_uncertain,
    }


# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Model & server health check",
    description=(
        "Returns the current status of the server and the loaded sentiment model. "
        "Call this on page load to detect cold-start delays — on Render's free tier "
        "the model can take 30–60 seconds to load after the dyno spins up from idle. "
        "Returns **503** while the model is still loading."
    ),
    response_description="Model status, device, label list, and uncertainty threshold.",
)
def health():
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading.")

    model = model_loader.get_model()
    return HealthResponse(
        status="ok",
        model="distilbert-sentiment",
        device=model.device,
        labels=[model.id2label[i] for i in sorted(model.id2label)],
        uncertainty_threshold=model.uncertainty_threshold,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Score a single review",
    description=(
        "Runs a single review text through the fine-tuned DistilBERT model and "
        "returns the predicted sentiment label, the top-class confidence score, "
        "the full three-class probability distribution, and an `uncertain` flag.\n\n"
        "The `uncertain` flag is set when the top-class confidence is **below the "
        "configured threshold** (default 0.65). Uncertain predictions should be "
        "reviewed manually before acting on them.\n\n"
        "- Input length: 1–10 000 characters  \n"
        "- Tokenisation: DistilBERT WordPiece (max 512 tokens; longer texts are truncated)"
    ),
    response_description=(
        "Predicted label (Positive/Neutral/Negative), confidence, full score "
        "distribution, and uncertainty flag."
    ),
)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Review text must not be empty or consist only of whitespace.",
        )

    model = model_loader.get_model()

    try:
        result = model.predict(text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    summary="Score a list of reviews",
    description=(
        "Scores up to **2 000 reviews** in a single request. Returns per-review "
        "results plus aggregated summary statistics including class distribution, "
        "uncertainty rate, average confidence, and the most positive, most negative, "
        "and most uncertain reviews.\n\n"
        "The summary is also appended to `data/analytics_history.json` so the "
        "`/analytics` endpoint can show trends across multiple batch runs.\n\n"
        "**Empty or whitespace-only reviews** in the list are rejected with a 422 "
        "so callers know to clean their data before submitting."
    ),
    response_description=(
        "Per-review predictions plus a batch summary with class counts, percentages, "
        "average confidence, and the three highlight reviews."
    ),
)
def predict_batch(request: BatchPredictRequest):
    # Validate: no empty strings in the list
    empty_indices = [
        i for i, r in enumerate(request.reviews) if not r.strip()
    ]
    if empty_indices:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Reviews at indices {empty_indices[:10]} are empty or whitespace-only. "
                "Remove blank lines before submitting."
            ),
        )

    model = model_loader.get_model()
    results = model.predict_batch(request.reviews)
    summary = _build_summary(results)

    # Persist this run to analytics history
    run_record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_reviews": summary["total"],
        "positive_count": summary["positive_count"],
        "neutral_count": summary["neutral_count"],
        "negative_count": summary["negative_count"],
        "positive_pct": summary["positive_pct"],
        "neutral_pct": summary["neutral_pct"],
        "negative_pct": summary["negative_pct"],
        "uncertain_count": summary["uncertain_count"],
        "uncertain_pct": summary["uncertain_pct"],
        "avg_confidence": summary["avg_confidence"],
        "most_positive_text": (
            summary["most_positive"]["text"][:200] if summary["most_positive"] else None
        ),
        "most_negative_text": (
            summary["most_negative"]["text"][:200] if summary["most_negative"] else None
        ),
        "most_uncertain_text": (
            summary["most_uncertain"]["text"][:200] if summary["most_uncertain"] else None
        ),
        "most_uncertain_confidence": (
            summary["most_uncertain"]["confidence"] if summary["most_uncertain"] else None
        ),
    }
    _save_analytics_run(run_record)

    return BatchPredictResponse(
        results=[PredictionResult(**r) for r in results],
        summary=BatchSummary(**summary),
    )


@app.get(
    "/analytics",
    response_model=AnalyticsResponse,
    tags=["analytics"],
    summary="Batch run history",
    description=(
        "Returns the history of all batch prediction runs, most recent first. "
        "Each entry includes the run timestamp, review counts, class distribution "
        "percentages, uncertainty rate, average confidence, and the text excerpts "
        "for the most positive, most negative, and most uncertain reviews.\n\n"
        "Backed by `data/analytics_history.json`. Capped at the last "
        f"**{MAX_HISTORY_RUNS} runs**.\n\n"
        "> **Note:** On Render's free tier the filesystem is ephemeral — history "
        "resets on every redeploy. For production persistence, replace the JSON "
        "file with a small database table."
    ),
    response_description=(
        "Full run history (most recent first) and a convenience `latest` field "
        "pointing to the most recent run."
    ),
)
def get_analytics():
    history = _load_analytics_history()
    latest = history[0] if history else None

    return AnalyticsResponse(
        history=[AnalyticsRun(**run) for run in history],
        latest=AnalyticsRun(**latest) if latest else None,
    )


# ══════════════════════════════════════════════════════════════════════
#  STATIC FRONTEND
# ══════════════════════════════════════════════════════════════════════
# Mounted last so it doesn't shadow the API routes above.
# Serves frontend/index.html, batch.html, analytics.html, style.css, app.js

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def root_placeholder():
        return {
            "message": "Frontend not found. API is running — see /docs for the API.",
            "expected_path": _FRONTEND_DIR,
        }