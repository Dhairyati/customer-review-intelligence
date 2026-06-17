"""
backend/schemas.py

Pydantic models defining the request and response shapes for every
API endpoint. Kept separate from main.py so the contract is easy to
review independently and reuse in tests.

Endpoints covered:
  POST /predict
  POST /predict/batch
  GET  /analytics
  GET  /health
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Shared building blocks ──────────────────────────────────────────────

class SentimentScores(BaseModel):
    """Full softmax distribution over all 3 classes."""
    Positive: float
    Neutral: float
    Negative: float


class PredictionResult(BaseModel):
    """
    The shape of a single prediction — returned by /predict and as
    each item in the /predict/batch "results" list.
    """
    text: str
    label: str = Field(..., description="Positive, Neutral, or Negative")
    confidence: float = Field(..., description="Top class probability (0-1)")
    scores: SentimentScores
    uncertain: bool = Field(
        ..., description="True if confidence is below the uncertainty threshold"
    )


# ── POST /predict ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Review text — anywhere from a couple words to a full paragraph",
        examples=["The food was amazing but the service was painfully slow."],
    )


# PredictResponse is identical to PredictionResult — aliased for clarity
# in the OpenAPI docs.
class PredictResponse(PredictionResult):
    pass


# ── POST /predict/batch ──────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="List of review texts to score in one request (1–2000 items). "
                    "Each item must be between 1 and 10 000 characters.",
        examples=[[
            "Great product, fast shipping.",
            "Broke after two days, very disappointed.",
            "Decent quality for the price.",
        ]],
    )


class BatchSummary(BaseModel):
    """Aggregated statistics for one batch run — also persisted to analytics_history.json."""
    total: int
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    uncertain_count: int
    uncertain_pct: float
    avg_confidence: float
    most_positive: Optional[PredictionResult] = None
    most_negative: Optional[PredictionResult] = None
    most_uncertain: Optional[PredictionResult] = Field(
        default=None,
        description="The prediction with the lowest top-class confidence score — "
                    "the review the model is least sure about.",
    )


class BatchPredictResponse(BaseModel):
    results: List[PredictionResult]
    summary: BatchSummary


# ── GET /analytics ───────────────────────────────────────────────────────

class AnalyticsRun(BaseModel):
    """One row of analytics_history.json — a single batch run's summary + metadata."""
    run_id: str
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the run")
    total_reviews: int
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    uncertain_count: int
    uncertain_pct: float
    avg_confidence: float
    most_positive_text: Optional[str] = None
    most_negative_text: Optional[str] = None
    most_uncertain_text: Optional[str] = Field(
        default=None,
        description="First 200 chars of the review with the lowest confidence score.",
    )
    most_uncertain_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score of the most uncertain review (0–1).",
    )


class AnalyticsResponse(BaseModel):
    """Full analytics history, most recent run first."""
    history: List[AnalyticsRun]
    latest: Optional[AnalyticsRun] = None


# ── GET /health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    labels: List[str]
    uncertainty_threshold: float
