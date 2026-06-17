"""
src/predict.py

Core inference module for the Customer Review Intelligence System.

This module provides model loading and sentiment prediction utilities and is
used by the evaluation pipeline and FastAPI application.

Design notes:
  - The model is loaded once and reused across requests.
  - Predictions include probabilities for all sentiment classes
    (Positive, Neutral, Negative).
  - "Uncertain" is a post-prediction flag rather than a training label.
    If the highest class probability falls below the configured
    uncertainty threshold, the prediction is flagged as uncertain while
    still returning the most likely sentiment label.
"""
import json
import os
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Defaults ──────────────────────────────────────────────────────────────
# These are overridden by model/label_config.json if present.
# Kept here as fallbacks so this module still works if that file is missing.
DEFAULT_MAX_LENGTH = 256
DEFAULT_UNCERTAINTY_THRESHOLD = 0.65
DEFAULT_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


class SentimentModel:
    """
    Wraps the fine-tuned DistilBERT model and tokenizer for inference.

    Usage:
        model = SentimentModel("model/")
        result = model.predict("The food was great but service was slow.")
        results = model.predict_batch(["text 1", "text 2", ...])
    """

    def __init__(self, model_path: str = "model"):
        """
        Load model, tokenizer, and label config from model_path.

        Args:
            model_path: directory containing pytorch_model.bin / model.safetensors,
                        config.json, tokenizer files, and optionally label_config.json
        """
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found: '{model_path}'.\n"
                f"Download the trained model from Kaggle (Phase 1b output) "
                f"and place its contents in this directory."
            )

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load label config (saved during training) ──────────────────
        label_config_path = os.path.join(model_path, "label_config.json")
        if os.path.exists(label_config_path):
            with open(label_config_path, "r") as f:
                label_config = json.load(f)
            self.max_length = label_config.get("max_length", DEFAULT_MAX_LENGTH)
            self.uncertainty_threshold = label_config.get(
                "uncertainty_threshold", DEFAULT_UNCERTAINTY_THRESHOLD
            )
            # JSON keys are always strings — convert id2label keys back to int
            raw_id2label = label_config.get("id2label", DEFAULT_ID2LABEL)
            self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            print(
                f"WARNING: label_config.json not found in '{model_path}'. "
                f"Using defaults (max_length={DEFAULT_MAX_LENGTH}, "
                f"threshold={DEFAULT_UNCERTAINTY_THRESHOLD})."
            )
            self.max_length = DEFAULT_MAX_LENGTH
            self.uncertainty_threshold = DEFAULT_UNCERTAINTY_THRESHOLD
            self.id2label = DEFAULT_ID2LABEL

        # ── Load tokenizer and model ────────────────────────────────────
        print(f"Loading tokenizer from '{model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"Loading model from '{model_path}' onto {self.device}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # inference mode — disables dropout

        # If the model's own config has id2label (set during training via
        # from_pretrained(..., id2label=...)), prefer that — it's guaranteed
        # to match the model's output head ordering.
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = {
                int(k): v for k, v in self.model.config.id2label.items()
            }

        print(
            f"Model loaded successfully. "
            f"Labels: {self.id2label} | "
            f"max_length={self.max_length} | "
            f"uncertainty_threshold={self.uncertainty_threshold} | "
            f"device={self.device}"
        )

    # ──────────────────────────────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run sentiment prediction on a single review.

        Args:
            text: raw review text (any length — short phrases to long paragraphs)

        Returns:
            {
                "text": <original text>,
                "label": "Positive" | "Neutral" | "Negative",
                "confidence": float (0-1, the top score),
                "scores": {
                    "Positive": float,
                    "Neutral": float,
                    "Negative": float
                },
                "uncertain": bool
            }
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Tokenize — truncation handles long inputs, padding=True handles
        # the single-example case cleanly
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        return self._format_result(text, probs)

    # ──────────────────────────────────────────────────────────────────
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run sentiment prediction on a list of reviews efficiently.

        Texts are tokenized and processed in a single batched forward pass
        (with internal chunking for very large lists), which is
        significantly faster than calling predict() in a loop.

        Args:
            texts: list of raw review strings

        Returns:
            list of result dicts, same shape as predict()'s return value,
            one per input text, in the same order.
        """
        if not texts:
            return []

        # Filter and track empty strings — tokenizer chokes on empty input,
        # so we substitute a placeholder and let the model score it normally
        # rather than crashing on a malformed CSV row.
        cleaned_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        cleaned_texts = [t if t else " " for t in cleaned_texts]

        results = []
        chunk_size = 32  # process in chunks to bound memory on CPU

        for i in range(0, len(cleaned_texts), chunk_size):
            chunk = cleaned_texts[i : i + chunk_size]

            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs_batch = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for original_text, probs in zip(texts[i : i + chunk_size], probs_batch):
                results.append(self._format_result(original_text, probs))

        return results

    # ──────────────────────────────────────────────────────────────────
    def _format_result(self, text: str, probs) -> Dict[str, Any]:
        """
        Convert raw softmax probabilities into the standard API response shape.

        This is the single place where the uncertainty flag logic lives —
        any change to the threshold or rounding behavior happens here only.
        """
        scores = {self.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

        top_idx = int(probs.argmax())
        top_label = self.id2label[top_idx]
        top_confidence = round(float(probs[top_idx]), 4)

        return {
            "text": text,
            "label": top_label,
            "confidence": top_confidence,
            "scores": scores,
            "uncertain": top_confidence < self.uncertainty_threshold,
        }


# ── Module-level convenience functions ──────────────────────────────────
# These allow simple usage without manually instantiating SentimentModel,
# e.g. for quick scripts or notebook cells. The FastAPI backend uses the
# class directly (via model_loader.py) to avoid reloading on every call.

_default_model = None


def load_model(model_path: str = "model") -> SentimentModel:
    """Load and return a SentimentModel instance. Call once, reuse the result."""
    return SentimentModel(model_path)


def predict(text: str, model: SentimentModel = None, model_path: str = "model") -> Dict[str, Any]:
    """
    Convenience wrapper. If `model` is not provided, loads (and caches)
    a default instance — useful for one-off scripts, not for the API server.
    """
    global _default_model
    if model is None:
        if _default_model is None:
            _default_model = load_model(model_path)
        model = _default_model
    return model.predict(text)


# ── CLI smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick manual test:
        python src/predict.py

    Loads the model from ../model (relative to src/) or ./model
    (if run from project root) and runs it on a few example reviews.
    """
    # Try common relative paths so this works whether run from
    # project root or from inside src/
    candidate_paths = ["model", "../model"]
    model_dir = next((p for p in candidate_paths if os.path.isdir(p)), "model")

    print(f"Loading model from: {model_dir}\n")
    sentiment_model = load_model(model_dir)

    test_reviews = [
        "Absolutely amazing experience, will come back again!",
        "Terrible. The staff were rude and the food was cold.",
        "It was fine, nothing special.",
        "Good",
        "meh",
        (
            "We went here for our anniversary dinner. The ambiance was lovely "
            "and the appetizers were delicious, but our main courses took over "
            "an hour to arrive and were lukewarm when they finally did. The "
            "waiter apologized and comped our dessert, which was a nice gesture. "
            "Overall a mixed experience — great start, rough middle, decent end."
        ),
    ]

    print(f"{'Review':<70} {'Label':<10} {'Conf':<7} {'Uncertain'}")
    print("-" * 100)
    for review in test_reviews:
        result = sentiment_model.predict(review)
        short = review[:65] + "..." if len(review) > 68 else review
        print(
            f"{short:<70} {result['label']:<10} "
            f"{result['confidence']:<7.3f} {result['uncertain']}"
        )
        print(f"    scores: {result['scores']}")