"""
backend/model_loader.py

Loads the fine-tuned SentimentModel once and exposes it as a module-level
singleton. FastAPI's lifespan handler (in main.py) calls `get_model()` at
startup to trigger the load, then every request handler calls `get_model()`
again — which returns the already-loaded instance instantly.

This avoids the most common ML-serving mistake: reloading the model on
every request (which would make each call take seconds instead of
milliseconds).
"""

import os
import sys

# Allow importing from src/ regardless of where uvicorn is launched from
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from predict import SentimentModel  # noqa: E402


# Default model path — resolves to <project_root>/model regardless of
# the working directory uvicorn was launched from.
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "model")

# Allow overriding via environment variable — used in deployment (Render)
# where the model may be downloaded to a different path at build time.
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# If set, and MODEL_PATH is missing/empty, download the model from this
# Hugging Face Hub repo at startup instead of failing. This is what makes
# deployment work without committing ~250MB of weights to Git.
# Set via scripts/push_to_hub.py — see that file's docstring.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")  # e.g. "your-username/distilbert-review-sentiment"
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the HF repo is private


_model: SentimentModel | None = None


def _download_from_hub(repo_id: str, local_dir: str) -> None:
    """
    Download all files from a Hugging Face Hub model repo into local_dir.
    Used as a fallback when MODEL_PATH is empty — this is the path that
    runs on a fresh Render deployment.
    """
    from huggingface_hub import snapshot_download

    print(f"[model_loader] Local model not found. Downloading from HF Hub: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=HF_TOKEN,  # None is fine for public repos
    )
    print(f"[model_loader] Download complete: {local_dir}")


def load_model() -> SentimentModel:
    """
    Load the model from MODEL_PATH and store it as the module-level
    singleton. Called once during FastAPI startup (see main.py lifespan).

    If MODEL_PATH is empty/missing and HF_MODEL_REPO is set, downloads
    the model from Hugging Face Hub first. This is the path used in
    production (Render) — locally you typically already have the model
    downloaded from Kaggle, so the HF download is skipped entirely.

    Raises:
        FileNotFoundError if MODEL_PATH does not exist or is empty AND
        no HF_MODEL_REPO is configured — this is intentional. The API
        should fail fast at startup rather than start successfully and
        then 500 on every request.
    """
    global _model

    print("=" * 60)
    print(f"[DEBUG] MODEL_PATH = {MODEL_PATH}")
    print(f"[DEBUG] HF_MODEL_REPO = {HF_MODEL_REPO}")
    print(f"[DEBUG] MODEL_PATH exists = {os.path.isdir(MODEL_PATH)}")

    if os.path.isdir(MODEL_PATH):
        print(f"[DEBUG] MODEL_PATH contents = {os.listdir(MODEL_PATH)}")

    print("=" * 60)

    model_missing = not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH)

    if model_missing and HF_MODEL_REPO:
        os.makedirs(MODEL_PATH, exist_ok=True)
        _download_from_hub(HF_MODEL_REPO, MODEL_PATH)
        model_missing = not os.listdir(MODEL_PATH)

    if model_missing:
        raise FileNotFoundError(
            f"Model directory not found or empty: '{MODEL_PATH}'.\n"
            f"Expected files: config.json, model.safetensors (or pytorch_model.bin), "
            f"tokenizer files, label_config.json.\n"
            f"Either:\n"
            f"  (a) Download the trained model from Kaggle (see notebooks/02_train.ipynb, "
            f"Cell 13) and place its contents in '{MODEL_PATH}', or\n"
            f"  (b) Set the HF_MODEL_REPO environment variable to a Hugging Face Hub "
            f"repo containing the model (see scripts/push_to_hub.py)."
        )

    print(f"[model_loader] Loading model from: {MODEL_PATH}")
    _model = SentimentModel(MODEL_PATH)
    print("[model_loader] Model ready.")
    return _model


def get_model() -> SentimentModel:
    """
    Return the loaded model singleton.

    Raises:
        RuntimeError if called before load_model() — indicates the
        FastAPI lifespan startup did not run, which should not happen
        in normal operation.
    """
    if _model is None:
        raise RuntimeError(
            "Model has not been loaded yet. "
            "This should be called only after the FastAPI startup "
            "lifespan has run load_model()."
        )
    return _model


def is_loaded() -> bool:
    """Used by /health to report status without raising if not yet loaded."""
    return _model is not None