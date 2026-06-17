"""
scripts/push_to_hub.py

One-time utility: pushes the trained model in `model/` to a Hugging Face
Hub repository, so the deployed backend (Render) can download it at
container build time instead of needing it committed to Git.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login          (paste a token with WRITE access)
    3. python scripts/push_to_hub.py --repo-id your-username/distilbert-review-sentiment

The repo is created as PRIVATE by default — pass --public to make it
publicly downloadable (required if Render should pull it without a token).
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


def push_model(model_path: str, repo_id: str, private: bool = True):
    if not os.path.isdir(model_path) or not os.listdir(model_path):
        raise FileNotFoundError(
            f"Model directory '{model_path}' is missing or empty. "
            f"Train the model first (notebooks/02_train.ipynb) and download "
            f"its contents into '{model_path}'."
        )

    required_files = ["config.json", "tokenizer_config.json", "label_config.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        print(f"WARNING: expected files not found in '{model_path}': {missing}")
        print("Proceeding anyway — verify the model directory is complete.")

    print(f"Creating/verifying repo: {repo_id} (private={private})")
    create_repo(repo_id, exist_ok=True, private=private, repo_type="model")

    print(f"Uploading contents of '{model_path}' to '{repo_id}'...")
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned DistilBERT sentiment model",
    )

    print(f"\nDone. Model available at: https://huggingface.co/{repo_id}")
    print(f"\nNext step: set these environment variables in your Render dashboard:")
    print(f"  HF_MODEL_REPO = {repo_id}")
    if private:
        print(f"  HF_TOKEN      = <your huggingface token with read access>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push the trained model to Hugging Face Hub.")
    parser.add_argument("--model-path", default="model", help="Local model directory (default: model)")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/distilbert-review-sentiment")
    parser.add_argument("--public", action="store_true", help="Make the repo public (default: private)")
    args = parser.parse_args()

    push_model(args.model_path, args.repo_id, private=not args.public)
