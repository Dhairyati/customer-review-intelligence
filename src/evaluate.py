"""
src/evaluate.py

Evaluation script for the Customer Review Intelligence System.

Reproduces test-set metrics using the trained model and generates
evaluation artifacts such as classification reports, confusion matrices,
and error-analysis examples.

Useful for:
  - Validating model performance on the held-out test set
  - Regenerating evaluation figures and metrics
  - Inspecting common misclassification patterns

Usage:
    python src/evaluate.py
    python src/evaluate.py --model-path model --data-path data/test.csv
    python src/evaluate.py --error-analysis
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Allow running as `python src/evaluate.py` from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_model  # noqa: E402


LABELS_ORDERED = ["Negative", "Neutral", "Positive"]


def load_test_data(data_path: str) -> pd.DataFrame:
    """Load and validate the test CSV produced by 01_eda.ipynb."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Test data not found at '{data_path}'.\n"
            f"Run notebooks/01_eda.ipynb first to generate data/test.csv."
        )

    df = pd.read_csv(data_path)

    required_cols = {"text", "sentiment", "label_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")

    if df.isnull().sum().sum() > 0:
        print(f"WARNING: {data_path} contains {df.isnull().sum().sum()} null values. Dropping them.")
        df = df.dropna(subset=["text"]).reset_index(drop=True)

    return df


def run_evaluation(model_path: str, data_path: str, output_dir: str, error_analysis: bool = False):
    """
    Run full evaluation: load model, predict on test set, compute metrics,
    save confusion matrix plot, optionally print misclassified examples.
    """
    print("=" * 60)
    print("  CUSTOMER REVIEW INTELLIGENCE — MODEL EVALUATION")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────
    print(f"\nLoading model from '{model_path}'...")
    model = load_model(model_path)

    # ── Load test data ───────────────────────────────────────────────
    print(f"\nLoading test data from '{data_path}'...")
    df_test = load_test_data(data_path)
    print(f"Test set size: {len(df_test):,} reviews")
    print(f"Class distribution:\n{df_test['sentiment'].value_counts()}")

    # ── Run predictions (batched) ────────────────────────────────────
    print(f"\nRunning predictions on {len(df_test):,} reviews...")
    print("(This may take a few minutes on CPU — progress printed every 1000 rows)")

    texts = df_test["text"].tolist()
    true_labels = df_test["label_id"].tolist()

    pred_labels = []
    pred_confidences = []
    pred_uncertain = []

    chunk_size = 32
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        results = model.predict_batch(chunk)
        for r in results:
            # Map label name back to label_id for sklearn metrics
            label_id = [k for k, v in model.id2label.items() if v == r["label"]][0]
            pred_labels.append(label_id)
            pred_confidences.append(r["confidence"])
            pred_uncertain.append(r["uncertain"])

        if (i + chunk_size) % 1000 < chunk_size:
            print(f"  ...{min(i + chunk_size, len(texts)):,} / {len(texts):,}")

    print("Predictions complete.\n")

    # ── Core metrics ──────────────────────────────────────────────────
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    per_class_f1 = f1_score(true_labels, pred_labels, average=None)

    print("=" * 60)
    print("  TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(
        true_labels, pred_labels, target_names=LABELS_ORDERED, digits=4
    ))
    print(f"Overall accuracy : {acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")

    # ── Uncertainty analysis ──────────────────────────────────────────
    n_uncertain = sum(pred_uncertain)
    pct_uncertain = n_uncertain / len(pred_uncertain) * 100

    # Accuracy split by certain vs uncertain predictions — does the
    # uncertainty flag actually correlate with correctness?
    correct = np.array(true_labels) == np.array(pred_labels)
    certain_mask = ~np.array(pred_uncertain)
    uncertain_mask = np.array(pred_uncertain)

    acc_certain = correct[certain_mask].mean() if certain_mask.sum() > 0 else float("nan")
    acc_uncertain = correct[uncertain_mask].mean() if uncertain_mask.sum() > 0 else float("nan")

    print("\n" + "=" * 60)
    print("  UNCERTAINTY FLAG ANALYSIS")
    print("=" * 60)
    print(f"Threshold              : {model.uncertainty_threshold}")
    print(f"Flagged as uncertain   : {n_uncertain:,} / {len(pred_uncertain):,} ({pct_uncertain:.1f}%)")
    print(f"Accuracy when CERTAIN  : {acc_certain:.4f}  (n={certain_mask.sum():,})")
    print(f"Accuracy when UNCERTAIN: {acc_uncertain:.4f}  (n={uncertain_mask.sum():,})")
    if acc_certain > acc_uncertain:
        print(
            "\nThe uncertainty flag is doing its job: predictions below the "
            "threshold are meaningfully less reliable than confident predictions."
        )
    else:
        print(
            "\nNOTE: Uncertain predictions are not less accurate than certain ones. "
            "Threshold may need tuning."
        )

    # ── Confusion matrix ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(true_labels, pred_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS_ORDERED, yticklabels=LABELS_ORDERED,
        linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title("Confusion matrix — raw counts", fontweight="bold")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=LABELS_ORDERED, yticklabels=LABELS_ORDERED,
        linewidths=0.5, ax=axes[1], vmin=0, vmax=100
    )
    axes[1].set_title("Confusion matrix — row-normalised (%)", fontweight="bold")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    plt.suptitle("Test Set Confusion Matrix (local re-evaluation)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=120)
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.close(fig)

    # ── Confidence distribution plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_confidences, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(
        x=model.uncertainty_threshold, color="red", linestyle="--", linewidth=1.5,
        label=f"Uncertainty threshold ({model.uncertainty_threshold})"
    )
    ax.set_title("Prediction confidence distribution (test set)", fontweight="bold")
    ax.set_xlabel("Top-class confidence")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()

    conf_path = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(conf_path, bbox_inches="tight", dpi=120)
    print(f"Confidence distribution saved to: {conf_path}")
    plt.close(fig)

    # ── Save metrics summary as JSON (for README / resume reference) ──
    summary = {
        "test_set_size": len(df_test),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {
            LABELS_ORDERED[i]: round(float(per_class_f1[i]), 4)
            for i in range(len(LABELS_ORDERED))
        },
        "uncertainty_threshold": model.uncertainty_threshold,
        "pct_flagged_uncertain": round(pct_uncertain, 2),
        "accuracy_when_certain": round(float(acc_certain), 4),
        "accuracy_when_uncertain": round(float(acc_uncertain), 4),
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to: {summary_path}")

    # ── Error analysis ──────────────────────────────────────────────
    if error_analysis:
        print_error_analysis(df_test, true_labels, pred_labels, pred_confidences, model.id2label)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)

    return summary


def print_error_analysis(df_test, true_labels, pred_labels, pred_confidences, id2label, n_examples=5):
    """
    Print misclassified examples grouped by error type.

    Useful for the README's "expected challenges" section and for
    answering "what does your model get wrong?" in interviews.
    """
    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS — MISCLASSIFIED EXAMPLES")
    print("=" * 60)

    df_results = df_test.copy()
    df_results["true_label"] = [id2label[i] for i in true_labels]
    df_results["pred_label"] = [id2label[i] for i in pred_labels]
    df_results["confidence"] = pred_confidences
    df_results["correct"] = df_results["true_label"] == df_results["pred_label"]

    errors = df_results[~df_results["correct"]]
    print(f"\nTotal errors: {len(errors):,} / {len(df_results):,} ({len(errors)/len(df_results)*100:.1f}%)")

    # Error breakdown by (true -> predicted) pair
    print("\nError breakdown (true → predicted):")
    error_pairs = errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
    for (true_l, pred_l), count in error_pairs.items():
        print(f"  {true_l:>8} → {pred_l:<8} : {count:,}")

    # Most common error type: Neutral confused with Positive/Negative
    print("\n" + "-" * 60)
    print("Sample misclassifications (highest confidence errors — most surprising):")
    print("-" * 60)

    top_confident_errors = errors.sort_values("confidence", ascending=False).head(n_examples)
    for _, row in top_confident_errors.iterrows():
        text_preview = row["text"][:120] + "..." if len(row["text"]) > 120 else row["text"]
        print(f"\nTrue: {row['true_label']:<10} Predicted: {row['pred_label']:<10} (conf: {row['confidence']:.3f})")
        print(f"  \"{text_preview}\"")

    print("\n" + "-" * 60)
    print("Common patterns to look for in these examples:")
    print("  - Negation:    'not bad', 'wasn't great' — model may miss negation scope")
    print("  - Mixed:       'great food but slow service' — genuinely ambiguous for Neutral")
    print("  - Sarcasm:     'oh great, another broken item' — hardest case, expected to fail")
    print("  - Short text:  very short reviews may lack enough signal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned sentiment model on the test set."
    )
    parser.add_argument(
        "--model-path", default="model",
        help="Path to the trained model directory (default: model)"
    )
    parser.add_argument(
        "--data-path", default="data/test.csv",
        help="Path to test CSV (default: data/test.csv)"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to save confusion matrix / summary outputs (default: data)"
    )
    parser.add_argument(
        "--error-analysis", action="store_true",
        help="Print misclassified examples grouped by error type"
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        error_analysis=args.error_analysis,
    )