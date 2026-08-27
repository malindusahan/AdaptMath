"""One-Time Final Evaluation on the Frozen Held-Out Test Split (N=636)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topic_extraction.finetune_minilm import evaluate_model
from src.topic_extraction.keyword_baseline import KeywordBaselineClassifier
from src.topic_extraction.minilm_baseline import PretrainedMiniLMBaseline
from src.topic_extraction.tfidf_baseline import TfidfBaselineClassifier
from src.topic_extraction.topic_extractor import HybridTopicExtractor

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
FINAL_METRICS_JSON = ARTIFACTS_DIR / "final_test_metrics.json"


def run_final_test_evaluation() -> dict[str, Any]:
    test_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "test.csv"
    test_df = pd.read_csv(test_csv)
    print(f"Loaded frozen held-out test split: {len(test_df)} examples across {test_df['canonical_name'].nunique()} skills.")

    # -------------------------------------------------------------
    # 1. Keyword Baseline Evaluation
    # -------------------------------------------------------------
    print("\n[1/5] Evaluating Keyword Baseline on Test Split...")
    kw_clf = KeywordBaselineClassifier()
    kw_metrics = kw_clf.evaluate(test_df)

    # -------------------------------------------------------------
    # 2. TF-IDF Baseline Evaluation
    # -------------------------------------------------------------
    print("[2/5] Evaluating TF-IDF Baseline on Test Split...")
    tfidf_clf = TfidfBaselineClassifier.load_artifacts()
    tfidf_metrics = tfidf_clf.evaluate(test_df)

    # -------------------------------------------------------------
    # 3. Pretrained MiniLM Baseline Evaluation
    # -------------------------------------------------------------
    print("[3/5] Evaluating Pretrained MiniLM Baseline on Test Split...")
    pretrained_clf = PretrainedMiniLMBaseline()
    pretrained_clf.load_centroids(
        ARTIFACTS_DIR / "minilm_pretrained_skill_centroids.npz"
    )
    pretrained_metrics = pretrained_clf.evaluate(test_df)

    # -------------------------------------------------------------
    # 4. Fine-Tuned MiniLM Model Evaluation
    # -------------------------------------------------------------
    print("[4/5] Evaluating Fine-Tuned MiniLM on Test Split...")
    ft_model = SentenceTransformer(
        str(ARTIFACTS_DIR / "minilm_finetuned"), device="cpu"
    )
    ft_data = np.load(ARTIFACTS_DIR / "minilm_finetuned_skill_centroids.npz")
    ft_centroids = ft_data["centroids"]
    ft_names = list(ft_data["canonical_names"])
    ft_metrics = evaluate_model(ft_model, ft_centroids, ft_names, test_df)

    # -------------------------------------------------------------
    # 5. Final Hybrid Topic Extractor Evaluation
    # -------------------------------------------------------------
    print("[5/5] Evaluating Final Hybrid Topic Extractor on Test Split...")
    hybrid_extractor = HybridTopicExtractor()

    y_true = test_df["canonical_name"].tolist()
    y_pred = []
    hybrid_methods = []
    top3_hits = 0
    covered_indices = []
    confidences = []
    confusion_pairs = []

    all_canonical_classes = sorted(list(ft_names))

    for idx, row in test_df.iterrows():
        text = str(row["text"])
        true_name = row["canonical_name"]
        res = hybrid_extractor.extract(text)

        hybrid_methods.append(res.method)
        confidences.append(res.confidence)

        # Check top-3 candidates
        cand_names = [
            hybrid_extractor.ontology_lookup.get_by_skill_code(c.skill_code).canonical_name
            for c in res.top_candidates
            if hybrid_extractor.ontology_lookup.get_by_skill_code(c.skill_code) is not None
        ]
        if true_name in cand_names[:3]:
            top3_hits += 1

        if not res.is_abstain and res.canonical_skill_name:
            pred_name = res.canonical_skill_name
            y_pred.append(pred_name)
            covered_indices.append(idx)
            if pred_name != true_name:
                confusion_pairs.append({
                    "text": text,
                    "true_skill": row.get("skill_name", row["canonical_name"]),
                    "pred_skill": res.display_name,
                    "confidence": res.confidence,
                    "method": res.method,
                })
        else:
            y_pred.append("__ABSTAIN__")

    total_test = len(test_df)
    covered_count = len(covered_indices)
    coverage = covered_count / total_test
    abstention_rate = 1.0 - coverage

    correct_overall = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    overall_accuracy = correct_overall / total_test

    if covered_count > 0:
        covered_true = [y_true[i] for i in covered_indices]
        covered_pred = [y_pred[i] for i in covered_indices]
        covered_accuracy = sum(1 for yt, yp in zip(covered_true, covered_pred) if yt == yp) / covered_count
    else:
        covered_accuracy = 0.0

    top3_accuracy = top3_hits / total_test

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=all_canonical_classes,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=all_canonical_classes,
            average="weighted",
            zero_division=0,
        )
    )

    # Per-skill metrics
    p, r, f1s, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=all_canonical_classes,
        zero_division=0,
    )
    per_skill_metrics = {
        name: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1s[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(all_canonical_classes)
    }

    # Method usage breakdown
    method_counts = dict(Counter(hybrid_methods))

    hybrid_results = {
        "accuracy": float(overall_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": float(top3_accuracy),
        "coverage": float(coverage),
        "abstention_rate": float(abstention_rate),
        "covered_accuracy": float(covered_accuracy),
        "mean_confidence": float(np.mean(confidences)),
        "method_breakdown": method_counts,
        "confusion_count": len(confusion_pairs),
        "confusion_examples": confusion_pairs[:10],
    }

    final_payload = {
        "dataset": {
            "name": "frozen_test_split",
            "samples": total_test,
            "skills": test_df["canonical_name"].nunique(),
        },
        "benchmarks": {
            "keyword_baseline": {
                "accuracy": kw_metrics["accuracy"],
                "macro_f1": kw_metrics["macro_f1"],
                "coverage": kw_metrics["coverage"],
                "covered_accuracy": kw_metrics["covered_accuracy"],
            },
            "tfidf_baseline": {
                "accuracy": tfidf_metrics["accuracy"],
                "macro_f1": tfidf_metrics["macro_f1"],
                "weighted_f1": tfidf_metrics["weighted_f1"],
                "top3_accuracy": tfidf_metrics["top3_accuracy"],
            },
            "minilm_pretrained": {
                "accuracy": pretrained_metrics["accuracy"],
                "macro_f1": pretrained_metrics["macro_f1"],
                "weighted_f1": pretrained_metrics["weighted_f1"],
                "top3_accuracy": pretrained_metrics["top3_accuracy"],
            },
            "minilm_finetuned": {
                "accuracy": ft_metrics["accuracy"],
                "macro_f1": ft_metrics["macro_f1"],
                "weighted_f1": ft_metrics["weighted_f1"],
                "top3_accuracy": ft_metrics["top3_accuracy"],
            },
            "hybrid_topic_extractor": hybrid_results,
        },
        "per_skill_metrics": per_skill_metrics,
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    return final_payload


def main():
    res = run_final_test_evaluation()
    b = res["benchmarks"]
    h = b["hybrid_topic_extractor"]

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION (FROZEN TEST SPLIT N=636)")
    print("=" * 80)
    print(f"Keyword test Macro F1:             {b['keyword_baseline']['macro_f1']:.4f}")
    print(f"TF-IDF test Macro F1:              {b['tfidf_baseline']['macro_f1']:.4f}")
    print(f"Pretrained MiniLM test Macro F1:   {b['minilm_pretrained']['macro_f1']:.4f}")
    print(f"Fine-tuned MiniLM test Macro F1:   {b['minilm_finetuned']['macro_f1']:.4f}")
    print("-" * 80)
    print(f"Hybrid test Accuracy:              {h['accuracy']*100:.2f}%")
    print(f"Hybrid test Macro F1:              {h['macro_f1']:.4f}")
    print(f"Hybrid test Weighted F1:           {h['weighted_f1']:.4f}")
    print(f"Hybrid Top-3:                      {h['top3_accuracy']*100:.2f}%")
    print(f"Hybrid Coverage:                   {h['coverage']*100:.2f}%")
    print(f"Hybrid Abstention rate:            {h['abstention_rate']*100:.2f}%")
    print(f"Hybrid Covered accuracy:           {h['covered_accuracy']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
