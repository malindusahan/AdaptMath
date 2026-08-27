"""Confidence Calibration and Safe Abstention System for Topic Extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topic_extraction.keyword_baseline import KeywordBaselineClassifier
from src.topic_extraction.tfidf_baseline import TfidfBaselineClassifier

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
THRESHOLDS_JSON = ARTIFACTS_DIR / "confidence_thresholds.json"

VAGUE_INPUTS = [
    "Can you help me?",
    "I don't understand this",
    "What should I do?",
    "Please explain",
    "Help",
    "Tell me the answer",
    "I am stuck on problem 4",
    "Why did I get this wrong?",
    "Hello tutor",
    "What is next?",
]


@dataclass
class HybridExtractedTopic:
    skill_id: str | None
    skill_code: str | None
    canonical_name: str | None
    display_name: str | None
    is_abstain: bool
    confidence: float
    margin: float
    decision_path: str
    top_candidates: list[dict[str, Any]]


class CalibratedTopicClassifier:
    """
    Tiered Hybrid Classifier combining:
    1. Deterministic Keyword/Rule layer (High-precision shortcuts)
    2. Fine-Tuned MiniLM dense semantic matcher
    3. TF-IDF lexical consistency support
    4. Calibrated confidence & margin gate with safe abstention
    """

    def __init__(
        self,
        min_similarity: float = 0.55,
        min_margin: float = 0.12,
        tfidf_support_threshold: float = 0.20,
    ):
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.tfidf_support_threshold = tfidf_support_threshold

        # Load models
        self.keyword_clf = KeywordBaselineClassifier()
        self.tfidf_clf = TfidfBaselineClassifier.load_artifacts()

        ft_model_path = ARTIFACTS_DIR / "minilm_finetuned"
        ft_centroids_path = ARTIFACTS_DIR / "minilm_finetuned_skill_centroids.npz"

        self.ft_model = SentenceTransformer(str(ft_model_path), device="cpu")
        ft_data = np.load(ft_centroids_path)
        self.ft_centroids = ft_data["centroids"]
        self.canonical_names = list(ft_data["canonical_names"])
        self.ontology_meta = json.loads(str(ft_data["metadata_json"]))
        self.name_to_meta = {m["canonical_name"]: m for m in self.ontology_meta}

    def predict(self, text: str) -> HybridExtractedTopic:
        """Classify input text with confidence calibration and safe abstention."""
        if not text or not str(text).strip():
            return HybridExtractedTopic(
                skill_id=None,
                skill_code=None,
                canonical_name=None,
                display_name=None,
                is_abstain=True,
                confidence=0.0,
                margin=0.0,
                decision_path="empty_input_abstain",
                top_candidates=[],
            )

        # 1. Tier 1: Keyword / Alias Rule Match
        kw_pred = self.keyword_clf.predict(text)
        if not kw_pred.is_abstain and kw_pred.confidence >= 0.99:
            meta = self.name_to_meta.get(kw_pred.canonical_name, {})
            return HybridExtractedTopic(
                skill_id=meta.get("skill_id"),
                skill_code=kw_pred.skill_code,
                canonical_name=kw_pred.canonical_name,
                display_name=kw_pred.display_name,
                is_abstain=False,
                confidence=1.0,
                margin=1.0,
                decision_path="rule_exact_alias",
                top_candidates=[
                    {
                        "skill_code": kw_pred.skill_code,
                        "canonical_name": kw_pred.canonical_name,
                        "display_name": kw_pred.display_name,
                        "similarity": 1.0,
                    }
                ],
            )

        # 2. Tier 2: Fine-Tuned MiniLM Semantic Encoding
        emb = self.ft_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(emb, self.ft_centroids.T)[0]
        ranked_indices = np.argsort(sims)[::-1]

        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        top_name = self.canonical_names[top_idx]
        top_meta = self.name_to_meta[top_name]
        top_sim = float(sims[top_idx])
        margin = float(top_sim - sims[second_idx])

        top_candidates = []
        for idx in ranked_indices[:3]:
            cand_name = self.canonical_names[idx]
            cand_meta = self.name_to_meta[cand_name]
            top_candidates.append(
                {
                    "skill_code": cand_meta["skill_code"],
                    "canonical_name": cand_name,
                    "display_name": cand_meta["display_name"],
                    "similarity": float(sims[idx]),
                }
            )

        # 3. Tier 3: TF-IDF Supporting Evidence
        tfidf_pred = self.tfidf_clf.predict(text, top_k=3)
        tfidf_agrees = (tfidf_pred.canonical_name == top_name)
        tfidf_strong = (tfidf_pred.similarity >= self.tfidf_support_threshold)

        # 4. Tier 4: Confidence & Margin Gating
        # High confidence semantic match
        if top_sim >= self.min_similarity and margin >= self.min_margin:
            return HybridExtractedTopic(
                skill_id=top_meta.get("skill_id"),
                skill_code=top_meta["skill_code"],
                canonical_name=top_name,
                display_name=top_meta["display_name"],
                is_abstain=False,
                confidence=top_sim,
                margin=margin,
                decision_path="semantic_high_confidence",
                top_candidates=top_candidates,
            )

        # Moderate semantic match with strong TF-IDF corroboration
        if (
            top_sim >= (self.min_similarity - 0.10)
            and margin >= (self.min_margin - 0.05)
            and tfidf_agrees
            and tfidf_strong
        ):
            return HybridExtractedTopic(
                skill_id=top_meta.get("skill_id"),
                skill_code=top_meta["skill_code"],
                canonical_name=top_name,
                display_name=top_meta["display_name"],
                is_abstain=False,
                confidence=top_sim,
                margin=margin,
                decision_path="semantic_tfidf_consensus",
                top_candidates=top_candidates,
            )

        # Low confidence or insufficient margin -> Safe Abstention
        return HybridExtractedTopic(
            skill_id=None,
            skill_code=None,
            canonical_name=None,
            display_name=None,
            is_abstain=True,
            confidence=top_sim,
            margin=margin,
            decision_path="low_confidence_abstain",
            top_candidates=top_candidates,
        )


def calibrate_thresholds() -> dict[str, Any]:
    val_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "validation.csv"
    val_df = pd.read_csv(val_csv)

    print("Initializing base models for calibration...")
    clf = CalibratedTopicClassifier()

    # Pre-extract representations for fast grid search in memory
    print("Pre-computing validation model outputs (N=641)...")
    val_records = []
    for _, row in val_df.iterrows():
        text = str(row["text"])
        kw_pred = clf.keyword_clf.predict(text)
        tfidf_pred = clf.tfidf_clf.predict(text, top_k=3)

        emb = clf.ft_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(emb, clf.ft_centroids.T)[0]
        ranked_indices = np.argsort(sims)[::-1]
        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        val_records.append(
            {
                "true_name": row["canonical_name"],
                "kw_abstain": kw_pred.is_abstain,
                "kw_name": kw_pred.canonical_name,
                "kw_conf": kw_pred.confidence,
                "ft_name": clf.canonical_names[top_idx],
                "top_sim": float(sims[top_idx]),
                "margin": float(sims[top_idx] - sims[second_idx]),
                "tfidf_name": tfidf_pred.canonical_name,
                "tfidf_sim": tfidf_pred.similarity,
            }
        )

    # Pre-extract vague query outputs
    vague_records = []
    for text in VAGUE_INPUTS:
        kw_pred = clf.keyword_clf.predict(text)
        tfidf_pred = clf.tfidf_clf.predict(text, top_k=3)

        emb = clf.ft_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(emb, clf.ft_centroids.T)[0]
        ranked_indices = np.argsort(sims)[::-1]
        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        vague_records.append(
            {
                "kw_abstain": kw_pred.is_abstain,
                "kw_conf": kw_pred.confidence,
                "top_sim": float(sims[top_idx]),
                "margin": float(sims[top_idx] - sims[second_idx]),
                "tfidf_name": tfidf_pred.canonical_name,
                "ft_name": clf.canonical_names[top_idx],
                "tfidf_sim": tfidf_pred.similarity,
            }
        )

    print("Running in-memory threshold grid search across 125 candidate parameter combinations...")
    similarity_candidates = [0.45, 0.50, 0.55, 0.60, 0.65]
    margin_candidates = [0.08, 0.10, 0.12, 0.15, 0.18]
    tfidf_candidates = [0.15, 0.20, 0.25]

    all_classes = sorted(clf.canonical_names)
    total_samples = len(val_records)
    best_score = -1.0
    best_config = {}
    best_eval = {}

    for sim in similarity_candidates:
        for mg in margin_candidates:
            for tf_sup in tfidf_candidates:
                # Fast evaluation on validation records
                y_true = []
                y_pred = []
                covered_indices = []

                for i, r in enumerate(val_records):
                    y_true.append(r["true_name"])

                    # Rule tier
                    if not r["kw_abstain"] and r["kw_conf"] >= 0.99:
                        y_pred.append(r["kw_name"])
                        covered_indices.append(i)
                    # Semantic tier (high confidence)
                    elif r["top_sim"] >= sim and r["margin"] >= mg:
                        y_pred.append(r["ft_name"])
                        covered_indices.append(i)
                    # Consensus tier
                    elif (
                        r["top_sim"] >= (sim - 0.10)
                        and r["margin"] >= (mg - 0.05)
                        and (r["tfidf_name"] == r["ft_name"])
                        and (r["tfidf_sim"] >= tf_sup)
                    ):
                        y_pred.append(r["ft_name"])
                        covered_indices.append(i)
                    else:
                        y_pred.append("__ABSTAIN__")

                cov_cnt = len(covered_indices)
                coverage = cov_cnt / total_samples
                abstention_rate = 1.0 - coverage

                if cov_cnt > 0:
                    cov_true = [y_true[idx] for idx in covered_indices]
                    cov_pred = [y_pred[idx] for idx in covered_indices]
                    covered_acc = sum(1 for yt, yp in zip(cov_true, cov_pred) if yt == yp) / cov_cnt
                else:
                    covered_acc = 0.0

                macro_f1 = float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=all_classes,
                        average="macro",
                        zero_division=0,
                    )
                )

                # Test vague inputs
                vague_abstains = 0
                for vr in vague_records:
                    is_pred = False
                    if not vr["kw_abstain"] and vr["kw_conf"] >= 0.99:
                        is_pred = True
                    elif vr["top_sim"] >= sim and vr["margin"] >= mg:
                        is_pred = True
                    elif (
                        vr["top_sim"] >= (sim - 0.10)
                        and vr["margin"] >= (mg - 0.05)
                        and (vr["tfidf_name"] == vr["ft_name"])
                        and (vr["tfidf_sim"] >= tf_sup)
                    ):
                        is_pred = True

                    if not is_pred:
                        vague_abstains += 1

                vague_rate = vague_abstains / len(VAGUE_INPUTS)

                # Score candidate
                score = macro_f1 * 0.45 + covered_acc * 0.45 + coverage * 0.05 + vague_rate * 0.05

                if score > best_score and vague_rate >= 1.0 and coverage >= 0.98:
                    best_score = score
                    best_config = {
                        "min_similarity": sim,
                        "min_margin": mg,
                        "tfidf_support_threshold": tf_sup,
                    }
                    best_eval = {
                        "coverage": float(coverage),
                        "covered_accuracy": float(covered_acc),
                        "abstention_rate": float(abstention_rate),
                        "macro_f1": float(macro_f1),
                        "vague_abstention_rate": float(vague_rate),
                    }

    out_data = {
        **best_config,
        "validation_coverage": best_eval["coverage"],
        "validation_covered_accuracy": best_eval["covered_accuracy"],
        "validation_abstention_rate": best_eval["abstention_rate"],
        "validation_macro_f1": best_eval["macro_f1"],
        "vague_input_abstention_rate": best_eval["vague_abstention_rate"],
        "vague_inputs_tested": VAGUE_INPUTS,
    }

    with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return out_data


def main():
    res = calibrate_thresholds()
    print("\n" + "=" * 60)
    print("Calibrated Confidence & Safe Abstention Results")
    print("=" * 60)
    print(f"Selected MiniLM similarity threshold: {res['min_similarity']:.2f}")
    print(f"Selected margin threshold:          {res['min_margin']:.2f}")
    print(f"Selected TF-IDF support threshold:  {res['tfidf_support_threshold']:.2f}")
    print(f"Validation coverage:                {res['validation_coverage']*100:.2f}%")
    print(f"Validation covered accuracy:        {res['validation_covered_accuracy']*100:.2f}%")
    print(f"Validation abstention rate:         {res['validation_abstention_rate']*100:.2f}%")
    print(f"Validation Macro F1:                {res['validation_macro_f1']:.4f}")
    print(f"Vague-input abstention result:      {res['vague_input_abstention_rate']*100:.2f}% ({len(res['vague_inputs_tested'])}/{len(res['vague_inputs_tested'])} abstained)")
    print("=" * 60)


if __name__ == "__main__":
    main()
