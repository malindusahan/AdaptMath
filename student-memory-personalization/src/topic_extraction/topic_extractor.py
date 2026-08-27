"""Production Hybrid Topic Extractor for Canonical Educational Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_lookup_service import (
    CanonicalSkillDetail,
    OntologyLookupService,
)
from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    normalize_token,
)
from src.topic_extraction.keyword_baseline import KeywordBaselineClassifier
from src.topic_extraction.tfidf_baseline import TfidfBaselineClassifier

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
FT_MODEL_DIR = (
    ARTIFACTS_DIR / "minilm_finetuned_v2"
    if (ARTIFACTS_DIR / "minilm_finetuned_v2").exists()
    else ARTIFACTS_DIR / "minilm_finetuned"
)
FT_CENTROIDS_NPZ = (
    ARTIFACTS_DIR / "minilm_finetuned_v2_skill_centroids.npz"
    if (ARTIFACTS_DIR / "minilm_finetuned_v2_skill_centroids.npz").exists()
    else ARTIFACTS_DIR / "minilm_finetuned_skill_centroids.npz"
)
THRESHOLDS_JSON = ARTIFACTS_DIR / "confidence_thresholds.json"
MODEL_VERSION = "phase16-minilm-ft-v2"


@dataclass(frozen=True)
class TopicCandidate:
    """Individual candidate skill match."""

    skill_code: str
    display_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_code": self.skill_code,
            "display_name": self.display_name,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class TopicExtractionResult:
    """Production topic extraction response."""

    skill_id: str | None
    skill_code: str | None
    canonical_skill_name: str | None
    display_name: str | None
    confidence: float
    method: str
    needs_review: bool
    top_candidates: list[TopicCandidate] = field(default_factory=list)
    model_version: str = MODEL_VERSION

    @property
    def is_abstain(self) -> bool:
        return self.method == "abstain" or self.skill_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_code": self.skill_code,
            "canonical_skill_name": self.canonical_skill_name,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "needs_review": self.needs_review,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "model_version": self.model_version,
        }


class HybridTopicExtractor:
    """
    Pure Neural Sentence-Transformer Topic Extractor:
    1. Fine-Tuned MiniLM Dense Semantic Embedding Classifier
    2. Deep Semantic Understanding across all Canonical Math Skills
    3. Calibrated Similarity Gating with Safe Abstention for Non-Math / Out-of-Domain Queries
    """

    def __init__(
        self,
        ontology_lookup: OntologyLookupService | None = None,
        min_similarity: float = 0.33,
        min_margin: float = 0.02,
        tfidf_support_threshold: float = 0.15,
    ):
        self.ontology_lookup = (
            ontology_lookup
            if ontology_lookup is not None
            else OntologyLookupService()
        )
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.tfidf_support_threshold = tfidf_support_threshold

        # Load calibrated thresholds from disk if available
        if THRESHOLDS_JSON.exists():
            try:
                with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.min_similarity = float(cfg.get("min_similarity", self.min_similarity))
                    self.min_margin = float(cfg.get("min_margin", self.min_margin))
            except Exception:
                pass

        # Keep the optional neural runtime out of API module import. This lets
        # evidence-only Memory routes run in a minimal service image while the
        # topic endpoint retains its existing model and dependency requirements.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Topic extraction requires the sentence-transformers runtime."
            ) from exc

        # Fine-tuned MiniLM neural model and centroids
        self.ft_model = SentenceTransformer(str(FT_MODEL_DIR), device="cpu")
        ft_data = np.load(FT_CENTROIDS_NPZ, allow_pickle=True)
        self.ft_centroids = ft_data["centroids"]
        self.canonical_names = list(ft_data["canonical_names"])
        self.ontology_meta = json.loads(str(ft_data["metadata_json"]))
        self.name_to_meta = {m["canonical_name"]: m for m in self.ontology_meta}

    def _resolve_skill_details(self, canonical_name_or_code: str) -> CanonicalSkillDetail | None:
        """Resolve full canonical skill details from ontology lookup service."""
        detail = self.ontology_lookup.get_by_canonical_name(canonical_name_or_code)
        if detail is None:
            detail = self.ontology_lookup.get_by_skill_code(canonical_name_or_code)
        return detail

    def extract(self, text: str) -> TopicExtractionResult:
        """Extract canonical skill topic from input student text purely via Sentence Transformer neural embeddings."""
        # 1. Validate empty / whitespace text
        if not text or not str(text).strip():
            return TopicExtractionResult(
                skill_id=None,
                skill_code=None,
                canonical_skill_name=None,
                display_name=None,
                confidence=0.0,
                method="abstain",
                needs_review=True,
                top_candidates=[],
                model_version=MODEL_VERSION,
            )

        # 2. Compute 384-dimensional dense semantic embedding with fine-tuned MiniLM
        emb = self.ft_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(emb, self.ft_centroids.T)[0]
        ranked_indices = np.argsort(sims)[::-1]

        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        top_name = self.canonical_names[top_idx]
        top_sim = float(sims[top_idx])
        margin = float(top_sim - sims[second_idx])

        # Build deduplicated top MiniLM candidates list (sorted descending)
        minilm_candidates: list[TopicCandidate] = []
        seen_codes = set()
        for idx in ranked_indices:
            cand_name = self.canonical_names[idx]
            cand_meta = self.name_to_meta.get(cand_name, {})
            code = cand_meta.get("skill_code", "UNKNOWN")
            if code not in seen_codes:
                seen_codes.add(code)
                minilm_candidates.append(
                    TopicCandidate(
                        skill_code=code,
                        display_name=cand_meta.get("display_name", cand_name),
                        score=float(sims[idx]),
                    )
                )
            if len(minilm_candidates) >= 3:
                break

        # 3. Pure neural classification decision:
        # Math queries produce top_sim >= 0.28.
        # Non-math chit-chat ("hi", "who are you", "weather") produces top_sim < 0.25.
        if top_sim >= self.min_similarity:
            detail = self._resolve_skill_details(top_name)
            if detail is not None:
                return TopicExtractionResult(
                    skill_id=str(detail.skill_id),
                    skill_code=detail.skill_code,
                    canonical_skill_name=detail.canonical_name,
                    display_name=detail.display_name,
                    confidence=minilm_candidates[0].score,
                    method="minilm_finetuned",
                    needs_review=False,
                    top_candidates=minilm_candidates,
                    model_version=MODEL_VERSION,
                )

        # 4. Safe Abstention for non-math / general questions (triggers natural English redirect)
        top_abstain_score = minilm_candidates[0].score if minilm_candidates else top_sim
        return TopicExtractionResult(
            skill_id=None,
            skill_code=None,
            canonical_skill_name=None,
            display_name=None,
            confidence=top_abstain_score,
            method="abstain",
            needs_review=True,
            top_candidates=minilm_candidates,
            model_version=MODEL_VERSION,
        )


# Global singleton instance cache
_EXTRACTOR_INSTANCE: HybridTopicExtractor | None = None


def get_topic_extractor() -> HybridTopicExtractor:
    """Return the global cached singleton instance of HybridTopicExtractor."""
    global _EXTRACTOR_INSTANCE
    if _EXTRACTOR_INSTANCE is None:
        _EXTRACTOR_INSTANCE = HybridTopicExtractor()
    return _EXTRACTOR_INSTANCE
