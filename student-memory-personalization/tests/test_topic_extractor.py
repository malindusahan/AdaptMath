"""Unit tests for the production Hybrid Topic Extractor."""

from __future__ import annotations

import pytest
from src.ontology.ontology_lookup_service import OntologyLookupService
from src.topic_extraction.topic_extractor import (
    HybridTopicExtractor,
    TopicExtractionResult,
    get_topic_extractor,
)


@pytest.fixture(scope="module")
def extractor():
    return get_topic_extractor()


@pytest.fixture(scope="module")
def ontology_lookup():
    return OntologyLookupService()


def test_singleton_loader_reuses_instance():
    """Verify that get_topic_extractor() returns the same singleton instance."""
    inst1 = get_topic_extractor()
    inst2 = get_topic_extractor()
    assert inst1 is inst2


def test_exact_alias_match_returns_neural_prediction(extractor: HybridTopicExtractor, ontology_lookup: OntologyLookupService):
    """Verify that exact alias mentions trigger high-confidence neural topic extraction."""
    query = "Can you explain PEMDAS and how it works?"
    result = extractor.extract(query)

    assert result.method == "minilm_finetuned"
    assert result.confidence >= 0.40
    assert result.needs_review is False
    assert "Order of Operations" in result.display_name
    assert result.skill_id is not None

    # Verify returned skill exists in ontology
    detail = ontology_lookup.get_by_skill_id(result.skill_id)
    assert detail is not None
    assert detail.skill_code == result.skill_code


def test_pythagorean_theorem_alias_rule(extractor: HybridTopicExtractor):
    """Verify neural extraction for Pythagoras theorem."""
    query = "I am having trouble using Pythagoras theorem on triangles"
    result = extractor.extract(query)

    assert result.method == "minilm_finetuned"
    assert result.display_name == "Pythagorean Theorem"
    assert result.needs_review is False


def test_semantic_wording_returns_minilm(extractor: HybridTopicExtractor, ontology_lookup: OntologyLookupService):
    """Verify that paraphrased semantic questions without exact alias match use fine-tuned MiniLM."""
    query = "How do I work out the steepness of a line from two points on a graph?"
    result = extractor.extract(query)

    assert result.method == "minilm_finetuned"
    assert result.display_name in ("Slope", "Finding Slope from Ordered Pairs")
    assert result.confidence >= 0.45
    assert result.needs_review is False
    assert result.skill_id is not None

    detail = ontology_lookup.get_by_skill_id(result.skill_id)
    assert detail is not None


def test_semantic_median_concept_wording(extractor: HybridTopicExtractor):
    """Verify semantic classification for statistical median."""
    query = "I keep mixing up mean and median when analyzing numbers"
    result = extractor.extract(query)

    assert result.method == "minilm_finetuned"
    assert result.display_name in ("Median", "Mean")
    assert result.confidence >= 0.45
    assert result.needs_review is False


def test_semantic_similarity_congruence(extractor: HybridTopicExtractor):
    """Verify that geometry comparison queries extract geometry skills."""
    query = "What is the difference between congruent figures and similar figures?"
    result = extractor.extract(query)

    assert result.method == "minilm_finetuned"
    assert result.display_name in ("Calculations with Similar Figures", "Congruence")
    assert result.confidence >= 0.25
    assert result.needs_review is False


def test_vague_input_abstains(extractor: HybridTopicExtractor):
    """Verify that vague non-educational queries safely abstain."""
    vague_queries = [
        "Can you help me?",
        "I don't understand this",
        "What should I do?",
        "Help",
        "Tell me the answer",
    ]

    for q in vague_queries:
        res = extractor.extract(q)
        assert res.method == "abstain", f"Expected abstain for '{q}', got {res.method} -> {res.display_name}"
        assert res.skill_id is None
        assert res.skill_code is None
        assert res.needs_review is True
        assert res.is_abstain is True


def test_empty_and_whitespace_input_abstains(extractor: HybridTopicExtractor):
    """Verify empty or whitespace strings trigger immediate safe abstention."""
    for empty_val in ["", "   ", "\n\t  "]:
        res = extractor.extract(empty_val)
        assert res.method == "abstain"
        assert res.skill_id is None
        assert res.confidence == 0.0
        assert res.needs_review is True
        assert res.is_abstain is True
        assert len(res.top_candidates) == 0


def test_no_duplicate_candidates(extractor: HybridTopicExtractor):
    """Verify top candidates list contains unique skill codes only."""
    queries = [
        "What is the difference between congruent figures and similar figures?",
        "How do I add fractions with unlike denominators?",
        "Can you explain PEMDAS?",
        "Can you help me?",
    ]
    for q in queries:
        res = extractor.extract(q)
        codes = [c.skill_code for c in res.top_candidates]
        assert len(codes) == len(set(codes)), f"Found duplicate candidates for '{q}': {codes}"


def test_candidate_scores_descending(extractor: HybridTopicExtractor):
    """Verify top candidates list is sorted in strictly descending score order."""
    queries = [
        "How do I calculate the area of a circle?",
        "What is the difference between congruent figures and similar figures?",
        "How do I solve systems of equations?",
    ]
    for q in queries:
        res = extractor.extract(q)
        if res.top_candidates:
            scores = [c.score for c in res.top_candidates]
            assert scores == sorted(scores, reverse=True), f"Scores not descending for '{q}': {scores}"


def test_returned_skill_matches_rank1_candidate(extractor: HybridTopicExtractor):
    """Verify returned prediction matches rank-1 candidate when not abstaining."""
    queries = [
        "Can you explain PEMDAS and how it works?",
        "How do I work out the steepness of a line from two points on a graph?",
        "What is the difference between congruent figures and similar figures?",
    ]
    for q in queries:
        res = extractor.extract(q)
        if not res.is_abstain:
            assert res.top_candidates, f"No candidates for '{q}'"
            assert res.skill_code == res.top_candidates[0].skill_code
            assert res.display_name == res.top_candidates[0].display_name


def test_returned_confidence_matches_rank1_candidate_score(extractor: HybridTopicExtractor):
    """Verify returned confidence matches the rank-1 candidate's score."""
    queries = [
        "Can you explain PEMDAS?",
        "How do I calculate percentage discount on a sale item?",
        "What is the difference between congruent figures and similar figures?",
        "Can you help me?",
    ]
    for q in queries:
        res = extractor.extract(q)
        if res.top_candidates:
            assert pytest.approx(res.confidence, 1e-4) == pytest.approx(res.top_candidates[0].score, 1e-4)


def test_all_returned_skills_exist_in_ontology(extractor: HybridTopicExtractor, ontology_lookup: OntologyLookupService):
    """Verify every candidate and extracted skill resolves in the canonical ontology."""
    test_queries = [
        "How do I add fractions with unlike denominators?",
        "What is the mean and median of this dataset?",
        "How do I solve systems of linear equations using elimination?",
        "What is the absolute value of negative ten?",
    ]

    for q in test_queries:
        res = extractor.extract(q)
        if not res.is_abstain:
            detail = ontology_lookup.get_by_skill_id(res.skill_id)
            assert detail is not None
            assert detail.skill_code == res.skill_code
            assert detail.display_name == res.display_name


def test_deterministic_repeated_results(extractor: HybridTopicExtractor):
    """Verify extractor returns identical results when called repeatedly."""
    q = "How do I calculate percentage discount on a sale item?"
    res1 = extractor.extract(q).to_dict()
    res2 = extractor.extract(q).to_dict()
    assert res1 == res2
