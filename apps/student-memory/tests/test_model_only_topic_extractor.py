"""Unit and integration tests for pure neural SentenceTransformer Topic Extractor."""

from __future__ import annotations

import pytest
from src.topic_extraction.topic_extractor import HybridTopicExtractor, get_topic_extractor


@pytest.fixture(scope="module")
def extractor():
    return get_topic_extractor()


def test_pure_neural_extraction_subtraction_whole_numbers(extractor: HybridTopicExtractor):
    """Verify colloquial and natural subtraction queries extract valid arithmetic skills."""
    queries = [
        "if we separate 10 from 100, what is the answer?",
        "If we take away 15 from 60, how much is left?",
        "What is 100 minus 10?",
        "How do I subtract whole numbers with borrowing or regrouping?",
    ]
    for q in queries:
        res = extractor.extract(q)
        assert res.method == "minilm_finetuned", f"Failed for query: {q}"
        assert res.skill_code in {"SKILL_74", "SKILL_83", "SKILL_277"}, f"Got {res.skill_code} for {q}"
        assert res.confidence >= 0.25
        assert res.needs_review is False


def test_pure_neural_extraction_linear_equations(extractor: HybridTopicExtractor):
    """Verify natural informal and formal linear equation queries extract algebra skills via MiniLM neural embeddings."""
    queries = [
        ("How do I solve 2x + 3 = 11?", {"SKILL_166", "SKILL_193", "SKILL_311"}),
        ("Solve the linear equation 3x + 5 = 20", {"SKILL_166", "SKILL_193", "SKILL_311"}),
        ("How do I isolate the variable when numbers are on both sides of the equals sign?", {"SKILL_368", "SKILL_166", "SKILL_193"}),
        ("How do I solve one-step and two-step linear equations?", {"SKILL_193", "SKILL_166", "SKILL_311"}),
        ("How do I solve linear equations like 4x + 12 = 36 step by step?", {"SKILL_193", "SKILL_166", "SKILL_311"}),
    ]
    for q, expected_codes in queries:
        res = extractor.extract(q)
        assert res.method == "minilm_finetuned", f"Failed for query: {q}"
        assert res.skill_code in expected_codes, f"Expected one of {expected_codes} but got {res.skill_code} for {q}"
        assert res.confidence >= 0.25
        assert res.needs_review is False
        assert len(res.top_candidates) >= 1
        assert res.top_candidates[0].skill_code in expected_codes


def test_pure_neural_extraction_pythagorean_theorem(extractor: HybridTopicExtractor):
    """Verify geometry and Pythagorean theorem queries extract SKILL_27 via neural embeddings."""
    queries = [
        "What is the Pythagorean theorem hypotenuse if legs are 3 and 4?",
        "How do I find the longest side of a right triangle using a squared plus b squared?",
        "How do I calculate the hypotenuse length from legs a and b?",
        "Why is c squared equal to a squared plus b squared in right angled triangles?",
    ]
    for q in queries:
        res = extractor.extract(q)
        assert res.method == "minilm_finetuned", f"Failed for query: {q}"
        assert res.skill_code == "SKILL_27", f"Expected SKILL_27 but got {res.skill_code} for {q}"
        assert res.confidence >= 0.28
        assert res.needs_review is False


def test_pure_neural_extraction_fractions(extractor: HybridTopicExtractor):
    """Verify fraction queries extract fraction operations skill."""
    queries = [
        "How do I add fractions with different denominators like 1/3 and 2/5?",
        "Do I need to find a common denominator before subtracting fractions?",
    ]
    for q in queries:
        res = extractor.extract(q)
        assert res.method == "minilm_finetuned"
        assert res.skill_code == "SKILL_292" or "fraction" in (res.canonical_skill_name or "").lower()
        assert res.needs_review is False


def test_non_math_queries_abstain_for_natural_redirection(extractor: HybridTopicExtractor):
    """Verify non-math chit-chat and greetings are cleanly classified as out-of-domain (abstain)."""
    general_queries = [
        "Hello",
        "Hi there, how are you doing today?",
        "Who are you and what is your name?",
        "Tell me a funny joke or story",
        "What is the capital city of France?",
        "Can you write a poem about the ocean?",
    ]
    for q in general_queries:
        res = extractor.extract(q)
        assert res.method == "abstain", f"Expected abstain for non-math query '{q}', but got {res.method}"
        assert res.needs_review is True
        assert res.skill_id is None
        assert res.canonical_skill_name is None


def test_empty_and_whitespace_abstention(extractor: HybridTopicExtractor):
    """Verify empty strings and whitespace safely return abstain."""
    assert extractor.extract("").method == "abstain"
    assert extractor.extract("   ").method == "abstain"
    assert extractor.extract("").needs_review is True
