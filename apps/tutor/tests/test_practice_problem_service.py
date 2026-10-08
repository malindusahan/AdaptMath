from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.clients.memory.memory_client import MemoryTopicClassification
from app.services.practice_problem_service import (
    PracticeProblemGenerationError,
    PracticeProblemGenerator,
)


def _response(payload: dict):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


class FakeCompletions:
    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)

    def create(self, **kwargs):
        del kwargs
        return _response(self.payloads.pop(0))


class FakeClassifier:
    def __init__(self, topic: str):
        self.topic = topic

    def classify_question(self, *, question: str):
        assert question
        return MemoryTopicClassification(
            topic=self.topic,
            skill_id="SKILL_TEST",
            confidence=0.99,
            is_math=True,
            model_version="test",
        )


def _candidate():
    return {
        "question": "Which fraction is equivalent to 3/5: 6/10 or 6/15?",
        "expected_answer": "6/10",
    }


def _audit():
    return {
        "valid": True,
        "matches_target_skill": True,
        "self_contained": True,
        "unambiguous": True,
        "solvable": True,
        "answer_matches_question": True,
        "verification_checks": [
            {"left_expression": "3*10", "right_expression": "5*6"}
        ],
        "rejection_reason": "",
    }


def test_generated_problem_requires_all_validation_gates():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions([
        _candidate(),
        _audit(),
    ])))
    generator = PracticeProblemGenerator(
        client=client,
        model_name="test-model",
        max_attempts=1,
    )

    problem = generator.generate(
        target_skill="Equivalent Fractions",
        student_age=15,
        classifier=FakeClassifier("Equivalent Fractions"),
    )

    assert problem.question.startswith("Equivalent Fractions practice:")
    assert "Which fraction" in problem.question
    assert problem.expected_answer == "6/10"


def test_skill_classifier_mismatch_fails_closed():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions([
        _candidate(),
        _audit(),
    ])))
    generator = PracticeProblemGenerator(
        client=client,
        model_name="test-model",
        max_attempts=1,
    )

    with pytest.raises(PracticeProblemGenerationError):
        generator.generate(
            target_skill="Equivalent Fractions",
            student_age=15,
            classifier=FakeClassifier("Addition Whole Numbers"),
        )


def test_failed_independent_sympy_check_rejects_candidate():
    audit = _audit()
    audit["verification_checks"] = [
        {"left_expression": "3/5", "right_expression": "6/15"}
    ]
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions([
        _candidate(),
        audit,
    ])))
    generator = PracticeProblemGenerator(
        client=client,
        model_name="test-model",
        max_attempts=1,
    )

    with pytest.raises(PracticeProblemGenerationError):
        generator.generate(
            target_skill="Equivalent Fractions",
            student_age=15,
            classifier=FakeClassifier("Equivalent Fractions"),
        )
