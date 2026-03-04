"""Focused deterministic checks for the final move-to-language contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.realization_contract import (
    MOVE_REALIZATION_CONTRACT,
    MOVE_REALIZATION_PROFILES,
    build_move_audit_instruction,
    build_move_realization_instruction,
)
from app.agents.tutor.response_verifier import TeachingResponseVerifier
from app.agents.tutor.tutor_agent import build_teacher_turn_schema


MOVES = ("generic", "probing", "focus", "telling")
ROOT = Path(__file__).resolve().parents[3]
PACKET = (
    ROOT
    / "pedagogical-move-selection"
    / "results"
    / "tutor_move_realization_final_v1"
    / "test_cases.csv"
)


def _response(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def _passing_payload() -> dict[str, object]:
    return {
        "checkable_claims": [],
        "logical_issues": [],
        "turn_scope_issues": [],
        "turn_scope_verdict": "pass",
        "move_alignment_verdict": "pass",
        "scope_reason": "One atomic move.",
        "move_alignment_reason": "Matches the selected move.",
    }


def test_profiles_cover_four_distinct_authoritative_moves() -> None:
    assert tuple(MOVE_REALIZATION_PROFILES) == MOVES
    assert set(MOVE_REALIZATION_CONTRACT) == set(MOVES)
    assert len(set(MOVE_REALIZATION_CONTRACT.values())) == 4


def test_generic_is_useful_contextual_and_not_forced_to_question() -> None:
    contract = MOVE_REALIZATION_CONTRACT["generic"]
    assert "context-aware" in contract
    assert "non-vacuous" in contract
    assert "question is optional" in contract
    assert "1-3" in contract
    assert "actual quantity" in contract


def test_consecutive_generic_is_allowed_but_stock_repetition_is_not() -> None:
    contract = MOVE_REALIZATION_CONTRACT["generic"]
    assert "consecutive turns" in contract
    assert "avoid repeating stock phrases" in contract
    assert "block" not in contract.casefold()


def test_probing_asks_one_meaningful_reasoning_question() -> None:
    contract = MOVE_REALIZATION_CONTRACT["probing"]
    assert "ONE clear, specific main mathematical" in contract
    assert "learner must" in contract
    assert "Do not answer the question" in contract


def test_focus_is_between_probing_and_telling() -> None:
    contract = MOVE_REALIZATION_CONTRACT["focus"]
    assert "more direct than probing but less" in contract
    assert "leave meaningful mathematical work" in contract
    assert "does not have to be" in contract
    assert "specific sign" in contract


def test_telling_explains_before_optional_question() -> None:
    contract = MOVE_REALIZATION_CONTRACT["telling"]
    assert "information FIRST" in contract
    assert "2-5" in contract
    assert "only after" in contract
    assert "learner already supplied" in contract


@pytest.mark.parametrize("move", MOVES)
def test_private_generation_instruction_preserves_selected_move(move: str) -> None:
    instruction = build_move_realization_instruction(move)
    assert f"Selected pedagogical move: {move}" in instruction
    assert "authoritative" in instruction.casefold()
    assert "Do not expose its raw label" in instruction


@pytest.mark.parametrize("move", MOVES)
def test_structured_schema_cannot_substitute_selected_move(move: str) -> None:
    schema = build_teacher_turn_schema(move)
    assert schema["properties"]["selected_move"]["enum"] == [move]


@pytest.mark.parametrize("move", MOVES)
def test_verifier_receives_exact_selected_move_contract(move: str) -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: _response(_passing_payload()))
        )
    )
    verifier = TeachingResponseVerifier(client, "mock", SymPyMathVerifier())
    payload = verifier._extract_audit_payload(
        question="What is 3 + 4?",
        verified_evidence="3 + 4 = 7",
        teaching_response="Keep the two addends in view as you continue.",
        pedagogical_move=move,
        conversation_history=[],
    )
    assert payload["move_alignment_verdict"] == "pass"
    guidance = build_move_audit_instruction(move)
    assert f"Selected move: {move}" in guidance
    assert "Length guidance:" in guidance


def test_verifier_prompt_rejects_clear_move_mismatches() -> None:
    prompt = TeachingResponseVerifier.audit_prompt
    assert "Fail a worked solution followed by a token question" in prompt
    assert "Fail a response that only asks the learner what to do" in prompt
    assert "Fail\n  vacuous praise" in prompt
    assert "must be useful rather than merely" in prompt


def test_realization_contract_has_no_state_to_move_rules() -> None:
    source = Path(
        ROOT
        / "adaptive-math-tutor"
        / "backend"
        / "app"
        / "agents"
        / "tutor"
        / "realization_contract.py"
    ).read_text(encoding="utf-8")
    forbidden = ("TurnLinTS", "select_arm", "mastery_before", "uncertainty_probability")
    assert all(token not in source for token in forbidden)


def test_deterministic_packet_contains_ten_cases_per_move() -> None:
    assert PACKET.is_file()
    rows = PACKET.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 41
    for move in MOVES:
        assert sum(f",{move}," in row for row in rows[1:]) == 10
