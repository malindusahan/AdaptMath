"""Focused deterministic checks for the final move-to-language contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.tutor.realization_contract import (
    MOVE_REALIZATION_CONTRACT,
    MOVE_REALIZATION_PROFILES,
    build_move_realization_instruction,
)
from app.agents.tutor.tutor_agent import build_teacher_turn_schema


MOVES = ("generic", "probing", "focus", "telling")
ROOT = Path(__file__).resolve().parents[3]
PACKET = (
    ROOT
    / "packages"
    / "move-selection"
    / "results"
    / "tutor_move_realization_final_v1"
    / "test_cases.csv"
)

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
    assert "audit" not in schema["properties"]


def test_active_tutor_has_no_response_verifier_dependency() -> None:
    source = (
        ROOT
        / "apps"
        / "tutor"
        / "app"
        / "agents"
        / "tutor"
        / "tutor_agent.py"
    ).read_text(encoding="utf-8")
    assert "TeachingResponseVerifier" not in source
    assert "response_verifier" not in source
    assert "INLINE AUDIT CONTRACT" not in source


def test_realization_contract_has_no_state_to_move_rules() -> None:
    source = Path(
        ROOT
        / "apps"
        / "tutor"
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
