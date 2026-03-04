"""Generate 12 bounded Tutor realization cases through the real finalizer path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
BACKEND_ROOT = WORKSPACE_ROOT / "adaptive-math-tutor" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.tutor.tutor_agent import TutorAgent  # noqa: E402
from app.schemas.dialogue import DialogueTurn  # noqa: E402
from app.schemas.tutor import TutorInput  # noqa: E402


MOVES = ("generic", "probing", "focus", "telling")
OUTPUT = HERE / "runtime_smoke_raw.json"
PROGRESS = HERE / "_runtime_smoke_progress.json"


STATES = (
    {
        "case_group": "negative_product",
        "problem": "Simplify (-3)(-5) + 2.",
        "topic": "arithmetic",
        "subtopic": "integer operations",
        "age": 12,
        "history": [
            {"role": "teacher", "content": "Show me how you simplified the expression."},
            {"role": "student", "content": "Negative times negative is negative, so I got -13."},
        ],
        "evidence": [
            {
                "action": "trusted_geometry_definition",
                "result": (
                    "The area of a rectangle is found by multiplying its "
                    "length by its width. Equivalently, length = area / width "
                    "when width is nonzero."
                ),
            },
            {
                "action": "final_claim_verification",
                "result": json.dumps(
                    [{
                        "check": 1,
                        "left_expression": "(-3)*(-5)+2",
                        "right_expression": "17",
                        "verified": True,
                    }],
                    separators=(",", ":"),
                ),
            }
        ],
    },
    {
        "case_group": "rectangle_factor",
        "problem": "A rectangle has area 48 square units and width 6 units. Find its length.",
        "topic": "geometry",
        "subtopic": "area of rectangles",
        "age": 11,
        "history": [
            {"role": "teacher", "content": "How would you find the missing length?"},
            {"role": "student", "content": "I think I should add 48 and 6."},
        ],
        "evidence": [
            {
                "action": "final_claim_verification",
                "result": json.dumps(
                    [{
                        "check": 1,
                        "left_expression": "48/6",
                        "right_expression": "8",
                        "verified": True,
                    }],
                    separators=(",", ":"),
                ),
            }
        ],
    },
    {
        "case_group": "equation_first_step",
        "problem": "Solve 2(x - 3) = 10.",
        "topic": "algebra",
        "subtopic": "linear equations",
        "age": 13,
        "history": [
            {"role": "teacher", "content": "Try the first step."},
            {"role": "student", "content": "I do not know how to begin. Please help me with the first step."},
        ],
        "evidence": [
            {
                "action": "final_claim_verification",
                "result": json.dumps(
                    [{
                        "check": 1,
                        "left_expression": "2*(8-3)",
                        "right_expression": "10",
                        "verified": True,
                    }],
                    separators=(",", ":"),
                ),
            }
        ],
    },
)


class ModelsProxy:
    def __init__(self, original: Any, counter: dict[str, int]) -> None:
        self._original = original
        self._counter = counter

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        self._counter["generate_content_attempts"] += 1
        return self._original.generate_content(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ClientProxy:
    def __init__(self, original: Any, counter: dict[str, int]) -> None:
        self._original = original
        self.models = ModelsProxy(original.models, counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def run() -> dict[str, Any]:
    recovered_failure: dict[str, Any] | None = None
    if PROGRESS.exists():
        saved = json.loads(PROGRESS.read_text(encoding="utf-8"))
        records = list(saved.get("records", []))
        counter = dict(saved.get("counter", {}))
        logical_by_schema = dict(saved.get("logical_by_schema", {}))
        # The interrupted rectangle/telling case exhausted exactly three
        # finalizer+verifier attempts. The traceback was emitted only after all
        # three audits returned, so these six completed requests are recovered
        # into the cumulative count before resuming.
        if len(records) == 7 and counter == {
            "logical_completion_calls": 14,
            "generate_content_attempts": 14,
        }:
            counter["logical_completion_calls"] += 6
            counter["generate_content_attempts"] += 6
            logical_by_schema["adaptmath_atomic_teacher_turn"] = (
                logical_by_schema.get("adaptmath_atomic_teacher_turn", 0) + 3
            )
            logical_by_schema["adaptmath_teaching_response_audit"] = (
                logical_by_schema.get("adaptmath_teaching_response_audit", 0) + 3
            )
            recovered_failure = {
                "case_group": "rectangle_factor",
                "selected_move": "telling",
                "logical_completion_calls": 6,
                "generate_content_attempts": 6,
                "result": "correct response rejected by contextual-formula verification",
            }
    else:
        records = []
        counter = {"logical_completion_calls": 0, "generate_content_attempts": 0}
        logical_by_schema = {}

    agent = TutorAgent()
    completions = agent.client.chat.completions
    original_create = completions.create

    def counted_create(*args: Any, **kwargs: Any) -> Any:
        counter["logical_completion_calls"] += 1
        schema = (
            kwargs.get("response_format", {})
            .get("json_schema", {})
            .get("name", "unknown")
        )
        logical_by_schema[schema] = logical_by_schema.get(schema, 0) + 1
        return original_create(*args, **kwargs)

    completions.create = counted_create
    completions._client = ClientProxy(completions._client, counter)

    completed_keys = {
        (record["case_group"], record["selected_move"]) for record in records
    }
    for state in STATES:
        for move in MOVES:
            if (state["case_group"], move) in completed_keys:
                continue
            tutor_input = TutorInput(
                question=state["problem"],
                topic=state["topic"],
                subtopic=state["subtopic"],
                student_age=state["age"],
                complexity_score=0.50,
                pedagogical_move=move,
                conversation_history=[
                    DialogueTurn.model_validate(turn) for turn in state["history"]
                ],
                previous_errors=[],
                reteaching=False,
            )
            before_logical = counter["logical_completion_calls"]
            before_attempts = counter["generate_content_attempts"]
            try:
                output = agent.teach_turn(tutor_input, list(state["evidence"]))
            except Exception as exc:
                PROGRESS.write_text(
                    json.dumps(
                        {
                            "records": records,
                            "counter": counter,
                            "logical_by_schema": logical_by_schema,
                            "last_failure": {
                                "case_group": state["case_group"],
                                "selected_move": move,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                raise
            record = {
                "case_group": state["case_group"],
                "problem": state["problem"],
                "history": state["history"],
                "selected_move": move,
                "tutor_response": output.teaching_response,
                "logical_completion_calls": (
                    counter["logical_completion_calls"] - before_logical
                ),
                "generate_content_attempts": (
                    counter["generate_content_attempts"] - before_attempts
                ),
            }
            records.append(record)
            PROGRESS.write_text(
                json.dumps(
                    {"records": records, "counter": counter, "logical_by_schema": logical_by_schema},
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )

    result = {
        "schema_version": "tutor_move_realization_runtime_smoke_v1",
        "generation_path": "TutorAgent.teach_turn -> finalizer -> TeachingResponseVerifier",
        "response_count": len(records),
        "external_api": "Gemini Developer API",
        **counter,
        "logical_calls_by_schema": logical_by_schema,
        "recovered_failed_attempt": recovered_failure,
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    PROGRESS.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "response_count", "logical_completion_calls", "generate_content_attempts",
        "logical_calls_by_schema",
    )}, indent=2))
