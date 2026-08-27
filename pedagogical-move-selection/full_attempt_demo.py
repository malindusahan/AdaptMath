"""Interactive synthetic harness for the frozen v3 adaptive tutor pipeline.

``DemoTutor`` is deliberately deterministic and scenario-specific. It exists
only to exercise the integration path

    MD6 -> LinTS/overlay -> pedagogical move -> response -> MRB1 -> next turn

without pretending to be the production Tutor Agent. The production agent
will realize pedagogical strategies dynamically.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


DEFAULT_PROBLEM = (
    "A school library has 6 shelves. Each shelf holds 24 books. The librarian "
    "removes 18 books and then shares the remaining books equally among 9 "
    "students. How many books does each student receive? Explain your reasoning."
)
DEFAULT_INITIAL_STUDENT_MESSAGE = (
    "I think I should first find how many books are on all 6 shelves."
)
DEMO_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "self_improvement"
    / "full_attempt_demo_v3"
)
DEMO_LOG_PATH = DEMO_OUTPUT_DIR / "demo_synthetic_attempts_v3.jsonl"
DEMO_STATE_PATH = DEMO_OUTPUT_DIR / "demo_synthetic_policy_state_v3.json"
DEMO_TUTOR_TURNS = 3


class DialogueStage(str, Enum):
    """Simple stages inferred only from authoritative conversation history."""

    START = "start"
    TOTAL_BOOKS = "total_books"
    REMOVE_BOOKS = "remove_books"
    SHARE_BOOKS = "share_books"
    EXPLAIN = "explain"


@dataclass(frozen=True, slots=True)
class DialogueState:
    stage: DialogueStage
    latest_student: str
    previous_teacher: str


@dataclass(frozen=True, slots=True)
class DemoProblemFacts:
    shelves: int
    books_per_shelf: int
    removed_books: int
    students: int
    total_books: int
    remaining_books: int
    books_per_student: int


def infer_problem_facts(problem: str) -> DemoProblemFacts:
    """Extract the canonical shelf/remove/share quantities from the problem."""

    numbers = [int(value) for value in re.findall(r"\b\d+\b", problem)]
    if len(numbers) < 4:
        raise ValueError(
            "The deterministic demo scenario needs four integer quantities: "
            "shelves, books per shelf, removed books, and students."
        )
    shelves, books_per_shelf, removed_books, students = numbers[:4]
    if min(shelves, books_per_shelf, students) <= 0 or removed_books < 0:
        raise ValueError("Demo problem quantities must be nonnegative and valid.")
    total_books = shelves * books_per_shelf
    remaining_books = total_books - removed_books
    if remaining_books < 0 or remaining_books % students != 0:
        raise ValueError(
            "The deterministic demo requires a nonnegative remaining total "
            "that divides equally among the students."
        )
    return DemoProblemFacts(
        shelves=shelves,
        books_per_shelf=books_per_shelf,
        removed_books=removed_books,
        students=students,
        total_books=total_books,
        remaining_books=remaining_books,
        books_per_student=remaining_books // students,
    )


def _latest_text(
    conversation_history: Sequence[Mapping[str, object]],
    user: str,
) -> str:
    for turn in reversed(conversation_history):
        if turn.get("user") == user:
            text = turn.get("text")
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())
    return ""


def _contains_number(text: str, number: int) -> bool:
    return re.search(rf"\b{number}\b", text) is not None


def infer_dialogue_state(
    conversation_history: Sequence[Mapping[str, object]],
    problem_facts: DemoProblemFacts,
) -> DialogueState:
    """Infer canonical demo progress from the complete student dialogue."""

    student_messages = [
        " ".join(text.split())
        for turn in conversation_history
        if turn.get("user") == "student"
        and isinstance((text := turn.get("text")), str)
        and text.strip()
    ]
    combined_student_text = " ".join(student_messages).lower()
    latest_student = student_messages[-1] if student_messages else ""
    previous_teacher = _latest_text(conversation_history, "teacher")

    if _contains_number(combined_student_text, problem_facts.books_per_student):
        stage = DialogueStage.EXPLAIN
    elif _contains_number(combined_student_text, problem_facts.remaining_books):
        stage = DialogueStage.SHARE_BOOKS
    elif _contains_number(combined_student_text, problem_facts.total_books):
        stage = DialogueStage.REMOVE_BOOKS
    elif any(
        marker in combined_student_text
        for marker in (
            f"all {problem_facts.shelves} shelves",
            "total number of books",
            "find how many books",
            f"multiply {problem_facts.shelves}",
            f"{problem_facts.shelves} x {problem_facts.books_per_shelf}",
            f"{problem_facts.shelves} * {problem_facts.books_per_shelf}",
        )
    ):
        stage = DialogueStage.TOTAL_BOOKS
    else:
        stage = DialogueStage.START

    return DialogueState(
        stage=stage,
        latest_student=latest_student,
        previous_teacher=previous_teacher,
    )


class DemoTutor:
    """Realize each pedagogical strategy for the inferred dialogue stage."""

    CANONICAL_MOVES = ("generic", "probing", "focus", "telling")

    def __init__(self, *, verbose: bool = True) -> None:
        self.verbose = verbose

    @staticmethod
    def _problem_excerpt(problem: str, limit: int = 220) -> str:
        compact = " ".join(problem.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1].rstrip() + "…"

    @staticmethod
    def _generic(
        stage: DialogueStage,
        latest_student: str,
        problem: str,
        facts: DemoProblemFacts,
    ) -> str:
        responses = {
            DialogueStage.START: (
                f"How would you start solving this problem: “{problem}”? "
                "Explain your current thinking."
            ),
            DialogueStage.TOTAL_BOOKS: (
                "You have identified the first step. What total do you get for "
                f"the books on all {facts.shelves} shelves, and how did you "
                "calculate it?"
            ),
            DialogueStage.REMOVE_BOOKS: (
                "You found the total number of books. What would you do next, "
                "and why?"
            ),
            DialogueStage.SHARE_BOOKS: (
                "You now know how many books remain. What should you do next?"
            ),
            DialogueStage.EXPLAIN: (
                f"You answered “{latest_student}” Can you explain the sequence "
                "of calculations that led to that result?"
            ),
        }
        return responses[stage]

    @staticmethod
    def _probing(
        stage: DialogueStage,
        latest_student: str,
        problem: str,
        facts: DemoProblemFacts,
    ) -> str:
        del latest_student, problem
        responses = {
            DialogueStage.START: (
                "Which two quantities would you use first, and what operation "
                "connects them?"
            ),
            DialogueStage.TOTAL_BOOKS: (
                f"What is the total number of books on the {facts.shelves} shelves?"
            ),
            DialogueStage.REMOVE_BOOKS: (
                f"If {facts.removed_books} books are removed from "
                f"{facts.total_books}, how many books remain?"
            ),
            DialogueStage.SHARE_BOOKS: (
                f"If {facts.remaining_books} books are shared equally among "
                f"{facts.students} students, how many does each student receive?"
            ),
            DialogueStage.EXPLAIN: (
                "Can you explain the calculations that led you to "
                f"{facts.books_per_student} books per student?"
            ),
        }
        return responses[stage]

    @staticmethod
    def _focus(
        stage: DialogueStage,
        latest_student: str,
        problem: str,
        facts: DemoProblemFacts,
    ) -> str:
        del latest_student, problem
        responses = {
            DialogueStage.START: (
                f"Focus first on the {facts.shelves} shelves and "
                f"{facts.books_per_shelf} books per shelf. What calculation "
                "gives the total number of books?"
            ),
            DialogueStage.TOTAL_BOOKS: (
                f"Focus on combining the {facts.shelves} equal shelves of "
                f"{facts.books_per_shelf} books. What calculation represents "
                "those equal groups?"
            ),
            DialogueStage.REMOVE_BOOKS: (
                f"Focus on the {facts.removed_books} books that are removed. "
                f"What operation should you use with {facts.total_books} and "
                f"{facts.removed_books}?"
            ),
            DialogueStage.SHARE_BOOKS: (
                "Now focus on the phrase 'shared equally among "
                f"{facts.students} students.' What operation does that suggest?"
            ),
            DialogueStage.EXPLAIN: (
                "Focus on the order of the three steps: total the shelf books, "
                f"remove {facts.removed_books}, and then share equally among "
                f"{facts.students}. Explain why that order fits the problem."
            ),
        }
        return responses[stage]

    @staticmethod
    def _telling(
        stage: DialogueStage,
        latest_student: str,
        problem: str,
        facts: DemoProblemFacts,
    ) -> str:
        del latest_student, problem
        responses = {
            DialogueStage.START: (
                f"First multiply {facts.shelves} by {facts.books_per_shelf} "
                "to find the total number of books."
            ),
            DialogueStage.TOTAL_BOOKS: (
                f"Multiply {facts.shelves} by {facts.books_per_shelf} now; that "
                "gives the total number of books before any are removed."
            ),
            DialogueStage.REMOVE_BOOKS: (
                f"Next subtract the {facts.removed_books} removed books from "
                f"{facts.total_books}."
            ),
            DialogueStage.SHARE_BOOKS: (
                f"Now divide {facts.remaining_books} by {facts.students} to find "
                "how many books each student receives."
            ),
            DialogueStage.EXPLAIN: (
                f"The result is {facts.books_per_student} books per student. "
                "Explain the three calculations you used: multiply, subtract, "
                "then divide."
            ),
        }
        return responses[stage]

    @staticmethod
    def _nonrepeating_follow_up(
        pedagogical_move: str,
        latest_student: str,
    ) -> str:
        student_reference = latest_student or "your current idea"
        follow_ups = {
            "generic": (
                f"Build on “{student_reference}” and describe one concrete next "
                "step you can take."
            ),
            "probing": (
                f"You replied “{student_reference}” Which specific numbers and "
                "operation will you use for this step?"
            ),
            "focus": (
                f"Look again at “{student_reference}” and focus on the one "
                "quantity that must be calculated next."
            ),
            "telling": (
                "Carry out that instructed calculation now and state the result "
                "before moving to the following step."
            ),
        }
        return follow_ups[pedagogical_move]

    def generate(
        self,
        *,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
        pedagogical_move: str,
    ) -> str:
        if pedagogical_move not in self.CANONICAL_MOVES:
            raise ValueError(
                f"Unknown pedagogical move: {pedagogical_move!r}."
            )
        if self.verbose:
            print("\n[TUTOR AGENT RECEIVED]")
            print("pedagogical_move =", pedagogical_move)

        problem_facts = infer_problem_facts(problem)
        dialogue = infer_dialogue_state(conversation_history, problem_facts)
        problem_excerpt = self._problem_excerpt(problem)
        realizers = {
            "generic": self._generic,
            "probing": self._probing,
            "focus": self._focus,
            "telling": self._telling,
        }
        response = realizers[pedagogical_move](
            dialogue.stage,
            dialogue.latest_student,
            problem_excerpt,
            problem_facts,
        )

        if (
            dialogue.previous_teacher
            and response.strip() == dialogue.previous_teacher.strip()
        ):
            response = self._nonrepeating_follow_up(
                pedagogical_move,
                dialogue.latest_student,
            )
        return response


def _history(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"user": user, "text": text} for user, text in turns]


def run_demo_tutor_smoke_check() -> None:
    """Validate deterministic dialogue progression without loading MD6/MRB1."""

    tutor = DemoTutor(verbose=False)
    total_stage = _history(
        ("student", DEFAULT_INITIAL_STUDENT_MESSAGE),
    )
    remove_stage = _history(
        ("student", DEFAULT_INITIAL_STUDENT_MESSAGE),
        ("teacher", "What is the total number of books on the six shelves?"),
        ("student", "144 books"),
    )
    share_stage = _history(
        ("student", DEFAULT_INITIAL_STUDENT_MESSAGE),
        ("teacher", "What is the total number of books on the six shelves?"),
        ("student", "144 books"),
        ("teacher", "How many remain after 18 are removed?"),
        ("student", "126 books"),
    )
    explain_stage = _history(
        *[(turn["user"], turn["text"]) for turn in share_stage],
        ("teacher", "How many does each student receive?"),
        ("student", "14 books"),
    )

    probing = [
        tutor.generate(
            problem=DEFAULT_PROBLEM,
            conversation_history=history,
            pedagogical_move="probing",
        )
        for history in (total_stage, remove_stage, share_stage, explain_stage)
    ]
    assert len(set(probing)) == 4
    assert "total number" in probing[0].lower()
    assert "18" in probing[1]
    assert "9" in probing[2]
    assert "explain" in probing[3].lower()

    focus = tutor.generate(
        problem=DEFAULT_PROBLEM,
        conversation_history=remove_stage,
        pedagogical_move="focus",
    )
    telling = tutor.generate(
        problem=DEFAULT_PROBLEM,
        conversation_history=share_stage,
        pedagogical_move="telling",
    )
    assert focus != probing[1]
    assert "focus" in focus.lower()
    assert "divide" in telling.lower()

    first = probing[0]
    repeated_stage = _history(
        ("student", DEFAULT_INITIAL_STUDENT_MESSAGE),
        ("teacher", first),
        ("student", "I still need help finding the total."),
    )
    second = tutor.generate(
        problem=DEFAULT_PROBLEM,
        conversation_history=repeated_stage,
        pedagogical_move="probing",
    )
    assert second != first

    print("DEMO TUTOR SMOKE CHECK: PASS")
    print("- probing progression: PASS")
    print("- focus differs from probing: PASS")
    print("- telling gives explicit guidance: PASS")
    print("- consecutive exact repetition avoided: PASS")


def print_turn(result, policy, turn_feature_names: Sequence[str]) -> None:
    print("\n" + "=" * 78)
    print(f"TURN {result['turn_index']}")
    print("=" * 78)

    print("\n[MD6 OUTPUT]")
    for name, value in result["md6_probabilities"].items():
        print(f"{name:10s}: {value:.6f}")

    print("\nMD6 base move :", result["base_move"])

    print("\n[9-D LINTS CONTEXT]")
    for name, value in zip(turn_feature_names, result["context"]):
        print(f"{name:35s}: {value:.6f}")

    print("\n[POLICY DECISION]")
    print("Eligible arms :", result["eligible_arms"])
    print("Selected arm  :", result["selected_arm"])
    print("Base move     :", result["base_move"])
    print("Overridden    :", result["overridden"])
    print("FINAL MOVE    :", result["pedagogical_move"])

    print("\n[TUTOR RESPONSE]")
    print(result["tutor_response"])

    print("\n[MRB1 OUTPUT]")
    for name, value in result["mrb1_scores"].items():
        print(f"{name:28s}: {value:.6f}")

    print("\nPosterior updates so far:", policy.total_updates)


def main() -> None:
    # Runtime/model imports are intentionally local so --smoke-check remains
    # deterministic and does not instantiate or load MD6/MRB1.
    from src.self_improvement.adaptive_tutor_pipeline import AdaptiveTutorPipeline
    from src.self_improvement.experience_logger import ExperienceLogger
    from src.self_improvement.lints_policy import TrueDisjointLinTS
    from src.self_improvement.md6_inference import FrozenMD6Inference
    from src.self_improvement.mrb1_inference import FrozenMRB1Inference
    from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES
    from src.self_improvement.turn_level_controller import (
        TurnLevelAttemptController,
    )

    print("DEFAULT DEMO PROBLEM")
    print(DEFAULT_PROBLEM)
    entered_problem = input(
        "\nProblem [press Enter to use the default above]: "
    ).strip()
    problem = entered_problem or DEFAULT_PROBLEM

    entered_student_message = input(
        "Initial student message "
        f"[press Enter for: {DEFAULT_INITIAL_STUDENT_MESSAGE}]: "
    ).strip()
    student_message = (
        entered_student_message or DEFAULT_INITIAL_STUDENT_MESSAGE
    )
    mastery_before = float(input("Mastery before [0-1]: ").strip())

    memory = {
        "attempt_id": "manual-synthetic-demo-attempt-001",
        "problem": problem,
        "conversation_history": [
            {"user": "student", "text": student_message}
        ],
        "mastery_before": mastery_before,
    }

    print("\nLoading frozen MD6...")
    md6 = FrozenMD6Inference()

    print("Loading frozen MRB1...")
    mrb1 = FrozenMRB1Inference()

    policy = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=20260824,
        data_mode="synthetic",
    )
    controller = TurnLevelAttemptController(policy)
    logger = ExperienceLogger(
        DEMO_LOG_PATH,
        data_mode="synthetic",
        source_policy=policy,
    )
    pipeline = AdaptiveTutorPipeline(
        md6=md6,
        mrb1=mrb1,
        controller=controller,
        experience_logger=logger,
        policy_state_path=DEMO_STATE_PATH,
    )
    tutor = DemoTutor()

    print("\n" + "=" * 78)
    print("ATTEMPT START")
    print("=" * 78)
    print("Problem        :", problem)
    print("Mastery before :", mastery_before)
    print("Data mode      : synthetic (demo-only lineage)")
    print("Demo log path  :", DEMO_LOG_PATH)
    print("Demo state path:", DEMO_STATE_PATH)

    pipeline.start_attempt(memory)
    updates_at_attempt_start = policy.total_updates
    print("\nPosterior updates at start:", updates_at_attempt_start)

    for turn_number in range(1, DEMO_TUTOR_TURNS + 1):
        result = pipeline.run_tutor_turn(memory, tutor)
        print_turn(result, policy, TURN_FEATURE_NAMES)

        if policy.total_updates != updates_at_attempt_start:
            raise RuntimeError(
                "DEMO VALIDATION FAILED: LinTS updated before attempt completion."
            )

        print("\n" + "-" * 78)
        next_student = input(
            f"Enter student reply after Turn {turn_number}: "
        ).strip()
        if next_student:
            memory["conversation_history"].append(
                {"user": "student", "text": next_student}
            )
            print("\nStudent response added to authoritative memory.")
        else:
            print("\nNo student response entered; history was not extended.")

    print("\n" + "=" * 78)
    print("FINAL CONVERSATION HISTORY")
    print("=" * 78)
    print(json.dumps(
        memory["conversation_history"],
        indent=2,
        ensure_ascii=False,
    ))

    print("\nPosterior updates BEFORE attempt completion:")
    print(policy.total_updates)
    print("Expected:", updates_at_attempt_start)
    print("No within-attempt updates: CONFIRMED")

    print("\n" + "=" * 78)
    print("ATTEMPT COMPLETION")
    print("=" * 78)
    print(
        "DEMO ONLY: learning_outcome is manually supplied; this is not a real "
        "BKT/Malindu output."
    )

    skill = input("Skill: ").strip()
    mastery_after = float(input("Mastery after [0-1]: ").strip())
    delta_mastery = float(
        input("Manually supplied signed delta mastery [-1,1]: ").strip()
    )
    completion = pipeline.finish_attempt(
        memory,
        learning_outcome={
            "skill": skill,
            "mastery_before": mastery_before,
            "mastery_after": mastery_after,
            "delta_mastery": delta_mastery,
        },
        metadata={"source": "manual-synthetic-full-attempt-demo"},
    )

    print("\n[COMPLETION RESULT]")
    print("Skill                  :", completion["skill"])
    print("Signed mastery reward  :", completion["reward"])
    print("Mastery before         :", completion["mastery_before"])
    print("Mastery after          :", completion["mastery_after"])
    print("Mastery delta          :", completion["mastery_delta"])
    print("Tutor turn count       :", completion["turn_count"])
    print("Weight per turn        :", completion["sample_weight_per_turn"])
    print("Total attempt weight   :", completion["total_attempt_weight"])

    expected_updates_after_completion = (
        updates_at_attempt_start + completion["turn_count"]
    )
    print("\nPosterior updates AFTER completion:")
    print(policy.total_updates)
    print("Expected:", expected_updates_after_completion)
    if policy.total_updates != expected_updates_after_completion:
        raise RuntimeError(
            "DEMO VALIDATION FAILED: completion did not apply exactly T updates."
        )
    print("Exactly T delayed updates after completion: CONFIRMED")

    print("\n[EXPERIENCE RECORD]")
    print(json.dumps(
        completion["experience_record"],
        indent=2,
        ensure_ascii=False,
    ))

    print("\n[POLICY STATE]")
    print("Path :", DEMO_STATE_PATH)
    print("Saved:", DEMO_STATE_PATH.is_file())

    print("\n[EXPERIENCE LOG]")
    print("Path :", DEMO_LOG_PATH)
    print("Saved:", DEMO_LOG_PATH.is_file())

    print("\n" + "=" * 78)
    print("FULL SYNTHETIC DEMO ATTEMPT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke-check"]:
        run_demo_tutor_smoke_check()
    elif sys.argv[1:]:
        raise SystemExit("Usage: python full_attempt_demo.py [--smoke-check]")
    else:
        main()
