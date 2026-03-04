"""Opt-in controlled live MD7-R1 test with frozen MD6 shadow inference.

This diagnostic uses the real integrated Tutor turn path and MRB1 scorer.  It
never completes an adaptive attempt: every attempt exits through the existing
abort path, which preserves inference-time LinTS sampling while preventing
posterior updates, experience logging, and policy-state persistence.

Run explicitly with ``--run-live``.  Importing this module has no live effects.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import patch
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
BACKEND_ROOT = WORKSPACE_ROOT / "adaptive-math-tutor" / "backend"
STUDENT_MODEL_ROOT = WORKSPACE_ROOT / "student-modeling"
MD6_DIR = PROJECT_ROOT / "models" / "frozen" / "md6"
MD7_DIR = PROJECT_ROOT / "models" / "candidates" / "md7r1_epoch3"
OUTPUT_DIR = PROJECT_ROOT / "results" / "md7r1_controlled_live_v1"
MOVE_ORDER: Final[tuple[str, ...]] = (
    "generic",
    "probing",
    "focus",
    "telling",
)
GAP_THRESHOLD: Final[float] = 0.10

for import_root in (BACKEND_ROOT, STUDENT_MODEL_ROOT, PROJECT_ROOT):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)


from app.agents.tutor.tutor_agent import TutorAgent  # noqa: E402
from app.integrations.adaptive_component_coordinator import (  # noqa: E402
    AdaptiveAttemptComponents,
    AdaptiveComponentCoordinator,
    TutorAssessmentStudentModelPipeline,
)
from app.integrations.tutor_state_memory_adapter import (  # noqa: E402
    TutorStateMemoryAdapter,
)
from app.schemas.dialogue import DialogueTurn  # noqa: E402
from app.schemas.tutor import TutorMathInput  # noqa: E402
from bkt.predict import BKTPredictor  # noqa: E402
from core.cross_session_pipeline import CrossSessionStudentModelPipeline  # noqa: E402
from core.detector_service import DetectorService  # noqa: E402
import core.knowledge_graph as knowledge_graph_module  # noqa: E402
from core.knowledge_graph import KnowledgeGraph  # noqa: E402
from db.database import get_connection, initialise_database  # noqa: E402
from src.integration.student_model_v3_bridge import (  # noqa: E402
    StudentModelFrozenV3Bridge,
)
from src.self_improvement.adaptive_tutor_pipeline import (  # noqa: E402
    AdaptiveTutorPipeline,
)
from src.self_improvement.conservative_overlay import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD,
)
from src.self_improvement.controlled_live_selector import (  # noqa: E402
    ActiveMD7WithMD6Shadow,
)
from src.self_improvement.experience_logger import ExperienceLogger  # noqa: E402
from src.self_improvement.lints_policy import TrueDisjointLinTS  # noqa: E402
from src.self_improvement.md6_inference import FrozenMD6Inference  # noqa: E402
from src.self_improvement.mrb1_inference import FrozenMRB1Inference  # noqa: E402
from src.self_improvement.state_io import load_policy_state  # noqa: E402
from src.self_improvement.turn_context_builder import TURN_FEATURE_NAMES  # noqa: E402
from src.self_improvement.turn_level_controller import (  # noqa: E402
    TurnLevelAttemptController,
)


ATTEMPT_SPECS: Final[tuple[dict[str, object], ...]] = (
    {
        "name": "unit_rate_short_and_correct",
        "skill": "Unit Rate",
        "problem": (
            "A plant grows 3 centimeters every 5 days at a constant rate. "
            "How many days will it take to grow 9 centimeters?"
        ),
        "topic": "Rates and proportional reasoning",
        "subtopic": "Unit rate",
        "complexity_score": 0.35,
        "turns": (
            {
                "case": "opening_question",
                "student_text": None,
                "plausible_moves": ("generic",),
            },
            {
                "case": "short_numeric_answer",
                "student_text": "15 days",
                "plausible_moves": ("probing",),
            },
            {
                "case": "short_ambiguous_answer",
                "student_text": "Maybe.",
                "plausible_moves": ("probing",),
            },
            {
                "case": "fully_correct_reasoning",
                "student_text": (
                    "Nine centimeters is three groups of 3 centimeters, and "
                    "each group takes 5 days, so 3 times 5 is 15 days."
                ),
                "plausible_moves": ("generic", "focus", "telling"),
            },
        ),
    },
    {
        "name": "percent_errors_and_uncertainty",
        "skill": "Percent Of",
        "problem": "What is 30 percent of 80?",
        "topic": "Percentages",
        "subtopic": "Percent of a quantity",
        "complexity_score": 0.30,
        "turns": (
            {
                "case": "opening_question",
                "student_text": None,
                "plausible_moves": ("generic",),
            },
            {
                "case": "specific_arithmetic_error",
                "student_text": "30% of 80 is 18 because 0.3 times 80 is 18.",
                "plausible_moves": ("focus",),
            },
            {
                "case": "clear_misconception",
                "student_text": (
                    "Thirty percent means subtract 30 from 80, so the answer is 50."
                ),
                "plausible_moves": ("focus",),
            },
            {
                "case": "early_i_dont_know",
                "student_text": "I don't know.",
                "plausible_moves": ("probing",),
            },
        ),
    },
    {
        "name": "repeated_failure_and_help",
        "skill": "Percent Of",
        "problem": (
            "A class has 40 students. If 35 percent are absent, how many "
            "students are absent?"
        ),
        "topic": "Percentages",
        "subtopic": "Percent of a quantity",
        "complexity_score": 0.40,
        "turns": (
            {
                "case": "opening_question",
                "student_text": None,
                "plausible_moves": ("generic",),
            },
            {
                "case": "initial_failed_attempt",
                "student_text": (
                    "I think it is 26 because 35 plus 40 is 75 and then I "
                    "subtracted something."
                ),
                "plausible_moves": ("probing", "focus"),
            },
            {
                "case": "repeated_failure_after_hint",
                "student_text": "I still get 26. The hint didn't help me.",
                "plausible_moves": ("focus", "telling"),
            },
            {
                "case": "explicit_direct_help_request",
                "student_text": "Just tell me - show me the next step.",
                "plausible_moves": ("telling",),
            },
        ),
    },
)


class DiagnosticFailure(RuntimeError):
    """Raised when a controlled-live safety invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(root: Path) -> dict[str, dict[str, object]]:
    require(root.is_dir(), f"Required directory does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def selected_files_snapshot(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if not resolved.exists():
            result[key] = {"exists": False}
        else:
            require(resolved.is_file(), f"Expected a regular file: {resolved}")
            stat = resolved.stat()
            result[key] = {
                "exists": True,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(resolved),
            }
    return result


def changed_entry_count(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> int:
    return sum(
        before.get(key) != after.get(key)
        for key in set(before) | set(after)
    )


def posterior_snapshot(policy: TrueDisjointLinTS) -> dict[str, object]:
    state = policy.state_dict()
    return {
        "A": state["A"],
        "b": state["b"],
        "arm_update_counts": state["arm_update_counts"],
        "total_updates": state["total_updates"],
    }


class StartOnlyEvidenceSource:
    """Production-compatible assessment boundary that this test never calls."""

    def __init__(self, skill: str) -> None:
        self.context = SimpleNamespace(assessed_skills=(skill,))

    def extract(self, transcript: list[dict[str, object]]) -> dict[str, object]:
        raise DiagnosticFailure(
            f"Assessment extraction was not authorized ({len(transcript)} turns)."
        )

    def __call__(self, *_: object, **__: object) -> object:
        raise DiagnosticFailure("Assessment evaluation was not authorized.")


def probability_diagnostics(probabilities: Mapping[str, float]) -> dict[str, object]:
    ranked = sorted(
        ((move, float(probabilities[move])) for move in MOVE_ORDER),
        key=lambda item: (-item[1], MOVE_ORDER.index(item[0])),
    )
    return {
        "argmax": ranked[0][0],
        "top1_probability": ranked[0][1],
        "top2_probability": ranked[1][1],
        "top1_top2_gap": float(ranked[0][1] - ranked[1][1]),
    }


def latest_student_text(history: Sequence[Mapping[str, object]]) -> str | None:
    for turn in reversed(history):
        if turn.get("user") == "student":
            value = turn.get("text")
            return str(value) if value is not None else None
    return None


KNOWN_RISK_MOVE: Final[dict[str, str]] = {
    "short_numeric_answer": "focus",
    "specific_arithmetic_error": "generic",
    "clear_misconception": "generic",
    "early_i_dont_know": "generic",
}


def semantic_judgment(
    *,
    case: str,
    plausible_moves: Sequence[str],
    md6_move: str,
    md7_move: str,
) -> tuple[str, str]:
    plausible = set(plausible_moves)
    known_risk = KNOWN_RISK_MOVE.get(case)
    if md7_move == md6_move:
        return (
            "UNCHANGED",
            "MD7 and MD6 select the same raw move; this is not a gold-label claim.",
        )
    if md7_move in plausible and md6_move not in plausible:
        return (
            "PLAUSIBLY IMPROVED",
            f"MD7 moved into the scenario-plausible set {sorted(plausible)}.",
        )
    if known_risk == md7_move:
        return (
            "POSSIBLE NEW BIAS",
            f"MD7 reproduced the watched {case} -> {md7_move} risk transition.",
        )
    if md7_move not in plausible:
        return (
            "QUESTIONABLE",
            f"MD7 raw move is outside the diagnostic plausible set {sorted(plausible)}.",
        )
    return (
        "UNCHANGED",
        "Both raw moves remain qualitatively plausible; no gold label is asserted.",
    )


def make_state(
    *,
    spec: Mapping[str, object],
    student_id: str,
    thread_id: str,
    attempt_id: str,
    verified_math_evidence: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "student_id": student_id,
        "thread_id": thread_id,
        "attempt_id": attempt_id,
        "adaptive_attempt_index": 1,
        "target_skill": str(spec["skill"]),
        "adaptive_lifecycle_status": "not_started",
        "question": str(spec["problem"]),
        "topic": str(spec["topic"]),
        "subtopic": str(spec["subtopic"]),
        "age": 14,
        "complexity_score": float(spec["complexity_score"]),
        "planner_output": {},
        "previous_errors": [],
        "teaching_phase": "initial",
        "verified_math_evidence": copy.deepcopy(verified_math_evidence),
        "conversation_history": [],
        "turn_count": 0,
    }


def append_student_turn(state: dict[str, object], text: str) -> None:
    history = state.get("conversation_history")
    require(isinstance(history, list), "TutorState history is not a mutable list.")
    history.append(DialogueTurn(role="student", content=text).model_dump())


def production_paths() -> tuple[Path, Path]:
    runtime = BACKEND_ROOT / "runtime"
    policy_path = Path(
        os.getenv(
            "ADAPTIVE_POLICY_STATE_PATH",
            str(runtime / "frozen_v3_policy_state.json"),
        )
    ).resolve()
    experience_path = Path(
        os.getenv(
            "ADAPTIVE_EXPERIENCE_LOG_PATH",
            str(runtime / "frozen_v3_experience.jsonl"),
        )
    ).resolve()
    return policy_path, experience_path


def distribution(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    counts = Counter(str(row[field]) for row in rows)
    total = len(rows)
    return {
        "total": total,
        "counts": {move: int(counts[move]) for move in MOVE_ORDER},
        "percentages": {
            move: (100.0 * counts[move] / total if total else 0.0)
            for move in MOVE_ORDER
        },
    }


def transition_matrix(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    matrix = {
        source: {target: 0 for target in MOVE_ORDER}
        for source in MOVE_ORDER
    }
    for row in rows:
        matrix[str(row["md6_argmax"])][str(row["md7_argmax"])] += 1
    return matrix


def collapse_risk(share: float) -> str:
    if share >= 80.0:
        return "HIGH"
    if share >= 50.0:
        return "MEDIUM"
    return "LOW"


def summarize_overlay(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(rows)
    baseline_only = sum(len(row["eligible_arms_at_0_10"]) == 1 for row in rows)

    def group_summary(group: Sequence[Mapping[str, object]]) -> dict[str, object]:
        n = len(group)
        only = sum(len(row["eligible_arms_at_0_10"]) == 1 for row in group)
        return {
            "n": n,
            "baseline_only_count": only,
            "baseline_only_percentage": 100.0 * only / n if n else None,
            "alternative_available_count": n - only,
            "alternative_available_percentage": (
                100.0 * (n - only) / n if n else None
            ),
            "mean_eligible_arms": (
                sum(len(row["eligible_arms_at_0_10"]) for row in group) / n
                if n
                else None
            ),
        }

    return {
        "gap_threshold": GAP_THRESHOLD,
        "total_turns": total,
        "baseline_only_count": baseline_only,
        "baseline_only_percentage": 100.0 * baseline_only / total,
        "alternative_available_count": total - baseline_only,
        "alternative_available_percentage": 100.0 * (total - baseline_only) / total,
        "mean_eligible_arms": (
            sum(len(row["eligible_arms_at_0_10"]) for row in rows) / total
        ),
        "by_md7_base_move": {
            move: group_summary(
                [row for row in rows if row["md7_argmax"] == move]
            )
            for move in MOVE_ORDER
        },
    }


def flatten_turn_for_csv(row: Mapping[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {
        key: value
        for key, value in row.items()
        if key not in {"md6_probabilities", "md7_probabilities", "mrb1_scores"}
    }
    for prefix in ("md6", "md7"):
        probabilities = row[f"{prefix}_probabilities"]
        require(isinstance(probabilities, Mapping), "Probability row is not a mapping.")
        for move in MOVE_ORDER:
            flat[f"{prefix}_p_{move}"] = probabilities[move]
    scores = row["mrb1_scores"]
    require(isinstance(scores, Mapping), "MRB1 row is not a mapping.")
    for task, value in scores.items():
        flat[f"mrb1_{task}"] = value
    flat["eligible_arms_at_0_10"] = json.dumps(
        row["eligible_arms_at_0_10"], separators=(",", ":")
    )
    return flat


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    require(bool(rows), f"Cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: Mapping[str, object]) -> str:
    md6 = summary["md6_distribution"]
    md7 = summary["md7_distribution"]
    matrix = summary["transition_matrix"]
    semantic = summary["semantic_review"]
    collapse = summary["collapse_check"]
    overlay = summary["overlay_summary"]
    safety = summary["side_effects"]
    attempts = summary["attempts"]
    paths = summary["output_artifacts"]
    lines: list[str] = [
        "1. LIVE TEST IMPLEMENTATION",
        "Changed: experiments/controlled_live_md7r1_shadow_md6_v1.py (new, opt-in --run-live runner); no production source file changed.",
        "MD7 activation: ActiveMD7WithMD6Shadow.predict_probabilities returns FrozenMD6Inference(models/candidates/md7r1_epoch3) probabilities to AdaptiveTutorPipeline.run_tutor_turn.",
        "MD6 shadow: the same wrapper calls FrozenMD6Inference(models/frozen/md6) on deep-copied identical problem/history and records it without returning it to C3/LinTS/overlay.",
        "Update suppression: each attempt uses AdaptiveComponentCoordinator.abort_attempt -> AdaptiveTutorPipeline.abort_attempt -> TurnLevelAttemptController.abort_attempt; finish_attempt is never called.",
        "Persistence suppression: diagnostic policy/log paths are temporary and the completion path that writes them is never called; real production paths are hash-checked before/after.",
        "",
        "2. MODEL PATHS",
        f"MD6: {summary['model_paths']['md6']}",
        f"MD7-R1: {summary['model_paths']['md7r1_epoch3']}",
        "",
        "3. ATTEMPTS RUN",
    ]
    for item in attempts:
        lines.append(
            f"{item['attempt_id']}: {item['turn_count']} turns ({item['name']})"
        )
    lines.extend(
        [
            f"Total live tutor turns: {summary['total_live_tutor_turns']}",
            "Tutor Agent: real TutorAgent.teach_turn with its unchanged response verifier.",
            "",
            "4. MD6 RAW MOVE DISTRIBUTION",
        ]
    )
    for move in MOVE_ORDER:
        lines.append(
            f"{move}: {md6['counts'][move]} ({md6['percentages'][move]:.2f}%)"
        )
    lines.extend(["", "5. MD7 RAW MOVE DISTRIBUTION"])
    for move in MOVE_ORDER:
        lines.append(
            f"{move}: {md7['counts'][move]} ({md7['percentages'][move]:.2f}%)"
        )
    lines.extend(["", "6. MD6 -> MD7 TRANSITIONS", "rows=MD6, columns=MD7"])
    lines.append("MD6\\MD7," + ",".join(MOVE_ORDER))
    for source in MOVE_ORDER:
        lines.append(
            source + "," + ",".join(str(matrix[source][target]) for target in MOVE_ORDER)
        )
    lines.extend(["", "7. CRITICAL LIVE CASES"])
    for item in semantic:
        md6_probs = ", ".join(
            f"{move}={item['md6_probabilities'][move]:.6f}" for move in MOVE_ORDER
        )
        md7_probs = ", ".join(
            f"{move}={item['md7_probabilities'][move]:.6f}" for move in MOVE_ORDER
        )
        lines.extend(
            [
                f"{item['case']} | {item['attempt_id']} turn {item['turn_index']}",
                f"  latest student: {item['latest_student_text']!r}",
                f"  MD6: {md6_probs} | {item['md6_argmax']}",
                f"  MD7: {md7_probs} | {item['md7_argmax']}",
                f"  final move: {item['final_move']}",
                f"  {item['judgment']}: {item['rationale']}",
            ]
        )
    lines.extend(["", "8. KNOWN-RISK CHECK"])
    for case in (
        "short_numeric_answer",
        "specific_arithmetic_error",
        "clear_misconception",
        "early_i_dont_know",
        "repeated_failure_after_hint",
        "explicit_direct_help_request",
    ):
        matches = [item for item in semantic if item["case"] == case]
        if not matches:
            lines.append(f"{case}: NOT OBSERVED")
        else:
            item = matches[0]
            lines.append(
                f"{case}: {item['md6_argmax']} -> {item['md7_argmax']}; "
                f"final={item['final_move']}; {item['judgment']}"
            )
    lines.extend(
        [
            "",
            "9. NEW CLASS COLLAPSE CHECK",
            "MD7 distribution: "
            + ", ".join(f"{move}={md7['counts'][move]}" for move in MOVE_ORDER),
            f"Maximum single-class share: {collapse['maximum_single_class_share_percent']:.2f}%",
            f"NEW_CLASS_COLLAPSE: {str(collapse['new_class_collapse']).upper()}",
        ]
    )
    for move in MOVE_ORDER:
        lines.append(
            f"{move.upper()}_COLLAPSE_RISK: {collapse['class_risks'][move]}"
        )
    lines.extend(
        [
            "",
            "10. OVERLAY OPPORTUNITY",
            f"Baseline-only: {overlay['baseline_only_percentage']:.2f}%",
            f"Alternative-available: {overlay['alternative_available_percentage']:.2f}%",
            f"Mean eligible arms: {overlay['mean_eligible_arms']:.6f}",
            "By MD7 base move:",
        ]
    )
    for move in MOVE_ORDER:
        item = overlay["by_md7_base_move"][move]
        lines.append(
            f"  {move}: n={item['n']}, baseline-only={item['baseline_only_percentage']}, "
            f"alternative-available={item['alternative_available_percentage']}, "
            f"mean eligible arms={item['mean_eligible_arms']}"
        )
    lines.extend(["", "11. TUTOR TRAJECTORY QUALITY"])
    for item in summary["trajectory_review"]:
        lines.append(
            f"{item['attempt_id']}: raw MD7 {' -> '.join(item['md7_raw_moves'])}; "
            f"final {' -> '.join(item['final_moves'])}. {item['interpretation']}"
        )
    lines.extend(["", "12. SIDE EFFECTS"])
    for name, value in safety.items():
        lines.append(f"{name}: {value}")
    lines.extend(["", "13. OUTPUT ARTIFACTS"])
    lines.extend(str(path) for path in paths)
    lines.extend(
        [
            "",
            "14. FINAL VERDICT",
            str(summary["final_verdict"]),
            "",
            "Interpretation boundary: these are qualitative live diagnostics, not gold labels or an accuracy estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def trajectory_review(
    rows: Sequence[Mapping[str, object]],
    attempts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for attempt in attempts:
        attempt_rows = [row for row in rows if row["attempt_id"] == attempt["attempt_id"]]
        raw = [str(row["md7_argmax"]) for row in attempt_rows]
        final = [str(row["final_move"]) for row in attempt_rows]
        later_support = any(move in {"focus", "telling"} for move in final[1:])
        interpretation = (
            "The sequence introduced targeted support after the opening; it need not contain all four moves."
            if later_support
            else "The sequence did not progress to focus/telling despite later difficulty; review warranted."
        )
        result.append(
            {
                "attempt_id": attempt["attempt_id"],
                "md7_raw_moves": raw,
                "final_moves": final,
                "interpretation": interpretation,
            }
        )
    return result


def choose_verdict(
    *,
    collapse: Mapping[str, object],
    semantic: Sequence[Mapping[str, object]],
    side_effects: Mapping[str, object],
) -> str:
    zero_required = (
        "MD6 model writes",
        "MD7 model writes",
        "training",
        "LinTS posterior updates",
        "real policy-state writes",
        "real adaptive experience-log writes",
        "overlay threshold changes",
        "BKT equation changes",
        "MRB1 changes",
        "Tutor Agent semantic changes",
        "external diagnostic API calls",
    )
    if any(side_effects[name] != 0 for name in zero_required):
        return "D. CONTROLLED TEST INVALID DUE TO IMPLEMENTATION OR SIDE EFFECTS"
    if collapse["new_class_collapse"]:
        return "C. MD7-R1 SHOWS NEW CLASS COLLAPSE OR UNSTABLE LIVE BEHAVIOR - REJECT"
    concerning = sum(
        item["judgment"] in {"QUESTIONABLE", "POSSIBLE NEW BIAS"}
        for item in semantic
    )
    if concerning:
        return "B. MD7-R1 IMPROVES DIVERSITY BUT HAS LIVE SEMANTIC ISSUES - DO NOT PROMOTE YET"
    return "A. MD7-R1 PASSES CONTROLLED LIVE TEST - READY FOR ACTIVE SELECTOR PROMOTION WITH MD6 ROLLBACK"


def validate_contract() -> None:
    require(DEFAULT_GAP_THRESHOLD == GAP_THRESHOLD, "Overlay threshold is not 0.10.")
    require(MOVE_ORDER == ("generic", "probing", "focus", "telling"), "Move order changed.")
    require(MD6_DIR.is_dir(), f"MD6 directory is missing: {MD6_DIR}")
    require(MD7_DIR.is_dir(), f"MD7 directory is missing: {MD7_DIR}")
    require(not OUTPUT_DIR.exists(), f"Refusing to overwrite output directory: {OUTPUT_DIR}")
    for model_dir in (MD6_DIR, MD7_DIR):
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        id2label = tuple(config["id2label"][str(index)] for index in range(4))
        require(id2label == MOVE_ORDER, f"Class order mismatch in {model_dir}")


def run_live() -> dict[str, object]:
    validate_contract()
    production_policy_path, production_experience_path = production_paths()
    production_before = selected_files_snapshot(
        (production_policy_path, production_experience_path)
    )
    md6_before = tree_snapshot(MD6_DIR)
    md7_before = tree_snapshot(MD7_DIR)
    protected_sources = (
        PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py",
        STUDENT_MODEL_ROOT / "bkt" / "predict.py",
        STUDENT_MODEL_ROOT / "bkt" / "train.py",
        PROJECT_ROOT / "src" / "self_improvement" / "mrb1_inference.py",
        BACKEND_ROOT / "app" / "agents" / "tutor" / "tutor_agent.py",
        BACKEND_ROOT / "app" / "agents" / "tutor" / "response_verifier.py",
        BACKEND_ROOT / "app" / "integrations" / "adaptive_tutor_agent_adapter.py",
    )
    protected_before = selected_files_snapshot(protected_sources)

    print("Loading active MD7-R1, shadow MD6, MRB1, BKT, and detectors...")
    active_md7 = FrozenMD6Inference(MD7_DIR)
    shadow_md6 = FrozenMD6Inference(MD6_DIR)
    selector = ActiveMD7WithMD6Shadow(
        active_md7=active_md7,
        shadow_md6=shadow_md6,
    )
    mrb1 = FrozenMRB1Inference()
    predictor = BKTPredictor.load(STUDENT_MODEL_ROOT / "models" / "bkt_params.json")
    detectors = DetectorService.from_project_defaults(STUDENT_MODEL_ROOT)
    policy_mode = os.getenv("ADAPTIVE_DATA_MODE", "real").strip()
    require(policy_mode in {"real", "synthetic"}, "Invalid ADAPTIVE_DATA_MODE.")
    policy = TrueDisjointLinTS(
        context_dim=len(TURN_FEATURE_NAMES),
        seed=42,
        data_mode=policy_mode,
    )
    if production_policy_path.exists():
        load_policy_state(
            policy,
            production_policy_path,
            expected_data_mode=policy_mode,
        )
    posterior_before = posterior_snapshot(policy)

    tutor = TutorAgent()
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    run_token = uuid4().hex[:12]

    with tempfile.TemporaryDirectory(prefix="adaptmath_md7_live_") as temp_name:
        temp_root = Path(temp_name)
        student_database = temp_root / "student-model" / "meta_agent.db"
        diagnostic_policy_path = temp_root / "policy" / "diagnostic_policy.json"
        diagnostic_experience_path = temp_root / "policy" / "diagnostic_experience.jsonl"
        logger = ExperienceLogger(
            diagnostic_experience_path,
            data_mode=policy_mode,
            source_policy=policy,
        )

        with (
            patch.object(
                knowledge_graph_module,
                "initialise_database",
                lambda: initialise_database(student_database),
            ),
            patch.object(
                knowledge_graph_module,
                "get_connection",
                lambda: get_connection(student_database),
            ),
        ):
            knowledge_graph = KnowledgeGraph(predictor=predictor)

            def component_factory(**identity: object) -> AdaptiveAttemptComponents:
                skill = str(identity["target_skill"])
                evidence_source = StartOnlyEvidenceSource(skill)
                cross_session = CrossSessionStudentModelPipeline(
                    concept_extractor=evidence_source,
                    evaluator=evidence_source,
                    detectors=detectors,
                    knowledge_graph=knowledge_graph,
                    curriculum=None,
                )
                student_pipeline = TutorAssessmentStudentModelPipeline(
                    target_skill=skill,
                    cross_session_pipeline=cross_session,
                    evidence_source=evidence_source,
                )
                memory_adapter = TutorStateMemoryAdapter()
                adaptive_pipeline = AdaptiveTutorPipeline(
                    md6=selector,
                    mrb1=mrb1,
                    controller=TurnLevelAttemptController(
                        policy,
                        gap_threshold=GAP_THRESHOLD,
                    ),
                    experience_logger=logger,
                    policy_state_path=diagnostic_policy_path,
                    memory_adapter=memory_adapter,
                )
                bridge = StudentModelFrozenV3Bridge(
                    student_model_pipeline=student_pipeline,
                    adaptive_pipeline=adaptive_pipeline,
                )
                return AdaptiveAttemptComponents(
                    bridge=bridge,
                    student_model_pipeline=student_pipeline,
                    memory_adapter=memory_adapter,
                )

            coordinator = AdaptiveComponentCoordinator(
                component_factory=component_factory,
                skill_validator=predictor.has_skill,
            )

            for attempt_number, spec in enumerate(ATTEMPT_SPECS, start=1):
                thread_id = f"md7-live-{run_token}-{attempt_number}"
                student_id = f"controlled-live-{run_token}"
                attempt_id = coordinator.derive_attempt_id(thread_id, 1)
                require(
                    predictor.has_skill(str(spec["skill"])),
                    f"Untrained BKT skill: {spec['skill']}",
                )
                print(f"Preparing verified Tutor math evidence for {attempt_id}...")
                evidence = tutor.prepare_math_evidence(
                    TutorMathInput(
                        question=str(spec["problem"]),
                        topic=str(spec["topic"]),
                        subtopic=str(spec["subtopic"]),
                        student_age=14,
                        complexity_score=float(spec["complexity_score"]),
                        planner_output=None,
                        previous_errors=[],
                        reteaching=False,
                    )
                )
                state = make_state(
                    spec=spec,
                    student_id=student_id,
                    thread_id=thread_id,
                    attempt_id=attempt_id,
                    verified_math_evidence=evidence,
                )
                state.update(coordinator.start_attempt(state))
                spec_turns = spec["turns"]
                require(isinstance(spec_turns, Sequence), "Attempt turns are invalid.")
                attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "name": spec["name"],
                        "target_skill": spec["skill"],
                        "turn_count": len(spec_turns),
                    }
                )

                for turn_spec in spec_turns:
                    require(isinstance(turn_spec, Mapping), "Turn spec is invalid.")
                    student_text = turn_spec["student_text"]
                    if student_text is not None:
                        append_student_turn(state, str(student_text))
                    selector_count_before = len(selector.records)
                    print(
                        f"Running {attempt_id} / {turn_spec['case']} "
                        f"(turn {len([r for r in rows if r['attempt_id'] == attempt_id]) + 1})..."
                    )
                    result = coordinator.run_tutor_turn(state, tutor_agent=tutor)
                    require(
                        len(selector.records) == selector_count_before + 1,
                        "Dual selector did not run exactly once for a live turn.",
                    )
                    shadow_record = selector.records[-1]
                    md6_probs = dict(shadow_record["md6_probabilities"])
                    md7_probs = dict(shadow_record["md7_probabilities"])
                    require(
                        all(
                            abs(float(result["md6_probabilities"][move]) - md7_probs[move])
                            <= 1e-9
                            for move in MOVE_ORDER
                        ),
                        "The probabilities entering C3 are not the active MD7 probabilities.",
                    )
                    md6_diag = probability_diagnostics(md6_probs)
                    md7_diag = probability_diagnostics(md7_probs)
                    active_context = coordinator._contexts[thread_id]
                    completed = active_context.bridge.adaptive_pipeline.controller.completed_turns
                    decision = completed[-1].decision
                    require(
                        decision.base_move == md7_diag["argmax"],
                        "Overlay base move is not the MD7 raw argmax.",
                    )
                    history = shadow_record["history"]
                    require(isinstance(history, Sequence), "Shadow history is invalid.")
                    plausible_moves = tuple(turn_spec["plausible_moves"])
                    judgment, rationale = semantic_judgment(
                        case=str(turn_spec["case"]),
                        plausible_moves=plausible_moves,
                        md6_move=str(md6_diag["argmax"]),
                        md7_move=str(md7_diag["argmax"]),
                    )
                    row: dict[str, object] = {
                        "attempt_id": attempt_id,
                        "attempt_name": spec["name"],
                        "turn_index": int(result["turn_index"]),
                        "case": turn_spec["case"],
                        "problem": spec["problem"],
                        "latest_student_text": latest_student_text(history),
                        "history_turn_count": len(history),
                        "md6_probabilities": md6_probs,
                        "md6_argmax": md6_diag["argmax"],
                        "md6_top1_probability": md6_diag["top1_probability"],
                        "md6_top2_probability": md6_diag["top2_probability"],
                        "md6_top1_top2_gap": md6_diag["top1_top2_gap"],
                        "md7_probabilities": md7_probs,
                        "md7_argmax": md7_diag["argmax"],
                        "md7_top1_probability": md7_diag["top1_probability"],
                        "md7_top2_probability": md7_diag["top2_probability"],
                        "md7_top1_top2_gap": md7_diag["top1_top2_gap"],
                        "md6_to_md7_transition": (
                            f"{md6_diag['argmax']} -> {md7_diag['argmax']}"
                        ),
                        "eligible_arms_at_0_10": list(result["eligible_arms"]),
                        "selected_arm": result["selected_arm"],
                        "final_move": result["pedagogical_move"],
                        "overridden": bool(result["overridden"]),
                        "overlay_gap": float(decision.gap),
                        "tutor_response": result["tutor_response"],
                        "mrb1_scores": dict(result["mrb1_scores"]),
                        "semantic_judgment": judgment,
                        "semantic_rationale": rationale,
                    }
                    rows.append(row)
                    state.update(result)
                    require(
                        posterior_snapshot(policy) == posterior_before,
                        "LinTS posterior changed inside an active attempt.",
                    )

                aborted = coordinator.abort_attempt(state)
                require(
                    aborted == {"adaptive_lifecycle_status": "aborted"},
                    f"Attempt did not exit through the safe abort path: {attempt_id}",
                )

        temporary_write_checks = {
            "diagnostic_policy_state_exists": diagnostic_policy_path.exists(),
            "diagnostic_experience_log_exists": diagnostic_experience_path.exists(),
        }
        require(
            not any(temporary_write_checks.values()),
            "Completion persistence ran despite the abort-only test contract.",
        )

    require(rows, "No live tutor turns were recorded.")
    require(len(attempts) == 3, "Controlled test did not run exactly three attempts.")
    require(
        posterior_snapshot(policy) == posterior_before,
        "LinTS posterior differs after all aborted attempts.",
    )

    md6_after = tree_snapshot(MD6_DIR)
    md7_after = tree_snapshot(MD7_DIR)
    production_after = selected_files_snapshot(
        (production_policy_path, production_experience_path)
    )
    protected_after = selected_files_snapshot(protected_sources)
    md6_writes = changed_entry_count(md6_before, md6_after)
    md7_writes = changed_entry_count(md7_before, md7_after)
    production_changes = {
        path: int(production_before[path] != production_after[path])
        for path in production_before
    }
    protected_changes = {
        path: int(protected_before[path] != protected_after[path])
        for path in protected_before
    }
    policy_key = str(production_policy_path)
    experience_key = str(production_experience_path)
    side_effects: dict[str, object] = {
        "MD6 model writes": md6_writes,
        "MD7 model writes": md7_writes,
        "training": 0,
        "LinTS posterior updates": (
            0 if posterior_snapshot(policy) == posterior_before else 1
        ),
        "real policy-state writes": production_changes[policy_key],
        "real adaptive experience-log writes": production_changes[experience_key],
        "overlay threshold changes": protected_changes[
            str((PROJECT_ROOT / "src" / "self_improvement" / "conservative_overlay.py").resolve())
        ],
        "BKT equation changes": sum(
            protected_changes[str(path.resolve())]
            for path in protected_sources[1:3]
        ),
        "MRB1 changes": protected_changes[str(protected_sources[3].resolve())],
        "Tutor Agent semantic changes": sum(
            protected_changes[str(path.resolve())]
            for path in protected_sources[4:]
        ),
        "external diagnostic API calls": 0,
        "normal Tutor Agent API use": (
            "YES - only normal math-evidence, turn-generation, and response-audit calls"
        ),
        "temporary policy/log writes": 0,
        "BKT storage": "temporary isolated SQLite; removed after run",
        "Memory writes": 0,
    }

    md6_distribution = distribution(rows, "md6_argmax")
    md7_distribution = distribution(rows, "md7_argmax")
    matrix = transition_matrix(rows)
    max_share = max(md7_distribution["percentages"].values())
    collapse = {
        "threshold_percent": 80.0,
        "maximum_single_class_share_percent": max_share,
        "new_class_collapse": max_share >= 80.0,
        "class_risks": {
            move: collapse_risk(md7_distribution["percentages"][move])
            for move in MOVE_ORDER
        },
        "risk_rule": "LOW < 50%; MEDIUM >= 50% and < 80%; HIGH >= 80%.",
    }
    overlay = summarize_overlay(rows)
    semantic = [
        {
            "attempt_id": row["attempt_id"],
            "turn_index": row["turn_index"],
            "case": row["case"],
            "latest_student_text": row["latest_student_text"],
            "md6_probabilities": row["md6_probabilities"],
            "md6_argmax": row["md6_argmax"],
            "md7_probabilities": row["md7_probabilities"],
            "md7_argmax": row["md7_argmax"],
            "final_move": row["final_move"],
            "judgment": row["semantic_judgment"],
            "rationale": row["semantic_rationale"],
            "interpretation_boundary": "Qualitative diagnostic only; no gold label.",
        }
        for row in rows
    ]
    trajectories = trajectory_review(rows, attempts)

    artifact_names = (
        "live_turns.jsonl",
        "live_turns.csv",
        "live_summary.json",
        "md6_vs_md7_transitions.csv",
        "semantic_review.json",
        "overlay_summary.json",
        "live_report.txt",
    )
    artifact_paths = [str((OUTPUT_DIR / name).resolve()) for name in artifact_names]
    summary: dict[str, object] = {
        "diagnostic_name": "controlled_live_md7r1_shadow_md6_v1",
        "diagnostic_only": True,
        "model_paths": {
            "md6": str(MD6_DIR.resolve()),
            "md7r1_epoch3": str(MD7_DIR.resolve()),
        },
        "model_hashes": {
            "md6": {name: item["sha256"] for name, item in md6_after.items()},
            "md7r1_epoch3": {
                name: item["sha256"] for name, item in md7_after.items()
            },
        },
        "attempts": attempts,
        "total_live_tutor_turns": len(rows),
        "md6_distribution": md6_distribution,
        "md7_distribution": md7_distribution,
        "transition_matrix": matrix,
        "semantic_review": semantic,
        "collapse_check": collapse,
        "overlay_summary": overlay,
        "trajectory_review": trajectories,
        "overconfidence_check": {
            "threshold": 0.90,
            "md6_top1_at_or_above_threshold": sum(
                float(row["md6_top1_probability"]) >= 0.90 for row in rows
            ),
            "md7_top1_at_or_above_threshold": sum(
                float(row["md7_top1_probability"]) >= 0.90 for row in rows
            ),
            "md7_max_top1_probability": max(
                float(row["md7_top1_probability"]) for row in rows
            ),
        },
        "side_effects": side_effects,
        "production_paths_checked": {
            "policy_state": str(production_policy_path),
            "experience_log": str(production_experience_path),
        },
        "output_artifacts": artifact_paths,
        "implementation_discovery": {
            "MD6 load": "adaptive-math-tutor/backend/app/integrations/adaptive_component_coordinator.py::_ProductionAdaptiveResources._ensure_shared",
            "base probabilities": "pedagogical-move-selection/src/self_improvement/adaptive_tutor_pipeline.py::AdaptiveTutorPipeline.run_tutor_turn",
            "C3/LinTS/overlay": "pedagogical-move-selection/src/self_improvement/turn_level_controller.py::TurnLevelAttemptController.select_turn",
            "Tutor handoff": "adaptive-math-tutor/backend/app/integrations/adaptive_tutor_agent_adapter.py::AdaptiveTutorAgentAdapter.generate",
            "LinTS update": "pedagogical-move-selection/src/self_improvement/turn_level_controller.py::TurnLevelAttemptController.finish_attempt",
            "experience persistence": "pedagogical-move-selection/src/self_improvement/adaptive_tutor_pipeline.py::AdaptiveTutorPipeline.finish_attempt -> ExperienceLogger.append_attempt",
            "policy persistence": "pedagogical-move-selection/src/self_improvement/adaptive_tutor_pipeline.py::AdaptiveTutorPipeline.finish_attempt -> save_policy_state",
            "history representation": "TutorState.conversation_history as DialogueTurn dictionaries, projected by TutorStateMemoryAdapter.get_conversation_history",
        },
        "interpretation_boundary": (
            "Controlled qualitative behavior diagnostic; scenario-plausible sets are not gold labels."
        ),
    }
    summary["final_verdict"] = choose_verdict(
        collapse=collapse,
        semantic=semantic,
        side_effects=side_effects,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    jsonl_path = OUTPUT_DIR / "live_turns.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    write_csv(OUTPUT_DIR / "live_turns.csv", [flatten_turn_for_csv(row) for row in rows])
    (OUTPUT_DIR / "live_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    transition_rows = [
        {
            "md6_move": source,
            "md7_move": target,
            "count": matrix[source][target],
        }
        for source in MOVE_ORDER
        for target in MOVE_ORDER
    ]
    write_csv(OUTPUT_DIR / "md6_vs_md7_transitions.csv", transition_rows)
    (OUTPUT_DIR / "semantic_review.json").write_text(
        json.dumps(semantic, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "overlay_summary.json").write_text(
        json.dumps(overlay, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "live_report.txt").write_text(
        build_report(summary),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Explicitly authorize the controlled real-Tutor diagnostic run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_live:
        validate_contract()
        print("Validation passed. Re-run with --run-live to execute the live diagnostic.")
        return
    summary = run_live()
    print(
        json.dumps(
            {
                "attempts": summary["attempts"],
                "total_live_tutor_turns": summary["total_live_tutor_turns"],
                "md6_distribution": summary["md6_distribution"],
                "md7_distribution": summary["md7_distribution"],
                "collapse_check": summary["collapse_check"],
                "overlay_summary": summary["overlay_summary"],
                "side_effects": summary["side_effects"],
                "final_verdict": summary["final_verdict"],
                "output_artifacts": summary["output_artifacts"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
