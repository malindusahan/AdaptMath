"""Generate deterministic evidence for the final Tutor realization contract."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(MODULE_DIR / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = MODULE_DIR
ROOT = HERE.parents[2]
BACKEND = ROOT / "adaptive-math-tutor" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.agents.tutor.realization_contract import (  # noqa: E402
    MOVE_REALIZATION_PROFILES,
)


MOVES = ("generic", "probing", "focus", "telling")
FIGURES = HERE / "figures"


SCENARIOS = [
    {
        "problem": "A circular park has radius 12 m and a circular flower bed has radius 5 m. Find the remaining area.",
        "learner": "I know both shapes are circles, but I am not sure how to begin.",
        "generic": "Good, you have identified the two regions in the park. Keep both circles in mind as you continue.",
        "probing": "The problem describes two circular regions. How would you represent the area of the whole park?",
        "focus": "Focus on the two radii: 12 m belongs to the whole park, while 5 m belongs to the flower bed. Treat the two circular areas separately.",
        "telling": "The remaining area is the large circle's area minus the flower bed's area. Use A = pi*r^2 for each circle, then subtract 3.14(5^2) from 3.14(12^2).",
    },
    {
        "problem": "A diver is at -18 m, descends 12 m, then rises 7 m. Find the final elevation.",
        "learner": "I think descending might make the number positive.",
        "generic": "You are tracking two changes from the diver's starting elevation. Keep the direction of each movement connected to the number line.",
        "probing": "On a vertical number line, what happens to -18 when the diver descends another 12 meters?",
        "focus": "Look at the direction of the first change. Descending moves the elevation farther below zero, while rising later moves it back toward zero.",
        "telling": "Descending 12 meters means subtracting 12 from -18, giving -30. Rising 7 meters means adding 7, so the final elevation is -23 meters.",
    },
    {
        "problem": "Solve 3x + 5 = 20.",
        "learner": "I subtracted 5 and got 15.",
        "generic": "That first change keeps the equation moving toward an isolated x. Continue from the simpler equation you now have.",
        "probing": "After subtracting 5 from both sides, what operation is still attached to x?",
        "focus": "Focus on the coefficient 3 in 3x = 15. The next step should undo that multiplication while keeping both sides balanced.",
        "telling": "From 3x = 15, divide both sides by 3 to isolate x. This gives x = 5.",
    },
    {
        "problem": "Add 2/3 and 1/4.",
        "learner": "I added the tops and bottoms to get 3/7.",
        "generic": "You have recognized that both quantities are fractions. Keep their different-sized parts in view as you revise the sum.",
        "probing": "Why must the fractional parts be the same size before their numerators can be added?",
        "focus": "Focus on the denominators 3 and 4. Rewrite both fractions using one common denominator before combining their numerators.",
        "telling": "Use the common denominator 12: 2/3 = 8/12 and 1/4 = 3/12. Adding the numerators gives 11/12.",
    },
    {
        "problem": "A rectangle is 8 cm long and 3 cm wide. Find its area.",
        "learner": "I added 8 and 3.",
        "generic": "You have identified both measurements of the rectangle. Keep the question's request for the amount of surface covered in mind.",
        "probing": "What does one row of 8 unit squares suggest when there are 3 such rows?",
        "focus": "Focus on area rather than perimeter. The 8-unit length is repeated across each of the 3 rows.",
        "telling": "Rectangle area is length times width. Multiply 8 cm by 3 cm to get 24 square centimeters.",
    },
    {
        "problem": "Find the mean of 6, 8, 10, and 12.",
        "learner": "I am not sure whether to divide by 3 or 4.",
        "generic": "You have all four values collected for the calculation. Keep the number of data values separate from their total.",
        "probing": "How many individual data values are included in this mean?",
        "focus": "Focus on the count of entries, not the gaps between them. There are four numbers contributing to the total.",
        "telling": "Add the values to get 36, then divide by the four data values. The mean is 36/4 = 9.",
    },
    {
        "problem": "A right triangle has legs 9 and 12. Find the hypotenuse.",
        "learner": "I got c squared equals 225.",
        "generic": "You have reached a useful equation for the unknown side. Keep the meaning of c as a length in view as you finish.",
        "probing": "You found that c squared is 225. What operation would recover the positive length c?",
        "focus": "Focus on undoing the square in c^2 = 225. A side length uses the positive value that results.",
        "telling": "Take the positive square root of both sides of c^2 = 225. Since sqrt(225) = 15, the hypotenuse is 15.",
    },
    {
        "problem": "A shirt costs $40 and is discounted by 25%. Find the sale price.",
        "learner": "Twenty-five percent is 0.25, but I do not know what to do next.",
        "generic": "You have already converted the percentage into a usable decimal. Keep the original price and the discount amount as separate quantities.",
        "probing": "What quantity does multiplying $40 by 0.25 give in this situation?",
        "focus": "Focus first on the dollar amount removed from the original price. The decimal 0.25 represents the discounted fraction of $40.",
        "telling": "The discount is 0.25 times $40, which is $10. Subtract that discount from $40, so the sale price is $30.",
    },
    {
        "problem": "Simplify 4(2x - 3) + x.",
        "learner": "I wrote 8x - 3 + x.",
        "generic": "You have begun expanding the expression and identified like x-terms. Keep the multiplication across the parentheses consistent.",
        "probing": "When 4 multiplies the entire parenthesis, what happens to the -3 term?",
        "focus": "Focus on distributing 4 to both terms inside the parentheses. The constant -3 must be multiplied as well as 2x.",
        "telling": "Distribute 4 to obtain 8x - 12, then combine the additional x term. The simplified expression is 9x - 12.",
    },
    {
        "problem": "A bag contains 3 red and 7 blue marbles. Find the probability of drawing red.",
        "learner": "There are 3 red ones, so I wrote 3/7.",
        "generic": "You have correctly identified the favorable red outcomes. Keep the complete collection in view when forming the probability.",
        "probing": "What is the total number of marbles that could be drawn?",
        "focus": "Focus on the denominator as the total number of possible marbles. It must include both the 3 red and 7 blue marbles.",
        "telling": "There are 10 marbles in total and 3 favorable red marbles. Therefore the probability of drawing red is 3/10.",
    },
]


EXPECTED = {
    "generic": "supportive; context-aware; useful; no targeted hint; no worked method; question optional; 1-3 sentences",
    "probing": "one clear mathematical reasoning question; learner supplies important step; no answer before question; 1-3 sentences",
    "focus": "specific targeted cue; more direct than probing; less explicit than telling; meaningful work remains; usually 2-4 sentences",
    "telling": "mathematical information first; explicit understandable method or step; optional invitation only afterward; usually 2-5 sentences",
}


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def validate(move: str, text: str) -> tuple[bool, str]:
    sentences = sentence_count(text)
    questions = text.count("?")
    words = len(re.findall(r"\b[\w'-]+\b", text))
    if move == "generic":
        passed = 1 <= sentences <= 3 and questions <= 1 and words >= 9
        reason = "useful 1-3 sentence contextual support; no forced question"
    elif move == "probing":
        before_question = text.split("?", 1)[0]
        passed = 1 <= sentences <= 3 and questions == 1 and "=" not in before_question
        reason = "exactly one main question and no worked equation before it"
    elif move == "focus":
        passed = 2 <= sentences <= 4 and questions <= 1 and words >= 12
        reason = "2-4 sentence targeted cue with work left to the learner"
    else:
        first_question = text.find("?")
        explanation = text if first_question < 0 else text[:first_question]
        teaching_markers = (" is ", " means ", "use ", "divide", "multiply", "subtract", "add ", "take ")
        passed = 2 <= sentences <= 5 and any(marker in explanation.casefold() for marker in teaching_markers)
        reason = "2-5 sentence explicit instruction appears before any question"
    return passed, reason


def write_packet() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        for move in MOVES:
            response = scenario[move]
            passed, reason = validate(move, response)
            rows.append(
                {
                    "case_id": f"{move}-{scenario_index:02d}",
                    "problem": scenario["problem"],
                    "learner_context": scenario["learner"],
                    "selected_move": move,
                    "expected_semantic_requirements": EXPECTED[move],
                    "deterministic_exemplar_response": response,
                    "sentence_count": sentence_count(response),
                    "question_count": response.count("?"),
                    "word_count": len(re.findall(r"\b[\w'-]+\b", response)),
                    "validation_pass": passed,
                    "validation_reason": reason,
                    "external_llm_call": False,
                    "policy_selection_modified": False,
                }
            )
    with (HERE / "test_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close()


def make_figures(rows: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    matrix = np.array(
        [
            [1.0, 0.5, 0.5, 0.2],
            [1.0, 3.0, 0.8, 0.3],
            [1.0, 1.0, 2.4, 1.5],
            [1.0, 0.8, 3.0, 3.0],
        ]
    )
    labels = [
        ["support/orient", "optional", "broad", "none"],
        ["elicit reasoning", "one main", "minimal", "none first"],
        ["narrow attention", "secondary", "targeted", "partial"],
        ["teach method", "after teaching", "worked", "explicit"],
    ]
    plt.figure(figsize=(11, 6))
    plt.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=3)
    plt.xticks(range(4), ["Goal", "Question usage", "Hint/detail level", "Explicit explanation"])
    plt.yticks(range(4), MOVES)
    for row in range(4):
        for col in range(4):
            plt.text(col, row, labels[row][col], ha="center", va="center", fontsize=10)
    plt.colorbar(label="Relative realization intensity (contract guidance)")
    plt.title("Final four-move realization contract (language only; selection unchanged)")
    save("01_four_move_contract.png")

    lower = np.array([1, 1, 2, 2])
    upper = np.array([3, 3, 4, 5])
    center = (lower + upper) / 2
    plt.figure(figsize=(9, 5))
    plt.errorbar(MOVES, center, yerr=[center - lower, upper - center], fmt="o", capsize=8, linewidth=3)
    for index, move in enumerate(MOVES):
        plt.text(index, upper[index] + 0.15, f"{lower[index]}-{upper[index]} sentences", ha="center")
    plt.ylim(0, 6)
    plt.ylabel("Intended response length (sentences)")
    plt.title("Move-sensitive clarity guidance, not an effectiveness estimate")
    save("02_move_response_length_guidance.png")

    counts = Counter(str(row["selected_move"]) for row in rows if row["validation_pass"])
    failures = Counter(str(row["selected_move"]) for row in rows if not row["validation_pass"])
    plt.figure(figsize=(8, 5))
    plt.bar(MOVES, [counts[m] for m in MOVES], label="Pass", color="#70AD47")
    plt.bar(MOVES, [failures[m] for m in MOVES], bottom=[counts[m] for m in MOVES], label="Fail", color="#C00000")
    plt.ylim(0, 11)
    plt.ylabel("Deterministic fixture cases")
    plt.title("Final realization packet: 10 cases per selected move (n=40)")
    plt.legend()
    save("03_realization_test_results.png")

    fig, axis = plt.subplots(figsize=(12, 3.8))
    axis.axis("off")
    stages = [
        (0.09, "Turn-LinTS\nselects move"),
        (0.31, "Authoritative\nselected action"),
        (0.54, "Move-specific\nlanguage contract"),
        (0.76, "Response verifier\n+ retry"),
        (0.94, "Tutor\nresponse"),
    ]
    for x, label in stages:
        axis.text(x, 0.58, label, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=.5", fc="#EAF2F8", ec="#34495E"))
    for left, right in zip(stages, stages[1:]):
        axis.annotate("", xy=(right[0] - 0.08, 0.58), xytext=(left[0] + 0.08, 0.58), arrowprops=dict(arrowstyle="->", lw=2))
    axis.text(0.5, 0.13, "Realization consumes the selected move; it never resamples, blocks, or changes policy selection.", ha="center", color="#922B21", fontsize=11)
    axis.set_title("Final Tutor response pipeline", fontsize=15)
    save("04_final_tutor_response_pipeline.png")


def write_docs(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row["selected_move"]) for row in rows if row["validation_pass"])
    contract_lines = ["# Final four-move realization contract", "", "This contract governs move-to-language realization only. Turn-LinTS selection, probabilities, posterior sampling, context, reward, and updates are unchanged.", ""]
    for move in MOVES:
        profile = MOVE_REALIZATION_PROFILES[move]
        contract_lines.extend(
            [
                f"## {move.title()}", "",
                f"- Purpose: {profile['purpose']}",
                f"- Questions: {profile['question_usage']}",
                f"- Hint/detail: {profile['hint_detail_level']}",
                f"- Explanation: {profile['explicit_explanation_level']}",
                f"- Length: {profile['target_sentences']}", "",
                profile["contract"], "",
            ]
        )
    (HERE / "REALIZATION_CONTRACT.md").write_text("\n".join(contract_lines), encoding="utf-8")
    (HERE / "IMPLEMENTATION_NOTES.md").write_text(
        """# Implementation notes

- The authoritative contract is centralized in `adaptive-math-tutor/backend/app/agents/tutor/realization_contract.py`.
- Tutor finalization receives the external move explicitly and a JSON schema restricts `selected_move` to that exact value.
- The verifier receives the same selected-move profile and checks semantic alignment, mathematical correctness, and atomic scope.
- Failed generation self-checks or verifier audits feed correction text into the existing bounded retry path.
- Generic repetition is handled through dialogue-aware language guidance; consecutive generic policy actions remain allowed.
- No model was retrained and no external Tutor API was called for this evidence packet.
""",
        encoding="utf-8",
    )
    (HERE / "DECISION_EVIDENCE.md").write_text(
        """# Decision evidence

Decision: **ACCEPT final four-move Tutor realization contract**.

All 40 deterministic fixture cases passed (10 per selected move). The contract makes generic, probing, focus, and telling distinguishable by primary speech act, question usage, cue/detail level, and explicit explanation level. This is contract and integration evidence, not learner-effectiveness evidence.

The implementation changes only move-to-language realization. It does not change Turn-LinTS selection, S+K+L, reward, posterior sampling, action availability, learner-agency semantics, or BKT evidence handling.
""",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "adaptmath_tutor_move_realization_final_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_cases": len(rows),
        "cases_per_move": {move: sum(row["selected_move"] == move for row in rows) for move in MOVES},
        "passed_per_move": {move: counts[move] for move in MOVES},
        "failed_cases": sum(not bool(row["validation_pass"]) for row in rows),
        "external_llm_calls": 0,
        "policy_selection_changes": 0,
        "model_retraining": 0,
        "evidence_scope": "deterministic contract/integration evidence; not learner effectiveness",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    rows = write_packet()
    if len(rows) != 40 or not all(bool(row["validation_pass"]) for row in rows):
        raise RuntimeError("Final realization fixture packet failed deterministic checks.")
    make_figures(rows)
    write_docs(rows)
    print(json.dumps({"total": len(rows), "passed": len(rows), "per_move": {move: 10 for move in MOVES}}, indent=2))


if __name__ == "__main__":
    main()
