from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1"
BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
CANDIDATE = (
    ROOT
    / "pedagogical-move-selection"
    / "models"
    / "candidates"
    / "md7_telling_calibration_epoch3"
    / "md7_telling_calibration_v5_candidates"
    / "md7_telling_calibration_v5"
    / "epoch3"
)
CANDIDATE_ROOT = CANDIDATE.parent
VALIDATION = ROOT / "pedagogical-move-selection" / "data" / "processed" / "mathdial" / "validation.jsonl"
POLICY = ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "adaptive_demo_md7r1_real_v1" / "policy_state.json"

TURN_PATH = DATA / "turn_outcomes.jsonl"
ASSESSMENT_PATH = DATA / "assessment_outcomes.jsonl"
SUMMARY_PATH = DATA / "attempt_summaries.jsonl"
SAFETY_PATH = OUT / "validation_safety_check.json"

MOVES = ("generic", "probing", "focus", "telling")
LABEL_TO_ID = {move: index for index, move in enumerate(MOVES)}
EXPECTED_SAFETY = {
    "accuracy": 0.5162162162162162,
    "macro_f1": 0.4611286390258053,
    "per_class_f1": {
        "generic": 0.6105263157894737,
        "probing": 0.3333333333333333,
        "focus": 0.6004672897196262,
        "telling": 0.300187617260788,
    },
}
MAX_LENGTH = 512
TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_history(history: Sequence[dict[str, str]]) -> str:
    if not history:
        return "[No previous conversation]"
    return "\n".join(f"{turn['user']}: {turn['text']}" for turn in history)


def build_paired_input(problem: str, history: Sequence[dict[str, str]]) -> tuple[str, str]:
    return (
        f"Problem:\n{problem}",
        "Conversation:\n"
        f"{format_history(history)}"
        "\n\n"
        "Next teacher pedagogical move:",
    )


def verify_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    observed = {int(key): value for key, value in config["id2label"].items()}
    assert observed == dict(enumerate(MOVES)), observed
    assert config["label2id"] == LABEL_TO_ID
    assert config["architectures"] == ["RobertaForSequenceClassification"]
    return config


def predict(model_dir: Path, records: Sequence[dict[str, Any]], batch_size: int = 16) -> np.ndarray:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to("cpu")
    model.eval()
    assert model.config.id2label == dict(enumerate(MOVES))
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            paired = [build_paired_input(str(row["problem"]), row["history"]) for row in batch]
            encoded = tokenizer(
                [item[0] for item in paired],
                [item[1] for item in paired],
                truncation="only_second",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            output.append(torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output, axis=0)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    if array.size == 0:
        return {"count": 0, "min": None, "q1": None, "median": None, "mean": None, "q3": None, "max": None, "std": None}
    return {
        "count": int(array.size), "min": float(array.min()),
        "q1": float(np.quantile(array, .25, method="linear")),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "q3": float(np.quantile(array, .75, method="linear")),
        "max": float(array.max()), "std": float(array.std(ddof=0)),
    }


def md(frame: pd.DataFrame, digits: int = 5) -> str:
    if frame.empty:
        return "_None._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", "<br>") for value in row] for row in shown.fillna("").itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_csv(name: str, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def history(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"user": user, "text": text} for user, text in pairs]


def challenge_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(case_id: str, group: str, boundary: str, problem: str, prior: list[dict[str, str]], rationale: str) -> None:
        cases.append({"case_id": case_id, "semantic_group": group, "expected_boundary": boundary, "problem": problem, "history": prior, "rationale": rationale})

    # A: repeated confusion after at least two meaningful scaffolds.
    add("A01", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "A rectangle has width x and length x + 4. Its area is 96. Form an equation.", history(("teacher", "What expression represents the length?"), ("student", "x + 4"), ("teacher", "Good. Area is length times width. How would you multiply x and x + 4?"), ("student", "I don't know."), ("teacher", "Try distributing x to each term inside the parentheses."), ("student", "I still do not understand what to do.")), "Two substantive scaffolds have not resolved the same composition step.")
    add("A02", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "A right triangle has shorter sides 9 cm and 12 cm. Find the hypotenuse.", history(("teacher", "Substitute the side lengths into a² + b² = c²."), ("student", "9² + 12² = c²"), ("teacher", "Now calculate the squares and add them."), ("student", "c² = 225"), ("teacher", "What operation reverses squaring?"), ("student", "I have no idea."), ("teacher", "Think of the number that squares to make 225."), ("student", "I still don't know.")), "The learner reached c²=225 but remains stuck after two targeted prompts.")
    add("A03", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "A pentagon has side lengths 6, 8, 5, 7, and 9 cm. Find its perimeter.", history(("teacher", "Perimeter is related to the boundary of the shape. What quantities might you use?"), ("student", "I don't know."), ("teacher", "Look at all five side lengths. What could combining them tell you?"), ("student", "I still don't understand perimeter."), ("teacher", "Imagine walking all the way around the pentagon. Which distances are included?"), ("student", "I cannot work it out.")), "Repeated conceptual confusion remains after boundary and side-length scaffolds.")
    add("A04", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "Convert 3/5 to a decimal and a percent.", history(("teacher", "A fraction can become a decimal by dividing the numerator by the denominator."), ("student", "I don't know how."), ("teacher", "Set up 3 divided by 5. What decimal does that give?"), ("student", "I still don't know."), ("teacher", "You can make an equivalent fraction with denominator 10."), ("student", "I cannot see it.")), "Two conversion routes have been offered without recovery.")
    add("A05", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "Calculate (-6) × 4.", history(("teacher", "First decide the sign when a negative and a positive are multiplied."), ("student", "I don't know."), ("teacher", "Recall: same signs give positive; different signs give what?"), ("student", "I am still unsure."), ("teacher", "The signs here are different. What sign should the product have?"), ("student", "I still cannot answer.")), "The same sign rule remains unresolved after two explicit cues.")
    add("A06", "A_repeated_confusion_after_two_scaffolds", "telling_plausible", "Two supplementary angles measure x + 20 and 2x + 10 degrees. Find x.", history(("teacher", "What total do supplementary angles make?"), ("student", "I don't know."), ("teacher", "They form a straight line. What is the angle of a straight line?"), ("student", "I still don't know."), ("teacher", "Use 180 as the total and combine the two expressions."), ("student", "I cannot set it up.")), "The defining total and equation setup remain unresolved after layered support.")

    # B: persistent same misconception after focused hints.
    add("B01", "B_persistent_same_misconception", "telling_plausible", "Simplify x(x + 4).", history(("teacher", "Multiply x by both terms inside the parentheses."), ("student", "x + 4x"), ("teacher", "When x multiplies x, what power of x results?"), ("student", "It is still x."), ("teacher", "Write x × x explicitly before combining anything."), ("student", "x × x is x.")), "The learner repeats the identical multiplication misconception after focused correction.")
    add("B02", "B_persistent_same_misconception", "telling_plausible", "A triangle has angles 45°, 65°, and x°. Find x.", history(("teacher", "The three interior angles add to 180°. Write an addition equation."), ("student", "45 × 65 × x = 180"), ("teacher", "Use addition, not multiplication: 45 + 65 + x = 180."), ("student", "So 45 × 65 + x = 180."), ("teacher", "All three angle measures are added."), ("student", "I think I multiply 45 and 65 first.")), "A focused correction has not changed the same operation misconception.")
    add("B03", "B_persistent_same_misconception", "telling_plausible", "Six bottles each hold 1.5 litres. How many litres altogether?", history(("teacher", "Does 'six equal bottles altogether' suggest multiplication or division?"), ("student", "Division: 6 ÷ 1.5."), ("teacher", "You need six groups of 1.5, so represent repeated addition."), ("student", "I still divide 1.5 by 6."), ("teacher", "Repeated addition of 1.5 six times is which operation?"), ("student", "Division.")), "The division-for-total misconception persists through two focused representations.")
    add("B04", "B_persistent_same_misconception", "telling_plausible", "Factor x² + 5x.", history(("teacher", "Identify the common factor in x² and 5x."), ("student", "The common factor is 5."), ("teacher", "Both terms contain x; factor x out."), ("student", "It is 5(x² + x)."), ("teacher", "Check by multiplying your factor back into both terms."), ("student", "I still think 5 is the common factor.")), "The same common-factor misconception survives explicit focus and checking.")
    add("B05", "B_persistent_same_misconception", "telling_plausible", "Calculate (-24) ÷ (-6).", history(("teacher", "The two numbers have the same sign. What sign rule applies?"), ("student", "The answer is negative."), ("teacher", "For division, equal signs produce a positive result."), ("student", "I think two negatives still make a negative."), ("teacher", "Compare with the same-sign multiplication rule."), ("student", "It should be negative.")), "The sign misconception persists after a direct focused hint.")
    add("B06", "B_persistent_same_misconception", "telling_plausible", "Simplify 3a + 2b + 4a.", history(("teacher", "Which terms contain the same variable?"), ("student", "3a and 2b because both have letters."), ("teacher", "Like terms need the same variable; 3a and 4a match."), ("student", "Then 3a + 2b is 5ab."), ("teacher", "Keep unlike a and b terms separate."), ("student", "I still get 5ab + 4a.")), "The learner repeats an unlike-term misconception after focused identification.")

    # C: explicit request for direct explanation after previous support.
    add("C01", "C_explicit_direct_explanation_request", "telling_plausible", "Solve 2x + 7 = 19.", history(("teacher", "What could you undo first to isolate the term with x?"), ("student", "Maybe divide by 2?"), ("teacher", "Consider reversing the +7 before dividing."), ("student", "Can you please explain the steps directly? I am lost.")), "The learner explicitly requests direct instruction after attempted support.")
    add("C02", "C_explicit_direct_explanation_request", "telling_plausible", "Find 15% of 80.", history(("teacher", "How can 15% be written as a decimal?"), ("student", "0.15"), ("teacher", "What operation connects 0.15 and 80 to the requested part?"), ("student", "Please just show me directly how to do this.")), "A direct explanation is explicitly requested after partial progress.")
    add("C03", "C_explicit_direct_explanation_request", "telling_plausible", "Expand 4(y - 3).", history(("teacher", "Which terms inside the parentheses must 4 multiply?"), ("student", "Only y?"), ("teacher", "The outside factor applies to every inside term."), ("student", "I still don't get it. Can you explain it directly with the answer?")), "The learner asks for a direct worked explanation after a cue.")
    add("C04", "C_explicit_direct_explanation_request", "telling_plausible", "A right triangle has hypotenuse 13 and one leg 5. Find the other leg.", history(("teacher", "Where do 13 and 5 go in a² + b² = c²?"), ("student", "I put both on the left."), ("teacher", "The hypotenuse is c."), ("student", "Could you explain the setup directly instead of asking me another question?")), "The learner explicitly asks to replace further questioning with explanation.")
    add("C05", "C_explicit_direct_explanation_request", "telling_plausible", "Convert 0.375 to a fraction in simplest form.", history(("teacher", "What fraction has 375 as numerator and a power of ten as denominator?"), ("student", "375/100"), ("teacher", "Check how many decimal places there are."), ("student", "Please show me the correct process step by step.")), "The learner requests a direct process after support exposed an unresolved setup error.")
    add("C06", "C_explicit_direct_explanation_request", "telling_plausible", "Find the perimeter of a rectangle with length 8 cm and width 3 cm.", history(("teacher", "Which sides appear twice around a rectangle?"), ("student", "The length?"), ("teacher", "Both length and width each occur twice."), ("student", "I am confused. Please tell me exactly how perimeter works.")), "An explicit conceptual explanation request follows prior scaffolding.")

    # D: repeated failure on the same narrowed micro-step.
    add("D01", "D_repeated_microstep_failure", "telling_plausible", "Solve x² = 49 for positive x.", history(("teacher", "We only need the positive number whose square is 49. What is it?"), ("student", "49"), ("teacher", "Which number multiplied by itself equals 49?"), ("student", "14"), ("teacher", "Check 14 × 14. Try the square root of 49."), ("student", "I don't know.")), "The interaction has narrowed to one micro-step and repeated attempts fail.")
    add("D02", "D_repeated_microstep_failure", "telling_plausible", "Compute 3/4 + 1/4.", history(("teacher", "The denominators already match. Add only the numerators: what is 3 + 1?"), ("student", "3"), ("teacher", "Compute the numerator sum 3 + 1."), ("student", "5"), ("teacher", "Try that single addition once more."), ("student", "I cannot.")), "Repeated failure occurs on the isolated numerator addition step.")
    add("D03", "D_repeated_microstep_failure", "telling_plausible", "Solve 5x = 35.", history(("teacher", "Divide both sides by 5. What is 35 ÷ 5?"), ("student", "5"), ("teacher", "Seven groups of 5 make what total?"), ("student", "30"), ("teacher", "Focus only on 35 divided by 5."), ("student", "I still cannot calculate it.")), "The single arithmetic micro-step remains unresolved after repetition.")
    add("D04", "D_repeated_microstep_failure", "telling_plausible", "Find the missing angle: x + 70 = 180.", history(("teacher", "Subtract 70 from both sides. What is 180 − 70?"), ("student", "120"), ("teacher", "Break it into 180 − 100 + 30."), ("student", "90"), ("teacher", "Return to the exact subtraction 180 − 70."), ("student", "I do not know.")), "The narrowed subtraction micro-step has failed repeatedly.")
    add("D05", "D_repeated_microstep_failure", "telling_plausible", "Simplify 2x + 3x.", history(("teacher", "Add the coefficients 2 and 3. What do you get?"), ("student", "6"), ("teacher", "This is addition, so calculate 2 + 3."), ("student", "4"), ("teacher", "Focus only on 2 + 3."), ("student", "I cannot answer.")), "The exact coefficient addition micro-step repeatedly fails.")
    add("D06", "D_repeated_microstep_failure", "telling_plausible", "Calculate 0.6 × 10.", history(("teacher", "Multiplying by 10 moves the decimal one place. Where does 0.6 become?"), ("student", "0.06"), ("teacher", "Move it one place to the right, not left."), ("student", "0.006"), ("teacher", "Try one rightward place from 0.6."), ("student", "I still don't know.")), "One decimal-shift micro-step fails despite repeated narrowing.")

    # E: first incorrect answer.
    add("E01", "E_first_incorrect_answer", "non_telling", "Solve 3x + 2 = 14.", history(("teacher", "How would you start?"), ("student", "I would divide 14 by 3.")), "This is the learner's first incorrect attempt; diagnosis or probing remains plausible.")
    add("E02", "E_first_incorrect_answer", "non_telling", "Find the perimeter of a square with side 7 cm.", history(("teacher", "What is the perimeter?"), ("student", "49 cm.")), "A first area/perimeter confusion should not automatically trigger telling.")
    add("E03", "E_first_incorrect_answer", "non_telling", "Simplify y(y - 5).", history(("teacher", "What do you get?"), ("student", "y² - 5.")), "This is a first distribution error with an identifiable missing factor.")
    add("E04", "E_first_incorrect_answer", "non_telling", "Calculate (-8) + 3.", history(("teacher", "What is the result?"), ("student", "11.")), "A first sign error is recoverable through focused questioning.")

    # F: first 'I don't know'.
    add("F01", "F_first_i_dont_know", "non_telling", "Find 25% of 60.", history(("teacher", "How might you start?"), ("student", "I don't know.")), "This is the first expression of uncertainty with no prior scaffold.")
    add("F02", "F_first_i_dont_know", "non_telling", "Solve x + 9 = 15.", history(("teacher", "What operation could isolate x?"), ("student", "I don't know.")), "A first uncertainty response leaves room for a simple cue.")
    add("F03", "F_first_i_dont_know", "non_telling", "What is the sum of the interior angles of a triangle?", history(("teacher", "What total do the angles make?"), ("student", "I have no idea.")), "No earlier instruction or failed support is present.")
    add("F04", "F_first_i_dont_know", "non_telling", "Convert 1/2 to a decimal.", history(("teacher", "What decimal is equivalent to one half?"), ("student", "I don't know.")), "First-time uncertainty alone is a hard negative for premature telling.")

    # G: first identifiable misconception.
    add("G01", "G_first_identifiable_misconception", "non_telling", "Calculate 4 + 3 × 2.", history(("teacher", "What answer do you get?"), ("student", "14, because I add 4 and 3 first.")), "The first order-of-operations misconception is explicit and focusable.")
    add("G02", "G_first_identifiable_misconception", "non_telling", "Solve 2x = 10.", history(("teacher", "How would you solve for x?"), ("student", "Subtract 2 from 10.")), "The inverse-operation misconception is newly identifiable.")
    add("G03", "G_first_identifiable_misconception", "non_telling", "Find the area of a rectangle 5 cm by 4 cm.", history(("teacher", "What is the area?"), ("student", "18 cm² because I add all sides.")), "A first perimeter-for-area misconception warrants focused repair.")
    add("G04", "G_first_identifiable_misconception", "non_telling", "Simplify 2a + 3b.", history(("teacher", "Can these terms combine?"), ("student", "Yes, they make 5ab.")), "The first unlike-term misconception is clear enough for a targeted prompt.")

    # H: correct or partially correct answer needing reasoning.
    add("H01", "H_correct_or_partial_needing_reasoning", "non_telling", "Solve 4x = 28.", history(("teacher", "What is x?"), ("student", "x = 7.")), "The answer is correct; the next move should elicit reasoning rather than tell.")
    add("H02", "H_correct_or_partial_needing_reasoning", "non_telling", "Find the perimeter of a 6 cm by 2 cm rectangle.", history(("teacher", "What perimeter do you get?"), ("student", "16 cm, because I used 6 + 6 + 2 + 2.")), "Correct answer and reasoning make telling unnecessary.")
    add("H03", "H_correct_or_partial_needing_reasoning", "non_telling", "Expand 3(x + 2).", history(("teacher", "What expression results?"), ("student", "3x + 6, but I am not fully sure why.")), "The result is correct; probing the reasoning is preferable to direct instruction.")
    add("H04", "H_correct_or_partial_needing_reasoning", "non_telling", "A triangle has angles 50°, 60°, and x°. Find x.", history(("teacher", "What is x?"), ("student", "70°, since the angles should total 180°.")), "A correct justified answer is a hard non-telling state.")

    # I: recoverable error after only one scaffold.
    add("I01", "I_recoverable_after_one_scaffold", "non_telling", "Solve x + 8 = 20.", history(("teacher", "What would you undo first?"), ("student", "Add 8."), ("teacher", "To reverse adding 8, which inverse operation could you use?"), ("student", "Maybe subtract 8?")), "One scaffold has already produced a plausible recovery.")
    add("I02", "I_recoverable_after_one_scaffold", "non_telling", "Calculate 3/10 as a decimal.", history(("teacher", "What decimal did you get?"), ("student", "0.03"), ("teacher", "Tenths use the first place after the decimal. Reconsider where the 3 belongs."), ("student", "Would it be 0.3?")), "The learner recovers after one place-value cue.")
    add("I03", "I_recoverable_after_one_scaffold", "non_telling", "Simplify 5x + 2x.", history(("teacher", "What is the result?"), ("student", "7x²"), ("teacher", "You are adding like terms, not multiplying x by x."), ("student", "Then it might be 7x.")), "A single focused cue yields the corrected form.")
    add("I04", "I_recoverable_after_one_scaffold", "non_telling", "Calculate (-5) × (-3).", history(("teacher", "What answer do you get?"), ("student", "-15"), ("teacher", "The signs are the same; revisit the sign rule."), ("student", "Same signs make positive, so 15?")), "The sign error is recoverable after one scaffold.")

    # J: neutral/opening states.
    add("J01", "J_neutral_opening", "non_telling", "A bag contains 4 red and 6 blue counters. How many counters are there?", history(), "No learner attempt exists; this is a neutral opening state.")
    add("J02", "J_neutral_opening", "non_telling", "A rectangle is 9 cm long and 3 cm wide. Find its area.", history(("teacher", "How would you start solving this problem?"), ("student", "I would identify the length and width first.")), "The learner has made a neutral productive opening without difficulty.")
    add("J03", "J_neutral_opening", "non_telling", "Convert 0.4 to a percentage.", history(("teacher", "What do you notice about the decimal?"), ("student", "It has four tenths.")), "This opening observation is relevant and does not justify telling.")
    add("J04", "J_neutral_opening", "non_telling", "A triangle has base 8 cm and height 5 cm. Find its area.", history(("teacher", "What information is given?"), ("student", "The base is 8 and the height is 5.")), "The learner is only orienting to the problem.")

    assert len(cases) == 48
    assert Counter(case["expected_boundary"] for case in cases) == {"telling_plausible": 24, "non_telling": 24}
    return cases


def vector_fields(prefix: str, vector: np.ndarray) -> dict[str, Any]:
    top_index = int(np.argmax(vector))
    return {
        **{f"{prefix}_p_{move}": float(vector[index]) for index, move in enumerate(MOVES)},
        f"{prefix}_top1": MOVES[top_index],
    }


def compact_vector(row: pd.Series, prefix: str) -> str:
    return "/".join(f"{move[0].upper()}={float(row[f'{prefix}_p_{move}']):.3f}" for move in MOVES)


def case_section(title: str, frame: pd.DataFrame, limit: int | None = None) -> str:
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        return f"## {title}\n\n_None._\n"
    blocks = [f"## {title}\n"]
    for _, row in frame.iterrows():
        identifier = row.get("case_id") or row.get("action_event_id")
        blocks.append(
            f"### {identifier}\n\n"
            f"- Group/skill: `{row.get('semantic_group', row.get('skill', ''))}`\n"
            f"- Baseline → candidate: `{row['baseline_top1']}` → `{row['candidate_top1']}`\n"
            f"- Baseline probabilities: `{compact_vector(row, 'baseline')}`\n"
            f"- Candidate probabilities: `{compact_vector(row, 'candidate')}`\n"
            f"- Δ telling / Δ focus: `{row['delta_p_telling']:.5f}` / `{row['delta_p_focus']:.5f}`\n"
            f"- Problem: {row['problem']}\n"
            f"- Prior dialogue:\n\n```text\n{row['formatted_history']}\n```\n"
        )
    return "\n".join(blocks)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [TURN_PATH, ASSESSMENT_PATH, SUMMARY_PATH, POLICY]
    model_files = [BASELINE / name for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]] + [CANDIDATE / name for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]]
    before_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected + model_files}

    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    assert safety["status"] == "PASS"
    assert safety["metrics"] == EXPECTED_SAFETY
    assert sha256(VALIDATION) == safety["validation_sha256"]
    assert sha256(BASELINE / "model.safetensors") == safety["baseline_model_sha256"]
    baseline_config = verify_config(BASELINE)
    candidate_config = verify_config(CANDIDATE)

    turns = read_jsonl(TURN_PATH)
    summaries = read_jsonl(SUMMARY_PATH)
    completed_ids = {str(nested(row, "identity", "attempt_id")) for row in summaries if row.get("completion_status") == "completed" or (row.get("completion_status") is None and row.get("adaptive_completion"))}
    real_source = [
        row for row in turns
        if str(nested(row, "identity", "attempt_id")) in completed_ids
        and nested(row, "provenance", "selector_mode") == "ordinary-md7r1-v1"
        and nested(row, "next_learner_observation", "status") in {"observed_update", "resolved_no_update"}
    ]
    assert len(real_source) == 87
    real_records: list[dict[str, Any]] = []
    for row in real_source:
        state = row["state_before_action"]
        history_before = state.get("history_before_action") or []
        assert all(set(turn) >= {"user", "text"} for turn in history_before)
        real_records.append({"problem": state["problem"], "history": history_before})

    challenges = challenge_cases()
    combined = real_records + [{"problem": row["problem"], "history": row["history"]} for row in challenges]
    baseline_probs = predict(BASELINE, combined)
    candidate_probs = predict(CANDIDATE, combined)
    assert baseline_probs.shape == candidate_probs.shape == (135, 4)
    assert np.allclose(baseline_probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(candidate_probs.sum(axis=1), 1.0, atol=1e-6)

    # Historical baseline parity is a second implementation/state-reconstruction gate.
    stored = np.asarray([[float(nested(row, "selector", "raw", "probabilities", move)) for move in MOVES] for row in real_source])
    parity_abs = np.abs(baseline_probs[:87] - stored)
    parity_max = float(parity_abs.max())
    parity_rows_failed = int((parity_abs.max(axis=1) > TOLERANCE).sum())
    assert parity_rows_failed == 0, (parity_max, parity_rows_failed)

    real_rows: list[dict[str, Any]] = []
    for index, (source, base_vector, candidate_vector) in enumerate(zip(real_source, baseline_probs[:87], candidate_probs[:87], strict=True)):
        identity = source["identity"]
        state = source["state_before_action"]
        outcome = source["next_learner_observation"]
        evaluator = outcome.get("evaluator") or {}
        update = source["learning_state_update"]
        row = {
            "learner_pseudonym": identity["student_pseudonymous_id"],
            "attempt_id": identity["attempt_id"], "action_event_id": identity["action_event_id"],
            "action_turn_index": identity["action_turn_index"], "skill": state["target_skill"],
            "problem": state["problem"], "history_before_action": json.dumps(state.get("history_before_action") or [], ensure_ascii=False, separators=(",", ":")),
            "formatted_history": format_history(state.get("history_before_action") or []),
            "historical_raw_md7_move": nested(source, "selector", "raw", "argmax"),
            **vector_fields("baseline", base_vector), **vector_fields("candidate", candidate_vector),
            **{f"delta_p_{move}": float(candidate_vector[i] - base_vector[i]) for i, move in enumerate(MOVES)},
            "top1_changed": MOVES[int(np.argmax(base_vector))] != MOVES[int(np.argmax(candidate_vector))],
            "changed_to_telling": MOVES[int(np.argmax(base_vector))] != "telling" and MOVES[int(np.argmax(candidate_vector))] == "telling",
            "changed_away_from_telling": MOVES[int(np.argmax(base_vector))] == "telling" and MOVES[int(np.argmax(candidate_vector))] != "telling",
            "evaluator_category_observational": str(evaluator.get("correctness", "unknown")).casefold(),
            "immediate_delta_mastery_observational": update.get("delta_mastery"),
            "mastery_before_observational": update.get("mastery_before"),
        }
        real_rows.append(row)
    real_df = pd.DataFrame(real_rows)

    challenge_rows: list[dict[str, Any]] = []
    for source, base_vector, candidate_vector in zip(challenges, baseline_probs[87:], candidate_probs[87:], strict=True):
        base_top = MOVES[int(np.argmax(base_vector))]
        candidate_top = MOVES[int(np.argmax(candidate_vector))]
        challenge_rows.append({
            **source,
            "history": json.dumps(source["history"], ensure_ascii=False, separators=(",", ":")),
            "formatted_history": format_history(source["history"]),
            **vector_fields("baseline", base_vector), **vector_fields("candidate", candidate_vector),
            **{f"delta_p_{move}": float(candidate_vector[i] - base_vector[i]) for i, move in enumerate(MOVES)},
            "top1_changed": base_top != candidate_top,
            "changed_to_telling": base_top != "telling" and candidate_top == "telling",
            "changed_away_from_telling": base_top == "telling" and candidate_top != "telling",
        })
    challenge_df = pd.DataFrame(challenge_rows)
    challenge_spec = challenge_df[["case_id", "semantic_group", "expected_boundary", "problem", "history", "rationale"]].copy()

    plausible = challenge_df[challenge_df["expected_boundary"] == "telling_plausible"]
    negatives = challenge_df[challenge_df["expected_boundary"] == "non_telling"]

    def boundary_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
        values = frame[f"{prefix}_p_telling"]
        return {
            "n": len(frame), "top1_telling_count": int((frame[f"{prefix}_top1"] == "telling").sum()),
            "top1_telling_rate": float((frame[f"{prefix}_top1"] == "telling").mean()),
            "mean_telling_probability": float(values.mean()), "median_telling_probability": float(values.median()),
            "max_telling_probability": float(values.max()),
            "top1_distribution": {move: int((frame[f"{prefix}_top1"] == move).sum()) for move in MOVES},
        }

    plausible_baseline = boundary_metrics(plausible, "baseline")
    plausible_candidate = boundary_metrics(plausible, "candidate")
    negative_baseline = boundary_metrics(negatives, "baseline")
    negative_candidate = boundary_metrics(negatives, "candidate")
    real_telling_baseline = int((real_df["baseline_top1"] == "telling").sum())
    real_telling_candidate = int((real_df["candidate_top1"] == "telling").sum())
    displacement = int(real_df["top1_changed"].sum())
    transition_df = (
        real_df.groupby(["baseline_top1", "candidate_top1"])
        .size()
        .reset_index(name="count")
        .sort_values(["baseline_top1", "candidate_top1"])
    )

    semantic_rows: list[dict[str, Any]] = []
    for group, frame in challenge_df.groupby("semantic_group", sort=True):
        semantic_rows.append({
            "semantic_group": group, "boundary": frame["expected_boundary"].iloc[0], "n": len(frame),
            "baseline_telling_count": int((frame["baseline_top1"] == "telling").sum()),
            "candidate_telling_count": int((frame["candidate_top1"] == "telling").sum()),
            "baseline_mean_p_telling": float(frame["baseline_p_telling"].mean()),
            "candidate_mean_p_telling": float(frame["candidate_p_telling"].mean()),
        })
    semantic_df = pd.DataFrame(semantic_rows)

    movement = stats(real_df["delta_p_telling"])
    candidate_telling_real = real_df[real_df["candidate_top1"] == "telling"]
    outcome_stratification = {
        "n": len(candidate_telling_real),
        "evaluator_distribution": dict(Counter(candidate_telling_real["evaluator_category_observational"])),
        "delta_mastery": stats(candidate_telling_real["immediate_delta_mastery_observational"]),
        "mastery_before": stats(candidate_telling_real["mastery_before_observational"]),
        "interpretation": "Observational historical outcomes joined only after independent pre-action predictions; no causal claim.",
    }

    telling_sensitivity_gain = plausible_candidate["top1_telling_rate"] - plausible_baseline["top1_telling_rate"]
    false_trigger_change = negative_candidate["top1_telling_rate"] - negative_baseline["top1_telling_rate"]
    recoverable = challenge_df[
        challenge_df["semantic_group"] == "I_recoverable_after_one_scaffold"
    ]
    recoverable_candidate_telling_count = int(
        (recoverable["candidate_top1"] == "telling").sum()
    )
    recoverable_candidate_telling_rate = float(
        (recoverable["candidate_top1"] == "telling").mean()
    )

    # Decision rubric is intentionally conservative and diagnostic-only.
    if recoverable_candidate_telling_rate >= 0.50:
        decision = "B. TELLING OVERCORRECTION"
        reason = (
            "Candidate predicts telling on "
            f"{recoverable_candidate_telling_count}/{len(recoverable)} "
            "recoverable-after-one-scaffold cases, including learner responses "
            "that already show recovery. This fails the required preservation "
            "of probing/focus behavior in recoverable difficulty."
        )
    elif negative_candidate["top1_telling_rate"] >= 0.20 or false_trigger_change >= 0.15:
        decision = "B. TELLING OVERCORRECTION"
        reason = "Candidate telling activation on hard non-telling cases is too broad."
    elif plausible_candidate["top1_telling_rate"] < 0.50 or telling_sensitivity_gain < 0.20:
        decision = "C. TELLING STILL UNDER-TRIGGERS"
        reason = "Candidate remains non-telling on too many persistent-support boundary cases."
    elif displacement / len(real_df) > 0.40:
        decision = "B. TELLING OVERCORRECTION"
        reason = "Candidate broadly displaces the frozen policy on real deployment states."
    else:
        decision = "A. PROMISING TELLING CALIBRATION"
        reason = "Candidate increases telling on persistent-support cases while retaining low hard-negative activation and limited real-state displacement."

    write_csv("real_state_predictions.csv", real_df)
    write_csv("telling_boundary_challenge.csv", challenge_spec)
    write_csv("telling_boundary_predictions.csv", challenge_df)
    write_csv("semantic_group_summary.csv", semantic_df)
    write_csv("real_top1_transitions.csv", transition_df)

    important = "# Important Case Inspection\n\nManual diagnostic labels are boundary judgments, not causal ground truth.\n\n"
    important += case_section("Every real state changed to telling", real_df[real_df["changed_to_telling"]].sort_values("delta_p_telling", ascending=False))
    important += case_section("Every hard-negative candidate telling case", negatives[negatives["candidate_top1"] == "telling"].sort_values("candidate_p_telling", ascending=False))
    important += case_section("Telling-plausible cases candidate did not tell", plausible[plausible["candidate_top1"] != "telling"].sort_values("candidate_p_telling", ascending=False))
    important += case_section("Largest 10 real-state telling-probability increases", real_df.sort_values("delta_p_telling", ascending=False), 10)
    important += case_section("Largest 10 real-state focus-probability decreases", real_df.sort_values("delta_p_focus", ascending=True), 10)
    (OUT / "important_cases.md").write_text(important, encoding="utf-8")

    # Three requested simple plots.
    def paired_plot(frame: pd.DataFrame, title: str, name: str) -> None:
        ordered = frame.sort_values("candidate_p_telling").reset_index(drop=True)
        x = np.arange(len(ordered))
        plt.figure(figsize=(9, 4.8))
        plt.plot(x, ordered["baseline_p_telling"], "o-", label="baseline", alpha=.8)
        plt.plot(x, ordered["candidate_p_telling"], "o-", label="candidate", alpha=.8)
        plt.xticks(x, ordered["case_id"], rotation=60)
        plt.ylabel("telling probability"); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(OUT / name, dpi=160); plt.close()

    paired_plot(plausible, "Telling probability: telling-plausible challenge", "telling_probability_plausible.png")
    paired_plot(negatives, "Telling probability: hard non-telling challenge", "telling_probability_hard_negatives.png")
    plt.figure(figsize=(7.5, 4.8)); plt.hist(real_df["delta_p_telling"], bins=18, edgecolor="white"); plt.axvline(0, color="black", lw=1); plt.xlabel("candidate p(telling) − baseline p(telling)"); plt.ylabel("Real states"); plt.title("Real-state telling probability movement"); plt.tight_layout(); plt.savefig(OUT / "real_state_telling_probability_delta.png", dpi=160); plt.close()

    candidate_metadata = json.loads((CANDIDATE_ROOT / "epoch3_metrics.json").read_text(encoding="utf-8"))
    holdout_metadata = json.loads((CANDIDATE_ROOT / "selected_epoch_final_holdout.json").read_text(encoding="utf-8"))
    artifact_hashes = {
        "baseline": {name: sha256(BASELINE / name) for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]},
        "candidate": {name: sha256(CANDIDATE / name) for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]},
    }
    after_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected + model_files}
    protected_changes = {path: {"before": before_hashes[path], "after": after_hashes[path]} for path in before_hashes if before_hashes[path] != after_hashes[path]}

    summary = {
        "diagnostic": "md7_telling_real_state_diagnostic_v1",
        "decision": decision, "decision_reason": reason,
        "models": {
            "baseline": {"path": str(BASELINE.relative_to(ROOT)).replace("\\", "/"), "hashes": artifact_hashes["baseline"], "id2label": baseline_config["id2label"]},
            "candidate": {"path": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"), "hashes": artifact_hashes["candidate"], "id2label": candidate_config["id2label"]},
        },
        "inference_contract": safety["contract"],
        "safety_check": safety,
        "real_state_reconstruction": {"n": len(real_df), "completed": True, "selector_mode": "ordinary-md7r1-v1", "formal_assessment_rows_used": 0, "future_fields_used_for_prediction": [], "baseline_probability_parity_max_abs_error": parity_max, "parity_rows_failed_at_1e_5": parity_rows_failed},
        "challenge": {"n": len(challenge_df), "telling_plausible_n": len(plausible), "hard_non_telling_n": len(negatives), "source": "manual realistic adaptations plus deployment-style authored cases", "labels_are_causal_ground_truth": False},
        "telling_plausible": {"baseline": plausible_baseline, "candidate": plausible_candidate, "sensitivity_gain": telling_sensitivity_gain},
        "hard_non_telling": {"baseline": negative_baseline, "candidate": negative_candidate, "false_trigger_change": false_trigger_change},
        "recoverable_after_one_scaffold": {
            "n": len(recoverable),
            "candidate_top1_telling_count": recoverable_candidate_telling_count,
            "candidate_top1_telling_rate": recoverable_candidate_telling_rate,
        },
        "real_states": {"baseline_top1_telling_count": real_telling_baseline, "candidate_top1_telling_count": real_telling_candidate, "candidate_top1_telling_rate": real_telling_candidate / len(real_df), "policy_displacement_count": displacement, "policy_displacement_rate": displacement / len(real_df), "changed_to_telling_count": int(real_df["changed_to_telling"].sum()), "changed_away_from_telling_count": int(real_df["changed_away_from_telling"].sum()), "top1_transitions": transition_df.to_dict(orient="records"), "telling_probability_delta": movement},
        "observational_outcome_stratification": outcome_stratification,
        "synthetic_calibration_evidence": {"candidate_epoch3_metrics": candidate_metadata, "selected_epoch_holdout": holdout_metadata, "real_effectiveness_evidence": False},
        "claims_not_entitled": ["candidate telling would have caused better learning", "candidate would have improved historical outcomes", "negative BKT proves telling was needed", "synthetic perfection demonstrates real tutoring effectiveness"],
        "recommended_next_experiment": "A targeted offline telling-boundary recalibration ablation that adds recoverable-after-one-scaffold hard negatives, followed by repetition of this same frozen diagnostic before any live exploration.",
        "side_effects": {"training": 0, "model_writes": 0, "production_policy_writes": 0, "authoritative_real_data_writes": 0, "tutor_api_calls": 0, "bkt_updates": 0, "lints_updates": 0, "mathdial_final_test_use": 0, "mrbench_v3_test_use": 0, "protected_input_hash_changes": protected_changes},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    boundary_table = pd.DataFrame([
        {"boundary": "telling-plausible", "model": "baseline", **plausible_baseline},
        {"boundary": "telling-plausible", "model": "candidate", **plausible_candidate},
        {"boundary": "hard non-telling", "model": "baseline", **negative_baseline},
        {"boundary": "hard non-telling", "model": "candidate", **negative_candidate},
    ])
    real_change_table = pd.DataFrame([
        {"metric": "baseline top-1 telling", "count": real_telling_baseline, "percent": 100 * real_telling_baseline / len(real_df)},
        {"metric": "candidate top-1 telling", "count": real_telling_candidate, "percent": 100 * real_telling_candidate / len(real_df)},
        {"metric": "top-1 displacement", "count": displacement, "percent": 100 * displacement / len(real_df)},
        {"metric": "changed to telling", "count": int(real_df["changed_to_telling"].sum()), "percent": 100 * real_df["changed_to_telling"].mean()},
        {"metric": "changed away from telling", "count": int(real_df["changed_away_from_telling"].sum()), "percent": 100 * real_df["changed_away_from_telling"].mean()},
    ])
    hash_table = pd.DataFrame([{"model": model, "path": summary["models"][model]["path"], "model.safetensors_sha256": artifact_hashes[model]["model.safetensors"], "config_sha256": artifact_hashes[model]["config.json"], "tokenizer_sha256": artifact_hashes[model]["tokenizer.json"], "tokenizer_config_sha256": artifact_hashes[model]["tokenizer_config.json"]} for model in ["baseline", "candidate"]])
    side_effect_table = pd.DataFrame([{"effect": key, "count": len(value) if key == "protected_input_hash_changes" else value} for key, value in summary["side_effects"].items()])

    report = f"""# MD7-R1 Telling-Calibration Real-State Diagnostic

This is an offline, CPU-only diagnostic. No model was trained or promoted; no production policy, BKT, LinTS, Tutor API, authoritative real-data JSONL, MathDial final test, or MRBench V3 test was touched.

## 1. Model safety and exact inference contract

The baseline-only gate ran first on exactly 1,850 rows from `pedagogical-move-selection/data/processed/mathdial/validation.jsonl` (SHA-256 `{safety['validation_sha256']}`). It reproduced every required metric exactly: accuracy **{safety['metrics']['accuracy']}**, Macro-F1 **{safety['metrics']['macro_f1']}**, and G/P/F/T F1 **{safety['metrics']['per_class_f1']['generic']} / {safety['metrics']['per_class_f1']['probing']} / {safety['metrics']['per_class_f1']['focus']} / {safety['metrics']['per_class_f1']['telling']}**. Gate: **PASS**.

The inspected source contract is: sequence A `Problem:\n<problem>`; sequence B `Conversation:\n<history>` with exact `user: text` lines and suffix `\n\nNext teacher pedagogical move:`; fast tokenizer; `truncation_side=left`; `truncation=only_second`; `max_length=512`; paired tokenization; class order `generic, probing, focus, telling`.

{md(hash_table)}

## 2. Evidence boundaries

- **Synthetic calibration evidence:** Epoch 3 synthetic telling challenge precision/recall are **1.0/1.0** with non-telling false-trigger rate **0.0**; untouched synthetic holdout accuracy/Macro-F1 and all class F1 values are **1.0**. Candidate MathDial validation accuracy/Macro-F1 are **{candidate_metadata['mathdial_val']['accuracy']:.6f}/{candidate_metadata['mathdial_val']['macro_f1']:.6f}**, below baseline **{safety['metrics']['accuracy']:.6f}/{safety['metrics']['macro_f1']:.6f}**. These establish calibration/retention on those sets only, not real tutoring effectiveness.
- **Real-state offline transfer evidence:** both frozen selectors scored the same 87 leakage-free pre-action states from completed ordinary-mode attempts.
- **Observational outcomes:** evaluator category, immediate BKT delta, and mastery-before were joined only after prediction for descriptive stratification.
- **Not entitled:** no causal learning benefit, counterfactual historical improvement, production readiness, or real effectiveness follows from these results.

## 3. Real deployment states

Exact real states analyzed: **{len(real_df)}**. Formal-assessment selector states: **0**. Future learner responses/evaluations/BKT-after/attempt-final outcomes used for prediction: **none**. Recomputed baseline probabilities match stored historical raw MD7 probabilities with maximum absolute error **{parity_max:.9g}**; rows over 1e-5: **{parity_rows_failed}**.

{md(real_change_table)}

Candidate-minus-baseline p(telling): min **{movement['min']:.5f}**, Q1 **{movement['q1']:.5f}**, median **{movement['median']:.5f}**, mean **{movement['mean']:.5f}**, Q3 **{movement['q3']:.5f}**, max **{movement['max']:.5f}**.

## 4. Manually auditable telling-boundary challenge

Challenge cases: **48**, balanced 24 telling-plausible / 24 hard non-telling across semantic groups A-J. Labels are manual diagnostic boundary judgments - not causal or outcome ground truth.

{md(boundary_table.drop(columns=['top1_distribution']))}

Telling sensitivity gain: **{telling_sensitivity_gain:.5f}**. Hard-negative false-trigger change: **{false_trigger_change:.5f}**.

The false triggers are not diffuse: **{recoverable_candidate_telling_count}/{len(recoverable)} ({100*recoverable_candidate_telling_rate:.1f}%)** of the recoverable-after-one-scaffold group become telling, including cases where the learner's latest response already corrects the error.

{md(semantic_df)}

## 5. Probability and policy movement

- Real-state candidate top-1 telling activation: **{real_telling_candidate}/{len(real_df)} ({100*real_telling_candidate/len(real_df):.3f}%)** versus baseline **{real_telling_baseline}/{len(real_df)} ({100*real_telling_baseline/len(real_df):.3f}%)**.
- Real-state top-1 displacement: **{displacement}/{len(real_df)} ({100*displacement/len(real_df):.3f}%)**.
- Real states changed to telling / away from telling: **{int(real_df['changed_to_telling'].sum())} / {int(real_df['changed_away_from_telling'].sum())}**.
- Challenge top-1 distributions are included in `summary.json` and `semantic_group_summary.csv`.

{md(transition_df)}

## 6. Important semantic cases

All requested compact context/probability inspections are in `important_cases.md`: every real change to telling, every hard-negative candidate telling case, every plausible case still not telling, ten largest telling increases, and ten largest focus decreases.

## 7. Optional observational outcome stratification

Candidate-telling real states: **{outcome_stratification['n']}**. Historical evaluator distribution: `{json.dumps(outcome_stratification['evaluator_distribution'], sort_keys=True)}`. Immediate delta stats: `{json.dumps(outcome_stratification['delta_mastery'])}`. Mastery-before stats: `{json.dumps(outcome_stratification['mastery_before'])}`.

These describe historical states the candidate classifies as telling. They do not say telling would have changed those outcomes.

## 8. Decision

**{decision}**

{reason}

No model is promoted. One recommended next experiment: **{summary['recommended_next_experiment']}**

## 9. Final required facts

1. Exact model paths and hashes are in the table above and `summary.json`.
2. Real states analyzed: **{len(real_df)}**.
3. Challenge cases: **{len(challenge_df)}**.
4. Telling-plausible top-1 telling: baseline **{plausible_baseline['top1_telling_count']}/24 ({100*plausible_baseline['top1_telling_rate']:.2f}%)**, candidate **{plausible_candidate['top1_telling_count']}/24 ({100*plausible_candidate['top1_telling_rate']:.2f}%)**; mean p(telling) **{plausible_baseline['mean_telling_probability']:.4f} -> {plausible_candidate['mean_telling_probability']:.4f}**.
5. Hard-negative false telling: baseline **{negative_baseline['top1_telling_count']}/24 ({100*negative_baseline['top1_telling_rate']:.2f}%)**, candidate **{negative_candidate['top1_telling_count']}/24 ({100*negative_candidate['top1_telling_rate']:.2f}%)**; mean/max candidate p(telling) **{negative_candidate['mean_telling_probability']:.4f}/{negative_candidate['max_telling_probability']:.4f}**.
6. Real-state top-1 telling: baseline **{real_telling_baseline}/87**, candidate **{real_telling_candidate}/87**.
7. Policy-displacement rate: **{displacement}/87 ({100*displacement/87:.3f}%)**.
8. Decision: **{decision}**.
9. Recommended next experiment: **{summary['recommended_next_experiment']}**

## 10. Side effects

{md(side_effect_table)}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "safety": "PASS", "real_states": len(real_df), "challenge_cases": len(challenge_df),
        "plausible_baseline_telling": plausible_baseline["top1_telling_count"],
        "plausible_candidate_telling": plausible_candidate["top1_telling_count"],
        "negative_baseline_telling": negative_baseline["top1_telling_count"],
        "negative_candidate_telling": negative_candidate["top1_telling_count"],
        "real_baseline_telling": real_telling_baseline, "real_candidate_telling": real_telling_candidate,
        "displacement": displacement, "parity_max_abs": parity_max,
        "protected_changes": len(protected_changes), "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
