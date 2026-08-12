from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
SEED = 42
MOVE_ORDER = ["generic", "probing", "focus", "telling"]
SCHEMA_VERSION = "md7_r2_tell_c1_correction_v1"

BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
MATHDIAL_TRAIN = ROOT / "pedagogical-move-selection" / "data" / "processed" / "mathdial" / "train.jsonl"
MATHDIAL_VAL = ROOT / "pedagogical-move-selection" / "data" / "processed" / "mathdial" / "validation.jsonl"
ORIGINAL_SYNTHETIC = DOWNLOADS / "telling_calibration_synthetic_v1.jsonl"
ORIGINAL_CHALLENGE = DOWNLOADS / "telling_boundary_challenge_val_v1.jsonl"
ORIGINAL_MANIFEST = DOWNLOADS / "dataset_manifest.json"
FROZEN_DIAGNOSTIC = (
    ROOT / "pedagogical-move-selection" / "results" / "md7_telling_real_state_diagnostic_v1"
    / "telling_boundary_challenge.csv"
)
ABLATION_SUMMARY = (
    ROOT / "pedagogical-move-selection" / "results" / "md7_telling_epoch_ablation_v1" / "summary.json"
)

EXPECTED_HASHES = {
    "baseline_model": "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13",
    "mathdial_train": "2d0c9605d86d99c956b62466fcea6cfc0dbfed7d4ebfe2d39229689a5c10af76",
    "mathdial_validation": "225e6b8f671bf1e335af175343722aa9ac2a603a96d67bcb5220be1e92c0b2b6",
    "original_synthetic": "006a0c7015e5f9ff3166556d2b9291ce3a40dd73893e0a14c60a9f1d14bc4540",
    "original_challenge": "5d62f16b16ab2ee88f4f6b860f4a3480dcf844e7fb21e09bd4b2ef25f9698a61",
    "frozen_diagnostic": "c415a0207fb3f4b960606317fc2d32c19cacf73f477d4900649321654bee141e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized_text(text: str) -> str:
    text = text.casefold().replace("tutor:", "teacher:").replace("learner:", "student:")
    return re.sub(r"\s+", " ", text).strip()


def canonical_input(problem: str, history: str) -> tuple[str, str]:
    return normalized_text(problem), normalized_text(history)


def topic_payload(topic_index: int, variant: int) -> dict[str, str]:
    n = variant + 1
    if topic_index == 0:
        a, b = 4 + n, 23 + 2 * n
        return {"topic": "one_step_addition_equations", "problem": f"Solve x + {a} = {b}.",
                "wrong": f"x = {b + a}", "partial": f"x = {b} - {a}", "structure": "the added number must be undone",
                "microstep": f"subtracting {a} from {b}",
                "hint1": f"Use the inverse of adding {a}; apply the same operation to both sides.",
                "hint2": f"After subtracting {a}, the isolated equation is x = {b} - {a}. Evaluate that subtraction."}
    if topic_index == 1:
        a, b = 3 + n, 18 + 2 * n
        return {"topic": "one_step_subtraction_equations", "problem": f"Solve y - {a} = {b}.",
                "wrong": f"y = {b - a}", "partial": f"y = {b} + {a}", "structure": "subtracting must be undone by addition",
                "microstep": f"adding {a} to {b}",
                "hint1": f"To undo minus {a}, add {a} to both sides.",
                "hint2": f"That gives y = {b} + {a}. Now combine those two numbers."}
    if topic_index == 2:
        a, x = 3 + (n % 7), 6 + n
        b = a * x
        return {"topic": "one_step_multiplication_equations", "problem": f"Solve {a}m = {b}.",
                "wrong": f"m = {b - a}", "partial": f"m = {b} ÷ {a}", "structure": "the coefficient multiplies the variable",
                "microstep": f"dividing {b} by {a}",
                "hint1": f"The {a} is multiplying m, so use division on both sides.",
                "hint2": f"Write m = {b} ÷ {a}, then evaluate that quotient."}
    if topic_index == 3:
        a, x, b = 2 + (n % 6), 5 + n, 3 + (n % 8)
        c = a * x + b
        return {"topic": "two_step_equations", "problem": f"Solve {a}x + {b} = {c}.",
                "wrong": f"x = {c - b}", "partial": f"{a}x = {c - b}", "structure": "the constant is removed before the coefficient",
                "microstep": f"dividing {c - b} by {a} after isolating {a}x",
                "hint1": f"First subtract {b} from both sides to isolate the term {a}x.",
                "hint2": f"Now {a}x = {c - b}; divide both sides by {a}."}
    if topic_index == 4:
        d, a, b = 8 + n, 2 + (n % 4), 3 + (n % 5)
        return {"topic": "like_denominator_fractions", "problem": f"Add {a}/{d} + {b}/{d}.",
                "wrong": f"{a + b}/{2 * d}", "partial": f"({a}+{b})/{d}", "structure": "equal-size parts keep the same denominator",
                "microstep": "adding only the numerators while keeping the denominator",
                "hint1": f"Both fractions already name {d}ths, so the size of each part does not change.",
                "hint2": f"Keep denominator {d} and add the numerators {a} and {b}."}
    if topic_index == 5:
        a, b = 2 + (n % 4), 5 + (n % 6)
        return {"topic": "unlike_denominator_fractions", "problem": f"Add 1/{a} + 1/{b} without using a calculator.",
                "wrong": f"2/{a + b}", "partial": f"I need a common denominator related to {a} and {b}", "structure": "the pieces must be renamed to a common size",
                "microstep": "finding and using a common denominator",
                "hint1": f"Before adding, rename both fractions with one denominator divisible by {a} and {b}.",
                "hint2": f"Use {a * b} as a common denominator, rewrite each numerator, and then add."}
    if topic_index == 6:
        a = 10 + n / 10
        b = 0.25 + n / 100
        return {"topic": "decimal_addition", "problem": f"Calculate {a:.1f} + {b:.2f}.",
                "wrong": f"{a + b * 10:.2f}", "partial": f"{a:.2f} + {b:.2f}", "structure": "decimal places must line up by value",
                "microstep": "aligning tenths with tenths and hundredths with hundredths",
                "hint1": "Write both numbers with two decimal places and align the decimal points.",
                "hint2": f"Rewrite {a:.1f} as {a:.2f}, place {b:.2f} below it, then add each column."}
    if topic_index == 7:
        pct, amount = 10 + 5 * (n % 9), 80 + 4 * n
        return {"topic": "percent_of", "problem": f"Find {pct}% of {amount}.",
                "wrong": f"{amount + pct}", "partial": f"({pct}/100) × {amount}", "structure": "percent means per hundred",
                "microstep": f"multiplying {amount} by {pct}/100",
                "hint1": f"Rewrite {pct}% as the fraction {pct}/100 before using 'of'.",
                "hint2": f"Set up ({pct}/100) × {amount}, simplify, and compute the product."}
    if topic_index == 8:
        a, b, scale = 2 + (n % 5), 3 + (n % 7), 4 + n
        return {"topic": "equivalent_ratios", "problem": f"The ratio of red to blue tiles is {a}:{b}. If there are {a * scale} red tiles, how many blue tiles are there?",
                "wrong": f"{b + scale}", "partial": f"The scale factor is {scale}", "structure": "both ratio parts use the same scale factor",
                "microstep": f"multiplying {b} by the scale factor {scale}",
                "hint1": f"Compare {a} with {a * scale} to identify the multiplicative scale factor.",
                "hint2": f"The scale factor is {scale}; apply it to the blue part {b}."}
    if topic_index == 9:
        values = [n + 4, n + 8, n + 12, n + 16]
        total = sum(values)
        return {"topic": "arithmetic_mean", "problem": f"Find the mean of {values[0]}, {values[1]}, {values[2]}, and {values[3]}.",
                "wrong": str(total), "partial": f"The total is {total}", "structure": "a mean divides the total by the number of values",
                "microstep": f"dividing the total {total} by 4",
                "hint1": "After adding the values, count how many values contributed to the total.",
                "hint2": f"There are 4 values, so divide the total {total} by 4."}
    if topic_index == 10:
        length, width = 8 + n, 3 + (n % 9)
        return {"topic": "rectangle_area", "problem": f"A rectangle is {length} cm long and {width} cm wide. Find its area.",
                "wrong": f"{2 * (length + width)} cm²", "partial": f"{length} × {width}", "structure": "area counts rows times columns",
                "microstep": f"multiplying {length} by {width}",
                "hint1": "Area uses length times width, not the distance around the rectangle.",
                "hint2": f"Use A = {length} × {width} and calculate the product."}
    if topic_index == 11:
        length, width = 9 + n, 4 + (n % 8)
        return {"topic": "rectangle_perimeter", "problem": f"A rectangle is {length} m by {width} m. Find its perimeter.",
                "wrong": f"{length * width} m", "partial": f"{length}+{width}+{length}+{width}", "structure": "perimeter adds all four side lengths",
                "microstep": "adding two lengths and two widths",
                "hint1": "Trace the boundary and include both copies of each side length.",
                "hint2": f"Write {length} + {width} + {length} + {width}, then add."}
    if topic_index == 12:
        base, height = 10 + 2 * n, 5 + (n % 10)
        return {"topic": "triangle_area", "problem": f"A triangle has base {base} cm and perpendicular height {height} cm. Find its area.",
                "wrong": f"{base * height} cm²", "partial": f"({base} × {height}) ÷ 2", "structure": "a triangle is half a matching rectangle",
                "microstep": "halving the base-times-height product",
                "hint1": "Compare the triangle with a rectangle having the same base and height.",
                "hint2": f"Use A = ({base} × {height}) ÷ 2 and evaluate it."}
    if topic_index == 13:
        k = n + 1
        a, b, c = 3 * k, 4 * k, 5 * k
        return {"topic": "pythagorean_theorem", "problem": f"A right triangle has legs {a} and {b}. Find the hypotenuse.",
                "wrong": str(a + b), "partial": f"c² = {a}² + {b}²", "structure": "the side lengths enter the theorem as squares",
                "microstep": "adding the squared legs and then taking the square root",
                "hint1": "Use a² + b² = c² rather than adding the two leg lengths directly.",
                "hint2": f"Compute {a}² + {b}², then take the positive square root to obtain c."}
    if topic_index == 14:
        p, q = 2 + (n % 6), 3 + n
        return {"topic": "product_of_powers", "problem": f"Simplify z^{p} · z^{q}.",
                "wrong": f"z^{p * q}", "partial": f"z^({p}+{q})", "structure": "like bases multiply by adding exponents",
                "microstep": f"adding exponents {p} and {q}",
                "hint1": "The base stays z; count the total number of z factors.",
                "hint2": f"Use z^({p}+{q}) and simplify the exponent."}
    if topic_index == 15:
        a, b, c = 2 + (n % 8), 3 + n, 1 + (n % 9)
        return {"topic": "distributive_property", "problem": f"Expand {a}({b}x + {c}).",
                "wrong": f"{a * b}x + {c}", "partial": f"({a}×{b})x + ({a}×{c})", "structure": "the outside factor multiplies every term",
                "microstep": f"multiplying {a} by both terms inside the parentheses",
                "hint1": f"Draw one arrow from {a} to {b}x and another from {a} to {c}.",
                "hint2": f"Compute ({a}×{b})x and ({a}×{c}), then write their sum."}
    if topic_index == 16:
        a, b = 4 + n, 2 + (n % 9)
        return {"topic": "combining_like_terms", "problem": f"Simplify {a}x + {b}x.",
                "wrong": f"{a + b}x²", "partial": f"({a}+{b})x", "structure": "the variable part stays x when coefficients are added",
                "microstep": f"adding coefficients {a} and {b} without changing x",
                "hint1": "Treat x as the common unit and add only how many x terms there are.",
                "hint2": f"Use ({a}+{b})x, add the coefficients, and keep x to the first power."}
    if topic_index == 17:
        a, b = 3 + n, 2 + (n % 11)
        return {"topic": "integer_multiplication", "problem": f"Calculate (-{a}) × (-{b}).",
                "wrong": str(-(a * b)), "partial": f"The magnitude is {a * b}", "structure": "two negative factors give a positive product",
                "microstep": "applying the sign rule after multiplying the magnitudes",
                "hint1": "Separate the sign decision from the multiplication of the magnitudes.",
                "hint2": f"The signs match, so the product is positive; now use {a} × {b}."}
    if topic_index == 18:
        a, b, c = 2 + (n % 7), 3 + n, 2 + (n % 5)
        return {"topic": "order_of_operations", "problem": f"Evaluate {a} + {b} × {c}.",
                "wrong": str((a + b) * c), "partial": f"{a} + {b * c}", "structure": "multiplication is completed before addition",
                "microstep": f"computing {b} × {c} before adding {a}",
                "hint1": "Identify the multiplication operation and complete it before the addition.",
                "hint2": f"Replace {b} × {c} with {b * c}, then add {a}."}
    if topic_index == 19:
        red, blue = 2 + (n % 9), 7 + n
        total = red + blue
        return {"topic": "simple_probability", "problem": f"A bag has {red} red counters and {blue} blue counters. What is the probability of drawing red?",
                "wrong": f"{red}/{blue}", "partial": f"the total is {total}, but I have not formed the probability", "structure": "probability uses favorable outcomes over all outcomes",
                "microstep": f"using the total {total} as the denominator",
                "hint1": "The denominator counts every counter that could be drawn, not only the other color.",
                "hint2": f"There are {total} counters altogether, with {red} favorable outcomes."}
    if topic_index == 20:
        x1, x2, y1, rise = n, n + 4, 2 * n + 1, 8 + (n % 5)
        y2 = y1 + rise
        return {"topic": "slope", "problem": f"Find the slope through ({x1}, {y1}) and ({x2}, {y2}).",
                "wrong": f"4/{rise}", "partial": f"({y2}-{y1})/({x2}-{x1})", "structure": "slope is vertical change divided by horizontal change",
                "microstep": "placing rise over run in the correct order",
                "hint1": f"Write the change in y over the change in x as ({y2}-{y1})/({x2}-{x1}), keeping the same point order.",
                "hint2": f"Use ({y2}-{y1})/({x2}-{x1}), simplify each difference, then divide."}
    if topic_index == 21:
        a, b, c = 2 + (n % 7), 5 + (n % 9), 3 + n
        return {"topic": "proportions", "problem": f"Solve the proportion {a}/{b} = x/{c}.",
                "wrong": f"x = {a + c - b}", "partial": f"{b}x = {a * c}", "structure": "cross products form an equation",
                "microstep": f"dividing the cross product {a * c} by {b}",
                "hint1": f"Set the cross products equal: {a}×{c} = {b}×x.",
                "hint2": f"This gives {b}x = {a * c}; divide by {b}."}
    if topic_index == 22:
        d, num = 20 + 2 * n, 3 + (n % 8)
        return {"topic": "fraction_to_decimal", "problem": f"Write {num}/{d} as a decimal.",
                "wrong": f"0.{num}{d}", "partial": f"{num} ÷ {d}", "structure": "the fraction bar represents division",
                "microstep": f"dividing {num} by {d} with place value",
                "hint1": f"Interpret {num}/{d} as the division {num} ÷ {d}.",
                "hint2": f"Add a decimal point and zeros to {num}, then carry out {num} ÷ {d}."}
    if topic_index == 23:
        angle = 35 + n
        return {"topic": "supplementary_angles", "problem": f"Two angles are supplementary. One angle is {angle}°. Find the other angle.",
                "wrong": f"{180 + angle}°", "partial": f"180° - {angle}°", "structure": "supplementary angles total 180 degrees",
                "microstep": f"subtracting {angle} from 180",
                "hint1": "Supplementary angles combine to make a straight angle of 180°.",
                "hint2": f"Write the missing angle as 180° - {angle}° and evaluate it."}
    raise AssertionError(topic_index)


OPENERS = [
    "Try the problem and show the step where you decide what operation to use.",
    "Work through this problem and write your first calculation.",
    "Start with the smallest useful step and show what you obtain.",
    "Give this problem a try and include the operation you chose.",
    "Attempt the calculation and explain the setup briefly.",
]


def focus_response(payload: dict[str, str], variant: int) -> tuple[str, str]:
    options = [
        (f"I still get {payload['wrong']}.", "one_scaffold_same_error"),
        (f"I used that clue, but I am still ending at {payload['wrong']}.", "one_scaffold_persistent_once"),
        (f"I see that {payload['structure']}, but I do not know how to finish the calculation.", "one_scaffold_incomplete_progress"),
        (f"I tried the suggested step. My current work is {payload['partial']}, but I am not sure what comes next.", "one_scaffold_partial_structure"),
        (f"I still don't get this part; I keep returning to {payload['wrong']}.", "one_scaffold_local_confusion"),
        (f"I can name the operation, but when I apply it I get {payload['wrong']}.", "one_scaffold_incorrect_execution"),
    ]
    return options[variant % len(options)]


def telling_response(payload: dict[str, str], variant: int) -> tuple[str, str]:
    options = [
        (f"I still get {payload['wrong']}; the same step is blocking me.", "two_scaffolds_same_error"),
        (f"I tried both clues and still cannot carry out {payload['microstep']}.", "two_scaffolds_microstep_failure"),
        (f"I understand the words, but I still do not know how to do {payload['microstep']}.", "two_scaffolds_repeated_confusion"),
        (f"I repeated it and arrived at {payload['wrong']} again.", "two_scaffolds_same_misconception"),
        ("I cannot do this narrowed step even after those two clues.", "two_scaffolds_exhausted_narrow_step"),
        (f"Could you explain {payload['microstep']} directly now?", "direct_explanation_request_after_two_scaffolds"),
    ]
    return options[variant % len(options)]


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_number = 0
    for topic_index in range(24):
        for variant in range(50):
            payload = topic_payload(topic_index, variant)
            family_id = f"md7r2c1:pair:{family_number:04d}"
            opening = OPENERS[(topic_index + variant) % len(OPENERS)]
            focus_latest, focus_tag = focus_response(payload, variant)
            telling_latest, telling_tag = telling_response(payload, variant)
            common = [
                f"Tutor: {opening}",
                f"Student: {payload['wrong']}",
                f"Tutor: {payload['hint1']}",
                f"Student: {focus_latest}",
            ]
            focus_history = "\n".join(common)
            telling_history = "\n".join(common + [f"Tutor: {payload['hint2']}", f"Student: {telling_latest}"])
            base = {
                "schema_version": SCHEMA_VERSION,
                "family_id": family_id,
                "split": "train",
                "source": "md7_r2_tell_c1_deterministic_matched_synthetic",
                "is_real_conversation": False,
                "problem": payload["problem"],
                "topic": payload["topic"],
                "generation_seed": SEED,
            }
            rows.append({
                **base,
                "example_id": f"md7r2c1:focus:{family_number:04d}",
                "history": focus_history,
                "latest_student_text": focus_latest,
                "target_move": "focus",
                "label": 2,
                "boundary_type": "recoverable_after_exactly_one_meaningful_scaffold",
                "rationale_tag": focus_tag,
                "meaningful_scaffold_count": 1,
                "matched_pair_role": "one_scaffold_hard_negative",
            })
            rows.append({
                **base,
                "example_id": f"md7r2c1:telling:{family_number:04d}",
                "history": telling_history,
                "latest_student_text": telling_latest,
                "target_move": "telling",
                "label": 3,
                "boundary_type": "persistent_after_two_meaningful_scaffolds",
                "rationale_tag": telling_tag,
                "meaningful_scaffold_count": 2,
                "matched_pair_role": "two_scaffold_matched_positive",
            })
            family_number += 1
    assert family_number == 1200
    random.Random(SEED).shuffle(rows)
    return rows


def validate_structure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "example_id", "family_id", "split", "source", "is_real_conversation",
        "problem", "history", "latest_student_text", "target_move", "label", "boundary_type",
        "rationale_tag", "topic", "generation_seed", "meaningful_scaffold_count", "matched_pair_role",
    }
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direct_request_re = re.compile(r"\b(just give|give me the answer|just tell|answer for me|explain .* directly)\b", re.I)
    artificial_token_re = re.compile(r"\b(generic|probing|focus|telling|target_move|class label)\b", re.I)
    for row in rows:
        assert set(row) == expected_keys
        assert row["split"] == "train" and row["is_real_conversation"] is False
        assert row["label"] == {"focus": 2, "telling": 3}[row["target_move"]]
        lines = row["history"].splitlines()
        assert all(lines[index].startswith("Tutor:" if index % 2 == 0 else "Student:") for index in range(len(lines)))
        assert lines[-1] == f"Student: {row['latest_student_text']}"
        if row["target_move"] == "focus":
            assert len(lines) == 4 and row["meaningful_scaffold_count"] == 1
            assert direct_request_re.search(row["latest_student_text"]) is None
        else:
            assert len(lines) == 6 and row["meaningful_scaffold_count"] == 2
            assert lines[2] != lines[4]
        assert artificial_token_re.search(row["problem"] + "\n" + row["history"]) is None
        family_rows[row["family_id"]].append(row)
    assert len(family_rows) == 1200
    assert all(len(items) == 2 and {item["target_move"] for item in items} == {"focus", "telling"}
               and len({item["problem"] for item in items}) == 1 for items in family_rows.values())
    return {
        "family_count": len(family_rows),
        "all_families_exactly_one_focus_one_telling": True,
        "focus_rows_with_exactly_one_scaffold": sum(r["target_move"] == "focus" and r["meaningful_scaffold_count"] == 1 for r in rows),
        "telling_rows_with_at_least_two_scaffolds": sum(r["target_move"] == "telling" and r["meaningful_scaffold_count"] >= 2 for r in rows),
        "focus_rows_with_direct_answer_request": sum(r["target_move"] == "focus" and direct_request_re.search(r["latest_student_text"]) is not None for r in rows),
        "rows_with_artificial_label_tokens_in_model_text": sum(artificial_token_re.search(r["problem"] + "\n" + r["history"]) is not None for r in rows),
    }


def notebook(correction_hash: str, packaged_hashes: dict[str, str], mixture: dict[str, Any]) -> dict[str, Any]:
    def md(text: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text: str) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

    cells = [
        md("""# MD7-R2-TELL-C1 — targeted telling-boundary recalibration

This Kaggle-GPU notebook starts from frozen **MD7-R1 epoch 3** and trains three offline candidates. It corrects one narrow error: one failed meaningful scaffold remains `focus`, while persistent difficulty after multiple scaffolds may become `telling`.

Scientific constraints: ordinary four-class cross-entropy only; no weighting, RL, BKT, MRB1, LinTS, propensity adjustment, automatic epoch selection, production promotion, MathDial test, or final synthetic test. Stop after the epoch comparison table and evaluate all exported epochs with the separately frozen 48-case real-state diagnostic.
"""),
        code("""# 1 — imports and exact experiment constants
import os, json, math, random, hashlib, zipfile, gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback

SEED = 42
MAX_LENGTH = 512
LABELS = ["generic", "probing", "focus", "telling"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator."
GPU_COUNT = torch.cuda.device_count()
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 16 // (PER_DEVICE_BATCH * GPU_COUNT)
assert GRAD_ACCUM >= 1 and PER_DEVICE_BATCH * GPU_COUNT * GRAD_ACCUM == 16
OUTPUT_DIR = Path("/kaggle/working/md7_r2_tell_c1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("GPU count:", GPU_COUNT, "effective global batch:", PER_DEVICE_BATCH * GPU_COUNT * GRAD_ACCUM)
"""),
        code(f"""# 2 — discover and verify the three mounted Kaggle inputs
INPUT_ROOT = Path("/kaggle/input")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

manifests = list(INPUT_ROOT.rglob("dataset_manifest.json"))
matches = []
for path in manifests:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("experiment") == "MD7-R2-TELL-C1": matches.append((path, obj))
    except Exception:
        pass
assert len(matches) == 1, f"Expected exactly one MD7-R2-TELL-C1 manifest, found {{len(matches)}}"
MANIFEST_PATH, MANIFEST = matches[0]
PACKAGE = MANIFEST_PATH.parent
CORRECTION = PACKAGE / "correction_train.jsonl"
ORIGINAL_TRAIN = PACKAGE / "telling_calibration_train_v1.jsonl"
ORIGINAL_VAL = PACKAGE / "telling_calibration_validation_v1.jsonl"
ORIGINAL_CHALLENGE = PACKAGE / "telling_boundary_challenge_val_v1.jsonl"
package_relpaths = (
    "correction_train.jsonl",
    "dataset_manifest.json",
    "dataset_audit.json",
    "telling_calibration_train_v1.jsonl",
    "telling_calibration_validation_v1.jsonl",
    "telling_boundary_challenge_val_v1.jsonl",
)
for rel in package_relpaths:
    path = PACKAGE / rel
    assert path.is_file(), path
    expected = MANIFEST["kaggle_file_hashes"].get(rel)
    if expected is not None:
        observed = sha256(path)
        assert observed == expected, (rel, observed, expected)
assert not (PACKAGE / "test.jsonl").exists(), "A protected test file was placed in the preparation dataset root."

# Resolve MathDial TRAIN and VALIDATION only. A separately mounted dataset may
# contain test.jsonl; this notebook never opens or hashes that file.
expected_md_train = MANIFEST["kaggle_file_hashes"]["mathdial/train.jsonl"]
expected_md_val = MANIFEST["kaggle_file_hashes"]["mathdial/validation.jsonl"]
local_md_train = PACKAGE / "mathdial" / "train.jsonl"
local_md_val = PACKAGE / "mathdial" / "validation.jsonl"
if (local_md_train.is_file() and local_md_val.is_file()
        and sha256(local_md_train) == expected_md_train
        and sha256(local_md_val) == expected_md_val):
    MATHDIAL_TRAIN, MATHDIAL_VAL = local_md_train, local_md_val
else:
    md_pairs = []
    for train_path in INPUT_ROOT.rglob("train.jsonl"):
        val_path = train_path.parent / "validation.jsonl"
        if not val_path.is_file():
            continue
        lower = str(train_path.parent).lower()
        score = (100 if "mathdial" in lower else 0) + (10 if "move-selection" in lower else 0)
        md_pairs.append((score, train_path, val_path))
    md_pairs.sort(key=lambda x: (-x[0], str(x[1])))
    selected_md = None
    for _, train_path, val_path in md_pairs:
        if sha256(train_path) == expected_md_train and sha256(val_path) == expected_md_val:
            selected_md = (train_path, val_path)
            break
    assert selected_md is not None, (
        "Could not find the exact authorized MathDial train.jsonl and validation.jsonl under /kaggle/input. "
        "Attach the mathdial-move-selection dataset."
    )
    MATHDIAL_TRAIN, MATHDIAL_VAL = selected_md

# Resolve the complete frozen MD7-R1 Hugging Face export. Notebook-output
# mounts are supported; loose epoch state_dict files are intentionally rejected.
expected_model = MANIFEST["kaggle_file_hashes"]["md7r1_epoch3/model.safetensors"]
required_model_hashes = (
    ("config.json", MANIFEST["kaggle_file_hashes"]["md7r1_epoch3/config.json"]),
    ("tokenizer.json", MANIFEST["kaggle_file_hashes"]["md7r1_epoch3/tokenizer.json"]),
    ("tokenizer_config.json", MANIFEST["kaggle_file_hashes"]["md7r1_epoch3/tokenizer_config.json"]),
)

def model_dir_matches(path):
    weight = path / "model.safetensors"
    if not weight.is_file() or sha256(weight) != expected_model:
        return False
    for name, expected in required_model_hashes:
        file_path = path / name
        if not file_path.is_file() or sha256(file_path) != expected:
            return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        labels = [config["id2label"][str(i)] for i in range(4)]
    except Exception:
        return False
    return labels == ID2LABEL

local_model = PACKAGE / "md7r1_epoch3"
if local_model.is_dir() and model_dir_matches(local_model):
    MODEL_DIR = local_model
else:
    model_candidates = []
    for config_path in INPUT_ROOT.rglob("config.json"):
        model_path = config_path.parent
        if not (model_path / "model.safetensors").is_file():
            continue
        lower = str(model_path).lower()
        score = 0
        score += 200 if "md7r1_epoch3" in lower else 0
        score += 80 if "md7r1" in lower else 0
        score += 40 if "epoch3" in lower else 0
        score += 20 if "results" in lower else 0
        score -= 200 if any(x in lower for x in ("md6", "mrb", "telling_calibration", "epoch1", "epoch2")) else 0
        model_candidates.append((score, model_path))
    model_candidates.sort(key=lambda x: (-x[0], str(x[1])))
    MODEL_DIR = None
    for _, model_path in model_candidates:
        if model_dir_matches(model_path):
            MODEL_DIR = model_path
            break
    assert MODEL_DIR is not None, (
        "Could not find the exact complete MD7-R1 epoch3 model export under /kaggle/input. "
        "Attach the MD7-R1 AdaptMath notebook output containing config/tokenizer files and model.safetensors."
    )

assert sha256(CORRECTION) == "{correction_hash}"
assert MANIFEST["guardrails"]["mathdial_final_test_use"] == 0
assert MANIFEST["guardrails"]["mrbench_v3_test_use"] == 0
assert MANIFEST["guardrails"]["production_changes"] == 0
print("Preparation package:", PACKAGE)
print("MathDial train:", MATHDIAL_TRAIN)
print("MathDial validation:", MATHDIAL_VAL)
print("Frozen MD7-R1 model:", MODEL_DIR)
print("INPUT HASH GUARD: PASS")
"""),
        code("""# 3 — load only authorized training/validation inputs
def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

correction = read_jsonl(CORRECTION)
original_train = read_jsonl(ORIGINAL_TRAIN)
original_val = read_jsonl(ORIGINAL_VAL)
challenge_val = read_jsonl(ORIGINAL_CHALLENGE)
mathdial_train = read_jsonl(MATHDIAL_TRAIN)
mathdial_val = read_jsonl(MATHDIAL_VAL)

assert len(correction) == 2400 and Counter(r["target_move"] for r in correction) == {"focus": 1200, "telling": 1200}
assert len(original_train) == 6000 and Counter(r["target_move"] for r in original_train) == {x: 1500 for x in LABELS}
assert len(original_val) == 2400 and Counter(r["target_move"] for r in original_val) == {x: 600 for x in LABELS}
assert len(challenge_val) == 800
assert len(mathdial_train) == 14905 and len(mathdial_val) == 1850
print("AUTHORIZED INPUT COUNTS: PASS")
"""),
        code("""# 4 — exact 70/30 mixture using MathDial TRAIN only
def norm_synthetic(rows, source):
    return [{"example_id": str(r["example_id"]), "problem": r["problem"], "history": r["history"],
             "target_move": r["target_move"], "source": source, "history_format": "preformatted_string"} for r in rows]
def norm_mathdial(rows, source):
    return [{"example_id": str(r["example_id"]), "problem": r["problem"], "history": r["history"],
             "target_move": r["target_move"], "source": source, "history_format": "mathdial_turn_list"} for r in rows]
def proportional_stratified_sample(rows, total_n, seed):
    rng = random.Random(seed)
    by_label = {label: [r for r in rows if r["target_move"] == label] for label in LABELS}
    counts = {label: len(by_label[label]) for label in LABELS}; total = sum(counts.values())
    raw = {label: total_n * counts[label] / total for label in LABELS}
    quota = {label: math.floor(raw[label]) for label in LABELS}
    order = sorted(LABELS, key=lambda label: (raw[label]-quota[label], label), reverse=True)
    for label in order[:total_n-sum(quota.values())]: quota[label] += 1
    sample = []
    for label in LABELS: sample.extend(rng.sample(by_label[label], quota[label]))
    rng.shuffle(sample); return sample, quota

original_rows = norm_synthetic(original_train, "original_telling_calibration_train")
correction_rows = norm_synthetic(correction, "md7_r2_tell_c1_correction_train")
md_train_rows = norm_mathdial(mathdial_train, "mathdial_train_replay")
md_val_rows = norm_mathdial(mathdial_val, "mathdial_validation")
md_replay, replay_quota = proportional_stratified_sample(md_train_rows, 3600, SEED)
train_rows = original_rows + correction_rows + md_replay
random.Random(SEED).shuffle(train_rows)
assert len(train_rows) == 12000 and replay_quota == {"generic":864,"probing":824,"focus":1315,"telling":597}
assert Counter(r["source"] for r in train_rows) == {"original_telling_calibration_train":6000,"md7_r2_tell_c1_correction_train":2400,"mathdial_train_replay":3600}
print("TRAINING MIXTURE:", Counter(r["source"] for r in train_rows), Counter(r["target_move"] for r in train_rows), replay_quota)
"""),
        code("""# 5 — exact deployed MD7-R1 paired preprocessing
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
tokenizer.truncation_side = "left"
def format_mathdial_history(history):
    if not history: return "[No previous conversation]"
    assert isinstance(history, list)
    return "\\n".join(f"{turn['user']}: {turn['text']}" for turn in history)
def normalized_history(row):
    if row["history_format"] == "mathdial_turn_list": return format_mathdial_history(row["history"])
    text = str(row["history"]); return text if text.strip() else "[No previous conversation]"
def paired_text(row):
    return "Problem:\\n" + str(row["problem"]), "Conversation:\\n" + normalized_history(row) + "\\n\\nNext teacher pedagogical move:"
def encode(row):
    first, second = paired_text(row)
    return tokenizer(first, second, truncation="only_second", max_length=MAX_LENGTH)
class MoveDataset(Dataset):
    def __init__(self, rows): self.rows = list(rows)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; item = encode(row); item["labels"] = LABEL2ID[row["target_move"]]; return item
collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_ds = MoveDataset(train_rows)
target_val_ds = MoveDataset(norm_synthetic(original_val, "original_telling_calibration_validation"))
challenge_val_ds = MoveDataset(norm_synthetic(challenge_val, "telling_boundary_challenge_validation"))
md_val_ds = MoveDataset(md_val_rows)
print(paired_text(md_val_rows[0])[0]); print(paired_text(md_val_rows[0])[1][:1000])
assert tokenizer.truncation_side == "left"
"""),
        code("""# 6 — evaluation helpers
def metrics_from_logits(logits, labels):
    pred = np.argmax(logits, axis=-1)
    per = f1_score(labels, pred, average=None, labels=[0,1,2,3], zero_division=0)
    return {"accuracy":float(accuracy_score(labels,pred)), "macro_f1":float(f1_score(labels,pred,average="macro",zero_division=0)),
            "per_class_f1":{LABELS[i]:float(per[i]) for i in range(4)},
            "pred_distribution":{LABELS[i]:int((pred==i).sum()) for i in range(4)}}
def challenge_metrics(logits, labels):
    out = metrics_from_logits(logits, labels); pred=np.argmax(logits,axis=-1); y=np.asarray(labels); tell=3
    tp=int(((pred==tell)&(y==tell)).sum()); fp=int(((pred==tell)&(y!=tell)).sum()); fn=int(((pred!=tell)&(y==tell)).sum()); non=int((y!=tell).sum())
    out.update({"telling_precision":tp/(tp+fp) if tp+fp else 0.0,"telling_recall":tp/(tp+fn) if tp+fn else 0.0,
                "telling_false_trigger_rate_on_non_telling":fp/non if non else 0.0,"telling_tp":tp,"telling_fp":fp,"telling_fn":fn})
    return out
@torch.no_grad()
def predict_dataset(model, dataset, batch_size=16):
    model.eval(); loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collator)
    logits=[]; labels=[]; device=next(model.parameters()).device
    for batch in loader:
        labels.append(batch.pop("labels").numpy()); batch={k:v.to(device) for k,v in batch.items()}; logits.append(model(**batch).logits.detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)
"""),
        code("""# 7 — REQUIRED PRETRAINING GUARD: frozen baseline on MathDial VALIDATION
config = json.loads((MODEL_DIR/"config.json").read_text(encoding="utf-8"))
assert {int(k):v for k,v in config["id2label"].items()} == ID2LABEL and config["label2id"] == LABEL2ID
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to("cuda")
md_logits, md_labels = predict_dataset(model, md_val_ds)
baseline_md = metrics_from_logits(md_logits, md_labels)
expected = {"accuracy":0.5162162162162162,"macro_f1":0.4611286390258053,
            "per_class_f1":{"generic":0.6105263157894737,"probing":0.3333333333333333,"focus":0.6004672897196262,"telling":0.300187617260788},
            "pred_distribution":{"generic":413,"probing":188,"focus":1034,"telling":215}}
assert abs(baseline_md["accuracy"]-expected["accuracy"]) <= 1e-12
assert abs(baseline_md["macro_f1"]-expected["macro_f1"]) <= 1e-12
assert all(abs(baseline_md["per_class_f1"][k]-v) <= 1e-12 for k,v in expected["per_class_f1"].items())
assert baseline_md["pred_distribution"] == expected["pred_distribution"]
(OUTPUT_DIR/"baseline_mathdial_validation.json").write_text(json.dumps(baseline_md,indent=2))
print("PRETRAINING GUARD: PASS", json.dumps(baseline_md,indent=2))
del model, md_logits; gc.collect(); torch.cuda.empty_cache()
"""),
        code("""# 8 — export and evaluate every epoch; no automatic selection
EPOCH_RESULTS=[]
class EpochAuditCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch=int(round(state.epoch)); model_=kwargs["model"]; path=OUTPUT_DIR/f"epoch{epoch}"; path.mkdir(parents=True,exist_ok=True)
        model_.save_pretrained(path); tokenizer.save_pretrained(path)
        row={"epoch":epoch}
        for name,ds,fn in [("original_synthetic_val",target_val_ds,metrics_from_logits),("telling_challenge_val",challenge_val_ds,challenge_metrics),("mathdial_val",md_val_ds,metrics_from_logits)]:
            logits,labels=predict_dataset(model_,ds); row[name]=fn(logits,labels)
        EPOCH_RESULTS.append(row); (OUTPUT_DIR/f"epoch{epoch}_metrics.json").write_text(json.dumps(row,indent=2)); print(json.dumps(row,indent=2))
        return control
"""),
        code("""# 9 — TRAIN ON KAGGLE GPU ONLY: ordinary four-class cross-entropy
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
args=TrainingArguments(output_dir=str(OUTPUT_DIR/"trainer_tmp"),learning_rate=1e-5,num_train_epochs=3,
    per_device_train_batch_size=PER_DEVICE_BATCH,per_device_eval_batch_size=16,gradient_accumulation_steps=GRAD_ACCUM,
    weight_decay=0.01,warmup_ratio=0.10,logging_strategy="epoch",save_strategy="no",load_best_model_at_end=False,
    fp16=True,report_to="none",seed=SEED,data_seed=SEED)
trainer=Trainer(model=model,args=args,train_dataset=train_ds,data_collator=collator,callbacks=[EpochAuditCallback()])
print("INITIALIZATION:", MODEL_DIR); print("ROWS:",len(train_rows),"LOSS: ordinary cross-entropy")
trainer.train()
"""),
        code("""# 10 — validation-only epoch table; DO NOT select or promote here
rows=[]
for item in EPOCH_RESULTS:
    rows.append({"epoch":item["epoch"],"mathdial_accuracy":item["mathdial_val"]["accuracy"],"mathdial_macro_f1":item["mathdial_val"]["macro_f1"],
      **{f"mathdial_f1_{k}":v for k,v in item["mathdial_val"]["per_class_f1"].items()},
      **{f"mathdial_pred_{k}":v for k,v in item["mathdial_val"]["pred_distribution"].items()},
      "synthetic_val_macro_f1":item["original_synthetic_val"]["macro_f1"],"challenge_telling_precision":item["telling_challenge_val"]["telling_precision"],
      "challenge_telling_recall":item["telling_challenge_val"]["telling_recall"],"challenge_false_telling_rate":item["telling_challenge_val"]["telling_false_trigger_rate_on_non_telling"]})
comparison=pd.DataFrame(rows); display(comparison); comparison.to_csv(OUTPUT_DIR/"epoch_comparison.csv",index=False)
assert sorted(comparison.epoch.tolist()) == [1,2,3]
print("STOP: export all epochs. Do not select automatically and do not evaluate a final synthetic test.")
"""),
        code("""# 11 — package all three offline candidates
zip_path=Path("/kaggle/working/md7_r2_tell_c1_epoch_candidates.zip")
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file() and "trainer_tmp" not in str(path): archive.write(path,path.relative_to(OUTPUT_DIR.parent))
print("Download:",zip_path)
"""),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        BASELINE / "model.safetensors", BASELINE / "config.json", BASELINE / "tokenizer.json",
        BASELINE / "tokenizer_config.json", MATHDIAL_TRAIN, MATHDIAL_VAL, ORIGINAL_SYNTHETIC,
        ORIGINAL_CHALLENGE, ORIGINAL_MANIFEST, FROZEN_DIAGNOSTIC, ABLATION_SUMMARY,
    ]
    before = {str(path): sha256(path) for path in sources}
    assert before[str(BASELINE / "model.safetensors")] == EXPECTED_HASHES["baseline_model"]
    assert before[str(MATHDIAL_TRAIN)] == EXPECTED_HASHES["mathdial_train"]
    assert before[str(MATHDIAL_VAL)] == EXPECTED_HASHES["mathdial_validation"]
    assert before[str(ORIGINAL_SYNTHETIC)] == EXPECTED_HASHES["original_synthetic"]
    assert before[str(ORIGINAL_CHALLENGE)] == EXPECTED_HASHES["original_challenge"]
    assert before[str(FROZEN_DIAGNOSTIC)] == EXPECTED_HASHES["frozen_diagnostic"]

    # Generate independently; diagnostic cases enter only the subsequent overlap audit.
    correction = make_rows()
    structure = validate_structure(correction)
    assert len(correction) == 2400
    assert Counter(row["target_move"] for row in correction) == {"focus": 1200, "telling": 1200}
    assert len({row["example_id"] for row in correction}) == 2400
    assert len({canonical_input(row["problem"], row["history"]) for row in correction}) == 2400
    correction_path = OUT / "correction_train.jsonl"
    write_jsonl(correction_path, correction)

    original = read_jsonl(ORIGINAL_SYNTHETIC)
    challenge = read_jsonl(ORIGINAL_CHALLENGE)
    assert len(original) == 9600 and Counter(row["split"] for row in original) == {"train": 6000, "val": 2400, "test": 1200}
    original_train = [row for row in original if row["split"] == "train"]
    original_val = [row for row in original if row["split"] == "val"]
    original_test = [row for row in original if row["split"] == "test"]
    write_jsonl(OUT / "telling_calibration_train_v1.jsonl", original_train)
    write_jsonl(OUT / "telling_calibration_validation_v1.jsonl", original_val)
    write_jsonl(OUT / "telling_boundary_challenge_val_v1.jsonl", challenge)

    import pandas as pd
    diagnostic = pd.read_csv(FROZEN_DIAGNOSTIC)
    diagnostic_histories = set()
    diagnostic_inputs = set()
    for row in diagnostic.itertuples(index=False):
        turns = json.loads(row.history)
        history = "\n".join(f"{turn['user']}: {turn['text']}" for turn in turns) if turns else "[No previous conversation]"
        diagnostic_histories.add(normalized_text(history))
        diagnostic_inputs.add(canonical_input(str(row.problem), history))

    generated_inputs = {canonical_input(row["problem"], row["history"]) for row in correction}
    generated_histories = {normalized_text(row["history"]) for row in correction}
    original_all_inputs = {canonical_input(row["problem"], row["history"]) for row in original + challenge}
    existing_val_test_inputs = {canonical_input(row["problem"], row["history"]) for row in original_val + original_test + challenge}
    existing_val_test_histories = {normalized_text(row["history"]) for row in original_val + original_test + challenge}

    audit = {
        "experiment": "MD7-R2-TELL-C1",
        "status": "PASS",
        "generation_seed": SEED,
        "correction_rows": len(correction),
        "target_counts": dict(Counter(row["target_move"] for row in correction)),
        "split_counts": dict(Counter(row["split"] for row in correction)),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in correction).items())),
        "rationale_tag_counts": dict(sorted(Counter(row["rationale_tag"] for row in correction).items())),
        "boundary_type_counts": dict(Counter(row["boundary_type"] for row in correction)),
        "structure": structure,
        "exact_problem_history_duplicate_count": len(correction) - len(generated_inputs),
        "exact_history_duplicate_count": len(correction) - len(generated_histories),
        "overlap": {
            "frozen_48_case_diagnostic_problem_history": len(generated_inputs & diagnostic_inputs),
            "frozen_48_case_diagnostic_history": len(generated_histories & diagnostic_histories),
            "original_synthetic_all_plus_challenge_problem_history": len(generated_inputs & original_all_inputs),
            "existing_synthetic_validation_test_challenge_problem_history": len(generated_inputs & existing_val_test_inputs),
            "existing_synthetic_validation_test_challenge_history": len(generated_histories & existing_val_test_histories),
        },
        "privacy": {"learner_identifiers_present": 0, "is_real_conversation_true": 0},
        "label_provenance": {"bkt_derived": False, "mrb1_derived": False, "manual_semantic_contract": True},
        "protected_data": {
            "mathdial_final_test_read": False,
            "mrbench_v3_test_read": False,
            "original_synthetic_final_test_used_for_overlap_audit_only": len(original_test),
            "original_synthetic_final_test_packaged_or_used_for_training_or_selection": False,
        },
    }
    assert audit["exact_problem_history_duplicate_count"] == 0
    assert audit["exact_history_duplicate_count"] == 0
    assert audit["overlap"] == {key: 0 for key in audit["overlap"]}
    assert structure["focus_rows_with_exactly_one_scaffold"] == 1200
    assert structure["telling_rows_with_at_least_two_scaffolds"] == 1200
    assert structure["focus_rows_with_direct_answer_request"] == 0
    assert structure["rows_with_artificial_label_tokens_in_model_text"] == 0

    (OUT / "mathdial").mkdir(exist_ok=True)
    shutil.copyfile(MATHDIAL_TRAIN, OUT / "mathdial" / "train.jsonl")
    shutil.copyfile(MATHDIAL_VAL, OUT / "mathdial" / "validation.jsonl")
    model_out = OUT / "md7r1_epoch3"
    model_out.mkdir(exist_ok=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(BASELINE / name, model_out / name)

    correction_hash = sha256(correction_path)
    packaged_hashes = {
        "correction_train.jsonl": correction_hash,
        "telling_calibration_train_v1.jsonl": sha256(OUT / "telling_calibration_train_v1.jsonl"),
        "telling_calibration_validation_v1.jsonl": sha256(OUT / "telling_calibration_validation_v1.jsonl"),
        "telling_boundary_challenge_val_v1.jsonl": sha256(OUT / "telling_boundary_challenge_val_v1.jsonl"),
        "mathdial/train.jsonl": sha256(OUT / "mathdial" / "train.jsonl"),
        "mathdial/validation.jsonl": sha256(OUT / "mathdial" / "validation.jsonl"),
        "md7r1_epoch3/model.safetensors": sha256(model_out / "model.safetensors"),
        "md7r1_epoch3/config.json": sha256(model_out / "config.json"),
        "md7r1_epoch3/tokenizer.json": sha256(model_out / "tokenizer.json"),
        "md7r1_epoch3/tokenizer_config.json": sha256(model_out / "tokenizer_config.json"),
    }
    mathdial_counts = Counter(row["target_move"] for row in read_jsonl(MATHDIAL_TRAIN))
    replay_quota = {"generic": 864, "probing": 824, "focus": 1315, "telling": 597}
    mixture = {
        "original_telling_calibration_train": 6000,
        "matched_boundary_correction_train": 2400,
        "mathdial_train_replay": 3600,
        "total": 12000,
        "overall_synthetic_fraction": 0.70,
        "mathdial_replay_fraction": 0.30,
        "component_fractions": {"original_telling_calibration_train": 0.50, "matched_boundary_correction_train": 0.20, "mathdial_train_replay": 0.30},
        "mathdial_train_available_counts": dict(mathdial_counts),
        "mathdial_replay_quota_by_class": replay_quota,
        "resulting_training_label_counts": {"generic": 2364, "probing": 2324, "focus": 4015, "telling": 3297},
    }
    assert sum(mixture["resulting_training_label_counts"].values()) == 12000

    manifest = {
        "schema_version": "md7_r2_tell_c1_manifest_v1",
        "experiment": "MD7-R2-TELL-C1",
        "status": "prepared_not_trained_not_promoted",
        "seed": SEED,
        "move_order": MOVE_ORDER,
        "correction_dataset": {
            "file": "correction_train.jsonl", "sha256": correction_hash, "rows": 2400,
            "target_counts": {"focus": 1200, "telling": 1200}, "split": "train_only",
            "family_count": 1200, "schema_version": SCHEMA_VERSION,
        },
        "original_calibration_source": {
            "path_discovered": str(ORIGINAL_SYNTHETIC), "source_sha256": EXPECTED_HASHES["original_synthetic"],
            "source_rows": 9600, "packaged_train_rows": 6000, "packaged_validation_rows": 2400,
            "source_final_test_rows": 1200, "source_final_test_packaged": False,
            "challenge_source_sha256": EXPECTED_HASHES["original_challenge"], "challenge_rows": 800,
        },
        "training_mixture": mixture,
        "model_initialization": {
            "path": "md7r1_epoch3", "source_path": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": EXPECTED_HASHES["baseline_model"], "not_telling_calibration_epoch": True,
        },
        "mathdial": {
            "train_rows": 14905, "validation_rows": 1850,
            "train_sha256": EXPECTED_HASHES["mathdial_train"], "validation_sha256": EXPECTED_HASHES["mathdial_validation"],
            "test_packaged_or_read": False,
        },
        "hyperparameters": {
            "seed": 42, "learning_rate": 1e-5, "epochs": 3, "weight_decay": 0.01,
            "warmup_ratio": 0.10, "effective_global_batch": 16, "max_length": 512,
            "objective": "ordinary_four_class_cross_entropy",
        },
        "preprocessing": {
            "first_sequence_prefix": "Problem:\n", "second_sequence_prefix": "Conversation:\n",
            "dialogue_format": "{user}: {text}", "suffix": "\n\nNext teacher pedagogical move:",
            "tokenizer_truncation_side": "left", "truncation": "only_second", "label_order": MOVE_ORDER,
        },
        "pretraining_guard_expected": {
            "accuracy": 0.5162162162162162, "macro_f1": 0.4611286390258053,
            "per_class_f1": {"generic": 0.6105263157894737, "probing": 0.3333333333333333,
                             "focus": 0.6004672897196262, "telling": 0.300187617260788},
            "prediction_distribution": {"generic": 413, "probing": 188, "focus": 1034, "telling": 215},
        },
        "kaggle_file_hashes": packaged_hashes,
        "guardrails": {
            "local_training": False, "automatic_epoch_selection": False, "production_changes": 0,
            "authoritative_real_data_writes": 0, "api_calls": 0, "bkt_updates": 0, "lints_updates": 0,
            "mathdial_final_test_use": 0, "mrbench_v3_test_use": 0,
        },
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    nb = notebook(correction_hash, packaged_hashes, mixture)
    (OUT / "MD7-R2-TELL-C1-kaggle.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    upload_files = [
        "correction_train.jsonl", "dataset_audit.json", "dataset_manifest.json",
        "telling_boundary_challenge_val_v1.jsonl", "telling_calibration_train_v1.jsonl",
        "telling_calibration_validation_v1.jsonl",
    ]
    upload_manifest = {
        "experiment": "MD7-R2-TELL-C1",
        "upload_root": str(OUT),
        "preparation_dataset_name": "md7-r2-tell-c1-preparation-v1",
        "files": upload_files,
        "notebook_file": "MD7-R2-TELL-C1-kaggle.ipynb",
        "attach_existing_inputs": [
            {
                "name": "mathdial-move-selection",
                "required_files": ["train.jsonl", "validation.jsonl"],
                "note": "test.jsonl may be mounted but is never opened or used",
            },
            {
                "name": "MD7-R1 - AdaptMath notebook output",
                "required_model_files": ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"],
                "required_model_sha256": EXPECTED_HASHES["baseline_model"],
            },
        ],
        "excluded_from_use": ["MathDial test", "MRBench V3 test", "original synthetic final test", "telling-calibration Epoch 1/2/3 models"],
    }
    (OUT / "kaggle_upload_manifest.json").write_text(json.dumps(upload_manifest, indent=2) + "\n", encoding="utf-8")

    readme = f"""# MD7-R2-TELL-C1 preparation

Status: **prepared only — not trained, selected, promoted, or deployed**.

## Purpose

Teach the boundary that continued difficulty after exactly one meaningful scaffold normally remains `focus`, while persistent unresolved difficulty after two distinct meaningful scaffolds may become `telling`.

## Correction dataset audit

- Rows: **2,400** (`focus=1,200`, `telling=1,200`), all `split=train`.
- Matched families: **1,200**, each containing one one-scaffold hard negative and one two-scaffold positive on the same problem.
- Exact problem+history duplicates: **0**.
- Exact history duplicates: **0**.
- Exact overlap with frozen 48-case diagnostic: **0**.
- Exact overlap with original synthetic validation/final-test/challenge: **0**.
- Focus rows containing a direct-answer request: **0**.
- Model-text artificial label-token occurrences: **0**.
- Real learner identifiers: **0**.

The frozen diagnostic was used only after generation for exact-overlap auditing. Its cases were not copied, individually paraphrased, or mechanically transformed.

## Exact training pool

| component | rows | fraction |
| --- | ---: | ---: |
| Original telling-calibration synthetic train | 6,000 | 50% |
| New matched correction train | 2,400 | 20% |
| MathDial train replay | 3,600 | 30% |
| Total | 12,000 | 100% |

Overall synthetic/MathDial ratio is exactly **70%/30%**. MathDial replay quota is `generic=864, probing=824, focus=1,315, telling=597`. Resulting class counts are `generic=2,364, probing=2,324, focus=4,015, telling=3,297`.

## Kaggle execution

1. Create/attach the `md7-r2-tell-c1-preparation-v1` dataset containing the six root files listed in `kaggle_upload_manifest.json`.
2. Separately attach `mathdial-move-selection`, containing `train.jsonl` and `validation.jsonl`. Its mounted `test.jsonl` is never opened or used.
3. Separately attach the `MD7-R1 - AdaptMath` notebook output containing the complete frozen epoch-3 Hugging Face export (`model.safetensors`, config, and tokenizer files).
4. Open `MD7-R2-TELL-C1-kaggle.ipynb`, enable a GPU, and run top to bottom. The notebook discovers these inputs by exact SHA-256 rather than fixed Kaggle owner paths.
5. The notebook blocks training unless frozen MD7-R1 reproduces the exact MathDial validation metrics.
6. Download `md7_r2_tell_c1_epoch_candidates.zip` after all three epochs finish.
7. Do not choose or promote an epoch in the notebook. Run the unchanged frozen offline diagnostic on all three exports.

The package intentionally excludes MathDial test, MRBench V3 test, the original final synthetic test, and all telling-calibration Epoch 1/2/3 models.

## Reproduction

`generate_package.py` deterministically regenerates the correction data and derived package from seed 42. It also rechecks all source and protected-artifact hashes before and after generation.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    after = {str(path): sha256(path) for path in sources}
    changes = {path: {"before": before[path], "after": after[path]} for path in before if before[path] != after[path]}
    assert not changes, changes
    audit["source_and_protected_hash_changes"] = changes
    audit["production_changes"] = 0
    audit["local_training_runs"] = 0
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(correction), "targets": Counter(r["target_move"] for r in correction),
                      "duplicates": audit["exact_problem_history_duplicate_count"], "overlap": audit["overlap"],
                      "mixture": mixture, "protected_changes": changes}, indent=2))


if __name__ == "__main__":
    main()
