from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


OUT = Path(__file__).resolve().parent
MOVES = ("generic", "probing", "focus", "telling")
PROGRESS_STRATA = {
    "G_correct_progress_after_previous_difficulty",
    "H_partial_recovering_progress_after_previous_difficulty",
}
ONE_SCAFFOLD_STRATA = {"C_recoverable_after_one_meaningful_scaffold"}
PERSISTENT_STRATA = {
    "D_repeated_misconception_after_multiple_scaffolds",
    "E_repeated_confusion_after_multiple_scaffolds",
    "F_direct_explanation_request_after_previous_support",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_binomial_two_sided(successes: int, failures: int) -> float | None:
    n = successes + failures
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(successes, failures) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def model_metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    return {
        "n": len(truth),
        "agreement_count": sum(a == b for a, b in zip(truth, predicted, strict=True)),
        "agreement_rate": sum(a == b for a, b in zip(truth, predicted, strict=True)) / len(truth),
        "macro_f1": float(f1_score(truth, predicted, labels=list(MOVES), average="macro", zero_division=0)),
        "per_class_f1": {
            move: float(value)
            for move, value in zip(
                MOVES,
                f1_score(truth, predicted, labels=list(MOVES), average=None, zero_division=0),
                strict=True,
            )
        },
        "telling_precision": float(
            precision_score(truth, predicted, labels=["telling"], average="macro", zero_division=0)
        ),
        "telling_recall": float(
            recall_score(truth, predicted, labels=["telling"], average="macro", zero_division=0)
        ),
        "prediction_counts": dict(Counter(predicted)),
        "expert_label_counts": dict(Counter(truth)),
        "confusion_matrix": {
            "label_order": list(MOVES),
            "rows_expert_columns_prediction": confusion_matrix(
                truth, predicted, labels=list(MOVES)
            ).tolist(),
        },
    }


def subset_counts(
    rows: Iterable[dict[str, Any]],
    strata: set[str],
) -> dict[str, Any]:
    selected = [row for row in rows if row["semantic_stratum"] in strata]
    expert_telling = sum(row["expert_move"] == "telling" for row in selected)
    return {
        "n": len(selected),
        "expert_telling_count": expert_telling,
        "reference_telling_count": sum(row["reference_top1"] == "telling" for row in selected),
        "recalibrated_telling_count": sum(row["recalibrated_top1"] == "telling" for row in selected),
        "reference_false_telling_count": sum(
            row["expert_move"] != "telling" and row["reference_top1"] == "telling" for row in selected
        ),
        "recalibrated_false_telling_count": sum(
            row["expert_move"] != "telling" and row["recalibrated_top1"] == "telling" for row in selected
        ),
        "expert_non_telling_n": len(selected) - expert_telling,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(clean(value) for value in row) + " |" for row in rows],
        ]
    )


def main() -> None:
    manifest = json.loads((OUT / "fresh_review_states_manifest.json").read_text(encoding="utf-8"))
    states_path = OUT / "fresh_review_states.jsonl"
    if sha256(states_path) != manifest["review_set"]["sha256"]:
        raise RuntimeError("INTEGRITY FAILURE: frozen review-set SHA-256 mismatch")
    states = read_jsonl(states_path)
    predictions = read_csv(OUT / "hidden_model_predictions.csv")
    mapping = json.loads((OUT / "hidden_model_mapping.json").read_text(encoding="utf-8"))
    judgments = read_csv(OUT / "expert_review_form.csv")

    expected_packet = {
        row["case_id"] for row in predictions if row["in_expert_packet"] == "True"
    }
    if len(judgments) != len(expected_packet) or {row["case_id"] for row in judgments} != expected_packet:
        raise RuntimeError("Expert form case IDs do not exactly match the frozen packet")
    if len({row["case_id"] for row in judgments}) != len(judgments):
        raise RuntimeError("Expert form contains duplicate case IDs")

    incomplete = [
        row["case_id"]
        for row in judgments
        if row["expert_move"] not in MOVES or row["expert_confidence"] not in {"1", "2", "3"}
    ]
    if incomplete:
        raise RuntimeError(
            "EXPERT REVIEW NOT COMPLETE: fill expert_move and expert_confidence for every packet row; "
            f"incomplete/invalid rows={len(incomplete)}"
        )
    for row in judgments:
        reason = row["expert_reason"].strip()
        if "\n" in reason or "\r" in reason:
            raise RuntimeError("expert_reason must be at most one sentence on one line")

    role_to_system = {
        details["model_role"]: system.casefold().replace(" ", "_")
        for system, details in mapping["system_mapping"].items()
    }
    if set(role_to_system) != {"reference", "recalibrated"}:
        raise RuntimeError("Hidden mapping does not identify both model roles")
    state_by_id = {row["case_id"]: row for row in states}
    prediction_by_id = {row["case_id"]: row for row in predictions}

    joined: list[dict[str, Any]] = []
    for judgment in judgments:
        case_id = judgment["case_id"]
        state = state_by_id[case_id]
        prediction = prediction_by_id[case_id]
        joined.append(
            {
                "case_id": case_id,
                "semantic_stratum": state["semantic_stratum"],
                "expert_move": judgment["expert_move"],
                "expert_confidence": int(judgment["expert_confidence"]),
                "expert_reason": judgment["expert_reason"].strip(),
                "reference_top1": prediction[f"{role_to_system['reference']}_top1"],
                "recalibrated_top1": prediction[f"{role_to_system['recalibrated']}_top1"],
                "models_disagree": prediction["systems_agree"] == "False",
            }
        )

    truth = [row["expert_move"] for row in joined]
    reference_predictions = [row["reference_top1"] for row in joined]
    recalibrated_predictions = [row["recalibrated_top1"] for row in joined]
    reference_metrics = model_metrics(truth, reference_predictions)
    recalibrated_metrics = model_metrics(truth, recalibrated_predictions)

    disagreements = [row for row in joined if row["models_disagree"]]
    recalibrated_wins = sum(
        row["recalibrated_top1"] == row["expert_move"]
        and row["reference_top1"] != row["expert_move"]
        for row in disagreements
    )
    reference_wins = sum(
        row["reference_top1"] == row["expert_move"]
        and row["recalibrated_top1"] != row["expert_move"]
        for row in disagreements
    )
    both_wrong = sum(
        row["reference_top1"] != row["expert_move"]
        and row["recalibrated_top1"] != row["expert_move"]
        for row in disagreements
    )
    decisive = recalibrated_wins + reference_wins

    expert_telling_n = sum(value == "telling" for value in truth)
    expert_non_telling_n = len(truth) - expert_telling_n
    telling_specific = {
        "expert_telling_cases": {
            "n": expert_telling_n,
            "reference_telling_recall": reference_metrics["telling_recall"],
            "recalibrated_telling_recall": recalibrated_metrics["telling_recall"],
        },
        "expert_non_telling_cases": {
            "n": expert_non_telling_n,
            "reference_false_telling_count": sum(
                expert != "telling" and predicted == "telling"
                for expert, predicted in zip(truth, reference_predictions, strict=True)
            ),
            "recalibrated_false_telling_count": sum(
                expert != "telling" and predicted == "telling"
                for expert, predicted in zip(truth, recalibrated_predictions, strict=True)
            ),
        },
        "G_H_correct_or_recovering_progress": subset_counts(joined, PROGRESS_STRATA),
        "C_recoverable_after_one_scaffold": subset_counts(joined, ONE_SCAFFOLD_STRATA),
        "D_E_F_persistent_or_support_exhausted": subset_counts(joined, PERSISTENT_STRATA),
    }

    progress = telling_specific["G_H_correct_or_recovering_progress"]
    progress_non_telling_n = progress["expert_non_telling_n"]
    reference_progress_rate = (
        progress["reference_false_telling_count"] / progress_non_telling_n
        if progress_non_telling_n else 0.0
    )
    recalibrated_progress_rate = (
        progress["recalibrated_false_telling_count"] / progress_non_telling_n
        if progress_non_telling_n else 0.0
    )
    substantive_progress_overtrigger = (
        progress["recalibrated_false_telling_count"]
        >= progress["reference_false_telling_count"] + 2
        or recalibrated_progress_rate - reference_progress_rate >= 0.20
    )

    confidences = [row["expert_confidence"] for row in joined]
    integrity_problems: list[str] = []
    if decisive < 8:
        integrity_problems.append("fewer_than_8_decisive_disagreements")
    if median(confidences) < 2:
        integrity_problems.append("median_expert_confidence_below_2")
    if expert_telling_n == 0 or expert_non_telling_n == 0:
        integrity_problems.append("expert_labels_lack_telling_or_non_telling_coverage")

    if integrity_problems:
        decision_code = "D"
        decision = "D. REVIEW INCONCLUSIVE"
    elif substantive_progress_overtrigger:
        decision_code = "B"
        decision = "B. RESIDUAL TELLING OVERCORRECTION"
    elif (
        recalibrated_wins > reference_wins
        and recalibrated_metrics["telling_precision"]
        >= reference_metrics["telling_precision"] - 0.05
    ):
        decision_code = "A"
        decision = "A. ADVANCE TO CONTROLLED LIVE VALIDATION"
    else:
        decision_code = "C"
        decision = "C. TELLING BENEFIT NOT CONFIRMED"

    summary = {
        "schema_version": "md7_r2_tell_c1_blinded_review_analysis_v1",
        "frozen_review_set_sha256": manifest["review_set"]["sha256"],
        "reviewed_n": len(joined),
        "expert_confidence_counts": dict(Counter(confidences)),
        "expert_confidence_median": float(median(confidences)),
        "models": {
            "MD7-R1": reference_metrics,
            "MD7-R2-TELL-C1_epoch2": recalibrated_metrics,
        },
        "disagreement_win_analysis": {
            "disagreement_n": len(disagreements),
            "recalibrated_wins": recalibrated_wins,
            "reference_wins": reference_wins,
            "both_wrong": both_wrong,
            "decisive_n": decisive,
            "exact_two_sided_binomial_p": exact_binomial_two_sided(recalibrated_wins, reference_wins),
            "interpretation": "Small diagnostic sample; raw counts are primary and no causal or production-effectiveness claim is made.",
        },
        "telling_specific": telling_specific,
        "decision_inputs": {
            "substantive_G_H_progress_overtrigger": substantive_progress_overtrigger,
            "reference_G_H_false_telling_rate": reference_progress_rate,
            "recalibrated_G_H_false_telling_rate": recalibrated_progress_rate,
            "integrity_or_inconclusive_reasons": integrity_problems,
        },
        "decision_code": decision_code,
        "decision": decision,
        "production_promotion_authorized": False,
    }
    (OUT / "review_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    model_table = [
        [
            name,
            metrics["agreement_count"],
            f"{metrics['agreement_rate']:.6f}",
            f"{metrics['macro_f1']:.6f}",
            f"{metrics['telling_precision']:.6f}",
            f"{metrics['telling_recall']:.6f}",
        ]
        for name, metrics in summary["models"].items()
    ]
    f1_table = [
        [move, f"{reference_metrics['per_class_f1'][move]:.6f}", f"{recalibrated_metrics['per_class_f1'][move]:.6f}"]
        for move in MOVES
    ]
    telling_table = [
        [name, values["n"], values.get("expert_telling_count", "-"), values.get("reference_telling_count", "-"), values.get("recalibrated_telling_count", "-")]
        for name, values in (
            ("G/H correct or recovering progress", telling_specific["G_H_correct_or_recovering_progress"]),
            ("C one-scaffold recovery", telling_specific["C_recoverable_after_one_scaffold"]),
            ("D/E/F persistent difficulty", telling_specific["D_E_F_persistent_or_support_exhausted"]),
        )
    ]
    report = f"""# MD7-R2-TELL-C1 Blinded Expert Review Analysis

Frozen review-set SHA-256: `{manifest['review_set']['sha256']}`. Reviewed cases: **{len(joined)}**.

## Model agreement with frozen expert judgments

{markdown_table(['model', 'exact matches', 'agreement rate', 'Macro-F1', 'telling precision', 'telling recall'], model_table)}

## Per-class F1

{markdown_table(['move', 'MD7-R1', 'MD7-R2-TELL-C1 Epoch 2'], f1_table)}

## Disagreement wins

- Recalibrated wins: **{recalibrated_wins}**
- Reference wins: **{reference_wins}**
- Both wrong: **{both_wrong}**
- Decisive disagreements: **{decisive}**
- Two-sided exact binomial p-value: **{summary['disagreement_win_analysis']['exact_two_sided_binomial_p']}**

Raw counts are primary. This small diagnostic does not establish production effectiveness or causal learning benefit.

## Telling-specific strata

{markdown_table(['stratum group', 'reviewed n', 'expert telling', 'MD7-R1 telling', 'MD7-R2 telling'], telling_table)}

Expert-telling cases: **{expert_telling_n}**; MD7-R1 recall **{reference_metrics['telling_recall']:.6f}**, recalibrated recall **{recalibrated_metrics['telling_recall']:.6f}**.

Expert-non-telling cases: **{expert_non_telling_n}**; MD7-R1 false telling **{telling_specific['expert_non_telling_cases']['reference_false_telling_count']}**, recalibrated false telling **{telling_specific['expert_non_telling_cases']['recalibrated_false_telling_count']}**.

## Preregistered decision

**{decision}**

No outcome authorizes production promotion.
"""
    (OUT / "review_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"reviewed_n": len(joined), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
