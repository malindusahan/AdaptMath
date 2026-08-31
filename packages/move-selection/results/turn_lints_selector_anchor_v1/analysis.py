"""Calibrate the frozen MD7 action anchor from pre-action distributions only.

This is an initialization/policy-design experiment. It never fits rewards,
updates a posterior, reads protected test data, or writes runtime policy state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
MOVE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = MOVE_ROOT.parent
PASSIVE_TURNS = (
    MOVE_ROOT / "results" / "md_self_improvement_turn_data_v1" / "turn_outcomes.jsonl"
)
if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.context_builder import MOVE_ORDER  # noqa: E402
from src.self_improvement.turn_lints_context import (  # noqa: E402
    BLOCK_ORDER,
    build_turn_lints_context,
    context_schema_id,
    feature_names_for_blocks,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    ANCHOR_PROBABILITY_FLOOR,
    DirectTurnLinTS,
)


SELECTOR_VERSION = "MD7-R2-TELL-C1 epoch 2"
SELECTOR_SHA256 = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
SPLIT_SALT = "turn-lints-selector-anchor-v1-grouped-split"
CALIBRATION_ATTEMPT_FRACTION = 2.0 / 3.0
SEED_COUNT = 512
GAMMAS = tuple(round(0.02 * index, 2) for index in range(41))
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20
REWARD_SD = 0.2075
REWARD_Q1 = -0.0457
REWARD_Q3 = 0.2406
REWARD_IQR = REWARD_Q3 - REWARD_Q1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("Passive turn rows must be JSON objects.")
                rows.append(value)
    return rows


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def stored_context(row: dict[str, Any]) -> dict[str, float]:
    decision = row.get("adaptive_decision")
    if not isinstance(decision, dict):
        return {}
    names = decision.get("feature_names") or decision.get("context_feature_names")
    vector = decision.get("vector") or decision.get("context")
    if not isinstance(names, list) or not isinstance(vector, list):
        return {}
    if len(names) != len(vector):
        return {}
    return {
        str(name): float(value)
        for name, value in zip(names, vector, strict=True)
    }


def reconstruct_pre_action_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(nested(row, "identity", "attempt_id")),
            int(nested(row, "identity", "action_turn_index") or 0),
        ),
    )
    previous: dict[str, dict[str, Any]] = {}
    contexts: list[dict[str, Any]] = []
    for row in ordered:
        identity = row.get("identity") or {}
        attempt_id = str(identity.get("attempt_id"))
        turn_index = int(identity.get("action_turn_index") or 0)
        selector = nested(row, "selector", "raw", "probabilities")
        if not isinstance(selector, dict) or set(selector) != set(MOVE_ORDER):
            raise ValueError("Every context requires four frozen MD7 probabilities.")
        probabilities = {arm: float(selector[arm]) for arm in MOVE_ORDER}
        mastery = nested(row, "state_before_action", "mastery_at_action")
        if mastery is None:
            mastery = nested(row, "learning_state_update", "mastery_before")
        if mastery is None:
            raise ValueError("Pre-action mastery is unavailable.")

        prior = previous.get(attempt_id)
        previous_delta = None
        if prior is not None and prior["reward_observed"]:
            previous_delta = float(prior["delta"])
        existing = stored_context(row)
        l_names = (
            "previous_reasoning_probability",
            "previous_uncertainty_probability",
            "previous_clarification_probability",
        )
        learner_signals = None
        if existing.get("previous_learner_signals_missing") == 0.0 and all(
            name in existing for name in l_names
        ):
            learner_signals = {
                "reasoning_probability": existing[l_names[0]],
                "uncertainty_probability": existing[l_names[1]],
                "clarification_probability": existing[l_names[2]],
            }
        context = build_turn_lints_context(
            enabled_blocks=BLOCK_ORDER,
            selector_probabilities=probabilities,
            mastery_before=float(mastery),
            previous_mastery_delta=previous_delta,
            previous_learner_signals=learner_signals,
            previous_move=None if prior is None else prior["actual_move"],
            prior_tutor_turn_count=turn_index - 1,
            previous_mrb1_scores=None if prior is None else prior["mrb1"],
        )
        contexts.append(
            {
                "attempt_id": attempt_id,
                "action_event_id": identity.get("action_event_id"),
                "turn_index": turn_index,
                "probabilities": probabilities,
                "md7_top": max(MOVE_ORDER, key=probabilities.__getitem__),
                "vector": context.vector(),
            }
        )

        scores = nested(row, "tutor_quality", "scores")
        if not isinstance(scores, dict) or len(scores) != 4:
            scores = None
        actual_move = nested(row, "adaptive_decision", "final_move")
        if actual_move not in MOVE_ORDER:
            actual_move = nested(row, "selector", "effective", "argmax")
        learning = row.get("learning_state_update") or {}
        observed = (
            learning.get("should_update") is True
            and nested(row, "next_learner_observation", "status")
            == "observed_update"
            and learning.get("delta_mastery") is not None
        )
        previous[attempt_id] = {
            "actual_move": actual_move,
            "mrb1": scores,
            "reward_observed": observed,
            "delta": learning.get("delta_mastery"),
        }
    return contexts


def grouped_split(
    contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    attempts = sorted({str(item["attempt_id"]) for item in contexts})
    ranked = sorted(
        attempts,
        key=lambda attempt: hashlib.sha256(
            f"{SPLIT_SALT}|{attempt}".encode("utf-8")
        ).hexdigest(),
    )
    calibration_count = round(len(ranked) * CALIBRATION_ATTEMPT_FRACTION)
    calibration_attempts = sorted(ranked[:calibration_count])
    validation_attempts = sorted(ranked[calibration_count:])
    if set(calibration_attempts) & set(validation_attempts):
        raise RuntimeError("Grouped split leaked an attempt across partitions.")
    calibration = [
        item for item in contexts if item["attempt_id"] in calibration_attempts
    ]
    validation = [
        item for item in contexts if item["attempt_id"] in validation_attempts
    ]
    if len(calibration) + len(validation) != len(contexts):
        raise RuntimeError("Grouped split lost contexts.")
    return calibration, validation, calibration_attempts, validation_attempts


def sample_reward_scores(contexts: list[dict[str, Any]]) -> np.ndarray:
    scores = np.empty(
        (len(contexts), SEED_COUNT, len(MOVE_ORDER)), dtype=np.float64
    )
    for seed in range(SEED_COUNT):
        policy = DirectTurnLinTS(
            context_schema_id=context_schema_id(BLOCK_ORDER),
            enabled_context_blocks=BLOCK_ORDER,
            feature_names=feature_names_for_blocks(BLOCK_ORDER),
            selector_version=SELECTOR_VERSION,
            selector_sha256=SELECTOR_SHA256,
            reward_mode="headroom_normalized",
            anchor_mode="none",
            anchor_gamma=0.0,
            ridge_lambda=RIDGE_LAMBDA,
            exploration_scale=EXPLORATION_SCALE,
            seed=seed,
            data_mode="real",
        )
        for context_index, item in enumerate(contexts):
            result = policy.select_arm(item["vector"])
            scores[context_index, seed, :] = [
                result["contextual_ts_reward_scores"][arm] for arm in MOVE_ORDER
            ]
        if policy.total_updates != 0:
            raise RuntimeError("Selection-only calibration mutated a posterior.")
    return scores


def empirical_policy_distributions(
    contexts: list[dict[str, Any]], reward_scores: np.ndarray, gamma: float
) -> np.ndarray:
    log_probabilities = np.asarray(
        [
            [
                math.log(
                    max(item["probabilities"][arm], ANCHOR_PROBABILITY_FLOOR)
                )
                for arm in MOVE_ORDER
            ]
            for item in contexts
        ],
        dtype=np.float64,
    )
    choices = np.argmax(
        reward_scores + gamma * log_probabilities[:, np.newaxis, :], axis=2
    )
    distributions = np.empty((len(contexts), len(MOVE_ORDER)), dtype=np.float64)
    for arm_index in range(len(MOVE_ORDER)):
        distributions[:, arm_index] = np.mean(choices == arm_index, axis=1)
    return distributions


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (p + q)

    def contribution(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(left)
        mask = left > 0.0
        terms[mask] = left[mask] * np.log(left[mask] / right[mask])
        return terms

    return 0.5 * np.sum(contribution(p, midpoint), axis=1) + 0.5 * np.sum(
        contribution(q, midpoint), axis=1
    )


def percentile(values: Iterable[float], q: float) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    return None if data.size == 0 else float(np.quantile(data, q))


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {
            key: 0 if key == "count" else None
            for key in ("count", "min", "q1", "median", "mean", "q3", "p90", "p95", "max")
        }
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "q1": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "q3": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def matching_metrics(
    contexts: list[dict[str, Any]], distributions: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(
        [[item["probabilities"][arm] for arm in MOVE_ORDER] for item in contexts],
        dtype=np.float64,
    )
    js = js_divergence_rows(target, distributions)
    safe_q = np.maximum(distributions, ANCHOR_PROBABILITY_FLOOR)
    safe_q /= safe_q.sum(axis=1, keepdims=True)
    target_to_q_kl = np.sum(target * np.log(target / safe_q), axis=1)
    top_indices = np.argmax(target, axis=1)
    expected_agreement = float(
        np.mean(distributions[np.arange(len(contexts)), top_indices])
    )
    telling_index = MOVE_ORDER.index("telling")
    non_telling_mask = top_indices != telling_index
    return {
        "js_nats": distribution_summary(js),
        "target_to_q_kl_nats": distribution_summary(target_to_q_kl),
        "expected_top1_agreement": expected_agreement,
        "empirical_action_distribution": {
            arm: float(np.mean(distributions[:, arm_index]))
            for arm_index, arm in enumerate(MOVE_ORDER)
        },
        "target_md7_distribution": {
            arm: float(np.mean(target[:, arm_index]))
            for arm_index, arm in enumerate(MOVE_ORDER)
        },
        "telling_selection_rate": float(np.mean(distributions[:, telling_index])),
        "telling_when_md7_top1_not_telling_rate": (
            float(np.mean(distributions[non_telling_mask, telling_index]))
            if np.any(non_telling_mask)
            else None
        ),
        "context_js_nats": [float(value) for value in js],
    }


def anchor_barriers(
    contexts: list[dict[str, Any]], gamma: float
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {
        "all_pairwise_absolute": [],
        "top1_to_second_best": [],
        "top1_to_each_alternative": [],
        "non_telling_top1_to_telling": [],
        "telling_top1_to_best_non_telling": [],
    }
    for item in contexts:
        probabilities = item["probabilities"]
        logp = {
            arm: math.log(max(probabilities[arm], ANCHOR_PROBABILITY_FLOOR))
            for arm in MOVE_ORDER
        }
        for index, arm in enumerate(MOVE_ORDER):
            for other in MOVE_ORDER[index + 1 :]:
                values["all_pairwise_absolute"].append(
                    abs(gamma * (logp[arm] - logp[other]))
                )
        ranked = sorted(MOVE_ORDER, key=probabilities.__getitem__, reverse=True)
        top, second = ranked[0], ranked[1]
        values["top1_to_second_best"].append(gamma * (logp[top] - logp[second]))
        for alternative in ranked[1:]:
            values["top1_to_each_alternative"].append(
                gamma * (logp[top] - logp[alternative])
            )
        if top != "telling":
            values["non_telling_top1_to_telling"].append(
                gamma * (logp[top] - logp["telling"])
            )
        else:
            best_non_telling = max(
                (arm for arm in MOVE_ORDER if arm != "telling"),
                key=probabilities.__getitem__,
            )
            values["telling_top1_to_best_non_telling"].append(
                gamma * (logp[top] - logp[best_non_telling])
            )
    return values


def barrier_rows(contexts: list[dict[str, Any]], gamma: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, values in anchor_barriers(contexts, gamma).items():
        summary = distribution_summary(values)
        row: dict[str, Any] = {"category": category, **summary}
        for field in ("q1", "median", "q3", "p90", "p95", "max"):
            value = summary[field]
            row[f"{field}_reward_sd"] = (
                None if value is None else float(value) / REWARD_SD
            )
            row[f"{field}_reward_iqr"] = (
                None if value is None else float(value) / REWARD_IQR
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    contexts = reconstruct_pre_action_contexts(load_jsonl(PASSIVE_TURNS))
    calibration, validation, calibration_attempts, validation_attempts = grouped_split(
        contexts
    )
    reward_scores = sample_reward_scores(contexts)
    index_by_event = {
        str(item["action_event_id"]): index for index, item in enumerate(contexts)
    }

    calibration_indices = np.asarray(
        [index_by_event[str(item["action_event_id"])] for item in calibration]
    )
    validation_indices = np.asarray(
        [index_by_event[str(item["action_event_id"])] for item in validation]
    )
    calibration_rows: list[dict[str, Any]] = []
    for gamma in GAMMAS:
        distributions = empirical_policy_distributions(
            calibration, reward_scores[calibration_indices, :, :], gamma
        )
        metrics = matching_metrics(calibration, distributions)
        js = metrics["js_nats"]
        calibration_rows.append(
            {
                "gamma": gamma,
                "calibration_context_count": len(calibration),
                "mean_js_nats": js["mean"],
                "median_js_nats": js["median"],
                "q1_js_nats": js["q1"],
                "q3_js_nats": js["q3"],
                "p90_js_nats": js["p90"],
                "p95_js_nats": js["p95"],
                "max_js_nats": js["max"],
                "expected_top1_agreement": metrics["expected_top1_agreement"],
                "selected_generic_rate": metrics["empirical_action_distribution"]["generic"],
                "selected_probing_rate": metrics["empirical_action_distribution"]["probing"],
                "selected_focus_rate": metrics["empirical_action_distribution"]["focus"],
                "selected_telling_rate": metrics["empirical_action_distribution"]["telling"],
                "telling_when_md7_top1_not_telling_rate": metrics[
                    "telling_when_md7_top1_not_telling_rate"
                ],
            }
        )
    selected_row = min(
        calibration_rows, key=lambda row: (row["mean_js_nats"], row["gamma"])
    )
    selected_gamma = float(selected_row["gamma"])

    validation_distributions = empirical_policy_distributions(
        validation, reward_scores[validation_indices, :, :], selected_gamma
    )
    validation_metrics = matching_metrics(validation, validation_distributions)
    validation_js = validation_metrics["js_nats"]
    validation_row = {
        "selected_gamma": selected_gamma,
        "validation_attempt_count": len(validation_attempts),
        "validation_context_count": len(validation),
        "mean_js_nats": validation_js["mean"],
        "median_js_nats": validation_js["median"],
        "q1_js_nats": validation_js["q1"],
        "q3_js_nats": validation_js["q3"],
        "p90_js_nats": validation_js["p90"],
        "p95_js_nats": validation_js["p95"],
        "max_js_nats": validation_js["max"],
        "mean_target_to_q_kl_nats": validation_metrics["target_to_q_kl_nats"]["mean"],
        "expected_top1_agreement": validation_metrics["expected_top1_agreement"],
        "selected_generic_rate": validation_metrics["empirical_action_distribution"]["generic"],
        "selected_probing_rate": validation_metrics["empirical_action_distribution"]["probing"],
        "selected_focus_rate": validation_metrics["empirical_action_distribution"]["focus"],
        "selected_telling_rate": validation_metrics["empirical_action_distribution"]["telling"],
        "telling_when_md7_top1_not_telling_rate": validation_metrics[
            "telling_when_md7_top1_not_telling_rate"
        ],
        "md7_generic_rate": validation_metrics["target_md7_distribution"]["generic"],
        "md7_probing_rate": validation_metrics["target_md7_distribution"]["probing"],
        "md7_focus_rate": validation_metrics["target_md7_distribution"]["focus"],
        "md7_telling_rate": validation_metrics["target_md7_distribution"]["telling"],
    }

    barriers = barrier_rows(contexts, selected_gamma)
    write_csv(HERE / "gamma_calibration.csv", calibration_rows)
    write_csv(HERE / "gamma_validation_summary.csv", [validation_row])
    write_csv(HERE / "override_barrier_summary.csv", barriers)

    selected = {
        "schema_version": "turn_lints_selected_anchor_v1",
        "scientific_decision": "B_P1_TOO_STRONG_POORLY_CALIBRATED",
        "activation_authorized": False,
        "anchor_mode": "md7_logprob_anchor",
        "anchor_gamma": selected_gamma,
        "anchor_probability_floor": ANCHOR_PROBABILITY_FLOOR,
        "anchor_selector_version": SELECTOR_VERSION,
        "anchor_selector_sha256": SELECTOR_SHA256,
        "selection_objective": "minimum calibration mean per-context Jensen-Shannon divergence in nats",
        "gamma_candidates": list(GAMMAS),
        "seed_count": SEED_COUNT,
        "group_split": {
            "salt": SPLIT_SALT,
            "calibration_attempts": calibration_attempts,
            "validation_attempts": validation_attempts,
            "calibration_context_count": len(calibration),
            "validation_context_count": len(validation),
            "attempt_overlap_count": 0,
        },
        "calibration_selected_row": selected_row,
        "held_out_validation": {
            **validation_metrics,
            "attempt_count": len(validation_attempts),
            "context_count": len(validation),
        },
        "posterior_updates": 0,
        "reward_outcomes_used_for_gamma_selection": 0,
    }
    (HERE / "cold_start_selected_gamma.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "turn_lints_selector_anchor_audit_v1",
        "source": str(PASSIVE_TURNS.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(PASSIVE_TURNS.read_bytes()).hexdigest(),
        "selector_version": SELECTOR_VERSION,
        "selector_sha256": SELECTOR_SHA256,
        "context_instrumentation": "S+K+L+H+Q",
        "context_dimension": 22,
        "context_scientific_status": "INCONCLUSIVE",
        "total_context_count": len(contexts),
        "total_attempt_count": len({item["attempt_id"] for item in contexts}),
        "calibration_attempt_count": len(calibration_attempts),
        "calibration_context_count": len(calibration),
        "validation_attempt_count": len(validation_attempts),
        "validation_context_count": len(validation),
        "selected_gamma": selected_gamma,
        "selected_by": "calibration mean JS divergence only",
        "scientific_decision": "B_P1_TOO_STRONG_POORLY_CALIBRATED",
        "runtime_anchor_activation": False,
        "calibration": selected_row,
        "validation": validation_metrics,
        "override_barriers": {row["category"]: row for row in barriers},
        "empirical_headroom_reward_scale": {
            "count": 96,
            "sd": REWARD_SD,
            "q1": REWARD_Q1,
            "q3": REWARD_Q3,
            "iqr": REWARD_IQR,
            "used_for_gamma_selection": False,
        },
        "safety": {
            "training": 0,
            "live_updates": 0,
            "external_tutor_api_calls": 0,
            "protected_test_rows": 0,
            "authoritative_historical_rewrites": 0,
        },
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "selected_gamma": result["selected_gamma"],
                "calibration_contexts": result["calibration_context_count"],
                "validation_contexts": result["validation_context_count"],
                "validation_mean_js_nats": result["validation"]["js_nats"]["mean"],
                "validation_median_js_nats": result["validation"]["js_nats"]["median"],
            },
            indent=2,
        )
    )
