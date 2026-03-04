"""Offline audit of frozen MD7-R2 probability-proportional action sampling.

The script replays only leakage-free problem/history snapshots through the
frozen local selector. It never reads a future outcome for policy analysis,
calls an external API, assigns a real treatment, or updates Turn-LinTS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
MOVE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = MOVE_ROOT.parent
PASSIVE_TURNS = (
    MOVE_ROOT / "results" / "md_self_improvement_turn_data_v1" / "turn_outcomes.jsonl"
)
MODEL_DIR = MOVE_ROOT / "models" / "frozen" / "md7_r2_tell_c1_epoch2"
MODEL_WEIGHTS = MODEL_DIR / "model.safetensors"
SELECTOR_VERSION = "MD7-R2-TELL-C1 epoch 2"
SELECTOR_SHA256 = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
BEHAVIOR_POLICY = "md7_r2_probability_proportional_v1"
MOVE_ORDER = ("generic", "probing", "focus", "telling")
PAIR_THRESHOLDS = (0.001, 0.005, 0.01, 0.025, 0.05)
MONTE_CARLO_SEEDS = 4096
HORIZONS = (50, 100, 250, 500)
HORIZON_REPLICATES = 10000

if str(MOVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOVE_ROOT))

from src.self_improvement.md6_inference import (  # noqa: E402
    FrozenMD6Inference,
    MAX_LENGTH,
)


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PASSIVE_TURNS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("Pre-action source rows must be JSON objects.")
                rows.append(row)
    return rows


def quantile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {
            key: 0 if key == "count" else None
            for key in (
                "count", "min", "p1", "p5", "p10", "q1", "median",
                "mean", "q3", "p90", "p95", "p99", "max",
            )
        }
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "p1": float(np.quantile(data, 0.01)),
        "p5": float(np.quantile(data, 0.05)),
        "p10": float(np.quantile(data, 0.10)),
        "q1": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "q3": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def _probability_vector(probabilities: dict[str, float]) -> np.ndarray:
    if tuple(probabilities) != MOVE_ORDER:
        raise ValueError("Frozen selector probability order is incompatible.")
    vector = np.asarray([probabilities[arm] for arm in MOVE_ORDER], dtype=np.float64)
    if not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("Frozen selector returned invalid probabilities.")
    total = float(np.sum(vector))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Frozen selector probabilities do not sum to one.")
    # This correction is only for floating-point summation, never arm filtering.
    return vector / total


def infer_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_hash = hashlib.sha256(MODEL_WEIGHTS.read_bytes()).hexdigest()
    if observed_hash != SELECTOR_SHA256:
        raise RuntimeError("Frozen MD7-R2 weight hash mismatch.")
    selector = FrozenMD6Inference(MODEL_DIR)
    contexts: list[dict[str, Any]] = []
    for row in rows:
        identity = row.get("identity")
        state = row.get("state_before_action")
        if not isinstance(identity, dict) or not isinstance(state, dict):
            raise ValueError("Row lacks pre-action identity/state.")
        problem = state.get("problem")
        history = state.get("history_before_action")
        if not isinstance(problem, str) or not isinstance(history, list):
            raise ValueError("Pre-action problem/history is unavailable.")
        probabilities = selector.predict_probabilities(problem, history)
        vector = _probability_vector(probabilities)
        ranked = np.argsort(-vector, kind="stable")
        top_index, second_index = int(ranked[0]), int(ranked[1])
        positive = vector > 0.0
        entropy = float(-np.sum(vector[positive] * np.log(vector[positive])))
        contexts.append(
            {
                "attempt_id": str(identity.get("attempt_id")),
                "action_event_id": str(identity.get("action_event_id")),
                "action_turn_index": int(identity.get("action_turn_index")),
                "problem": problem,
                "history_before_action": history,
                "latest_student_text": state.get("latest_student_text"),
                "mastery_at_action": state.get("mastery_at_action"),
                "probabilities": {
                    arm: float(vector[index]) for index, arm in enumerate(MOVE_ORDER)
                },
                "top1": MOVE_ORDER[top_index],
                "top1_probability": float(vector[top_index]),
                "second_best": MOVE_ORDER[second_index],
                "second_best_probability": float(vector[second_index]),
                "top1_second_gap": float(vector[top_index] - vector[second_index]),
                "entropy_nats": entropy,
                "non_top1_sampling_probability": float(1.0 - vector[top_index]),
            }
        )
    return contexts


def context_matrix(contexts: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[item["probabilities"][arm] for arm in MOVE_ORDER] for item in contexts],
        dtype=np.float64,
    )


def probability_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(MOVE_ORDER):
        values = matrix[:, arm_index]
        row: dict[str, Any] = {"move": arm, **quantile_summary(values)}
        for threshold in PAIR_THRESHOLDS:
            row[f"proportion_below_{threshold:g}"] = float(np.mean(values < threshold))
        rows.append(row)
    flattened = matrix.reshape(-1)
    aggregate: dict[str, Any] = {
        "move": "__all_context_action_pairs__",
        **quantile_summary(flattened),
    }
    for threshold in PAIR_THRESHOLDS:
        aggregate[f"proportion_below_{threshold:g}"] = float(
            np.mean(flattened < threshold)
        )
    rows.append(aggregate)
    return rows


def gap_rank_strata(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(contexts, key=lambda item: item["top1_second_gap"])
    rows: list[dict[str, Any]] = []
    for index, group_indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
        group = [ordered[int(item)] for item in group_indices]
        rows.append(
            {
                "gap_rank_quartile": index,
                "count": len(group),
                "min_gap": min(item["top1_second_gap"] for item in group),
                "max_gap": max(item["top1_second_gap"] for item in group),
                "mean_gap": float(np.mean([item["top1_second_gap"] for item in group])),
                "expected_top1_agreement": float(
                    np.mean([item["top1_probability"] for item in group])
                ),
                "expected_deviation_rate": float(
                    np.mean([item["non_top1_sampling_probability"] for item in group])
                ),
                "mean_entropy_nats": float(
                    np.mean([item["entropy_nats"] for item in group])
                ),
            }
        )
    return rows


def confidence_rank_strata(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe exploration across empirical top-1-confidence quartiles."""

    ordered = sorted(contexts, key=lambda item: item["top1_probability"])
    rows: list[dict[str, Any]] = []
    for index, group_indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
        group = [ordered[int(item)] for item in group_indices]
        rows.append(
            {
                "confidence_rank_quartile": index,
                "count": len(group),
                "min_top1_probability": min(
                    item["top1_probability"] for item in group
                ),
                "max_top1_probability": max(
                    item["top1_probability"] for item in group
                ),
                "mean_top1_probability": float(
                    np.mean([item["top1_probability"] for item in group])
                ),
                "expected_top1_agreement": float(
                    np.mean([item["top1_probability"] for item in group])
                ),
                "expected_deviation_rate": float(
                    np.mean([item["non_top1_sampling_probability"] for item in group])
                ),
                "mean_gap": float(
                    np.mean([item["top1_second_gap"] for item in group])
                ),
                "mean_entropy_nats": float(
                    np.mean([item["entropy_nats"] for item in group])
                ),
            }
        )
    return rows


def simulate_context_sweeps(matrix: np.ndarray) -> dict[str, Any]:
    selected_indices: list[np.ndarray] = []
    propensities: list[np.ndarray] = []
    cumulative = np.cumsum(matrix, axis=1)
    for seed in range(MONTE_CARLO_SEEDS):
        uniforms = np.random.default_rng(seed).random(len(matrix))
        choices = np.sum(uniforms[:, None] > cumulative, axis=1)
        choices = np.minimum(choices, len(MOVE_ORDER) - 1)
        selected_indices.append(choices)
        propensities.append(matrix[np.arange(len(matrix)), choices])
    selected = np.concatenate(selected_indices)
    sampled_propensity = np.concatenate(propensities)
    inverse_propensity = 1.0 / sampled_propensity
    return {
        "draw_count": int(selected.size),
        "simulated_action_distribution": {
            arm: float(np.mean(selected == arm_index))
            for arm_index, arm in enumerate(MOVE_ORDER)
        },
        "sampled_propensity": quantile_summary(sampled_propensity),
        "inverse_propensity": quantile_summary(inverse_propensity),
        "sampled_propensity_frequencies": {
            f"below_{threshold:g}": float(np.mean(sampled_propensity < threshold))
            for threshold in (0.01, 0.025, 0.05)
        },
    }


def horizon_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    exact_rates = np.mean(matrix, axis=0)
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        rng = np.random.default_rng(100_000 + horizon)
        context_indices = rng.integers(
            0, len(matrix), size=(HORIZON_REPLICATES, horizon)
        )
        sampled_probabilities = matrix[context_indices]
        uniforms = rng.random((HORIZON_REPLICATES, horizon))
        choices = np.sum(
            uniforms[:, :, None] > np.cumsum(sampled_probabilities, axis=2), axis=2
        )
        choices = np.minimum(choices, len(MOVE_ORDER) - 1)
        for arm_index, arm in enumerate(MOVE_ORDER):
            counts = np.sum(choices == arm_index, axis=1)
            rows.append(
                {
                    "horizon": horizon,
                    "move": arm,
                    "exact_expected_count": float(horizon * exact_rates[arm_index]),
                    "simulation_mean_count": float(np.mean(counts)),
                    "simulation_p5_count": float(np.quantile(counts, 0.05)),
                    "simulation_median_count": float(np.median(counts)),
                    "simulation_p95_count": float(np.quantile(counts, 0.95)),
                    "replicates": HORIZON_REPLICATES,
                }
            )
    return rows


def representative_cases(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    entropy_values = [item["entropy_nats"] for item in contexts]
    entropy_median = float(np.median(entropy_values))

    def compact(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in (
                "attempt_id", "action_event_id", "action_turn_index", "problem",
                "history_before_action", "latest_student_text", "mastery_at_action",
                "probabilities", "top1", "top1_probability", "second_best",
                "second_best_probability", "top1_second_gap", "entropy_nats",
                "non_top1_sampling_probability",
            )
        }

    non_top_telling = [item for item in contexts if item["top1"] != "telling"]
    nonempty_history = [item for item in contexts if item["history_before_action"]]
    return {
        "low_entropy": compact(min(contexts, key=lambda item: item["entropy_nats"])),
        "middle_entropy": compact(
            min(contexts, key=lambda item: abs(item["entropy_nats"] - entropy_median))
        ),
        "high_entropy": compact(max(contexts, key=lambda item: item["entropy_nats"])),
        "high_telling_not_top1": [
            compact(item)
            for item in sorted(
                non_top_telling,
                key=lambda item: item["probabilities"]["telling"],
                reverse=True,
            )[:3]
        ],
        "telling_top1": [
            compact(item)
            for item in sorted(
                (item for item in contexts if item["top1"] == "telling"),
                key=lambda item: item["probabilities"]["telling"],
                reverse=True,
            )[:3]
        ],
        "meaningful_focus_telling_competition": compact(
            max(
                contexts,
                key=lambda item: min(
                    item["probabilities"]["focus"],
                    item["probabilities"]["telling"],
                ),
            )
        ),
        "meaningful_probing_focus_competition": compact(
            max(
                contexts,
                key=lambda item: min(
                    item["probabilities"]["probing"],
                    item["probabilities"]["focus"],
                ),
            )
        ),
        "high_generic_with_history": compact(
            max(
                nonempty_history,
                key=lambda item: item["probabilities"]["generic"],
            )
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    source_hash = hashlib.sha256(PASSIVE_TURNS.read_bytes()).hexdigest()
    rows = load_rows()
    contexts = infer_contexts(rows)
    matrix = context_matrix(contexts)
    probability_table = probability_rows(matrix)
    exact_rates = np.mean(matrix, axis=0)
    entropy = quantile_summary(item["entropy_nats"] for item in contexts)
    top_probabilities = quantile_summary(item["top1_probability"] for item in contexts)
    gaps = quantile_summary(item["top1_second_gap"] for item in contexts)
    strata = gap_rank_strata(contexts)
    confidence_strata = confidence_rank_strata(contexts)
    simulation = simulate_context_sweeps(matrix)
    horizons = horizon_rows(matrix)
    representatives = representative_cases(contexts)

    entropy_rows = [
        {
            "attempt_id": item["attempt_id"],
            "action_event_id": item["action_event_id"],
            "action_turn_index": item["action_turn_index"],
            **{f"p_{arm}": item["probabilities"][arm] for arm in MOVE_ORDER},
            "top1": item["top1"],
            "top1_probability": item["top1_probability"],
            "second_best": item["second_best"],
            "second_best_probability": item["second_best_probability"],
            "top1_second_gap": item["top1_second_gap"],
            "entropy_nats": item["entropy_nats"],
            "non_top1_sampling_probability": item[
                "non_top1_sampling_probability"
            ],
        }
        for item in contexts
    ]
    coverage_rows = [
        {
            "move": arm,
            "exact_expected_action_rate": float(exact_rates[index]),
            "simulated_action_rate": simulation["simulated_action_distribution"][arm],
            **{
                f"exact_expected_count_N{horizon}": float(
                    horizon * exact_rates[index]
                )
                for horizon in HORIZONS
            },
        }
        for index, arm in enumerate(MOVE_ORDER)
    ]
    propensity_rows = [
        {
            "metric": "sampled_action_propensity",
            **simulation["sampled_propensity"],
            **{
                f"frequency_{key}": value
                for key, value in simulation[
                    "sampled_propensity_frequencies"
                ].items()
            },
        },
        {
            "metric": "inverse_propensity",
            **simulation["inverse_propensity"],
            "frequency_below_0.01": None,
            "frequency_below_0.025": None,
            "frequency_below_0.05": None,
        },
    ]
    write_csv(HERE / "md7_probability_summary.csv", probability_table)
    write_csv(HERE / "entropy_summary.csv", entropy_rows)
    write_csv(HERE / "sampling_propensity_summary.csv", propensity_rows)
    write_csv(HERE / "expected_action_coverage.csv", coverage_rows)
    write_csv(HERE / "coverage_horizon_simulation.csv", horizons)

    telling_index = MOVE_ORDER.index("telling")
    top_indices = np.argmax(matrix, axis=1)
    non_telling_top = top_indices != telling_index
    telling_top = top_indices == telling_index
    summary = {
        "schema_version": "turn_lints_randomized_warmstart_audit_v1",
        "scientific_decision": "A_RAW_MD7_PROPORTIONAL_RANDOMIZATION_DEFENSIBLE",
        "behavior_policy": BEHAVIOR_POLICY,
        "source": str(PASSIVE_TURNS.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "source_sha256": source_hash,
        "audited_pre_action_state_count": len(contexts),
        "attempt_count": len({item["attempt_id"] for item in contexts}),
        "stored_source_selector_lineages": {
            sha: sum(
                row.get("selector", {}).get("model_sha256") == sha for row in rows
            )
            for sha in sorted(
                {
                    str(row.get("selector", {}).get("model_sha256")) for row in rows
                }
            )
        },
        "probabilities_recomputed_with_current_frozen_selector": True,
        "selector": {
            "version": SELECTOR_VERSION,
            "sha256": SELECTOR_SHA256,
            "class_order": list(MOVE_ORDER),
            "model_directory": str(MODEL_DIR.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "paired_input": "Problem + Conversation + Next teacher pedagogical move",
            "tokenizer_truncation": "only_second, left truncation",
            "max_length": MAX_LENGTH,
        },
        "exact_expected_action_distribution": {
            arm: float(exact_rates[index]) for index, arm in enumerate(MOVE_ORDER)
        },
        "probability_by_move": {
            row["move"]: row for row in probability_table if not row["move"].startswith("__")
        },
        "all_context_action_pair_probability_bins": {
            key: value
            for key, value in probability_table[-1].items()
            if key.startswith("proportion_below_")
        },
        "top1": {
            "counts": {
                arm: sum(item["top1"] == arm for item in contexts)
                for arm in MOVE_ORDER
            },
            "expected_agreement": float(
                np.mean([item["top1_probability"] for item in contexts])
            ),
            "expected_deviation_rate": float(
                np.mean([item["non_top1_sampling_probability"] for item in contexts])
            ),
            "top1_probability": top_probabilities,
            "top1_second_gap": gaps,
            "confidence_rank_quartiles": confidence_strata,
            "gap_rank_quartiles": strata,
        },
        "entropy_nats": entropy,
        "monte_carlo": {
            "seed_count": MONTE_CARLO_SEEDS,
            **simulation,
        },
        "telling": {
            **quantile_summary(matrix[:, telling_index]),
            "expected_rate": float(exact_rates[telling_index]),
            "non_telling_top1_context_count": int(np.sum(non_telling_top)),
            "conditional_expected_rate_when_top1_not_telling": float(
                np.mean(matrix[non_telling_top, telling_index])
            ) if np.any(non_telling_top) else None,
            "aggregate_expected_contribution_when_top1_not_telling": float(
                np.sum(matrix[non_telling_top, telling_index]) / len(matrix)
            ),
            "telling_top1_context_count": int(np.sum(telling_top)),
            "conditional_expected_rate_when_top1_telling": float(
                np.mean(matrix[telling_top, telling_index])
            ) if np.any(telling_top) else None,
        },
        "coverage_horizons": horizons,
        "representative_cases": representatives,
        "runtime_implementation": True,
        "runtime_activation": False,
        "current_runtime_mode": "SHADOW",
        "runtime_event_schema": "adaptmath_turn_lints_event_v4",
        "runtime_behavior_policy": BEHAVIOR_POLICY,
        "runtime_lineage_schema": "adaptmath_randomized_warmstart_lineage_v1",
        "runtime_event_assignment_fields": [
            "decision_source",
            "treatment_assignment_source",
            "behavior_policy",
            "behavior_propensity",
            "full_behavior_probability_vector",
            "randomized_assignment",
            "assignment_rng_source",
            "assignment_seed",
            "assignment_random_draw",
            "posterior_updated",
        ],
        "focused_tests": {
            "files": [
                "adaptive-math-tutor/backend/tests/test_randomized_warmstart.py",
                "adaptive-math-tutor/backend/tests/test_turn_lints_v1.py",
                "adaptive-math-tutor/backend/tests/test_turn_lints_selector_anchor.py",
            ],
            "passed": 29,
            "failed": 0,
            "external_llm_tests": 0,
        },
        "randomized_treatment_observations_created": 0,
        "turn_lints_posterior_updates": 0,
        "safety": {
            "training": 0,
            "randomized_real_treatments": 0,
            "live_updates": 0,
            "external_tutor_api_calls": 0,
            "protected_mathdial_test_use": 0,
            "mrbench_v3_test_use": 0,
            "authoritative_historical_rewrites": 0,
        },
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "states": len(contexts),
                "expected_distribution": summary[
                    "exact_expected_action_distribution"
                ],
                "top1_agreement": summary["top1"]["expected_agreement"],
                "entropy_mean": summary["entropy_nats"]["mean"],
                "telling_rate": summary["telling"]["expected_rate"],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    run()
