"""Evaluate the recovered final integrity-filtered adaptive-policy table.

The input is the canonical 67-column derived CSV.  This is evaluation-only:
no selector, BKT state, runtime architecture, or raw turn ledger is mutated.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTPUT_DIR / "figures"
WORKSPACE_ROOT = OUTPUT_DIR.parents[2]
CANONICAL_DATASET = (
    WORKSPACE_ROOT
    / "pedagogical-move-selection"
    / "results"
    / "turn_lints_final_architecture_selection_v1"
    / "derived_analysis_dataset.csv"
)

MOVES = ("generic", "probing", "focus", "telling")
BLOCK_ORDER = ("S", "K", "L", "H", "Q")
BLOCK_FEATURES: dict[str, tuple[str, ...]] = {
    "S": tuple(f"selector_p_{move}" for move in MOVES),
    "K": (
        "mastery_before",
        "previous_mastery_delta",
        "previous_mastery_delta_missing",
    ),
    "L": (
        "previous_reasoning_probability",
        "previous_uncertainty_probability",
        "previous_clarification_probability",
        "previous_learner_signals_missing",
    ),
    "H": (
        *(f"previous_move_{move}" for move in MOVES),
        "previous_move_missing",
        "prior_tutor_turn_count",
    ),
    "Q": (
        "previous_mistake_identification",
        "previous_mistake_location",
        "previous_providing_guidance",
        "previous_actionability",
        "previous_mrb1_missing",
    ),
}
CONTEXTS: tuple[tuple[str, str | None], ...] = (
    ("Action only", None),
    ("S", "S"),
    ("S+K", "S+K"),
    ("S+L", "S+L"),
    ("S+H", "S+H"),
    ("S+Q", "S+Q"),
    ("S+K+L", "S+K+L"),
    ("S+K+H", "S+K+H"),
    ("S+K+Q", "S+K+Q"),
    ("S+K+L+H", "S+K+L+H"),
    ("Full S+K+L+H+Q", "S+K+L+H+Q"),
)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260829

EXPECTED_COLUMNS = (
    "action_event_id", "attempt_id", "learner_id", "skill_id",
    "turn_index_within_attempt", "relative_attempt_position", "actual_move",
    "runtime_mode", "selector_mode", "policy_lineage", "decision_source",
    "behavior_policy", "randomized_assignment", "behavior_propensity",
    "full_behavior_probability_vector", "mastery_before", "mastery_after",
    "raw_mastery_delta", "reward_raw_delta", "reward_headroom_normalized",
    "reward_provenance_at_collection", "current_mrb1_mistake_identification",
    "current_mrb1_mistake_location", "current_mrb1_providing_guidance",
    "current_mrb1_actionability", "resolution_status", "resolver_event_id",
    "observation_source", "valid_linked_bkt_outcome",
    "pre_action_context_representable", "include_in_reward_experiment",
    "include_in_context_experiment", "exclusion_reason",
    "no_knowledge_evidence_category", "mastery_state_contaminated",
    "stored_runtime_context", "learner_response_text", "evaluator_correctness",
    "evaluator_reason", "selector_p_generic", "selector_p_probing",
    "selector_p_focus", "selector_p_telling", "previous_mastery_delta",
    "previous_mastery_delta_missing", "previous_reasoning_probability",
    "previous_uncertainty_probability", "previous_clarification_probability",
    "previous_learner_signals_missing", "previous_move_generic",
    "previous_move_probing", "previous_move_focus", "previous_move_telling",
    "previous_move_missing", "prior_tutor_turn_count",
    "previous_mistake_identification", "previous_mistake_location",
    "previous_providing_guidance", "previous_actionability",
    "previous_mrb1_missing", "S_real_observed", "K_real_observed",
    "L_real_observed", "H_real_observed", "Q_real_observed", "mastery_band",
    "position_band",
)
CONTEXT_FEATURES = tuple(
    feature for block in BLOCK_ORDER for feature in BLOCK_FEATURES[block]
)
INDICATOR_ASSOCIATES = {
    "previous_mastery_delta_missing": ("previous_mastery_delta",),
    "previous_learner_signals_missing": BLOCK_FEATURES["L"][:-1],
    "previous_move_missing": BLOCK_FEATURES["H"][:4],
    "previous_mrb1_missing": BLOCK_FEATURES["Q"][:-1],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_blocks(preset: str) -> tuple[str, ...]:
    supplied = tuple(preset.split("+"))
    if not supplied or not set(supplied) <= set(BLOCK_ORDER):
        raise ValueError(f"Invalid context preset: {preset}")
    return tuple(block for block in BLOCK_ORDER if block in supplied)


def features_for(preset: str | None) -> tuple[str, ...]:
    if preset is None:
        return ()
    return tuple(
        feature for block in canonical_blocks(preset) for feature in BLOCK_FEATURES[block]
    )


def headroom_reward(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    delta = after - before
    result = np.zeros_like(delta, dtype=float)
    positive = delta > 0.0
    negative = delta < 0.0
    result[positive] = delta[positive] / (1.0 - before[positive])
    result[negative] = delta[negative] / before[negative]
    return np.clip(result, -1.0, 1.0)


def count_mismatches(mask: Iterable[bool]) -> int:
    return int(np.count_nonzero(np.asarray(list(mask), dtype=bool)))


def run_preflight(frame: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []

    def check(name: str, passed: bool) -> None:
        if not passed:
            failures.append(name)

    check("exact_67_column_schema_and_order", tuple(frame.columns) == EXPECTED_COLUMNS)
    check("turn_count_713", len(frame) == 713)
    check("learner_count_31", frame["learner_id"].nunique() == 31)
    check("attempt_count_92", frame["attempt_id"].nunique() == 92)
    check("skill_count_22", frame["skill_id"].nunique() == 22)

    action_counts = {
        move: int((frame["actual_move"] == move).sum()) for move in MOVES
    }
    check("exact_action_domain", set(frame["actual_move"].dropna()) == set(MOVES))
    check(
        "expected_action_counts",
        action_counts == {"generic": 168, "probing": 185, "focus": 212, "telling": 148},
    )

    core_ids = ("action_event_id", "attempt_id", "learner_id", "skill_id")
    null_core_ids = {column: int(frame[column].isna().sum()) for column in core_ids}
    duplicate_counts = {
        "action_event_id": int(frame["action_event_id"].duplicated().sum()),
        "resolver_event_id_nonnull": int(
            frame.loc[frame["resolver_event_id"].notna(), "resolver_event_id"].duplicated().sum()
        ),
        "attempt_id_plus_turn_index": int(
            frame.duplicated(["attempt_id", "turn_index_within_attempt"]).sum()
        ),
        "exact_rows": int(frame.duplicated().sum()),
    }
    check("no_null_core_ids", not any(null_core_ids.values()))
    check("no_duplicate_ids_or_rows", not any(duplicate_counts.values()))

    numeric_context = frame.loc[:, CONTEXT_FEATURES].apply(pd.to_numeric, errors="coerce")
    nonfinite_context = {
        column: int((~np.isfinite(numeric_context[column].to_numpy(float))).sum())
        for column in CONTEXT_FEATURES
    }
    check("finite_complete_context", not any(nonfinite_context.values()))

    indicator_summary: dict[str, dict[str, int]] = {}
    zero_fill_violations: dict[str, int] = {}
    for indicator, associated in INDICATOR_ASSOCIATES.items():
        values = numeric_context[indicator].to_numpy(float)
        indicator_summary[indicator] = {
            "observed_0": int((values == 0.0).sum()),
            "missing_1": int((values == 1.0).sum()),
            "invalid_other": int((~np.isin(values, (0.0, 1.0))).sum()),
        }
        missing = values == 1.0
        zero_fill_violations[indicator] = int(
            np.count_nonzero(~np.isclose(numeric_context.loc[missing, associated], 0.0).all(axis=1))
        )
    check(
        "binary_missingness_indicators",
        all(item["invalid_other"] == 0 for item in indicator_summary.values()),
    )
    check("missing_values_are_zero_filled", not any(zero_fill_violations.values()))

    l_observed = int((numeric_context["previous_learner_signals_missing"] == 0.0).sum())
    l_missing = int((numeric_context["previous_learner_signals_missing"] == 1.0).sum())
    l_coverage = l_observed / len(frame)
    check("L_coverage_approximately_92_percent", 0.90 <= l_coverage <= 0.94)

    probabilities = numeric_context.loc[:, BLOCK_FEATURES["S"]].to_numpy(float)
    check("selector_probabilities_in_unit_interval", bool(np.all((probabilities >= 0) & (probabilities <= 1))))
    check("selector_probabilities_sum_to_one", bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)))
    for column in (*BLOCK_FEATURES["L"][:-1], *BLOCK_FEATURES["Q"][:-1], "mastery_before"):
        values = numeric_context[column].to_numpy(float)
        check(f"unit_interval_{column}", bool(np.all((values >= 0) & (values <= 1))))
    delta_values = numeric_context["previous_mastery_delta"].to_numpy(float)
    check("previous_mastery_delta_in_signed_unit_interval", bool(np.all((delta_values >= -1) & (delta_values <= 1))))

    before = pd.to_numeric(frame["mastery_before"], errors="coerce").to_numpy(float)
    after = pd.to_numeric(frame["mastery_after"], errors="coerce").to_numpy(float)
    raw = pd.to_numeric(frame["reward_raw_delta"], errors="coerce").to_numpy(float)
    raw_mastery = pd.to_numeric(frame["raw_mastery_delta"], errors="coerce").to_numpy(float)
    headroom = pd.to_numeric(frame["reward_headroom_normalized"], errors="coerce").to_numpy(float)
    check("finite_reward_inputs", bool(np.isfinite(np.column_stack([before, after, raw, headroom])).all()))
    check("mastery_values_in_unit_interval", bool(np.all((before >= 0) & (before <= 1) & (after >= 0) & (after <= 1))))
    reward_tolerance = 2e-12
    raw_equation_max_error = float(np.max(np.abs(raw - (after - before))))
    raw_alias_max_error = float(np.max(np.abs(raw - raw_mastery)))
    headroom_max_error = float(np.max(np.abs(headroom - headroom_reward(before, after))))
    check("raw_reward_equation", raw_equation_max_error <= reward_tolerance)
    check("raw_reward_alias", raw_alias_max_error <= reward_tolerance)
    check("headroom_reward_equation", headroom_max_error <= 2e-11)

    required_true_flags = (
        "valid_linked_bkt_outcome",
        "pre_action_context_representable",
        "include_in_reward_experiment",
        "include_in_context_experiment",
    )
    false_flag_counts = {
        column: int((frame[column] != True).sum())  # noqa: E712
        for column in required_true_flags
    }
    check("all_rows_scientifically_included", not any(false_flag_counts.values()))

    temporal = {
        "stored_runtime_context_rows": int(frame["stored_runtime_context"].eq(True).sum()),  # noqa: E712
        "reconstructed_context_rows": int(frame["stored_runtime_context"].ne(True).sum()),  # noqa: E712
        "attempt_learner_mismatches": 0,
        "attempt_skill_mismatches": 0,
        "turn_start_mismatches": 0,
        "turn_contiguity_mismatches": 0,
        "relative_position_mismatches": 0,
        "prior_tutor_turn_count_mismatches": 0,
        "previous_delta_mismatches": 0,
        "previous_move_mismatches": 0,
        "previous_Q_mismatches": 0,
    }
    move_columns = BLOCK_FEATURES["H"][:4]
    previous_q = BLOCK_FEATURES["Q"][:-1]
    current_q = (
        "current_mrb1_mistake_identification",
        "current_mrb1_mistake_location",
        "current_mrb1_providing_guidance",
        "current_mrb1_actionability",
    )
    for _, group in frame.groupby("attempt_id", sort=False):
        group = group.sort_values("turn_index_within_attempt").reset_index(drop=True)
        temporal["attempt_learner_mismatches"] += int(group["learner_id"].nunique() != 1)
        temporal["attempt_skill_mismatches"] += int(group["skill_id"].nunique() != 1)
        turns = group["turn_index_within_attempt"].astype(int).tolist()
        temporal["turn_start_mismatches"] += int(turns[0] != 1)
        temporal["turn_contiguity_mismatches"] += int(turns != list(range(1, len(group) + 1)))
        # The recovered final table uses endpoint-normalized position: the
        # first retained action is 0 and the last is 1.
        expected_positions = (
            (group["turn_index_within_attempt"].to_numpy(float) - 1.0)
            / max(1, len(group) - 1)
        )
        temporal["relative_position_mismatches"] += int(
            not np.allclose(group["relative_attempt_position"].to_numpy(float), expected_positions, atol=1e-12)
        )
        temporal["prior_tutor_turn_count_mismatches"] += int(
            not np.allclose(
                group["prior_tutor_turn_count"].to_numpy(float),
                group["turn_index_within_attempt"].to_numpy(float) - 1.0,
                atol=1e-12,
            )
        )
        for index, row in group.iterrows():
            # The canonical derivation treats the recorded decision-time context as
            # authoritative when present.  Reconstructing K/H/Q from this flat
            # table is valid only for historical rows without stored context; the
            # current outcome/quality columns may have been re-resolved later.
            stored_context = bool(row["stored_runtime_context"])
            if index == 0:
                if not stored_context and row["previous_mastery_delta_missing"] != 1.0:
                    temporal["previous_delta_mismatches"] += 1
                if not stored_context and row["previous_move_missing"] != 1.0:
                    temporal["previous_move_mismatches"] += 1
                continue
            prior = group.iloc[index - 1]
            if not stored_context and (
                row["previous_mastery_delta_missing"] != 0.0 or not math.isclose(
                float(row["previous_mastery_delta"]), float(prior["reward_raw_delta"]), abs_tol=1e-12
                )
            ):
                temporal["previous_delta_mismatches"] += 1
            expected_move = np.asarray([float(prior["actual_move"] == move) for move in MOVES])
            if not stored_context and (
                row["previous_move_missing"] != 0.0 or not np.allclose(
                row.loc[list(move_columns)].to_numpy(float), expected_move, atol=1e-12
                )
            ):
                temporal["previous_move_mismatches"] += 1
            if not stored_context and row["previous_mrb1_missing"] == 0.0:
                if not np.allclose(
                    row.loc[list(previous_q)].to_numpy(float),
                    prior.loc[list(current_q)].to_numpy(float),
                    atol=1e-12,
                ):
                    temporal["previous_Q_mismatches"] += 1
    temporal_mismatch_keys = tuple(key for key in temporal if key.endswith("_mismatches"))
    check("temporally_ordered_context", not any(temporal[key] for key in temporal_mismatch_keys))

    forbidden_tokens = (
        "mastery_after", "current_", "reward", "delta_mastery", "learner_response",
        "evaluator", "outcome", "correctness", "tutor_response",
    )
    leakage_features = [
        feature for feature in CONTEXT_FEATURES
        if any(token in feature.casefold() for token in forbidden_tokens)
    ]
    check("no_post_action_or_current_action_context_features", not leakage_features)

    missingness = pd.DataFrame({
        "column": frame.columns,
        "dtype_after_csv_load": [str(frame[column].dtype) for column in frame.columns],
        "null_count": [int(frame[column].isna().sum()) for column in frame.columns],
        "null_fraction": [float(frame[column].isna().mean()) for column in frame.columns],
    })
    missingness.to_csv(OUTPUT_DIR / "missingness_summary.csv", index=False)

    checks = {
        name: name not in failures
        for name in (
            "exact_67_column_schema_and_order", "turn_count_713", "learner_count_31",
            "attempt_count_92", "skill_count_22", "exact_action_domain",
            "expected_action_counts", "no_null_core_ids", "no_duplicate_ids_or_rows",
            "finite_complete_context", "binary_missingness_indicators",
            "missing_values_are_zero_filled", "L_coverage_approximately_92_percent",
            "selector_probabilities_in_unit_interval", "selector_probabilities_sum_to_one",
            "previous_mastery_delta_in_signed_unit_interval", "finite_reward_inputs",
            "mastery_values_in_unit_interval", "raw_reward_equation", "raw_reward_alias",
            "headroom_reward_equation", "all_rows_scientifically_included",
            "temporally_ordered_context", "no_post_action_or_current_action_context_features",
        )
    }
    for column in (*BLOCK_FEATURES["L"][:-1], *BLOCK_FEATURES["Q"][:-1], "mastery_before"):
        checks[f"unit_interval_{column}"] = f"unit_interval_{column}" not in failures

    report = {
        "status": "PASS" if not failures else "FAIL",
        "canonical_path": str(CANONICAL_DATASET.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "canonical_sha256": sha256(CANONICAL_DATASET),
        "inventory": {
            "turns": int(len(frame)),
            "learners": int(frame["learner_id"].nunique()),
            "attempts": int(frame["attempt_id"].nunique()),
            "skills": int(frame["skill_id"].nunique()),
            "action_counts": action_counts,
        },
        "L_coverage": {
            "observed": l_observed,
            "missing": l_missing,
            "total": int(len(frame)),
            "fraction": float(l_coverage),
            "percent": float(100.0 * l_coverage),
        },
        "semantic_missingness": indicator_summary,
        "zero_fill_violations": zero_fill_violations,
        "null_core_ids": null_core_ids,
        "nonfinite_context": nonfinite_context,
        "duplicates": duplicate_counts,
        "temporal_checks": temporal,
        "leakage_or_post_action_context_features": leakage_features,
        "reward_equation_max_absolute_error": {
            "raw": raw_equation_max_error,
            "raw_alias": raw_alias_max_error,
            "headroom": headroom_max_error,
        },
        "false_required_flag_counts": false_flag_counts,
        "checks": checks,
        "failures": failures,
    }
    json_dump(OUTPUT_DIR / "preflight_audit.json", report)
    return report


def reward_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = frame.loc[frame["include_in_reward_experiment"] == True].copy()  # noqa: E712
    rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for label, column in (
        ("RAW_BKT_DELTA", "reward_raw_delta"),
        ("HEADROOM_NORMALIZED", "reward_headroom_normalized"),
    ):
        values = usable[column].astype(float)
        result = spearmanr(values.abs(), usable["mastery_before"].astype(float))
        row = {
            "reward": label,
            "column": column,
            "N": int(values.notna().sum()),
            "min": float(values.min()),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "max": float(values.max()),
            "spearman_rho_abs_reward_vs_mastery_before": float(result.statistic),
            "abs_spearman_rho": float(abs(result.statistic)),
            "spearman_p_value": float(result.pvalue),
        }
        rows.append(row)
        correlations[label] = {
            "rho": row["spearman_rho_abs_reward_vs_mastery_before"],
            "absolute_rho": row["abs_spearman_rho"],
            "p_value": row["spearman_p_value"],
            "N": row["N"],
        }
    summary = pd.DataFrame(rows)
    detail = {"rewards": rows, "correlations": correlations}
    return summary, detail


def action_onehot(actions: np.ndarray) -> np.ndarray:
    return np.column_stack([actions == move for move in MOVES]).astype(float)


def build_design(
    frame: pd.DataFrame,
    indices: np.ndarray,
    context_features: tuple[str, ...],
    *,
    scaler: StandardScaler | None = None,
    fit_scaler: bool,
) -> tuple[np.ndarray, StandardScaler | None]:
    actions = frame.iloc[indices]["actual_move"].to_numpy()
    action_matrix = action_onehot(actions)
    if not context_features:
        return action_matrix, None
    context = frame.iloc[indices].loc[:, context_features].to_numpy(float)
    if fit_scaler:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(context)
    else:
        if scaler is None:
            raise RuntimeError("A fitted scaler is required for held-out context.")
        standardized = scaler.transform(context)
    interactions = np.hstack(
        [standardized * action_matrix[:, [index]] for index in range(len(MOVES))]
    )
    return np.hstack([action_matrix, standardized, interactions]), scaler


def choose_alpha(
    training_frame: pd.DataFrame,
    target_column: str,
    context_features: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    groups = training_frame["learner_id"].to_numpy()
    target = training_frame[target_column].to_numpy(float)
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, len(np.unique(groups))))
    scores = {alpha: [] for alpha in RIDGE_GRID}
    for train, validation in splitter.split(training_frame, target, groups):
        x_train, scaler = build_design(
            training_frame, train, context_features, fit_scaler=True
        )
        x_validation, _ = build_design(
            training_frame, validation, context_features, scaler=scaler, fit_scaler=False
        )
        for alpha in RIDGE_GRID:
            model = Ridge(alpha=alpha)
            model.fit(x_train, target[train])
            scores[alpha].append(
                float(mean_absolute_error(target[validation], model.predict(x_validation)))
            )
    means = {str(alpha): float(np.mean(values)) for alpha, values in scores.items()}
    selected = min(RIDGE_GRID, key=lambda alpha: (means[str(alpha)], alpha))
    return float(selected), means


def run_ablation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = frame.loc[frame["include_in_context_experiment"] == True].copy()  # noqa: E712
    usable = usable.reset_index(drop=True)
    groups = usable["learner_id"].to_numpy()
    target = usable["reward_headroom_normalized"].to_numpy(float)
    splitter = GroupKFold(n_splits=OUTER_FOLDS)
    split_indices = list(splitter.split(usable, target, groups))
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for order, (label, preset) in enumerate(CONTEXTS):
        context_features = features_for(preset)
        out_of_fold = np.full(len(usable), np.nan, dtype=float)
        candidate_folds: list[dict[str, Any]] = []
        for fold, (train, test) in enumerate(split_indices, 1):
            outer_training = usable.iloc[train].reset_index(drop=True)
            alpha, inner_scores = choose_alpha(
                outer_training, "reward_headroom_normalized", context_features
            )
            x_train, scaler = build_design(
                usable, train, context_features, fit_scaler=True
            )
            x_test, _ = build_design(
                usable, test, context_features, scaler=scaler, fit_scaler=False
            )
            model = Ridge(alpha=alpha)
            model.fit(x_train, target[train])
            predicted = model.predict(x_test)
            out_of_fold[test] = predicted
            result = {
                "candidate_order": order,
                "context": label,
                "preset": preset or "ACTION_ONLY",
                "context_dimension": len(context_features),
                "design_dimension": int(x_train.shape[1]),
                "fold": fold,
                "train_N": int(len(train)),
                "test_N": int(len(test)),
                "train_learners": int(usable.iloc[train]["learner_id"].nunique()),
                "test_learners": int(usable.iloc[test]["learner_id"].nunique()),
                "test_learner_ids": json.dumps(
                    sorted(usable.iloc[test]["learner_id"].astype(str).unique().tolist()),
                    separators=(",", ":"),
                ),
                "selected_alpha": alpha,
                "inner_mean_MAE_by_alpha": json.dumps(inner_scores, sort_keys=True),
                "MAE": float(mean_absolute_error(target[test], predicted)),
                "RMSE": float(mean_squared_error(target[test], predicted) ** 0.5),
                "R2": float(r2_score(target[test], predicted)),
            }
            candidate_folds.append(result)
            fold_rows.append(result)
        if not np.isfinite(out_of_fold).all():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {label}.")
        folds = pd.DataFrame(candidate_folds)
        summary_rows.append({
            "candidate_order": order,
            "context": label,
            "preset": preset or "ACTION_ONLY",
            "context_dimension": len(context_features),
            "ordered_context_features": json.dumps(context_features, separators=(",", ":")),
            "N": int(len(usable)),
            "learner_count": int(usable["learner_id"].nunique()),
            "attempt_count": int(usable["attempt_id"].nunique()),
            "fold_count": int(len(folds)),
            "MAE_mean": float(folds["MAE"].mean()),
            "MAE_SD": float(folds["MAE"].std(ddof=1)),
            "MAE_SE": float(folds["MAE"].std(ddof=1) / math.sqrt(len(folds))),
            "RMSE_mean": float(folds["RMSE"].mean()),
            "RMSE_SD": float(folds["RMSE"].std(ddof=1)),
            "RMSE_SE": float(folds["RMSE"].std(ddof=1) / math.sqrt(len(folds))),
            "R2_mean": float(folds["R2"].mean()),
            "R2_SD": float(folds["R2"].std(ddof=1)),
            "R2_SE": float(folds["R2"].std(ddof=1) / math.sqrt(len(folds))),
            "OOF_MAE": float(mean_absolute_error(target, out_of_fold)),
            "OOF_RMSE": float(mean_squared_error(target, out_of_fold) ** 0.5),
            "OOF_R2": float(r2_score(target, out_of_fold)),
            "selected_alphas": json.dumps(folds["selected_alpha"].tolist(), separators=(",", ":")),
        })
        prediction_rows.append(pd.DataFrame({
            "context": label,
            "action_event_id": usable["action_event_id"],
            "learner_id": usable["learner_id"],
            "attempt_id": usable["attempt_id"],
            "actual": target,
            "predicted": out_of_fold,
            "absolute_error": np.abs(target - out_of_fold),
            "squared_error": np.square(target - out_of_fold),
        }))
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True)


def learner_bootstrap(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = predictions.loc[predictions["context"] == "S+K+L"].reset_index(drop=True)
    right = predictions.loc[predictions["context"] == "S"].reset_index(drop=True)
    identity_columns = ["action_event_id", "learner_id", "attempt_id", "actual"]
    if not left[identity_columns].equals(right[identity_columns]):
        raise RuntimeError("S+K+L and S out-of-fold identities do not match.")
    learners = left["learner_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    replicate_rows: list[dict[str, Any]] = []
    left_actual = left["actual"].to_numpy(float)
    left_predicted = left["predicted"].to_numpy(float)
    right_predicted = right["predicted"].to_numpy(float)
    learner_values = left["learner_id"].to_numpy()
    for replicate in range(1, BOOTSTRAP_REPLICATES + 1):
        sampled = rng.choice(learners, size=len(learners), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(learner_values == learner) for learner in sampled]
        )
        actual = left_actual[indices]
        skl_mae = mean_absolute_error(actual, left_predicted[indices])
        s_mae = mean_absolute_error(actual, right_predicted[indices])
        skl_rmse = mean_squared_error(actual, left_predicted[indices]) ** 0.5
        s_rmse = mean_squared_error(actual, right_predicted[indices]) ** 0.5
        replicate_rows.append({
            "replicate": replicate,
            "sampled_learner_clusters": int(len(sampled)),
            "sampled_turn_rows": int(len(indices)),
            "delta_MAE_S_K_L_minus_S": float(skl_mae - s_mae),
            "delta_RMSE_S_K_L_minus_S": float(skl_rmse - s_rmse),
        })
    replicates = pd.DataFrame(replicate_rows)
    point_values = {
        "MAE": float(
            mean_absolute_error(left_actual, left_predicted)
            - mean_absolute_error(left_actual, right_predicted)
        ),
        "RMSE": float(
            mean_squared_error(left_actual, left_predicted) ** 0.5
            - mean_squared_error(left_actual, right_predicted) ** 0.5
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric in ("MAE", "RMSE"):
        column = f"delta_{metric}_S_K_L_minus_S"
        values = replicates[column].to_numpy(float)
        rows.append({
            "comparison": "S+K+L minus S",
            "metric": metric,
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "learner_id",
            "point_delta": point_values[metric],
            "bootstrap_mean_delta": float(values.mean()),
            "bootstrap_SE": float(values.std(ddof=1)),
            "CI_2.5": float(np.quantile(values, 0.025)),
            "CI_97.5": float(np.quantile(values, 0.975)),
            "probability_improvement": float((values < 0.0).mean()),
        })
    return pd.DataFrame(rows), replicates


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(frame: pd.DataFrame, rewards: pd.DataFrame, ablation: pd.DataFrame) -> None:
    usable = frame.loc[frame["include_in_reward_experiment"] == True]  # noqa: E712
    colors = ("#4C78A8", "#F28E2B")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, (label, column), color in zip(
        axes,
        (("Raw BKT delta", "reward_raw_delta"), ("HEADROOM_NORMALIZED", "reward_headroom_normalized")),
        colors,
        strict=True,
    ):
        values = usable[column].to_numpy(float)
        axis.hist(values, bins=30, color=color, alpha=0.85, edgecolor="white")
        axis.axvline(np.median(values), color="#222222", linestyle="--", linewidth=1.2, label="Median")
        axis.axvline(np.mean(values), color="#B22222", linestyle=":", linewidth=1.4, label="Mean")
        axis.set_title(label)
        axis.set_xlabel("Reward")
        axis.set_ylabel("Turns")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Adaptive-policy reward distributions")
    fig.tight_layout()
    save_figure(fig, "reward_distribution_comparison")

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    values = rewards["abs_spearman_rho"].to_numpy(float)
    labels = ["Raw BKT delta", "HEADROOM_NORMALIZED"]
    bars = axis.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}", ha="center", va="bottom")
    axis.set_ylabel(r"Absolute Spearman $|\rho|$")
    axis.set_title(r"Reward-magnitude dependence on mastery before")
    axis.set_ylim(0, max(0.1, values.max() * 1.18))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "reward_magnitude_dependence")

    metrics = (("MAE", "lower is better"), ("RMSE", "lower is better"), ("R2", "higher is better"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
    positions = np.arange(len(ablation))
    for axis, (metric, subtitle) in zip(axes, metrics, strict=True):
        means = ablation[f"{metric}_mean"].to_numpy(float)
        errors = ablation[f"{metric}_SE"].to_numpy(float)
        axis.errorbar(means, positions, xerr=errors, fmt="o", color="#3B6EA8", ecolor="#8AA9C7", capsize=3)
        axis.set_title(f"{metric} ({subtitle})")
        axis.axvline(means.min() if metric != "R2" else means.max(), color="#B22222", linestyle="--", alpha=0.6)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(positions, ablation["context"].tolist())
    axes[0].invert_yaxis()
    fig.suptitle("Learner-grouped pre-action context ablation: mean ± SE across 5 folds")
    fig.tight_layout()
    save_figure(fig, "learner_grouped_context_ablation")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CANONICAL_DATASET.exists():
        raise FileNotFoundError(f"Canonical derived dataset is missing: {CANONICAL_DATASET}")
    frame = pd.read_csv(CANONICAL_DATASET)
    audit = run_preflight(frame)
    print("PREFLIGHT AUDIT")
    print(json.dumps({
        "status": audit["status"],
        "inventory": audit["inventory"],
        "L_coverage": audit["L_coverage"],
        "semantic_missingness": audit["semantic_missingness"],
        "duplicates": audit["duplicates"],
        "leakage_or_post_action_context_features": audit["leakage_or_post_action_context_features"],
        "temporal_checks": audit["temporal_checks"],
        "failures": audit["failures"],
    }, indent=2))
    if audit["status"] != "PASS":
        print("FATAL: preflight failed; no experiment was run.", file=sys.stderr)
        return 2

    rewards, reward_detail = reward_summary(frame)
    rewards.to_csv(OUTPUT_DIR / "reward_summary.csv", index=False)
    json_dump(OUTPUT_DIR / "reward_summary.json", reward_detail)

    ablation, folds, predictions = run_ablation(frame)
    ablation.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "ablation_fold_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "ablation_oof_predictions.csv", index=False)

    bootstrap, bootstrap_replicates = learner_bootstrap(predictions)
    bootstrap.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)
    bootstrap_replicates.to_csv(OUTPUT_DIR / "bootstrap_replicates.csv", index=False)
    make_plots(frame, rewards, ablation)

    best = ablation.sort_values(["MAE_mean", "RMSE_mean", "candidate_order"]).iloc[0]
    result = {
        "analysis_version": "adaptive_policy_recovered_final_v1",
        "input": {
            "canonical_path": audit["canonical_path"],
            "sha256": audit["canonical_sha256"],
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        },
        "preflight": audit,
        "reward_experiment": reward_detail,
        "ablation_configuration": {
            "outer_cv": "5-fold GroupKFold by learner_id",
            "inner_cv": "4-fold GroupKFold by learner_id within each outer training fold",
            "ridge_alpha_grid": list(RIDGE_GRID),
            "inner_selection_metric": "MAE",
            "standardization": "context only; fit on each training fold only",
            "design": "action main effects + context main effects + action-by-context interactions",
            "target": "reward_headroom_normalized",
            "contexts": [label for label, _ in CONTEXTS],
        },
        "ablation_results": ablation.to_dict(orient="records"),
        "best_context_by_mean_MAE": {
            "context": str(best["context"]),
            "MAE_mean": float(best["MAE_mean"]),
            "MAE_SE": float(best["MAE_SE"]),
            "RMSE_mean": float(best["RMSE_mean"]),
            "RMSE_SE": float(best["RMSE_SE"]),
            "R2_mean": float(best["R2_mean"]),
            "R2_SE": float(best["R2_SE"]),
        },
        "bootstrap_S_K_L_minus_S": bootstrap.to_dict(orient="records"),
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "preflight_audit": "preflight_audit.json",
            "missingness_summary": "missingness_summary.csv",
            "reward_summary_csv": "reward_summary.csv",
            "reward_summary_json": "reward_summary.json",
            "ablation_summary": "ablation_summary.csv",
            "ablation_fold_results": "ablation_fold_results.csv",
            "ablation_oof_predictions": "ablation_oof_predictions.csv",
            "bootstrap_summary": "bootstrap_summary.csv",
            "bootstrap_replicates": "bootstrap_replicates.csv",
            "figures": [
                "figures/reward_distribution_comparison.png",
                "figures/reward_distribution_comparison.pdf",
                "figures/reward_magnitude_dependence.png",
                "figures/reward_magnitude_dependence.pdf",
                "figures/learner_grouped_context_ablation.png",
                "figures/learner_grouped_context_ablation.pdf",
            ],
        },
    }
    json_dump(OUTPUT_DIR / "final_results.json", result)

    raw = rewards.set_index("reward").loc["RAW_BKT_DELTA"]
    head = rewards.set_index("reward").loc["HEADROOM_NORMALIZED"]
    bootstrap_indexed = bootstrap.set_index("metric")
    print("FINAL RESULT")
    print(json.dumps({
        "N": int(len(frame)),
        "reward": {
            "raw": raw.to_dict(),
            "headroom_normalized": head.to_dict(),
        },
        "best_context": result["best_context_by_mean_MAE"],
        "bootstrap_S_K_L_minus_S": {
            "MAE": bootstrap_indexed.loc["MAE"].to_dict(),
            "RMSE": bootstrap_indexed.loc["RMSE"].to_dict(),
        },
        "final_results_json": str((OUTPUT_DIR / "final_results.json").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
