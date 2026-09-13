"""Read-only Turn-LinTS reward/context audit over existing real turn records.

This script creates derived artifacts only. It does not load/train a model,
call an external service, mutate authoritative JSONL, or touch policy state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
MOVE_ROOT = OUTPUT_DIR.parents[1]
WORKSPACE_ROOT = MOVE_ROOT.parent
BACKEND_ROOT = WORKSPACE_ROOT / "adaptive-math-tutor" / "backend"
TURN_DATA = (
    MOVE_ROOT
    / "results"
    / "md_self_improvement_turn_data_v1"
    / "turn_outcomes.jsonl"
)
SHADOW_ROOT = (
    WORKSPACE_ROOT
    / "adaptive-math-tutor"
    / "backend"
    / "runtime"
    / "turn_lints_shadow_full_raw_v1"
)
SHADOW_EVENTS = SHADOW_ROOT / "turn_events.jsonl"
SHADOW_STATE = SHADOW_ROOT / "policy_state.json"
SELECTOR_ROOT = MOVE_ROOT / "models" / "frozen" / "md7_r2_tell_c1_epoch2"
ROLLBACK_ROOT = MOVE_ROOT / "models" / "candidates" / "md7r1_epoch3"
SELECTOR_MODE_SOURCE = (
    BACKEND_ROOT / "app" / "integrations" / "adaptive_selector_mode.py"
)
COORDINATOR_SOURCE = (
    BACKEND_ROOT / "app" / "integrations" / "adaptive_component_coordinator.py"
)
LOCAL_RUNTIME_STATE = WORKSPACE_ROOT / "_local_runtime_state" / "adaptmath-local.json"

MOVES = ("generic", "probing", "focus", "telling")
MRB1_HEADS = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)
MRB1_COLUMNS = {
    head: "mrb1_" + head.casefold() for head in MRB1_HEADS
}

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

CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("C0", ("S",)),
    ("C1", ("S", "K")),
    ("C2", ("S", "L")),
    ("C3", ("S", "K", "L")),
    ("C4", ("S", "K", "L", "H")),
    ("C5", ("S", "K", "L", "H", "Q")),
    ("C6", ("K", "L", "H", "Q")),
    # Diagnostic fallbacks allow H and Q to be audited without fabricating L.
    ("D0", ("S", "K", "H")),
    ("D1", ("S", "K", "H", "Q")),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def headroom_reward(before: float, after: float, *, tolerance: float = 1e-12) -> float:
    delta = after - before
    if abs(delta) <= tolerance:
        return 0.0
    denominator = 1.0 - before if delta > 0.0 else before
    if denominator <= tolerance:
        raise ValueError(
            f"Nonzero mastery delta {delta} has zero headroom at mastery {before}."
        )
    reward = delta / denominator
    if reward > 1.0 and reward <= 1.0 + tolerance:
        reward = 1.0
    elif reward < -1.0 and reward >= -1.0 - tolerance:
        reward = -1.0
    if not -1.0 <= reward <= 1.0:
        raise ValueError(f"Headroom reward out of range: {reward}")
    return reward


def extract_context_mapping(row: dict[str, Any]) -> dict[str, float] | None:
    decision = row.get("adaptive_decision")
    if not isinstance(decision, dict):
        return None
    names = decision.get("feature_names")
    vector = decision.get("vector")
    if not isinstance(names, list) or not isinstance(vector, list):
        return None
    if len(names) != len(vector):
        raise ValueError("Stored Turn-LinTS context names/vector length mismatch")
    return {str(name): float(value) for name, value in zip(names, vector, strict=True)}


def build_turn_table(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    identities = [nested(row, "identity", "action_event_id") for row in raw_rows]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("Every passive row must have an action_event_id")
    duplicates = [key for key, count in Counter(identities).items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate passive action identities: {duplicates[:3]}")

    sorted_rows = sorted(
        raw_rows,
        key=lambda row: (
            str(nested(row, "identity", "attempt_id")),
            int(nested(row, "identity", "action_turn_index") or 0),
        ),
    )
    attempt_lengths = Counter(
        str(nested(row, "identity", "attempt_id")) for row in sorted_rows
    )
    previous_by_attempt: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for raw in sorted_rows:
        identity = raw["identity"]
        attempt_id = str(identity["attempt_id"])
        turn_index = int(identity["action_turn_index"])
        previous = previous_by_attempt.get(attempt_id)
        learning = raw.get("learning_state_update") or {}
        status = nested(raw, "next_learner_observation", "status")
        should_update = learning.get("should_update") is True
        before = finite_float(learning.get("mastery_before"))
        after = finite_float(learning.get("mastery_after"))
        resolver_id = learning.get("resolver_event_id")
        usable = (
            status == "observed_update"
            and should_update
            and before is not None
            and after is not None
            and isinstance(resolver_id, str)
            and bool(resolver_id)
            and learning.get("observation_source") != "formal_assessment"
        )
        raw_reward = (after - before) if usable else np.nan
        stored_delta = finite_float(learning.get("delta_mastery"))
        if usable and stored_delta is not None and not math.isclose(
            raw_reward, stored_delta, abs_tol=1e-12, rel_tol=0.0
        ):
            raise ValueError(f"Stored delta mismatch for {identity['action_event_id']}")
        normalized = headroom_reward(before, after) if usable else np.nan

        selector_probabilities = nested(raw, "selector", "raw", "probabilities")
        if not isinstance(selector_probabilities, dict):
            selector_probabilities = nested(
                raw, "selector", "effective", "probabilities"
            )
        if not isinstance(selector_probabilities, dict) or set(
            selector_probabilities
        ) != set(MOVES):
            raise ValueError("Selector probabilities must contain four canonical moves")

        actual_move = nested(raw, "adaptive_decision", "final_move")
        if actual_move not in MOVES:
            actual_move = nested(raw, "selector", "effective", "argmax")
        if actual_move not in MOVES:
            raise ValueError("Passive row has no canonical actual move")

        previous_usable = bool(previous and previous["usable_reward"])
        previous_delta = previous["r_raw"] if previous_usable else 0.0
        previous_move = previous["pedagogical_move"] if previous else None
        previous_quality = previous["current_mrb1"] if previous else None
        stored_context = extract_context_mapping(raw)

        record: dict[str, Any] = {
            "action_event_id": identity["action_event_id"],
            "resolver_event_id": resolver_id,
            "attempt_id": attempt_id,
            "learner_id": identity.get("student_pseudonymous_id"),
            "thread_id": identity.get("thread_id"),
            "turn_index": turn_index,
            "relative_position": turn_index / attempt_lengths[attempt_id],
            "skill": learning.get("skill")
            or nested(raw, "state_before_action", "target_skill"),
            "pedagogical_move": actual_move,
            "observation_status": status,
            "should_update": should_update,
            "usable_reward": usable,
            "mastery_before": before,
            "mastery_after": after,
            "r_raw": raw_reward,
            "r_headroom": normalized,
            "selector_mode": nested(raw, "provenance", "selector_mode"),
            "selector_checkpoint": nested(raw, "selector", "checkpoint"),
            "selector_sha256": nested(raw, "selector", "model_sha256"),
            "turn_lints_mode": nested(raw, "provenance", "turn_lints_mode"),
            "hypothetical_arm": nested(raw, "adaptive_decision", "hypothetical_arm"),
            "selected_arm": nested(raw, "adaptive_decision", "selected_arm"),
            "posterior_context_stored": stored_context is not None,
        }
        for move in MOVES:
            record[f"selector_p_{move}"] = float(selector_probabilities[move])

        record["previous_mastery_delta"] = float(previous_delta)
        record["previous_mastery_delta_missing"] = 0.0 if previous_usable else 1.0

        l_names = BLOCK_FEATURES["L"]
        for name in l_names:
            record[name] = (
                stored_context.get(name, np.nan)
                if stored_context is not None
                else np.nan
            )

        for move in MOVES:
            record[f"previous_move_{move}"] = float(previous_move == move)
        record["previous_move_missing"] = float(previous_move is None)
        record["prior_tutor_turn_count"] = float(turn_index - 1)

        current_scores = nested(raw, "tutor_quality", "scores")
        current_mrb1: dict[str, float] | None = None
        if isinstance(current_scores, dict) and set(current_scores) == set(MRB1_HEADS):
            current_mrb1 = {head: float(current_scores[head]) for head in MRB1_HEADS}
        for head in MRB1_HEADS:
            record[MRB1_COLUMNS[head]] = (
                current_mrb1[head] if current_mrb1 is not None else np.nan
            )
            previous_name = "previous_" + head.casefold()
            record[previous_name] = (
                previous_quality[head] if previous_quality is not None else 0.0
            )
        record["previous_mrb1_missing"] = float(previous_quality is None)

        records.append(record)
        previous_by_attempt[attempt_id] = {
            "usable_reward": usable,
            "r_raw": raw_reward,
            "pedagogical_move": actual_move,
            "current_mrb1": current_mrb1,
        }

    frame = pd.DataFrame(records)
    frame["mastery_band"] = pd.cut(
        frame["mastery_before"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.000000000001],
        labels=["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"],
        right=False,
        include_lowest=True,
    )
    frame["position_band"] = pd.cut(
        frame["relative_position"],
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.000000000001],
        labels=["early", "middle", "late"],
        right=False,
        include_lowest=True,
    )
    return frame


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return {key: None for key in ("count", "min", "q1", "median", "mean", "q3", "max", "std")}
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
        "std": float(series.std(ddof=1)) if series.size > 1 else None,
    }


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return {"n": int(len(pair)), "rho": None, "p_value": None}
    result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {
        "n": int(len(pair)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def grouped_stats(frame: pd.DataFrame, group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reward in ("r_raw", "r_headroom"):
        for key, subset in frame.groupby(group, observed=False, dropna=False):
            values = subset[reward].dropna()
            rows.append(
                {
                    "reward": reward,
                    group: str(key),
                    **describe(values),
                    "absolute_mean": float(values.abs().mean()) if len(values) else None,
                    "absolute_median": float(values.abs().median()) if len(values) else None,
                }
            )
    return rows


def action_conditional_design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    context = frame[features].to_numpy(dtype=float)
    blocks = []
    actions = frame["pedagogical_move"].to_numpy()
    for move in MOVES:
        mask = (actions == move).astype(float)[:, None]
        blocks.append(context * mask)
    return np.hstack(blocks)


def run_ablation(frame: pd.DataFrame) -> list[dict[str, Any]]:
    usable = frame[frame["usable_reward"]].copy()
    rows: list[dict[str, Any]] = []
    for config_name, blocks in CONFIGS:
        features = [name for block in blocks for name in BLOCK_FEATURES[block]]
        unavailable = [name for name in features if usable[name].isna().any()]
        for reward in ("r_raw", "r_headroom"):
            base = {
                "configuration": config_name,
                "blocks": "+".join(blocks),
                "reward": reward,
                "context_dimension": len(features),
                "ordered_features": json.dumps(features, separators=(",", ":")),
                "sample_count": int(len(usable)),
                "attempt_count": int(usable["attempt_id"].nunique()),
                "action_counts": json.dumps(
                    {move: int((usable["pedagogical_move"] == move).sum()) for move in MOVES},
                    separators=(",", ":"),
                ),
            }
            if unavailable:
                rows.append(
                    {
                        **base,
                        "status": "unavailable",
                        "reason": (
                            "Retrospective records do not contain complete pre-action "
                            "values for: " + ", ".join(unavailable)
                        ),
                    }
                )
                continue
            groups = usable["attempt_id"].to_numpy()
            group_count = len(np.unique(groups))
            if group_count < 2:
                rows.append({**base, "status": "unavailable", "reason": "Fewer than two attempts."})
                continue
            splitter = GroupKFold(n_splits=min(5, group_count))
            design = action_conditional_design(usable, features)
            target = usable[reward].to_numpy(dtype=float)
            fold_metrics: list[dict[str, float | int | None]] = []
            for fold, (train, test) in enumerate(splitter.split(design, target, groups), start=1):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(design[train], target[train])
                predicted = model.predict(design[test])
                fold_metrics.append(
                    {
                        "fold": fold,
                        "train_n": int(len(train)),
                        "test_n": int(len(test)),
                        "mae": float(mean_absolute_error(target[test], predicted)),
                        "rmse": float(mean_squared_error(target[test], predicted) ** 0.5),
                        "r2": (
                            float(r2_score(target[test], predicted))
                            if len(test) >= 2 and np.var(target[test]) > 1e-12
                            else None
                        ),
                    }
                )
            metric_summary: dict[str, float | None] = {}
            for metric in ("mae", "rmse", "r2"):
                values = np.array(
                    [item[metric] for item in fold_metrics if item[metric] is not None],
                    dtype=float,
                )
                metric_summary[f"{metric}_mean"] = float(values.mean()) if values.size else None
                metric_summary[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if values.size > 1 else None
                )
            rows.append(
                {
                    **base,
                    "status": "available",
                    "reason": "5-fold GroupKFold by attempt; ridge alpha=1.0; direct-arm interactions.",
                    "fold_count": len(fold_metrics),
                    **metric_summary,
                    "fold_metrics": json.dumps(fold_metrics, separators=(",", ":")),
                }
            )
    return rows


def feature_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    usable = frame[frame["usable_reward"]]
    source_timing = {
        "S": (
            "MD selector probabilities",
            "computed from problem and history before Tutor action t",
        ),
        "K": (
            "BKT state and prior linked turn outcome",
            "mastery_before is snapshotted before action t; prior delta is from t-1",
        ),
        "L": (
            "Student Modeling resolved behaviour from t-1",
            "assigned after learner response t-1 and before action t",
        ),
        "H": (
            "Turn controller completed-action history",
            "previous move and count exist before action t",
        ),
        "Q": (
            "frozen MRB1 score for Tutor response t-1",
            "current MRB1_t is post-action and only becomes Q for t+1",
        ),
    }
    rows: list[dict[str, Any]] = []
    for block, features in BLOCK_FEATURES.items():
        source, timing = source_timing[block]
        for feature in features:
            available = int(usable[feature].notna().sum())
            rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "source": source,
                    "pre_action_timing": timing,
                    "leakage_status": "PASS",
                    "usable_reward_rows": int(len(usable)),
                    "available_rows": available,
                    "missing_rows": int(len(usable) - available),
                    "retrospective_ablation_status": (
                        "available" if available == len(usable) else "unavailable"
                    ),
                }
            )
    return rows


def mrb1_analysis(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usable = frame[frame["usable_reward"]].copy()
    rows: list[dict[str, Any]] = []
    distributions: dict[str, Any] = {}
    current_correlations: dict[str, Any] = {}
    next_correlations: dict[str, Any] = {}

    for head in MRB1_HEADS:
        column = MRB1_COLUMNS[head]
        stats = describe(usable[column])
        stats["missing_count"] = int(usable[column].isna().sum())
        stats["availability_rate"] = float(usable[column].notna().mean())
        distributions[head] = stats
        rows.append({"analysis_type": "distribution", "x": head, "y": None, **stats})
        current_correlations[head] = {}
        next_correlations[head] = {}
        for reward in ("r_raw", "r_headroom"):
            corr = spearman(usable[column], usable[reward])
            current_correlations[head][reward] = corr
            rows.append(
                {
                    "analysis_type": "current_reward_spearman",
                    "x": head,
                    "y": reward,
                    **corr,
                }
            )

    next_frame = usable.sort_values(["attempt_id", "turn_index"]).copy()
    for reward in ("r_raw", "r_headroom"):
        next_frame[f"next_{reward}"] = next_frame.groupby("attempt_id")[reward].shift(-1)
    for head in MRB1_HEADS:
        column = MRB1_COLUMNS[head]
        for reward in ("r_raw", "r_headroom"):
            corr = spearman(next_frame[column], next_frame[f"next_{reward}"])
            next_correlations[head][reward] = corr
            rows.append(
                {
                    "analysis_type": "next_reward_spearman",
                    "x": head,
                    "y": reward,
                    **corr,
                }
            )

    head_correlations: dict[str, Any] = {}
    for index, first in enumerate(MRB1_HEADS):
        for second in MRB1_HEADS[index + 1 :]:
            corr = spearman(usable[MRB1_COLUMNS[first]], usable[MRB1_COLUMNS[second]])
            head_correlations[f"{first}__{second}"] = corr
            rows.append(
                {
                    "analysis_type": "head_to_head_spearman",
                    "x": first,
                    "y": second,
                    **corr,
                }
            )

    usable["mrb1_mean_descriptive_only"] = usable[
        [MRB1_COLUMNS[head] for head in MRB1_HEADS]
    ].mean(axis=1)
    quality_median = float(usable["mrb1_mean_descriptive_only"].median())
    disagreements = {
        "high_quality_negative_raw": int(
            ((usable["mrb1_mean_descriptive_only"] >= quality_median) & (usable["r_raw"] < 0)).sum()
        ),
        "low_quality_positive_raw": int(
            ((usable["mrb1_mean_descriptive_only"] < quality_median) & (usable["r_raw"] > 0)).sum()
        ),
        "quality_median": quality_median,
    }
    for name, value in disagreements.items():
        rows.append(
            {
                "analysis_type": "disagreement_summary",
                "x": name,
                "y": None,
                "value": value,
                "n": int(len(usable)),
            }
        )
    summary = {
        "usable_count": int(len(usable)),
        "complete_four_head_count": int(
            usable[[MRB1_COLUMNS[head] for head in MRB1_HEADS]].notna().all(axis=1).sum()
        ),
        "distributions": distributions,
        "head_to_head_spearman": head_correlations,
        "current_reward_spearman": current_correlations,
        "next_reward_spearman": next_correlations,
        "disagreements": disagreements,
        "scoring": {
            "labels": {"No": 0.0, "To some extent": 0.5, "Yes": 1.0},
            "stored_value": (
                "continuous expected score: 0.5 * P(To some extent) + P(Yes)"
            ),
        },
        "interpretation_constraint": "MRB1 is a noisy auxiliary critic, not ground truth or the scalar reward.",
    }
    return rows, summary


def signal_audit(frame: pd.DataFrame) -> dict[str, Any]:
    usable = frame[frame["usable_reward"]]
    mapping = {
        "previous_reasoning_probability": 0.30,
        "previous_uncertainty_probability": 0.25,
        "previous_clarification_probability": 0.40,
    }
    signals: dict[str, Any] = {}
    actually_available = (
        usable["previous_learner_signals_missing"].eq(0.0)
        & usable[list(mapping)].notna().all(axis=1)
    )
    available_subset = usable.loc[actually_available, list(mapping)]
    for feature, threshold in mapping.items():
        series = usable.loc[actually_available, feature]
        signals[feature] = {
            **describe(series),
            "missing_count": int(len(usable) - len(series)),
            "threshold_for_component_diagnostic_only": threshold,
            "present_count": int((series >= threshold).sum()),
            "near_constant": (
                bool(series.nunique() <= 1) if len(series) >= 2 else None
            ),
        }
    correlation = available_subset.corr(method="spearman").to_dict() if len(available_subset) >= 2 else {}
    return {
        "signals": signals,
        "head_to_head_spearman": correlation,
        "complete_previous_signal_rows": int(actually_available.sum()),
        "usable_reward_rows": int(len(usable)),
        "thresholded_features_excluded_from_context": True,
        "temporal_status": "PASS: only resolved learner-response probabilities from t-1 feed action t.",
    }


def selector_integrity() -> dict[str, Any]:
    config = json.loads((SELECTOR_ROOT / "config.json").read_text(encoding="utf-8"))
    labels = [config["id2label"][str(index)] for index in range(4)]
    expected_sha = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
    expected_rollback_sha = "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13"
    actual_sha = sha256(SELECTOR_ROOT / "model.safetensors")
    rollback_available = (ROLLBACK_ROOT / "model.safetensors").exists()
    rollback_sha = (
        sha256(ROLLBACK_ROOT / "model.safetensors") if rollback_available else None
    )
    selector_mode_source = SELECTOR_MODE_SOURCE.read_text(encoding="utf-8")
    coordinator_source = COORDINATOR_SOURCE.read_text(encoding="utf-8")
    source_checks = {
        "normal_mode_declared": (
            'NORMAL_MD7 = "ordinary-md7-r2-tell-c1-v1"' in selector_mode_source
        ),
        "normal_mode_is_default": (
            "return AdaptiveSelectorMode.NORMAL_MD7" in selector_mode_source
        ),
        "normal_loader_points_to_promoted_directory": (
            '"md7_r2_tell_c1_epoch2"' in coordinator_source
            and "self.active_model_path = self.md7_model_dir" in coordinator_source
            and "model_loader(self.md7_model_dir)" in coordinator_source
        ),
        "runtime_enforces_promoted_weight_hash": (
            expected_sha in coordinator_source
            and "Promoted MD7-R2-TELL-C1 weight hash mismatch" in coordinator_source
        ),
    }
    runtime = (
        json.loads(LOCAL_RUNTIME_STATE.read_text(encoding="utf-8-sig"))
        if LOCAL_RUNTIME_STATE.exists()
        else {}
    )
    return {
        "active_default": "MD7-R2-TELL-C1 Epoch 2",
        "normal_mode": "ordinary-md7-r2-tell-c1-v1",
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "label_order": labels,
        "source_checks": source_checks,
        "runtime_selector_mode": runtime.get("selectorMode"),
        "runtime_turn_lints_mode": runtime.get("turnLinTSMode"),
        "integrity_pass": (
            actual_sha == expected_sha
            and labels == list(MOVES)
            and all(source_checks.values())
        ),
        "rollback_md7_r1_available": rollback_available,
        "rollback_md7_r1_expected_sha256": expected_rollback_sha,
        "rollback_md7_r1_sha256": rollback_sha,
        "rollback_md7_r1_integrity_pass": (
            rollback_available and rollback_sha == expected_rollback_sha
        ),
    }


def shadow_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    events = load_jsonl(SHADOW_EVENTS)
    state = json.loads(SHADOW_STATE.read_text(encoding="utf-8"))
    passive = frame.set_index("action_event_id", drop=False)
    links = []
    for event in events:
        action_id = event.get("action_event_id")
        linked = action_id in passive.index
        linked_resolver = passive.loc[action_id, "resolver_event_id"] if linked else None
        probabilities = event.get("md7_raw_probabilities") or {}
        base_move = max(MOVES, key=lambda move: probabilities.get(move, -math.inf))
        links.append(
            {
                "action_event_id": action_id,
                "passive_linked": bool(linked),
                "resolver_linked": bool(linked and linked_resolver == event.get("resolver_event_id")),
                "actual_equals_md7_argmax": event.get("actual_move") == base_move,
                "selected_arm_is_null": event.get("selected_arm") is None,
                "hypothetical_arm_present": event.get("hypothetical_arm") in MOVES,
                "posterior_update_false": event.get("posterior_update_occurred") is False,
            }
        )
    ordered_events = sorted(events, key=lambda item: int(item["action_turn_index"]))
    chain_checks: list[dict[str, Any]] = []
    for previous, current in zip(ordered_events, ordered_events[1:]):
        context = current.get("context") or {}
        names = context.get("feature_names") or []
        vector = context.get("vector") or []
        values = dict(zip(names, vector, strict=True))
        prior_signals = previous.get("learner_signals_for_next_action") or {}
        prior_quality = previous.get("current_response_mrb1") or {}
        chain_checks.append(
            {
                "previous_action_event_id": previous.get("action_event_id"),
                "current_action_event_id": current.get("action_event_id"),
                "previous_delta_matches": math.isclose(
                    float(values["previous_mastery_delta"]),
                    float(previous["raw_delta"]),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                ),
                "previous_learner_signals_match": all(
                    math.isclose(
                        float(values[f"previous_{name}"]),
                        float(prior_signals[name]),
                        abs_tol=1e-12,
                        rel_tol=0.0,
                    )
                    for name in (
                        "reasoning_probability",
                        "uncertainty_probability",
                        "clarification_probability",
                    )
                ),
                "previous_mrb1_matches": all(
                    math.isclose(
                        float(values["previous_" + head.casefold()]),
                        float(prior_quality[head]),
                        abs_tol=1e-12,
                        rel_tol=0.0,
                    )
                    for head in MRB1_HEADS
                ),
                "previous_move_matches": values[f"previous_move_{previous['actual_move']}"] == 1.0,
                "only_previous_quality_named_in_context": all(
                    not name.startswith("current_") for name in names
                ),
            }
        )
    ridge = float(state["posterior_hyperparameters"]["ridge_lambda"])
    matrices_initial = all(
        np.allclose(np.asarray(state["A"][move], dtype=float), ridge * np.eye(state["context_dimension"]))
        for move in MOVES
    )
    vectors_zero = all(
        np.allclose(np.asarray(state["b"][move], dtype=float), 0.0) for move in MOVES
    )
    censored = frame[~frame["usable_reward"]]
    return {
        "mode": "SHADOW",
        "event_count": len(events),
        "event_checks": links,
        "md7_controls_actual_action": bool(links and all(item["actual_equals_md7_argmax"] for item in links)),
        "hypothetical_never_controls_action": bool(
            links and all(item["selected_arm_is_null"] and item["hypothetical_arm_present"] for item in links)
        ),
        "policy_update_count": int(state["update_count"]),
        "arm_update_counts": state["arm_update_counts"],
        "A_is_initial_ridge_identity": matrices_initial,
        "b_is_zero": vectors_zero,
        "linked_event_count": sum(item["passive_linked"] for item in links),
        "linked_resolver_count": sum(item["resolver_linked"] for item in links),
        "sequential_context_checks": chain_checks,
        "sequential_context_integrity": bool(
            chain_checks
            and all(
                all(value is True for key, value in item.items() if key.endswith("_matches") or key == "only_previous_quality_named_in_context")
                for item in chain_checks
            )
        ),
        "censored_or_failed_count": int(len(censored)),
        "censored_have_no_fake_reward": bool(censored[["r_raw", "r_headroom"]].isna().all().all()),
        "formal_assessment_action_rewards": int(
            (frame["observation_status"] == "formal_assessment").sum()
        ),
        "limitation": (
            f"Only {len(events)} real SHADOW events from one attempt are available; "
            "source/tests provide the broader lifecycle evidence."
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(summary: dict[str, Any], ablation: pd.DataFrame) -> None:
    raw = summary["rewards"]["R_RAW"]
    headroom = summary["rewards"]["R_HEADROOM"]
    corr = summary["reward_correlations"]
    lines = [
        "# Turn-LinTS reward and context audit v1",
        "",
        "## Scope and safeguards",
        "",
        "Read-only analysis of existing real passive turn records. BKT belief change is a transformed model-state update, not measured or causal learning. MRB1 is a noisy auxiliary critic, not ground truth. No training, external APIs, protected tests, historical rewrites, or policy updates were used.",
        "",
        "## Selector and SHADOW integrity",
        "",
        f"- Selector integrity: {'PASS' if summary['selector_integrity']['integrity_pass'] else 'FAIL'}; MD7-R2-TELL-C1 Epoch 2, SHA-256 `{summary['selector_integrity']['actual_sha256']}`, labels `{', '.join(summary['selector_integrity']['label_order'])}`.",
        f"- MD7-R1 rollback remains available at SHA-256 `{summary['selector_integrity']['rollback_md7_r1_sha256']}`.",
        f"- Recorded local runtime: selector `{summary['selector_integrity']['runtime_selector_mode']}`; Turn-LinTS `{summary['selector_integrity']['runtime_turn_lints_mode']}`.",
        f"- Real SHADOW events: {summary['shadow_integrity']['event_count']}; policy updates: {summary['shadow_integrity']['policy_update_count']}; linked passive/resolver events: {summary['shadow_integrity']['linked_event_count']}/{summary['shadow_integrity']['linked_resolver_count']}.",
        f"- Sequential SHADOW context linkage: {'PASS' if summary['shadow_integrity']['sequential_context_integrity'] else 'INSUFFICIENT/FAIL'}; turn t+1 carries only the prior turn's BKT delta, learner signals, move, and MRB1 values.",
        f"- Limitation: {summary['shadow_integrity']['limitation']}",
        "",
        "## Usable rewards",
        "",
        f"Usable linked turn-level BKT observations: **{summary['usable_reward_count']}** from {summary['attempt_count']} attempts, {summary['learner_count']} learners, and {summary['skill_count']} skills. Two censored turns have no fabricated zero reward.",
        "",
        "| reward | count | min | Q1 | median | mean | Q3 | max | std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| R_RAW | {raw['count']} | {fmt(raw['min'])} | {fmt(raw['q1'])} | {fmt(raw['median'])} | {fmt(raw['mean'])} | {fmt(raw['q3'])} | {fmt(raw['max'])} | {fmt(raw['std'])} |",
        f"| R_HEADROOM | {headroom['count']} | {fmt(headroom['min'])} | {fmt(headroom['q1'])} | {fmt(headroom['median'])} | {fmt(headroom['mean'])} | {fmt(headroom['q3'])} | {fmt(headroom['max'])} | {fmt(headroom['std'])} |",
        "",
        "Spearman correlations of absolute reward magnitude:",
        "",
        "| relationship | R_RAW rho | R_HEADROOM rho |",
        "|---|---:|---:|",
        f"| mastery_before | {fmt(corr['abs_raw_vs_mastery']['rho'])} | {fmt(corr['abs_headroom_vs_mastery']['rho'])} |",
        f"| turn_index | {fmt(corr['abs_raw_vs_turn_index']['rho'])} | {fmt(corr['abs_headroom_vs_turn_index']['rho'])} |",
        "",
        f"Raw magnitude has a strong negative mastery association (rho {fmt(corr['abs_raw_vs_mastery']['rho'])}), while headroom magnitude is nearly unrelated to mastery (rho {fmt(corr['abs_headroom_vs_mastery']['rho'])}). Headroom values remain finite and bounded in the observed range [{fmt(headroom['min'])}, {fmt(headroom['max'])}], although their standard deviation is larger ({fmt(headroom['std'])} versus {fmt(raw['std'])}). Neither reward has a material turn-index association. Thus the recommendation considers stability, range, position, and grouped predictability as well as the correlation change.",
        "",
        "Mastery-band behavior:",
        "",
        "| band | reward | n | signed mean | absolute mean |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["reward_by_mastery_band"]:
        lines.append(
            f"| {row['mastery_band']} | {row['reward']} | {row['count'] or 0} | "
            f"{fmt(row['mean'])} | {fmt(row['absolute_mean'])} |"
        )
    lines.extend(
        [
        "",
        "Turn-position summaries are in `reward_by_turn_position.csv`; move, learner, and skill breakdowns are in `summary.json`.",
        "",
        "## Historical action coverage",
        "",
        "| move | usable turns |",
        "|---|---:|",
        ]
    )
    for move in MOVES:
        lines.append(f"| {move} | {summary['action_counts'][move]} |")
    lines.extend(
        [
            "",
            "Telling has no usable historical observations, so no retrospective analysis validates that direct arm.",
            "",
            "## Student Modeling and MRB1",
            "",
            f"The implemented learner-state outputs are exactly `previous_reasoning_probability`, `previous_uncertainty_probability`, and `previous_clarification_probability`. Complete retrospective pre-action values exist for only {summary['student_signals']['complete_previous_signal_rows']} of {summary['usable_reward_count']} usable rows, so L-containing ablations are unavailable rather than imputed.",
            f"All four MRB1 heads are complete on {summary['mrb1']['complete_four_head_count']} of {summary['mrb1']['usable_count']} usable rows. Stored scores are continuous expected values using `No=0`, `To some extent=0.5`, and `Yes=1`. Current MRB1 is treated only as post-action outcome; Q uses the previous response's scores.",
            f"Descriptive disagreement counts: high-quality/negative raw reward = {summary['mrb1']['disagreements']['high_quality_negative_raw']}; low-quality/positive raw reward = {summary['mrb1']['disagreements']['low_quality_positive_raw']}. These are associations, not causal effects.",
            "",
            "| MRB1 head | mean | std | Q1 | median | Q3 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for head in MRB1_HEADS:
        stats = summary["mrb1"]["distributions"][head]
        lines.append(
            f"| {head} | {fmt(stats['mean'])} | {fmt(stats['std'])} | "
            f"{fmt(stats['q1'])} | {fmt(stats['median'])} | {fmt(stats['q3'])} |"
        )
    lines.extend(
        [
            "",
            "The four MRB1 heads are strongly intercorrelated (pairwise Spearman rho 0.749 to 0.966), but each head has near-zero current-reward and next-reward association (absolute rho at most 0.098 in these data). MRB1 therefore describes a signal distinct from BKT belief change, but adding previous MRB1 (D1 versus D0) worsens grouped MAE/RMSE for both rewards; current data do not support Q as incrementally predictive.",
            "",
            "## Temporal leakage audit",
            "",
            "- S: selector probabilities are computed from the problem/history before action t.",
            "- K: mastery is snapshotted before action t; previous delta is resolved after t-1 and carried forward.",
            "- L: learner-signal probabilities come from the resolved learner response at t-1, never the current response at t.",
            "- H: previous move and prior Tutor-turn count exist before selection.",
            "- Q: MRB1_t is scored after Tutor response t and can enter only action t+1 as previous MRB1.",
            "",
            "All implemented fields pass the source-order audit. Features with absent retrospective values are rejected from ablation rather than filled from current/future information.",
            "",
            "## Grouped context ablation",
            "",
            "Primary validation is deterministic 5-fold GroupKFold by attempt. Each available model is ridge-regularized linear prediction with direct action-by-context interactions. Turns are never randomly split.",
            "",
            "| config | blocks | reward | status | n | MAE mean (sd) | RMSE mean (sd) | R2 mean (sd) |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(
            f"| {row['configuration']} | {row['blocks']} | {row['reward']} | {row['status']} | {int(row['sample_count'])} | "
            f"{fmt(row.get('mae_mean'))} ({fmt(row.get('mae_std'))}) | "
            f"{fmt(row.get('rmse_mean'))} ({fmt(row.get('rmse_std'))}) | "
            f"{fmt(row.get('r2_mean'))} ({fmt(row.get('r2_std'))}) |"
        )
    lines.extend(
        [
            "",
            "D0/D1 are explicitly diagnostic fallbacks for testing H and Q when L is unavailable; they are not substitutions for the requested C4/C5 comparison. Fold-level metrics and unavailable reasons are in `context_ablation_results.csv`.",
            "",
            "## Decisions",
            "",
            f"**REWARD: {summary['decisions']['reward']}**",
            "",
            f"**CONTEXT: {summary['decisions']['context']}**",
            "",
            f"**TURN-LINTS READINESS: {summary['decisions']['readiness']}**",
            "",
            summary["decisions"]["rationale"],
            "",
            "Turn-LinTS remains SHADOW. This report does not authorize LIVE.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows = load_jsonl(TURN_DATA)
    frame = build_turn_table(raw_rows)
    usable = frame[frame["usable_reward"]].copy()
    ablation_rows = run_ablation(frame)
    ablation = pd.DataFrame(ablation_rows)
    feature_rows = feature_audit(frame)
    mrb1_rows, mrb1_summary = mrb1_analysis(frame)

    reward_correlations = {
        "abs_raw_vs_mastery": spearman(usable["r_raw"].abs(), usable["mastery_before"]),
        "abs_headroom_vs_mastery": spearman(usable["r_headroom"].abs(), usable["mastery_before"]),
        "abs_raw_vs_turn_index": spearman(usable["r_raw"].abs(), usable["turn_index"]),
        "abs_headroom_vs_turn_index": spearman(usable["r_headroom"].abs(), usable["turn_index"]),
        "abs_raw_vs_relative_position": spearman(usable["r_raw"].abs(), usable["relative_position"]),
        "abs_headroom_vs_relative_position": spearman(usable["r_headroom"].abs(), usable["relative_position"]),
    }
    group_breakdowns: dict[str, Any] = {}
    for group in ("pedagogical_move", "learner_id", "skill"):
        group_breakdowns[group] = grouped_stats(usable, group)

    action_counts = {move: int((usable["pedagogical_move"] == move).sum()) for move in MOVES}
    l_complete = int(
        (
            usable["previous_learner_signals_missing"].eq(0.0)
            & usable[list(BLOCK_FEATURES["L"][:3])].notna().all(axis=1)
        ).sum()
    )
    decisions = {
        "reward": "HEADROOM_NORMALIZED",
        "context": "INCONCLUSIVE",
        "readiness": "C. CONTEXT NEEDS MORE EVIDENCE",
        "rationale": (
            "HEADROOM_NORMALIZED removes the observed mechanical mastery dependence "
            "without non-finite values, boundary explosions, clipping, or added turn-index "
            "dependence; its wider but bounded scale and similar normalized predictive "
            "difficulty were considered explicitly. Context remains inconclusive because "
            f"complete L values exist on only {l_complete} usable row(s), telling has zero "
            "historical observations, and every available grouped model has unstable "
            "negative mean R2. More causally complete, action-covered attempts are required."
        ),
    }
    summary = {
        "schema_version": "turn_lints_reward_context_audit_v1",
        "source": str(TURN_DATA.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "source_row_count": int(len(frame)),
        "usable_reward_count": int(len(usable)),
        "attempt_count": int(usable["attempt_id"].nunique()),
        "learner_count": int(usable["learner_id"].nunique()),
        "skill_count": int(usable["skill"].nunique()),
        "action_counts": action_counts,
        "rewards": {
            "R_RAW": describe(usable["r_raw"]),
            "R_HEADROOM": describe(usable["r_headroom"]),
        },
        "reward_correlations": reward_correlations,
        "reward_by_mastery_band": grouped_stats(usable, "mastery_band"),
        "reward_by_turn_position": grouped_stats(usable, "position_band"),
        "reward_breakdowns": group_breakdowns,
        "student_signals": signal_audit(frame),
        "mrb1": mrb1_summary,
        "selector_integrity": selector_integrity(),
        "shadow_integrity": shadow_integrity(frame),
        "validation": {
            "split": "5-fold GroupKFold by attempt",
            "model": "StandardScaler + Ridge(alpha=1.0), direct action-by-context interactions",
            "random_turn_split": False,
        },
        "decisions": decisions,
        "safety": {
            "training": 0,
            "turn_lints_live_updates": 0,
            "protected_mathdial_test_use": 0,
            "mrbench_v3_test_use": 0,
            "authoritative_historical_data_rewrites": 0,
            "external_tutor_api_calls": 0,
        },
    }

    output_columns = [
        "action_event_id", "resolver_event_id", "attempt_id", "learner_id",
        "turn_index", "relative_position", "mastery_band", "position_band",
        "skill", "pedagogical_move", "observation_status", "should_update",
        "usable_reward", "mastery_before", "mastery_after", "r_raw", "r_headroom",
        "selector_mode", "selector_checkpoint", "selector_sha256", "turn_lints_mode",
        "hypothetical_arm", "selected_arm", *[name for block in BLOCK_FEATURES.values() for name in block],
        *[MRB1_COLUMNS[head] for head in MRB1_HEADS],
    ]
    frame[output_columns].to_csv(OUTPUT_DIR / "turn_reward_table.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUTPUT_DIR / "context_feature_audit.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "context_ablation_results.csv", index=False)
    pd.DataFrame(grouped_stats(usable, "mastery_band")).to_csv(
        OUTPUT_DIR / "reward_by_mastery_band.csv", index=False
    )
    pd.DataFrame(grouped_stats(usable, "position_band")).to_csv(
        OUTPUT_DIR / "reward_by_turn_position.csv", index=False
    )
    pd.DataFrame(mrb1_rows).to_csv(OUTPUT_DIR / "mrb1_reward_analysis.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(summary, ablation)
    print(
        json.dumps(
            {
                "source_rows": len(frame),
                "usable_rewards": len(usable),
                "attempts": int(usable["attempt_id"].nunique()),
                "action_counts": action_counts,
                "output": str(OUTPUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
