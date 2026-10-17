"""Final, leakage-free AdaptMath Turn-LinTS v1 architecture selection.

This analysis is intentionally derived-only.  It reads authoritative logs and
source, but never rewrites them, trains a neural model, calls an external API,
uses protected MathDial final test data, uses MRBench V3 official test data, or
mutates a Turn-LinTS posterior.

Predeclared reward rule (declared before ``main`` computes any result):
1. preserve BKT posterior-update sign exactly;
2. reject non-finite or numerically unstable candidates;
3. retain non-zero contextual-bandit reward variation;
4. prefer HEADROOM only when it materially reduces mechanical starting-mastery
   dependence (>= .20 absolute-Spearman reduction for reward magnitude), stays
   bounded without tuned constants, and introduces no strong (|rho| >= .30)
   turn/relative-position artifact; otherwise prefer RAW when stable, or report
   INCONCLUSIVE.

Predeclared context rule:
* nested GroupKFold by attempt; no random turn split;
* fold-local context standardization;
* Ridge alpha selected only inside each outer training fold from
  [0.01, 0.1, 1, 10, 100] using grouped MAE;
* primary score is mean outer-fold MAE on the selected reward, with RMSE/R2 as
  corroborating metrics;
* among contexts within one standard error of the lowest-MAE context, select
  the smallest dimension (then lower RMSE, then canonical candidate order).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTPUT_DIR / "figures"
MOVE_ROOT = OUTPUT_DIR.parents[1]
WORKSPACE_ROOT = MOVE_ROOT.parent
BACKEND_ROOT = WORKSPACE_ROOT / "adaptive-math-tutor" / "backend"
TURN_DATA_ROOT = MOVE_ROOT / "results" / "md_self_improvement_turn_data_v1"
TURN_OUTCOMES = TURN_DATA_ROOT / "turn_outcomes.jsonl"
ASSESSMENT_OUTCOMES = TURN_DATA_ROOT / "assessment_outcomes.jsonl"
ATTEMPT_SUMMARIES = TURN_DATA_ROOT / "attempt_summaries.jsonl"
LOCAL_RUNTIME_STATE = WORKSPACE_ROOT / "_local_runtime_state" / "adaptmath-local.json"
SELECTOR_MODEL = (
    MOVE_ROOT / "models" / "frozen" / "md7_r2_tell_c1_epoch2" / "model.safetensors"
)
CONTEXT_SOURCE = MOVE_ROOT / "src" / "self_improvement" / "turn_lints_context.py"
REWARD_SOURCE = MOVE_ROOT / "src" / "self_improvement" / "turn_lints_reward.py"
RUNTIME_SOURCE = MOVE_ROOT / "src" / "self_improvement" / "turn_lints_runtime.py"
WARMSTART_SOURCE = MOVE_ROOT / "src" / "self_improvement" / "randomized_warmstart.py"
POLICY_SOURCE = MOVE_ROOT / "src" / "self_improvement" / "turn_lints_policy.py"
SELECTOR_SHA256 = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
SELECTOR_NAME = "MD7-R2-TELL-C1 Epoch 2"
MOVES = ("generic", "probing", "focus", "telling")
BLOCK_ORDER = ("S", "K", "L", "H", "Q")
MRB1_HEADS = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)
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
CONTEXT_CANDIDATES: tuple[str, ...] = (
    "S",
    "S+K",
    "S+L",
    "S+H",
    "S+Q",
    "S+K+L",
    "S+K+H",
    "S+K+Q",
    "S+K+L+H",
    "S+K+L+H+Q",
    "K+L+H+Q",
    "S+L+H+Q",       # FULL-minus-K
    "S+K+L+Q",       # FULL-minus-H
    "S+K+H+Q",       # FULL-minus-L and required Q matched pair
)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
OUTER_SPLITS = 5
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260829
REWARD_NOTE = "BKT posterior belief update, not measured learning"

REWARD_DECISION_CRITERIA = (
    "Preserve the sign/direction of the BKT posterior belief update.",
    "Avoid mechanical domination by starting mastery/headroom.",
    "Avoid a strong artificial dependence on turn or relative attempt position.",
    "Remain finite and stable at mastery boundaries.",
    "Use no arbitrary clipping or tuned weighting constants.",
    "Retain usable contextual-bandit reward variation.",
)
CONTEXT_DECISION_CRITERIA = (
    "Minimize grouped out-of-sample MAE on the selected reward.",
    "Inspect RMSE, R2, and fold/group variation.",
    "Use the one-standard-error rule for statistically/practically indistinguishable contexts.",
    "Retain a block only for reproducible out-of-sample benefit or correction of a clear deficiency.",
    "Treat sparse substantive coverage as a limitation, not proof of uselessness or usefulness.",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
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


def nested(value: Any, *keys: str) -> Any:
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


def canonical_blocks(preset: str) -> tuple[str, ...]:
    supplied = set(preset.split("+"))
    if not supplied or not supplied <= set(BLOCK_ORDER):
        raise ValueError(f"Invalid context preset {preset!r}")
    return tuple(block for block in BLOCK_ORDER if block in supplied)


def feature_names(preset: str) -> list[str]:
    return [name for block in canonical_blocks(preset) for name in BLOCK_FEATURES[block]]


def context_schema_id(preset: str) -> str:
    blocks = canonical_blocks(preset)
    payload = json.dumps(
        {
            "version": "turn_lints_context_v1",
            "blocks": blocks,
            "features": feature_names(preset),
        },
        separators=(",", ":"),
    )
    return f"turn_lints_context_v1:{'+'.join(blocks)}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def headroom_reward(before: float, after: float) -> float:
    delta = after - before
    if delta > 0.0:
        result = delta / (1.0 - before) if before < 1.0 else 0.0
    elif delta < 0.0:
        result = delta / before if before > 0.0 else 0.0
    else:
        result = 0.0
    if not math.isfinite(result) or not -1.0 - 1e-12 <= result <= 1.0 + 1e-12:
        raise ValueError(f"Invalid headroom reward {result} from {before}, {after}")
    return max(-1.0, min(1.0, float(result)))


def _normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.casefold().replace("â€™", "'").replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def no_knowledge_evidence_category(row: dict[str, Any]) -> str | None:
    """Classify logged unknown behavioural proxies without exact-string rules.

    The stored evaluator's semantic reason is authoritative for this derived
    audit.  Explicit knowledge-state evidence (confusion, inability, lack of
    understanding/knowledge) remains behavioural evidence.  Pure social
    acknowledgements, procedural interaction management, transcript echoes,
    and problem repetitions do not constitute knowledge evidence.
    """

    if nested(row, "learning_state_update", "should_update") is not True:
        return None
    if nested(row, "learning_state_update", "observation_source") != "behavioural_proxy":
        return None
    if nested(row, "next_learner_observation", "evaluator", "correctness") != "unknown":
        return None
    reason = _normalized_text(nested(row, "next_learner_observation", "evaluator", "reason"))
    text = _normalized_text(nested(row, "next_learner_observation", "student_text"))
    knowledge_markers = (
        "lack of knowledge",
        "lack of understanding",
        "does not understand",
        "not understanding",
        "uncertainty",
        "confusion",
        "confused",
        "need for further teaching",
        "lack of a required concept",
        "lack of conceptual understanding",
    )
    text_knowledge_markers = (
        "dont know",
        "do not know",
        "no idea",
        "dont understand",
        "do not understand",
        "cannot understand",
        "cant understand",
        "confused",
        "not sure",
        "stuck",
    )
    has_knowledge_state = any(marker in reason for marker in knowledge_markers) or any(
        marker in text for marker in text_knowledge_markers
    )
    if has_knowledge_state:
        return None
    if any(marker in reason for marker in ("verbatim repetition", "direct repetition")):
        return "no_knowledge_evidence_transcript_echo"
    if "repetition of the problem" in reason:
        return "no_knowledge_evidence_problem_repetition"
    if any(marker in reason for marker in ("acknowledgement", "social affirmation")):
        return "interaction_management_acknowledgement"
    if "procedural question" in reason:
        return "interaction_management_request"
    if (
        "does not contain any mathematical content" in reason
        or "does not contain assessable mathematical content" in reason
    ):
        return "no_knowledge_evidence_metacognitive_or_social"
    return None


def extract_stored_context(row: dict[str, Any]) -> dict[str, float] | None:
    decision = row.get("adaptive_decision")
    if not isinstance(decision, dict):
        return None
    mapping = decision.get("feature_values")
    if isinstance(mapping, dict):
        result = {str(key): float(value) for key, value in mapping.items()}
        return result if set(result) >= set(sum((list(v) for v in BLOCK_FEATURES.values()), [])) else None
    names, vector = decision.get("feature_names"), decision.get("vector")
    if not isinstance(names, list) or not isinstance(vector, list) or len(names) != len(vector):
        return None
    return {str(name): float(value) for name, value in zip(names, vector, strict=True)}


def current_mrb1(row: dict[str, Any]) -> dict[str, float] | None:
    scores = nested(row, "tutor_quality", "scores")
    if not isinstance(scores, dict) or set(scores) != set(MRB1_HEADS):
        return None
    values = {head: finite_float(scores[head]) for head in MRB1_HEADS}
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items() if value is not None}


def resolution_status(row: dict[str, Any]) -> str:
    status = nested(row, "next_learner_observation", "status")
    if status == "observed_update":
        return "resolved_reward"
    if status == "resolved_no_update":
        return "resolved_no_update"
    if status == "processing_failed":
        return "failed"
    return "censored"


def build_derived_dataset(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    identities = [nested(row, "identity", "action_event_id") for row in raw_rows]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("All rows require action_event_id")
    duplicates = [key for key, count in Counter(identities).items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate action identities: {duplicates[:3]}")

    rows = sorted(
        raw_rows,
        key=lambda row: (
            str(nested(row, "identity", "attempt_id")),
            int(nested(row, "identity", "action_turn_index") or 0),
        ),
    )
    attempt_lengths = Counter(str(nested(row, "identity", "attempt_id")) for row in rows)
    prior_by_attempt: dict[str, dict[str, Any]] = {}
    contamination_by_attempt: dict[str, str] = {}
    records: list[dict[str, Any]] = []

    for row in rows:
        identity = row["identity"]
        attempt_id = str(identity["attempt_id"])
        action_id = str(identity["action_event_id"])
        turn = int(identity["action_turn_index"])
        prior = prior_by_attempt.get(attempt_id)
        stored = extract_stored_context(row)
        learning = row.get("learning_state_update") or {}
        outcome_before = finite_float(learning.get("mastery_before"))
        after = finite_float(learning.get("mastery_after"))
        before = outcome_before
        if before is None and stored is not None:
            before = finite_float(stored.get("mastery_before"))
        if before is None:
            before = finite_float(nested(row, "state_before_action", "mastery_at_action"))
        linked = (
            resolution_status(row) == "resolved_reward"
            and learning.get("should_update") is True
            and outcome_before is not None
            and after is not None
            and isinstance(learning.get("resolver_event_id"), str)
            and bool(learning.get("resolver_event_id"))
            and learning.get("observation_source") != "formal_assessment"
        )
        invalid_category = no_knowledge_evidence_category(row)
        prior_contamination = contamination_by_attempt.get(attempt_id)
        if invalid_category is not None and attempt_id not in contamination_by_attempt:
            contamination_by_attempt[attempt_id] = action_id

        exclusion: str | None = None
        if not linked:
            exclusion = f"no_scientifically_resolved_reward:{resolution_status(row)}"
        elif prior_contamination is not None:
            exclusion = f"downstream_mastery_state_contaminated_by:{prior_contamination}"
        elif invalid_category is not None:
            exclusion = invalid_category

        move = nested(row, "adaptive_decision", "final_move")
        if move not in MOVES:
            move = nested(row, "selector", "effective", "argmax")
        if move not in MOVES:
            raise ValueError(f"No actual move for {action_id}")
        probabilities = nested(row, "selector", "raw", "probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(MOVES):
            probabilities = nested(row, "selector", "effective", "probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(MOVES):
            raise ValueError(f"No four-arm selector probabilities for {action_id}")

        raw_reward = float(after - outcome_before) if linked else np.nan
        stored_delta = finite_float(learning.get("delta_mastery"))
        if linked and stored_delta is not None and not math.isclose(
            raw_reward, stored_delta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Stored reward mismatch for {action_id}")
        headroom = headroom_reward(outcome_before, after) if linked else np.nan

        runtime_mode = nested(row, "provenance", "turn_lints_mode")
        selector_mode = nested(row, "provenance", "selector_mode")
        lineage = nested(row, "provenance", "lints_policy_lineage")
        if not runtime_mode:
            runtime_mode = "LEGACY_POLICY_BOUND"
        if not lineage:
            lineage = "legacy_md7r1_attempt_policy"
        behavior_policy = nested(row, "adaptive_decision", "behavior_policy")
        if not behavior_policy:
            behavior_policy = "historical_policy_bound"
        randomized = nested(row, "adaptive_decision", "randomized_assignment") is True
        propensity = finite_float(nested(row, "adaptive_decision", "behavior_propensity"))

        record: dict[str, Any] = {
            "action_event_id": action_id,
            "attempt_id": attempt_id,
            "learner_id": identity.get("student_pseudonymous_id"),
            "skill_id": learning.get("skill") or nested(row, "state_before_action", "target_skill"),
            "turn_index_within_attempt": turn,
            "relative_attempt_position": turn / attempt_lengths[attempt_id],
            "actual_move": move,
            "runtime_mode": runtime_mode,
            "selector_mode": selector_mode,
            "policy_lineage": lineage,
            "decision_source": nested(row, "adaptive_decision", "treatment_assignment_source")
            or "historical_policy_bound",
            "behavior_policy": behavior_policy,
            "randomized_assignment": randomized,
            "behavior_propensity": propensity,
            "full_behavior_probability_vector": json.dumps(
                nested(row, "adaptive_decision", "full_behavior_probability_vector") or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "mastery_before": before,
            "mastery_after": after,
            "raw_mastery_delta": raw_reward,
            "reward_raw_delta": raw_reward,
            "reward_headroom_normalized": headroom,
            "reward_provenance_at_collection": nested(row, "adaptive_decision", "reward_mode")
            or "not_configured_in_historical_row",
            "current_mrb1_mistake_identification": np.nan,
            "current_mrb1_mistake_location": np.nan,
            "current_mrb1_providing_guidance": np.nan,
            "current_mrb1_actionability": np.nan,
            "resolution_status": resolution_status(row),
            "resolver_event_id": learning.get("resolver_event_id"),
            "observation_source": learning.get("observation_source"),
            "valid_linked_bkt_outcome": linked,
            "pre_action_context_representable": before is not None,
            "include_in_reward_experiment": linked and exclusion is None,
            "include_in_context_experiment": linked and exclusion is None and before is not None,
            "exclusion_reason": exclusion or "included",
            "no_knowledge_evidence_category": invalid_category,
            "mastery_state_contaminated": prior_contamination is not None,
            "stored_runtime_context": stored is not None,
            "learner_response_text": nested(row, "next_learner_observation", "student_text"),
            "evaluator_correctness": nested(row, "next_learner_observation", "evaluator", "correctness"),
            "evaluator_reason": nested(row, "next_learner_observation", "evaluator", "reason"),
        }
        for move_name in MOVES:
            record[f"selector_p_{move_name}"] = float(probabilities[move_name])

        quality = current_mrb1(row)
        if quality is not None:
            for head in MRB1_HEADS:
                record[f"current_mrb1_{head.casefold()}"] = quality[head]

        # K: stored direct context is exact when present; otherwise use only the
        # immediately prior scientifically resolved outcome and explicit missingness.
        if stored is not None:
            record["previous_mastery_delta"] = stored["previous_mastery_delta"]
            record["previous_mastery_delta_missing"] = stored["previous_mastery_delta_missing"]
        elif prior is not None and prior["scientifically_valid_outcome"]:
            record["previous_mastery_delta"] = prior["raw_delta"]
            record["previous_mastery_delta_missing"] = 0.0
        else:
            record["previous_mastery_delta"] = 0.0
            record["previous_mastery_delta_missing"] = 1.0

        # L: never synthesize historical learner probabilities.  Runtime contract
        # uses zeros plus a shared missing indicator.
        for name in BLOCK_FEATURES["L"]:
            record[name] = stored[name] if stored is not None else (1.0 if name.endswith("missing") else 0.0)

        # H: exact prior treatment history, available before action t.
        if stored is not None:
            for name in BLOCK_FEATURES["H"]:
                record[name] = stored[name]
        else:
            previous_move = prior["actual_move"] if prior is not None else None
            for move_name in MOVES:
                record[f"previous_move_{move_name}"] = float(previous_move == move_name)
            record["previous_move_missing"] = float(previous_move is None)
            record["prior_tutor_turn_count"] = float(turn - 1)

        # Q: prior Tutor response's separate four-head MRB1 output only.
        if stored is not None:
            for name in BLOCK_FEATURES["Q"]:
                record[name] = stored[name]
        else:
            previous_quality = prior["current_mrb1"] if prior is not None else None
            for head in MRB1_HEADS:
                record[f"previous_{head.casefold()}"] = (
                    previous_quality[head] if previous_quality is not None else 0.0
                )
            record["previous_mrb1_missing"] = float(previous_quality is None)

        # Block-level real substantive availability is distinct from numerical
        # representability under the missing-indicator contract.
        record["S_real_observed"] = True
        record["K_real_observed"] = before is not None and record["previous_mastery_delta_missing"] == 0.0
        record["L_real_observed"] = record["previous_learner_signals_missing"] == 0.0
        record["H_real_observed"] = record["previous_move_missing"] == 0.0
        record["Q_real_observed"] = record["previous_mrb1_missing"] == 0.0
        records.append(record)

        prior_by_attempt[attempt_id] = {
            "actual_move": move,
            "raw_delta": raw_reward,
            "current_mrb1": quality,
            "scientifically_valid_outcome": linked and exclusion is None,
        }

    frame = pd.DataFrame(records).sort_values(
        ["attempt_id", "turn_index_within_attempt"], kind="stable"
    ).reset_index(drop=True)
    for feature in sum((list(values) for values in BLOCK_FEATURES.values()), []):
        if feature not in frame or not np.isfinite(frame[feature].to_numpy(float)).all():
            raise RuntimeError(f"Context feature {feature} is not finite/complete")
    frame["mastery_band"] = pd.cut(
        frame["mastery_before"],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.000000000001],
        labels=["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"],
        right=False,
        include_lowest=True,
    )
    frame["position_band"] = pd.cut(
        frame["relative_attempt_position"],
        [0.0, 1 / 3, 2 / 3, 1.000000000001],
        labels=["early", "middle", "late"],
        right=False,
        include_lowest=True,
    )
    return frame


def spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int | None]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return {"n": int(len(pair)), "rho": None, "p_value": None}
    result = spearmanr(pair.x, pair.y)
    return {"n": int(len(pair)), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def reward_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = frame[frame.include_in_reward_experiment].copy()
    mapping = {
        "RAW_DELTA": "reward_raw_delta",
        "HEADROOM_NORMALIZED": "reward_headroom_normalized",
    }
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {
        "predeclared_criteria": list(REWARD_DECISION_CRITERIA),
        "scientific_semantics": REWARD_NOTE,
        "candidates": {},
    }
    for name, column in mapping.items():
        values = usable[column].astype(float)
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        correlations = {
            "reward_vs_mastery_before": spearman(values, usable.mastery_before),
            "absolute_reward_vs_mastery_before": spearman(values.abs(), usable.mastery_before),
            "reward_vs_turn_index": spearman(values, usable.turn_index_within_attempt),
            "absolute_reward_vs_turn_index": spearman(values.abs(), usable.turn_index_within_attempt),
            "reward_vs_relative_attempt_position": spearman(values, usable.relative_attempt_position),
            "absolute_reward_vs_relative_attempt_position": spearman(values.abs(), usable.relative_attempt_position),
        }
        row = {
            "reward": name,
            "N": int(len(values)),
            "min": float(values.min()),
            "P1": float(quantiles.loc[0.01]),
            "P5": float(quantiles.loc[0.05]),
            "Q1": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "mean": float(values.mean()),
            "Q3": float(quantiles.loc[0.75]),
            "P95": float(quantiles.loc[0.95]),
            "P99": float(quantiles.loc[0.99]),
            "max": float(values.max()),
            "SD": float(values.std(ddof=1)),
            "IQR": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
            "positive_proportion": float((values > 0).mean()),
            "zero_proportion": float((values == 0).mean()),
            "negative_proportion": float((values < 0).mean()),
            "nonfinite_count": int((~np.isfinite(values)).sum()),
            "boundary_before_le_0_01_N": int((usable.mastery_before <= 0.01).sum()),
            "boundary_before_ge_0_99_N": int((usable.mastery_before >= 0.99).sum()),
            "sign_mismatch_count": int(
                (np.sign(values.to_numpy()) != np.sign(usable.reward_raw_delta.to_numpy())).sum()
            ),
        }
        for key, result in correlations.items():
            row[f"spearman_{key}_rho"] = result["rho"]
            row[f"spearman_{key}_p"] = result["p_value"]
        rows.append(row)
        attempt_means = usable.assign(value=values).groupby("attempt_id").value.mean()
        learner_means = usable.assign(value=values).groupby("learner_id").value.mean()
        detail["candidates"][name] = {
            **row,
            "correlations": correlations,
            "per_attempt_mean": {
                "N": int(len(attempt_means)),
                "mean": float(attempt_means.mean()),
                "SD": float(attempt_means.std(ddof=1)) if len(attempt_means) > 1 else None,
                "min": float(attempt_means.min()),
                "max": float(attempt_means.max()),
            },
            "per_learner_mean": {
                "N": int(len(learner_means)),
                "values": {str(k): float(v) for k, v in learner_means.items()},
            },
        }
    return pd.DataFrame(rows), detail


def select_reward(results: pd.DataFrame) -> tuple[str, str]:
    raw = results.set_index("reward").loc["RAW_DELTA"]
    head = results.set_index("reward").loc["HEADROOM_NORMALIZED"]
    stable = (
        raw.nonfinite_count == 0
        and head.nonfinite_count == 0
        and raw.sign_mismatch_count == 0
        and head.sign_mismatch_count == 0
        and raw.SD > 0
        and head.SD > 0
        and -1.0 - 1e-12 <= head["min"] <= head["max"] <= 1.0 + 1e-12
    )
    raw_dep = abs(float(raw.spearman_absolute_reward_vs_mastery_before_rho))
    head_dep = abs(float(head.spearman_absolute_reward_vs_mastery_before_rho))
    position_fields = (
        "spearman_reward_vs_turn_index_rho",
        "spearman_absolute_reward_vs_turn_index_rho",
        "spearman_reward_vs_relative_attempt_position_rho",
        "spearman_absolute_reward_vs_relative_attempt_position_rho",
    )
    head_position = max(abs(float(head[field])) for field in position_fields)
    if stable and raw_dep - head_dep >= 0.20 and head_position < 0.30:
        return (
            "HEADROOM_NORMALIZED",
            "HEADROOM_NORMALIZED preserved sign, stayed finite and bounded without tuned constants, "
            f"reduced absolute-reward mastery dependence from |rho|={raw_dep:.3f} to {head_dep:.3f}, "
            f"and introduced no strong turn/relative-position artifact (maximum |rho|={head_position:.3f}).",
        )
    if stable and raw_dep - head_dep < 0.20:
        return (
            "RAW_DELTA",
            "HEADROOM_NORMALIZED did not materially reduce mechanical mastery dependence enough to justify its transformation; RAW_DELTA retained sign, stability, and usable variation.",
        )
    return "INCONCLUSIVE", "The predeclared stability/dependence criteria did not support exactly one reward."


def reward_group_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = frame[frame.include_in_reward_experiment].copy()
    rows_band: list[dict[str, Any]] = []
    rows_position: list[dict[str, Any]] = []
    for reward, column in (("RAW_DELTA", "reward_raw_delta"), ("HEADROOM_NORMALIZED", "reward_headroom_normalized")):
        for band, subset in usable.groupby("mastery_band", observed=False):
            values = subset[column].dropna()
            rows_band.append({
                "reward": reward,
                "mastery_band": str(band),
                "N": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "absolute_mean": float(values.abs().mean()) if len(values) else np.nan,
                "Q1": float(values.quantile(.25)) if len(values) else np.nan,
                "Q3": float(values.quantile(.75)) if len(values) else np.nan,
            })
        for band, subset in usable.groupby("position_band", observed=False):
            values = subset[column].dropna()
            rows_position.append({
                "reward": reward,
                "position_band": str(band),
                "N": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "absolute_mean": float(values.abs().mean()) if len(values) else np.nan,
                "Q1": float(values.quantile(.25)) if len(values) else np.nan,
                "Q3": float(values.quantile(.75)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows_band), pd.DataFrame(rows_position)


def build_design(
    context: np.ndarray,
    actions: Sequence[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
    standardized = scaler.fit_transform(context) if fit_scaler else scaler.transform(context)
    action_onehot = np.column_stack([np.asarray(actions) == move for move in MOVES]).astype(float)
    interactions = np.hstack([standardized * action_onehot[:, [i]] for i in range(len(MOVES))])
    return np.hstack([action_onehot, standardized, interactions]), scaler


@dataclass
class AblationOutput:
    results: pd.DataFrame
    folds: pd.DataFrame
    predictions: dict[str, pd.DataFrame]


def inner_alpha(
    context: np.ndarray,
    actions: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 1.0
    splitter = GroupKFold(n_splits=min(4, len(unique_groups)))
    scores = {alpha: [] for alpha in RIDGE_GRID}
    for train, validation in splitter.split(context, target, groups):
        x_train, scaler = build_design(context[train], actions[train], fit_scaler=True)
        x_validation, _ = build_design(context[validation], actions[validation], scaler=scaler)
        for alpha in RIDGE_GRID:
            model = Ridge(alpha=alpha)
            model.fit(x_train, target[train])
            scores[alpha].append(mean_absolute_error(target[validation], model.predict(x_validation)))
    return min(RIDGE_GRID, key=lambda alpha: (float(np.mean(scores[alpha])), alpha))


def run_context_ablation(frame: pd.DataFrame, reward_column: str) -> AblationOutput:
    usable = frame[frame.include_in_context_experiment].copy().reset_index(drop=True)
    groups = usable.attempt_id.to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("Context ablation needs at least two attempts")
    splitter = GroupKFold(n_splits=min(OUTER_SPLITS, len(unique_groups)))
    actions = usable.actual_move.to_numpy()
    target = usable[reward_column].to_numpy(float)
    result_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_tables: dict[str, pd.DataFrame] = {}

    for candidate_index, preset in enumerate(CONTEXT_CANDIDATES):
        names = feature_names(preset)
        context = usable[names].to_numpy(float)
        oof = np.full(len(usable), np.nan)
        metrics: list[dict[str, Any]] = []
        for fold, (train, test) in enumerate(splitter.split(context, target, groups), 1):
            alpha = inner_alpha(context[train], actions[train], target[train], groups[train])
            x_train, scaler = build_design(context[train], actions[train], fit_scaler=True)
            x_test, _ = build_design(context[test], actions[test], scaler=scaler)
            model = Ridge(alpha=alpha)
            model.fit(x_train, target[train])
            predicted = model.predict(x_test)
            oof[test] = predicted
            fold_result = {
                "context": preset,
                "fold": fold,
                "train_N": int(len(train)),
                "test_N": int(len(test)),
                "test_attempts": json.dumps(sorted(set(groups[test])), separators=(",", ":")),
                "selected_alpha": alpha,
                "MAE": float(mean_absolute_error(target[test], predicted)),
                "RMSE": float(mean_squared_error(target[test], predicted) ** .5),
                "R2": float(r2_score(target[test], predicted)) if len(test) >= 2 and np.var(target[test]) > 1e-12 else np.nan,
            }
            metrics.append(fold_result)
            fold_rows.append(fold_result)
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Incomplete OOF predictions for {preset}")
        metric_frame = pd.DataFrame(metrics)
        row: dict[str, Any] = {
            "candidate_order": candidate_index,
            "context": preset,
            "blocks": "+".join(canonical_blocks(preset)),
            "context_dimension": len(names),
            "ordered_features": json.dumps(names, separators=(",", ":")),
            "N": int(len(usable)),
            "attempt_count": int(len(unique_groups)),
            "fold_count": int(len(metric_frame)),
            "MAE_mean": float(metric_frame.MAE.mean()),
            "MAE_SD": float(metric_frame.MAE.std(ddof=1)),
            "MAE_SE": float(metric_frame.MAE.std(ddof=1) / math.sqrt(len(metric_frame))),
            "RMSE_mean": float(metric_frame.RMSE.mean()),
            "RMSE_SD": float(metric_frame.RMSE.std(ddof=1)),
            "RMSE_SE": float(metric_frame.RMSE.std(ddof=1) / math.sqrt(len(metric_frame))),
            "R2_mean": float(metric_frame.R2.mean()),
            "R2_SD": float(metric_frame.R2.std(ddof=1)),
            "R2_SE": float(metric_frame.R2.std(ddof=1) / math.sqrt(metric_frame.R2.notna().sum())),
            "OOF_MAE": float(mean_absolute_error(target, oof)),
            "OOF_RMSE": float(mean_squared_error(target, oof) ** .5),
            "OOF_R2": float(r2_score(target, oof)),
            "ridge_grid": json.dumps(RIDGE_GRID),
            "selected_alphas": json.dumps(metric_frame.selected_alpha.tolist(), separators=(",", ":")),
        }
        result_rows.append(row)
        prediction_tables[preset] = pd.DataFrame({
            "attempt_id": usable.attempt_id,
            "action_event_id": usable.action_event_id,
            "actual": target,
            "predicted": oof,
        })
    return AblationOutput(pd.DataFrame(result_rows), pd.DataFrame(fold_rows), prediction_tables)


def select_context(results: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    best = results.sort_values(["MAE_mean", "RMSE_mean", "candidate_order"]).iloc[0]
    threshold = float(best.MAE_mean + best.MAE_SE)
    eligible = results[results.MAE_mean <= threshold + 1e-15].sort_values(
        ["context_dimension", "RMSE_mean", "candidate_order"]
    )
    selected = eligible.iloc[0]
    rationale = {
        "lowest_mae_context": best.context,
        "lowest_mae": float(best.MAE_mean),
        "lowest_mae_se": float(best.MAE_SE),
        "one_se_threshold": threshold,
        "eligible_contexts": eligible.context.tolist(),
        "selected_context": selected.context,
        "selected_dimension": int(selected.context_dimension),
        "selection_rule": "smallest dimension within one SE of lowest mean grouped MAE; then RMSE; then preregistered order",
    }
    return str(selected.context), rationale


def incremental_value(results: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "K": [("S", "S+K"), ("S+L", "S+K+L"), ("S+H", "S+K+H"), ("S+Q", "S+K+Q"), ("S+L+H+Q", "S+K+L+H+Q")],
        "L": [("S", "S+L"), ("S+K", "S+K+L"), ("S+K+H", "S+K+L+H"), ("S+K+H+Q", "S+K+L+H+Q")],
        "H": [("S", "S+H"), ("S+K", "S+K+H"), ("S+K+L", "S+K+L+H"), ("S+K+L+Q", "S+K+L+H+Q")],
        "Q": [("S", "S+Q"), ("S+K", "S+K+Q"), ("S+K+H", "S+K+H+Q"), ("S+K+L+H", "S+K+L+H+Q")],
    }
    indexed = results.set_index("context")
    fold_indexed = folds.set_index(["context", "fold"])
    rows: list[dict[str, Any]] = []
    for block, comparisons in pairs.items():
        for base, added in comparisons:
            if base not in indexed.index or added not in indexed.index:
                continue
            base_folds = fold_indexed.loc[base]
            added_folds = fold_indexed.loc[added]
            for metric in ("MAE", "RMSE", "R2"):
                delta = added_folds[metric] - base_folds[metric]
                rows.append({
                    "block_added": block,
                    "base_context": base,
                    "added_context": added,
                    "metric": metric,
                    "delta_added_minus_base": float(indexed.loc[added, f"{metric}_mean"] - indexed.loc[base, f"{metric}_mean"]),
                    "fold_delta_mean": float(delta.mean()),
                    "fold_delta_SD": float(delta.std(ddof=1)),
                    "fold_delta_SE": float(delta.std(ddof=1) / math.sqrt(delta.notna().sum())),
                    "improvement_direction": "negative" if metric in {"MAE", "RMSE"} else "positive",
                })
    return pd.DataFrame(rows)


def cluster_bootstrap(
    predictions: dict[str, pd.DataFrame],
    comparisons: list[tuple[str, str, str]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for label, first, second in comparisons:
        left = predictions[first]
        right = predictions[second]
        if not left[["attempt_id", "action_event_id"]].equals(right[["attempt_id", "action_event_id"]]):
            raise RuntimeError("OOF prediction identities differ across contexts")
        attempts = left.attempt_id.unique()
        replicate_values = {"MAE": [], "RMSE": []}
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(attempts, size=len(attempts), replace=True)
            indices = np.concatenate([np.flatnonzero(left.attempt_id.to_numpy() == attempt) for attempt in sampled])
            actual = left.actual.to_numpy()[indices]
            for metric in replicate_values:
                if metric == "MAE":
                    a = mean_absolute_error(actual, left.predicted.to_numpy()[indices])
                    b = mean_absolute_error(actual, right.predicted.to_numpy()[indices])
                else:
                    a = mean_squared_error(actual, left.predicted.to_numpy()[indices]) ** .5
                    b = mean_squared_error(actual, right.predicted.to_numpy()[indices]) ** .5
                replicate_values[metric].append(a - b)
        for metric, values_list in replicate_values.items():
            values = np.asarray(values_list)
            rows.append({
                "comparison": label,
                "first_context": first,
                "second_context": second,
                "metric": metric,
                "difference_definition": "first minus second; negative favors first",
                "attempt_count": int(len(attempts)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "CI95_low": float(np.quantile(values, .025)),
                "CI95_high": float(np.quantile(values, .975)),
                "seed": RANDOM_SEED,
            })
    return pd.DataFrame(rows)


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[frame.include_in_context_experiment]
    rows: list[dict[str, Any]] = []
    for block, features in BLOCK_FEATURES.items():
        observed = usable[f"{block}_real_observed"].astype(bool)
        for feature in features:
            rows.append({
                "block": block,
                "feature": feature,
                "analysis_N": int(len(usable)),
                "runtime_representable_N": int(np.isfinite(usable[feature]).sum()),
                "runtime_representable_percent": float(np.isfinite(usable[feature]).mean() * 100),
                "real_substantive_observed_N": int(observed.sum()),
                "real_substantive_observed_percent": float(observed.mean() * 100),
                "missing_representation": {
                    "S": "none",
                    "K": "previous delta zero + explicit missing indicator",
                    "L": "three zeros + shared missing indicator",
                    "H": "zero one-hot + missing indicator; count remains observed",
                    "Q": "four zeros + shared missing indicator",
                }[block],
            })
    return pd.DataFrame(rows)


def provenance_table(
    frame: pd.DataFrame,
    assessments: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    runtime_event_rows: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(metric: str, category: str, value: Any, denominator: Any = "", note: str = "") -> None:
        rows.append({"metric": metric, "category": category, "value": value, "denominator": denominator, "note": note})
    add("candidate_turns", "all", len(frame))
    add("valid_linked_bkt_outcomes", "all", int(frame.valid_linked_bkt_outcome.sum()), len(frame), "before scientific integrity exclusions")
    add("valid_pre_action_contexts", "runtime_representable", int(frame.pre_action_context_representable.sum()), len(frame))
    add("included_reward_experiment", "all", int(frame.include_in_reward_experiment.sum()), len(frame))
    add("included_context_experiment", "all", int(frame.include_in_context_experiment.sum()), len(frame))
    add("attempt_count", "all", frame.attempt_id.nunique())
    add("learner_count", "all", frame.learner_id.dropna().nunique())
    add("skill_count", "all", frame.skill_id.dropna().nunique())
    add("assessment_rows_separate", "all", len(assessments), note="never credited to Tutor actions")
    add("attempt_summary_rows", "all", len(attempts))
    for move, count in frame.actual_move.value_counts().reindex(MOVES, fill_value=0).items():
        add("move_count", move, int(count), len(frame))
    for mode, count in frame.runtime_mode.value_counts(dropna=False).items():
        add("runtime_mode_count", str(mode), int(count), len(frame))
    for randomized, count in frame.randomized_assignment.value_counts().items():
        add("assignment_count", "randomized" if randomized else "non_randomized", int(count), len(frame))
    for status, count in frame.resolution_status.value_counts().items():
        add("resolution_count", status, int(count), len(frame))
    for reason, count in frame.exclusion_reason.value_counts().items():
        add("exclusion_count", reason, int(count), len(frame))
    for lineage, event_rows in runtime_event_rows.items():
        add("local_runtime_event_rows", lineage, len(event_rows), note="audited; collector remains canonical analysis source")
    usable = frame[frame.include_in_context_experiment]
    combos = ("S", "K", "L", "H", "Q", "S+K", "S+L", "S+K+L", "S+K+L+H", "S+K+L+H+Q", "K+L+H+Q")
    for combo in combos:
        blocks = canonical_blocks(combo)
        substantive = np.logical_and.reduce([usable[f"{block}_real_observed"].to_numpy(bool) for block in blocks])
        add("context_block_completeness", combo, int(substantive.sum()), len(usable), "real substantive observations; runtime representation is complete via explicit indicators")
    return pd.DataFrame(rows)


def mrb1_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    usable = frame[frame.include_in_context_experiment]
    columns = [f"current_mrb1_{head.casefold()}" for head in MRB1_HEADS]
    complete = usable[columns].dropna()
    correlations: dict[str, Any] = {}
    for i, left in enumerate(MRB1_HEADS):
        for right in MRB1_HEADS[i + 1:]:
            correlations[f"{left}__{right}"] = spearman(
                complete[f"current_mrb1_{left.casefold()}"],
                complete[f"current_mrb1_{right.casefold()}"],
            )
    return {
        "complete_current_rows": int(len(complete)),
        "head_correlations": correlations,
        "role_constraint": "four separate previous-turn Q features only; never scalar reward",
    }


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.grid": True,
        "grid.alpha": .22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / name, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_provenance(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    status = frame.resolution_status.value_counts().reindex(["resolved_reward", "resolved_no_update", "censored", "failed"], fill_value=0)
    axes[0].bar(status.index, status.values, color="#4c78a8")
    axes[0].set_title("Outcome resolution")
    axes[0].set_ylabel("Tutor treatment turns")
    axes[0].tick_params(axis="x", rotation=25)
    assignment = frame.randomized_assignment.value_counts().reindex([False, True], fill_value=0)
    axes[1].bar(["Policy-bound", "Randomized"], assignment.values, color=["#9c755f", "#59a14f"])
    axes[1].set_title("Treatment assignment provenance")
    axes[1].set_ylabel("Tutor treatment turns")
    usable = frame[frame.include_in_context_experiment]
    observed = [usable[f"{block}_real_observed"].mean() * 100 for block in BLOCK_ORDER]
    axes[2].bar(BLOCK_ORDER, observed, color="#f28e2b")
    axes[2].set_ylim(0, 110)
    axes[2].set_title("Real substantive pre-action coverage")
    axes[2].set_ylabel("Included context rows (%)")
    for index, value in enumerate(observed):
        axes[2].text(index, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    fig.suptitle("Data provenance and context completeness", fontsize=15, fontweight="bold")
    fig.text(.5, .01, f"Candidate N={len(frame)}; context-analysis N={len(usable)}. Numeric context is runtime-representable via explicit missing indicators; bars show real signal coverage.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, .93))
    savefig(fig, "01_data_provenance_and_context_completeness.png")


def plot_rewards(frame: pd.DataFrame, results: pd.DataFrame) -> None:
    usable = frame[frame.include_in_reward_experiment]
    rewards = [("RAW_DELTA", "reward_raw_delta"), ("HEADROOM_NORMALIZED", "reward_headroom_normalized")]
    values = [usable[column].to_numpy(float) for _, column in rewards]
    shared_min, shared_max = min(v.min() for v in values), max(v.max() for v in values)
    pad = max((shared_max - shared_min) * .05, .01)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    bins = np.linspace(shared_min - pad, shared_max + pad, 25)
    for ax, (name, column) in zip(axes, rewards, strict=True):
        ax.hist(usable[column], bins=bins, color="#4c78a8", alpha=.82, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Reward (posterior belief-update scale)")
        ax.set_ylabel("Turn count")
    fig.suptitle("Reward distribution comparison (shared axes)", fontsize=15, fontweight="bold")
    fig.text(.5, .01, f"N={len(usable)} per candidate. {REWARD_NOTE}.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, .92))
    savefig(fig, "02_reward_distribution_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, (name, column) in zip(axes, rewards, strict=True):
        ax.scatter(usable.mastery_before, usable[column], s=32, alpha=.72, color="#4c78a8", edgecolor="white", linewidth=.3)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("Reward")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(shared_min - pad, shared_max + pad)
    fig.suptitle("Reward versus starting mastery (shared axes)", fontsize=15, fontweight="bold", y=.97)
    fig.supxlabel("Mastery before action (BKT posterior probability)", y=.075)
    fig.text(.5, .015, f"N={len(usable)}. {REWARD_NOTE}.", ha="center")
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.84, wspace=.08)
    savefig(fig, "03_reward_vs_mastery_before.png")

    correlation_labels = ["signed vs mastery", "magnitude vs mastery", "signed vs turn", "magnitude vs turn", "signed vs relative position", "magnitude vs relative position"]
    fields = [
        "spearman_reward_vs_mastery_before_rho",
        "spearman_absolute_reward_vs_mastery_before_rho",
        "spearman_reward_vs_turn_index_rho",
        "spearman_absolute_reward_vs_turn_index_rho",
        "spearman_reward_vs_relative_attempt_position_rho",
        "spearman_absolute_reward_vs_relative_attempt_position_rho",
    ]
    result_index = results.set_index("reward")
    x = np.arange(len(fields)); width = .36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, name, color in ((-.5, "RAW_DELTA", "#4c78a8"), (.5, "HEADROOM_NORMALIZED", "#f28e2b")):
        ax.bar(x + offset * width, [result_index.loc[name, field] for field in fields], width, label=name, color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, correlation_labels, rotation=22, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-1, 1)
    ax.set_title("Reward dependence diagnostics")
    ax.legend()
    fig.text(.5, .01, f"N={len(usable)} paired turns; correlations are descriptive under clustered observations. {REWARD_NOTE}.", ha="center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    savefig(fig, "04_reward_mastery_dependence.png")

    bands = ["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (name, column) in zip(axes, rewards, strict=True):
        data = [usable.loc[usable.mastery_band.astype(str) == band, column].abs().to_numpy() for band in bands]
        ax.boxplot(data, tick_labels=bands, showfliers=True)
        ax.set_title(name)
        ax.set_xlabel("Starting-mastery band")
        ax.set_ylabel("Absolute reward magnitude")
    fig.suptitle("Reward magnitude by starting-mastery band", fontsize=15, fontweight="bold")
    fig.text(.5, .01, f"N={len(usable)} total; boxes show IQR and median. {REWARD_NOTE}.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, .92))
    savefig(fig, "05_reward_by_mastery_band.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    for ax, (name, column) in zip(axes, rewards, strict=True):
        grouped = usable.groupby("turn_index_within_attempt")[column].agg(["mean", "count", "std"]).reset_index()
        se = grouped["std"].fillna(0) / np.sqrt(grouped["count"])
        ax.errorbar(grouped.turn_index_within_attempt, grouped["mean"], yerr=se, marker="o", capsize=3, color="#4c78a8")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Turn index within attempt")
        ax.set_ylabel("Mean reward")
    fig.suptitle("Reward by turn position", fontsize=15, fontweight="bold")
    fig.text(.5, .01, f"N={len(usable)}; error bars are SE across turns at each index (descriptive, not independent). {REWARD_NOTE}.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, .92))
    savefig(fig, "06_reward_by_turn_position.png")


def plot_context(
    results: pd.DataFrame,
    folds: pd.DataFrame,
    increments: pd.DataFrame,
    frame: pd.DataFrame,
    selected: str,
) -> None:
    ordered = results.sort_values("MAE_mean").reset_index(drop=True)
    def error_plot(metric: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 7))
        y = np.arange(len(ordered))
        ax.errorbar(ordered[f"{metric}_mean"], y, xerr=ordered[f"{metric}_SE"], fmt="o", capsize=3, color="#4c78a8")
        ax.set_yticks(y, ordered.context)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean outer-fold {metric} (error bars: SE across attempt-group folds)")
        ax.set_title(title)
        ax.axvline(float(ordered.iloc[0][f"{metric}_mean"]), color="#e15759", linestyle="--", linewidth=1)
        fig.text(.5, .01, f"N={int(ordered.N.iloc[0])}, {int(ordered.attempt_count.iloc[0])} attempts; nested grouped CV; lower is better for {metric}.", ha="center")
        fig.tight_layout(rect=(0, .04, 1, 1))
        savefig(fig, filename)
    error_plot("MAE", "07_context_ablation_mae.png", "Context ablation: grouped MAE")
    error_plot("RMSE", "08_context_ablation_rmse.png", "Context ablation: grouped RMSE")

    r2_ordered = results.sort_values("R2_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(r2_ordered))
    ax.errorbar(r2_ordered.R2_mean, y, xerr=r2_ordered.R2_SE, fmt="o", capsize=3, color="#59a14f")
    ax.set_yticks(y, r2_ordered.context.tolist()); ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean outer-fold R2 (error bars: SE across attempt-group folds)")
    ax.set_title("Context ablation: grouped R2")
    fig.text(.5, .01, "R2 can be strongly negative on small heterogeneous held-out attempt groups; higher is better.", ha="center")
    fig.tight_layout(rect=(0, .04, 1, 1))
    savefig(fig, "09_context_ablation_r2.png")

    baseline = results.set_index("context").loc["S"]
    delta = results.copy()
    delta["delta_MAE_vs_S"] = delta.MAE_mean - baseline.MAE_mean
    delta["delta_RMSE_vs_S"] = delta.RMSE_mean - baseline.RMSE_mean
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(delta)); width = .38
    ax.bar(x - width / 2, delta.delta_MAE_vs_S, width, label="Delta MAE", color="#4c78a8")
    ax.bar(x + width / 2, delta.delta_RMSE_vs_S, width, label="Delta RMSE", color="#f28e2b")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, delta.context, rotation=40, ha="right")
    ax.set_ylabel("Context minus S (negative improves over S)")
    ax.set_title("Grouped error change relative to S")
    ax.legend()
    fig.text(.5, .01, "Outer-fold mean differences; same folds and target for every context.", ha="center")
    fig.tight_layout(rect=(0, .04, 1, 1))
    savefig(fig, "10_context_delta_vs_S.png")

    leaders = results.sort_values("MAE_mean").head(4).context.tolist()
    if "S" not in leaders:
        leaders[-1] = "S"
    stability = folds[folds.context.isin(leaders)].pivot(index="context", columns="fold", values="MAE").reindex(leaders)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(stability, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(np.arange(len(stability)), stability.index)
    ax.set_xticks(np.arange(len(stability.columns)), [f"Fold {i}" for i in stability.columns])
    for i in range(stability.shape[0]):
        for j in range(stability.shape[1]):
            ax.text(j, i, f"{stability.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="MAE")
    ax.set_title("Fold stability for leading contexts")
    fig.text(.5, .01, "Each column holds out entire attempts; lower is better.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, 1))
    savefig(fig, "11_context_fold_stability.png")

    summary = increments[increments.metric == "MAE"].groupby("block_added").agg(
        mean_delta=("delta_added_minus_base", "mean"),
        min_delta=("delta_added_minus_base", "min"),
        max_delta=("delta_added_minus_base", "max"),
    ).reindex(BLOCK_ORDER[1:])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.index, summary.mean_delta, color="#76b7b2")
    ax.errorbar(
        np.arange(len(summary)),
        summary.mean_delta,
        yerr=np.vstack([summary.mean_delta - summary.min_delta, summary.max_delta - summary.mean_delta]),
        fmt="none", color="black", capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean matched delta MAE (added minus base)")
    ax.set_title("Context-block incremental value across matched pairs")
    fig.text(.5, .01, "Bars average valid matched comparisons; whiskers span comparison minima/maxima. Negative favors adding the block.", ha="center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    savefig(fig, "12_context_block_incremental_value.png")

    usable = frame[frame.include_in_context_experiment]
    observed = np.array([usable[f"{block}_real_observed"].sum() for block in BLOCK_ORDER])
    missing = len(usable) - observed
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(BLOCK_ORDER, observed, label="Real substantive observation", color="#59a14f")
    ax.bar(BLOCK_ORDER, missing, bottom=observed, label="Missing represented by runtime contract", color="#d9d9d9")
    ax.set_ylabel("Included context rows")
    ax.set_title("Real context-block coverage and missingness")
    ax.legend()
    for i, value in enumerate(observed):
        ax.text(i, value / 2 if value else 1, f"{value}/{len(usable)}", ha="center", va="center", fontsize=9)
    fig.text(.5, .01, "No future outcomes, synthetic learner signals, or outcome-derived imputations were used.", ha="center")
    fig.tight_layout(rect=(0, .05, 1, 1))
    savefig(fig, "13_context_missingness.png")


def plot_architecture(selected_context: str, selected_reward: str) -> None:
    blocks = canonical_blocks(selected_context)
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")
    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08,rounding_size=.12", facecolor=color, edgecolor="#334", linewidth=1.2)
        ax.add_patch(patch); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", wrap=True, fontsize=9.5)
    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#334"))
        if label: ax.text((x1 + x2) / 2 + .12, (y1 + y2) / 2, label, fontsize=8, color="#444")
    box(3, 14.6, 4, .7, "Learner message", "#e8f1fb")
    box(3, 13.5, 4, .7, "Explicit learner agency layer\n(request != inferred state)", "#fff2cc")
    box(3, 12.4, 4, .7, "Frozen MD7-R2-TELL-C1 Epoch 2", "#d9ead3")
    context_description = "selector probabilities only" if selected_context == "S" else f"blocks {', '.join(blocks)}"
    box(.4, 10.8, 3.8, 1.0, f"PRE-ACTION CONTEXT\n{selected_context} ({context_description})", "#cfe2f3")
    box(5.8, 10.8, 3.8, 1.0, "Cold start: raw MD7\nproportional randomized warm-start\n(no posterior mutation)", "#fce5cd")
    box(3, 9.5, 4, .8, "Future direct disjoint Turn-LinTS\n(or warm-start assignment now)", "#d9d2e9")
    box(3, 8.3, 4, .7, "Selected move\ngeneric | probing | focus | telling", "#ead1dc")
    box(3, 7.2, 4, .7, "Existing Tutor realization contract", "#e2f0d9")
    box(3, 6.1, 4, .7, "Tutor response", "#e8f1fb")
    box(.4, 4.7, 3.8, .9, "Current MRB1 auxiliary outcome\npost-action; never scalar reward", "#fff2cc")
    box(5.8, 4.7, 3.8, .9, "Learner response\nsemantic evidence category", "#f4cccc")
    box(3, 3.3, 4, .9, "Student Modeling / BKT\nvalid mathematical or knowledge-state evidence", "#d9ead3")
    box(3, 1.9, 4, .8, f"Selected reward: {selected_reward}\n{REWARD_NOTE}", "#cfe2f3")
    box(3, .5, 4, .8, "Future immediate LinTS update\nLIVE only; zero updates during warm-start", "#d9d2e9")
    for a, b in ((14.6, 14.2), (13.5, 13.1), (12.4, 11.8), (10.8, 10.3), (9.5, 9.0), (8.3, 7.9), (7.2, 6.8), (6.1, 5.6), (4.7, 4.2), (3.3, 2.7), (1.9, 1.3)):
        arrow(5, a, 5, b)
    arrow(4.2, 11.3, 4.9, 10.3, "x_t")
    arrow(5.8, 11.3, 5.1, 10.3)
    arrow(5, 6.1, 2.3, 5.6)
    arrow(5, 6.1, 7.7, 5.6)
    arrow(7.7, 4.7, 5.8, 4.2)
    ax.text(2.3, 4.45, "logged auxiliary only; Q not selected", ha="center", fontsize=8, color="#555")
    ax.text(5, 15.8, "AdaptMath Turn-LinTS v1 final architecture", ha="center", fontsize=16, fontweight="bold")
    ax.text(5, .08, "Intentionally absent: no tau gate, no eligibility threshold, no P1/P2 anchor, no handwritten pedagogical action rules.", ha="center", fontsize=8.5)
    savefig(fig, "14_final_architecture.png")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}")
    labels = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for values in view.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in values)
            + " |"
        )
    return "\n".join(lines)


def write_reports(
    frame: pd.DataFrame,
    provenance: pd.DataFrame,
    reward_results: pd.DataFrame,
    reward_detail: dict[str, Any],
    reward_decision: str,
    reward_reason: str,
    context_results: pd.DataFrame,
    context_selection: str,
    context_rule: dict[str, Any],
    bootstrap: pd.DataFrame,
    increments: pd.DataFrame,
    coverage: pd.DataFrame,
    mrb1: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    selected_row = context_results.set_index("context").loc[context_selection]
    s_row = context_results.set_index("context").loc["S"]
    full_row = context_results.set_index("context").loc["S+K+L+H+Q"]
    selected_features = feature_names(context_selection)
    reward_column = "reward_headroom_normalized" if reward_decision == "HEADROOM_NORMALIZED" else "reward_raw_delta"
    randomized = frame[frame.randomized_assignment]
    randomized_valid = randomized[randomized.include_in_context_experiment]
    exclusion_counts = {str(k): int(v) for k, v in frame.exclusion_reason.value_counts().items() if k != "included"}
    action_support = {move: int((frame[frame.include_in_context_experiment].actual_move == move).sum()) for move in MOVES}
    learner_coverage = frame[frame.include_in_context_experiment].L_real_observed
    current_mode = runtime_state.get("turnLinTSMode", "unknown")
    live_updates = 0
    for state_path in BACKEND_ROOT.glob("runtime/turn_lints_*/policy_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        live_updates += int(state.get("update_count", 0) or 0)

    support_strength = "qualified_limited"
    decision_label = "B. FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT"
    limitations = [
        f"Only {int(learner_coverage.sum())}/{len(learner_coverage)} included rows contain real non-missing L signals; missing-indicator representation permits evaluation but does not establish substantive L value.",
        f"Telling has {action_support['telling']} included observations, so telling-specific reward interactions are not empirically established.",
        f"Only {len(randomized_valid)} scientifically valid randomized observations from {randomized_valid.attempt_id.nunique()} attempt(s) exist; insufficient for standalone context selection.",
        "The primary context evidence is observational and policy-bound; no causal action-effect claim is made.",
        "Grouped R2 is sensitive to small heterogeneous held-out attempt groups and is interpreted with MAE/RMSE and fold stability.",
    ]
    manifest: dict[str, Any] = {
        "architecture_version": "adaptmath_turn_lints_v1_final",
        "freeze_status": support_strength,
        "algorithm": "direct_disjoint_linear_thompson_sampling",
        "arms": list(MOVES),
        "context_preset": context_selection,
        "ordered_context_features": selected_features,
        "context_dimension": len(selected_features),
        "context_schema_id": context_schema_id(context_selection),
        "reward_mode": "headroom_normalized" if reward_decision == "HEADROOM_NORMALIZED" else "raw_delta",
        "reward_decision_label": reward_decision,
        "selector_name": SELECTOR_NAME,
        "selector_sha256": SELECTOR_SHA256,
        "cold_start_behavior_policy": "md7_r2_probability_proportional_v1",
        "known_behavior_propensity_logging": True,
        "mrb1_role": "previous-turn four-head auxiliary context only if selected; never scalar reward",
        "mrb1_selected_in_context": "Q" in canonical_blocks(context_selection),
        "student_model_role": "pre-action BKT state and prior valid learner-signal probabilities per selected blocks; linked valid BKT posterior update supplies reward",
        "formal_assessment_role": "separate external policy-health signal; never Tutor-action reward",
        "learner_agency_semantics": "explicit interaction requests are handled above the policy, logged non-randomized when overriding; inferred learner state never maps to handwritten actions",
        "turn_update_timing": "immediate after valid linked BKT evidence in future LIVE mode",
        "tau_gate": "none",
        "action_eligibility_threshold": "none",
        "p1_anchor": "inactive",
        "p2": "not_used",
        "handwritten_pedagogical_action_rules": "none",
        "tutor_realization": "existing generic/probing/focus/telling contract",
        "experiment_artifact_path": "pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1",
        "analysis_script_sha256": None,
        "derived_dataset_sha256": None,
        "architecture_manifest_sha256": None,
        "architecture_manifest_hash_note": "Self-hash is stored in FINAL_ARCHITECTURE.sha256 after this JSON is finalized.",
        "selection_metrics": {
            "grouped_MAE_mean": float(selected_row.MAE_mean),
            "grouped_MAE_SE": float(selected_row.MAE_SE),
            "grouped_RMSE_mean": float(selected_row.RMSE_mean),
            "grouped_RMSE_SE": float(selected_row.RMSE_SE),
            "grouped_R2_mean": float(selected_row.R2_mean),
            "grouped_R2_SE": float(selected_row.R2_SE),
            "delta_MAE_vs_S": float(selected_row.MAE_mean - s_row.MAE_mean),
            "delta_RMSE_vs_S": float(selected_row.RMSE_mean - s_row.RMSE_mean),
            "delta_MAE_vs_full": float(selected_row.MAE_mean - full_row.MAE_mean),
            "delta_RMSE_vs_full": float(selected_row.RMSE_mean - full_row.RMSE_mean),
        },
        "current_runtime_at_analysis": {
            "mode": current_mode,
            "context_blocks": runtime_state.get("turnLinTSContextBlocks"),
            "reward_mode": runtime_state.get("turnLinTSRewardMode"),
            "live_update_count_across_direct_states": live_updates,
        },
        "randomized_warmstart": {
            "treatment_count": int(len(randomized)),
            "scientifically_valid_observation_count": int(len(randomized_valid)),
            "attempt_count": int(randomized_valid.attempt_id.nunique()),
            "standalone_context_selection": "insufficient",
        },
        "limitations": limitations,
        "decision_label": decision_label,
    }

    # Hash values are written after the derived CSV and stable analysis source exist.
    manifest["analysis_script_sha256"] = sha256(Path(__file__))
    manifest["derived_dataset_sha256"] = sha256(OUTPUT_DIR / "derived_analysis_dataset.csv")
    (OUTPUT_DIR / "FINAL_ARCHITECTURE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = sha256(OUTPUT_DIR / "FINAL_ARCHITECTURE.json")
    (OUTPUT_DIR / "FINAL_ARCHITECTURE.sha256").write_text(f"{manifest_hash}  FINAL_ARCHITECTURE.json\n", encoding="utf-8")

    reward_report = f"""# Reward decision

## Predeclared criteria

{chr(10).join(f'{i}. {value}' for i, value in enumerate(REWARD_DECISION_CRITERIA, 1))}

## Results

{markdown_table(reward_results, ['reward','N','min','P1','P5','Q1','median','mean','Q3','P95','P99','max','SD','IQR','positive_proportion','zero_proportion','negative_proportion'])}

## Selected option

**{reward_decision}**

{reward_reason}

The reward is a {REWARD_NOTE}. Observations are clustered within attempts and learners; correlations are descriptive, not independent causal evidence.
"""
    (OUTPUT_DIR / "REWARD_DECISION.md").write_text(reward_report, encoding="utf-8")

    final_architecture_md = f"""# AdaptMath Turn-LinTS v1 FINAL ARCHITECTURE

- Frozen base selector: {SELECTOR_NAME}, SHA-256 `{SELECTOR_SHA256}`
- Bandit: direct disjoint linear Thompson Sampling
- Arms: {', '.join(MOVES)}
- Final context: **{context_selection}**, dimension **{len(selected_features)}**
- Ordered features: {', '.join(f'`{name}`' for name in selected_features)}
- Final reward: **{reward_decision}** (`{manifest['reward_mode']}`)
- Cold start: raw MD7 proportional randomized warm-start (`md7_r2_probability_proportional_v1`), known propensities logged
- P1 anchor: inactive; P2: not used; tau gate: none
- Handwritten pedagogical action rules/action filtering: none
- MRB1: {'previous-turn four-head Q context' if 'Q' in canonical_blocks(context_selection) else 'auxiliary diagnostic only'}; never scalar reward
- Formal assessment: separate external policy-health signal
- Learner agency: explicit interaction requests are above-policy and non-randomized when overriding; inferred learner state does not trigger action rules
- Turn update: immediate after valid linked BKT evidence in future LIVE mode
- LIVE activation: not authorized by this freeze; current direct posterior updates remain {live_updates}

Support is qualified because action support is {action_support}, real L coverage is {int(learner_coverage.sum())}/{len(learner_coverage)}, and randomized-only evidence is insufficient for standalone selection.

Manifest SHA-256: `{manifest_hash}`

**{decision_label}**
"""
    (OUTPUT_DIR / "FINAL_ARCHITECTURE.md").write_text(final_architecture_md, encoding="utf-8")

    decision_evidence = f"""# Decision evidence

## REWARD DECISION

### QUESTION
Which BKT posterior-update transformation should be the Turn-LinTS scalar reward?

### CANDIDATES
RAW_DELTA and HEADROOM_NORMALIZED.

### DATA
{int(frame.include_in_reward_experiment.sum())} scientifically valid linked turns from {frame.loc[frame.include_in_reward_experiment, 'attempt_id'].nunique()} attempts; corrupt/no-evidence updates and contaminated downstream states excluded.

### TEST
Predeclared sign, mastery-dependence, position, boundary, numerical-stability, and variation criteria. Spearman results are descriptive under clustering.

### RESULT
{markdown_table(reward_results, ['reward','N','mean','SD','spearman_absolute_reward_vs_mastery_before_rho','spearman_absolute_reward_vs_turn_index_rho','nonfinite_count','sign_mismatch_count'])}

### VISUAL EVIDENCE
Figures 02-06.

### SELECTED OPTION
**{reward_decision}**

### WHY IT WON
{reward_reason}

### WHAT THIS DOES NOT PROVE
It does not establish measured learning or a causal move effect.

### LIMITATIONS
BKT is a model posterior; observations cluster within attempts/learners.

## CONTEXT DECISION

### QUESTION
Which pre-action block preset has the best supported grouped generalization?

### CANDIDATES
{', '.join(CONTEXT_CANDIDATES)}.

### DATA
{int(frame.include_in_context_experiment.sum())} turns, {frame.loc[frame.include_in_context_experiment, 'attempt_id'].nunique()} attempts, action support {action_support}.

### TEST
Nested GroupKFold by attempt, fold-local standardization, inner-only ridge tuning on {RIDGE_GRID}; one-standard-error parsimony rule.

### RESULT
{markdown_table(context_results.sort_values('MAE_mean'), ['context','context_dimension','MAE_mean','MAE_SE','RMSE_mean','RMSE_SE','R2_mean','R2_SE'])}

### VISUAL EVIDENCE
Figures 07-13.

### SELECTED OPTION
**{context_selection}**

### WHY IT WON
Lowest-MAE context was {context_rule['lowest_mae_context']} at {context_rule['lowest_mae']:.4f}; the one-SE threshold was {context_rule['one_se_threshold']:.4f}. {context_selection} was the smallest eligible preset under the predeclared rule.

### WHAT THIS DOES NOT PROVE
It does not establish causal action heterogeneity or that sparsely observed blocks are useless.

### LIMITATIONS
{chr(10).join(f'- {item}' for item in limitations)}

## COLD-START DECISION

### QUESTION
How are treatments assigned before a learned posterior is justified?

### CANDIDATES
Previously frozen decision; not reopened here.

### DATA
{len(randomized)} logged randomized treatments, {len(randomized_valid)} scientifically valid after provenance/evidence audit.

### TEST
Lineage, propensity-vector, assignment-source, and zero-posterior-update audit.

### RESULT
Known-propensity raw MD7 probability-proportional assignment is implemented; standalone context selection is insufficient.

### VISUAL EVIDENCE
Figure 01 and randomized rows in the derived dataset.

### SELECTED OPTION
`md7_r2_probability_proportional_v1` (frozen prior decision).

### WHY IT WON
Not re-adjudicated; retained as the established cold-start contract.

### WHAT THIS DOES NOT PROVE
It does not justify LIVE posterior activation.

### LIMITATIONS
Scientifically valid randomized evidence is sparse.

## MRB1 ROLE

### QUESTION
Does previous MRB1 earn Q context inclusion?

### CANDIDATES
Matched with/without-Q presets; four heads remain separate.

### DATA
{mrb1['complete_current_rows']} included rows with complete current four-head MRB1 outcomes.

### TEST
Matched grouped deltas, including S+K+H versus S+K+H+Q and S+K+L+H versus full.

### RESULT
See `context_block_incremental_value.csv`; selected context {'includes' if 'Q' in canonical_blocks(context_selection) else 'does not include'} Q.

### VISUAL EVIDENCE
Figures 10 and 12.

### SELECTED OPTION
{'Previous-turn four-head Q context' if 'Q' in canonical_blocks(context_selection) else 'Auxiliary critic/diagnostic only for v1'}.

### WHY IT WON
Generated by the same grouped one-SE context rule, not by MRB1 semantics.

### WHAT THIS DOES NOT PROVE
MRB1 is not a scalar reward or ground-truth learning measure.

### LIMITATIONS
MRB1 heads are correlated and observations are policy-bound.

## STUDENT-MODELING ROLE

### QUESTION
Does real previous learner-state L evidence support retention?

### CANDIDATES
Matched models with/without L under the exact runtime missing representation.

### DATA
Real L non-missing: {int(learner_coverage.sum())}/{len(learner_coverage)} ({learner_coverage.mean()*100:.1f}%).

### TEST
Grouped matched ablations; no future or synthetic probabilities.

### RESULT
L contribution is {'retained by the selected preset but only weakly established' if 'L' in canonical_blocks(context_selection) else 'not established with available data'}.

### VISUAL EVIDENCE
Figures 12-13.

### SELECTED OPTION
{'Retain L in v1 policy vector' if 'L' in canonical_blocks(context_selection) else 'Do not include L in the frozen v1 policy vector; keep diagnostic logging'}.

### WHY IT WON
The result follows grouped predictive evidence and one-SE parsimony.

### WHAT THIS DOES NOT PROVE
It does not prove learner signals are useless.

### LIMITATIONS
Substantive L coverage is sparse.
"""
    (OUTPUT_DIR / "DECISION_EVIDENCE.md").write_text(decision_evidence, encoding="utf-8")

    report = f"""# Turn-LinTS final architecture selection v1

## Outcome

Reward **{reward_decision}** and context **{context_selection}** (dimension {len(selected_features)}) are frozen with qualified/limited support. This is an architecture freeze, not LIVE activation.

## Data integrity

- Candidate Tutor treatment turns: {len(frame)}
- Valid linked BKT outcomes before scientific exclusions: {int(frame.valid_linked_bkt_outcome.sum())}
- Included reward/context turns: {int(frame.include_in_reward_experiment.sum())}/{int(frame.include_in_context_experiment.sum())}
- Attempts/learners/skills: {frame.attempt_id.nunique()}/{frame.learner_id.nunique()}/{frame.skill_id.nunique()}
- Exclusions: {json.dumps(exclusion_counts, sort_keys=True)}
- Current runtime at analysis: {current_mode}, context {runtime_state.get('turnLinTSContextBlocks')}, reward {runtime_state.get('turnLinTSRewardMode')}

Pure acknowledgement, interaction-management, echo, and problem-repetition behavioural-proxy updates were excluded, together with downstream rows whose mastery state was contaminated. Authoritative logs were not changed.

## Reward

{markdown_table(reward_results, ['reward','N','min','Q1','median','mean','Q3','max','SD','spearman_absolute_reward_vs_mastery_before_rho','spearman_absolute_reward_vs_turn_index_rho'])}

{reward_reason}

## Context

{markdown_table(context_results.sort_values('MAE_mean'), ['context','context_dimension','MAE_mean','MAE_SE','RMSE_mean','RMSE_SE','R2_mean','R2_SE','OOF_MAE','OOF_RMSE','OOF_R2'])}

The selected preset is **{context_selection}** with ordered features {selected_features}. Relative to S, delta MAE={selected_row.MAE_mean - s_row.MAE_mean:+.4f} and delta RMSE={selected_row.RMSE_mean - s_row.RMSE_mean:+.4f}. Relative to full context, delta MAE={selected_row.MAE_mean - full_row.MAE_mean:+.4f} and delta RMSE={selected_row.RMSE_mean - full_row.RMSE_mean:+.4f}.

## Randomized versus observational evidence

Randomized warm-start treatments: {len(randomized)}; scientifically valid randomized observations: {len(randomized_valid)} from {randomized_valid.attempt_id.nunique()} attempt(s). This is **insufficient for standalone context selection**. Primary results are predictive observational evidence and do not identify causal action effects.

## Safety

Neural training=0; protected MathDial final test use=0; MRBench V3 official test use=0; authoritative historical rewrites=0; external Tutor API calls=0; LIVE posterior activations=0.

## Limitations

{chr(10).join(f'- {item}' for item in limitations)}

**{decision_label}**
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "architecture": manifest,
        "architecture_manifest_sha256": manifest_hash,
        "reward": {"selected": reward_decision, "reason": reward_reason, **reward_detail},
        "context": {
            "selected": context_selection,
            "ordered_features": selected_features,
            "dimension": len(selected_features),
            "selection_rule_result": context_rule,
            "metrics": {key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer,)) else value) for key, value in selected_row.to_dict().items()},
        },
        "data": {
            "candidate_turns": len(frame),
            "valid_linked_bkt_outcomes": int(frame.valid_linked_bkt_outcome.sum()),
            "included_reward": int(frame.include_in_reward_experiment.sum()),
            "included_context": int(frame.include_in_context_experiment.sum()),
            "attempts": int(frame.attempt_id.nunique()),
            "learners": int(frame.learner_id.nunique()),
            "skills": int(frame.skill_id.nunique()),
            "exclusions": exclusion_counts,
            "action_support_included": action_support,
        },
        "runtime": {"current_mode": current_mode, "live_update_count": live_updates},
        "randomized_warmstart": manifest["randomized_warmstart"],
        "mrb1": mrb1,
        "bootstrap": json.loads(bootstrap.to_json(orient="records")),
        "decision_label": decision_label,
        "safety": {
            "neural_training": 0,
            "protected_mathdial_final_test_use": 0,
            "mrbench_v3_official_test_use": 0,
            "authoritative_historical_rewrites": 0,
            "external_tutor_api_calls": 0,
            "live_activation": 0,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {"manifest": manifest, "manifest_hash": manifest_hash, "summary": summary}


def main() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if sha256(SELECTOR_MODEL) != SELECTOR_SHA256:
        raise RuntimeError("Frozen selector hash mismatch")
    raw_rows = load_jsonl(TURN_OUTCOMES)
    assessments = load_jsonl(ASSESSMENT_OUTCOMES)
    attempts = load_jsonl(ATTEMPT_SUMMARIES)
    runtime_paths = {
        "turn_lints_shadow_full_raw_v1": BACKEND_ROOT / "runtime" / "turn_lints_shadow_full_raw_v1" / "turn_events.jsonl",
        "turn_lints_randomized_warmstart_v1": BACKEND_ROOT / "runtime" / "turn_lints_randomized_warmstart_v1" / "turn_events.jsonl",
        "legacy_direct_fixture": BACKEND_ROOT / "runtime" / "adaptive_turn_lints_v1_real" / "turn_events.jsonl",
    }
    runtime_rows = {name: load_jsonl(path) for name, path in runtime_paths.items()}
    collector_ids = {nested(row, "identity", "action_event_id") for row in raw_rows}
    for lineage in ("turn_lints_shadow_full_raw_v1", "turn_lints_randomized_warmstart_v1"):
        missing = {row.get("action_event_id") for row in runtime_rows[lineage]} - collector_ids
        if missing:
            raise RuntimeError(f"Runtime events missing from collector: {sorted(missing)}")

    frame = build_derived_dataset(raw_rows)
    frame.to_csv(OUTPUT_DIR / "derived_analysis_dataset.csv", index=False)
    provenance = provenance_table(frame, assessments, attempts, runtime_rows)
    provenance.to_csv(OUTPUT_DIR / "data_provenance.csv", index=False)
    coverage = feature_coverage(frame)
    coverage.to_csv(OUTPUT_DIR / "context_feature_coverage.csv", index=False)

    reward_results, reward_detail = reward_statistics(frame)
    reward_decision, reward_reason = select_reward(reward_results)
    if reward_decision == "INCONCLUSIVE":
        reward_column = "reward_raw_delta"
    else:
        reward_column = "reward_headroom_normalized" if reward_decision == "HEADROOM_NORMALIZED" else "reward_raw_delta"
    reward_results.to_csv(OUTPUT_DIR / "reward_experiment_results.csv", index=False)
    (OUTPUT_DIR / "reward_summary.json").write_text(
        json.dumps({"selected": reward_decision, "reason": reward_reason, **reward_detail}, indent=2) + "\n",
        encoding="utf-8",
    )
    reward_bands, reward_positions = reward_group_tables(frame)
    reward_bands.to_csv(OUTPUT_DIR / "reward_by_mastery_band.csv", index=False)
    reward_positions.to_csv(OUTPUT_DIR / "reward_by_turn_position.csv", index=False)

    ablation = run_context_ablation(frame, reward_column)
    selected_context, context_rule = select_context(ablation.results)
    ablation.results.to_csv(OUTPUT_DIR / "context_ablation_results.csv", index=False)
    ablation.folds.to_csv(OUTPUT_DIR / "context_fold_results.csv", index=False)
    increments = incremental_value(ablation.results, ablation.folds)
    increments.to_csv(OUTPUT_DIR / "context_block_incremental_value.csv", index=False)
    ranking = ablation.results.sort_values(["MAE_mean", "RMSE_mean", "candidate_order"]).context.tolist()
    larger = next(name for name in ranking if ablation.results.set_index("context").loc[name, "context_dimension"] > 4)
    second = next(name for name in ranking if name != selected_context)
    comparisons = [
        ("best_larger_context_vs_S", larger, "S"),
        ("selected_context_vs_second_best", selected_context, second),
        ("selected_context_vs_full", selected_context, "S+K+L+H+Q"),
    ]
    bootstrap = cluster_bootstrap(ablation.predictions, comparisons)
    bootstrap.to_csv(OUTPUT_DIR / "context_bootstrap_comparisons.csv", index=False)
    mrb1 = mrb1_analysis(frame)
    runtime_state = json.loads(LOCAL_RUNTIME_STATE.read_text(encoding="utf-8-sig")) if LOCAL_RUNTIME_STATE.exists() else {}

    setup_style()
    plot_provenance(frame)
    plot_rewards(frame, reward_results)
    plot_context(ablation.results, ablation.folds, increments, frame, selected_context)
    plot_architecture(selected_context, reward_decision)
    result = write_reports(
        frame, provenance, reward_results, reward_detail, reward_decision,
        reward_reason, ablation.results, selected_context, context_rule,
        bootstrap, increments, coverage, mrb1, runtime_state,
    )
    print(json.dumps({
        "reward": reward_decision,
        "context": selected_context,
        "dimension": len(feature_names(selected_context)),
        "included_reward_N": int(frame.include_in_reward_experiment.sum()),
        "included_context_N": int(frame.include_in_context_experiment.sum()),
        "manifest_sha256": result["manifest_hash"],
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
