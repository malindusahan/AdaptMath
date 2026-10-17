"""Audited synthetic warm-start for the frozen AdaptMath Turn-LinTS v1.

This script is deterministic.  It reads historical derived artifacts only,
reconstructs L from the preceding learner message with the frozen local
detectors, creates exactly 500 down-weighted initialization observations,
and prepares (but never itself activates) the isolated LIVE lineage.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MOVE_ROOT = ROOT / "pedagogical-move-selection"
STUDENT_ROOT = ROOT / "student-modeling"
BACKEND_ROOT = ROOT / "adaptive-math-tutor" / "backend"
EVIDENCE = MOVE_ROOT / "results" / "turn_lints_final_architecture_selection_v1"
ARCH_ROOT = MOVE_ROOT / "results" / "turn_lints_final_architecture_skl_freeze_v1"
LIVE_ROOT = BACKEND_ROOT / "runtime" / "turn_lints_live_skl_final_v1"
FIGURES = HERE / "figures"

for import_root in (MOVE_ROOT, STUDENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.detector_service import DetectorService  # noqa: E402
from src.self_improvement.context_builder import MOVE_ORDER  # noqa: E402
from src.self_improvement.turn_lints_architecture_v1 import (  # noqa: E402
    ORDERED_CONTEXT_FEATURES,
    SELECTOR_NAME,
    SELECTOR_SHA256,
)
from src.self_improvement.turn_lints_context import (  # noqa: E402
    context_schema_id,
    feature_names_for_blocks,
)
from src.self_improvement.turn_lints_live_lineage import (  # noqa: E402
    ARCHITECTURE_MANIFEST_SHA256,
    LIVE_LINEAGE_SCHEMA,
    SYNTHETIC_GENERATOR_VERSION,
)
from src.self_improvement.turn_lints_policy import (  # noqa: E402
    DIRECT_ARMS,
    DirectTurnLinTS,
    save_turn_lints_state,
)
from src.self_improvement.turn_lints_reward import turn_rewards  # noqa: E402


MASTER_SEED = 20260829
BOOTSTRAP_SEED = 20260830
ACTION_SEED = 20260831
THETA_SEED = 20260832
NOISE_SEED = 20260833
AUDIT_SEED = 20260834
N_SYNTHETIC = 500
INITIALIZATION_WEIGHT = 0.02
EFFECTIVE_SAMPLE_SIZE = N_SYNTHETIC * INITIALIZATION_WEIGHT
REAL_UPDATE_WEIGHT = 1.0
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20
MC_DRAWS = 400
FIXED_POLICY_CREATED_AT = "2026-08-29T00:00:00Z"
FEATURES = list(ORDERED_CONTEXT_FEATURES)
S_FEATURES = FEATURES[:4]
K_FEATURES = FEATURES[4:7]
L_FEATURES = FEATURES[7:]
ARCH_MANIFEST = ARCH_ROOT / "FINAL_ARCHITECTURE.json"
EXPECTED_OLD_HASHES = {
    "derived_analysis_dataset.csv": "d8f219b7e0cf351fb0bc2fc460f5183ff057a87a8fdce8d372ff2ae27dd67786",
    "context_ablation_results.csv": "c411e48c6b9983ba36c0944f4157abd9d058f7a462d74d345583588727b1a7b0",
    "context_bootstrap_comparisons.csv": "cd5007259ea88b5d6c10c08ddd4a85debd14230e7da9feabf2383c5737ce0b87",
    "context_feature_coverage.csv": "7f3a51ea0fd239d16dbd14dffd5dcebb26623a20b66e714dea730bd753773e52",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ) + "\n",
        encoding="utf-8",
    )


def make_policy(seed: int = 42) -> DirectTurnLinTS:
    blocks = ("S", "K", "L")
    return DirectTurnLinTS(
        context_schema_id=context_schema_id(blocks),
        enabled_context_blocks=blocks,
        feature_names=feature_names_for_blocks(blocks),
        selector_version=SELECTOR_NAME,
        selector_sha256=SELECTOR_SHA256,
        reward_mode="headroom_normalized",
        anchor_mode="none",
        anchor_gamma=0.0,
        anchor_selector_version=SELECTOR_NAME,
        anchor_selector_sha256=SELECTOR_SHA256,
        ridge_lambda=RIDGE_LAMBDA,
        exploration_scale=EXPLORATION_SCALE,
        seed=seed,
        data_mode="real",
        created_at=FIXED_POLICY_CREATED_AT,
    )


def verify_inputs() -> dict[str, str]:
    observed = {name: sha256(EVIDENCE / name) for name in EXPECTED_OLD_HASHES}
    if observed != EXPECTED_OLD_HASHES:
        raise RuntimeError(f"Historical evidence hash mismatch: {observed}")
    if sha256(ARCH_MANIFEST) != ARCHITECTURE_MANIFEST_SHA256:
        raise RuntimeError("Frozen S+K+L architecture manifest hash mismatch.")
    arch = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))
    required = {
        "context_preset": "S+K+L",
        "ordered_context_features": FEATURES,
        "context_dimension": 11,
        "reward_mode": "headroom_normalized",
        "selector_sha256": SELECTOR_SHA256,
        "anchor_mode": "none",
    }
    for key, value in required.items():
        if arch.get(key) != value:
            raise RuntimeError(f"Frozen architecture mismatch for {key}.")
    return observed


def reconstruct_preaction_rows() -> pd.DataFrame:
    all_rows = pd.read_csv(EVIDENCE / "derived_analysis_dataset.csv")
    included = all_rows.loc[all_rows["include_in_context_experiment"] == True].copy()  # noqa: E712
    if len(included) != 84:
        raise RuntimeError(f"Expected 84 included real rows, found {len(included)}.")
    detectors = DetectorService.from_project_defaults(STUDENT_ROOT)
    history: dict[str, list[str]] = {}
    reconstructed: dict[str, tuple[float, float, float, float, str]] = {}
    ordered = all_rows.sort_values(["attempt_id", "turn_index_within_attempt"])
    for row in ordered.itertuples(index=False):
        prior = history.setdefault(str(row.attempt_id), [])
        if prior:
            values = detectors.predict_all(
                current_text=prior[-1],
                previous_student_text=prior[-2] if len(prior) > 1 else None,
            )
            reconstructed[str(row.action_event_id)] = (
                float(values["reasoning_probability"]),
                float(values["uncertainty_probability"]),
                float(values["clarification_probability"]),
                0.0,
                "retrospective_preaction_recomputation",
            )
        else:
            reconstructed[str(row.action_event_id)] = (
                0.0, 0.0, 0.0, 1.0, "runtime_missing_representation"
            )
        response = str(row.learner_response_text or "").strip()
        if response:
            prior.append(response)
    for index, row in included.iterrows():
        values = reconstructed[str(row["action_event_id"])]
        included.loc[index, L_FEATURES] = values[:4]
        included.loc[index, "L_feature_source"] = values[4]
    included["S_feature_source"] = "real_frozen_selector_probabilities"
    included["K_feature_source"] = "real_preaction_bkt_state"
    included["context_origin"] = "real_preaction_bootstrap_source"
    return included.reset_index(drop=True)


def hidden_theta() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(THETA_SEED)
    result: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(DIRECT_ARMS):
        beta = np.zeros(11, dtype=float)
        beta[:4] = 0.20 * rng.normal(size=4)
        beta[arm_index] += 1.0
        k = rng.normal(size=3)
        l = rng.normal(size=4)
        beta[4:7] = 0.18 * k / max(float(np.linalg.norm(k)), 1e-12)
        beta[7:] = 0.18 * l / max(float(np.linalg.norm(l)), 1e-12)
        result[arm] = beta
    return result


def synthesize(real: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    action_rng = np.random.default_rng(ACTION_SEED)
    noise_rng = np.random.default_rng(NOISE_SEED)
    source_indices = bootstrap_rng.integers(0, len(real), size=N_SYNTHETIC)
    theta = hidden_theta()
    staged: list[dict[str, object]] = []
    raw_utilities: list[float] = []
    for synthetic_index, source_index in enumerate(source_indices, start=1):
        row = real.iloc[int(source_index)]
        x = np.asarray([float(row[name]) for name in FEATURES], dtype=float)
        p = x[:4].copy()
        p /= p.sum()
        draw = float(action_rng.random())
        action_index = min(int(np.searchsorted(np.cumsum(p), draw, side="right")), 3)
        arm = DIRECT_ARMS[action_index]
        utility = float(theta[arm] @ x + noise_rng.normal(0.0, 0.42))
        raw_utilities.append(utility)
        staged.append({
            "synthetic_observation_id": f"synthetic-init:{synthetic_index:04d}",
            "source_action_event_id": str(row["action_event_id"]),
            "source_attempt_id": str(row["attempt_id"]),
            "source_turn_index": int(row["turn_index_within_attempt"]),
            "context_vector": x.tolist(),
            "context": {name: float(value) for name, value in zip(FEATURES, x)},
            "selected_arm": arm,
            "synthetic_action": arm,
            "behavior_propensity": float(p[action_index]),
            "synthetic_behavior_propensity": float(p[action_index]),
            "full_behavior_probability_vector": {
                name: float(value) for name, value in zip(DIRECT_ARMS, p)
            },
            "full_MD7_probability_vector": {
                name: float(value) for name, value in zip(DIRECT_ARMS, p)
            },
            "assignment_random_draw": draw,
            "S_feature_source": str(row["S_feature_source"]),
            "K_feature_source": str(row["K_feature_source"]),
            "L_feature_source": str(row["L_feature_source"]),
            "L_source": str(row["L_feature_source"]),
        })
    target_mean = 0.14744802180779332
    target_sd = 0.20819256625143093
    raw = np.asarray(raw_utilities)
    rewards = target_mean + target_sd * (raw - raw.mean()) / raw.std(ddof=1)
    rewards = np.clip(rewards, -1.0, 1.0)
    for record, reward in zip(staged, rewards):
        before = float(record["context"]["mastery_before"])  # type: ignore[index]
        after = (
            before + float(reward) * (1.0 - before)
            if reward > 0.0
            else before + float(reward) * before if reward < 0.0 else before
        )
        roundtrip = turn_rewards(before, after)["headroom_normalized"]
        if abs(roundtrip - float(reward)) > 1e-10:
            raise RuntimeError("Synthetic mastery transition failed reward round-trip.")
        record.update({
            "schema_version": "adaptmath_skl_synthetic_observation_v1",
            "observation_origin": "synthetic_initialization",
            "synthetic_generator_version": SYNTHETIC_GENERATOR_VERSION,
            "real_live_observation": False,
            "posterior_initialization_observation": True,
            "synthetic_initialization": True,
            "policy_context_preset": "S+K+L",
            "context_dimension": 11,
            "ordered_context_features": FEATURES,
            "temporal_semantics": "pre_action_only",
            "action_assignment_source": "raw_md7_probability_sample",
            "behavior_policy": "md7_r2_probability_proportional_v1",
            "randomized_assignment": True,
            "reward_mode": "headroom_normalized",
            "synthetic_reward": float(reward),
            "synthetic_mastery_before": before,
            "synthetic_mastery_after": after,
            "reward_roundtrip_error": float(roundtrip - reward),
            "posterior_update_weight": INITIALIZATION_WEIGHT,
            "posterior_update_origin": "synthetic_initialization",
            "current_learner_response_in_context": False,
            "current_mrb1_in_context": False,
            "mastery_after_in_context": False,
            "future_information_in_context": False,
        })
    generator = {
        "generator_version": SYNTHETIC_GENERATOR_VERSION,
        "master_seed": MASTER_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "action_seed": ACTION_SEED,
        "theta_seed": THETA_SEED,
        "noise_seed": NOISE_SEED,
        "audit_seed": AUDIT_SEED,
        "observation_count": N_SYNTHETIC,
        "per_observation_weight": INITIALIZATION_WEIGHT,
        "effective_sample_size": EFFECTIVE_SAMPLE_SIZE,
        "reward_target_source": "historical HEADROOM_NORMALIZED N=84 mean/SD",
        "reward_target_mean": target_mean,
        "reward_target_sd": target_sd,
        "reward_noise_sd_before_global_scaling": 0.42,
        "hidden_theta_by_arm": {arm: vector.tolist() for arm, vector in theta.items()},
        "theta_design": (
            "fixed seeded full-11D linear utility; selector-preference diagonal plus "
            "lower-norm seeded K/L terms; no semantic state-action rules"
        ),
        "reward_scaling": "single global affine scaling followed by [-1,1] clipping",
    }
    return staged, generator


def initialize_policy(records: list[dict[str, object]]) -> DirectTurnLinTS:
    policy = make_policy()
    for record in records:
        applied = policy.update_once(
            update_id=str(record["synthetic_observation_id"]),
            arm=str(record["selected_arm"]),
            context=record["context_vector"],
            reward=record["synthetic_reward"],
            weight=INITIALIZATION_WEIGHT,
        )
        if not applied:
            raise RuntimeError("Unexpected duplicate synthetic update identity.")
    return policy


def posterior_draw_probabilities(
    policy: DirectTurnLinTS, contexts: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros((len(contexts), 4), dtype=float)
    means = []
    covariances = []
    for arm in DIRECT_ARMS:
        inv = np.linalg.inv(policy.A[arm])
        means.append(inv @ policy.b[arm])
        covariances.append((EXPLORATION_SCALE**2) * inv)
    for row_index, x in enumerate(contexts):
        sampled_scores = np.column_stack([
            rng.multivariate_normal(means[index], covariances[index], size=MC_DRAWS) @ x
            for index in range(4)
        ])
        counts = np.bincount(np.argmax(sampled_scores, axis=1), minlength=4)
        output[row_index] = counts / MC_DRAWS
    return output


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def audit(
    policy: DirectTurnLinTS,
    records: list[dict[str, object]],
    real: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arm_counts = Counter(str(row["selected_arm"]) for row in records)
    rewards = np.asarray([float(row["synthetic_reward"]) for row in records])
    matrix_rows = []
    coefficient_rows = []
    eigen_rows = []
    for arm in DIRECT_ARMS:
        eigenvalues = np.linalg.eigvalsh(policy.A[arm])
        mean = np.linalg.solve(policy.A[arm], policy.b[arm])
        matrix_rows.append({
            "arm": arm,
            "dimension": 11,
            "synthetic_updates": arm_counts[arm],
            "effective_weight": arm_counts[arm] * INITIALIZATION_WEIGHT,
            "condition_number": float(np.linalg.cond(policy.A[arm])),
            "minimum_eigenvalue": float(eigenvalues.min()),
            "maximum_eigenvalue": float(eigenvalues.max()),
            "frobenius_norm_A_minus_prior": float(
                np.linalg.norm(policy.A[arm] - RIDGE_LAMBDA * np.eye(11))
            ),
            "b_l2_norm": float(np.linalg.norm(policy.b[arm])),
            "posterior_mean_l2_norm": float(np.linalg.norm(mean)),
        })
        for name, value in zip(FEATURES, mean):
            coefficient_rows.append({"arm": arm, "feature": name, "posterior_mean": value})
        for index, value in enumerate(eigenvalues, start=1):
            eigen_rows.append({"arm": arm, "eigen_index": index, "eigenvalue": value})
    matrix_df = pd.DataFrame(matrix_rows)
    coefficient_df = pd.DataFrame(coefficient_rows)
    eigen_df = pd.DataFrame(eigen_rows)

    contexts = real[FEATURES].to_numpy(dtype=float)
    md7 = contexts[:, :4].copy()
    md7 /= md7.sum(axis=1, keepdims=True)
    prior = make_policy(seed=7)
    p0 = posterior_draw_probabilities(prior, contexts, AUDIT_SEED)
    psynth = posterior_draw_probabilities(policy, contexts, AUDIT_SEED + 1)
    posterior_mean_scores = np.column_stack([
        contexts @ np.linalg.solve(policy.A[arm], policy.b[arm])
        for arm in DIRECT_ARMS
    ])
    rows = []
    for index, source in real.iterrows():
        for policy_name, probabilities in (
            ("P0_uninformed_prior", p0[index]),
            ("MD7_behavior", md7[index]),
            ("P_SYNTH", psynth[index]),
        ):
            rows.append({
                "source_action_event_id": source["action_event_id"],
                "policy": policy_name,
                **{f"p_{arm}": float(probabilities[i]) for i, arm in enumerate(DIRECT_ARMS)},
                "top1_arm": DIRECT_ARMS[int(np.argmax(probabilities))],
                "entropy_nats": float(entropy_rows(probabilities[None, :])[0]),
                "md7_top1_agreement": int(
                    int(np.argmax(probabilities)) == int(np.argmax(md7[index]))
                ),
                "total_variation_from_md7": float(
                    0.5 * np.abs(probabilities - md7[index]).sum()
                ),
                "unavailable_action_selected": 0,
                **{
                    f"posterior_mean_reward_score_{arm}": (
                        float(posterior_mean_scores[index, arm_index])
                        if policy_name == "P_SYNTH" else ""
                    )
                    for arm_index, arm in enumerate(DIRECT_ARMS)
                },
            })
    heldout = pd.DataFrame(rows)
    summary_rows = []
    for policy_name, group in heldout.groupby("policy", sort=False):
        probs = group[[f"p_{arm}" for arm in DIRECT_ARMS]].to_numpy()
        aggregate = probs.mean(axis=0)
        non_telling = real[S_FEATURES].to_numpy().argmax(axis=1) != 3
        summary_rows.append({
            "policy": policy_name,
            "n_real_contexts": len(real),
            **{f"action_rate_{arm}": float(aggregate[i]) for i, arm in enumerate(DIRECT_ARMS)},
            "aggregate_entropy_nats": float(entropy_rows(aggregate[None, :])[0]),
            "mean_context_entropy_nats": float(group["entropy_nats"].mean()),
            "md7_top1_agreement_rate": float(group["md7_top1_agreement"].mean()),
            "mean_total_variation_from_md7": float(group["total_variation_from_md7"].mean()),
            "telling_rate_when_md7_top1_not_telling": float(probs[non_telling, 3].mean()),
            "unavailable_action_frequency": 0.0,
        })
    heldout_summary = pd.DataFrame(summary_rows)

    sensitivity_rows = []
    base_probs = posterior_draw_probabilities(policy, contexts, AUDIT_SEED + 2)
    for arm_index, arm in enumerate(DIRECT_ARMS):
        source_index = int(np.argmax(md7[:, arm_index]))
        clone = copy.deepcopy(policy)
        clone.update_once(
            update_id=f"sensitivity:{arm}", arm=arm,
            context=contexts[source_index], reward=0.7, weight=REAL_UPDATE_WEIGHT,
        )
        after_probs = posterior_draw_probabilities(clone, contexts, AUDIT_SEED + 2)
        sensitivity_rows.append({
            "arm": arm,
            "hypothetical_real_reward": 0.7,
            "hypothetical_real_weight": REAL_UPDATE_WEIGHT,
            "source_action_event_id": real.iloc[source_index]["action_event_id"],
            "mean_selection_probability_before": float(base_probs[:, arm_index].mean()),
            "mean_selection_probability_after": float(after_probs[:, arm_index].mean()),
            "mean_selection_probability_change": float(
                after_probs[:, arm_index].mean() - base_probs[:, arm_index].mean()
            ),
            "posterior_mean_parameter_change_l2": float(
                np.linalg.norm(
                    np.linalg.solve(clone.A[arm], clone.b[arm])
                    - np.linalg.solve(policy.A[arm], policy.b[arm])
                )
            ),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    ps = heldout_summary.loc[heldout_summary["policy"] == "P_SYNTH"].iloc[0]
    p0_summary = heldout_summary.loc[
        heldout_summary["policy"] == "P0_uninformed_prior"
    ].iloc[0]
    checks = {
        "exactly_500_records": len(records) == N_SYNTHETIC,
        "all_four_arms_observed_at_least_10": min(arm_counts.values()) >= 10,
        "reward_mean_within_0_02_of_target": abs(rewards.mean() - 0.1474480218) <= 0.02,
        "reward_sd_within_0_02_of_target": abs(rewards.std(ddof=1) - 0.2081925663) <= 0.02,
        "reward_bounds_valid": bool(np.all((-1.0 <= rewards) & (rewards <= 1.0))),
        "matrix_condition_number_below_1e6": float(matrix_df.condition_number.max()) < 1e6,
        "all_matrices_positive_definite": float(matrix_df.minimum_eigenvalue.min()) > 0.0,
        "psynth_all_arm_rates_at_least_0_01": min(float(ps[f"action_rate_{arm}"]) for arm in DIRECT_ARMS) >= 0.01,
        "psynth_mean_context_entropy_at_least_0_20": float(ps["mean_context_entropy_nats"]) >= 0.20,
        "psynth_not_p0_like_top1_agreement_margin_at_least_0_20": (
            float(ps["md7_top1_agreement_rate"])
            - float(p0_summary["md7_top1_agreement_rate"]) >= 0.20
        ),
        "psynth_not_p0_like_entropy_reduction_at_least_0_03": (
            float(p0_summary["mean_context_entropy_nats"])
            - float(ps["mean_context_entropy_nats"]) >= 0.03
        ),
        "psynth_mean_tv_from_md7_below_0_60": float(ps["mean_total_variation_from_md7"]) <= 0.60,
        "psynth_nontop1_telling_rate_below_0_40": float(ps["telling_rate_when_md7_top1_not_telling"]) <= 0.40,
        "no_unavailable_action_selections": float(ps["unavailable_action_frequency"]) == 0.0,
        "one_real_update_changes_each_arm_posterior": bool(
            (sensitivity.posterior_mean_parameter_change_l2 > 1e-6).all()
        ),
    }
    decision = "A" if all(checks.values()) else "B"
    audit_summary = {
        "activation_decision": decision,
        "activation_decision_meaning": (
            "synthetic initialization accepted; activate isolated LIVE lineage"
            if decision == "A" else "do not activate; continue evidence collection"
        ),
        "predeclared_sanity_checks": checks,
        "synthetic_observation_count": len(records),
        "per_observation_weight": INITIALIZATION_WEIGHT,
        "effective_sample_size": EFFECTIVE_SAMPLE_SIZE,
        "real_live_observation_weight": REAL_UPDATE_WEIGHT,
        "synthetic_arm_counts": dict(arm_counts),
        "reward": {
            "mean": float(rewards.mean()),
            "sd": float(rewards.std(ddof=1)),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "positive_fraction": float((rewards > 0).mean()),
            "negative_fraction": float((rewards < 0).mean()),
        },
        "maximum_matrix_condition_number": float(matrix_df.condition_number.max()),
        "minimum_matrix_eigenvalue": float(matrix_df.minimum_eigenvalue.min()),
        "heldout_real_context_count": len(real),
        "monte_carlo_draws_per_context": MC_DRAWS,
        "uncertainty_note": "Monte Carlo policy sampling variability; no outcome labels used",
        "limitations": [
            "All initialization rewards are synthetic and down-weighted.",
            "Held-out audit contexts are historical pre-action contexts, not a prospective trial.",
            "Retrospective L values use frozen local detectors on t-1 text; they do not prove L usefulness.",
            "Activation does not convert synthetic evidence into real evidence.",
        ],
    }
    return audit_summary, matrix_df, coefficient_df, eigen_df, heldout, heldout_summary, sensitivity


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_figures(records, real, policy, matrix_df, coefficient_df, eigen_df,
                 heldout_summary, sensitivity) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    arm_counts = Counter(str(row["selected_arm"]) for row in records)
    rewards = np.array([float(row["synthetic_reward"]) for row in records])
    mastery = np.array([float(row["synthetic_mastery_before"]) for row in records])
    sources = Counter(str(row["L_feature_source"]) for row in records)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, cols, title in zip(axes, (S_FEATURES, K_FEATURES, L_FEATURES), ("S", "K", "L")):
        axis.boxplot(real[cols].to_numpy(), tick_labels=[c.replace("selector_p_", "p_").replace("previous_", "prev_") for c in cols])
        axis.set_title(f"{title} real pre-action source (n=84)"); axis.tick_params(axis="x", rotation=70)
    fig.suptitle("Synthetic S+K+L coverage from real pre-action source states (n=500 bootstrap)")
    save_figure(FIGURES / "01_synthetic_SKL_context_coverage.png")

    plt.figure(figsize=(7, 5)); plt.bar(DIRECT_ARMS, [arm_counts[a] for a in DIRECT_ARMS], color="#70AD47")
    plt.ylabel("Assigned observations"); plt.title("MD7-proportional synthetic assignments (n=500)")
    save_figure(FIGURES / "02_synthetic_action_distribution.png")

    plt.figure(figsize=(8, 5)); plt.hist(rewards, bins=25, color="#5B9BD5", edgecolor="white")
    plt.axvline(rewards.mean(), color="black", linestyle="--", label=f"mean={rewards.mean():.3f}")
    plt.xlabel("HEADROOM_NORMALIZED synthetic reward"); plt.ylabel("Count"); plt.legend();
    plt.title("Synthetic reward distribution (n=500; globally scaled)")
    save_figure(FIGURES / "03_synthetic_reward_distribution.png")

    historical = pd.read_csv(EVIDENCE / "derived_analysis_dataset.csv")
    historical = historical.loc[historical["include_in_reward_experiment"] == True, "reward_headroom_normalized"].astype(float)  # noqa: E712
    plt.figure(figsize=(8, 5)); bins = np.linspace(-.5, .8, 27)
    plt.hist(rewards, bins=bins, density=True, alpha=.58, color="#5B9BD5", label="Synthetic initialization (n=500)")
    plt.hist(historical, bins=bins, density=True, alpha=.58, color="#ED7D31", label="Historical genuine HEADROOM (n=84)")
    plt.xlabel("HEADROOM_NORMALIZED reward"); plt.ylabel("Density"); plt.legend()
    plt.title("Global reward-scale comparison (historical outcomes used only for scale)")
    save_figure(FIGURES / "04_reward_scale_comparison.png")

    mastery_after = np.array([float(row["synthetic_mastery_after"]) for row in records])
    plt.figure(figsize=(8, 5)); plt.scatter(mastery, mastery_after, c=rewards, cmap="coolwarm", s=14, alpha=.55)
    plt.plot([0,1],[0,1], color="black", linestyle="--", linewidth=.8); plt.colorbar(label="Synthetic HEADROOM reward")
    plt.xlabel("mastery_before (pre-action)"); plt.ylabel("synthetic mastery_after")
    plt.title("BKT-consistent synthetic transitions (n=500; round-trip checked)")
    save_figure(FIGURES / "05_synthetic_mastery_transitions.png")

    counts = np.array([arm_counts[a] for a in DIRECT_ARMS]); weighted = counts * INITIALIZATION_WEIGHT
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax1.bar(DIRECT_ARMS, counts, color="#A5A5A5", label="Record count")
    ax1.set_ylabel("Synthetic record count"); ax2 = ax1.twinx(); ax2.plot(DIRECT_ARMS, weighted, color="#C00000", marker="o", linewidth=2, label="Effective weight")
    ax2.set_ylabel("Effective posterior weight"); ax1.set_title("Per-arm initialization: each record weight = 0.02; total effective n = 10")
    save_figure(FIGURES / "06_effective_weight_by_arm.png")

    coverage = [(real[name] != 0).mean() for name in FEATURES]
    plt.figure(figsize=(10, 5)); plt.bar(range(11), coverage, color=["#4472C4"]*4+["#70AD47"]*3+["#FFC000"]*4)
    plt.xticks(range(11), FEATURES, rotation=70, ha="right"); plt.ylabel("Nonzero fraction in 84 source contexts")
    plt.title("Frozen 11-feature context coverage (missing indicators included)")
    save_figure(FIGURES / "06_feature_coverage.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, arm in zip(axes.ravel(), DIRECT_ARMS):
        im = axis.imshow(policy.A[arm], cmap="viridis", aspect="auto"); axis.set_title(f"A: {arm} (11×11)")
        fig.colorbar(im, ax=axis, fraction=.046)
    fig.suptitle("Weighted posterior precision matrices: λI + 0.02Σxxᵀ")
    save_figure(FIGURES / "07_posterior_matrix_heatmaps.png")

    plt.figure(figsize=(8, 5))
    for arm in DIRECT_ARMS:
        vals = eigen_df.loc[eigen_df.arm == arm, "eigenvalue"].to_numpy()
        plt.plot(range(1, 12), vals, marker="o", label=arm)
    plt.xlabel("Ordered eigenvalue index"); plt.ylabel("Eigenvalue"); plt.legend(); plt.title("Positive eigenvalue spectra")
    save_figure(FIGURES / "08_posterior_eigenvalues.png")

    plt.figure(figsize=(7, 5)); plt.bar(matrix_df.arm, matrix_df.condition_number, color="#A5A5A5")
    plt.ylabel("2-norm condition number"); plt.title("Posterior matrix conditioning (lower is safer)")
    save_figure(FIGURES / "09_posterior_condition_numbers.png")

    plt.figure(figsize=(8, 5)); plt.bar(matrix_df.arm, matrix_df.posterior_mean_l2_norm, color="#8064A2")
    plt.ylabel("||A^-1 b||_2"); plt.title("Posterior mean coefficient norms (synthetic effective n=10)")
    save_figure(FIGURES / "11_posterior_coefficient_norms.png")

    positions = np.arange(4); width = .24
    plt.figure(figsize=(9, 5))
    for i, row in heldout_summary.reset_index(drop=True).iterrows():
        plt.bar(positions + (i-1)*width, [row[f"action_rate_{a}"] for a in DIRECT_ARMS], width, label=row.policy)
    plt.xticks(positions, DIRECT_ARMS); plt.ylabel("Mean action probability across 84 contexts"); plt.legend()
    plt.title(f"Held-out pre-action audit ({MC_DRAWS} deterministic draws/context for TS)")
    save_figure(FIGURES / "07_policy_action_distribution_comparison.png")

    metrics = ["md7_top1_agreement_rate", "mean_total_variation_from_md7", "mean_context_entropy_nats", "telling_rate_when_md7_top1_not_telling"]
    table = heldout_summary.set_index("policy")[metrics].to_numpy()
    plt.figure(figsize=(9, 4)); plt.imshow(table, cmap="Blues", aspect="auto")
    plt.xticks(range(len(metrics)), ["top-1 agree", "TV from MD7", "entropy", "telling | MD7≠telling"], rotation=18)
    plt.yticks(range(len(heldout_summary)), heldout_summary.policy)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]): plt.text(j, i, f"{table[i,j]:.3f}", ha="center", va="center")
    plt.colorbar(label="Rate / nats"); plt.title("Held-out policy comparison (n=84; no outcomes used)")
    save_figure(FIGURES / "08_policy_md7_agreement.png")

    plt.figure(figsize=(8, 5)); plt.bar(heldout_summary.policy, heldout_summary.telling_rate_when_md7_top1_not_telling, color=["#A5A5A5", "#ED7D31", "#4472C4"])
    plt.ylim(0, 1); plt.ylabel("P(telling | MD7 top-1 is not telling)"); plt.xticks(rotation=12)
    plt.title("Telling behavior on held-out real pre-action contexts (n=84)")
    save_figure(FIGURES / "09_policy_telling_rate.png")

    plt.figure(figsize=(8, 5)); plt.bar(heldout_summary.policy, heldout_summary.mean_context_entropy_nats, color=["#A5A5A5", "#ED7D31", "#4472C4"])
    plt.ylim(0, math.log(4)); plt.ylabel("Mean context-level entropy (nats)"); plt.xticks(rotation=12)
    plt.title(f"Policy entropy (n=84; TS uses {MC_DRAWS} deterministic draws/context)")
    save_figure(FIGURES / "10_policy_entropy.png")

    plt.figure(figsize=(8, 5)); plt.bar(sensitivity.arm, sensitivity.mean_selection_probability_change, color="#C55A11")
    plt.axhline(0, color="black", linewidth=.8); plt.ylabel("Mean action-probability change")
    plt.title("One cloned real update sensitivity (weight=1, reward=+0.7)")
    save_figure(FIGURES / "12_real_update_responsiveness.png")

    fig, ax = plt.subplots(figsize=(11, 3.8)); ax.axis("off")
    provenance = [(0.12,"500 generated\nsynthetic records"),(0.38,"weight 0.02 each\neffective total = 10"),(0.64,"isolated LIVE\nposterior"),(0.88,"future genuine update\nweight = 1 each")]
    for x,label in provenance:
        ax.text(x,.55,label,ha="center",va="center",fontsize=11,bbox=dict(boxstyle="round,pad=.55",fc="#EAF2F8",ec="#2E4053"))
    for left,right in zip(provenance, provenance[1:]):
        ax.annotate("",xy=(right[0]-.10,.55),xytext=(left[0]+.10,.55),arrowprops=dict(arrowstyle="->",lw=2))
    ax.text(.5,.10,"Synthetic provenance remains explicit; it is never counted as real learner evidence.",ha="center",fontsize=10,color="#922B21")
    ax.set_title("Turn-LinTS initialization provenance and influence", fontsize=14)
    save_figure(FIGURES / "13_initialization_provenance.png")

    fig, ax = plt.subplots(figsize=(11, 8)); ax.axis("off")
    boxes = [
        (0.5,.94,"PRE-ACTION: learner message → explicit agency check"),
        (0.5,.81,"Frozen MD7-R2 → S probabilities"), (0.25,.68,"K: mastery_before + previous Δ"),
        (0.75,.68,"L: detector signals from learner response t−1"), (0.5,.54,"S+K+L vector (11 dimensions)"),
        (0.5,.40,"Direct disjoint Turn-LinTS LIVE\n4 arms; no tau/eligibility/anchor"),
        (0.5,.25,"Tutor realization → response → learner response"),
        (0.5,.10,"POST-ACTION: BKT posterior-belief update\nHEADROOM_NORMALIZED reward → immediate weight-1 update"),
    ]
    for x,y,label in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=10,bbox=dict(boxstyle="round,pad=.45",fc="#EAF2F8" if y>.5 else "#FDEDEC",ec="#34495E"))
    arrows = [((.5,.90),(.5,.85)),((.5,.77),(.29,.72)),((.5,.77),(.71,.72)),((.25,.64),(.45,.58)),((.75,.64),(.55,.58)),((.5,.50),(.5,.44)),((.5,.36),(.5,.29)),((.5,.21),(.5,.14))]
    for start,end in arrows: ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.text(.02,.49,"Agency override: non-randomized, null propensity, no policy update",fontsize=9,color="#922B21")
    ax.text(.02,.02,"MRB1: auxiliary diagnostic only • formal assessment: separate policy-health signal",fontsize=9)
    ax.set_title("AdaptMath Turn-LinTS v1 LIVE architecture — synthetic initialization remains labeled",fontsize=14)
    save_figure(FIGURES / "14_final_live_architecture.png")


def write_docs(summary: dict[str, object]) -> None:
    decision = summary["activation_decision"]
    report = f"""# Synthetic warm-start report

Decision: **{decision}**.

Exactly 500 synthetic initialization observations were generated from bootstrapped real pre-action S and K states. L was deterministically recomputed from the preceding learner response with the frozen local detectors; first-turn L uses the implemented neutral values plus the missing indicator. Actions were sampled from the raw frozen MD7 vector. Rewards were produced by fixed seeded, arm-specific full-11D linear functions plus seeded noise and one global scaling operation to the historical HEADROOM_NORMALIZED scale.

Each synthetic observation has posterior weight 0.02, for an effective initialization sample size of 10. Real linked LIVE BKT posterior-belief updates have weight 1.0. Synthetic evidence is permanently labeled and is not described as real evidence.

The audit used 84 historical pre-action contexts without outcome labels to compare the uninformed prior, raw MD7 behavior, and the initialized posterior. Decision A requires every predeclared sanity check in `summary.json` to pass.
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")
    decision_text = """# Activation decision

## A — activate isolated LIVE S+K+L lineage

All predeclared audit gates passed. This accepts a deliberately weak synthetic prior; it does not establish that L improves outcomes and does not relabel synthetic observations as real. LIVE may be activated only through the explicit `-FinalSKLLive` launcher after an activation timestamp is written to the lineage manifest and focused tests pass.
""" if decision == "A" else """# Activation decision

## B — remain in evidence collection

At least one predeclared audit gate failed. LIVE must not be activated.
"""
    (HERE / "DECISION_EVIDENCE.md").write_text(decision_text, encoding="utf-8")


def prepare_live_lineage(policy: DirectTurnLinTS, summary: dict[str, object]) -> None:
    if summary["activation_decision"] != "A":
        return
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = LIVE_ROOT / "policy_state.json"
    manifest_path = LIVE_ROOT / "lineage_manifest.json"
    if state_path.exists() or manifest_path.exists():
        if not (state_path.is_file() and manifest_path.is_file()):
            raise RuntimeError("Incomplete existing LIVE lineage; refusing replacement.")
        old_state = json.loads(state_path.read_text(encoding="utf-8"))
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_ids = [f"synthetic-init:{index:04d}" for index in range(1, 501)]
        replaceable = (
            old_manifest.get("live_activated_at") is None
            and old_state.get("update_count") == 500
            and old_state.get("applied_update_ids") == expected_ids
        )
        if not replaceable:
            raise RuntimeError(
                "Refusing to overwrite an activated or evolved LIVE lineage."
            )
    save_turn_lints_state(policy, state_path)
    # Counts are sourced from the persisted state, not recomputed from outcomes.
    manifest = {
        "schema_version": LIVE_LINEAGE_SCHEMA,
        "architecture_version": "adaptmath_turn_lints_v1_skl_final_amendment_1",
        "mode": "LIVE",
        "architecture_manifest_path": str(ARCH_MANIFEST),
        "architecture_manifest_sha256": ARCHITECTURE_MANIFEST_SHA256,
        "context_preset": "S+K+L",
        "context_blocks": ["S", "K", "L"],
        "context_dimension": 11,
        "ordered_context_features": FEATURES,
        "diagnostic_context_preset": "S+K+L+H+Q",
        "reward_mode": "headroom_normalized",
        "selector_name": SELECTOR_NAME,
        "selector_sha256": SELECTOR_SHA256,
        "algorithm": "direct_disjoint_turn_lints_v1",
        "arms": list(DIRECT_ARMS),
        "tau_gate": "none",
        "action_eligibility": "none",
        "anchor_mode": "none",
        "anchor_gamma": 0.0,
        "P1": "inactive",
        "P2": "not_used",
        "synthetic_generator_version": SYNTHETIC_GENERATOR_VERSION,
        "posterior_initialization": "synthetic_warmstart_v1",
        "synthetic_observation_count": N_SYNTHETIC,
        "synthetic_observation_weight": INITIALIZATION_WEIGHT,
        "synthetic_effective_sample_size": EFFECTIVE_SAMPLE_SIZE,
        "synthetic_arm_update_counts": dict(policy.arm_update_counts),
        "real_live_observation_weight": REAL_UPDATE_WEIGHT,
        "initial_state_sha256": sha256(state_path),
        "initial_total_update_count": policy.total_updates,
        "initial_real_live_update_count": 0,
        "activation_decision": "A",
        "activation_decision_artifact": str(HERE / "DECISION_EVIDENCE.md"),
        "analysis_artifact_path": str(HERE),
        "analysis_summary_sha256": sha256(HERE / "summary.json"),
        "initialized_at": utc_now(),
        "live_activated_at": None,
        "live_enabled": True,
        "learner_agency_update_semantics": "explicit agency overrides receive no policy update",
        "no_reward_update_semantics": "no valid linked BKT posterior-belief update means no policy update",
        "MRB1_role": "auxiliary diagnostic only; not context or reward",
        "formal_assessment_role": "separate external policy-health signal",
    }
    write_json(manifest_path, manifest)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    input_hashes = verify_inputs()
    real = reconstruct_preaction_rows()
    records, generator = synthesize(real)
    with (HERE / "synthetic_warmstart_observations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(HERE / "synthetic_generator.json", generator)
    policy = initialize_policy(records)
    (HERE / "posterior_state_before.json").write_text(
        json.dumps(make_policy().state_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "posterior_state_after.json").write_text(
        json.dumps(policy.state_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "synthetic_action_summary.csv").write_text(
        "arm,count,effective_weight\n" + "".join(
            f"{arm},{policy.arm_update_counts[arm]},{policy.arm_update_counts[arm]*INITIALIZATION_WEIGHT}\n"
            for arm in DIRECT_ARMS
        ), encoding="utf-8"
    )
    reward_frame = pd.DataFrame({
        "synthetic_observation_id": [row["synthetic_observation_id"] for row in records],
        "mastery_before": [row["synthetic_mastery_before"] for row in records],
        "mastery_after": [row["synthetic_mastery_after"] for row in records],
        "reward": [row["synthetic_reward"] for row in records],
        "roundtrip_error": [row["reward_roundtrip_error"] for row in records],
    })
    reward_frame.to_csv(HERE / "synthetic_reward_summary.csv", index=False)
    context_summary_rows = []
    for name in FEATURES:
        values = np.asarray([float(row["context"][name]) for row in records])
        context_summary_rows.append({
            "feature": name,
            "block": "S" if name in S_FEATURES else "K" if name in K_FEATURES else "L",
            "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1),
            "min": values.min(), "max": values.max(),
            "nonzero_fraction": (values != 0).mean(),
        })
    pd.DataFrame(context_summary_rows).to_csv(
        HERE / "synthetic_context_summary.csv", index=False
    )
    result = audit(policy, records, real)
    summary, matrix_df, coefficient_df, eigen_df, heldout, heldout_summary, sensitivity = result
    summary["historical_input_hashes"] = input_hashes
    summary["architecture_manifest_sha256"] = ARCHITECTURE_MANIFEST_SHA256
    summary["generated_at"] = utc_now()
    write_json(HERE / "summary.json", summary)
    matrix_df.to_csv(HERE / "posterior_initialization_summary.csv", index=False)
    coefficient_df.to_csv(HERE / "posterior_coefficient_summary.csv", index=False)
    eigen_df.to_csv(HERE / "posterior_eigenvalue_audit.csv", index=False)
    heldout.to_csv(HERE / "heldout_policy_audit.csv", index=False)
    heldout_summary.to_csv(HERE / "heldout_context_policy_summary.csv", index=False)
    sensitivity.to_csv(HERE / "single_real_update_sensitivity.csv", index=False)
    make_figures(records, real, policy, matrix_df, coefficient_df, eigen_df,
                 heldout_summary, sensitivity)
    write_docs(summary)
    prepare_live_lineage(policy, summary)
    print(json.dumps(summary, indent=2, default=lambda item: item.item()))


if __name__ == "__main__":
    main()
