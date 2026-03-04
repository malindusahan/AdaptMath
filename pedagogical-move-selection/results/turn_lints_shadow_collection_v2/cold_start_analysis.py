"""Offline cold-start diagnostic for direct-arm Turn-LinTS.

Reads existing passive real-turn records, reconstructs only pre-action context,
and performs selection-only simulations. It never updates or saves a policy.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
MOVE_ROOT = OUTPUT_DIR.parents[1]
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
from src.self_improvement.turn_lints_policy import DirectTurnLinTS  # noqa: E402


SEED_COUNT = 256
RIDGE_LAMBDA = 1.0
EXPLORATION_SCALE = 0.20
SELECTOR_SHA = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
P1_GAMMAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40)
P2_KAPPAS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.00)


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
    if not isinstance(names, list) or not isinstance(vector, list) or len(names) != len(vector):
        return {}
    return {str(name): float(value) for name, value in zip(names, vector, strict=True)}


def reconstruct_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            raise ValueError("Every row needs four MD7 probabilities.")
        probabilities = {move: float(selector[move]) for move in MOVE_ORDER}
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
        l_fields = (
            "previous_reasoning_probability",
            "previous_uncertainty_probability",
            "previous_clarification_probability",
        )
        learner_signals = None
        if existing.get("previous_learner_signals_missing") == 0.0 and all(
            name in existing for name in l_fields
        ):
            learner_signals = {
                "reasoning_probability": existing[l_fields[0]],
                "uncertainty_probability": existing[l_fields[1]],
                "clarification_probability": existing[l_fields[2]],
            }
        previous_move = None if prior is None else prior["actual_move"]
        previous_mrb1 = None if prior is None else prior["mrb1"]
        context = build_turn_lints_context(
            enabled_blocks=BLOCK_ORDER,
            selector_probabilities=probabilities,
            mastery_before=float(mastery),
            previous_mastery_delta=previous_delta,
            previous_learner_signals=learner_signals,
            previous_move=previous_move,
            prior_tutor_turn_count=turn_index - 1,
            previous_mrb1_scores=previous_mrb1,
        )
        md7_top = max(MOVE_ORDER, key=probabilities.__getitem__)
        contexts.append(
            {
                "action_event_id": identity.get("action_event_id"),
                "attempt_id": attempt_id,
                "turn_index": turn_index,
                "probabilities": probabilities,
                "md7_top": md7_top,
                "vector": context.vector(),
            }
        )

        current_scores = nested(row, "tutor_quality", "scores")
        if not isinstance(current_scores, dict) or len(current_scores) != 4:
            current_scores = None
        actual_move = nested(row, "adaptive_decision", "final_move")
        if actual_move not in MOVE_ORDER:
            actual_move = nested(row, "selector", "effective", "argmax")
        learning = row.get("learning_state_update") or {}
        observed = (
            learning.get("should_update") is True
            and nested(row, "next_learner_observation", "status") == "observed_update"
            and learning.get("delta_mastery") is not None
        )
        previous[attempt_id] = {
            "actual_move": actual_move,
            "mrb1": current_scores,
            "reward_observed": observed,
            "delta": learning.get("delta_mastery"),
        }
    return contexts


def summarize(
    contexts: list[dict[str, Any]],
    selections: list[str],
) -> dict[str, Any]:
    if len(selections) % len(contexts):
        raise ValueError("Selections do not contain complete context sweeps.")
    repeated_top = [item["md7_top"] for _ in range(len(selections) // len(contexts)) for item in contexts]
    arm_counts = Counter(selections)
    transition: dict[str, Counter[str]] = defaultdict(Counter)
    for md7_top, selected in zip(repeated_top, selections, strict=True):
        transition[md7_top][selected] += 1
    non_telling = [
        selected for md7_top, selected in zip(repeated_top, selections, strict=True)
        if md7_top != "telling"
    ]
    return {
        "draw_count": len(selections),
        "arm_counts": {arm: arm_counts[arm] for arm in MOVE_ORDER},
        "arm_rates": {arm: arm_counts[arm] / len(selections) for arm in MOVE_ORDER},
        "md7_top1_agreement_count": sum(
            selected == md7_top
            for selected, md7_top in zip(selections, repeated_top, strict=True)
        ),
        "md7_top1_agreement_rate": mean(
            selected == md7_top
            for selected, md7_top in zip(selections, repeated_top, strict=True)
        ),
        "telling_when_md7_not_telling_count": sum(value == "telling" for value in non_telling),
        "telling_when_md7_not_telling_rate": mean(value == "telling" for value in non_telling),
        "transition_counts": {
            source: {target: transition[source][target] for target in MOVE_ORDER}
            for source in MOVE_ORDER
        },
        "transition_row_rates": {
            source: {
                target: (
                    transition[source][target] / sum(transition[source].values())
                    if sum(transition[source].values()) else None
                )
                for target in MOVE_ORDER
            }
            for source in MOVE_ORDER
        },
    }


def margin_summary(contexts: list[dict[str, Any]], strength: float, kind: str) -> dict[str, float]:
    margins: list[float] = []
    for item in contexts:
        probabilities = item["probabilities"]
        top = item["md7_top"]
        runner = max((arm for arm in MOVE_ORDER if arm != top), key=probabilities.__getitem__)
        if kind == "log":
            margin = strength * math.log(max(probabilities[top], 1e-12) / max(probabilities[runner], 1e-12))
        else:
            margin = strength * (probabilities[top] - probabilities[runner])
        margins.append(float(margin))
    return {"min": min(margins), "median": median(margins), "mean": mean(margins), "max": max(margins)}


def run() -> dict[str, Any]:
    contexts = reconstruct_contexts(load_jsonl(PASSIVE_TURNS))
    p0_scores: list[dict[str, float]] = []
    p0_selections: list[str] = []
    for seed in range(SEED_COUNT):
        policy = DirectTurnLinTS(
            context_schema_id=context_schema_id(BLOCK_ORDER),
            enabled_context_blocks=BLOCK_ORDER,
            feature_names=feature_names_for_blocks(BLOCK_ORDER),
            selector_version="MD7-R2-TELL-C1 epoch 2",
            selector_sha256=SELECTOR_SHA,
            reward_mode="headroom_normalized",
            ridge_lambda=RIDGE_LAMBDA,
            exploration_scale=EXPLORATION_SCALE,
            seed=seed,
            data_mode="real",
        )
        for item in contexts:
            result = policy.select_arm(item["vector"])
            scores = {arm: float(result["sampled_scores"][arm]) for arm in MOVE_ORDER}
            p0_scores.append(scores)
            p0_selections.append(str(result["selected_arm"]))

    repeated_contexts = contexts * SEED_COUNT
    p1: dict[str, Any] = {}
    for gamma in P1_GAMMAS:
        selections = []
        for item, scores in zip(repeated_contexts, p0_scores, strict=True):
            adjusted = {
                arm: scores[arm] + gamma * math.log(max(item["probabilities"][arm], 1e-12))
                for arm in MOVE_ORDER
            }
            selections.append(max(MOVE_ORDER, key=adjusted.__getitem__))
        p1[str(gamma)] = {
            **summarize(contexts, selections),
            "md7_top_vs_runner_prior_margin": margin_summary(contexts, gamma, "log"),
        }

    p2: dict[str, Any] = {}
    for kappa in P2_KAPPAS:
        selections = []
        for item, scores in zip(repeated_contexts, p0_scores, strict=True):
            adjusted = {
                arm: scores[arm] + kappa * item["probabilities"][arm]
                for arm in MOVE_ORDER
            }
            selections.append(max(MOVE_ORDER, key=adjusted.__getitem__))
        p2[str(kappa)] = {
            **summarize(contexts, selections),
            "md7_top_vs_runner_prior_margin": margin_summary(contexts, kappa, "linear"),
        }

    summary = {
        "schema_version": "turn_lints_cold_start_audit_v1",
        "source": str(PASSIVE_TURNS.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "context_count": len(contexts),
        "attempt_count": len({item["attempt_id"] for item in contexts}),
        "context_blocks": list(BLOCK_ORDER),
        "context_dimension": len(feature_names_for_blocks(BLOCK_ORDER)),
        "seed_count": SEED_COUNT,
        "hyperparameters": {
            "ridge_lambda": RIDGE_LAMBDA,
            "exploration_scale": EXPLORATION_SCALE,
            "posterior_updates": 0,
        },
        "md7_top1_counts": dict(Counter(item["md7_top"] for item in contexts)),
        "P0_current_zero_mean": summarize(contexts, p0_selections),
        "P1_log_selector_prior_sensitivity": p1,
        "P2_bayesian_coefficient_prior_sensitivity": p2,
        "P2_formulation": {
            "mean_prior": "m0_a = kappa * e(selector_p_a)",
            "information_form": "A0 = lambda I; b0_a = A0 m0_a",
            "pseudo_reward_observations": 0,
            "status": "mathematically coherent parameter prior; not selected or activated",
        },
        "interpretation": {
            "reward_claims": 0,
            "live_policy_changes": 0,
            "context_selection": "INCONCLUSIVE",
        },
    }
    return summary


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def write_report(summary: dict[str, Any]) -> None:
    p0 = summary["P0_current_zero_mean"]
    lines = [
        "# Turn-LinTS cold-start audit",
        "",
        "Selection-only offline diagnostic. No posterior update, training, reward fitting, or LIVE state was used.",
        "",
        "## Data and current prior",
        "",
        f"- Pre-action contexts: {summary['context_count']} across {summary['attempt_count']} attempts.",
        f"- Deterministic seeds: {summary['seed_count']} ({p0['draw_count']} selections).",
        "- Context: S+K+L+H+Q, dimension 22.",
        "- Current fresh prior: A=lambda I, b=0, lambda=1.0, exploration=0.20.",
        "",
        "## P0 — current zero-mean direct LinTS",
        "",
        f"MD7 top-1 agreement: **{fmt(p0['md7_top1_agreement_rate'])}**.",
        f"Telling when MD7 was not telling: **{fmt(p0['telling_when_md7_not_telling_rate'])}**.",
        "",
        "| selected arm | rate |",
        "|---|---:|",
    ]
    for arm in MOVE_ORDER:
        lines.append(f"| {arm} | {fmt(p0['arm_rates'][arm])} |")
    lines.extend(["", "Transition row rates (MD7 top-1 → fresh P0 selection):", "", "| MD7 | generic | probing | focus | telling |", "|---|---:|---:|---:|---:|"])
    for source in MOVE_ORDER:
        rates = p0["transition_row_rates"][source]
        lines.append("| " + source + " | " + " | ".join(fmt(rates[target]) for target in MOVE_ORDER) + " |")

    lines.extend([
        "",
        "Because all four arms have identical zero-mean covariance, selector probabilities in x do not create an initial preference by themselves. P0 is therefore effectively near-uniform at cold start.",
        "",
        "## P1 — additive log-selector action prior",
        "",
        "Score: `TS_reward_score_a + gamma * log(max(p_MD7(a), 1e-12))`.",
        "",
        "| gamma | MD7 agreement | telling when MD7 not telling |",
        "|---:|---:|---:|",
    ])
    for gamma, result in summary["P1_log_selector_prior_sensitivity"].items():
        lines.append(f"| {gamma} | {fmt(result['md7_top1_agreement_rate'])} | {fmt(result['telling_when_md7_not_telling_rate'])} |")
    lines.extend([
        "",
        "The prior is finite, so a learned reward-score advantage can override it once the advantage exceeds the logged selector-prior margin. Gamma is therefore a prior-strength parameter, not an eligibility gate, but a fixed gamma does not decay automatically.",
        "",
        "## P2 — Bayesian coefficient-mean prior",
        "",
        "Use `A0=lambda I`, `m0_a=kappa*e(selector_p_a)`, and `b0_a=A0*m0_a`. This is a Gaussian parameter prior whose initial mean score is `kappa*p_MD7(a)`; it does not create pseudo reward observations.",
        "",
        "| kappa | MD7 agreement | telling when MD7 not telling |",
        "|---:|---:|---:|",
    ])
    for kappa, result in summary["P2_bayesian_coefficient_prior_sensitivity"].items():
        lines.append(f"| {kappa} | {fmt(result['md7_top1_agreement_rate'])} | {fmt(result['telling_when_md7_not_telling_rate'])} |")
    lines.extend([
        "",
        "P2 is mathematically coherent with Bayesian linear regression and its influence naturally dilutes as A accumulates real observations. It still encodes a subjective reward-model prior scale, so it is not scientifically selected here and is not implemented in production.",
        "",
        "## Engineering conclusion",
        "",
        "A selector-centered cold-start prior is necessary before LIVE if initial behavior is expected to remain near MD7 rather than random across four arms. P1 is the smallest operational formulation; P2 is the more internally Bayesian formulation. Neither is activated. A separate prior-specification decision and offline verification are required before LIVE.",
        "",
        "Context remains **INCONCLUSIVE**. This diagnostic does not rerun or supersede the context ablation.",
    ])
    (OUTPUT_DIR / "cold_start_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    (OUTPUT_DIR / "cold_start_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(result)
    print(json.dumps({
        "contexts": result["context_count"],
        "seeds": result["seed_count"],
        "draws": result["P0_current_zero_mean"]["draw_count"],
        "p0_agreement": result["P0_current_zero_mean"]["md7_top1_agreement_rate"],
    }, indent=2))
