"""Frozen AdaptMath Turn-LinTS v1 architecture constants.

The scientific evidence lives in
``results/turn_lints_final_architecture_selection_v1``.  This module is the
small runtime-facing contract; it does not activate LIVE or initialize a
learned posterior.
"""

from __future__ import annotations

from typing import Final


ARCHITECTURE_VERSION: Final[str] = "adaptmath_turn_lints_v1_skl_final"
ALGORITHM: Final[str] = "direct_disjoint_linear_thompson_sampling"
ARMS: Final[tuple[str, ...]] = ("generic", "probing", "focus", "telling")
CONTEXT_PRESET: Final[str] = "S+K+L"
ORDERED_CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "selector_p_generic",
    "selector_p_probing",
    "selector_p_focus",
    "selector_p_telling",
    "mastery_before",
    "previous_mastery_delta",
    "previous_mastery_delta_missing",
    "previous_reasoning_probability",
    "previous_uncertainty_probability",
    "previous_clarification_probability",
    "previous_learner_signals_missing",
)
CONTEXT_DIMENSION: Final[int] = 11
REWARD_MODE: Final[str] = "headroom_normalized"
SELECTOR_NAME: Final[str] = "MD7-R2-TELL-C1 Epoch 2"
SELECTOR_SHA256: Final[str] = (
    "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"
)
COLD_START_BEHAVIOR_POLICY: Final[str] = (
    "md7_r2_probability_proportional_v1"
)
MRB1_ROLE: Final[str] = "auxiliary_diagnostic_not_policy_context_or_reward"
STUDENT_MODEL_ROLE: Final[str] = (
    "previous_learner_state_L_context_provider_and_diagnostics;"
    "valid_linked_bkt_posterior_updates_define_turn_reward"
)
FORMAL_ASSESSMENT_ROLE: Final[str] = "separate_external_policy_health_signal"
LEARNER_AGENCY_SEMANTICS: Final[str] = (
    "explicit_interaction_requests_may_override_above_policy_nonrandomized;"
    "inferred_learner_state_never_forces_actions"
)
ANCHOR_MODE: Final[str] = "none"
ANCHOR_GAMMA: Final[float] = 0.0
TAU_GATE: Final[None] = None
P1_ANCHOR_ACTIVE: Final[bool] = False
P2_ACTIVE: Final[bool] = False
HANDWRITTEN_PEDAGOGICAL_STATE_ACTION_RULES: Final[bool] = False
EXPERIMENT_ARTIFACT: Final[str] = (
    "pedagogical-move-selection/results/"
    "turn_lints_final_architecture_selection_v1"
)


__all__ = (
    "ALGORITHM",
    "ANCHOR_GAMMA",
    "ANCHOR_MODE",
    "ARCHITECTURE_VERSION",
    "ARMS",
    "COLD_START_BEHAVIOR_POLICY",
    "CONTEXT_DIMENSION",
    "CONTEXT_PRESET",
    "EXPERIMENT_ARTIFACT",
    "FORMAL_ASSESSMENT_ROLE",
    "HANDWRITTEN_PEDAGOGICAL_STATE_ACTION_RULES",
    "ORDERED_CONTEXT_FEATURES",
    "P1_ANCHOR_ACTIVE",
    "P2_ACTIVE",
    "REWARD_MODE",
    "LEARNER_AGENCY_SEMANTICS",
    "MRB1_ROLE",
    "SELECTOR_NAME",
    "SELECTOR_SHA256",
    "TAU_GATE",
    "STUDENT_MODEL_ROLE",
)
