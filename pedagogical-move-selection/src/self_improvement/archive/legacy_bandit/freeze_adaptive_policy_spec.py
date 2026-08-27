import json
from pathlib import Path


OUTPUT_PATH = Path(
    "results/self_improvement/"
    "frozen_adaptive_policy_spec.json"
)


def main():

    spec = {
        "schema_version":
            "adaptive_policy_v1",

        "status":
            "frozen_primary_architecture",

        "purpose":
            (
                "Primary self-improvement architecture for "
                "pedagogical move selection."
            ),

        # ====================================================
        # BASE SUPERVISED SELECTOR
        # ====================================================

        "base_selector": {
            "architecture":
                "context_routed_hybrid",

            "input_contract": {
                "runtime_inputs": [
                    "problem",
                    "conversation_history",
                ],

                "forbidden_runtime_inputs": [
                    "ground_truth",
                    "reference_answer",
                    "student_incorrect_solution",
                    "future_turns",
                    "historical_move_labels",
                    "teacher_described_confusion",
                    "self_correctness",
                    "self_typical_confusion",
                    "self_typical_interactions",
                    "scenario",
                ],
            },

            "move_space": [
                "generic",
                "probing",
                "focus",
                "telling",
            ],

            "routing": {
                "routing_tokenizer_source":
                    "MD4_DistilRoBERTa",

                "threshold_tokens":
                    512,

                "le_512_model": {
                    "experiment":
                        "MD4_weighted_DistilRoBERTa",

                    "model":
                        "distilbert/distilroberta-base",

                    "training_loss":
                        "inverse_frequency_weighted_cross_entropy",

                    "frozen_epoch":
                        3,
                },

                "gt_512_model": {
                    "experiment":
                        "MD3_ModernBERT",

                    "model":
                        "answerdotai/ModernBERT-base",

                    "max_length":
                        1792,

                    "training_loss":
                        "ordinary_cross_entropy",

                    "frozen_epoch":
                        3,
                },
            },

            "validation": {
                "examples":
                    1850,

                "md4_route_examples":
                    1507,

                "md3_route_examples":
                    343,

                "accuracy":
                    0.5005405405405405,

                "macro_f1":
                    0.48676777751987244,
            },

            "current_artifact_status": {
                "validation_behavior_recovered":
                    True,

                "offline_prediction_replay_available":
                    True,

                "new_input_inference_available":
                    False,

                "reason":
                    (
                        "MD3 and MD4 deployable model weights "
                        "have not yet been recovered/reproduced."
                    ),
            },
        },

        # ====================================================
        # ONLINE ADAPTATION
        # ====================================================

        "adaptive_policy": {
            "level":
                "attempt",

            "algorithm":
                "disjoint_linear_thompson_sampling",

            "policy_class":
                "AttemptLevelBanditPolicy",

            "bandit_class":
                "DisjointLinTS",

            "separate_bandit_per_skill":
                True,

            "action_space": [
                "baseline",
                "probing_bias",
                "focus_bias",
                "telling_bias",
                "generic_bias",
            ],

            "arm_to_move": {
                "baseline":
                    None,

                "probing_bias":
                    "probing",

                "focus_bias":
                    "focus",

                "telling_bias":
                    "telling",

                "generic_bias":
                    "generic",
            },

            "arm_selection_timing":
                "once_at_attempt_start",

            "arm_fixed_during_attempt":
                True,

            "turn_level_arm_reselection":
                False,

            "turn_level_bandit_updates":
                False,

            "updates_per_completed_attempt":
                1,

            # This is the current reusable implementation
            # configuration used by SI7-C/SI7-D.
            "conservative_overlay": {
                "enabled":
                    True,

                "max_probability_gap":
                    0.15,

                "rule":
                    (
                        "Override the frozen selector only when "
                        "P(base_move) - P(preferred_move) <= 0.15."
                    ),

                "baseline_never_overrides":
                    True,
            },

            # Current implementation default.
            #
            # Keep the structural architecture frozen even if
            # exploration-scale sensitivity is later reported
            # as an ablation.
            "lints": {
                "exploration_scale":
                    0.50,

                "exploration_scale_status":
                    "current_primary_implementation_value",

                "posterior_sampling":
                    True,
            },
        },

        # ====================================================
        # ATTEMPT CONTEXT
        # ====================================================

        "attempt_context": {
            "feature_names": [
                "bias",
                "mastery_before",
                "attempt_index_normalized",
                "has_previous_feedback",
                "previous_evaluator_rate",
                "previous_mastery_delta",
            ],

            "runtime_availability_requirement":
                "all_features_available_before_arm_selection",

            "uses_future_information":
                False,

            "previous_attempt_feedback_only":
                True,
        },

        # ====================================================
        # REWARD
        # ====================================================

        "reward": {
            "primary_source":
                "external_evaluator",

            "evaluator_score_domain": [
                0,
                1,
                2,
                3,
            ],

            "formula":
                "evaluator_score / 3.0",

            "reward_range": [
                0.0,
                1.0,
            ],

            "mastery_in_primary_reward":
                False,

            "mastery_usage": [
                "attempt_context",
                "secondary_outcome_analysis",
                "diagnostics",
                "previous_attempt_feedback",
            ],

            "credit_assignment_level":
                "attempt",

            "individual_turns_receive_terminal_reward":
                False,
        },

        # ====================================================
        # ATTEMPT LIFECYCLE
        # ====================================================

        "lifecycle": {
            "sequence": [
                "receive_problem_and_history",
                "receive_mastery_before",
                "start_attempt",
                "build_attempt_context",
                "select_one_lints_arm",
                "freeze_arm_for_attempt",
                "get_frozen_selector_probabilities_each_turn",
                "apply_conservative_overlay_each_turn",
                "send_final_move_to_tutor_agent",
                "continue_attempt_without_bandit_update",
                "receive_external_evaluator_score",
                "receive_mastery_after",
                "finish_attempt",
                "perform_exactly_one_lints_update",
            ],

            "learning_allowed_inside_apply_overlay":
                False,

            "learning_location":
                "finish_attempt_only",
        },

        # ====================================================
        # EXTERNAL COMPONENTS
        # ====================================================

        "external_components": {
            "tutor_agent": {
                "owned_by_this_component":
                    False,

                "receives_selected_move":
                    True,

                "generates_teacher_utterance":
                    True,
            },

            "evaluator_agent": {
                "owned_by_this_component":
                    False,

                "score_domain":
                    "0_to_3",

                "three_out_of_three_success":
                    True,
            },

            "mastery_estimator": {
                "owned_by_this_component":
                    False,

                "expected_range":
                    "0_to_1",
            },
        },

        # ====================================================
        # SAFETY / DATA ORIGIN
        # ====================================================

        "research_constraints": {
            "test_split_touched_for_adaptive_development":
                False,

            "synthetic_data_allowed_for":
                [
                    "software_validation",
                    "algorithm_validation",
                    "integration_validation",
                ],

            "synthetic_data_forbidden_for":
                [
                    "seeding_real_policy_state",
                    "training_final_real_policy",
                    "educational_effectiveness_claims",
                ],

            "synthetic_bandit_state_must_not_be_reused_as_real":
                True,
        },

        # ====================================================
        # EXPLICITLY REJECTED / NON-PRIMARY DESIGNS
        # ====================================================

        "not_in_primary_architecture": {
            "weighted_evaluator_mastery_reward": {
                "formula":
                    "0.8 * evaluator + 0.2 * mastery_signal",

                "status":
                    "provisional_experiment_not_used",
            },

            "direct_move_bandit_arms": {
                "arms": [
                    "generic",
                    "probing",
                    "focus",
                    "telling",
                ],

                "status":
                    "not_primary",
            },

            "turn_level_pedagogical_reward_model": {
                "status":
                    "possible_future_comparison_or_extension",

                "used_in_primary_policy":
                    False,
            },

            "turn_level_lints_updates": {
                "used_in_primary_policy":
                    False,
            },
        },

        # ====================================================
        # VALIDATION EVIDENCE
        # ====================================================

        "validation_evidence": {
            "true_lints_convergence":
                "results/self_improvement/"
                "si6e_lints_convergence.json",

            "frozen_selector_recovery":
                "results/self_improvement/"
                "si7a_frozen_selector_recovery_validation.json",

            "real_probability_overlay":
                "results/self_improvement/"
                "si7c_real_frozen_overlay_validation.json",

            "attempt_level_integration":
                "results/self_improvement/"
                "si7d_recovered_selector_attempt_lints_integration.json",
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            spec,
            f,
            indent=2,
        )

    print("=" * 76)
    print(
        "SI8-A ADAPTIVE POLICY SPECIFICATION FREEZE"
    )
    print("=" * 76)
    print()

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_PATH,
    )

    print()

    print(
        "Architecture:",
        spec[
            "adaptive_policy"
        ][
            "algorithm"
        ],
    )

    print(
        "Adaptive level:",
        spec[
            "adaptive_policy"
        ][
            "level"
        ],
    )

    print(
        "Reward:",
        spec[
            "reward"
        ][
            "formula"
        ],
    )

    print(
        "Max probability gap:",
        spec[
            "adaptive_policy"
        ][
            "conservative_overlay"
        ][
            "max_probability_gap"
        ],
    )

    print()

    print("=" * 76)
    print(
        "SPECIFICATION WRITTEN"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()