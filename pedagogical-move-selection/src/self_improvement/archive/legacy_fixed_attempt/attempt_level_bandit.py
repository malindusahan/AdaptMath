from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ============================================================
# PEDAGOGICAL MOVE SPACE
# ============================================================

MOVES = [
    "generic",
    "probing",
    "focus",
    "telling",
]


ARMS = [
    "baseline",
    "probing_bias",
    "focus_bias",
    "telling_bias",
    "generic_bias",
]


ARM_TO_MOVE = {
    "baseline": None,
    "probing_bias": "probing",
    "focus_bias": "focus",
    "telling_bias": "telling",
    "generic_bias": "generic",
}


# ============================================================
# CONTEXT
#
# Context exists BEFORE an attempt starts.
#
# No evaluator result from the current attempt.
# No mastery_after from the current attempt.
# No future conversation information.
# ============================================================

CONTEXT_FEATURE_NAMES = [
    "bias",
    "mastery_before",
    "attempt_index_normalized",
    "has_previous_feedback",
    "previous_evaluator_rate",
    "previous_mastery_delta",
]


CONTEXT_DIM = len(
    CONTEXT_FEATURE_NAMES
)


def build_attempt_context(
    mastery_before: float,
    attempt_index: int,
    previous_evaluator_score: Optional[int] = None,
    previous_mastery_delta: Optional[float] = None,
) -> np.ndarray:

    mastery_before = float(
        mastery_before
    )

    if not 0.0 <= mastery_before <= 1.0:
        raise ValueError(
            "mastery_before must be in [0, 1]."
        )


    attempt_index = int(
        attempt_index
    )

    if attempt_index < 0:
        raise ValueError(
            "attempt_index must be >= 0."
        )


    # Smooth bounded transformation:
    #
    # 0 -> 0.00
    # 1 -> 0.50
    # 2 -> 0.67
    # 3 -> 0.75
    #
    # Avoids inventing a hard maximum attempt count.
    attempt_index_normalized = (
        attempt_index
        /
        (
            attempt_index
            + 1.0
        )
    )


    has_previous_feedback = (
        previous_evaluator_score
        is not None
    )


    if has_previous_feedback:

        previous_evaluator_score = int(
            previous_evaluator_score
        )

        if previous_evaluator_score not in {
            0,
            1,
            2,
            3,
        }:
            raise ValueError(
                "previous_evaluator_score must be "
                "0, 1, 2, 3, or None."
            )

        previous_evaluator_rate = (
            previous_evaluator_score
            / 3.0
        )

    else:

        previous_evaluator_rate = 0.0


    if previous_mastery_delta is None:

        previous_mastery_delta_value = 0.0

    else:

        previous_mastery_delta_value = float(
            np.clip(
                previous_mastery_delta,
                -1.0,
                1.0,
            )
        )


    context = np.array(
        [
            1.0,

            mastery_before,

            attempt_index_normalized,

            float(
                has_previous_feedback
            ),

            previous_evaluator_rate,

            previous_mastery_delta_value,
        ],
        dtype=np.float64,
    )


    return context


# ============================================================
# LinUCB
# ============================================================

class DisjointLinUCB:

    def __init__(
        self,
        arms,
        context_dim,
        alpha=0.75,
    ):

        self.arms = list(
            arms
        )

        self.context_dim = int(
            context_dim
        )

        self.alpha = float(
            alpha
        )


        self.A = {
            arm: np.eye(
                self.context_dim,
                dtype=np.float64,
            )

            for arm in self.arms
        }


        self.b = {
            arm: np.zeros(
                self.context_dim,
                dtype=np.float64,
            )

            for arm in self.arms
        }


        self.counts = Counter()

        self.total_updates = 0


    def score_arm(
        self,
        arm,
        context,
    ):

        A_inv = np.linalg.inv(
            self.A[
                arm
            ]
        )


        theta = (
            A_inv
            @
            self.b[
                arm
            ]
        )


        expected_reward = float(
            theta
            @
            context
        )


        uncertainty = float(
            np.sqrt(
                context
                @
                A_inv
                @
                context
            )
        )


        ucb_score = (
            expected_reward
            +
            self.alpha
            * uncertainty
        )


        return {
            "expected_reward":
                expected_reward,

            "uncertainty":
                uncertainty,

            "selection_score":
                ucb_score,
        }


    def select_arm(
        self,
        context,
    ):

        diagnostics = {
            arm: self.score_arm(
                arm,
                context,
            )

            for arm in self.arms
        }


        # Deterministic tie-breaking.
        #
        # Because baseline appears first,
        # identical initial UCB scores choose baseline.
        selected_arm = max(
            self.arms,

            key=lambda arm: (
                diagnostics[
                    arm
                ][
                    "selection_score"
                ],

                -self.arms.index(
                    arm
                ),
            ),
        )


        return (
            selected_arm,
            diagnostics,
        )


    def update(
        self,
        arm,
        context,
        reward,
    ):

        reward = float(
            reward
        )


        self.A[
            arm
        ] += np.outer(
            context,
            context,
        )


        self.b[
            arm
        ] += (
            reward
            *
            context
        )


        self.counts[
            arm
        ] += 1

        self.total_updates += 1


    def state_dict(
        self,
    ):

        return {
            "type":
                "DisjointLinUCB",

            "arms":
                list(
                    self.arms
                ),

            "context_dim":
                int(
                    self.context_dim
                ),

            "alpha":
                float(
                    self.alpha
                ),

            "A": {
                arm:
                    self.A[
                        arm
                    ].tolist()

                for arm
                in self.arms
            },

            "b": {
                arm:
                    self.b[
                        arm
                    ].tolist()

                for arm
                in self.arms
            },

            "counts": {
                arm:
                    int(
                        self.counts[
                            arm
                        ]
                    )

                for arm
                in self.arms
            },

            "total_updates":
                int(
                    self.total_updates
                ),
        }


    def load_state_dict(
        self,
        state,
    ):

        if (
            state[
                "type"
            ]
            !=
            "DisjointLinUCB"
        ):
            raise ValueError(
                "Invalid LinUCB state type."
            )


        if (
            state[
                "arms"
            ]
            !=
            self.arms
        ):
            raise ValueError(
                "LinUCB arm configuration mismatch."
            )


        if (
            int(
                state[
                    "context_dim"
                ]
            )
            !=
            self.context_dim
        ):
            raise ValueError(
                "LinUCB context dimension mismatch."
            )


        self.A = {
            arm:
                np.asarray(
                    state[
                        "A"
                    ][
                        arm
                    ],
                    dtype=np.float64,
                )

            for arm
            in self.arms
        }


        self.b = {
            arm:
                np.asarray(
                    state[
                        "b"
                    ][
                        arm
                    ],
                    dtype=np.float64,
                )

            for arm
            in self.arms
        }


        self.counts = Counter({
            arm:
                int(
                    state[
                        "counts"
                    ][
                        arm
                    ]
                )

            for arm
            in self.arms
        })


        self.total_updates = int(
            state[
                "total_updates"
            ]
        )


# ============================================================
# Linear Thompson Sampling
#
# Comparator only.
#
# Uses the same linear reward assumptions and
# the same context/action space as LinUCB.
# ============================================================

class DisjointLinTS:

    def __init__(
        self,
        arms,
        context_dim,
        exploration_scale=0.50,
        seed=42,
    ):

        self.arms = list(
            arms
        )

        self.context_dim = int(
            context_dim
        )

        self.exploration_scale = float(
            exploration_scale
        )


        self.rng = np.random.default_rng(
            seed
        )


        self.A = {
            arm: np.eye(
                self.context_dim,
                dtype=np.float64,
            )

            for arm in self.arms
        }


        self.b = {
            arm: np.zeros(
                self.context_dim,
                dtype=np.float64,
            )

            for arm in self.arms
        }


        self.counts = Counter()

        self.total_updates = 0


    def sample_arm_score(
        self,
        arm,
        context,
    ):

        A_inv = np.linalg.inv(
            self.A[
                arm
            ]
        )


        posterior_mean = (
            A_inv
            @
            self.b[
                arm
            ]
        )


        posterior_covariance = (
            self.exploration_scale
            ** 2
            *
            A_inv
        )


        sampled_theta = (
            self.rng.multivariate_normal(
                mean=posterior_mean,
                cov=posterior_covariance,
            )
        )


        sampled_score = float(
            sampled_theta
            @
            context
        )


        expected_reward = float(
            posterior_mean
            @
            context
        )


        return {
            "expected_reward":
                expected_reward,

            "sampled_score":
                sampled_score,

            "selection_score":
                sampled_score,
        }


    def select_arm(
        self,
        context,
    ):

        diagnostics = {
            arm: self.sample_arm_score(
                arm,
                context,
            )

            for arm in self.arms
        }


        selected_arm = max(
            self.arms,

            key=lambda arm:
                diagnostics[
                    arm
                ][
                    "selection_score"
                ],
        )


        return (
            selected_arm,
            diagnostics,
        )


    def update(
        self,
        arm,
        context,
        reward,
    ):

        reward = float(
            reward
        )


        self.A[
            arm
        ] += np.outer(
            context,
            context,
        )


        self.b[
            arm
        ] += (
            reward
            *
            context
        )


        self.counts[
            arm
        ] += 1

        self.total_updates += 1


    def state_dict(
        self,
    ):

        return {
            "type":
                "DisjointLinTS",

            "arms":
                list(
                    self.arms
                ),

            "context_dim":
                int(
                    self.context_dim
                ),

            "exploration_scale":
                float(
                    self.exploration_scale
                ),

            "A": {
                arm:
                    self.A[
                        arm
                    ].tolist()

                for arm
                in self.arms
            },

            "b": {
                arm:
                    self.b[
                        arm
                    ].tolist()

                for arm
                in self.arms
            },

            "counts": {
                arm:
                    int(
                        self.counts[
                            arm
                        ]
                    )

                for arm
                in self.arms
            },

            "total_updates":
                int(
                    self.total_updates
                ),

            # Needed so resumed LinTS experiments remain
            # exactly reproducible.
            "rng_state":
                self.rng.bit_generator.state,
        }


    def load_state_dict(
        self,
        state,
    ):

        if (
            state[
                "type"
            ]
            !=
            "DisjointLinTS"
        ):
            raise ValueError(
                "Invalid LinTS state type."
            )


        if (
            state[
                "arms"
            ]
            !=
            self.arms
        ):
            raise ValueError(
                "LinTS arm configuration mismatch."
            )


        if (
            int(
                state[
                    "context_dim"
                ]
            )
            !=
            self.context_dim
        ):
            raise ValueError(
                "LinTS context dimension mismatch."
            )


        self.A = {
            arm:
                np.asarray(
                    state[
                        "A"
                    ][
                        arm
                    ],
                    dtype=np.float64,
                )

            for arm
            in self.arms
        }


        self.b = {
            arm:
                np.asarray(
                    state[
                        "b"
                    ][
                        arm
                    ],
                    dtype=np.float64,
                )

            for arm
            in self.arms
        }


        self.counts = Counter({
            arm:
                int(
                    state[
                        "counts"
                    ][
                        arm
                    ]
                )

            for arm
            in self.arms
        })


        self.total_updates = int(
            state[
                "total_updates"
            ]
        )


        self.rng.bit_generator.state = (
            state[
                "rng_state"
            ]
        )


# ============================================================
# ATTEMPT-LEVEL SELF-IMPROVEMENT POLICY
# ============================================================

class AttemptLevelBanditPolicy:

    def __init__(
        self,
        algorithm="linucb",
        alpha=0.75,
        lints_exploration_scale=0.50,
        max_probability_gap=0.15,
        seed=42,
        data_origin="synthetic",
    ):

        algorithm = (
            algorithm
            .strip()
            .lower()
        )


        if algorithm not in {
            "linucb",
            "lints",
        }:
            raise ValueError(
                "algorithm must be "
                "'linucb' or 'lints'."
            )


        data_origin = (
            str(
                data_origin
            )
            .strip()
            .lower()
        )


        if data_origin not in {
            "synthetic",
            "real",
        }:

            raise ValueError(
                "data_origin must be "
                "'synthetic' or 'real'."
            )


        self.data_origin = (
            data_origin
        )


        self.algorithm = algorithm

        self.alpha = float(
            alpha
        )

        self.lints_exploration_scale = float(
            lints_exploration_scale
        )

        self.max_probability_gap = float(
            max_probability_gap
        )

        self.seed = int(
            seed
        )


        # One independent bandit per skill.
        self.bandits: Dict[
            str,
            object
        ] = {}


        # Number of completed attempts per skill.
        self.completed_attempts = Counter()


        # At most ONE attempt may be active.
        self.active_attempt = None
    # ========================================================
    # SERIALIZATION
    #
    # JSON only.
    # We deliberately avoid pickle for research artifacts.
    # ========================================================

    def state_dict(
        self,
    ):

        if (
            self.active_attempt
            is not None
        ):

            raise RuntimeError(
                "Cannot serialize bandit state while "
                "a tutoring attempt is active."
            )


        return {
            "schema_version":
                "attempt_level_bandit_v1",

            "data_origin":
                self.data_origin,

            "algorithm":
                self.algorithm,

            "context_feature_names":
                list(
                    CONTEXT_FEATURE_NAMES
                ),

            "arms":
                list(
                    ARMS
                ),

            "alpha":
                float(
                    self.alpha
                ),

            "lints_exploration_scale":
                float(
                    self.lints_exploration_scale
                ),

            "max_probability_gap":
                float(
                    self.max_probability_gap
                ),

            "seed":
                int(
                    self.seed
                ),

            "completed_attempts": {
                skill_id:
                    int(
                        count
                    )

                for skill_id, count
                in self.completed_attempts.items()
            },

            "bandits": {
                skill_id:
                    bandit.state_dict()

                for skill_id, bandit
                in self.bandits.items()
            },
        }


    def save_state(
        self,
        path,
    ):

        path = Path(
            path
        )


        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        state = (
            self.state_dict()
        )


        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                state,
                f,
                indent=2,
            )


        return path


    def load_state(
        self,
        path,
    ):

        if (
            self.active_attempt
            is not None
        ):

            raise RuntimeError(
                "Cannot load state while "
                "an attempt is active."
            )


        path = Path(
            path
        )


        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(
                f
            )


        if (
            state[
                "schema_version"
            ]
            !=
            "attempt_level_bandit_v1"
        ):

            raise ValueError(
                "Unsupported bandit state schema."
            )


        # ====================================================
        # CRITICAL SAFETY BARRIER
        #
        # Synthetic state may NEVER be loaded into a
        # policy configured for real research data.
        # ====================================================

        if (
            state[
                "data_origin"
            ]
            !=
            self.data_origin
        ):

            raise ValueError(
                "Bandit data-origin mismatch. "
                f"Policy expects '{self.data_origin}' "
                f"but state contains "
                f"'{state['data_origin']}'."
            )


        if (
            state[
                "algorithm"
            ]
            !=
            self.algorithm
        ):

            raise ValueError(
                "Bandit algorithm mismatch."
            )


        if (
            state[
                "context_feature_names"
            ]
            !=
            list(
                CONTEXT_FEATURE_NAMES
            )
        ):

            raise ValueError(
                "Context-feature configuration mismatch."
            )


        if (
            state[
                "arms"
            ]
            !=
            list(
                ARMS
            )
        ):

            raise ValueError(
                "Bandit arm configuration mismatch."
            )


        # Start from an empty state.
        self.reset_state()


        self.completed_attempts = Counter({
            skill_id:
                int(
                    count
                )

            for skill_id, count
            in state[
                "completed_attempts"
            ].items()
        })


        for (
            skill_id,
            bandit_state,
        ) in state[
            "bandits"
        ].items():

            bandit = (
                self._create_bandit(
                    skill_id
                )
            )


            bandit.load_state_dict(
                bandit_state
            )


            self.bandits[
                skill_id
            ] = bandit


        return {
            "path":
                str(
                    path
                ),

            "data_origin":
                self.data_origin,

            "algorithm":
                self.algorithm,

            "total_updates":
                self.get_total_updates(),

            "skills":
                sorted(
                    self.bandits.keys()
                ),
        }


    # ========================================================
    # INTERNAL BANDIT CREATION
    # ========================================================

    def _create_bandit(
        self,
        skill_id,
    ):

        if self.algorithm == "linucb":

            return DisjointLinUCB(
                arms=ARMS,
                context_dim=CONTEXT_DIM,
                alpha=self.alpha,
            )


        # Stable but skill-dependent random seed.
        #
        # Used only so separate skill bandits do not
        # generate identical Thompson samples.
        skill_seed = (
            self.seed
            +
            sum(
                ord(character)

                for character
                in skill_id
            )
        )


        return DisjointLinTS(
            arms=ARMS,
            context_dim=CONTEXT_DIM,

            exploration_scale=
                self.lints_exploration_scale,

            seed=skill_seed,
        )


    def _get_bandit(
        self,
        skill_id,
    ):

        if skill_id not in self.bandits:

            self.bandits[
                skill_id
            ] = self._create_bandit(
                skill_id
            )


        return self.bandits[
            skill_id
        ]


    # ========================================================
    # START ATTEMPT
    #
    # Bandit selects ONE overlay here.
    #
    # No more arm selection is allowed until this
    # attempt finishes.
    # ========================================================

    def start_attempt(
        self,
        skill_id,
        mastery_before,
        attempt_index=0,
        previous_evaluator_score=None,
        previous_mastery_delta=None,
    ):

        if self.active_attempt is not None:

            raise RuntimeError(
                "Cannot start a new attempt while "
                "another attempt is active."
            )


        skill_id = str(
            skill_id
        ).strip()


        if not skill_id:

            raise ValueError(
                "skill_id cannot be empty."
            )


        context = build_attempt_context(
            mastery_before=
                mastery_before,

            attempt_index=
                attempt_index,

            previous_evaluator_score=
                previous_evaluator_score,

            previous_mastery_delta=
                previous_mastery_delta,
        )


        bandit = self._get_bandit(
            skill_id
        )


        selected_arm, diagnostics = (
            bandit.select_arm(
                context
            )
        )


        self.active_attempt = {
            "skill_id":
                skill_id,

            "mastery_before":
                float(
                    mastery_before
                ),

            "attempt_index":
                int(
                    attempt_index
                ),

            "context":
                context,

            "selected_arm":
                selected_arm,

            "selection_diagnostics":
                diagnostics,

            "turn_count":
                0,

            "override_count":
                0,

            "updates_at_start":
                int(
                    bandit.total_updates
                ),
        }


        return {
            "skill_id":
                skill_id,

            "selected_arm":
                selected_arm,

            "algorithm":
                self.algorithm,

            "selection_diagnostics":
                diagnostics,
        }


    # ========================================================
    # APPLY FIXED OVERLAY
    #
    # Called every tutoring turn.
    #
    # IMPORTANT:
    # This method NEVER updates the bandit.
    # ========================================================

    def apply_overlay(
        self,
        base_move_probs,
    ):

        if self.active_attempt is None:

            raise RuntimeError(
                "No active attempt. "
                "Call start_attempt() first."
            )


        missing_moves = [
            move

            for move in MOVES

            if move not in base_move_probs
        ]


        if missing_moves:

            raise ValueError(
                "Missing move probabilities: "
                + ", ".join(
                    missing_moves
                )
            )


        values = {
            move:
                float(
                    base_move_probs[
                        move
                    ]
                )

            for move in MOVES
        }


        if any(
            value < 0.0

            for value
            in values.values()
        ):
            raise ValueError(
                "Move probabilities cannot be negative."
            )


        total_probability = sum(
            values.values()
        )


        if total_probability <= 0.0:

            raise ValueError(
                "Move probabilities must have "
                "positive total mass."
            )


        # Normalize defensively.
        normalized = {
            move:
                value
                /
                total_probability

            for move, value
            in values.items()
        }


        base_move = max(
            MOVES,

            key=lambda move:
                normalized[
                    move
                ],
        )


        selected_arm = (
            self.active_attempt[
                "selected_arm"
            ]
        )


        preferred_move = (
            ARM_TO_MOVE[
                selected_arm
            ]
        )


        overridden = False

        probability_gap = None

        final_move = base_move


        if preferred_move is not None:

            if preferred_move != base_move:

                probability_gap = (
                    normalized[
                        base_move
                    ]
                    -
                    normalized[
                        preferred_move
                    ]
                )


                if (
                    probability_gap
                    <=
                    self.max_probability_gap
                ):

                    final_move = (
                        preferred_move
                    )

                    overridden = True


        self.active_attempt[
            "turn_count"
        ] += 1


        if overridden:

            self.active_attempt[
                "override_count"
            ] += 1


        # Critical assertion:
        #
        # no bandit update may occur during
        # an active tutoring attempt.
        skill_id = (
            self.active_attempt[
                "skill_id"
            ]
        )


        bandit = self.bandits[
            skill_id
        ]


        if (
            bandit.total_updates
            !=
            self.active_attempt[
                "updates_at_start"
            ]
        ):

            raise RuntimeError(
                "Bandit state changed during an "
                "active tutoring attempt."
            )


        return {
            "selected_arm":
                selected_arm,

            "base_move":
                base_move,

            "final_move":
                final_move,

            "preferred_move":
                preferred_move,

            "overridden":
                bool(
                    overridden
                ),

            "probability_gap":
                probability_gap,

            "base_move_probs":
                normalized,
        }


    # ========================================================
    # FINISH ATTEMPT
    #
    # This is the ONLY place where learning occurs.
    # ========================================================

    def finish_attempt(
        self,
        evaluator_score,
        mastery_after,
    ):

        if self.active_attempt is None:

            raise RuntimeError(
                "No active attempt to finish."
            )


        evaluator_score = int(
            evaluator_score
        )


        if evaluator_score not in {
            0,
            1,
            2,
            3,
        }:
            raise ValueError(
                "evaluator_score must be "
                "0, 1, 2, or 3."
            )


        mastery_after = float(
            mastery_after
        )


        if not 0.0 <= mastery_after <= 1.0:

            raise ValueError(
                "mastery_after must be in [0, 1]."
            )


        attempt = self.active_attempt


        skill_id = attempt[
            "skill_id"
        ]


        bandit = self.bandits[
            skill_id
        ]


        # --------------------------------------------
        # Primary contextual-bandit reward.
        #
        # Mastery is deliberately NOT mixed into
        # this reward using arbitrary weights.
        # --------------------------------------------

        reward = (
            evaluator_score
            /
            3.0
        )


        mastery_delta = (
            mastery_after
            -
            attempt[
                "mastery_before"
            ]
        )


        updates_before = (
            bandit.total_updates
        )


        # ====================================================
        # ONE AND ONLY ONE LEARNING UPDATE
        # ====================================================

        bandit.update(
            arm=
                attempt[
                    "selected_arm"
                ],

            context=
                attempt[
                    "context"
                ],

            reward=
                reward,
        )


        if (
            bandit.total_updates
            !=
            updates_before
            + 1
        ):

            raise RuntimeError(
                "Expected exactly one bandit update "
                "when finishing an attempt."
            )


        self.completed_attempts[
            skill_id
        ] += 1


        result = {
            "skill_id":
                skill_id,

            "algorithm":
                self.algorithm,

            "selected_arm":
                attempt[
                    "selected_arm"
                ],

            "evaluator_score":
                evaluator_score,

            "reward":
                reward,

            "mastery_before":
                attempt[
                    "mastery_before"
                ],

            "mastery_after":
                mastery_after,

            "mastery_delta":
                mastery_delta,

            "turn_count":
                attempt[
                    "turn_count"
                ],

            "override_count":
                attempt[
                    "override_count"
                ],

            "bandit_updates_after":
                int(
                    bandit.total_updates
                ),
        }


        # Attempt is now closed.
        self.active_attempt = None


        return result


    # ========================================================
    # INSPECTION
    # ========================================================

    def get_total_updates(
        self,
    ):

        return sum(
            bandit.total_updates

            for bandit
            in self.bandits.values()
        )


    def get_skill_updates(
        self,
        skill_id,
    ):

        if skill_id not in self.bandits:
            return 0


        return int(
            self.bandits[
                skill_id
            ].total_updates
        )


    # ========================================================
    # RESET
    #
    # Must be called before beginning the REAL research
    # experiment if this object has been used with dummy data.
    # ========================================================

    def reset_state(
        self,
    ):

        self.bandits = {}

        self.completed_attempts = Counter()

        self.active_attempt = None
