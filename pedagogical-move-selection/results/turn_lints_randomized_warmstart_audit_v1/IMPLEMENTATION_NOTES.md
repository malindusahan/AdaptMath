# RANDOMIZED_WARMSTART implementation notes

## Status

Decision A supported an implementation, but **did not authorize activation**.
The runtime default remains `SHADOW`; no randomized treatment event was
created, and no Turn-LinTS posterior was updated.

## Isolated implementation

- `randomized_warmstart.py` implements one assignment rule only:
  `action ~ Categorical(raw frozen MD7 probabilities)`.
- Production assignment uses `secrets.SystemRandom`, backed by operating-system
  entropy. It is not seeded from learner or attempt identity and is not globally
  deterministic.
- Tests may inject a deterministic random source or nonnegative NumPy seed.
- Validation requires exactly the four canonical moves, finite nonnegative
  probabilities, and a sum within `1e-5` of one. The only renormalization is a
  whole-vector floating-point correction.
- There is no temperature, tau gate, minimum exploration probability,
  threshold, subset renormalization, eligibility filter, or pedagogical rule.

`TurnLinTSMode` now recognizes `OFF`, `SHADOW`, `RANDOMIZED_WARMSTART`, and
`LIVE`. In warm-start mode the sampled MD7 move is the actual treatment passed
to the existing Tutor realization path. A valid BKT transition logs raw and
headroom-normalized deltas, but the posterior update branch remains restricted
to `LIVE`.

## Lineage and schema

Prepared, inactive lineage:

`adaptive-math-tutor/backend/runtime/turn_lints_randomized_warmstart_v1/`

It contains only:

- `lineage_manifest.json`
- a fresh `policy_state.json` with zero applied updates and zero per-arm counts
- an empty `pending_actions/` directory

It intentionally contains no `turn_events.jsonl`.

The event schema is `adaptmath_turn_lints_event_v4`. New treatment-assignment
fields are:

- `decision_source`
- `treatment_assignment_source`
- `behavior_policy`
- `behavior_propensity`
- `full_behavior_probability_vector`
- `randomized_assignment`
- `assignment_rng_source`
- `assignment_seed`
- `assignment_random_draw`
- `posterior_updated`

Existing action/outcome fields retain the linkage: `attempt_id`,
`action_event_id`, `action_turn_index`, actual move, Tutor response, current
MRB1 heads, `resolver_event_id`, mastery before/after, both reward transforms,
and resolution status.

Behavior-policy identifier:
`md7_r2_probability_proportional_v1`.

Lineage schema:
`adaptmath_randomized_warmstart_lineage_v1`.

An externally forced action is explicitly marked `randomized_assignment=false`,
uses behavior policy `external_override`, and receives no MD7 behavior
propensity. In the current normal MD7/direct-Turn-LinTS path, the legacy
learner-agency escalation wrapper is disabled. The legacy manual move endpoint
also fails with HTTP 409 while the adaptive lifecycle is active. Rollback and
manual-controlled modes retain their existing, separate behavior and were not
changed.

## Reproducible preparation

`adaptive-math-tutor/backend/scripts/prepare_randomized_warmstart_lineage.py`
creates or verifies the fresh zero-update state and manifest. It fails if the
state has updates or if an event log exists. Running it does not start an
attempt or sample an action.
