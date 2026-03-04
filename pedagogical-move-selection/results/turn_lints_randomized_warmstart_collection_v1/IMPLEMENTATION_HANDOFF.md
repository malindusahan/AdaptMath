# RANDOMIZED_WARMSTART collection handoff

## Active collection contract

| Field | Value |
|---|---|
| Runtime mode | `RANDOMIZED_WARMSTART` |
| Runtime lineage | `adaptive-math-tutor/backend/runtime/turn_lints_randomized_warmstart_v1/` |
| Event log | `adaptive-math-tutor/backend/runtime/turn_lints_randomized_warmstart_v1/turn_events.jsonl` |
| Behavior policy | `md7_r2_probability_proportional_v1` |
| Selector | `MD7-R2-TELL-C1 epoch 2` |
| Selector SHA-256 | `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5` |
| Reward | `headroom_normalized` |
| Context | `S+K+L+H+Q`, 22 ordered features |
| Event schema | `adaptmath_turn_lints_event_v4` |
| Lineage schema | `adaptmath_randomized_warmstart_lineage_v1` |
| Posterior updates | Disabled; count must remain zero |

The supporting decision tables, limitations, and four focused figures are in
[`DECISION_EVIDENCE.md`](DECISION_EVIDENCE.md).

## Verified activation state

The local research stack was restarted through the existing launcher with
`ADAPTIVE_TURN_LINTS_MODE=RANDOMIZED_WARMSTART`. The post-start status check
reported all services ready and confirmed:

- mode `RANDOMIZED_WARMSTART`;
- behavior policy `md7_r2_probability_proportional_v1`;
- frozen MD7-R2-TELL-C1 epoch 2 and the expected SHA-256;
- real data mode, `headroom_normalized`, and 22-feature `S+K+L+H+Q`;
- LIVE disabled;
- posterior updates disabled and update count zero; and
- the dedicated state/event root shown above.

The post-activation readiness check still reported zero completed events, zero
pending actions, zero randomized observations, and zero posterior updates. No
tutoring interaction was generated during setup.

## Assignment and propensity semantics

For every normal action, frozen MD7-R2 produces the complete ordered vector
`generic, probing, focus, telling`. The actual treatment is sampled once from
that unchanged categorical distribution. There is no tau, threshold,
temperature, epsilon mixture, action block, top-1 override, or state-based rule.

Every genuine randomized event records:

- `randomized_assignment = true`
- `behavior_policy = md7_r2_probability_proportional_v1`
- `actual_treatment_move`
- `behavior_propensity = P_MD7(actual_treatment_move | x_t)`
- `full_behavior_probability_vector`
- decision/treatment source and RNG provenance

The sampled move passes through the existing authoritative Tutor realization
contract. The internal move label is not exposed to the learner.

If an external mechanism supplies the action, the event is
`randomized_assignment = false`, records the true external source, and has no
fabricated MD7 behavior propensity.

## Context, outcome, and update guarantees

The 22-feature context retains exact ordered feature names and values. `L_t`
uses only the preceding learner response; `Q_t` uses only preceding Tutor MRB1.
Current learner response, current MRB1, and current BKT after-state are excluded
from current `x_t`.

After valid resolution, the event links the actual treatment to current MRB1,
resolver identity/status, mastery before/after, raw delta, and
headroom-normalized delta. Formal assessment remains separate.

Only `LIVE` enters the posterior-update branch. In randomized warm-start,
`posterior_updated=false`; `A`, `b`, per-arm counts, and total update count stay
unchanged.

## Readiness command

From the workspace root:

```powershell
python pedagogical-move-selection/results/turn_lints_randomized_warmstart_collection_v1/randomized_data_readiness.py
```

The report is descriptive and has no automatic threshold. It reports genuine
randomized actions, resolved rewards, attempts, linked learners/skills, context
completeness, action counts, propensity and reward distributions, resolution
statuses, external overrides, behavior-policy/propensity integrity violations,
and posterior update counts.

## Activation and status commands

The existing launcher environment is the sole configuration path. A future
restart must set the mode explicitly because the safe launcher default remains
SHADOW:

```powershell
$env:ADAPTIVE_TURN_LINTS_MODE = 'RANDOMIZED_WARMSTART'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\stop-adaptmath-local.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start-adaptmath-local.ps1
```

Check the active process configuration with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\status-adaptmath-local.ps1
```

The status output must show the frozen selector/hash, behavior policy,
`RANDOMIZED_WARMSTART`, `headroom_normalized`, `S+K+L+H+Q` with dimension 22,
LIVE disabled, posterior updates disabled, and update count zero.

## Baseline safety state

The collection lineage began with zero completed events, zero pending actions,
zero randomized observations, zero applied update IDs, and zero arm updates.
No artificial tutoring session or external Tutor API call was used for setup.
