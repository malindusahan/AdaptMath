# Frozen MD7 action-anchor implementation notes

## Scope

The new code adds opt-in `md7_logprob_anchor` support to the fresh direct-arm
Turn-LinTS implementation only. Legacy LinTS code is unchanged. The calibrated
gamma was not activated because the scientific decision is B.

## Policy interface and state

`DirectTurnLinTS.select_arm` now accepts `selector_probabilities` separately
from the reward-model context. It returns:

- `contextual_ts_reward_scores` from A/b only;
- `anchor_scores` from frozen MD7 only;
- `policy_scores` as their sum;
- `p0_selected_arm` and, when configured, `p1_selected_arm`.

State schema v2 adds and fail-closes on:

- `anchor_mode`
- `anchor_gamma`
- `anchor_selector_version`
- `anchor_selector_sha256`
- `anchor_probability_floor`

The fixed floor is `1e-12`; it prevents log(0) and does not filter an arm.
`anchor_mode=none` requires gamma zero. Existing direct Turn-LinTS state v1 is
accepted only by an unanchored policy. An anchored policy cannot implicitly
migrate or overwrite it.

## Runtime diagnostics

Derived event schema v3 supports:

- `actual_treatment_move`
- `shadow_p0_move`
- `shadow_p1_move`
- `shadow_moves_are_observed_treatments=false`
- separate contextual reward, anchor, and combined policy scores
- anchor mode/gamma/selector provenance

In SHADOW, the actual move remains MD7 and posterior updates remain zero. A
reward continues to be attributed only to `actual_treatment_move`.

## Configuration

The local launcher and coordinator accept:

```text
ADAPTIVE_TURN_LINTS_ANCHOR_MODE=none|md7_logprob_anchor
ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA=<finite nonnegative scalar>
```

Defaults remain `none` and `0.0`. This task did not alter the current runtime
state, start a new anchored lineage, enable LIVE, or generate Tutor calls.

