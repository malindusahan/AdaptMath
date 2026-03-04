# Turn-LinTS SHADOW collection v2 handoff

## Preserved scientific decisions

- Frozen production selector: MD7-R2-TELL-C1 Epoch 2,
  SHA-256 `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`.
- Future SHADOW reward configuration: `headroom_normalized`.
- Reward interpretation: a transformed BKT posterior-belief update, not
  measured learning.
- Context status: **INCONCLUSIVE**. S+K+L+H+Q is retained as an instrumentation
  context for prospective ablation, not as the selected LIVE context.
- Turn-LinTS mode: **SHADOW**; posterior updates: zero.

## Runtime configuration and lineage

The normal local launch defaults now resolve to:

```text
ADAPTIVE_TURN_LINTS_MODE=SHADOW
ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS=S+K+L+H+Q
ADAPTIVE_TURN_LINTS_REWARD_MODE=headroom_normalized
adaptive-math-tutor/backend/runtime/turn_lints_shadow_full_headroom_v1/
```

The isolated directory contains a fresh 22-dimensional direct-disjoint state.
Every arm has `A=I`, `b=0`, and update count zero. It does not reuse or modify
`turn_lints_shadow_full_raw_v1`.

The event ledger now records the full named context, response provenance, both
reward candidates, resolution status, and explicit actual-versus-hypothetical
treatment attribution. See `shadow_event_schema.md`.

## SHADOW guarantees

Focused tests verify that a SHADOW draw leaves the MD7 Tutor move unchanged,
does not mutate A or b, does not increment arm or global update counts, and
cannot place an update identity into a later LIVE state. Outcome reward is
attributed to the actual MD7 action, never to the hypothetical draw.

## Cold-start result

P0 was simulated on 98 historical pre-action contexts over 256 fixed seeds
(25,088 selection-only draws) with the real fresh prior (`lambda=1`,
exploration scale `0.20`). It was effectively uniform: MD7 agreement 0.2457 and
telling on non-telling MD7 contexts 0.2522.

P1 adds `gamma * log(p_MD7(a))` to each sampled reward score. It is minimal and
always overridable by a sufficiently large learned reward advantage, but a
fixed gamma does not decay automatically. The sensitivity grid is in
`cold_start_audit.md`.

P2 uses a coherent Gaussian coefficient-mean prior:
`A0=lambda*I`, `m0_a=kappa*e(selector_p_a)`, `b0_a=A0*m0_a`. It does not pretend
that a selector label is a reward observation, and its influence dilutes as
real design information accumulates. Its reward-scale interpretation and state
schema need an explicit scientific decision.

A selector-centered prior is necessary before LIVE if the desired cold start
is stable MD7 behavior. Neither P1 nor P2 has been selected or connected to the
production LIVE path. LIVE must remain disabled until that design is chosen and
verified offline.

## Prospective readiness

Run from the workspace root:

```powershell
python pedagogical-move-selection/results/turn_lints_shadow_collection_v2/shadow_data_readiness.py
```

The script prints raw action/resolution counts, L/Q availability, complete
S/K/L/H/Q rows, both actual and hypothetical arm counts, attempts/learners/
skills when linkable, and reward distributions. It deliberately has no
automatic pass threshold and performs no model fit.

At handoff, the new lineage has no collected user action yet, so it is ready to
collect but does not resolve the earlier context-data limitation.

## Verification completed

- 80 focused pytest cases plus 4 parameterized subtests passed, covering
  SHADOW actual/hypothetical isolation, zero updates, context and reward
  logging, L/Q timing, event identity/idempotency, lineage isolation, the
  selector bridge, and Tutor response/prompt contracts.
- Re-running the 25,088-draw cold-start analysis produced the identical summary
  SHA-256 `7334c69001797ee6ff0f1ed17fcd532ad02de03cc2e75177182cf2b5972883e7`.
- The real Tutor-path realization smoke audit passed 12/12 retained responses;
  exact external API attempts were 34. See
  `../tutor_move_realization_v1/runtime_smoke_report.md`.
- The active Tutor backend reports MD7-R2-TELL-C1 Epoch 2, SHADOW,
  S+K+L+H+Q, `headroom_normalized`, and the new lineage. The fresh state still
  has zero updates for every arm.

## Safety accounting

Training: 0. LIVE updates: 0. Protected MathDial test use: 0. MRBench V3 test
use: 0. Historical authoritative rewrites: 0.
