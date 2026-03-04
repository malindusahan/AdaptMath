# Turn-LinTS frozen selector action-anchor decision

## Decision

**B. P1 TOO STRONG / POORLY CALIBRATED.**

P1 is the primary v1 formulation, but the gamma that best reproduces the full
frozen MD7 distribution creates persistent override barriers that are large
relative to the observed HEADROOM_NORMALIZED reward scale. P1 is therefore
implemented as isolated, fail-closed support but is not activated in SHADOW or
LIVE. Turn-LinTS remains on anchor mode `none`.

This is a policy-initialization result. It makes no reward-effectiveness or
causal claim.

## Formulation and scientific separation

P1 is a **frozen selector action anchor**:

```text
policy_score_a(x)
  = contextual_TS_reward_score_a(x)
  + gamma * log(max(P_MD7(a | x), 1e-12))
```

The first term comes only from the Turn-LinTS BKT-reward posterior. The second
comes only from frozen MD7-R2-TELL-C1. MD7 probabilities are not rewards and
were not inserted into A or b.

The anchor receives the current MD7 vector through a separate policy argument.
Context S may contain the same vector as candidate reward-model features, but S
can be disabled without disabling the anchor. No coefficient prior, tau gate,
eligibility rule, arm block, or decay schedule was introduced.

P2 remains mathematically coherent as a Gaussian coefficient-mean prior, but it
requires stronger assumptions about reward scale, coefficient meaning, and the
mapping from selector probability to expected BKT reward. P1 leaves the learned
reward posterior untouched and is therefore the clearer primary v1 candidate.

## Grouped distribution-matching calibration

The input was 98 leakage-free pre-action contexts from 15 attempts. Attempts
were ranked by a fixed salted SHA-256 key and split as groups:

- Calibration: 10 attempts, 73 contexts.
- Held-out validation: 5 attempts, 25 contexts.
- Attempt overlap: 0.

No current/future learner response, current MRB1 score, or future BKT outcome
entered action selection. Past quantities were used only where they already
formed the pre-action context. Reward values were not used in gamma selection.

For each context, 512 deterministic fresh-posterior samples estimated
`q_gamma(a|x)`. The search contained 41 values from 0.00 through 0.80 in steps
of 0.02. Gamma was selected only by minimum calibration mean per-context
Jensen-Shannon divergence in natural-log units.

Selected calibration gamma: **0.66**.

Calibration mean JS was 0.010010 nats and median JS was 0.006227. Neighboring
values were close (0.64: 0.010100; 0.68: 0.010069), so the objective is smooth
near its interior minimum rather than driven by an edge of the search.

## Held-out selector-distribution matching

The exact selected gamma was evaluated once on the held-out attempts.

| Metric | Held-out result |
|---|---:|
| mean JS divergence | 0.010996 nats |
| median JS divergence | 0.007083 nats |
| JS Q1 / Q3 | 0.003497 / 0.013182 |
| JS P90 / P95 | 0.026697 / 0.031832 |
| JS maximum | 0.049632 |
| expected MD7 top-1 agreement | 0.745859 |
| telling when MD7 top-1 was not telling | 0.041719 |

| Move | Frozen MD7 mean probability | P1 empirical probability |
|---|---:|---:|
| generic | 0.347050 | 0.355469 |
| probing | 0.412522 | 0.416484 |
| focus | 0.206493 | 0.186328 |
| telling | 0.033935 | 0.041719 |

Top-1 agreement and telling are diagnostics only; neither selected gamma. The
small held-out JS confirms that gamma 0.66 reproduces the full selector
distribution reasonably well, including MD7 uncertainty.

## Persistent override barriers

For an MD7-preferred arm a over alternative b, the learned reward-score
advantage needed to reverse the deterministic anchor ordering is:

```text
B(a,b) = 0.66 * [log P_MD7(a|x) - log P_MD7(b|x)]
```

The following uses all 98 pre-action contexts after gamma selection. Reward SD
is 0.2075 and reward IQR is 0.2863; neither scale was used to tune gamma.

| Barrier family | N | Q1 | Median | Q3 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top-1 to second-best | 98 | 0.406 | 0.734 | 1.874 | 3.209 | 4.375 | 4.951 |
| top-1 to each alternative | 294 | 0.831 | 1.669 | 2.960 | 4.319 | 5.032 | 5.687 |
| non-telling top-1 to telling | 98 | 1.667 | 2.248 | 3.284 | 4.188 | 4.738 | 5.127 |
| telling top-1 to best non-telling | 0 | NA | NA | NA | NA | NA | NA |

No historical MD7 top-1 context was telling, so the final requested direction
cannot be estimated from these contexts.

The median top-1/runner barrier is 3.54 reward SDs or 2.56 reward IQRs. Its 95th
percentile is 21.08 SDs. The median non-telling-to-telling barrier is 10.83 SDs
or 7.85 IQRs. For comparison, the observed retrospective reward range was
-0.2622 to 0.7005.

Very uncertain MD7 contexts remain overridable (the smallest top/runner barrier
was 0.0031), but typical and strong-preference contexts do not present a
credible adaptation scale. Linear scores are not formally bounded, so override
is mathematically possible; the empirical evidence says it would often require
implausibly large reward-score differences. This is why good distribution
matching alone does not make P1 ready.

## Operational disposition

- P1 support is opt-in and fail closed on anchor mode, gamma, selector version,
  selector SHA-256, or numerical-floor mismatch.
- Probability floor: `1e-12`, used only inside log for numerical safety.
- All four arms are always sampled and scored.
- Current SHADOW configuration remains `anchor_mode=none`, gamma 0.
- The existing zero-update v1 state is load-compatible only with anchor mode
  none; implicit migration to an anchored state is refused.
- No P1 runtime event was manufactured. When explicitly enabled in a new
  isolated state, event schema v3 can log `actual_treatment_move`,
  `shadow_p0_move`, and `shadow_p1_move`, with both shadow moves marked as
  non-treatments and never assigned the observed reward.

Context remains **INCONCLUSIVE**. Full S+K+L+H+Q SHADOW instrumentation and Q
retention continue unchanged. The next analysis is frozen in
`context_ablation_preregistration.md` and must wait for prospective complete
data.

## Safety

Training: 0. LIVE updates: 0. External Tutor API calls: 0. Protected test use:
0. Historical authoritative rewrites: 0.

