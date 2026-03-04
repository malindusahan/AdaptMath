# Raw MD7 probability-proportional warm-start audit

## Decision

**A — RAW MD7 PROPORTIONAL WARM-START IS DEFENSIBLE.**

The raw frozen MD7 vector provides non-deterministic support for all four moves
under the audited empirical context distribution. It preserves very strong MD7
preferences while exploring much more often where MD7 is uncertain. Very-low-
propensity treatments are possible but rare under the behavior policy. The
semantic review found no widespread implausible alternative-action mass.

This is a behavior-policy and data-collection decision, not a reward or causal
claim. The mode has been implemented in isolation but remains inactive.

## Source and artifact verification

- Frozen selector: `MD7-R2-TELL-C1 Epoch 2`.
- Checkpoint SHA-256:
  `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`.
- Class order: `generic, probing, focus, telling` (IDs 0–3).
- Inference contract: paired inputs `Problem:\n{problem}` and
  `Conversation:\n{formatted history}\n\nNext teacher pedagogical move:`, frozen
  fast tokenizer, left truncation, `truncation=only_second`, maximum length
  512, then four-class softmax.
- Audited source: 98 leakage-free pre-action problem/history snapshots from 15
  attempts in `md_self_improvement_turn_data_v1/turn_outcomes.jsonl`; source
  SHA-256 `4c984c0ead929c5dd0159abe262c51286cff13bb5c8d00d088120b4932410445`.
- Important lineage correction: 96 stored vectors in that file came from
  MD7-R1 and only two from MD7-R2. The analysis therefore reused only the
  leakage-free pre-action snapshots and recomputed all 98 probability vectors
  locally with the hash-verified frozen MD7-R2. It used no future outcomes.

The current Turn-LinTS defaults are still SHADOW, full instrumentation
`S+K+L+H+Q` (22 features), and `headroom_normalized`; the existing SHADOW
lineage remains separate. Action/reward linkage uses `attempt_id`,
`action_event_id`, `action_turn_index`, and `resolver_event_id`.

## Exact MD7 probability coverage

The exact expected action rate is the mean frozen probability across contexts.
The Monte Carlo check used 4,096 deterministic seed sweeps, totaling 401,408
selection-only draws.

| Move | Min | P10 | Q1 | Median | Mean/exact rate | Q3 | P90 | P95 | Max | MC rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generic | 0.000085 | 0.000172 | 0.004893 | 0.076537 | 0.244670 | 0.420063 | 0.795656 | 0.904229 | 0.987085 | 0.245411 |
| probing | 0.006738 | 0.036523 | 0.099431 | 0.166062 | 0.316653 | 0.456417 | 0.982321 | 0.997980 | 0.999225 | 0.316020 |
| focus | 0.000211 | 0.002633 | 0.053097 | 0.278685 | 0.337327 | 0.651768 | 0.742015 | 0.770816 | 0.823873 | 0.337452 |
| telling | 0.000317 | 0.002058 | 0.008266 | 0.029261 | 0.101350 | 0.095707 | 0.279695 | 0.424467 | 0.819198 | 0.101117 |

Balance is not required. These rates imply meaningful expected observations
for every arm, including telling, without forcing any move.

## Confidence, top-1 deviation, and entropy

The exact expected top-1 agreement is **71.329%** and deviation is **28.671%**.
Top-1 counts among contexts were generic 27, probing 24, focus 41, telling 6.
The top-1 probability had min/Q1/median/mean/Q3/P90/max of
`0.3364/0.5779/0.7199/0.7133/0.8231/0.9851/0.9992`. The top-to-second gap had
`0.0057/0.2537/0.5614/0.5276/0.6920/0.9739/0.9989` at the same summaries.

Top-1-confidence quartiles independently show that deviation falls as selector
confidence rises:

| Confidence quartile | Contexts | Top-1 probability range | Mean top-1 probability | Expected deviation | Mean gap | Mean entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|
| Q1, lowest | 25 | 0.3364-0.5776 | 0.4794 | 0.5206 | 0.1785 | 1.1408 |
| Q2 | 25 | 0.5788-0.7203 | 0.6560 | 0.3440 | 0.4048 | 0.8463 |
| Q3 | 24 | 0.7291-0.8239 | 0.7734 | 0.2266 | 0.6180 | 0.6986 |
| Q4, highest | 24 | 0.8376-0.9992 | 0.9565 | 0.0435 | 0.9286 | 0.1819 |

Gap-ranked quartiles show the same confidence/exploration relationship:

| Gap quartile | Contexts | Mean gap | Expected top-1 agreement | Expected deviation | Mean entropy (nats) |
|---|---:|---:|---:|---:|---:|
| Q1, closest | 25 | 0.1480 | 0.4901 | 0.5099 | 1.0411 |
| Q2 | 25 | 0.4287 | 0.6512 | 0.3488 | 0.9101 |
| Q3 | 24 | 0.6250 | 0.7671 | 0.2329 | 0.7359 |
| Q4, strongest | 24 | 0.9286 | 0.9565 | 0.0435 | 0.1819 |

Action entropy in nats had min/Q1/median/mean/Q3/max of
`0.0071/0.5540/0.7852/0.7225/1.0129/1.3472` (maximum possible for four arms is
`ln(4)=1.3863`). Representative low-, middle-, and high-entropy states and
competition cases are reviewed in `semantic_review_cases.md`.

## Low-probability and overlap diagnostics

Across all 392 context/action probabilities, proportions below the descriptive
bins were as follows. These are observations, not eligibility rules.

| Move | <.001 | <.005 | <.010 | <.025 | <.050 |
|---|---:|---:|---:|---:|---:|
| generic | 17.347% | 25.510% | 28.571% | 35.714% | 42.857% |
| probing | 0.000% | 0.000% | 2.041% | 7.143% | 14.286% |
| focus | 6.122% | 12.245% | 14.286% | 21.429% | 24.490% |
| telling | 6.122% | 20.408% | 25.510% | 45.918% | 61.224% |
| all context/action pairs | 7.398% | 14.541% | 17.602% | 27.551% | 35.714% |

Because low-probability arms are selected infrequently, the propensity of the
actually sampled action was much larger in typical draws. Its
min/P1/P5/P10/Q1/median were
`0.000091/0.028182/0.090073/0.151772/0.358392/0.683666`. Sampled actions had
propensity below `.01` in **0.179%**, below `.025` in **0.853%**, and below
`.05` in **2.025%** of draws.

Inverse propensity had min/P1/P5/P10/Q1/median/mean/Q3/P90/P95/P99/max of
`1.001/1.001/1.002/1.013/1.219/1.463/3.999/2.790/6.589/11.102/35.484/11004.339`.
Thus extreme weights are possible but occur in a very thin tail. Future
off-policy analysis must report overlap and weight diagnostics and should not
make claims from raw, unstabilized inverse-propensity estimates.

## Telling coverage

Telling probability had min/Q1/median/mean/Q3/P90/P95/max of
`0.000317/0.008266/0.029261/0.101350/0.095707/0.279695/0.424467/0.819198`.
The exact expected telling rate is **10.135%**. In 92 contexts where telling was
not top-1, its conditional expected rate was **6.930%** (an aggregate 6.506
percentage points of all assignments). In six telling-top-1 contexts, the
conditional expected telling rate was **59.273%**. This supplies natural telling
coverage without forcing telling.

## Practical-horizon planning

Each cell is `exact expected count; MC mean [P5, median, P95]` from 10,000
deterministic replicates using the empirical context distribution.

| N | generic | probing | focus | telling |
|---:|---:|---:|---:|---:|
| 50 | 12.23; 12.28 [7,12,17] | 15.83; 15.76 [11,16,21] | 16.87; 16.92 [11,17,22] | 5.07; 5.04 [2,5,9] |
| 100 | 24.47; 24.42 [17,24,32] | 31.67; 31.62 [24,32,40] | 33.73; 33.81 [26,34,42] | 10.14; 10.15 [5,10,15] |
| 250 | 61.17; 61.25 [50,61,73] | 79.16; 79.04 [67,79,91] | 84.33; 84.36 [72,84,97] | 25.34; 25.35 [18,25,33] |
| 500 | 122.33; 122.24 [107,122,138] | 158.33; 158.42 [141,158,176] | 168.66; 168.73 [152,169,186] | 50.68; 50.62 [40,51,62] |

These are planning quantities, not minimum sample-size requirements.

## Semantic and override review

The manual structural review found concentrated distributions in clear states
and plausible multi-move mass in ambiguous/recovering states. High non-top-1
telling cases generally reflected a reasonable choice between support, a
targeted cue, and an explicit next step. No post-hoc rule was introduced.

Normal MD7-R2/direct-Turn-LinTS does not use the legacy learner-agency wrapper;
that wrapper remains confined to rollback/manual modes. The legacy manual move
endpoint cannot mutate an active adaptive session. The new runtime contract also
supports explicit external-action provenance: an externally forced move is
`randomized_assignment=false`, has an external source, and has no MD7 sampling
propensity.

## Inactive runtime support

Because the decision is A, an isolated `RANDOMIZED_WARMSTART` mode was added.
Its behavior policy is `md7_r2_probability_proportional_v1`; lineage schema is
`adaptmath_randomized_warmstart_lineage_v1`; event schema is
`adaptmath_turn_lints_event_v4`.

The mode:

1. builds the same leakage-free pre-action `S+K+L+H+Q` context;
2. samples the actual move from the full frozen MD7 vector using system entropy;
3. passes that sampled move through the authoritative Tutor realization path;
4. logs the selected propensity and full vector;
5. records Tutor response, current MRB1 auxiliary scores, resolver linkage,
   mastery values, raw delta, headroom-normalized delta, and status; and
6. never updates the Turn-LinTS posterior.

Current MRB1 is not treatment input or scalar reward; previous-turn MRB1 remains
only the candidate Q block for the next action. Formal assessment remains
outside per-turn assignment and reward.

Prepared lineage:
`adaptive-math-tutor/backend/runtime/turn_lints_randomized_warmstart_v1/`.
It has a fresh state with zero updates and no event log. The active/default mode
remains SHADOW.

## Focused verification

The focused offline suite passed **29/29** tests:

- `test_randomized_warmstart.py`
- `test_turn_lints_v1.py`
- `test_turn_lints_selector_anchor.py`

The contracts cover exact categorical interval assignment, deterministic RNG
injection, availability of all four arms without tau or filtering, exact
selected-arm propensity and full-vector logging, an actual sampled Tutor move
that differs from MD7/SHADOW top-1, zero warm-start posterior updates, valid
HEADROOM_NORMALIZED reward and action linkage, censored responses, external
override provenance, leakage-free context timing, previous-only MRB1 context,
and the SHADOW default. Python compilation also passed. No external Tutor API
test was run.

## Safety accounting

- training: 0
- randomized real treatments: 0
- LIVE updates: 0
- external Tutor API calls: 0
- protected MathDial test use: 0
- MRBench V3 test use: 0
- authoritative historical rewrites: 0
