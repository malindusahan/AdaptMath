# Turn-LinTS reward and context audit v1

## Scope and safeguards

Read-only analysis of existing real passive turn records. BKT belief change is a transformed model-state update, not measured or causal learning. MRB1 is a noisy auxiliary critic, not ground truth. No training, external APIs, protected tests, historical rewrites, or policy updates were used.

## Selector and SHADOW integrity

- Selector integrity: PASS; MD7-R2-TELL-C1 Epoch 2, SHA-256 `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`, labels `generic, probing, focus, telling`.
- MD7-R1 rollback remains available at SHA-256 `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13`.
- Recorded local runtime: selector `ordinary-md7-r2-tell-c1-v1`; Turn-LinTS `SHADOW`.
- Real SHADOW events: 2; policy updates: 0; linked passive/resolver events: 2/2.
- Sequential SHADOW context linkage: PASS; turn t+1 carries only the prior turn's BKT delta, learner signals, move, and MRB1 values.
- Limitation: Only 2 real SHADOW events from one attempt are available; source/tests provide the broader lifecycle evidence.

## Usable rewards

Usable linked turn-level BKT observations: **96** from 14 attempts, 3 learners, and 12 skills. Two censored turns have no fabricated zero reward.

| reward | count | min | Q1 | median | mean | Q3 | max | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R_RAW | 96 | -0.1711 | -0.0278 | 0.0227 | 0.0235 | 0.0654 | 0.5565 | 0.0924 |
| R_HEADROOM | 96 | -0.2622 | -0.0457 | 0.1674 | 0.1358 | 0.2406 | 0.7005 | 0.2075 |

Spearman correlations of absolute reward magnitude:

| relationship | R_RAW rho | R_HEADROOM rho |
|---|---:|---:|
| mastery_before | -0.7035 | -0.0258 |
| turn_index | 0.0391 | 0.0110 |

Raw magnitude has a strong negative mastery association (rho -0.7035), while headroom magnitude is nearly unrelated to mastery (rho -0.0258). Headroom values remain finite and bounded in the observed range [-0.2622, 0.7005], although their standard deviation is larger (0.2075 versus 0.0924). Neither reward has a material turn-index association. Thus the recommendation considers stability, range, position, and grouped predictability as well as the correlation change.

Mastery-band behavior:

| band | reward | n | signed mean | absolute mean |
|---|---|---:|---:|---:|
| [0,.2) | r_raw | 1 | 0.5565 | 0.5565 |
| [.2,.4) | r_raw | 3 | 0.1382 | 0.1382 |
| [.4,.6) | r_raw | 16 | 0.0512 | 0.1013 |
| [.6,.8) | r_raw | 41 | 0.0090 | 0.0640 |
| [.8,1] | r_raw | 35 | 0.0027 | 0.0311 |
| [0,.2) | r_headroom | 1 | 0.5627 | 0.5627 |
| [.2,.4) | r_headroom | 3 | 0.1956 | 0.1956 |
| [.4,.6) | r_headroom | 16 | 0.1125 | 0.2122 |
| [.6,.8) | r_headroom | 41 | 0.0832 | 0.1631 |
| [.8,1] | r_headroom | 35 | 0.1908 | 0.2226 |

Turn-position summaries are in `reward_by_turn_position.csv`; move, learner, and skill breakdowns are in `summary.json`.

## Historical action coverage

| move | usable turns |
|---|---:|
| generic | 28 |
| probing | 33 |
| focus | 35 |
| telling | 0 |

Telling has no usable historical observations, so no retrospective analysis validates that direct arm.

## Student Modeling and MRB1

The implemented learner-state outputs are exactly `previous_reasoning_probability`, `previous_uncertainty_probability`, and `previous_clarification_probability`. Complete retrospective pre-action values exist for only 1 of 96 usable rows, so L-containing ablations are unavailable rather than imputed.
All four MRB1 heads are complete on 96 of 96 usable rows. Stored scores are continuous expected values using `No=0`, `To some extent=0.5`, and `Yes=1`. Current MRB1 is treated only as post-action outcome; Q uses the previous response's scores.
Descriptive disagreement counts: high-quality/negative raw reward = 16; low-quality/positive raw reward = 31. These are associations, not causal effects.

| MRB1 head | mean | std | Q1 | median | Q3 |
|---|---:|---:|---:|---:|---:|
| Mistake_Identification | 0.6644 | 0.2636 | 0.4278 | 0.7483 | 0.9078 |
| Mistake_Location | 0.5695 | 0.2789 | 0.3226 | 0.6897 | 0.8144 |
| Providing_Guidance | 0.6430 | 0.2627 | 0.5083 | 0.7389 | 0.8511 |
| Actionability | 0.7511 | 0.2844 | 0.7513 | 0.8987 | 0.9341 |

The four MRB1 heads are strongly intercorrelated (pairwise Spearman rho 0.749 to 0.966), but each head has near-zero current-reward and next-reward association (absolute rho at most 0.098 in these data). MRB1 therefore describes a signal distinct from BKT belief change, but adding previous MRB1 (D1 versus D0) worsens grouped MAE/RMSE for both rewards; current data do not support Q as incrementally predictive.

## Temporal leakage audit

- S: selector probabilities are computed from the problem/history before action t.
- K: mastery is snapshotted before action t; previous delta is resolved after t-1 and carried forward.
- L: learner-signal probabilities come from the resolved learner response at t-1, never the current response at t.
- H: previous move and prior Tutor-turn count exist before selection.
- Q: MRB1_t is scored after Tutor response t and can enter only action t+1 as previous MRB1.

All implemented fields pass the source-order audit. Features with absent retrospective values are rejected from ablation rather than filled from current/future information.

## Grouped context ablation

Primary validation is deterministic 5-fold GroupKFold by attempt. Each available model is ridge-regularized linear prediction with direct action-by-context interactions. Turns are never randomly split.

| config | blocks | reward | status | n | MAE mean (sd) | RMSE mean (sd) | R2 mean (sd) |
|---|---|---|---|---:|---:|---:|---:|
| C0 | S | r_raw | available | 96 | 0.0747 (0.0141) | 0.1031 (0.0201) | -1.5281 (2.9958) |
| C0 | S | r_headroom | available | 96 | 0.1942 (0.0377) | 0.2365 (0.0425) | -1.0965 (1.5707) |
| C1 | S+K | r_raw | available | 96 | 0.0719 (0.0134) | 0.0962 (0.0188) | -1.0544 (2.1721) |
| C1 | S+K | r_headroom | available | 96 | 0.1997 (0.0340) | 0.2402 (0.0376) | -1.5480 (2.7194) |
| C2 | S+L | r_raw | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C2 | S+L | r_headroom | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C3 | S+K+L | r_raw | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C3 | S+K+L | r_headroom | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C4 | S+K+L+H | r_raw | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C4 | S+K+L+H | r_headroom | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C5 | S+K+L+H+Q | r_raw | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C5 | S+K+L+H+Q | r_headroom | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C6 | K+L+H+Q | r_raw | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| C6 | K+L+H+Q | r_headroom | unavailable | 96 | NA (NA) | NA (NA) | NA (NA) |
| D0 | S+K+H | r_raw | available | 96 | 0.0696 (0.0194) | 0.0953 (0.0255) | -0.5779 (0.8672) |
| D0 | S+K+H | r_headroom | available | 96 | 0.2023 (0.0515) | 0.2501 (0.0608) | -1.1555 (1.2370) |
| D1 | S+K+H+Q | r_raw | available | 96 | 0.0751 (0.0285) | 0.1046 (0.0333) | -1.0506 (1.5015) |
| D1 | S+K+H+Q | r_headroom | available | 96 | 0.2320 (0.0734) | 0.2866 (0.0843) | -2.2976 (3.0471) |

D0/D1 are explicitly diagnostic fallbacks for testing H and Q when L is unavailable; they are not substitutions for the requested C4/C5 comparison. Fold-level metrics and unavailable reasons are in `context_ablation_results.csv`.

## Decisions

**REWARD: HEADROOM_NORMALIZED**

**CONTEXT: INCONCLUSIVE**

**TURN-LINTS READINESS: C. CONTEXT NEEDS MORE EVIDENCE**

HEADROOM_NORMALIZED removes the observed mechanical mastery dependence without non-finite values, boundary explosions, clipping, or added turn-index dependence; its wider but bounded scale and similar normalized predictive difficulty were considered explicitly. Context remains inconclusive because complete L values exist on only 1 usable row(s), telling has zero historical observations, and every available grouped model has unstable negative mean R2. More causally complete, action-covered attempts are required.

Turn-LinTS remains SHADOW. This report does not authorize LIVE.
