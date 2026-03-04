# First Multi-Attempt Analysis of Real Self-Improvement Data

Analysis version: `md_self_improvement_pilot_analysis_v1`  
Scope: read-only descriptive analysis; no training rule, reward, threshold, preference pair, or accepted-label decision is created.

## 1. Dataset inventory

- Real completed attempts: **6** (Attempt 1 is inferred completed from its pre-fix assessment/summary lifecycle; its missing recorded status is preserved.)
- Aborted attempts: **0**
- Dialogue turns: **57**
- Formal-assessment observations: **18**
- Pending actions / processing failures: **0 / 0**

| attempt_number | attempt_id | learner_pseudonym | skill | problem_id | problem_summary | attempt_started_at | completion_status | dialogue_turn_count | assessment_item_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | problem_c62f1095a393 | A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle. |  | completed_legacy_inferred | 15 | 3 |
| 2 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | problem_852ce183aa20 | A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m². Find the width and length of the g… | 2026-08-28T02:20:06.567784Z | completed | 8 | 3 |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | psn_f78c0817d43053e5bd31da20 | Complementary and Supplementary Angles | problem_7088565b00dc | Two angles are supplementary. One angle is 18° more than twice the other angle. Find the measure of both angles. | 2026-08-28T03:57:58.362166Z | completed | 9 | 3 |
| 4 | f3133173-8f14-44d9-b485-97a42dc45030:1 | psn_f78c0817d43053e5bd31da20 | Conversion of Fraction Decimals Percents | problem_93260778f367 | A school survey shows that \(\frac{3}{5}\) of the students prefer studying with digital tools. Convert \(\frac{3}{5}\) into: A decimal A pe… | 2026-08-28T04:10:58.458614Z | completed | 10 | 3 |
| 5 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | psn_f78c0817d43053e5bd31da20 | Multiplication Whole Numbers | problem_b8a2b1eb2d2a | A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks. How many notebooks are there altogether? | 2026-08-28T04:27:57.396329Z | completed | 7 | 3 |
| 6 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | psn_f78c0817d43053e5bd31da20 | Simplifying Expressions positive exponents | problem_388ba721ea09 | Simplify: $$ 3x^2 \cdot 4x^3 $$ Write your final answer using a single exponent for \(x\). | 2026-08-28T04:45:22.258527Z | completed | 8 | 3 |

## 2. Integrity / privacy / provenance

| check | mismatch_count |
| --- | --- |
| action_mastery_mismatches | 0 |
| assessment_chain_mismatches | 0 |
| assessment_separation_mismatches | 0 |
| conflicting_action_id_groups | 0 |
| conflicting_assessment_event_groups | 0 |
| conflicting_attempt_summary_groups | 0 |
| conflicting_resolver_id_groups | 0 |
| delta_equation_mismatches | 0 |
| duplicate_action_id_extra_records | 0 |
| duplicate_action_id_groups | 0 |
| duplicate_assessment_event_extra_records | 0 |
| duplicate_assessment_event_groups | 0 |
| duplicate_attempt_summary_extra_records | 0 |
| duplicate_attempt_summary_groups | 0 |
| duplicate_resolver_id_extra_records | 0 |
| duplicate_resolver_id_groups | 0 |
| exact_duplicate_assessment_records | 0 |
| exact_duplicate_summary_records | 0 |
| exact_duplicate_turn_records | 0 |
| index_contiguous_mismatches | 0 |
| index_monotonic_mismatches | 0 |
| index_starts_at_1_mismatches | 0 |
| mastery_continuity_mismatches | 0 |
| pending_actions | 0 |
| pending_processing_failures | 0 |
| resolver_identity_mismatches | 0 |
| text_history_mismatches | 0 |
| timestamp_order_mismatches | 0 |

All integrity mismatch and duplicate counts are zero. Formal assessment remains separate: every assessment `source_action_event_id` is null and its mastery chain begins after the final dialogue observation.

Post-fix scope: **5 attempts / 52 records**. Privacy pattern counts are `{"Lenovo": 0, "absolute_C_Users_paths": 0, "account_id": 0, "api_key": 0, "auth_token": 0, "email_address": 0, "email_word": 0, "secret": 0, "username": 0}`; rooted path fields: **0**; invalid pseudonyms: **0**.

Non-null runtime provenance has **0 conflicting fields**. The consistent deployment is: real data, ordinary-md7r1-v1, MD7 checkpoint md7r1_epoch3 / hash `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13`, move order G/P/F/T, learner agency `learner_agency_telling_escalation_v1`, C3 `turn_lints_v3_c3_9d`, LinTS `true_disjoint_lints_v3` / `MD7-R1 fresh LinTS v1`, tau `.10`, MRB1 `frozen_mrb1` / hash `9ad757e884beae6a242cf6fe17d5b3524e7599e413e9b9cc5f3b999a37a6f1d7`, resolver `2.0`, and BKT `confidence_weighted_bkt_v1` / hash `a82d8f066c965d931882ae3251db50b42f8033defb89da3e822b7b23a3b56e75`.

Documented legacy limitation: Attempt 1 omits post-fix lifecycle/move-order/portable-policy fields, and the older deployment `lineage_manifest.json` still stores absolute local paths plus its creation-time `total_updates=15`. Neither artifact was rewritten. Current authoritative policy state reports `57` updates.

## 3. Attempt table

Move count columns use `G/P/F/T`; evaluator columns are explicit counts. No causal interpretation is attached.

| attempt_number | attempt_id | learner_pseudonym | skill | dialogue_turn_count | mastery_start | mastery_pre_assessment | mastery_final | dialogue_net_delta | assessment_net_delta | total_delta | raw_G_P_F_T | final_G_P_F_T | learner_agency_triggers | lints_nonbaseline_selections | actual_lints_overrides | evaluator_correct | evaluator_partial | evaluator_incorrect | evaluator_unknown | mean_MRB1_MI | mean_MRB1_ML | mean_MRB1_PG | mean_MRB1_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 15 | 0.641663 | 0.805371 | 0.974046 | 0.163709 | 0.168674 | 0.332383 | 4/1/10/0 | 4/1/10/0 | 0 | 0 | 0 | 8 | 0 | 2 | 5 | 0.505205 | 0.412756 | 0.520732 | 0.647364 |
| 2 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 8 | 0.974046 | 0.965872 | 0.996467 | -0.008173 | 0.030595 | 0.022422 | 5/2/1/0 | 5/2/1/0 | 0 | 0 | 0 | 4 | 0 | 1 | 3 | 0.614412 | 0.518958 | 0.632381 | 0.802412 |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | psn_f78c0817d43053e5bd31da20 | Complementary and Supplementary Angles | 9 | 0.843622 | 0.746336 | 0.998109 | -0.097286 | 0.251772 | 0.154487 | 1/6/2/0 | 2/6/1/0 | 0 | 1 | 1 | 2 | 1 | 1 | 5 | 0.793426 | 0.694115 | 0.724265 | 0.829741 |
| 4 | f3133173-8f14-44d9-b485-97a42dc45030:1 | psn_f78c0817d43053e5bd31da20 | Conversion of Fraction Decimals Percents | 10 | 0.616757 | 0.723806 | 0.944455 | 0.107049 | 0.220649 | 0.327698 | 1/4/5/0 | 1/4/5/0 | 0 | 0 | 0 | 6 | 1 | 2 | 1 | 0.721080 | 0.613636 | 0.717337 | 0.806754 |
| 5 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | psn_f78c0817d43053e5bd31da20 | Multiplication Whole Numbers | 7 | 0.660628 | 0.671784 | 0.995153 | 0.011157 | 0.323368 | 0.334525 | 1/6/0/0 | 1/6/0/0 | 0 | 0 | 0 | 2 | 0 | 3 | 2 | 0.694543 | 0.658447 | 0.677082 | 0.814770 |
| 6 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | psn_f78c0817d43053e5bd31da20 | Simplifying Expressions positive exponents | 8 | 0.011042 | 0.995945 | 1.000000 | 0.984904 | 0.004055 | 0.988958 | 1/1/6/0 | 1/1/6/0 | 0 | 0 | 0 | 5 | 0 | 2 | 1 | 0.850676 | 0.746134 | 0.790035 | 0.876942 |

## 4. Starting mastery / headroom

Headroom is descriptive only: `1 - mastery_before`.

| variable | count | min | q1 | median | mean | q3 | max | std_population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mastery_before | 57 | 0.011042 | 0.592673 | 0.687696 | 0.702220 | 0.843622 | 0.989414 | 0.197578 |
| headroom | 57 | 0.010586 | 0.156378 | 0.312304 | 0.297780 | 0.407327 | 0.988958 | 0.197578 |

| mastery_band | turn_count | mean_raw_delta | median_raw_delta | mean_absolute_delta | median_absolute_delta | positive | zero | negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [0,.25) | 1 | 0.556531 | 0.556531 | 0.556531 | 0.556531 | 1 | 0 | 0 |
| [.25,.50) | 6 | 0.022969 | 0.019003 | 0.096123 | 0.088248 | 3 | 0 | 3 |
| [.50,.75) | 29 | 0.024052 | 0.055534 | 0.082692 | 0.071161 | 17 | 0 | 12 |
| [.75,.90) | 8 | -0.039060 | -0.044160 | 0.074931 | 0.055198 | 2 | 0 | 6 |
| [.90,.97) | 5 | 0.020144 | 0.012955 | 0.024341 | 0.012955 | 4 | 0 | 1 |
| [.97,1] | 8 | -0.002343 | -0.002758 | 0.007579 | 0.007464 | 3 | 0 | 5 |

The band table is descriptive. It is not a training threshold or reward transformation.

## 5. Turn-level BKT

Statuses: `{"censored_no_response": 0, "observed_update": 57, "processing_failed": 0, "resolved_no_update": 0}`. Delta signs: `{"negative": 27, "positive": 30, "zero": 0}`.

| variable | count | min | q1 | median | mean | q3 | max | std_population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta_mastery | 57 | -0.154909 | -0.040419 | 0.007025 | 0.020375 | 0.071161 | 0.556531 | 0.109391 |
| absolute_delta_mastery | 57 | 0.001371 | 0.027105 | 0.059972 | 0.075669 | 0.095028 | 0.556531 | 0.081582 |

Every delta is interpreted only as a **confidence-weighted BKT posterior belief update after resolved learner evidence**. Standard deviations in this report are population standard deviations (`ddof=0`); quartiles use linear interpolation.

## 6. Evaluator / resolver

| correctness | count | percent |
| --- | --- | --- |
| correct | 27 | 47.368421 |
| partial | 2 | 3.508772 |
| incorrect | 11 | 19.298246 |
| unknown | 17 | 29.824561 |
| <missing> | 0 | 0.000000 |

Evaluator correctness × delta sign (counts and within-row percentages):

| correctness | delta_sign | count | row_percent |
| --- | --- | --- | --- |
| correct | positive | 27 | 100.000000 |
| correct | zero | 0 | 0.000000 |
| correct | negative | 0 | 0.000000 |
| partial | positive | 2 | 100.000000 |
| partial | zero | 0 | 0.000000 |
| partial | negative | 0 | 0.000000 |
| incorrect | positive | 1 | 9.090909 |
| incorrect | zero | 0 | 0.000000 |
| incorrect | negative | 10 | 90.909091 |
| unknown | positive | 0 | 0.000000 |
| unknown | zero | 0 | 0.000000 |
| unknown | negative | 17 | 100.000000 |
| <missing> | positive | 0 | 0.000000 |
| <missing> | zero | 0 | 0.000000 |
| <missing> | negative | 0 | 0.000000 |

Resolver-field frequencies:

| field | value | count |
| --- | --- | --- |
| primary_signal | behavioural_difficulty | 17 |
| primary_signal | correct_explanation | 14 |
| primary_signal | correct_answer | 13 |
| primary_signal | incorrect_answer | 11 |
| primary_signal | partial_correct | 2 |
| should_update | true | 57 |
| observation_source | evaluator | 40 |
| observation_source | behavioural_proxy | 17 |
| reasoning_present | <missing> | 57 |
| uncertainty_present | <missing> | 57 |
| clarification_present | <missing> | 57 |
| repeated_misunderstanding | <missing> | 57 |

The four requested behavioural-presence fields are not stored in the turn-outcome schema; they are reported as missing rather than inferred from contributor names.

## 7. Raw / effective / final move distributions

| move | raw_count | raw_percent | effective_count | effective_percent | final_count | final_percent |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 13 | 22.807018 | 13 | 22.807018 | 14 | 24.561404 |
| probing | 20 | 35.087719 | 20 | 35.087719 | 20 | 35.087719 |
| focus | 24 | 42.105263 | 24 | 42.105263 | 23 | 40.350877 |
| telling | 0 | 0.000000 | 0 | 0.000000 | 0 | 0.000000 |

Transition counts are in `move_transitions.csv`. Learner-agency triggers: **0 / 57 (0.000%)**.

Agency-triggered turn details:

_No rows._

## 8. LinTS authority

- Total turns: **57**
- Baseline-only eligible: **54**
- At least one nonbaseline eligible: **3**
- Eligible-arm count mean / median / max: **1.070175 / 1.000000 / 3**
- Selected arms: `{"baseline": 56, "focus_bias": 0, "generic_bias": 1, "probing_bias": 0, "telling_bias": 0}`
- Actual overrides: **1 (1.754%)**
- Current policy updates: `{"baseline": 56, "focus_bias": 0, "generic_bias": 1, "probing_bias": 0, "telling_bias": 0}`; total **57**
- Experience log: **6 attempts / 57 turns**

Actual override details (outcomes are subsequent observations, not causal effects):

| attempt_number | attempt_id | action_turn_index | skill | latest_student_text | raw_probabilities | effective_probabilities | eligible_arms | selected_arm | base_move | final_move | mastery_before | correctness | delta_mastery | MI | ML | PG | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | 8 | Complementary and Supplementary Angles | i dont undersand | {"focus":0.3232975900173187,"generic":0.22414202988147736,"probing":0.31149759888648987,"telling":0.14106281101703644} | {"focus":0.3232975900173187,"generic":0.22414202988147736,"probing":0.31149759888648987,"telling":0.14106281101703644} | ["baseline","generic_bias","probing_bias"] | generic_bias | focus | generic | 0.642868 | unknown | -0.073632 | 0.686155 | 0.561044 | 0.730183 | 0.916986 |

## 9. MD7 confidence / gate

| count | min | q1 | median | mean | q3 | max | std_population |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 57 | 0.011800 | 0.247797 | 0.393084 | 0.507736 | 0.839537 | 0.997960 | 0.323956 |

| gap_at_or_below | count | percent |
| --- | --- | --- |
| 0.020000 | 2 | 3.508772 |
| 0.050000 | 2 | 3.508772 |
| 0.100000 | 3 | 5.263158 |
| 0.150000 | 6 | 10.526316 |
| 0.200000 | 10 | 17.543860 |

At tau `.10`, an alternative was empirically eligible on **3/57 turns (5.263%)**. This describes deployed LinTS authority and does not recommend a new tau.

## 10. Move × outcome observational tables

**OBSERVATIONAL ASSOCIATIONS ONLY.** Actions were not randomized; moves are not ranked.

| final_move | turn_count | mean_mastery_before | median_mastery_before | mean_delta | median_delta | positive | zero | negative | correct | partial | incorrect | unknown | mean_MI | mean_ML | mean_PG | mean_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic | 14 | 0.757956 | 0.804055 | 0.037417 | -0.008350 | 6 | 0 | 8 | 6 | 0 | 0 | 8 | 0.447157 | 0.318296 | 0.408751 | 0.568578 |
| probing | 20 | 0.673883 | 0.693870 | -0.010229 | -0.031493 | 8 | 0 | 12 | 5 | 2 | 8 | 5 | 0.846300 | 0.791255 | 0.810181 | 0.919365 |
| focus | 23 | 0.692935 | 0.640071 | 0.036613 | 0.059972 | 16 | 0 | 7 | 16 | 0 | 3 | 4 | 0.666349 | 0.566233 | 0.682413 | 0.784195 |
| telling | 0 |  |  |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |

Final move × evaluator correctness:

| final_move | correct | partial | incorrect | unknown | <missing> |
| --- | --- | --- | --- | --- | --- |
| generic | 6 | 0 | 0 | 8 | 0 |
| probing | 5 | 2 | 8 | 5 | 0 |
| focus | 16 | 0 | 3 | 4 | 0 |
| telling | 0 | 0 | 0 | 0 | 0 |

Final move × delta sign:

| final_move | positive | zero | negative |
| --- | --- | --- | --- |
| generic | 6 | 0 | 8 |
| probing | 8 | 0 | 12 |
| focus | 16 | 0 | 7 |
| telling | 0 | 0 | 0 |

Decision-modification groups:

| decision_group | turn_count | mastery_before_mean | mastery_before_median | delta_mean | delta_median | positive | zero | negative | correct | partial | incorrect | unknown | mean_MI | mean_ML | mean_PG | mean_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_unchanged | 56 | 0.703280 | 0.691573 | 0.022053 | 0.007464 | 30 | 0 | 26 | 27 | 2 | 11 | 16 | 0.675465 | 0.584706 | 0.658776 | 0.776195 |
| B_agency_only | 0 |  |  |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| C_lints_only | 1 | 0.642868 | 0.642868 | -0.073632 | -0.073632 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0.686155 | 0.561044 | 0.730183 | 0.916986 |
| D_both | 0 |  |  |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |

## 11. Skill coverage

Skills are kept separate; differences are not interpreted as equal-difficulty comparisons. Diagnostic flags are descriptive only: at least one start ≥.90, fewer than 10 turns, and near saturation based on ≥80% of turns at mastery ≥.97 or median final mastery ≥.99.

| skill | attempt_count | turn_count | start_mastery_min | start_mastery_median | start_mastery_max | turn_mastery_before_min | turn_mastery_before_median | turn_mastery_before_mean | turn_mastery_before_max | raw_G_P_F_T | final_G_P_F_T | correct | partial | incorrect | unknown | positive | zero | negative | dialogue_net_mastery_change | assessment_net_mastery_change | fraction_turns_mastery_ge_0_97 | descriptive_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Complementary and Supplementary Angles | 1 | 9 | 0.843622 | 0.843622 | 0.843622 | 0.569236 | 0.739781 | 0.738620 | 0.868796 | 1/6/2/0 | 2/6/1/0 | 2 | 1 | 1 | 5 | 3 | 0 | 6 | -0.097286 | 0.251772 | 0.000000 | low_turn_count_<10;near_complete_saturation |
| Conversion of Fraction Decimals Percents | 1 | 10 | 0.616757 | 0.616757 | 0.616757 | 0.436005 | 0.622409 | 0.619418 | 0.761219 | 1/4/5/0 | 1/4/5/0 | 6 | 1 | 2 | 1 | 7 | 0 | 3 | 0.107049 | 0.220649 | 0.000000 | none |
| Equation Solving Two or Fewer Steps | 2 | 23 | 0.641663 | 0.807854 | 0.974046 | 0.432256 | 0.744246 | 0.763096 | 0.981070 | 9/3/11/0 | 9/3/11/0 | 12 | 0 | 3 | 8 | 12 | 0 | 11 | 0.155535 | 0.199269 | 0.217391 | very_high_starting_mastery_present |
| Multiplication Whole Numbers | 1 | 7 | 0.660628 | 0.660628 | 0.660628 | 0.320690 | 0.501115 | 0.505262 | 0.660628 | 1/6/0/0 | 1/6/0/0 | 2 | 0 | 3 | 2 | 2 | 0 | 5 | 0.011157 | 0.323368 | 0.000000 | low_turn_count_<10;near_complete_saturation |
| Simplifying Expressions positive exponents | 1 | 8 | 0.011042 | 0.011042 | 0.011042 | 0.011042 | 0.936721 | 0.762094 | 0.989414 | 1/1/6/0 | 1/1/6/0 | 5 | 0 | 2 | 1 | 6 | 0 | 2 | 0.984904 | 0.004055 | 0.375000 | low_turn_count_<10;near_complete_saturation |

## 12. Learner dependence

Unique learner pseudonyms: **1** — `psn_f78c0817d43053e5bd31da20`.

**These attempts are longitudinal observations from one learner and are not six independent learners.**

Repeated learner-skill combinations:

| learner_pseudonym | skill | attempt_count |
| --- | --- | --- |
| psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 2 |

## 13. MRB1 distributions

MRB1 is a frozen model signal, not ground truth, and no cutoff is selected.

| criterion | count | min | q1 | median | mean | q3 | max | std_population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistake_Identification | 57 | 0.091523 | 0.485762 | 0.754165 | 0.675653 | 0.902460 | 0.973557 | 0.258828 |
| Mistake_Location | 57 | 0.062339 | 0.394947 | 0.698532 | 0.584291 | 0.806693 | 0.927603 | 0.265329 |
| Providing_Guidance | 57 | 0.116575 | 0.525385 | 0.747580 | 0.660029 | 0.850311 | 0.921500 | 0.247014 |
| Actionability | 57 | 0.088484 | 0.809370 | 0.907230 | 0.778665 | 0.938864 | 0.960767 | 0.269453 |

By final move:

| final_move | turn_count | mean_MI | mean_ML | mean_PG | mean_A |
| --- | --- | --- | --- | --- | --- |
| generic | 14 | 0.447157 | 0.318296 | 0.408751 | 0.568578 |
| probing | 20 | 0.846300 | 0.791255 | 0.810181 | 0.919365 |
| focus | 23 | 0.666349 | 0.566233 | 0.682413 | 0.784195 |
| telling | 0 |  |  |  |  |

By evaluator correctness:

| correctness | turn_count | mean_MI | mean_ML | mean_PG | mean_A |
| --- | --- | --- | --- | --- | --- |
| correct | 27 | 0.625829 | 0.530271 | 0.617503 | 0.704866 |
| partial | 2 | 0.860306 | 0.823489 | 0.767167 | 0.926625 |
| incorrect | 11 | 0.821757 | 0.750428 | 0.790036 | 0.920752 |
| unknown | 17 | 0.638523 | 0.534447 | 0.630843 | 0.786529 |
| <missing> | 0 |  |  |  |  |

## 14. Attempt-level versus turn-level credit

| attempt_number | attempt_id | skill | old_lints_cumulative_final_mastery_delta | dialogue_turn_count | dialogue_delta_min | dialogue_delta_median | dialogue_delta_mean | dialogue_delta_max | positive_turns | zero_turns | negative_turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | Equation Solving Two or Fewer Steps | 0.332383 | 15 | -0.138406 | 0.054455 | 0.010914 | 0.099359 | 8 | 0 | 7 |
| 2 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | Equation Solving Two or Fewer Steps | 0.022422 | 8 | -0.017476 | -0.001064 | -0.001022 | 0.012955 | 4 | 0 | 4 |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | Complementary and Supplementary Angles | 0.154487 | 9 | -0.139762 | -0.047900 | -0.010810 | 0.177100 | 3 | 0 | 6 |
| 4 | f3133173-8f14-44d9-b485-97a42dc45030:1 | Conversion of Fraction Decimals Percents | 0.327698 | 10 | -0.154909 | 0.057753 | 0.010705 | 0.080970 | 7 | 0 | 3 |
| 5 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | Multiplication Whole Numbers | 0.334525 | 7 | -0.106478 | -0.029859 | 0.001594 | 0.180425 | 2 | 0 | 5 |
| 6 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | Simplifying Expressions positive exponents | 0.988958 | 8 | -0.004145 | 0.043881 | 0.123113 | 0.556531 | 6 | 0 | 2 |

Attempts with positive cumulative outcome: **6**; among them, attempts containing at least one negative dialogue delta: **6**. Full immediate-delta lists are retained in `attempt_vs_turn_credit.csv`. This is descriptive evidence about delayed credit assignment.

## 15. Formal assessment contribution

`assessment_absolute_path_share = |assessment net| / (|dialogue net| + |assessment net|)`; it describes evidence-path magnitude and does not attach assessment evidence to Tutor moves.

| attempt_number | attempt_id | skill | dialogue_net_delta | assessment_net_delta | total_delta | assessment_absolute_path_share | dialogue_assessment_directions_differ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | Equation Solving Two or Fewer Steps | 0.163709 | 0.168674 | 0.332383 | 0.507470 | False |
| 2 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | Equation Solving Two or Fewer Steps | -0.008173 | 0.030595 | 0.022422 | 0.789174 | True |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | Complementary and Supplementary Angles | -0.097286 | 0.251772 | 0.154487 | 0.721290 | True |
| 4 | f3133173-8f14-44d9-b485-97a42dc45030:1 | Conversion of Fraction Decimals Percents | 0.107049 | 0.220649 | 0.327698 | 0.673330 | False |
| 5 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | Multiplication Whole Numbers | 0.011157 | 0.323368 | 0.334525 | 0.966650 | False |
| 6 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | Simplifying Expressions positive exponents | 0.984904 | 0.004055 | 0.988958 | 0.004100 | False |

Attempts with negative dialogue net but positive final total: `[2, 3]`.

## 16. Data sufficiency

- Total real turns: **57**
- Turns per final move: `{"focus": 23, "generic": 14, "probing": 20, "telling": 0}`
- Turns per skill: `{"Complementary and Supplementary Angles": 9, "Conversion of Fraction Decimals Percents": 10, "Equation Solving Two or Fewer Steps": 23, "Multiplication Whole Numbers": 7, "Simplifying Expressions positive exponents": 8}`
- Learner-agency modifications: **0**
- LinTS actual overrides: **1**
- Positive / zero / negative dialogue turns: **30 / 0 / 27**
- Distinct learners: **1**
- Starting-mastery range: **0.011042 to 0.989414**

This is enough for **pipeline and analysis prototyping**. It is not yet enough for a tiny scientific retraining pilot beyond an engineering smoke test, and it is not enough for any claim of policy improvement. The limiting factors are one learner, only five skills, sparse agency/LinTS intervention coverage, move imbalance, and non-randomized observational decisions.

## 17. Issues / limitations

1. One learner supplies all six attempts; within-learner and repeated-skill dependence is material.
2. Attempt 1 is pre-fix and legitimately lacks the new lifecycle/provenance fields.
3. The legacy deployment lineage manifest contains local absolute paths and a creation-time update count; post-fix research records themselves pass the portability/privacy audit.
4. Resolver behavioural-presence booleans are absent from turn-outcome records and cannot be reconstructed without inventing values.
5. Move and skill coverage are imbalanced, with observational rather than randomized selection.
6. High-mastery turns compress available BKT headroom; immediate deltas must not be treated as causal rewards.
7. MRB1 and evaluator outputs are model signals, not ground truth.

## 18. Derived artifact paths

All artifacts are under `pedagogical-move-selection/results/md_self_improvement_pilot_analysis_v1/`:

- `assessment_contribution.csv`
- `assessment_observations.csv`
- `attempt_table.csv`
- `attempt_vs_turn_credit.csv`
- `decision_group_summary.csv`
- `delta_mastery_histogram.png`
- `delta_vs_mastery_before_scatter.png`
- `descriptive_statistics.csv`
- `effective_gap_distribution.png`
- `evaluator_delta_crosstab.csv`
- `final_move_counts.png`
- `headroom_bands.csv`
- `integrity_by_attempt.csv`
- `integrity_totals.csv`
- `learner_agency_turns.csv`
- `learner_skill_counts.csv`
- `lints_eligible_arm_count_distribution.png`
- `lints_overrides.csv`
- `mastery_before_histogram.png`
- `move_outcome_observational.csv`
- `move_transitions.csv`
- `mrb1_by_evaluator.csv`
- `mrb1_by_final_move.csv`
- `mrb1_criterion_distributions.png`
- `mrb1_overall.csv`
- `provenance_audit.csv`
- `raw_md7_move_counts.png`
- `resolver_summary.csv`
- `skill_summary.csv`
- `summary.json`
- `turn_table.csv`
- `report.md`

## 19. Side effects

| effect | count |
| --- | --- |
| training | 0 |
| model writes | 0 |
| policy writes | 0 |
| authoritative research writes | 0 |
| API calls | 0 |
| Tutor conversations | 0 |
| protected evaluation use | 0 |
| protected-input hash changes | 0 |

Only the derived-analysis directory was written. No APIs, Tutor conversations, training, model/policy updates, authoritative JSONL writes, or protected MathDial/MRBench V3 test access occurred.

## 20. Final verdict

**B. PILOT DATASET CLEAN BUT COVERAGE TOO WEAK — COLLECT MORE BEFORE MD8 DESIGN**
