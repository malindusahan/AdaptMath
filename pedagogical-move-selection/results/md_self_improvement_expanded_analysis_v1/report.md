# Expanded Real-Data Analysis Before MD8-C1 Design

Analysis version: `md_self_improvement_expanded_analysis_v1`  
Primary inferential scope: **completed ordinary-mode cohort only** (12 attempts, 87 observed dialogue turns). Inventory and integrity additionally cover all 96 action records, including two censored attempts. All results are descriptive and non-causal.

## 1. DATASET INVENTORY

- Completed real attempts: **12** (old 6; **+6**).
- Aborted/censored attempts: **2** (old 0; **+2**). Attempt `4220e000…:1` has 7 observed turns plus one censored action; `978c3e3b…:2` has one censored action and no observed response.
- Action records: **96**; observed dialogue updates across all attempts: **94**; completed-cohort dialogue turns: **87** (old 57; **+30**).
- Formal-assessment observations: **36** (old 18; **+18**).
- Unique learners: **3** (old 1); completed-cohort skills: **11** (old 5); learner-skill pairs: **11**.
- Pending actions / processing failures: **0 / 0**.

| collection_sequence | attempt_id | learner_pseudonym | skill | completion_status | action_record_count | observed_dialogue_turns | censored_actions | assessment_item_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | completed | 8 | 8 | 0 | 3 |
| 2 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | psn_f78c0817d43053e5bd31da20 | Complementary and Supplementary Angles | completed | 9 | 9 | 0 | 3 |
| 3 | f3133173-8f14-44d9-b485-97a42dc45030:1 | psn_f78c0817d43053e5bd31da20 | Conversion of Fraction Decimals Percents | completed | 10 | 10 | 0 | 3 |
| 4 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | psn_f78c0817d43053e5bd31da20 | Multiplication Whole Numbers | completed | 7 | 7 | 0 | 3 |
| 5 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | psn_f78c0817d43053e5bd31da20 | Simplifying Expressions positive exponents | completed | 8 | 8 | 0 | 3 |
| 6 | b9a3b98d-ab91-4e24-b3f7-39a0623831b9:1 | psn_9a081eb5df240169b9f3c83b | Polynomial Factors | completed | 5 | 5 | 0 | 3 |
| 7 | 4220e000-557f-487a-949f-a6b3662998ac:1 | psn_9a081eb5df240169b9f3c83b | Pythagorean Theorem | aborted_censored | 8 | 7 | 1 | 0 |
| 8 | 0adabdf3-b861-4b0f-b802-451cc208dd0b:1 | psn_9a081eb5df240169b9f3c83b | Pythagorean Theorem | completed | 3 | 3 | 0 | 3 |
| 9 | 978c3e3b-ab82-4c73-9791-5dc47de68792:1 | psn_9a081eb5df240169b9f3c83b | Perimeter of a Polygon | completed | 2 | 2 | 0 | 3 |
| 10 | 978c3e3b-ab82-4c73-9791-5dc47de68792:2 | psn_9a081eb5df240169b9f3c83b | Perimeter of a Polygon | aborted_censored | 1 | 0 | 1 | 0 |
| 11 | bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1 | psn_b94f4a505b4ee07be23867fc | Multiplication and Division Integers | completed | 9 | 9 | 0 | 3 |
| 12 | 9bba79bb-b00a-40ca-a121-24da1db2e319:1 | psn_b94f4a505b4ee07be23867fc | Interior Angles Triangle | completed | 5 | 5 | 0 | 3 |
| 13 | eef1d04e-fe55-4457-9edf-f1c05786fd69:1 | psn_b94f4a505b4ee07be23867fc | Multiplication and Division Positive Decimals | completed | 6 | 6 | 0 | 3 |
| 14 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | completed_legacy_inferred | 15 | 15 | 0 | 3 |

## 2. INTEGRITY / PRIVACY / PROVENANCE

| check | count |
| --- | --- |
| action_mastery_mismatches | 0 |
| assessment_chain_mismatches | 0 |
| assessment_separation_mismatches | 0 |
| censored_action_mastery_continuity_mismatches | 0 |
| conflicting_action_id_groups | 0 |
| conflicting_assessment_id_groups | 0 |
| conflicting_resolver_id_groups | 0 |
| conflicting_summary_id_groups | 0 |
| delta_equation_mismatches | 0 |
| duplicate_action_id_extra_records | 0 |
| duplicate_action_id_groups | 0 |
| duplicate_assessment_id_extra_records | 0 |
| duplicate_assessment_id_groups | 0 |
| duplicate_resolver_id_extra_records | 0 |
| duplicate_resolver_id_groups | 0 |
| duplicate_summary_id_extra_records | 0 |
| duplicate_summary_id_groups | 0 |
| exact_duplicate_assessment_records | 0 |
| exact_duplicate_summary_records | 0 |
| exact_duplicate_turn_records | 0 |
| history_text_temporal_mismatches | 0 |
| index_contiguous_mismatches | 0 |
| index_monotonic_mismatches | 0 |
| index_starts_at_1_mismatches | 0 |
| mastery_continuity_mismatches | 0 |
| pending_actions | 0 |
| pending_processing_failures | 0 |
| resolver_action_identity_mismatches | 0 |
| timestamp_order_mismatches | 0 |

Blocking identity/temporal/mastery mismatch count: **0**. Duplicate action, resolver, assessment, and summary identifiers; conflicting duplicates; history continuity; mastery continuity; delta equations; assessment chains; and formal-assessment separation all pass.

The two censored rows correctly have no evaluator/resolver/BKT update and are not counted as outcome turns. The seven observed rows in the aborted manual attempt retain internally continuous mastery but have no final assessment or cumulative attempt outcome.

Privacy scan scope is every post-first-attempt turn, assessment, and summary, including censored attempts. Pattern counts: `{"absolute_C_Users_paths": 0, "account_id": 0, "api_key": 0, "auth_token": 0, "email_address": 0, "raw_username_Lenovo": 0, "secret": 0}`; rooted path fields: **0**; invalid pseudonyms: **0**. The known pre-fix first-attempt path was not rewritten or copied into the derived tables.

| field | expected | completed_cohort | all_action_records |
| --- | --- | --- | --- |
| data_mode | real | real (87/87) | real (96/96) |
| selector_mode | ordinary-md7r1-v1 | ordinary-md7r1-v1 (87/87) | {"manual-controlled-live-md7r1-v1": 1, "ordinary-md7r1-v1": 95} |
| MD7 | md7r1_epoch3 / d32d…a13 | consistent | consistent |
| move_order | G/P/F/T | consistent post-fix; legacy attempt missing | consistent post-fix; legacy attempt missing |
| agency/C3/LinTS/tau | agency_v1 / C3-9d / LinTS-v3 / .10 | consistent | consistent |
| MRB1/resolver/BKT | frozen_mrb1 / 2.0 / confidence_weighted_bkt_v1 | consistent | consistent |

Important provenance exception: all **87 completed-cohort turns** are `ordinary-md7r1-v1`; the aborted `4220e000…:1` contributes **8 manual-controlled action records** (7 observed, 1 censored). It is inventoried and integrity-audited but excluded from primary coverage/outcome statistics. The remaining censored `978c3e3b…:2` row is ordinary-mode.

## 3. LEARNER COVERAGE

| learner_pseudonym | attempts | turns | skills | start_min | start_median | start_mean | start_max | final_min | final_median | final_mean | final_max | move_generic | move_probing | move_focus | move_telling | eval_correct | eval_partial | eval_incorrect | eval_unknown | bkt_positive | bkt_zero | bkt_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psn_9a081eb5df240169b9f3c83b | 3 | 10 | 3 | 0.209177 | 0.732229 | 0.614106 | 0.900912 | 0.910146 | 0.956673 | 0.948763 | 0.979468 | 3 | 5 | 2 | 0 | 9 | 1 | 0 | 0 | 10 | 0 | 0 |
| psn_b94f4a505b4ee07be23867fc | 3 | 20 | 3 | 0.780739 | 0.813552 | 0.814764 | 0.850000 | 0.989841 | 0.991430 | 0.993401 | 0.998931 | 5 | 5 | 10 | 0 | 14 | 3 | 2 | 1 | 17 | 0 | 3 |
| psn_f78c0817d43053e5bd31da20 | 6 | 57 | 5 | 0.011042 | 0.651145 | 0.624626 | 0.974046 | 0.944455 | 0.995810 | 0.984705 | 1.000000 | 14 | 20 | 23 | 0 | 27 | 2 | 11 | 17 | 30 | 0 | 27 |

These are **within-learner repeated observations**: 12 attempts and 87 turns are not 12 or 87 independent learners. Between-learner evidence consists of only **3 pseudonymous learners**. Each learner contributes multiple turns and multiple skills; all distribution summaries above are descriptive.

## 4. SKILL COVERAGE

| skill | unique_learners | attempts | turns | start_min | start_median | start_mean | start_max | raw_generic | raw_probing | raw_focus | raw_telling | final_generic | final_probing | final_focus | final_telling | eval_correct | eval_partial | eval_incorrect | eval_unknown | bkt_positive | bkt_zero | bkt_negative | dialogue_net | assessment_net | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Complementary and Supplementary Angles | 1 | 1 | 9 | 0.843622 | 0.843622 | 0.843622 | 0.843622 | 1 | 6 | 2 | 0 | 2 | 6 | 1 | 0 | 2 | 1 | 1 | 5 | 3 | 0 | 6 | -0.097286 | 0.251772 | single_learner;single_attempt;very_small_turn_count_<10;near_saturation |
| Conversion of Fraction Decimals Percents | 1 | 1 | 10 | 0.616757 | 0.616757 | 0.616757 | 0.616757 | 1 | 4 | 5 | 0 | 1 | 4 | 5 | 0 | 6 | 1 | 2 | 1 | 7 | 0 | 3 | 0.107049 | 0.220649 | single_learner;single_attempt |
| Equation Solving Two or Fewer Steps | 1 | 2 | 23 | 0.641663 | 0.807854 | 0.807854 | 0.974046 | 9 | 3 | 11 | 0 | 9 | 3 | 11 | 0 | 12 | 0 | 3 | 8 | 12 | 0 | 11 | 0.155535 | 0.199269 | single_learner;high_starting_mastery |
| Interior Angles Triangle | 1 | 1 | 5 | 0.813552 | 0.813552 | 0.813552 | 0.813552 | 3 | 1 | 1 | 0 | 3 | 1 | 1 | 0 | 4 | 0 | 1 | 0 | 4 | 0 | 1 | 0.128792 | 0.056587 | single_learner;single_attempt;very_small_turn_count_<10;near_saturation |
| Multiplication Whole Numbers | 1 | 1 | 7 | 0.660628 | 0.660628 | 0.660628 | 0.660628 | 1 | 6 | 0 | 0 | 1 | 6 | 0 | 0 | 2 | 0 | 3 | 2 | 2 | 0 | 5 | 0.011157 | 0.323368 | single_learner;single_attempt;very_small_turn_count_<10;near_saturation |
| Multiplication and Division Integers | 1 | 1 | 9 | 0.780739 | 0.780739 | 0.780739 | 0.780739 | 1 | 1 | 7 | 0 | 1 | 1 | 7 | 0 | 7 | 2 | 0 | 0 | 9 | 0 | 0 | 0.173956 | 0.035146 | single_learner;single_attempt;very_small_turn_count_<10 |
| Multiplication and Division Positive Decimals | 1 | 1 | 6 | 0.850000 | 0.850000 | 0.850000 | 0.850000 | 1 | 3 | 2 | 0 | 1 | 3 | 2 | 0 | 3 | 1 | 1 | 1 | 4 | 0 | 2 | -0.035826 | 0.177257 | single_learner;single_attempt;very_small_turn_count_<10;near_saturation |
| Perimeter of a Polygon | 1 | 1 | 2 | 0.900912 | 0.900912 | 0.900912 | 0.900912 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0.058290 | -0.049055 | single_learner;single_attempt;very_small_turn_count_<10;high_starting_mastery |
| Polynomial Factors | 1 | 1 | 5 | 0.209177 | 0.209177 | 0.209177 | 0.209177 | 1 | 3 | 1 | 0 | 1 | 3 | 1 | 0 | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0.545461 | 0.224830 | single_learner;single_attempt;very_small_turn_count_<10 |
| Pythagorean Theorem | 1 | 1 | 3 | 0.732229 | 0.732229 | 0.732229 | 0.732229 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 2 | 1 | 0 | 0 | 3 | 0 | 0 | 0.095157 | 0.129288 | single_learner;single_attempt;very_small_turn_count_<10 |
| Simplifying Expressions positive exponents | 1 | 1 | 8 | 0.011042 | 0.011042 | 0.011042 | 0.011042 | 1 | 1 | 6 | 0 | 1 | 1 | 6 | 0 | 5 | 0 | 2 | 1 | 6 | 0 | 2 | 0.984904 | 0.004055 | single_learner;single_attempt;very_small_turn_count_<10;near_saturation |

All 11 skills are single-learner skills; 10/11 are single-attempt skills. Small cells, high starts, and saturation flags are explicit. Skill differences must not be interpreted as equal-difficulty action comparisons.

## 5. MOVE COVERAGE

| stage | move | turns | learners | skills |
| --- | --- | --- | --- | --- |
| raw | generic | 21 | 3 | 11 |
| raw | probing | 30 | 3 | 11 |
| raw | focus | 36 | 3 | 9 |
| raw | telling | 0 | 0 | 0 |
| effective | generic | 21 | 3 | 11 |
| effective | probing | 30 | 3 | 11 |
| effective | focus | 36 | 3 | 9 |
| effective | telling | 0 | 0 | 0 |
| final | generic | 22 | 3 | 11 |
| final | probing | 30 | 3 | 11 |
| final | focus | 35 | 3 | 9 |
| final | telling | 0 | 0 | 0 |

Raw/effective/final G/P/F/T totals are **21/30/36/0**, **21/30/36/0**, and **22/30/35/0**.

**There are no real telling actions.** Telling origin counts are raw MD7 **0**, learner-agency **0**, and LinTS override **0**. This is a major action-support gap.

## 6. LEARNER-AGENCY COVERAGE

Triggers: **0/87 (0.000%)**.

_None._

Known documented semantic non-engagement misses that are detectable in the completed cohort: **2**.

| learner_pseudonym | skill | action_turn_index | latest_student_text | raw_move | effective_move | final_move | correctness | delta_mastery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 6 | i cannot understand clearly | focus | focus | focus | incorrect | -0.138406 |
| psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 15 | i dont know about that | generic | generic | generic | correct | 0.061126 |

No agency-policy change is proposed here.

## 7. LINTS AUTHORITY

- Baseline-only eligible: **79/87 (90.805%)**.
- Nonbaseline alternative eligible: **8/87 (9.195%)**.
- Exactly 2 / exactly 3 / 4+ eligible arms: **7 / 1 / 0**.
- Eligible-arm mean / median / max: **1.103448 / 1 / 3**.
- Selected arms: `{"baseline": 86, "focus_bias": 0, "generic_bias": 1, "probing_bias": 0, "telling_bias": 0}`.
- Actual final-move overrides: **1/87 (1.149%)**.
- Current per-arm updates: `{"baseline": 86, "focus_bias": 0, "generic_bias": 1, "probing_bias": 0, "telling_bias": 0}`; total **87**. This equals the 87 completed-cohort turns; aborted attempts did not receive delayed attempt credit.

| learner_pseudonym | skill | action_turn_index | latest_student_text | raw_probabilities | effective_probabilities | eligible_arms | selected_arm | base_move | final_move | mastery_before | correctness | delta_mastery | MI | ML | PG | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psn_f78c0817d43053e5bd31da20 | Complementary and Supplementary Angles | 8 | i dont undersand | {"focus":0.3232975900173187,"generic":0.22414202988147736,"probing":0.31149759888648987,"telling":0.14106281101703644} | {"focus":0.3232975900173187,"generic":0.22414202988147736,"probing":0.31149759888648987,"telling":0.14106281101703644} | ["baseline","generic_bias","probing_bias"] | generic_bias | focus | generic | 0.642868 | unknown | -0.073632 | 0.686155 | 0.561044 | 0.730183 | 0.916986 |

## 8. MD7 CONFIDENCE / GATE

Effective top1−top2 gap: min **0.001977**, Q1 **0.239213**, median **0.393084**, mean **0.500713**, Q3 **0.849716**, max **0.997960**.

| threshold | count | percent |
| --- | --- | --- |
| 0.020000 | 4 | 4.597701 |
| 0.050000 | 4 | 4.597701 |
| 0.100000 | 8 | 9.195402 |
| 0.150000 | 12 | 13.793103 |
| 0.200000 | 17 | 19.540230 |

At tau=.10, alternatives were allowed on **8/87 = 9.195%**, versus **3/57 = 5.263%** previously. Authority widened slightly but remains rare; no new tau is proposed.

## 9. BKT / HEADROOM

Mastery-before: `{"count": 87, "min": 0.0110415225804813, "q1": 0.6221267115815636, "median": 0.7397806131002348, "mean": 0.7285193946319061, "q3": 0.8988945098954768, "max": 0.9894141164608344, "std": 0.1954995169360748}`. Headroom: `{"count": 87, "min": 0.0105858835391656, "q1": 0.10110549010452319, "median": 0.2602193868997652, "mean": 0.2714806053680938, "q3": 0.37787328841843637, "max": 0.9889584774195188, "std": 0.19549951693607484}`. Delta: `{"count": 87, "min": -0.1710836699786672, "q1": -0.02848221711988145, "median": 0.0217703693489806, "mean": 0.024450430817018136, "q3": 0.06857612139128094, "max": 0.5565306848534494, "std": 0.09532161839036109}`. Absolute delta: `{"count": 87, "min": 0.0013711078406207, "q1": 0.02420705238002625, "median": 0.0494524306110741, "mean": 0.06793484743794173, "q3": 0.09188816873716675, "max": 0.5565306848534494, "std": 0.07119614458156424}`. Signs +/0/−: **57/0/30**.

| band | turns | learners | skills | mean_delta | median_delta | mean_abs_delta | median_abs_delta | positive | zero | negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [0,.25) | 2 | 2 | 2 | 0.331363 | 0.331363 | 0.331363 | 0.331363 | 2 | 0 | 0 |
| [.25,.50) | 8 | 2 | 4 | 0.048391 | 0.088426 | 0.103256 | 0.095455 | 5 | 0 | 3 |
| [.50,.75) | 36 | 3 | 8 | 0.028463 | 0.057753 | 0.080987 | 0.074665 | 23 | 0 | 13 |
| [.75,.90) | 19 | 3 | 7 | 0.001573 | 0.018872 | 0.049569 | 0.038780 | 13 | 0 | 6 |
| [.90,.97) | 14 | 3 | 6 | 0.002966 | 0.014227 | 0.035970 | 0.023277 | 11 | 0 | 3 |
| [.97,1] | 8 | 1 | 2 | -0.002343 | -0.002758 | 0.007579 | 0.007464 | 3 | 0 | 5 |

Headroom is descriptive only and is not converted into a weight.

## 10. EVALUATOR-BKT RELATION

| correctness | positive_count | zero_count | negative_count | positive_percent | zero_percent | negative_percent | row_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| correct | 50 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 | 50 |
| partial | 6 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 | 6 |
| incorrect | 1 | 0 | 12 | 7.692308 | 0.000000 | 92.307692 | 13 |
| unknown | 0 | 0 | 18 | 0.000000 | 0.000000 | 100.000000 | 18 |

The prior near-deterministic sign relationship persists to the degree shown in the row percentages. This is expected to make BKT delta partly an encoding of evaluator/resolver evidence; it is not, by itself, labeled a defect.

## 11. SIGNAL REDUNDANCY

| correctness | variable | count | min | q1 | median | mean | q3 | max | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | mastery_before | 50 | 0.011042 | 0.593966 | 0.746237 | 0.714277 | 0.903809 | 0.988043 | 0.219436 |
| correct | delta_mastery | 50 | 0.006018 | 0.027986 | 0.062910 | 0.075403 | 0.089645 | 0.556531 | 0.083604 |
| correct | update_confidence | 50 | 0.401012 | 0.428994 | 0.450362 | 0.449051 | 0.461459 | 0.509003 | 0.025962 |
| partial | mastery_before | 6 | 0.682055 | 0.734117 | 0.760260 | 0.774695 | 0.832685 | 0.863364 | 0.064774 |
| partial | delta_mastery | 6 | 0.009968 | 0.014439 | 0.016603 | 0.020118 | 0.022025 | 0.039987 | 0.009781 |
| partial | update_confidence | 6 | 0.157312 | 0.159170 | 0.164189 | 0.165694 | 0.170538 | 0.178324 | 0.007520 |
| incorrect | mastery_before | 13 | 0.402158 | 0.590915 | 0.720347 | 0.729470 | 0.905816 | 0.989414 | 0.188649 |
| incorrect | delta_mastery | 13 | -0.171084 | -0.138406 | -0.106478 | -0.083324 | -0.049452 | 0.127879 | 0.078714 |
| incorrect | update_confidence | 13 | 0.520059 | 0.538911 | 0.544840 | 0.552767 | 0.571475 | 0.593104 | 0.023233 |
| unknown | mastery_before | 18 | 0.475220 | 0.647308 | 0.731883 | 0.752004 | 0.838578 | 0.980109 | 0.148479 |
| unknown | delta_mastery | 18 | -0.095158 | -0.047831 | -0.037962 | -0.037804 | -0.026610 | -0.004145 | 0.022627 |
| unknown | update_confidence | 18 | 0.107374 | 0.157285 | 0.177689 | 0.175764 | 0.195966 | 0.237784 | 0.035714 |

| group | association | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| all | delta_mastery_vs_mastery_before | 87 | -0.454712 | -0.357002 |
| all | delta_mastery_vs_headroom | 87 | 0.454712 | 0.357002 |
| all | delta_mastery_vs_update_confidence | 87 | 0.155937 | 0.126894 |
| correct | delta_mastery_vs_mastery_before | 50 | -0.768121 | -0.905882 |
| correct | delta_mastery_vs_headroom | 50 | 0.768121 | 0.905882 |
| correct | delta_mastery_vs_update_confidence | 50 | 0.177000 | 0.234752 |
| partial | delta_mastery_vs_mastery_before | 6 | -0.526388 | -0.600000 |
| partial | delta_mastery_vs_headroom | 6 | 0.526388 | 0.600000 |
| partial | delta_mastery_vs_update_confidence | 6 | 0.233741 | 0.257143 |
| incorrect | delta_mastery_vs_mastery_before | 13 | 0.038448 | 0.076923 |
| incorrect | delta_mastery_vs_headroom | 13 | -0.038448 | -0.076923 |
| incorrect | delta_mastery_vs_update_confidence | 13 | -0.124606 | -0.016506 |
| unknown | delta_mastery_vs_mastery_before | 18 | 0.447950 | 0.308566 |
| unknown | delta_mastery_vs_headroom | 18 | -0.447950 | -0.308566 |
| unknown | delta_mastery_vs_update_confidence | 18 | -0.227682 | -0.175040 |

Evaluator-category eta-squared association with delta magnitude: **0.443624**. Conservatively, evaluator class strongly structures sign and location; mastery/headroom and update confidence structure magnitude within the BKT equation. Nonzero within-class ranges/std show residual variation, but it is not evidence of independent pedagogical-move effect.

## 12. MRB1

| group_type | group | criterion | count | min | q1 | median | mean | q3 | max | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | MI | 87 | 0.091523 | 0.455699 | 0.762043 | 0.676699 | 0.910428 | 0.973557 | 0.262623 |
| overall | all | ML | 87 | 0.062339 | 0.356222 | 0.698532 | 0.584972 | 0.816814 | 0.927603 | 0.274917 |
| overall | all | PG | 87 | 0.116575 | 0.518802 | 0.778082 | 0.655412 | 0.853994 | 0.921500 | 0.260700 |
| overall | all | A | 87 | 0.088484 | 0.788402 | 0.903506 | 0.758461 | 0.937409 | 0.961619 | 0.280912 |
| final_move | generic | MI | 22 | 0.174216 | 0.292968 | 0.329924 | 0.436722 | 0.564453 | 0.887411 | 0.190163 |
| final_move | generic | ML | 22 | 0.117926 | 0.117926 | 0.180662 | 0.298509 | 0.520882 | 0.745332 | 0.211522 |
| final_move | generic | PG | 22 | 0.141498 | 0.141498 | 0.243040 | 0.370055 | 0.623496 | 0.843544 | 0.253875 |
| final_move | generic | A | 22 | 0.159130 | 0.159130 | 0.436805 | 0.496671 | 0.885835 | 0.939281 | 0.339548 |
| final_move | probing | MI | 30 | 0.312332 | 0.768000 | 0.904455 | 0.843267 | 0.926038 | 0.973557 | 0.132678 |
| final_move | probing | ML | 30 | 0.357811 | 0.725592 | 0.815333 | 0.789941 | 0.881643 | 0.927603 | 0.109525 |
| final_move | probing | PG | 30 | 0.507907 | 0.784096 | 0.839890 | 0.814684 | 0.877980 | 0.921500 | 0.094279 |
| final_move | probing | A | 30 | 0.511338 | 0.898453 | 0.926625 | 0.908008 | 0.945840 | 0.959978 | 0.079942 |
| final_move | focus | MI | 35 | 0.091523 | 0.504704 | 0.822577 | 0.684771 | 0.906484 | 0.964838 | 0.271251 |
| final_move | focus | ML | 35 | 0.062339 | 0.421058 | 0.686368 | 0.589346 | 0.783595 | 0.891439 | 0.248700 |
| final_move | focus | PG | 35 | 0.116575 | 0.640676 | 0.800314 | 0.698261 | 0.851881 | 0.887563 | 0.214582 |
| final_move | focus | A | 35 | 0.088484 | 0.796582 | 0.907153 | 0.794832 | 0.936589 | 0.961619 | 0.233739 |
| final_move | telling | MI | 0 |  |  |  |  |  |  |  |
| final_move | telling | ML | 0 |  |  |  |  |  |  |  |
| final_move | telling | PG | 0 |  |  |  |  |  |  |  |
| final_move | telling | A | 0 |  |  |  |  |  |  |  |
| evaluator | correct | MI | 50 | 0.091523 | 0.422494 | 0.740005 | 0.658210 | 0.916514 | 0.973557 | 0.279947 |
| evaluator | correct | ML | 50 | 0.062339 | 0.352734 | 0.679374 | 0.567486 | 0.823012 | 0.927603 | 0.287098 |
| evaluator | correct | PG | 50 | 0.116575 | 0.512635 | 0.780638 | 0.646355 | 0.864880 | 0.921500 | 0.273257 |
| evaluator | correct | A | 50 | 0.088484 | 0.513044 | 0.895630 | 0.726338 | 0.931354 | 0.961619 | 0.296958 |
| evaluator | partial | MI | 6 | 0.292968 | 0.332723 | 0.628866 | 0.618932 | 0.887586 | 0.955054 | 0.281612 |
| evaluator | partial | ML | 6 | 0.117926 | 0.146483 | 0.486416 | 0.500676 | 0.851972 | 0.906299 | 0.350733 |
| evaluator | partial | PG | 6 | 0.141498 | 0.162370 | 0.474275 | 0.487757 | 0.788969 | 0.884229 | 0.323000 |
| evaluator | partial | A | 6 | 0.159130 | 0.225586 | 0.673501 | 0.586419 | 0.924586 | 0.927819 | 0.350084 |
| evaluator | incorrect | MI | 13 | 0.543446 | 0.754165 | 0.839768 | 0.808378 | 0.902460 | 0.926146 | 0.107727 |
| evaluator | incorrect | ML | 13 | 0.543937 | 0.706487 | 0.724697 | 0.743089 | 0.779314 | 0.888356 | 0.083323 |
| evaluator | incorrect | PG | 13 | 0.507907 | 0.794340 | 0.813418 | 0.785499 | 0.839034 | 0.878419 | 0.100622 |
| evaluator | incorrect | A | 13 | 0.825368 | 0.896078 | 0.919436 | 0.912088 | 0.943289 | 0.960767 | 0.042123 |
| evaluator | unknown | MI | 18 | 0.174216 | 0.403207 | 0.686016 | 0.652213 | 0.899800 | 0.950284 | 0.256247 |
| evaluator | unknown | ML | 18 | 0.117926 | 0.310055 | 0.609444 | 0.547449 | 0.799616 | 0.827980 | 0.260075 |
| evaluator | unknown | PG | 18 | 0.141498 | 0.515510 | 0.727437 | 0.642504 | 0.847919 | 0.880916 | 0.238758 |
| evaluator | unknown | A | 18 | 0.159130 | 0.818309 | 0.908917 | 0.794088 | 0.939331 | 0.959978 | 0.251536 |
| learner | psn_9a081eb5df240169b9f3c83b | MI | 10 | 0.292968 | 0.297809 | 0.578713 | 0.594987 | 0.861981 | 0.965075 | 0.280652 |
| learner | psn_9a081eb5df240169b9f3c83b | ML | 10 | 0.117926 | 0.176205 | 0.524003 | 0.513553 | 0.833863 | 0.893796 | 0.316423 |
| learner | psn_9a081eb5df240169b9f3c83b | PG | 10 | 0.141498 | 0.232617 | 0.655349 | 0.566061 | 0.831005 | 0.898354 | 0.305255 |
| learner | psn_9a081eb5df240169b9f3c83b | A | 10 | 0.159130 | 0.247182 | 0.808149 | 0.629331 | 0.911438 | 0.945302 | 0.329458 |
| learner | psn_b94f4a505b4ee07be23867fc | MI | 20 | 0.150568 | 0.526762 | 0.867054 | 0.720538 | 0.918191 | 0.955054 | 0.253867 |
| learner | psn_b94f4a505b4ee07be23867fc | ML | 20 | 0.110033 | 0.509322 | 0.739177 | 0.622622 | 0.834241 | 0.894618 | 0.272213 |
| learner | psn_b94f4a505b4ee07be23867fc | PG | 20 | 0.141498 | 0.654444 | 0.833359 | 0.686931 | 0.872437 | 0.887442 | 0.264790 |
| learner | psn_b94f4a505b4ee07be23867fc | A | 20 | 0.159130 | 0.748566 | 0.912102 | 0.765446 | 0.933815 | 0.961619 | 0.269298 |
| learner | psn_f78c0817d43053e5bd31da20 | MI | 57 | 0.091523 | 0.485762 | 0.754165 | 0.675653 | 0.902460 | 0.973557 | 0.258828 |
| learner | psn_f78c0817d43053e5bd31da20 | ML | 57 | 0.062339 | 0.394947 | 0.698532 | 0.584291 | 0.806693 | 0.927603 | 0.265329 |
| learner | psn_f78c0817d43053e5bd31da20 | PG | 57 | 0.116575 | 0.525385 | 0.747580 | 0.660029 | 0.850311 | 0.921500 | 0.247014 |
| learner | psn_f78c0817d43053e5bd31da20 | A | 57 | 0.088484 | 0.809370 | 0.907230 | 0.778665 | 0.938864 | 0.960767 | 0.269453 |

MRB1 values are frozen model outputs. No cutoff or good/bad label is introduced.

## 13. ATTEMPT VS TURN CREDIT

| attempt_number | attempt_id | learner_pseudonym | skill | dialogue_net_delta | assessment_net_delta | total_delta | positive_dialogue_turns | zero_dialogue_turns | negative_dialogue_turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5413cab2-22c8-4536-a66b-66619d887e84:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | 0.163709 | 0.168674 | 0.332383 | 8 | 0 | 7 |
| 2 | 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1 | psn_f78c0817d43053e5bd31da20 | Equation Solving Two or Fewer Steps | -0.008173 | 0.030595 | 0.022422 | 4 | 0 | 4 |
| 3 | ecaa67e8-e212-4866-877f-9e2dd4cced40:1 | psn_f78c0817d43053e5bd31da20 | Complementary and Supplementary Angles | -0.097286 | 0.251772 | 0.154487 | 3 | 0 | 6 |
| 4 | f3133173-8f14-44d9-b485-97a42dc45030:1 | psn_f78c0817d43053e5bd31da20 | Conversion of Fraction Decimals Percents | 0.107049 | 0.220649 | 0.327698 | 7 | 0 | 3 |
| 5 | a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1 | psn_f78c0817d43053e5bd31da20 | Multiplication Whole Numbers | 0.011157 | 0.323368 | 0.334525 | 2 | 0 | 5 |
| 6 | dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1 | psn_f78c0817d43053e5bd31da20 | Simplifying Expressions positive exponents | 0.984904 | 0.004055 | 0.988958 | 6 | 0 | 2 |
| 7 | b9a3b98d-ab91-4e24-b3f7-39a0623831b9:1 | psn_9a081eb5df240169b9f3c83b | Polynomial Factors | 0.545461 | 0.224830 | 0.770291 | 5 | 0 | 0 |
| 8 | 0adabdf3-b861-4b0f-b802-451cc208dd0b:1 | psn_9a081eb5df240169b9f3c83b | Pythagorean Theorem | 0.095157 | 0.129288 | 0.224444 | 3 | 0 | 0 |
| 9 | 978c3e3b-ab82-4c73-9791-5dc47de68792:1 | psn_9a081eb5df240169b9f3c83b | Perimeter of a Polygon | 0.058290 | -0.049055 | 0.009235 | 2 | 0 | 0 |
| 10 | bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1 | psn_b94f4a505b4ee07be23867fc | Multiplication and Division Integers | 0.173956 | 0.035146 | 0.209102 | 9 | 0 | 0 |
| 11 | 9bba79bb-b00a-40ca-a121-24da1db2e319:1 | psn_b94f4a505b4ee07be23867fc | Interior Angles Triangle | 0.128792 | 0.056587 | 0.185379 | 4 | 0 | 1 |
| 12 | eef1d04e-fe55-4457-9edf-f1c05786fd69:1 | psn_b94f4a505b4ee07be23867fc | Multiplication and Division Positive Decimals | -0.035826 | 0.177257 | 0.141430 | 4 | 0 | 2 |

Positive cumulative attempts: **12/12**. Of these, **8** contain at least one negative immediate dialogue turn. Attempts with dialogue net < 0 but total final delta > 0: **[2, 3, 12]**. This continues to expose the scientific weakness of assigning equal delayed attempt credit to every selected arm.

## 14. MOVE × OUTCOME OBSERVATIONS

**OBSERVATIONAL ASSOCIATIONS ONLY. ACTIONS WERE NOT RANDOMLY ASSIGNED. MOVES ARE NOT RANKED.**

| final_move | turns | learners | skills | mastery_mean | mastery_median | eval_correct | eval_partial | eval_incorrect | eval_unknown | bkt_positive | bkt_zero | bkt_negative | delta_mean | delta_median | MRB1_MI_mean | MRB1_ML_mean | MRB1_PG_mean | MRB1_A_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generic | 22 | 3 | 11 | 0.761682 | 0.818499 | 10 | 3 | 1 | 8 | 13 | 0 | 9 | 0.034750 | 0.013437 | 0.436722 | 0.298509 | 0.370055 | 0.496671 |
| probing | 30 | 3 | 11 | 0.688845 | 0.724690 | 13 | 2 | 9 | 6 | 16 | 0 | 14 | 0.002619 | 0.017064 | 0.843267 | 0.789941 | 0.814684 | 0.908008 |
| focus | 35 | 3 | 9 | 0.741682 | 0.704295 | 27 | 1 | 3 | 4 | 28 | 0 | 7 | 0.036689 | 0.032355 | 0.684771 | 0.589346 | 0.698261 | 0.794832 |
| telling | 0 | 0 | 0 |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |

## 15. OLD → NEW COVERAGE

| metric | old | new |
| --- | --- | --- |
| completed attempts | 6 | 12 |
| aborted attempts | 0 | 2 |
| learners | 1 | 3 |
| skills | 5 | 11 |
| completed-cohort turns | 57 | 87 |
| raw G/P/F/T | 13/20/24/0 | 21/30/36/0 |
| final G/P/F/T | 14/20/23/0 | 22/30/35/0 |
| agency triggers | 0 | 0 |
| LinTS overrides | 1 | 1 |
| baseline-only rate | 94.74% | 90.80% |
| alternative opportunity | 5.26% | 9.20% |
| BKT +/0/- | 30/0/27 | 57/0/30 |
| mastery-before range | .011042–.989414 | 0.011042–0.989414 |

- Learner diversity: improved from 1 to 3, but remains small.
- Skill diversity: improved from 5 to 11, though all skills remain single-learner and nearly all single-attempt.
- Move diversity: **not materially improved**; the raw ratio stayed similar and telling remains absent.
- Mastery-state diversity: range did not expand beyond the pilot endpoints; more mid/high-state observations were added.
- Alternative-action coverage: opportunity rose from 5.263% to 9.195%, but actual overrides stayed at one and three arms remain entirely unupdated.

## 16. MD8 SUFFICIENCY

- Data-pipeline prototyping: **Yes**.
- Engineering training smoke test: **Yes**, with no scientific claim.
- Tiny scientific SFT candidate: **Conditional/no at present**; only after explicit qualification plus replay and without treating turns as independent learners.
- Comparison against MD7: **No** for effectiveness; **yes only** for descriptive pipeline checks.
- Claim of tutoring-policy improvement: **No**.

## 17. RECOMMENDED NEXT EXPERIMENT CATEGORY

**B. Observational dataset is integrity-clean, but action coverage remains too policy-bound. Design controlled randomized research exploration before MD8-C1.**

Why: another six natural completed attempts added two learners and six skills but still produced zero telling, zero agency triggers, only 8/87 alternative opportunities, one actual override, and no updates at all for probing/focus/telling LinTS arms. Natural collection improved learner/skill breadth but did not break the deployed policy's action-support constraint.

No exploration percentage, selection algorithm, or eligibility threshold is chosen. Only the need is identified. State-region support includes **6** coarse mastery-band × evaluator regions with at least two learners and at least two observed final moves; these are coarse observational overlaps, not randomized comparable states. Full cells are in `state_region_action_coverage.csv`.

## 18. DERIVED ARTIFACTS

All outputs are under `pedagogical-move-selection/results/md_self_improvement_expanded_analysis_v1/`. Principal artifacts: `report.md`, `summary.json`, `attempt_table.csv`, `turn_table.csv`, `turn_table_completed_cohort.csv`, `learner_table.csv`, `skill_table.csv`, `move_table.csv`, integrity/provenance/agency/LinTS/BKT/evaluator/redundancy/MRB1/credit tables, and 11 descriptive plots.

## 19. SIDE EFFECTS

| effect | count |
| --- | --- |
| training | 0 |
| model_writes | 0 |
| policy_writes | 0 |
| authoritative_research_writes | 0 |
| api_calls | 0 |
| tutor_conversations | 0 |
| protected_evaluation_use | 0 |
| protected_input_hash_changes | 0 |

Only this derived-analysis directory was written. No API calls, Tutor conversations, training, model/policy writes, authoritative JSONL writes, or protected MathDial/MRBench V3 test access occurred.

## 20. FINAL VERDICT

**B. DATA CLEAN BUT ACTION COVERAGE POLICY-BOUND — DESIGN CONTROLLED EXPLORATION NEXT**
