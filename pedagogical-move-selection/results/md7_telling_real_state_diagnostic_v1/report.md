# MD7-R1 Telling-Calibration Real-State Diagnostic

This is an offline, CPU-only diagnostic. No model was trained or promoted; no production policy, BKT, LinTS, Tutor API, authoritative real-data JSONL, MathDial final test, or MRBench V3 test was touched.

## 1. Model safety and exact inference contract

The baseline-only gate ran first on exactly 1,850 rows from `pedagogical-move-selection/data/processed/mathdial/validation.jsonl` (SHA-256 `225e6b8f671bf1e335af175343722aa9ac2a603a96d67bcb5220be1e92c0b2b6`). It reproduced every required metric exactly: accuracy **0.5162162162162162**, Macro-F1 **0.4611286390258053**, and G/P/F/T F1 **0.6105263157894737 / 0.3333333333333333 / 0.6004672897196262 / 0.300187617260788**. Gate: **PASS**.

The inspected source contract is: sequence A `Problem:
<problem>`; sequence B `Conversation:
<history>` with exact `user: text` lines and suffix `

Next teacher pedagogical move:`; fast tokenizer; `truncation_side=left`; `truncation=only_second`; `max_length=512`; paired tokenization; class order `generic, probing, focus, telling`.

| model | path | model.safetensors_sha256 | config_sha256 | tokenizer_sha256 | tokenizer_config_sha256 |
| --- | --- | --- | --- | --- | --- |
| baseline | pedagogical-move-selection/models/candidates/md7r1_epoch3 | d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13 | 9877dd11007c8af1d2841a2c1bb470023ef2ceeb748b47c24686ea82479e9127 | 80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b | bb5706f71e3d5cb4bf9878ad9b14a2079e6fbd618fc0a0ba2ac1b257d7824cc9 |
| candidate | pedagogical-move-selection/models/candidates/md7_telling_calibration_epoch3/md7_telling_calibration_v5_candidates/md7_telling_calibration_v5/epoch3 | a24d364b7d7461144afa97bae70c3690c4ce7af919927010cfb3fce73df9619c | 419efd3794d791c4927dec0518ebe7d81765a22c3d8b8d28956589d625da52c0 | 80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b | bb5706f71e3d5cb4bf9878ad9b14a2079e6fbd618fc0a0ba2ac1b257d7824cc9 |

## 2. Evidence boundaries

- **Synthetic calibration evidence:** Epoch 3 synthetic telling challenge precision/recall are **1.0/1.0** with non-telling false-trigger rate **0.0**; untouched synthetic holdout accuracy/Macro-F1 and all class F1 values are **1.0**. Candidate MathDial validation accuracy/Macro-F1 are **0.496757/0.449797**, below baseline **0.516216/0.461129**. These establish calibration/retention on those sets only, not real tutoring effectiveness.
- **Real-state offline transfer evidence:** both frozen selectors scored the same 87 leakage-free pre-action states from completed ordinary-mode attempts.
- **Observational outcomes:** evaluator category, immediate BKT delta, and mastery-before were joined only after prediction for descriptive stratification.
- **Not entitled:** no causal learning benefit, counterfactual historical improvement, production readiness, or real effectiveness follows from these results.

## 3. Real deployment states

Exact real states analyzed: **87**. Formal-assessment selector states: **0**. Future learner responses/evaluations/BKT-after/attempt-final outcomes used for prediction: **none**. Recomputed baseline probabilities match stored historical raw MD7 probabilities with maximum absolute error **1.25169754e-06**; rows over 1e-5: **0**.

| metric | count | percent |
| --- | --- | --- |
| baseline top-1 telling | 0 | 0.00000 |
| candidate top-1 telling | 4 | 4.59770 |
| top-1 displacement | 24 | 27.58621 |
| changed to telling | 4 | 4.59770 |
| changed away from telling | 0 | 0.00000 |

Candidate-minus-baseline p(telling): min **-0.04556**, Q1 **-0.01137**, median **-0.00310**, mean **0.02098**, Q3 **0.00011**, max **0.67741**.

## 4. Manually auditable telling-boundary challenge

Challenge cases: **48**, balanced 24 telling-plausible / 24 hard non-telling across semantic groups A-J. Labels are manual diagnostic boundary judgments - not causal or outcome ground truth.

| boundary | model | n | top1_telling_count | top1_telling_rate | mean_telling_probability | median_telling_probability | max_telling_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| telling-plausible | baseline | 24 | 19 | 0.79167 | 0.77617 | 0.97343 | 0.99761 |
| telling-plausible | candidate | 24 | 24 | 1.00000 | 0.99690 | 0.99877 | 0.99923 |
| hard non-telling | baseline | 24 | 0 | 0.00000 | 0.00153 | 0.00112 | 0.00404 |
| hard non-telling | candidate | 24 | 3 | 0.12500 | 0.11217 | 0.00072 | 0.96342 |

Telling sensitivity gain: **0.20833**. Hard-negative false-trigger change: **0.12500**.

The false triggers are not diffuse: **3/4 (75.0%)** of the recoverable-after-one-scaffold group become telling, including cases where the learner's latest response already corrects the error.

| semantic_group | boundary | n | baseline_telling_count | candidate_telling_count | baseline_mean_p_telling | candidate_mean_p_telling |
| --- | --- | --- | --- | --- | --- | --- |
| A_repeated_confusion_after_two_scaffolds | telling_plausible | 6 | 6 | 6 | 0.99357 | 0.99768 |
| B_persistent_same_misconception | telling_plausible | 6 | 5 | 6 | 0.75226 | 0.99871 |
| C_explicit_direct_explanation_request | telling_plausible | 6 | 2 | 6 | 0.37892 | 0.99208 |
| D_repeated_microstep_failure | telling_plausible | 6 | 6 | 6 | 0.97995 | 0.99913 |
| E_first_incorrect_answer | non_telling | 4 | 0 | 0 | 0.00105 | 0.00057 |
| F_first_i_dont_know | non_telling | 4 | 0 | 0 | 0.00075 | 0.00064 |
| G_first_identifiable_misconception | non_telling | 4 | 0 | 0 | 0.00256 | 0.00300 |
| H_correct_or_partial_needing_reasoning | non_telling | 4 | 0 | 0 | 0.00142 | 0.00244 |
| I_recoverable_after_one_scaffold | non_telling | 4 | 0 | 3 | 0.00221 | 0.66569 |
| J_neutral_opening | non_telling | 4 | 0 | 0 | 0.00121 | 0.00068 |

## 5. Probability and policy movement

- Real-state candidate top-1 telling activation: **4/87 (4.598%)** versus baseline **0/87 (0.000%)**.
- Real-state top-1 displacement: **24/87 (27.586%)**.
- Real states changed to telling / away from telling: **4 / 0**.
- Challenge top-1 distributions are included in `summary.json` and `semantic_group_summary.csv`.

| baseline_top1 | candidate_top1 | count |
| --- | --- | --- |
| focus | focus | 21 |
| focus | probing | 13 |
| focus | telling | 2 |
| generic | focus | 1 |
| generic | generic | 18 |
| generic | probing | 2 |
| probing | focus | 4 |
| probing | probing | 24 |
| probing | telling | 2 |

## 6. Important semantic cases

All requested compact context/probability inspections are in `important_cases.md`: every real change to telling, every hard-negative candidate telling case, every plausible case still not telling, ten largest telling increases, and ten largest focus decreases.

## 7. Optional observational outcome stratification

Candidate-telling real states: **4**. Historical evaluator distribution: `{"correct": 4}`. Immediate delta stats: `{"count": 4, "min": 0.018871587353022856, "q1": 0.022376308830446168, "median": 0.0971070051937235, "mean": 0.09837768340440532, "q3": 0.17310837976768267, "max": 0.18042513587715142, "std": 0.07726432365638916}`. Mastery-before stats: `{"count": 4, "min": 0.32068980617750237, "q1": 0.45600865808536595, "median": 0.6872238985675669, "mean": 0.6480037519290092, "q3": 0.8792189924112103, "max": 0.8968774044034007, "std": 0.2456735848502619}`.

These describe historical states the candidate classifies as telling. They do not say telling would have changed those outcomes.

## 8. Decision

**B. TELLING OVERCORRECTION**

Candidate predicts telling on 3/4 recoverable-after-one-scaffold cases, including learner responses that already show recovery. This fails the required preservation of probing/focus behavior in recoverable difficulty.

No model is promoted. One recommended next experiment: **A targeted offline telling-boundary recalibration ablation that adds recoverable-after-one-scaffold hard negatives, followed by repetition of this same frozen diagnostic before any live exploration.**

## 9. Final required facts

1. Exact model paths and hashes are in the table above and `summary.json`.
2. Real states analyzed: **87**.
3. Challenge cases: **48**.
4. Telling-plausible top-1 telling: baseline **19/24 (79.17%)**, candidate **24/24 (100.00%)**; mean p(telling) **0.7762 -> 0.9969**.
5. Hard-negative false telling: baseline **0/24 (0.00%)**, candidate **3/24 (12.50%)**; mean/max candidate p(telling) **0.1122/0.9634**.
6. Real-state top-1 telling: baseline **0/87**, candidate **4/87**.
7. Policy-displacement rate: **24/87 (27.586%)**.
8. Decision: **B. TELLING OVERCORRECTION**.
9. Recommended next experiment: **A targeted offline telling-boundary recalibration ablation that adds recoverable-after-one-scaffold hard negatives, followed by repetition of this same frozen diagnostic before any live exploration.**

## 10. Side effects

| effect | count |
| --- | --- |
| training | 0 |
| model_writes | 0 |
| production_policy_writes | 0 |
| authoritative_real_data_writes | 0 |
| tutor_api_calls | 0 |
| bkt_updates | 0 |
| lints_updates | 0 |
| mathdial_final_test_use | 0 |
| mrbench_v3_test_use | 0 |
| protected_input_hash_changes | 0 |
