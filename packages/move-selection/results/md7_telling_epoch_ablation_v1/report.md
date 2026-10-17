# MD7 Telling-Calibration Epoch Ablation

## Outcome

**B. ALL EPOCHS OVERCORRECT**

Epoch 1 and Epoch 2 both fail the same unchanged v1 boundary rubric as Epoch 3; earlier stopping does not remove the substantive recoverable-state boundary failure. This is an offline candidate-selection result only; no model is promoted.

## Safety and frozen implementation

The deployed MD7-R1 baseline reproduced the established MathDial validation metrics exactly on 1,850 rows: accuracy **0.516216216216**, Macro-F1 **0.461128639026**, and per-class F1 **generic 0.610526315789, probing 0.333333333333, focus 0.600467289720, telling 0.300187617261**.

The historical real-state reconstruction parity check against original collector raw MD7 telemetry also passed: maximum absolute probability difference **1.25169754e-06**, rows failing `1e-5`: **0**, historical top-1 mismatches: **0**. Parity against the frozen v1 export had maximum difference **1.110223025e-16**. Challenge baseline parity had maximum difference **1.110223025e-16** and **0** failures.

The exact v1 implementation and inputs were reused. The contract remains paired `Problem:\n...` and `Conversation:\n...`, dialogue lines `{user}: {text}`, suffix `\n\nNext teacher pedagogical move:`, left tokenizer truncation, `only_second`, maximum length 512, and label order `generic, probing, focus, telling`. No cases, thresholds, or real states were changed.

## Frozen comparison results

| model | mathdial_macro_f1 | telling_plausible_telling_count | telling_plausible_mean_p_telling | telling_plausible_median_p_telling | hard_negative_false_telling_count | hard_negative_mean_p_telling | hard_negative_max_p_telling | recoverable_false_telling_count | recoverable_mean_p_telling | real_state_telling_count | real_state_displacement_count | real_state_displacement_rate | real_focus_to_telling_count | real_probing_to_telling_count | real_generic_to_telling_count | mean_real_delta_p_telling | median_real_delta_p_telling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.461129 | 19 | 0.776173 | 0.973431 | 0 | 0.001530 | 0.004042 | 0 | 0.002208 | 0 | 0 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 |
| epoch1 | 0.425292 | 23 | 0.961113 | 0.997780 | 2 | 0.074067 | 0.844631 | 2 | 0.431178 | 0 | 15 | 0.172414 | 0 | 0 | 0 | 0.024483 | 0.019809 |
| epoch2 | 0.441921 | 24 | 0.993936 | 0.998658 | 3 | 0.107380 | 0.975312 | 3 | 0.636893 | 5 | 20 | 0.229885 | 2 | 3 | 0 | 0.038394 | 0.002372 |
| epoch3 | 0.449797 | 24 | 0.996900 | 0.998770 | 3 | 0.112168 | 0.963416 | 3 | 0.665688 | 4 | 24 | 0.275862 | 2 | 2 | 0 | 0.020975 | -0.003099 |

The recoverable-after-one-scaffold subgroup contains the same four cases as v1. Probability order is `generic, probing, focus, telling`.

| case | model | top-1 | probability vector [generic, probing, focus, telling] |
| --- | --- | --- | --- |
| I01 | baseline | generic | [generic=0.546811, probing=0.446812, focus=0.003049, telling=0.003327] |
| I01 | epoch1 | focus | [generic=0.000290, probing=0.009642, focus=0.956671, telling=0.033397] |
| I01 | epoch2 | telling | [generic=0.002870, probing=0.002528, focus=0.453599, telling=0.541003] |
| I01 | epoch3 | telling | [generic=0.001294, probing=0.003980, focus=0.436106, telling=0.558619] |
| I02 | baseline | probing | [generic=0.020586, probing=0.970818, focus=0.005083, telling=0.003514] |
| I02 | epoch1 | focus | [generic=0.000293, probing=0.010015, focus=0.957222, telling=0.032470] |
| I02 | epoch2 | focus | [generic=0.001509, probing=0.003894, focus=0.775909, telling=0.218688] |
| I02 | epoch3 | focus | [generic=0.000785, probing=0.003947, focus=0.804939, telling=0.190328] |
| I03 | baseline | generic | [generic=0.996410, probing=0.002072, focus=0.000460, telling=0.001057] |
| I03 | epoch1 | telling | [generic=0.001165, probing=0.011638, focus=0.142565, telling=0.844631] |
| I03 | epoch2 | telling | [generic=0.002627, probing=0.000665, focus=0.021395, telling=0.975312] |
| I03 | epoch3 | telling | [generic=0.000890, probing=0.001773, focus=0.046948, telling=0.950389] |
| I04 | baseline | generic | [generic=0.996709, probing=0.002016, focus=0.000342, telling=0.000933] |
| I04 | epoch1 | telling | [generic=0.008423, probing=0.037351, focus=0.140012, telling=0.814214] |
| I04 | epoch2 | telling | [generic=0.121193, probing=0.005212, focus=0.061027, telling=0.812568] |
| I04 | epoch3 | telling | [generic=0.020311, probing=0.005598, focus=0.010675, telling=0.963416] |

The complete rows appear in `recoverable_one_scaffold_cases.csv`; dialogue-level inspection for Epoch 1 and Epoch 2 appears in `important_cases.md`.

## Synthetic calibration evidence

The saved training-run metadata reports perfect four-class scores on its synthetic target validation and perfect telling precision/recall with zero non-telling false triggers on its synthetic challenge for all three epochs. Those results describe synthetic calibration behavior only and are **not** evidence of real tutoring effectiveness.

## Real-state offline transfer evidence

All selectors were evaluated on the exact same 87 leakage-free pre-action states. Epoch 3 values are the frozen v1 results; baseline, Epoch 1, and Epoch 2 use the same inference implementation and formatting. Policy displacement means top-1 differs from deployed baseline; it is not a claim that the alternative action would have improved learning.

## Observational historical outcomes

The output preserves evaluator category, immediate mastery delta, and mastery-before fields solely as post-prediction observational joins. They were not used as selector inputs and do not establish counterfactual effects.

## Claims this diagnostic does not support

- Telling would have caused better learning.
- Any candidate would have improved a historical outcome.
- Negative BKT proves telling was needed.
- Synthetic perfection demonstrates real tutoring effectiveness.

## Model paths and hashes

| model | path | model.safetensors SHA-256 |
| --- | --- | --- |
| baseline | `pedagogical-move-selection/models/candidates/md7r1_epoch3` | `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13` |
| epoch1 | `pedagogical-move-selection/models/candidates/md7_telling_calibration_epoch3/md7_telling_calibration_v5_candidates/md7_telling_calibration_v5/epoch1` | `1dc67fa3a3296c386214c36e696cd5bf82cb7884c707381fb01f437e6a36103a` |
| epoch2 | `pedagogical-move-selection/models/candidates/md7_telling_calibration_epoch3/md7_telling_calibration_v5_candidates/md7_telling_calibration_v5/epoch2` | `6f4f19dcec16a92fe599b80428ce57a5e07fb8e828916c03ed0c99a4bc3c2314` |
| epoch3 | `pedagogical-move-selection/models/candidates/md7_telling_calibration_epoch3/md7_telling_calibration_v5_candidates/md7_telling_calibration_v5/epoch3` | `a24d364b7d7461144afa97bae70c3690c4ce7af919927010cfb3fce73df9619c` |


Exact analyzed counts: **87 real states**, **48 challenge cases** (**24 telling-plausible**, **24 hard non-telling**, including **4 recoverable-after-one-scaffold**).

## Final table

| metric | baseline | epoch1 | epoch2 | epoch3 |
| --- | --- | --- | --- | --- |
| MathDial Macro-F1 | 0.461129 | 0.425292 | 0.441921 | 0.449797 |
| telling-plausible telling count /24 | 19/24 | 23/24 | 24/24 | 24/24 |
| telling-plausible mean telling probability | 0.776173 | 0.961113 | 0.993936 | 0.996900 |
| hard-negative false telling count /24 | 0/24 | 2/24 | 3/24 | 3/24 |
| recoverable-one-scaffold false telling count /4 | 0/4 | 2/4 | 3/4 | 3/4 |
| real-state telling count /87 | 0/87 | 0/87 | 5/87 | 4/87 |
| real-state displacement count /87 | 0/87 | 15/87 | 20/87 | 24/87 |
| real-state displacement % | 0.000% | 17.241% | 22.989% | 27.586% |
| mean real-state telling-probability increase | 0.000000 | 0.024483 | 0.038394 | 0.020975 |

**DECISION: B**

Recommended next action: Design a targeted recalibration set with recoverable-after-one-scaffold hard negatives, train only under a separately authorized experiment, and then repeat this unchanged frozen diagnostic.
