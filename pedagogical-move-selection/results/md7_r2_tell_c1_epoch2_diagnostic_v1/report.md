# MD7-R2-TELL-C1 Epoch 2 frozen telling-boundary diagnostic

This is a derived-only offline evaluation. It trained and promoted nothing, changed no runtime behavior, called no Tutor API, and did not read MathDial test or MRBench V3 test.

## Artifact and contract verification

The candidate is a complete Hugging Face export. Both configs specify the exact move order `generic, probing, focus, telling` and `RobertaForSequenceClassification`.

| model | path | model.safetensors | config.json | tokenizer.json | tokenizer_config.json |
| --- | --- | --- | --- | --- | --- |
| MD7-R1 | pedagogical-move-selection/models/candidates/md7r1_epoch3 | d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13 | 9877dd11007c8af1d2841a2c1bb470023ef2ceeb748b47c24686ea82479e9127 | 80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b | bb5706f71e3d5cb4bf9878ad9b14a2079e6fbd618fc0a0ba2ac1b257d7824cc9 |
| MD7-R2-C1 epoch2 | pedagogical-move-selection/models/candidates/md7_r2_tell_c1/epoch2 | ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5 | 419efd3794d791c4927dec0518ebe7d81765a22c3d8b8d28956589d625da52c0 | 80d0fb9a6bbc3ab2b24ec149d6b47981036a5f96c753667be38709975753550b | bb5706f71e3d5cb4bf9878ad9b14a2079e6fbd618fc0a0ba2ac1b257d7824cc9 |

The reused contract is exactly: `Problem:
<problem>` paired with `Conversation:
<formatted dialogue>

Next teacher pedagogical move:`; dialogue lines are `user: text`; tokenizer truncation side is left; truncation is `only_second`; maximum length is 512.

Frozen artifacts were verified by SHA-256 before scoring: the v1 implementation, frozen 48-case CSV, frozen v1 real-state predictions, MathDial validation, and the C1 synthetic challenge. The generated case objects match all 48 frozen CSV rows field-for-field.

## Baseline safety reproduction

Gate: **PASS**. On all 1,850 MathDial validation rows, MD7-R1 reproduced accuracy **0.5162162162162162**, Macro-F1 **0.4611286390258053**, and per-class F1 generic/probing/focus/telling **0.6105263157894737 / 0.3333333333333333 / 0.6004672897196262 / 0.3001876172607880**.

The 87-state historical raw-probability parity check also passed: maximum absolute difference **1.25169754e-06**; rows above `1e-5`: **0**.

## Candidate validation-reference reproduction

Local candidate MathDial validation: accuracy **0.5070270270270271**, Macro-F1 **0.4625716485004503**, per-class F1 generic/probing/focus/telling **0.6036866359447005 / 0.3066439522998297 / 0.5921299188007495 / 0.3478260869565217**; prediction distribution `{"focus": 923, "generic": 426, "probing": 175, "telling": 326}`. Exact reference match: **True**.

Local existing synthetic telling challenge: precision **0.9925558312655087**, recall **1.0000000000000000**, false-trigger rate **0.0075000000000000**, FP **3**. Exact reference match: **True**.

These validation results are retention/calibration evidence only, not evidence of real tutoring effectiveness.

## Frozen 48-case telling-boundary challenge

| boundary | model | n | top1_telling_count | top1_telling_rate | mean_telling_probability | median_telling_probability | minimum_telling_probability | maximum_telling_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| telling-plausible | MD7-R1 | 24 | 19 | 0.791667 | 0.776173 | 0.973431 | 0.003691 | 0.997613 |
| telling-plausible | MD7-R2-C1 epoch2 | 24 | 24 | 1.000000 | 0.975433 | 0.998775 | 0.734738 | 0.999220 |
| hard non-telling | MD7-R1 | 24 | 0 | 0.000000 | 0.001530 | 0.001122 | 0.000415 | 0.004042 |
| hard non-telling | MD7-R2-C1 epoch2 | 24 | 0 | 0.000000 | 0.007379 | 0.000549 | 0.000113 | 0.099842 |

Top-1 prediction distributions:

- Telling-plausible MD7-R1: `{"focus": 0, "generic": 5, "probing": 0, "telling": 19}`
- Telling-plausible candidate: `{"focus": 0, "generic": 0, "probing": 0, "telling": 24}`
- Hard-negative MD7-R1: `{"focus": 1, "generic": 11, "probing": 12, "telling": 0}`
- Hard-negative candidate: `{"focus": 9, "generic": 2, "probing": 13, "telling": 0}`

Sensitivity gain relative to baseline: **0.208333**. Hard-negative false-trigger change: **0.000000**.

## Primary test: recoverable after one scaffold

Baseline false telling: **0/4**. Candidate false telling: **0/4**.

| case_id | problem | formatted_history | expected_boundary | baseline_p_generic | baseline_p_probing | baseline_p_focus | baseline_p_telling | baseline_top1 | candidate_p_generic | candidate_p_probing | candidate_p_focus | candidate_p_telling | candidate_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I01 | Solve x + 8 = 20. | teacher: What would you undo first?<br>student: Add 8.<br>teacher: To reverse adding 8, which inverse operation could you use?<br>student: Maybe subtract 8? | non_telling | 0.546811 | 0.446812 | 0.003049 | 0.003327 | generic | 0.002158 | 0.001892 | 0.961616 | 0.034334 | focus |
| I02 | Calculate 3/10 as a decimal. | teacher: What decimal did you get?<br>student: 0.03<br>teacher: Tenths use the first place after the decimal. Reconsider where the 3 belongs.<br>student: Would it be 0.3? | non_telling | 0.020586 | 0.970818 | 0.005083 | 0.003514 | probing | 0.000217 | 0.000651 | 0.995574 | 0.003558 | focus |
| I03 | Simplify 5x + 2x. | teacher: What is the result?<br>student: 7x²<br>teacher: You are adding like terms, not multiplying x by x.<br>student: Then it might be 7x. | non_telling | 0.996410 | 0.002072 | 0.000460 | 0.001057 | generic | 0.000978 | 0.003748 | 0.968246 | 0.027028 | focus |
| I04 | Calculate (-5) × (-3). | teacher: What answer do you get?<br>student: -15<br>teacher: The signs are the same; revisit the sign rule.<br>student: Same signs make positive, so 15? | non_telling | 0.996709 | 0.002016 | 0.000342 | 0.000933 | generic | 0.717086 | 0.000468 | 0.182604 | 0.099842 | generic |

The full vectors and exact dialogue histories are also preserved in `recoverable_one_scaffold_cases.csv` and `important_cases.md`.

## Exact 87 real deployment states

- Candidate top-1 telling: **5/87 (5.747%)**; baseline **0/87**.
- Policy displacement: **20/87 (22.989%)**.
- Focus/probing/generic to telling: **2 / 3 / 0**.
- Telling to another move: **0**.
- Candidate-minus-baseline telling probability: mean **0.052146**, median **0.010344**, maximum increase **0.789545**.
- All transitions: `{"focus -> focus": 30, "focus -> generic": 2, "focus -> probing": 2, "focus -> telling": 2, "generic -> focus": 2, "generic -> generic": 19, "probing -> focus": 9, "probing -> probing": 18, "probing -> telling": 3}`.

Every real candidate-telling state, every hard-negative candidate-telling state, the four recoverable cases, the ten largest telling increases, and the ten largest focus decreases are shown with exact context and both vectors in `important_cases.md`.

## Semantic inspection

All five real candidate-telling rows were inspected. Three reflect repeated unresolved difficulty or sustained uncertainty after multiple meaningful scaffolds. Two are residual concerns because the latest learner response already shows correct progress: the warehouse learner has just recognized that division would give notebooks per box rather than the total, and the diver has just correctly computed the post-descent depth as -60. Neither is a recoverable-after-one-scaffold state. The candidate produces no top-1 telling prediction on any of the 24 hard negatives.

The ten largest telling-probability increases and ten largest focus-probability decreases were also inspected. On recent-progress rows from the rectangle and fraction attempts, the telling probability rises but top-1 remains probing, focus, or generic. Among the largest focus decreases, only the two diver rows cross to telling; other productive-progress rows remain non-telling. Thus the targeted frozen boundary is corrected, but the real-state sample still shows a narrower residual risk around multi-scaffold recovery. Historical outcomes do not establish what telling would have caused.

## Observational historical outcome join

Outcome fields were joined only after independent pre-action prediction. Candidate-telling real-state count: **5**; evaluator distribution `{"correct": 4, "incorrect": 1}`. These fields describe historical states only and support no causal claim about an alternative action.

## Comparison and interpretation

The old candidates all failed the recoverable boundary (2/4, 3/4, and 3/4) while displacing 17.241%, 22.989%, and 27.586% of real states. The C1 candidate is judged with the unchanged frozen rubric and the preregistered primary interpretation: 0/4 is strong correction evidence, 1/4 is meaningful improvement requiring inspection, and 2+/4 is inadequate correction.

**A. TARGETED RECALIBRATION SUCCESS**

The targeted recoverable-state correction passes while persistent-state sensitivity, hard-negative safety, and policy stability are retained. This remains an offline candidate diagnostic; no promotion is authorized.

## Safety audit

- Training: 0
- Production changes: 0
- Tutor API calls: 0
- BKT updates: 0
- LinTS updates: 0
- Authoritative-data writes: 0
- Protected MathDial test use: 0
- MRBench V3 test use: 0
- Protected input hash changes: 0

## Final comparison table

| metric | MD7-R1 | old tell epoch1 | old tell epoch2 | old tell epoch3 | MD7-R2-C1 epoch2 |
| --- | --- | --- | --- | --- | --- |
| MathDial Macro-F1 | 0.461129 | 0.425292 | 0.441921 | 0.449797 | 0.462572 |
| telling-plausible count /24 | 19/24 | 23/24 | 24/24 | 24/24 | 24/24 |
| hard-negative false telling /24 | 0/24 | 2/24 | 3/24 | 3/24 | 0/24 |
| recoverable false telling /4 | 0/4 | 2/4 | 3/4 | 3/4 | 0/4 |
| real-state telling /87 | 0/87 | 0/87 | 5/87 | 4/87 | 5/87 |
| real-state displacement /87 | 0/87 | 15/87 | 20/87 | 24/87 | 20/87 |
| displacement % | 0.000% | 17.241% | 22.989% | 27.586% | 22.989% |
| mean real-state telling-probability delta | 0.000000 | 0.024483 | 0.038394 | 0.020975 | 0.052146 |

DECISION: A

Recommended next experiment: Run one preregistered blinded expert review on a fresh untouched set of baseline-candidate disagreement states before any live experiment.
