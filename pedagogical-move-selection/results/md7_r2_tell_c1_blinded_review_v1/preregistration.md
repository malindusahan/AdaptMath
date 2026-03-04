# MD7-R2-TELL-C1 Blinded Expert Review Preregistration

Status: **PREPARED BEFORE MODEL COMPARISON; EXPERT LABELS NOT YET COLLECTED**

## Scientific question

Does the recalibrated selector improve pedagogical-move selection relative to the frozen reference around difficult telling-boundary states without introducing inappropriate telling after learner recovery or productive progress?

## Frozen review set

- Rows: 48 fresh independently authored deployment-style synthetic states.
- Strata: eight fixed semantic strata with six states each: `{"A_first_clear_incorrect_answer": 6, "B_first_confusion_or_first_i_dont_know": 6, "C_recoverable_after_one_meaningful_scaffold": 6, "D_repeated_misconception_after_multiple_scaffolds": 6, "E_repeated_confusion_after_multiple_scaffolds": 6, "F_direct_explanation_request_after_previous_support": 6, "G_correct_progress_after_previous_difficulty": 6, "H_partial_recovering_progress_after_previous_difficulty": 6}`.
- SHA-256: `58466681a7e40724f6b4c3acea28ac26c548a771872b9ed72d938dbe84353309`.
- The prior 48-case diagnostic was not reused as a case source and the new set was not produced by one-by-one paraphrase.
- Exact state overlap against the prior 48, all 87 prior deployment states, the 2,400-row correction set, and packaged calibration train/validation/challenge artifacts: `{"c1_correction_train": 0, "calibration_challenge": 0, "calibration_train": 0, "calibration_validation": 0, "prior_deployment_87": 0, "prior_manual_48": 0}`.
- Maximum normalized sequence similarity to any prior manually authored 48-case state: `0.560000`; preregistered rejection boundary: `>= 0.80`.
- No expected move labels appear in the frozen review set.

## Fixed inference contract

Both systems are scored only after the set is frozen. Paired input is `Problem:
<problem>` and `Conversation:
<formatted dialogue>

Next teacher pedagogical move:`. Dialogue uses `user: text`; tokenizer truncation side is left; truncation is `only_second`; maximum length is 512; move order is `generic, probing, focus, telling`.

Reference weight SHA-256: `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13`. Recalibrated weight SHA-256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`.

## Blinding and packet selection

- Model identities are deterministically randomized to System X/Y with seed `2026082801` and stored only in `hidden_model_mapping.json`.
- Predictions and probabilities are stored only in `hidden_model_predictions.csv`.
- Human-facing files are `expert_review_instructions.md` and `expert_review_form.csv`; neither contains model identity, predictions, probabilities, strata, latest-state annotations, BKT, evaluator outcomes, MRB1, or historical outcomes.
- Include every disagreement. If there are fewer than 20 disagreements, add agreement states by deterministic stratum-round-robin sampling to reach 24, seed `2026082802`. If there are more than 32 disagreements, retain all because at most 48 is auditable.
- Reviewers choose exactly one move, confidence 1-3, and an optional one-sentence justification.

## Preregistered post-review metrics

For each actual model: exact agreement count/rate, four-class Macro-F1, per-class F1, telling precision/recall, and confusion counts. On model disagreements: recalibrated-only wins, reference-only wins, both wrong, decisive count, and a two-sided exact binomial test on recalibrated-vs-reference wins. Raw counts are primary; small-sample p-values will not be overstated.

Telling-specific reports are fixed for: expert-telling recall; false telling on expert-non-telling cases; G/H correct-or-recovering progress; C one-scaffold recovery; and D/E/F persistent/support-exhausted difficulty.

## Preregistered decision rule after expert labels

Integrity is checked first. Decision **D (REVIEW INCONCLUSIVE)** applies if the frozen hash fails, forms are incomplete, fewer than 8 decisive disagreements exist, median expert confidence is below 2, or the reviewed labels contain no telling or no non-telling case.

Otherwise, define a substantive G/H recovery overtrigger as either (i) at least 2 more false-telling cases for the recalibrated model than the reference among expert-non-telling G/H cases, or (ii) a false-telling-rate increase of at least 0.20 there. If this occurs, choose **B (RESIDUAL TELLING OVERCORRECTION)**.

If there is no substantive G/H overtrigger, choose **A (ADVANCE TO CONTROLLED LIVE VALIDATION)** only when the recalibrated model has strictly more disagreement wins than the reference and its telling precision is no more than 0.05 below the reference. Otherwise choose **C (TELLING BENEFIT NOT CONFIRMED)**.

No outcome authorizes production promotion.

## Safety boundary

Training 0; promotion 0; production changes 0; Tutor API calls 0; BKT updates 0; LinTS modifications/updates 0; authoritative real-data JSONL writes 0; MathDial final test use 0; MRBench V3 test use 0.
