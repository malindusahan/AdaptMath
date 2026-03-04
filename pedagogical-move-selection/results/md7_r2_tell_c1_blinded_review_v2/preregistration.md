# Preregistration: MD7-R2-TELL-C1 Blinded Review Extension 2

This study is independent of Review 1. Review 1 remains D (inconclusive), with 3 candidate wins, 0 baseline wins, 2 neither, and 3 decisive disagreements. Its expert labels are not used to author, select, or judge Review 2 states.

## Frozen models

- Reference weight SHA-256: `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13`
- Recalibrated weight SHA-256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- Move order: `generic`, `probing`, `focus`, `telling`

## Prospectively frozen pool

- Rows: 128
- SHA-256: `0d3834b6aaaee16606e6bbd7c98f2fa3771cad0f5bfc43ccde833f5b7ae9376b`
- Maximum excluded-material similarity: `0.637362637363`
- Rejection rule: reject any state with normalized SequenceMatcher similarity >= `0.80` before inference.

- `A_first_clear_incorrect_answer`: 16
- `B_first_confusion_or_first_i_dont_know`: 16
- `C_recoverable_after_one_meaningful_scaffold`: 16
- `D_repeated_misconception_after_multiple_scaffolds`: 16
- `E_repeated_confusion_after_multiple_scaffolds`: 16
- `F_direct_explanation_request_after_previous_support`: 16
- `G_correct_progress_after_previous_difficulty`: 16
- `H_partial_recovering_progress_after_previous_difficulty`: 16

The entire pool is authored, audited, written, and hashed before either model is loaded. No state may be edited after model predictions are observed.

## Packet rule

All and only new reference-versus-recalibrated top-1 disagreements enter the primary packet. No disagreement may be cherry-picked. Agreement padding is not used. If fewer than 8 model disagreements occur, preparation stops with insufficient disagreement support and no model conclusion.

## Blinding

Reviewers see only opaque case IDs, problems, dialogue histories, fixed move definitions, confidence 1-3, and an optional one-sentence reason. They do not see model identities, System X/Y, probabilities, strata, BKT, MRB1, evaluator outcomes, historical outcomes, or Review 1 labels. A deterministic seed `2026082902` seals model identity as System X/Y in a separate file.

## Primary endpoint and stopping rule

Primary analysis is restricted to new model-disagreement cases. Each is classified as candidate win, baseline win, or neither. Decisive count is candidate wins plus baseline wins. If decisive count is below 8, Review 2 is D: REVIEW STILL INCONCLUSIVE. If it is at least 8, raw wins and an exact two-sided binomial test are reported.

## Secondary endpoints

Within the reviewed extension packet: exact agreement, Macro-F1, per-class F1, telling precision/recall, false telling among expert non-telling cases, and telling counts in C, G/H, and D/E/F. Every C/G/H disagreement is printed after review with problem, history, expert move/confidence, and both model moves.

## Operational decision rule

1. D if decisive new disagreements < 8 or an integrity check fails.
2. B if, among expert-non-telling C/G/H cases, the recalibrated model has at least 2 more false-telling predictions than the reference or a false-telling-rate increase of at least 0.20.
3. A only if decisive count >= 8, recalibrated wins exceed reference wins, recalibrated telling recall in expert-telling D/E/F cases is at least reference recall, and rule 2 is false.
4. Otherwise C.

The formal decision depends on Review 2 alone. Only after that decision is frozen, an explicitly exploratory cumulative table adds Review 1's 3 candidate wins and 0 baseline wins. No outcome authorizes promotion.

## Safety

Training = 0; parameter changes = 0; production changes = 0; Tutor API calls = 0; BKT/LinTS updates = 0; authoritative-data writes = 0; MathDial final-test use = 0; MRBench V3 test use = 0; promotion = 0.
