# Prospective Turn-LinTS context-ablation preregistration

## Status and trigger

This plan is frozen before adequate prospective SHADOW data exist. It must not
be executed merely because time has passed. First run the existing
`turn_lints_shadow_collection_v2/shadow_data_readiness.py` and report its raw
evidence. No automatic sample-size or winning-context threshold is encoded.

The historical result remains inconclusive, chiefly because only 1/96 usable
reward rows had a complete L block. The purpose of ongoing S+K+L+H+Q
instrumentation is to remedy that limitation prospectively.

## Eligible observations

Include only normal future SHADOW events that have:

- a uniquely linked actual MD7 treatment action and learner outcome;
- a valid HEADROOM_NORMALIZED reward derived from that actual action;
- a context snapshot created before action selection;
- the required candidate-block fields and explicit missingness indicators; and
- an attempt identity suitable for grouped validation.

Never attach the actual outcome to `shadow_p0_move`, `shadow_p1_move`, or any
other hypothetical action. Do not force telling or manufacture learner
sessions for coverage. Report natural actual counts for generic, probing,
focus, and telling separately from hypothetical counts.

## Frozen candidates

Primary candidates:

1. S
2. S+K
3. S+L
4. S+K+L
5. S+K+L+H
6. S+K+L+H+Q
7. K+L+H+Q

If prospective sample size and actual-arm coverage later support it, also run
leave-one-block-out variants of the full instrumentation context. No candidate
is designated to win in advance. The external MD7 action anchor, if a later
experiment approves one, must be held identical across context variants and
must not require S.

## Model and validation

- Model family: the same simple regularized arm-conditional linear
  reward-prediction family used in the earlier audit.
- Primary outcome: HEADROOM_NORMALIZED BKT belief-update reward.
- Secondary robustness outcome: R_RAW.
- Grouping unit: attempt; no turns from one attempt may cross training and
  validation partitions.
- Preprocessing and any regularization choice must be fit using training groups
  only.
- Report grouped out-of-sample R-squared, MAE, fold dispersion, usable rows,
  missing-block coverage, and actual treatment counts by arm.
- Preserve explicit missing indicators. Do not impute from future turns.
- Compare candidates on identical eligible rows where a paired comparison is
  claimed; also report each candidate's maximum usable cohort transparently.

Do not interpret predictive improvement as a causal treatment effect. Do not
select a final context when grouped performance is unstable, actual-arm
coverage is inadequate, or results depend on a few attempts.

## MRB1 handling

Q remains the previous Tutor response's four MRB1 heads and is a candidate
pre-action context block. Current-turn MRB1 remains an auxiliary response-
quality outcome. Neither is combined with the scalar reward. The absence of a
previous Q benefit is not evidence that Q is useless.

## Reporting

Report data lineage, code/configuration hashes, grouped splits, exact candidate
features, reward provenance, actual versus hypothetical action coverage, and
all exclusions. Preserve the conclusion `INCONCLUSIVE` unless prospective
complete-data evidence supports a reproducible model-specification decision.

