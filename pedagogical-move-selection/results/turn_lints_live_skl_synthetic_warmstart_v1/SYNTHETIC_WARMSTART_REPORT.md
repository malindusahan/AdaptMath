# Synthetic warm-start report

Decision: **A**.

Exactly 500 synthetic initialization observations were generated from bootstrapped real pre-action S and K states. L was deterministically recomputed from the preceding learner response with the frozen local detectors; first-turn L uses the implemented neutral values plus the missing indicator. Actions were sampled from the raw frozen MD7 vector. Rewards were produced by fixed seeded, arm-specific full-11D linear functions plus seeded noise and one global scaling operation to the historical HEADROOM_NORMALIZED scale.

Each synthetic observation has posterior weight 0.02, for an effective initialization sample size of 10. Real linked LIVE BKT posterior-belief updates have weight 1.0. Synthetic evidence is permanently labeled and is not described as real evidence.

The audit used 84 historical pre-action contexts without outcome labels to compare the uninformed prior, raw MD7 behavior, and the initialized posterior. Decision A requires every predeclared sanity check in `summary.json` to pass.
