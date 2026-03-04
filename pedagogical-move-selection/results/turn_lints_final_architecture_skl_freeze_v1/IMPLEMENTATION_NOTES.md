# Implementation notes

- Frozen constants are defined in `src/self_improvement/turn_lints_architecture_v1.py`.
- Policy construction uses exactly S+K+L and 11 ordered values.
- Runtime event records contain separate `policy_context` and `diagnostic_logged_context` objects. The diagnostic object uses the existing causal builder and contains previous H/Q only.
- Missing L is represented as three zeros plus `previous_learner_signals_missing=1`; real all-zero signals use indicator 0.
- Pure acknowledgement/interaction-management evidence is categorized semantically and cannot trigger behavioural-proxy BKT updates.
- Explicit agency uses decision/treatment source `learner_agency`, `randomized_assignment=false`, and `behavior_propensity=null`.
- The safe default remains SHADOW. `start-adaptmath-local.ps1 -FinalSKLWarmstart` explicitly selects the final randomized-warm-start lineage.
- The S-only and prior drifted state files are not migrated. The S+K+L lineage starts from fresh 11×11 identity A matrices, zero b vectors, and zero updates.
- No LIVE posterior was fitted or activated.
