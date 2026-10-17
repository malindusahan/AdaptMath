# Decision evidence

## REWARD DECISION

### QUESTION
Which BKT posterior-update transformation should be the Turn-LinTS scalar reward?

### CANDIDATES
RAW_DELTA and HEADROOM_NORMALIZED.

### DATA
84 scientifically valid linked turns from 14 attempts; corrupt/no-evidence updates and contaminated downstream states excluded.

### TEST
Predeclared sign, mastery-dependence, position, boundary, numerical-stability, and variation criteria. Spearman results are descriptive under clustering.

### RESULT
| reward | N | mean | SD | spearman_absolute_reward_vs_mastery_before_rho | spearman_absolute_reward_vs_turn_index_rho | nonfinite_count | sign_mismatch_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_DELTA | 84 | 0.0247 | 0.0926 | -0.7416 | -0.0242 | 0 | 0 |
| HEADROOM_NORMALIZED | 84 | 0.1474 | 0.2082 | -0.0247 | -0.0385 | 0 | 0 |

### VISUAL EVIDENCE
Figures 02-06.

### SELECTED OPTION
**HEADROOM_NORMALIZED**

### WHY IT WON
HEADROOM_NORMALIZED preserved sign, stayed finite and bounded without tuned constants, reduced absolute-reward mastery dependence from |rho|=0.742 to 0.025, and introduced no strong turn/relative-position artifact (maximum |rho|=0.155).

### WHAT THIS DOES NOT PROVE
It does not establish measured learning or a causal move effect.

### LIMITATIONS
BKT is a model posterior; observations cluster within attempts/learners.

## CONTEXT DECISION

### QUESTION
Which pre-action block preset has the best supported grouped generalization?

### CANDIDATES
S, S+K, S+L, S+H, S+Q, S+K+L, S+K+H, S+K+Q, S+K+L+H, S+K+L+H+Q, K+L+H+Q, S+L+H+Q, S+K+L+Q, S+K+H+Q.

### DATA
84 turns, 14 attempts, action support {'generic': 23, 'probing': 27, 'focus': 34, 'telling': 0}.

### TEST
Nested GroupKFold by attempt, fold-local standardization, inner-only ridge tuning on (0.01, 0.1, 1.0, 10.0, 100.0); one-standard-error parsimony rule.

### RESULT
| context | context_dimension | MAE_mean | MAE_SE | RMSE_mean | RMSE_SE | R2_mean | R2_SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S+K+L | 11 | 0.1616 | 0.0259 | 0.2046 | 0.0291 | -0.1949 | 0.0298 |
| S+K | 7 | 0.1626 | 0.0251 | 0.2052 | 0.0285 | -0.2053 | 0.0267 |
| S+K+L+H | 17 | 0.1638 | 0.0259 | 0.2074 | 0.0293 | -0.2290 | 0.0436 |
| S+L | 8 | 0.1640 | 0.0287 | 0.2091 | 0.0313 | -0.2417 | 0.0531 |
| S+K+H | 13 | 0.1649 | 0.0251 | 0.2078 | 0.0288 | -0.2371 | 0.0437 |
| S | 4 | 0.1649 | 0.0279 | 0.2096 | 0.0305 | -0.2517 | 0.0472 |
| S+K+L+Q | 16 | 0.1666 | 0.0244 | 0.2074 | 0.0283 | -0.2366 | 0.0425 |
| S+H | 10 | 0.1672 | 0.0280 | 0.2119 | 0.0307 | -0.2810 | 0.0606 |
| K+L+H+Q | 18 | 0.1676 | 0.0249 | 0.2089 | 0.0283 | -0.2565 | 0.0551 |
| S+K+L+H+Q | 22 | 0.1680 | 0.0244 | 0.2090 | 0.0284 | -0.2570 | 0.0536 |
| S+K+Q | 12 | 0.1680 | 0.0235 | 0.2087 | 0.0278 | -0.2561 | 0.0474 |
| S+K+H+Q | 18 | 0.1694 | 0.0235 | 0.2101 | 0.0280 | -0.2740 | 0.0586 |
| S+Q | 9 | 0.1709 | 0.0261 | 0.2130 | 0.0299 | -0.3010 | 0.0622 |
| S+L+H+Q | 19 | 0.1710 | 0.0271 | 0.2135 | 0.0308 | -0.3045 | 0.0733 |

### VISUAL EVIDENCE
Figures 07-13.

### SELECTED OPTION
**S**

### WHY IT WON
Lowest-MAE context was S+K+L at 0.1616; the one-SE threshold was 0.1876. S was the smallest eligible preset under the predeclared rule.

### WHAT THIS DOES NOT PROVE
It does not establish causal action heterogeneity or that sparsely observed blocks are useless.

### LIMITATIONS
- Only 3/84 included rows contain real non-missing L signals; missing-indicator representation permits evaluation but does not establish substantive L value.
- Telling has 0 included observations, so telling-specific reward interactions are not empirically established.
- Only 3 scientifically valid randomized observations from 1 attempt(s) exist; insufficient for standalone context selection.
- The primary context evidence is observational and policy-bound; no causal action-effect claim is made.
- Grouped R2 is sensitive to small heterogeneous held-out attempt groups and is interpreted with MAE/RMSE and fold stability.

## COLD-START DECISION

### QUESTION
How are treatments assigned before a learned posterior is justified?

### CANDIDATES
Previously frozen decision; not reopened here.

### DATA
8 logged randomized treatments, 3 scientifically valid after provenance/evidence audit.

### TEST
Lineage, propensity-vector, assignment-source, and zero-posterior-update audit.

### RESULT
Known-propensity raw MD7 probability-proportional assignment is implemented; standalone context selection is insufficient.

### VISUAL EVIDENCE
Figure 01 and randomized rows in the derived dataset.

### SELECTED OPTION
`md7_r2_probability_proportional_v1` (frozen prior decision).

### WHY IT WON
Not re-adjudicated; retained as the established cold-start contract.

### WHAT THIS DOES NOT PROVE
It does not justify LIVE posterior activation.

### LIMITATIONS
Scientifically valid randomized evidence is sparse.

## MRB1 ROLE

### QUESTION
Does previous MRB1 earn Q context inclusion?

### CANDIDATES
Matched with/without-Q presets; four heads remain separate.

### DATA
84 included rows with complete current four-head MRB1 outcomes.

### TEST
Matched grouped deltas, including S+K+H versus S+K+H+Q and S+K+L+H versus full.

### RESULT
See `context_block_incremental_value.csv`; selected context does not include Q.

### VISUAL EVIDENCE
Figures 10 and 12.

### SELECTED OPTION
Auxiliary critic/diagnostic only for v1.

### WHY IT WON
Generated by the same grouped one-SE context rule, not by MRB1 semantics.

### WHAT THIS DOES NOT PROVE
MRB1 is not a scalar reward or ground-truth learning measure.

### LIMITATIONS
MRB1 heads are correlated and observations are policy-bound.

## STUDENT-MODELING ROLE

### QUESTION
Does real previous learner-state L evidence support retention?

### CANDIDATES
Matched models with/without L under the exact runtime missing representation.

### DATA
Real L non-missing: 3/84 (3.6%).

### TEST
Grouped matched ablations; no future or synthetic probabilities.

### RESULT
L contribution is not established with available data.

### VISUAL EVIDENCE
Figures 12-13.

### SELECTED OPTION
Do not include L in the frozen v1 policy vector; keep diagnostic logging.

### WHY IT WON
The result follows grouped predictive evidence and one-SE parsimony.

### WHAT THIS DOES NOT PROVE
It does not prove learner signals are useless.

### LIMITATIONS
Substantive L coverage is sparse.
