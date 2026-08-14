# Turn-LinTS v1 architecture decision amendment: S+K+L

## Status

This is a transparent architecture-design amendment. The original experiment and its one-standard-error selection of **S** remain unchanged under `C:/Users/Lenovo/Documents/Research/Tutor Max/pedagogical-move-selection/results/turn_lints_final_architecture_selection_v1`.

## Amendment decision

AdaptMath Turn-LinTS v1 freezes **S+K+L** as its policy context with **qualified / limited support**. S+K+L achieved the best nominal grouped MAE/RMSE among tested contexts and provides explicit selector, knowledge-state, and learner-state representation. Its improvement over S was small and uncertainty intervals crossed zero, while substantive L coverage was only 3/84. Therefore the choice must not be interpreted as proof that L improves reward prediction.

## Empirical evidence

| Context | Dimension | MAE ± SE | RMSE ± SE | R² ± SE |
|---|---:|---:|---:|---:|
| S | 4 | 0.1649 ± 0.0279 | 0.2096 ± 0.0305 | -0.2517 ± 0.0472 |
| S+K | 7 | 0.1626 ± 0.0251 | 0.2052 ± 0.0285 | -0.2053 ± 0.0267 |
| S+K+L | 11 | 0.1616 ± 0.0259 | 0.2046 ± 0.0291 | -0.1949 ± 0.0298 |
| Full | 22 | 0.1680 ± 0.0244 | 0.2090 ± 0.0284 | -0.2570 ± 0.0536 |

Attempt-bootstrap S+K+L minus S: MAE mean -0.00312, 95% interval [-0.00913, +0.00319]; RMSE mean -0.00522, interval [-0.00941, +0.00031]. Both intervals cross zero; no statistically established superiority is claimed.

## Architecture rationale

- K is retained because S→S+K improved all nominal grouped metrics and adds current mastery plus previous mastery change. This is predictive, not causal, evidence.
- L is retained because it supplies the intended previous learner-response-state representation and avoids changing the policy vector after prospective collection begins. Its substantive coverage was only **3/84**, so L usefulness remains an unresolved empirical limitation.
- H is excluded: S→S+H changed MAE by approximately +0.00234, RMSE by +0.00233, and R² by −0.02929. H may remain diagnostic; no previous-move rule is introduced.
- Q is excluded: adding Q worsened matched MAE by approximately +0.00452 for S+K+H and +0.00418 for S+K+L+H. MRB1 remains auxiliary and is neither policy context nor scalar reward.
- Full context was nominally worse and doubles dimension from 11 to 22 without matched incremental evidence from H/Q.

## Reward

The original experiment-selected reward remains **HEADROOM_NORMALIZED**. It preserves BKT update sign, is finite and bounded [-1,1], uses no tuned α/β constants, and reduced `|ρ|` between reward magnitude and starting mastery from approximately 0.742 to 0.025. It is a **BKT posterior-belief update, not measured learning**.

## Decision boundary

This amendment freezes vector semantics and starts clean randomized collection. It does not fit or activate a LIVE posterior, establish L usefulness, or make causal action-effect claims.

**B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
