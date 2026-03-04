# Reward decision

## Predeclared criteria

1. Preserve the sign/direction of the BKT posterior belief update.
2. Avoid mechanical domination by starting mastery/headroom.
3. Avoid a strong artificial dependence on turn or relative attempt position.
4. Remain finite and stable at mastery boundaries.
5. Use no arbitrary clipping or tuned weighting constants.
6. Retain usable contextual-bandit reward variation.

## Results

| reward | N | min | P1 | P5 | Q1 | median | mean | Q3 | P95 | P99 | max | SD | IQR | positive_proportion | zero_proportion | negative_proportion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_DELTA | 84 | -0.1711 | -0.1577 | -0.1316 | -0.0197 | 0.0227 | 0.0247 | 0.0658 | 0.1269 | 0.2692 | 0.5565 | 0.0926 | 0.0855 | 0.6786 | 0.0000 | 0.3214 |
| HEADROOM_NORMALIZED | 84 | -0.2622 | -0.2318 | -0.1699 | -0.0237 | 0.1776 | 0.1474 | 0.2393 | 0.5400 | 0.6924 | 0.7005 | 0.2082 | 0.2630 | 0.6786 | 0.0000 | 0.3214 |

## Selected option

**HEADROOM_NORMALIZED**

HEADROOM_NORMALIZED preserved sign, stayed finite and bounded without tuned constants, reduced absolute-reward mastery dependence from |rho|=0.742 to 0.025, and introduced no strong turn/relative-position artifact (maximum |rho|=0.155).

The reward is a BKT posterior belief update, not measured learning. Observations are clustered within attempts and learners; correlations are descriptive, not independent causal evidence.
