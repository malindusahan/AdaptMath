# AdaptMath Turn-LinTS v1 final S+K+L architecture

- Architecture decision: **S+K+L**, 11 ordered pre-action features, qualified/limited support
- Algorithm: direct disjoint linear Thompson Sampling
- Arms: generic, probing, focus, telling
- Reward: HEADROOM_NORMALIZED BKT posterior-belief update
- Cold start: raw MD7 proportional randomized warm-start with exact propensity logging
- Diagnostic logging: full S+K+L+H+Q snapshot; H/Q never enter the frozen policy vector
- Explicit agency: above-policy, non-randomized override with null propensity
- No tau gate, filtering, P1/P2 anchor, or handwritten inferred-state action rules
- LIVE: disabled; warm-start posterior update count must remain zero

The original S-selection experiment remains unchanged. S+K+L was the nominal predictive winner, but its advantage over S was small and uncertain and L coverage was only 3/84. L usefulness remains unresolved.

Manifest SHA-256: `a860d385797eaaced4788463fb789197c614bef7d7934f8006705276fdb57dfa`

**B — S+K+L FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
