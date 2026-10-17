# AdaptMath Turn-LinTS v1 FINAL ARCHITECTURE

- Frozen base selector: MD7-R2-TELL-C1 Epoch 2, SHA-256 `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- Bandit: direct disjoint linear Thompson Sampling
- Arms: generic, probing, focus, telling
- Final context: **S**, dimension **4**
- Ordered features: `selector_p_generic`, `selector_p_probing`, `selector_p_focus`, `selector_p_telling`
- Final reward: **HEADROOM_NORMALIZED** (`headroom_normalized`)
- Cold start: raw MD7 proportional randomized warm-start (`md7_r2_probability_proportional_v1`), known propensities logged
- P1 anchor: inactive; P2: not used; tau gate: none
- Handwritten pedagogical action rules/action filtering: none
- MRB1: auxiliary diagnostic only; never scalar reward
- Formal assessment: separate external policy-health signal
- Learner agency: explicit interaction requests are above-policy and non-randomized when overriding; inferred learner state does not trigger action rules
- Turn update: immediate after valid linked BKT evidence in future LIVE mode
- LIVE activation: not authorized by this freeze; current direct posterior updates remain 0

Support is qualified because action support is {'generic': 23, 'probing': 27, 'focus': 34, 'telling': 0}, real L coverage is 3/84, and randomized-only evidence is insufficient for standalone selection.

Manifest SHA-256: `b90b8efc87079e59ab88e041c5e8e4c9a6150350d6847a1976367311996834a0`

**B. FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
