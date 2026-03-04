# Turn-LinTS final architecture selection v1

## Outcome

Reward **HEADROOM_NORMALIZED** and context **S** (dimension 4) are frozen with qualified/limited support. This is an architecture freeze, not LIVE activation.

## Data integrity

- Candidate Tutor treatment turns: 109
- Valid linked BKT outcomes before scientific exclusions: 106
- Included reward/context turns: 84/84
- Attempts/learners/skills: 18/3/14
- Exclusions: {"downstream_mastery_state_contaminated_by:00efa9aa-9749-4384-bc1c-a5c7d17012c8:1:action:4": 3, "downstream_mastery_state_contaminated_by:4220e000-557f-487a-949f-a6b3662998ac:1:action:1": 6, "downstream_mastery_state_contaminated_by:44442f3f-ef9b-4c06-8cd8-0c1eba70feef:1:action:1": 2, "downstream_mastery_state_contaminated_by:a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:3": 4, "downstream_mastery_state_contaminated_by:eef1d04e-fe55-4457-9edf-f1c05786fd69:1:action:4": 2, "interaction_management_acknowledgement": 1, "interaction_management_request": 1, "no_knowledge_evidence_metacognitive_or_social": 1, "no_knowledge_evidence_problem_repetition": 1, "no_knowledge_evidence_transcript_echo": 1, "no_scientifically_resolved_reward:censored": 3}
- Current runtime at analysis: SHADOW, context S+K+L+H+Q, reward raw_delta

Pure acknowledgement, interaction-management, echo, and problem-repetition behavioural-proxy updates were excluded, together with downstream rows whose mastery state was contaminated. Authoritative logs were not changed.

## Reward

| reward | N | min | Q1 | median | mean | Q3 | max | SD | spearman_absolute_reward_vs_mastery_before_rho | spearman_absolute_reward_vs_turn_index_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_DELTA | 84 | -0.1711 | -0.0197 | 0.0227 | 0.0247 | 0.0658 | 0.5565 | 0.0926 | -0.7416 | -0.0242 |
| HEADROOM_NORMALIZED | 84 | -0.2622 | -0.0237 | 0.1776 | 0.1474 | 0.2393 | 0.7005 | 0.2082 | -0.0247 | -0.0385 |

HEADROOM_NORMALIZED preserved sign, stayed finite and bounded without tuned constants, reduced absolute-reward mastery dependence from |rho|=0.742 to 0.025, and introduced no strong turn/relative-position artifact (maximum |rho|=0.155).

## Context

| context | context_dimension | MAE_mean | MAE_SE | RMSE_mean | RMSE_SE | R2_mean | R2_SE | OOF_MAE | OOF_RMSE | OOF_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S+K+L | 11 | 0.1616 | 0.0259 | 0.2046 | 0.0291 | -0.1949 | 0.0298 | 0.1598 | 0.2109 | -0.0385 |
| S+K | 7 | 0.1626 | 0.0251 | 0.2052 | 0.0285 | -0.2053 | 0.0267 | 0.1608 | 0.2112 | -0.0414 |
| S+K+L+H | 17 | 0.1638 | 0.0259 | 0.2074 | 0.0293 | -0.2290 | 0.0436 | 0.1620 | 0.2137 | -0.0664 |
| S+L | 8 | 0.1640 | 0.0287 | 0.2091 | 0.0313 | -0.2417 | 0.0531 | 0.1621 | 0.2163 | -0.0924 |
| S+K+H | 13 | 0.1649 | 0.0251 | 0.2078 | 0.0288 | -0.2371 | 0.0437 | 0.1631 | 0.2139 | -0.0684 |
| S | 4 | 0.1649 | 0.0279 | 0.2096 | 0.0305 | -0.2517 | 0.0472 | 0.1630 | 0.2164 | -0.0937 |
| S+K+L+Q | 16 | 0.1666 | 0.0244 | 0.2074 | 0.0283 | -0.2366 | 0.0425 | 0.1648 | 0.2132 | -0.0618 |
| S+H | 10 | 0.1672 | 0.0280 | 0.2119 | 0.0307 | -0.2810 | 0.0606 | 0.1653 | 0.2188 | -0.1180 |
| K+L+H+Q | 18 | 0.1676 | 0.0249 | 0.2089 | 0.0283 | -0.2565 | 0.0551 | 0.1658 | 0.2147 | -0.0767 |
| S+K+L+H+Q | 22 | 0.1680 | 0.0244 | 0.2090 | 0.0284 | -0.2570 | 0.0536 | 0.1663 | 0.2149 | -0.0787 |
| S+K+Q | 12 | 0.1680 | 0.0235 | 0.2087 | 0.0278 | -0.2561 | 0.0474 | 0.1663 | 0.2142 | -0.0716 |
| S+K+H+Q | 18 | 0.1694 | 0.0235 | 0.2101 | 0.0280 | -0.2740 | 0.0586 | 0.1678 | 0.2158 | -0.0872 |
| S+Q | 9 | 0.1709 | 0.0261 | 0.2130 | 0.0299 | -0.3010 | 0.0622 | 0.1692 | 0.2194 | -0.1244 |
| S+L+H+Q | 19 | 0.1710 | 0.0271 | 0.2135 | 0.0308 | -0.3045 | 0.0733 | 0.1692 | 0.2204 | -0.1344 |

The selected preset is **S** with ordered features ['selector_p_generic', 'selector_p_probing', 'selector_p_focus', 'selector_p_telling']. Relative to S, delta MAE=+0.0000 and delta RMSE=+0.0000. Relative to full context, delta MAE=-0.0031 and delta RMSE=+0.0005.

## Randomized versus observational evidence

Randomized warm-start treatments: 8; scientifically valid randomized observations: 3 from 1 attempt(s). This is **insufficient for standalone context selection**. Primary results are predictive observational evidence and do not identify causal action effects.

## Safety

Neural training=0; protected MathDial final test use=0; MRBench V3 official test use=0; authoritative historical rewrites=0; external Tutor API calls=0; LIVE posterior activations=0.

## Limitations

- Only 3/84 included rows contain real non-missing L signals; missing-indicator representation permits evaluation but does not establish substantive L value.
- Telling has 0 included observations, so telling-specific reward interactions are not empirically established.
- Only 3 scientifically valid randomized observations from 1 attempt(s) exist; insufficient for standalone context selection.
- The primary context evidence is observational and policy-bound; no causal action-effect claim is made.
- Grouped R2 is sensitive to small heterogeneous held-out attempt groups and is interpreted with MAE/RMSE and fold stability.

**B. FINAL ARCHITECTURE FROZEN WITH QUALIFIED / LIMITED SUPPORT**
