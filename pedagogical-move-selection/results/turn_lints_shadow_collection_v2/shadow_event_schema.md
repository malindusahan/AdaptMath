# Turn-LinTS SHADOW event schema v2

This is a derived instrumentation lineage. It does not rewrite the authoritative
passive collector. One completed JSON object is appended to `turn_events.jsonl`
per Tutor action; unresolved actions remain as idempotent files under
`pending_actions/`.

## Action-time fields

- Identity and lineage: `schema_version`, `collection_lineage`,
  `policy_lineage`, `policy_mode`, `attempt_id`, `action_event_id`, and
  `action_turn_index`.
- Frozen selector provenance: `selector_version`, `selector_sha256`, and
  `md7_raw_probabilities` for generic, probing, focus, and telling.
- Treatment separation: `actual_move`, `actual_selector_move`,
  `actual_treatment_move`, `selected_arm`, `hypothetical_arm`,
  `shadow_hypothetical_move`, `hypothetical_is_observed_treatment=false`, and
  `reward_attribution=actual_treatment_move`.
- Reward configuration: `reward_mode` and `configured_reward_mode`.
- Context envelope: `context.schema_version`, `context.schema_id`,
  `context.enabled_blocks`, `context.available_blocks`,
  `context.feature_names`, `context.context_dimension`, `context.vector`,
  `context.feature_values`, and `context.missingness`.

For the full S+K+L+H+Q instrumentation context, the ordered 22 features are:

1. `selector_p_generic`
2. `selector_p_probing`
3. `selector_p_focus`
4. `selector_p_telling`
5. `mastery_before`
6. `previous_mastery_delta`
7. `previous_mastery_delta_missing`
8. `previous_reasoning_probability`
9. `previous_uncertainty_probability`
10. `previous_clarification_probability`
11. `previous_learner_signals_missing`
12. `previous_move_generic`
13. `previous_move_probing`
14. `previous_move_focus`
15. `previous_move_telling`
16. `previous_move_missing`
17. `prior_tutor_turn_count`
18. `previous_mistake_identification`
19. `previous_mistake_location`
20. `previous_providing_guidance`
21. `previous_actionability`
22. `previous_mrb1_missing`

The named map makes every S/K/L/H/Q component recoverable without relying on
positional inference. Missing values retain explicit missing indicators.

## Response- and outcome-time fields

Once permitted derived response data exist, the same action receives
`tutor_response` and `current_response_mrb1` (the four MRB1 heads). Resolution
adds `resolver_event_id`, `mastery_before`, `mastery_after`, `raw_delta`,
`headroom_normalized_delta`, `configured_reward_mode`,
`configured_reward_value`, `reward_attributed_to_move`,
`reward_attributed_to_shadow_hypothetical=false`,
`learner_signals_for_next_action`, `outcome_observed`, `resolution_status`,
`posterior_update_occurred`, `update_id`, and `update_timestamp`.

`resolution_status` is `observed_bkt_update`, `no_bkt_update`,
`processing_failed`, or `censored_no_response`. The configured scalar is
clearly separate from the two retained reward candidates. There is only one
possible policy update, and SHADOW never takes it.

Learner and skill identifiers are not duplicated into this derived log. The
readiness script links them by `action_event_id` from the existing passive
lineage when available.

## Temporal-order proof

```text
Tutor response t-1 --> MRB1 t-1 ---------------------------> Q_t
Learner response t-1 --> learner-state resolver ----------> L_t
Prior completed turns ------------------------------------> H_t
BKT state before action t --------------------------------> K_t
Frozen MD7 probabilities at action t ---------------------> S_t
                                                              |
                    x_t = [S_t, K_t, L_t, H_t, Q_t] <---------+
                                      |
                                      v
                         MD7 actual action + SHADOW sample
                                      |
                                      v
                            Tutor response t --> MRB1 t
                                      |
                                      v
                  Learner response t --> BKT mastery_after_t
                                      |
                                      +--> inputs for action t+1 only
```

Thus learner response t, MRB1 t, and mastery-after t do not exist when x_t is
built. L_t comes only from learner response t-1; Q_t comes only from Tutor
response t-1.
