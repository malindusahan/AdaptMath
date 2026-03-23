# AdaptMath passive turn-outcome dataset v1

`turn_outcomes.jsonl` is the authoritative action-level dataset. Each row is
identified by `identity.action_event_id` and joins one Tutor action only to its
immediately following dialogue response and resolver/BKT result. Formal
three-question assessment evidence is written to `assessment_outcomes.jsonl`;
it is never attached to the last Tutor action. Cumulative attempt results are
written to `attempt_summaries.jsonl`. The human-readable, transaction-level BKT
audit stream is written to `bkt_activity.jsonl`; it includes both dialogue-turn
and formal-assessment mastery transitions using `adaptmath_bkt_activity_v2`.

The collector copies values already produced by MD7, learner-agency logic,
LinTS/C3, Tutor, frozen MRB1, the evaluator, resolver, and BKT. It does not run
any of those components and cannot select or modify a pedagogical move.

Privacy: raw student/account identifiers, credentials, age, and unrelated
profile fields are excluded. A deterministic dataset-scoped SHA-256 pseudonym
is retained. Problem text, selector input dialogue, the Tutor response, and
runtime thread/attempt identifiers are retained because they are necessary to
reconstruct the move-selection context. Mathematical dialogue is not
destructively redacted; researchers must therefore treat the JSONL files as
potentially identifying research data.

`learning_state_update.delta_mastery` means a confidence-weighted BKT posterior
belief update after resolved evidence. It is not a reward, a causal reward, a
learning effect caused by the move, or a true learning gain.

## Post-first-attempt provenance/privacy note

The first real attempt in this directory is preserved as immutable raw
observational evidence. Its pre-fix attempt summary contains the runtime-only
`adaptive_completion.policy_state_path`, including an absolute local user
profile path. Do not publish or copy that field into a derived research export.
Exporters must omit `policy_state_path`; post-fix collection retains portable
`provenance.policy` lineage, identifier, and repository-relative path fields
instead.

Post-fix attempt summaries explicitly store `attempt_started_at` and lifecycle
`completion_status`. Assessment and summary rows also carry standalone deployed
configuration provenance, including the explicit frozen move order.

Observed policy issue / future experiment candidate: the unchanged
`learner_agency_telling_escalation_v1` matcher did not classify phrases such as
"i cannot understand clearly" and "i dont know about that" as non-engagement.
This is documented for a separate learner-agency coverage experiment; the
active policy and its regular expressions were intentionally not changed.
