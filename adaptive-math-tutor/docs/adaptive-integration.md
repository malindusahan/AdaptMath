# Adaptive subsystem integration

The production Tutor graph delegates every teacher turn to the frozen-v3
`AdaptiveTutorPipeline`. `TutorState.conversation_history` is the only live
dialogue store; `TutorStateMemoryAdapter` projects it for MD6/MRB1 and owns the
single completed teacher-response append.

`thread_id` identifies one complete Tutor problem/session. `attempt_id`
identifies one assessment cycle and is derived as
`<thread_id>:<adaptive_attempt_index>`. A failed assessment completes the old
student-model/frozen-v3 attempt and starts the next index before reteaching.

Cross-session Memory is optional and isolated at two graph boundaries. Tutor
start performs one bounded, skill-scoped, service-authenticated retrieval after
the BKT skill is validated. Only structured misconceptions and objective
assessment summaries are admitted; current-session dialogue remains solely in
`TutorState.conversation_history`. After adaptive completion, `memory_update`
writes the captured `completed_attempt_id`. Transient failures are checkpointed
in `memory_pending_writes` and retried with the same stable IDs at the next
Memory write boundary without rerunning BKT, LinTS, or attempt completion.

Clients must submit an exact `target_skill` from the trained BKT vocabulary.
Topic and subtopic are never treated as skill aliases.

## Current safety limits

- Current research runtime supports one active adaptive attempt per backend
  process. Distinct Tutor sessions are isolated, but adaptive episodes execute
  sequentially.
- Only one adaptive attempt may be open in a backend process. Frozen
  `TurnLevelAttemptController` instances snapshot the shared policy update
  count at start; completing either of two overlapping attempts would
  invalidate the other's guard. The coordinator rejects overlap instead of
  creating per-student policies or merging posterior state.
- Active controller/bridge objects are process memory while Tutor dialogue is
  persisted by LangGraph. After a backend restart, an unfinished persisted
  attempt fails with an explicit recovery error. It is never reconstructed,
  silently restarted, or completed twice.
- Policy state and experience logs default to the Tutor backend `runtime`
  directory and can be redirected with `ADAPTIVE_POLICY_STATE_PATH` and
  `ADAPTIVE_EXPERIENCE_LOG_PATH`. `ADAPTIVE_DATA_MODE=synthetic` creates a
  lineage-isolated local-demo policy and log; it must never seed real research
  state. Tests inject synthetic components and use temporary paths only.

These limits preserve frozen policy and BKT semantics. Supporting concurrent
open attempts or full process-restart recovery requires a new public research
state contract, not an orchestration-only change.
