# Tutor move-realization implementation notes

## Finding

The selected move already survived the selector-to-Tutor path and was recorded with the Tutor response. No response postprocessor appended a question. The question-heavy behavior came from the learner-facing generation and verifier prompts: generic was defined as an invitation to reveal thinking, and the passing generic/focus examples were questions. Those instructions made several moves converge on probing-like language.

## Source changes

- `adaptive-math-tutor/backend/app/agents/tutor/tutor_agent.py`
  - defines four distinct primary speech-act contracts;
  - injects the selected move and its private contract into every finalizer request;
  - removes any global question expectation;
  - makes generic/focus examples declarative and telling explanation-first;
  - retries if `teacher_message` exposes a raw internal move prefix.
- `adaptive-math-tutor/backend/app/agents/tutor/response_verifier.py`
  - audits semantic move alignment using the same distinctions;
  - explicitly permits declarative generic/focus turns and explanation-first telling;
  - rejects learner-facing raw move prefixes.
- Focused prompt/mocked-generation tests cover distinct definitions, selected-move provenance, telling/probing/focus/generic requirements, absence of a universal question suffix, and raw-label suppression.

## Preserved behavior

- The selector and selected move are unchanged.
- All four moves remain available; no state-to-action eligibility rules were added.
- Atomic-turn, mathematical verification, learner-agency escalation, and formal-assessment ownership remain intact.
- Internal selected-move provenance remains available through `pedagogical_move`, `adaptive_turn_diagnostics`, conversation history metadata, and the passive turn collector. Only the natural `teacher_message` is learner-visible.

## Runtime validation boundary

Automated validation is prompt construction and mocked generation only; no external LLM was called. The manual cases in this directory are intended for later human review of real runtime outputs.

## Final status

**REALIZATION STATUS: A. MOVE REALIZATION CONTRACT IMPLEMENTED**
