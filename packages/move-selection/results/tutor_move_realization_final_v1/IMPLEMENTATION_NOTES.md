# Implementation notes

- The authoritative contract is centralized in `adaptive-math-tutor/backend/app/agents/tutor/realization_contract.py`.
- Tutor finalization receives the external move explicitly and a JSON schema restricts `selected_move` to that exact value.
- Learner-facing Tutor finalization has no response-verifier or inline-audit stage. A structurally valid teacher turn is returned directly after the selected-move and atomic-scope self-check fields pass.
- Failed structured-generation self-checks feed correction text into the bounded retry path.
- Generic repetition is handled through dialogue-aware language guidance; consecutive generic policy actions remain allowed.
- No model was retrained and no external Tutor API was called for this evidence packet.
