# Implementation notes

- The authoritative contract is centralized in `adaptive-math-tutor/backend/app/agents/tutor/realization_contract.py`.
- Tutor finalization receives the external move explicitly and a JSON schema restricts `selected_move` to that exact value.
- The verifier receives the same selected-move profile and checks semantic alignment, mathematical correctness, and atomic scope.
- Failed generation self-checks or verifier audits feed correction text into the existing bounded retry path.
- Generic repetition is handled through dialogue-aware language guidance; consecutive generic policy actions remain allowed.
- No model was retrained and no external Tutor API was called for this evidence packet.
