"""
Integration boundary for Omash's self-improving pedagogical-move selector.

The real teammate API contract is not present in this repository, so this file
intentionally does not invent an HTTP URL, authentication scheme, payload, or
response shape.

Final ownership:
- Omash selects one MathDial move for EACH teacher turn from student profile,
  conversation history up to that point, and the current mathematics problem;
- supported labels are telling, focus, generic, probing;
- this tutoring component consumes the selected move and never chooses a move
  itself.

Temporary integration behaviour until Omash's API is connected:
- the learner-facing TutorStartRequest does not carry a pedagogical move;
- after problem analysis, and again after each continuing student turn, the
  workflow interrupts with pedagogical_move_required;
- POST /tutor/{thread_id}/move is the temporary system-to-system integration
  seam through which Omash's externally selected move is supplied.

The frontend should not ask the learner to choose these moves. The /move
endpoint is an integration seam for the teammate component during development.
"""
