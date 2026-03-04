import json
from typing import Any

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.realization_contract import build_move_audit_instruction


TEACHING_RESPONSE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkable_claims": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "source_text": {"type": "string"},
                    "left_expression": {"type": "string"},
                    "right_expression": {"type": "string"},
                },
                "required": [
                    "source_text",
                    "left_expression",
                    "right_expression",
                ],
                "additionalProperties": False,
            },
        },
        "logical_issues": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "source_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["source_text", "reason"],
                "additionalProperties": False,
            },
        },
        "turn_scope_issues": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "source_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["source_text", "reason"],
                "additionalProperties": False,
            },
        },
        "turn_scope_verdict": {
            "type": "string",
            "enum": ["pass", "fail"],
        },
        "move_alignment_verdict": {
            "type": "string",
            "enum": ["pass", "fail"],
        },
        "scope_reason": {"type": "string"},
        "move_alignment_reason": {"type": "string"},
    },
    "required": [
        "checkable_claims",
        "logical_issues",
        "turn_scope_issues",
        "turn_scope_verdict",
        "move_alignment_verdict",
        "scope_reason",
        "move_alignment_reason",
    ],
    "additionalProperties": False,
}


class TeachingResponseVerifier:
    """
    Post-generation mathematical audit for learner-facing Tutor responses.

    GenAI extracts claims that can be represented symbolically and identifies
    clear internal/unsupported logical claims. SymPy then verifies the
    extracted arithmetic and symbolic equalities deterministically.

    This component does not choose pedagogy or routing. It is a correctness
    guardrail around already-generated Tutor text.
    """

    audit_prompt = """
You audit a learner-facing mathematics explanation before it is shown.

Return JSON matching the required schema. Do not rewrite the lesson.

TASK 1 - EXTRACT CHECKABLE MATHEMATICAL CLAIMS
Extract every important arithmetic or symbolic equality/equivalence that is
actually stated in the teaching response and can be checked with ordinary
algebra. Include intermediate calculations, substitutions, expansions,
factorisations, equation transformations, and final numerical/symbolic claims.

Convert expressions to safe Python/SymPy syntax:
- use ** for powers, never ^,
- use * for multiplication,
- use / for division,
- expression fields contain expressions only, never explanatory prose.

Examples of extraction format only:
- text "20 - 5 = 15" -> left_expression "20-5", right_expression "15"
- text "(x-2)(x+2) = x^2-4" -> left_expression "(x-2)*(x+2)",
  right_expression "x**2-4"

Do not extract a conditional/contextual equation merely because the teacher
restates or refers to it. For example, after a learner establishes c^2 = 225,
the teacher's reference to "c^2 = 225" is not a claim that c^2 and 225 are
universally equivalent. Check its support against the supplied context under
TASK 2 instead. Any contradiction with verified evidence belongs in
logical_issues.

Do not invent a claim that the response does not make. Do not extract a claim
if it cannot be represented reliably in the expression syntax above.

TASK 2 - FIND CLEAR LOGICAL/CONSISTENCY PROBLEMS
Flag only clear mathematical problems in prose that symbolic equality alone
cannot capture, especially:
- a statement contradicting another part of the same response,
- unsupported "only", "unique", "all", "no other", "must", or exhaustive
  claims,
- a justification that does not logically follow from its stated premise,
- a claim contradicted by the supplied verified mathematical evidence.

Do not flag style preferences. Do not demand a particular solution method.
Do not mark a statement as wrong merely because it is concise.

If no such issue exists, return an empty logical_issues array.

TASK 3 - AUDIT THE SCOPE OF THIS SINGLE TEACHER TURN
The Tutor is part of an interactive MathDial-style dialogue. Omash's component
has already selected the pedagogical move. You must NOT choose a different
move. Decide explicitly whether the response is one atomic teacher action or
whether it leaks future teaching work that should occur only after another
student response and another Omash selection.

Set turn_scope_verdict to "fail" whenever the draft does ANY of the following:
- asks the learner for the complete solution or complete multi-step plan;
- asks for several independent things at once, such as identifying numbers AND
  choosing operations AND explaining all steps;
- names or enumerates several future operations, transformations, or
  calculations before the learner has responded between them;
- supplies a sequence such as "multiply, then subtract, then divide";
- asks several sequential mathematical questions that should be separate
  teacher-student turns;
- exposes enough of the future solution path that Omash loses the opportunity
  to adapt after the learner's next response;
- combines more than one pedagogical action in the same turn.

The following is a FAILING generic turn:
"What numbers would you write down, and what operations like multiplication,
subtraction, and division might you need to use? Explain the steps you would
take."
It fails because it requests multiple learner tasks, names several future
operations, and asks for the whole solution plan.

The following is a PASSING generic turn:
"How would you start solving this problem?"
It asks for one broad learner contribution and then returns control.

Move-specific scope rules:
- generic: one neutral acknowledgement, encouragement, transition, or broadly
  supportive, context-aware statement. It must be useful rather than merely
  "Great" or "Okay", must not repeat stock acknowledgement from recent Tutor
  turns, target a specific gap, teach a method, or default to eliciting
  reasoning. A question is optional, never required.
- probing: one clear, specific diagnostic/reasoning question that elicits the
  learner's thinking without first supplying or answering the missing step.
  Fail a worked solution followed by a token question.
- focus: one targeted cue, observation, or directive pointing attention to a
  specific error, relationship, clue, representation, or sub-step. It should
  not be rejected merely because it is declarative and has no question. Fail
  vacuous praise and fail a complete worked explanation.
- telling: one needed fact, explanation, representation, method, or worked step
  supplied explicitly, with instruction/explanation before any optional
  follow-up. Fail a response that only asks the learner what to do. If the
  latest learner message explicitly requests the answer, or the latest two
  learner messages both communicate non-engagement, a short one-step problem
  may instead receive its verified answer directly with one brief clarification.
  Do not reject that direct answer merely because it contains no follow-up
  question.

A short explanation followed by one question can pass when both address the
same immediate target. Do not count punctuation mechanically; judge semantic
scope.

There is no universal requirement to ask a question or end with a question.
Judge the primary pedagogical speech act, not punctuation. In particular,
declarative generic and focus turns can pass, while telling must actually
provide information before any optional question.

TASK 4 - CHECK ALIGNMENT WITH THE EXTERNALLY SELECTED MOVE
Set move_alignment_verdict to "fail" if the teacher response does not actually
execute the supplied move, blends multiple moves, or substitutes a different
move. Do not decide that another move would be better; only judge fidelity to
the supplied move.

The raw internal labels generic, probing, focus, and telling are private. Fail
alignment if the learner-facing response is prefixed with one of those labels.

Use the supplied exact move-alignment contract. Length ranges are guidance for
clarity, not mechanical sentence-count gates. Reject material semantic
mismatches while allowing natural wording.

For both verdicts, provide a concise reason. If a verdict fails, include at
least one corresponding turn_scope_issue describing the learner-facing text
that caused the failure. If both verdicts pass and there are no mathematical
problems, turn_scope_issues may be empty.
""".strip()

    def __init__(
        self,
        client: Any,
        model_name: str,
        math_verifier: SymPyMathVerifier,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.math_verifier = math_verifier

    def audit(
        self,
        *,
        question: str,
        verified_evidence: str,
        teaching_response: str,
        pedagogical_move: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self._extract_audit_payload(
            question=question,
            verified_evidence=verified_evidence,
            teaching_response=teaching_response,
            pedagogical_move=pedagogical_move,
            conversation_history=conversation_history or [],
        )
        return self.evaluate_payload(payload)

    def _extract_audit_payload(
        self,
        *,
        question: str,
        verified_evidence: str,
        teaching_response: str,
        pedagogical_move: str | None,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_content = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"EXTERNALLY SELECTED PEDAGOGICAL MOVE:\n{pedagogical_move or 'not supplied'}\n\n"
            "EXACT MOVE-ALIGNMENT CONTRACT:\n"
            + (
                build_move_audit_instruction(pedagogical_move)
                if pedagogical_move is not None
                else "No move supplied; apply only correctness and scope checks."
            )
            + "\n\n"
            "CURRENT CONVERSATION HISTORY:\n"
            + json.dumps(
                conversation_history,
                ensure_ascii=False,
                default=str,
                indent=2,
            )
            + "\n\n"
            f"VERIFIED MATHEMATICAL EVIDENCE:\n{verified_evidence}\n\n"
            f"TEACHING RESPONSE TO AUDIT:\n{teaching_response}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.audit_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1800,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "adaptmath_teaching_response_audit",
                    "strict": True,
                    "schema": TEACHING_RESPONSE_AUDIT_SCHEMA,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(
                "Tutor response verifier returned an empty audit."
            )

        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise RuntimeError(
                "Tutor response verifier did not return a JSON object."
            )

        return self.prepare_inline_payload(
            payload,
            question=question,
            verified_evidence=verified_evidence,
            conversation_history=conversation_history,
        )

    def prepare_inline_payload(
        self,
        payload: dict[str, Any],
        *,
        question: str,
        verified_evidence: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add deterministic context markers to an inline model audit.

        The learner-facing Tutor can return the audit in the same structured
        response as ``teacher_message``.  Keeping contextual-equation handling
        here makes that one-call path use the exact same deterministic evaluator
        as the legacy independent-auditor path.
        """
        context_parts = [question, verified_evidence]
        for turn in conversation_history:
            if not isinstance(turn, dict):
                continue
            for key in ("content", "text"):
                value = turn.get(key)
                if isinstance(value, str):
                    context_parts.append(value)
        normalized_context = self._normalize_math_text("\n".join(context_parts))
        raw_claims = payload.get("checkable_claims", [])
        if isinstance(raw_claims, list):
            for claim in raw_claims:
                if not isinstance(claim, dict):
                    continue
                source_text = claim.get("source_text")
                if not isinstance(source_text, str):
                    continue
                normalized_source = self._normalize_math_text(source_text)
                if normalized_source and normalized_source in normalized_context:
                    claim["_established_contextual_equation"] = True
        return payload

    @staticmethod
    def _normalize_math_text(value: str) -> str:
        normalized = value.casefold().replace("**", "^")
        for token in ("\\(", "\\)", "\\[", "\\]", "$$", "$", "{", "}"):
            normalized = normalized.replace(token, "")
        return "".join(character for character in normalized if not character.isspace())

    def evaluate_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_claims = payload.get("checkable_claims", [])
        raw_logical_issues = payload.get("logical_issues", [])
        raw_turn_scope_issues = payload.get("turn_scope_issues", [])
        turn_scope_verdict = payload.get(
            "turn_scope_verdict",
            "fail" if raw_turn_scope_issues else "pass",
        )
        move_alignment_verdict = payload.get("move_alignment_verdict", "pass")
        scope_reason = payload.get("scope_reason", "")
        move_alignment_reason = payload.get("move_alignment_reason", "")

        if not isinstance(raw_claims, list):
            raise ValueError("checkable_claims must be an array.")
        if not isinstance(raw_logical_issues, list):
            raise ValueError("logical_issues must be an array.")
        if not isinstance(raw_turn_scope_issues, list):
            raise ValueError("turn_scope_issues must be an array.")
        if turn_scope_verdict not in {"pass", "fail"}:
            raise ValueError("turn_scope_verdict must be pass or fail.")
        if move_alignment_verdict not in {"pass", "fail"}:
            raise ValueError("move_alignment_verdict must be pass or fail.")

        deterministic_results: list[dict[str, Any]] = []
        deterministic_failures: list[dict[str, Any]] = []
        inconclusive_checks: list[dict[str, Any]] = []

        for index, claim in enumerate(raw_claims, start=1):
            if not isinstance(claim, dict):
                inconclusive_checks.append(
                    {
                        "claim": index,
                        "reason": "Extracted claim was not an object.",
                    }
                )
                continue

            source_text = claim.get("source_text")
            left_expression = claim.get("left_expression")
            right_expression = claim.get("right_expression")

            if not all(
                isinstance(value, str) and value.strip()
                for value in (
                    source_text,
                    left_expression,
                    right_expression,
                )
            ):
                inconclusive_checks.append(
                    {
                        "claim": index,
                        "source_text": source_text,
                        "reason": "Extracted claim had incomplete fields.",
                    }
                )
                continue

            if claim.get("_established_contextual_equation") is True:
                result = {
                    "claim": index,
                    "source_text": source_text,
                    "left_expression": left_expression,
                    "right_expression": right_expression,
                    "verified": None,
                    "verification_status": "established_contextual_equation",
                }
                deterministic_results.append(result)
                inconclusive_checks.append(
                    {
                        **result,
                        "reason": (
                            "Contextual equations are conditions or established "
                            "intermediate facts, not universal identities. Their "
                            "support is assessed against the supplied dialogue "
                            "and verified evidence."
                        ),
                    }
                )
                continue

            try:
                verified = self.math_verifier.equivalent(
                    left_expression,
                    right_expression,
                )
            except Exception as exc:
                inconclusive_checks.append(
                    {
                        "claim": index,
                        "source_text": source_text,
                        "left_expression": left_expression,
                        "right_expression": right_expression,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            result = {
                "claim": index,
                "source_text": source_text,
                "left_expression": left_expression,
                "right_expression": right_expression,
                "verified": verified is True,
            }
            deterministic_results.append(result)

            if verified is False:
                deterministic_failures.append(result)
            elif verified is None:
                inconclusive_checks.append(
                    {
                        **result,
                        "reason": "SymPy could not establish equivalence.",
                    }
                )

        logical_issues: list[dict[str, str]] = []
        for issue in raw_logical_issues:
            if not isinstance(issue, dict):
                continue
            source_text = issue.get("source_text")
            reason = issue.get("reason")
            if (
                isinstance(source_text, str)
                and source_text.strip()
                and isinstance(reason, str)
                and reason.strip()
            ):
                logical_issues.append(
                    {
                        "source_text": source_text.strip(),
                        "reason": reason.strip(),
                    }
                )

        turn_scope_issues: list[dict[str, str]] = []
        for issue in raw_turn_scope_issues:
            if not isinstance(issue, dict):
                continue
            source_text = issue.get("source_text")
            reason = issue.get("reason")
            if (
                isinstance(source_text, str)
                and source_text.strip()
                and isinstance(reason, str)
                and reason.strip()
            ):
                turn_scope_issues.append(
                    {
                        "source_text": source_text.strip(),
                        "reason": reason.strip(),
                    }
                )

        approved = (
            not deterministic_failures
            and not logical_issues
            and not turn_scope_issues
            and turn_scope_verdict == "pass"
            and move_alignment_verdict == "pass"
        )

        return {
            "approved": approved,
            "deterministic_results": deterministic_results,
            "deterministic_failures": deterministic_failures,
            "logical_issues": logical_issues,
            "turn_scope_issues": turn_scope_issues,
            "turn_scope_verdict": turn_scope_verdict,
            "move_alignment_verdict": move_alignment_verdict,
            "scope_reason": scope_reason if isinstance(scope_reason, str) else "",
            "move_alignment_reason": (
                move_alignment_reason
                if isinstance(move_alignment_reason, str)
                else ""
            ),
            "inconclusive_checks": inconclusive_checks,
        }

    @staticmethod
    def correction_feedback(audit_result: dict[str, Any]) -> str:
        failures = audit_result.get("deterministic_failures", [])
        logical_issues = audit_result.get("logical_issues", [])
        turn_scope_issues = audit_result.get("turn_scope_issues", [])
        turn_scope_verdict = audit_result.get("turn_scope_verdict", "pass")
        move_alignment_verdict = audit_result.get("move_alignment_verdict", "pass")
        scope_reason = audit_result.get("scope_reason", "")
        move_alignment_reason = audit_result.get("move_alignment_reason", "")

        parts = [
            "The previous teaching draft failed the Tutor response audit.",
            "Rewrite only the current teacher turn from the original problem, "
            "verified evidence, dialogue context, and externally selected move.",
            "Do not preserve a disputed mathematical claim or an over-scoped "
            "multi-step teaching plan.",
        ]

        if failures:
            parts.append(
                "Deterministically false mathematical claims:\n"
                + json.dumps(
                    failures,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if logical_issues:
            parts.append(
                "Clear logical/consistency issues:\n"
                + json.dumps(
                    logical_issues,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if turn_scope_issues:
            parts.append(
                "Single-turn pedagogy/scope issues:\n"
                + json.dumps(
                    turn_scope_issues,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if turn_scope_verdict == "fail":
            parts.append(
                "Turn-scope verdict: FAIL. "
                + (scope_reason if isinstance(scope_reason, str) else "")
            )

        if move_alignment_verdict == "fail":
            parts.append(
                "Move-alignment verdict: FAIL. "
                + (
                    move_alignment_reason
                    if isinstance(move_alignment_reason, str)
                    else ""
                )
            )

        parts.append(
            "The corrected response must address one immediate reasoning target, "
            "give the learner one coherent action, and stop. Do not name several "
            "future operations or ask for the complete solution plan. Preserve "
            "the exact selected move's primary speech act and move-sensitive "
            "clarity/length guidance."
        )

        return "\n\n".join(parts)
