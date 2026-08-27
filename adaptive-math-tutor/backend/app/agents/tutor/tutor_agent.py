import json
from typing import Any

from langchain_core.tools import tool

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.response_verifier import TeachingResponseVerifier
from app.clients.gemini import GeminiChatCompatibilityClient
from app.core.config import get_settings
from app.schemas.tutor import TutorInput, TutorMathInput, TutorOutput


math_verifier = SymPyMathVerifier()


@tool
def evaluate_expression(expression: str, substitutions: dict[str, str]) -> str:
    """Evaluate or simplify one mathematical expression after substitutions."""
    return math_verifier.evaluate(expression, substitutions)


@tool
def expand_expression(expression: str) -> str:
    """Expand one symbolic expression."""
    return math_verifier.expand(expression)


@tool
def factor_expression(expression: str) -> str:
    """Factor one symbolic expression."""
    return math_verifier.factor(expression)


@tool
def get_polynomial_degree(expression: str, variable: str = "x") -> str:
    """Return the polynomial degree."""
    return str(math_verifier.polynomial_degree(expression, variable))


@tool
def check_equivalence(left_expression: str, right_expression: str) -> str:
    """Check whether two expressions are symbolically equivalent."""
    return str(math_verifier.equivalent(left_expression, right_expression))


@tool
def check_polynomial_identity(left_expression: str, right_expression: str) -> str:
    """Check whether two polynomial expressions are identical."""
    return str(math_verifier.polynomial_identity(left_expression, right_expression))


@tool
def derive_polynomial_commutativity_constraints(
    p_expression: str,
    q_expression: str,
    variable: str = "x",
) -> str:
    """Derive every coefficient constraint implied by p(q(x)) = q(p(x))."""
    result = math_verifier.polynomial_commutativity_constraints(
        p_expression,
        q_expression,
        variable,
    )
    compact_result = {
        "degree": result["degree"],
        "equations": result["equations"],
    }
    return json.dumps(compact_result, default=str, separators=(",", ":"))


@tool
def reduce_polynomial_system_groebner(
    equations: list[list[str]],
    variables: list[str],
    nonzero_variables: list[str],
    order: str = "grevlex",
) -> str:
    """Reduce a nonlinear polynomial system using exact preprocessing/Groebner."""
    normalized_equations: list[tuple[str, str]] = []
    for equation in equations:
        if len(equation) != 2:
            raise ValueError(
                "Every equation must contain exactly "
                "[left_expression, right_expression]."
            )
        normalized_equations.append((equation[0], equation[1]))

    result = math_verifier.groebner_reduce_system(
        equations=normalized_equations,
        variables=variables,
        nonzero_variables=nonzero_variables,
        order=order,
    )
    compact_result = {
        "relations": result["relations"],
        "nonzero_variables": result["nonzero_variables"],
    }
    return json.dumps(compact_result, default=str, separators=(",", ":"))


@tool
def check_polynomial_commutativity(
    p_expression: str,
    q_expression: str,
    variable: str = "x",
) -> str:
    """Verify p(q(x)) = q(p(x)) identically."""
    return str(
        math_verifier.polynomials_commute(
            p_expression,
            q_expression,
            variable,
        )
    )


@tool
def solve_equation_system(
    equations: list[list[str]],
    variables: list[str],
) -> str:
    """Solve a relatively small/direct symbolic equation system."""
    normalized_equations: list[tuple[str, str]] = []
    for equation in equations:
        if len(equation) != 2:
            raise ValueError(
                "Every equation must contain exactly "
                "[left_expression, right_expression]."
            )
        normalized_equations.append((equation[0], equation[1]))

    solutions = math_verifier.solve_equations(
        normalized_equations,
        variables,
    )
    return json.dumps(solutions, default=str, separators=(",", ":"))


@tool
def submit_tutor_output(
    verification_checks: list[dict[str, str]],
) -> str:
    """
    Signal that mathematical work is ready for finalization.

    verification_checks are model-selected symbolic claims that
    Python verifies deterministically before finalization.
    """
    return "Tutor mathematics ready for finalization."


TUTOR_ACTION_NAMES = [
    "evaluate_expression",
    "expand_expression",
    "factor_expression",
    "get_polynomial_degree",
    "check_equivalence",
    "check_polynomial_identity",
    "derive_polynomial_commutativity_constraints",
    "reduce_polynomial_system_groebner",
    "check_polynomial_commutativity",
    "solve_equation_system",
    "submit_tutor_output",
]


TUTOR_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": TUTOR_ACTION_NAMES,
        },
        "arguments_json": {
            "type": "string",
        },
    },
    "required": ["action", "arguments_json"],
    "additionalProperties": False,
}


def build_teacher_turn_schema(selected_move: str) -> dict[str, Any]:
    """Structured contract for one learner-facing atomic teacher turn."""
    return {
        "type": "object",
        "properties": {
            "selected_move": {
                "type": "string",
                "enum": [selected_move],
            },
            "immediate_target": {"type": "string"},
            "learner_action": {"type": "string"},
            "teacher_message": {"type": "string"},
            "reveals_future_steps": {"type": "boolean"},
            "multiple_independent_requests": {"type": "boolean"},
            "blends_multiple_pedagogical_actions": {"type": "boolean"},
        },
        "required": [
            "selected_move",
            "immediate_target",
            "learner_action",
            "teacher_message",
            "reveals_future_steps",
            "multiple_independent_requests",
            "blends_multiple_pedagogical_actions",
        ],
        "additionalProperties": False,
    }


class DirectGeminiActionModel:
    """Ask Gemini for exactly one next mathematical action."""

    def __init__(
        self,
        client: GeminiChatCompatibilityClient,
        model_name: str,
    ) -> None:
        self.client = client
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1800,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "adaptmath_tutor_action",
                    "strict": True,
                    "schema": TUTOR_ACTION_SCHEMA,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Gemini returned an empty Tutor action.")

        outer = json.loads(content)
        if not isinstance(outer, dict):
            raise RuntimeError("Gemini Tutor response was not a JSON object.")

        action = outer.get("action")
        arguments_json = outer.get("arguments_json")

        if action not in TUTOR_ACTION_NAMES:
            raise RuntimeError(f"Unknown Tutor action: {action}")
        if not isinstance(arguments_json, str):
            raise RuntimeError("Tutor arguments_json was not a string.")

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            return {
                "action": action,
                "arguments": None,
                "arguments_json": arguments_json,
                "argument_error": f"arguments_json was not valid JSON: {exc}",
            }

        if not isinstance(arguments, dict):
            return {
                "action": action,
                "arguments": None,
                "arguments_json": arguments_json,
                "argument_error": "arguments_json must encode one JSON object.",
            }

        return {
            "action": action,
            "arguments": arguments,
            "arguments_json": arguments_json,
            "argument_error": None,
        }


class TutorAgent:
    """
    GenAI Tutor with iterative mathematical verification.

    Gemini selects mathematical actions. SymPy executes deterministic checks.
    Learner-facing teaching happens after the math loop; assessment is generated later by the Evaluator.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.gemini_model
        self.client = GeminiChatCompatibilityClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            max_retries=2,
        )
        self.action_model = DirectGeminiActionModel(
            client=self.client,
            model_name=self.model_name,
        )

        self.tools = [
            evaluate_expression,
            expand_expression,
            factor_expression,
            get_polynomial_degree,
            check_equivalence,
            check_polynomial_identity,
            derive_polynomial_commutativity_constraints,
            reduce_polynomial_system_groebner,
            check_polynomial_commutativity,
            solve_equation_system,
            submit_tutor_output,
        ]
        self.tool_map = {
            current_tool.name: current_tool
            for current_tool in self.tools
            if current_tool.name != "submit_tutor_output"
        }

        self.max_tool_rounds = 12
        self.max_response_verification_attempts = 3
        self.response_verifier = TeachingResponseVerifier(
            client=self.client,
            model_name=self.model_name,
            math_verifier=math_verifier,
        )

        self.system_prompt = """
You are the Tutor Agent in AdaptMath, an adaptive mathematics tutoring
system for learners aged 8 to 18.

On every turn choose exactly ONE action.

The response protocol is:
- action
- arguments_json

arguments_json must be a STRING containing one valid JSON object with the
arguments required by that action. Do not place markdown around it.

ACTION ARGUMENT FORMATS

The examples below demonstrate JSON/tool syntax only. They are unrelated to
any evaluation item and must never be treated as facts about the current
problem.

evaluate_expression
{"expression":"(18-6)/4","substitutions":{}}

evaluate_expression with substitution
{"expression":"2*y+7","substitutions":{"y":"4"}}

expand_expression
{"expression":"(t-3)**2"}

factor_expression
{"expression":"z**2-9"}

get_polynomial_degree
{"expression":"5*u**4-u+2","variable":"u"}

check_equivalence
{"left_expression":"6/8","right_expression":"3/4"}

check_polynomial_identity
{"left_expression":"(m-2)**2","right_expression":"m**2-4*m+4"}

derive_polynomial_commutativity_constraints
{"p_expression":"a*x**3+b*x**2+c*x+r","q_expression":"d*x**3+e*x**2+f*x+s","variable":"x"}

reduce_polynomial_system_groebner
{"equations":[["u+v-3","0"],["u-v-1","0"]],"variables":["u","v"],"nonzero_variables":[],"order":"grevlex"}

check_polynomial_commutativity
{"p_expression":"...","q_expression":"...","variable":"x"}

solve_equation_system
{"equations":[["x+y","7"],["x-y","1"]],"variables":["x","y"]}

submit_tutor_output
{"verification_checks":[
  {"left_expression":"(18-6)/4","right_expression":"3"}
]}

IMPORTANT: submit_tutor_output is only a READY signal. Do NOT put teaching
prose or assessment questions inside it. A separate finalization stage
creates those later.

Before finalization, include one or more model-selected symbolic verification
checks for the requested final mathematical claim(s). Each check must contain:
- left_expression
- right_expression

Python verifies every check deterministically with SymPy. If a check is false
or cannot be established, finalization is rejected and you must continue the
mathematical action loop. For a problem asking for multiple final values,
include a check for each requested value whenever representable symbolically.


MATHEMATICAL REQUIREMENTS

Solve the actual problem correctly. Never invent equations, coefficients,
values, functions, conditions, or learner information.

Use deterministic mathematical tools to check important claims whenever they
are checkable. If tool evidence contradicts your reasoning, correct your
reasoning. Never ignore verified mathematical evidence.

Before submit_tutor_output, verify the requested final numerical or symbolic
result with available mathematical tools whenever it can be expressed using
those tools. A reduction that only gives coefficient relations is not, by
itself, verification of a requested evaluation.

Tool expression fields contain ONLY mathematical expressions. Never write
"p(x) = a*x**3..." inside an expression field. Write only
"a*x**3...". Use Python syntax: x**2, 3*x, (x+1)/2. Never use ^.

evaluate_expression accepts ONE expression.

POLYNOMIAL COMPOSITION / COMMUTATIVITY

When a problem specifies an unknown polynomial's degree, construct a generic
symbolic polynomial with fresh coefficient symbols. Incorporate coefficient or
function-value constraints only when they are explicitly stated in the current
question or established by verified tool evidence. Never assume constants,
coefficient values, or polynomial forms from examples or previous tasks.

For polynomial composition identities such as p(q(x)) = q(p(x)), when they are
part of the current problem, derive the COMPLETE coefficient constraints with
derive_polynomial_commutativity_constraints before attempting to solve the
coefficient system. For a large nonlinear polynomial system, use
reduce_polynomial_system_groebner when appropriate. Do not use
solve_equation_system merely because many equations are available.

When using polynomial-system reduction, pass the complete derived equations,
the actual unknown coefficients, and only mathematically justified nonzero
variables. For example, a leading coefficient is nonzero when the current
problem explicitly states that the polynomial has that degree.

Free parameters are allowed. Never assign an arbitrary value to a free
parameter just to finish. Determine whether the requested result is independent
of it. Check important original conditions when practical.

PEDAGOGY / RESPONSIBILITIES

The later finalization stage writes the learner-facing lesson. This loop's job
is to obtain trustworthy mathematical evidence. Use supplied planner guidance, learner errors, and reteaching context when
preparing mathematical evidence. Per-turn pedagogical moves are applied only
in the learner-facing stage and come from Omash's external move-selector. Do
not take over Router, Evaluator, Memory, or move-selector responsibilities.

Choose submit_tutor_output only when the mathematics is ready. Its
arguments_json must contain a non-empty "verification_checks" array targeting
the final requested mathematical claim(s), not unrelated intermediate work.
""".strip()

        self.final_teaching_prompt = """
You are the learner-facing Tutor in AdaptMath, an interactive mathematics
 tutoring system.

Your output is ONE teacher move in a real teacher-student dialogue. It is not a
mini-lesson, not a solution outline, and not a set of future subquestions.
After this message the learner must respond before any second pedagogical action
is allowed.

You will return structured JSON matching the supplied schema. The fields
immediate_target and learner_action are PRIVATE planning metadata. Only
teacher_message is shown to the learner.

Use only the supplied problem, dialogue history, learner context, planner
context, and verified mathematical evidence. Planner output and verified
mathematical evidence are PRIVATE TEACHER NOTES. Never summarize, enumerate, or
expose their future solution sequence to the learner.

MATHEMATICAL REQUIREMENTS
- Every mathematical statement you make must be correct.
- Do not invent facts, restrictions, coefficients, learner information, or tool
  results.
- Preserve restrictions from the original problem.
- Do not claim uniqueness or exhaustiveness unless verified evidence supports
  it.
- The teacher turn will be independently audited before it is shown.

ATOMIC DIALOGUE CONTRACT
- Execute exactly ONE pedagogical action for the CURRENT moment in the dialogue.
- Address exactly ONE immediate reasoning target.
- Hand exactly ONE coherent piece of cognitive work back to the learner.
- If you can think of two things the learner could do next, choose only the
  FIRST one that is useful now. The next action belongs to a future Omash move.
- Never ask the learner for the complete solution, the full plan, all steps, all
  relevant numbers plus all operations, or several calculations at once.
- Never reveal an ordered chain such as "multiply, then subtract, then divide"
  before the learner has responded between those steps.
- Never list several future operations or transformations merely to help the
  learner plan ahead. That removes Omash's opportunity to adapt after each
  student response.
- A brief explanation plus one immediate question is acceptable only when both
  serve the SAME immediate target.
- Do not generate the three formal assessment questions. The Evaluator does
  that only after interactive teaching is complete.

PEDAGOGICAL MOVE OWNERSHIP
The selected MathDial move comes from Omash's external move-selector component.
You MUST execute exactly that move. Do not choose, replace, blend, reclassify,
or silently substitute another move.

Move execution rules:
- generic: make ONE broad conversational invitation for the learner to reveal
  their current thinking, initial approach, or next idea. Do not teach a method
  and do not name a sequence of operations or formulas that the learner has not
  already introduced. On an opening turn, a generic move should be genuinely
  open, for example "How would you start solving this?" or "What do you notice
  first?" rather than asking for the entire plan.
- focus: direct attention to ONE relevant quantity, relationship,
  representation, or immediate reasoning target. Do not point to multiple
  separate targets in the same turn.
- probing: investigate ONE current claim, reason, misconception, or piece of
  learner thinking. Ask for one justification or self-correction, not several.
- telling: explicitly provide ONE needed fact, idea, representation, or strategy
  step, then return one immediate piece of work to the learner. Telling does not
  authorize revealing all remaining steps.

EXPLICIT SCOPE EXAMPLES
These examples define scope only; they are not facts about the current problem.

PASS, generic:
"How would you start solving this problem?"

FAIL, generic:
"What numbers would you write down, and what operations like multiplication,
subtraction, and division might you need? Explain all the steps you would take."
Why it fails: it asks for multiple tasks, names several future operations, and
requests the whole solution plan before the next learner turn.

PASS, focus:
"You have found the total. Which quantity should you compare it with next?"

FAIL, focus:
"Find the total, subtract the removed amount, then divide what remains."
Why it fails: it exposes several future steps instead of one immediate target.

PASS, probing:
"Why do you think division is the right operation here?"

PASS, telling:
"Use division to compare these two quantities. What result do you get?"

For reteaching, address the diagnosed learner difficulty while still executing
only the newly supplied external move.

Before returning the JSON, inspect your own teacher_message. Set the scope
booleans truthfully. If it reveals future steps, asks multiple independent
learner tasks, or blends multiple pedagogical actions, rewrite it before
returning.
""".strip()

    def _build_initial_messages(
        self,
        tutor_input: TutorMathInput,
    ) -> list[dict[str, str]]:
        planner_output: Any = (
            tutor_input.planner_output.model_dump()
            if tutor_input.planner_output
            else None
        )
        user_context = {
            "question": tutor_input.question,
            "topic": tutor_input.topic,
            "subtopic": tutor_input.subtopic,
            "student_age": tutor_input.student_age,
            "complexity_score": tutor_input.complexity_score,
            "planner_output": planner_output,
            "previous_errors": tutor_input.previous_errors,
            "reteaching": tutor_input.reteaching,
        }

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Tutor context:\n"
                    + json.dumps(
                        user_context,
                        default=str,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n\nChoose the next Tutor action."
                ),
            },
        ]

    @staticmethod
    def _action_history_message(action: str, arguments_json: str) -> str:
        """Avoid duplicating huge action arguments in later model prompts."""
        if len(arguments_json) <= 900:
            stored_arguments = arguments_json
        else:
            stored_arguments = (
                "<validated arguments omitted from conversation history: "
                f"{len(arguments_json)} characters>"
            )

        return json.dumps(
            {
                "action": action,
                "arguments_json": stored_arguments,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _build_evidence_text(evidence: list[dict[str, str]]) -> str:
        if not evidence:
            return "No deterministic mathematical tool evidence was produced."

        parts: list[str] = []
        for index, item in enumerate(evidence, start=1):
            result = item["result"]
            if len(result) > 6000:
                result = (
                    result[:6000]
                    + "\n[tool result truncated for finalizer context]"
                )
            parts.append(
                f"EVIDENCE {index}\n"
                f"Action: {item['action']}\n"
                f"Result: {result}"
            )
        return "\n\n".join(parts)

    def _build_finalizer_context(
        self,
        tutor_input: TutorInput,
        evidence: list[dict[str, str]],
    ) -> str:
        planner_output: Any = (
            tutor_input.planner_output.model_dump()
            if tutor_input.planner_output
            else None
        )
        context = {
            "question": tutor_input.question,
            "topic": tutor_input.topic,
            "subtopic": tutor_input.subtopic,
            "student_age": tutor_input.student_age,
            "complexity_score": tutor_input.complexity_score,
            "planner_output": planner_output,
            "pedagogical_move": tutor_input.pedagogical_move,
            "conversation_history": [
                turn.model_dump()
                for turn in tutor_input.conversation_history
            ],
            "previous_errors": tutor_input.previous_errors,
            "reteaching": tutor_input.reteaching,
        }

        return (
            "LEARNER / PROBLEM / DIALOGUE CONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, default=str, indent=2)
            + "\n\nVERIFIED MATHEMATICAL EVIDENCE:\n"
            + self._build_evidence_text(evidence)
        )

    def _generate_teaching_response(
        self,
        tutor_input: TutorInput,
        evidence: list[dict[str, str]],
    ) -> str:
        base_context = self._build_finalizer_context(
            tutor_input,
            evidence,
        )
        verified_evidence = self._build_evidence_text(evidence)
        correction = ""
        last_audit: dict[str, Any] | None = None
        last_draft: dict[str, Any] | None = None

        for _ in range(self.max_response_verification_attempts):
            user_content = base_context
            if correction:
                user_content += (
                    "\n\nPREVIOUS DRAFT FEEDBACK:\n"
                    + correction
                    + "\n\nReturn a corrected atomic teacher turn."
                )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.final_teaching_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                reasoning_effort="low",
                include_reasoning=False,
                max_completion_tokens=650,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "adaptmath_atomic_teacher_turn",
                        "strict": True,
                        "schema": build_teacher_turn_schema(
                            str(tutor_input.pedagogical_move)
                        ),
                    },
                },
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                correction = (
                    "The previous response was empty. Produce one atomic "
                    "teacher turn only."
                )
                continue

            try:
                draft = json.loads(content)
            except json.JSONDecodeError as exc:
                correction = (
                    "The previous response was not valid structured JSON: "
                    f"{exc}."
                )
                continue

            if not isinstance(draft, dict):
                correction = (
                    "The previous response was not a JSON object. Produce one "
                    "atomic teacher-turn object."
                )
                continue

            last_draft = draft
            teaching_response = draft.get("teacher_message")
            if not isinstance(teaching_response, str) or not teaching_response.strip():
                correction = "teacher_message must contain one learner-facing turn."
                continue
            teaching_response = teaching_response.strip()

            declared_move = draft.get("selected_move")
            scope_failures: list[str] = []
            if declared_move != str(tutor_input.pedagogical_move):
                scope_failures.append(
                    "The draft did not preserve the externally selected move."
                )
            if draft.get("reveals_future_steps") is True:
                scope_failures.append(
                    "The draft reports that it reveals future solution steps."
                )
            if draft.get("multiple_independent_requests") is True:
                scope_failures.append(
                    "The draft reports multiple independent learner requests."
                )
            if draft.get("blends_multiple_pedagogical_actions") is True:
                scope_failures.append(
                    "The draft reports that it blends multiple pedagogical actions."
                )

            if scope_failures:
                correction = (
                    "The structured teacher-turn self-check failed:\n- "
                    + "\n- ".join(scope_failures)
                    + "\nRewrite around one immediate target and one learner action."
                )
                continue

            last_audit = self.response_verifier.audit(
                question=tutor_input.question,
                verified_evidence=verified_evidence,
                teaching_response=teaching_response,
                pedagogical_move=tutor_input.pedagogical_move,
                conversation_history=[
                    turn.model_dump()
                    for turn in tutor_input.conversation_history
                ],
            )

            if last_audit.get("approved") is True:
                return teaching_response

            correction = self.response_verifier.correction_feedback(
                last_audit
            )

        raise RuntimeError(
            "Tutor teaching response failed mathematical/atomic-turn "
            f"verification after {self.max_response_verification_attempts} "
            "attempts. "
            f"Last draft: {json.dumps(last_draft, ensure_ascii=False)}. "
            f"Last audit: {json.dumps(last_audit, ensure_ascii=False)}"
        )

    def _finalize_tutor_output(
        self,
        tutor_input: TutorInput,
        evidence: list[dict[str, str]],
    ) -> TutorOutput:
        teaching_response = self._generate_teaching_response(
            tutor_input,
            evidence,
        )
        return TutorOutput(teaching_response=teaching_response)

    def teach_turn(
        self,
        tutor_input: TutorInput,
        evidence: list[dict[str, str]],
    ) -> TutorOutput:
        """Generate one mathematically audited teacher turn."""
        return self._finalize_tutor_output(tutor_input, evidence)

    def prepare_math_evidence(
        self,
        tutor_input: TutorMathInput,
    ) -> list[dict[str, str]]:
        messages = self._build_initial_messages(tutor_input)
        evidence: list[dict[str, str]] = []

        for _ in range(self.max_tool_rounds):
            action_payload = self.action_model.invoke(messages)
            action = action_payload["action"]
            arguments_json = action_payload["arguments_json"]
            arguments = action_payload["arguments"]
            argument_error = action_payload["argument_error"]

            messages.append(
                {
                    "role": "assistant",
                    "content": self._action_history_message(
                        action,
                        arguments_json,
                    ),
                }
            )

            if argument_error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "ACTION ARGUMENT ERROR:\n"
                            f"{argument_error}\n\n"
                            "arguments_json must encode one valid JSON object. "
                            "Correct it and choose the next action."
                        ),
                    }
                )
                continue

            if not isinstance(arguments, dict):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "ACTION ARGUMENT ERROR:\n"
                            "arguments_json did not decode to a JSON object. "
                            "Correct it."
                        ),
                    }
                )
                continue

            if action == "submit_tutor_output":
                verification_checks = arguments.get(
                    "verification_checks"
                )

                if (
                    not isinstance(verification_checks, list)
                    or not verification_checks
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL VERIFICATION ERROR:\n"
                                "submit_tutor_output requires a non-empty "
                                "verification_checks array. Each item must "
                                "contain left_expression and right_expression "
                                "for a final requested mathematical claim."
                            ),
                        }
                    )
                    continue

                if len(verification_checks) > 8:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL VERIFICATION ERROR:\n"
                                "Too many verification checks. Provide at most "
                                "8 concise checks for the final requested "
                                "mathematical claim(s)."
                            ),
                        }
                    )
                    continue

                verification_results: list[dict[str, Any]] = []
                verification_failed = False

                for index, check in enumerate(
                    verification_checks,
                    start=1,
                ):
                    if not isinstance(check, dict):
                        verification_results.append(
                            {
                                "check": index,
                                "verified": False,
                                "error": "Check must be a JSON object.",
                            }
                        )
                        verification_failed = True
                        continue

                    left_expression = check.get("left_expression")
                    right_expression = check.get("right_expression")

                    if (
                        not isinstance(left_expression, str)
                        or not left_expression.strip()
                        or not isinstance(right_expression, str)
                        or not right_expression.strip()
                    ):
                        verification_results.append(
                            {
                                "check": index,
                                "verified": False,
                                "error": (
                                    "left_expression and right_expression "
                                    "must be non-empty strings."
                                ),
                            }
                        )
                        verification_failed = True
                        continue

                    try:
                        verified = math_verifier.equivalent(
                            left_expression,
                            right_expression,
                        )
                    except Exception as exc:
                        verification_results.append(
                            {
                                "check": index,
                                "left_expression": left_expression,
                                "right_expression": right_expression,
                                "verified": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        verification_failed = True
                        continue

                    verification_results.append(
                        {
                            "check": index,
                            "left_expression": left_expression,
                            "right_expression": right_expression,
                            "verified": verified is True,
                        }
                    )

                    if verified is not True:
                        verification_failed = True

                verification_text = json.dumps(
                    verification_results,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                if verification_failed:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL VERIFICATION FAILED:\n"
                                f"{verification_text}\n\n"
                                "At least one submitted final claim was false, "
                                "malformed, or not symbolically established. "
                                "Continue the mathematical action loop, repair "
                                "the reasoning if needed, and submit corrected "
                                "verification checks later."
                            ),
                        }
                    )
                    continue

                evidence.append(
                    {
                        "action": "final_claim_verification",
                        "result": verification_text,
                    }
                )

                return evidence

            selected_tool = self.tool_map.get(action)
            if selected_tool is None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"ACTION ERROR:\nThe action '{action}' has no "
                            "executable mathematical tool. Choose another valid action."
                        ),
                    }
                )
                continue

            try:
                result = selected_tool.invoke(arguments)
                tool_result = str(result)
            except Exception as exc:
                tool_result = (
                    "Mathematical tool error: "
                    f"{type(exc).__name__}: {exc}"
                )

            evidence.append({"action": action, "result": tool_result})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"RESULT OF {action}:\n{tool_result}\n\n"
                        "Treat successful deterministic mathematical results as "
                        "evidence. If this result is an error, repair the arguments "
                        "or choose another appropriate action. Then choose exactly "
                        "one next Tutor action."
                    ),
                }
            )

        raise RuntimeError(
            "Tutor exceeded the maximum mathematical action rounds "
            "without reaching finalization."
        )

    def teach(self, tutor_input: TutorInput) -> TutorOutput:
        """Compatibility helper: solve internally once, then emit one turn."""
        math_input = TutorMathInput(
            question=tutor_input.question,
            topic=tutor_input.topic,
            subtopic=tutor_input.subtopic,
            student_age=tutor_input.student_age,
            complexity_score=tutor_input.complexity_score,
            planner_output=tutor_input.planner_output,
            previous_errors=tutor_input.previous_errors,
            reteaching=tutor_input.reteaching,
        )
        evidence = self.prepare_math_evidence(math_input)
        return self.teach_turn(tutor_input, evidence)
