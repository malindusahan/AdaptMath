import json
import logging
import re
from typing import Any

from langchain_core.tools import tool

from app.agents.tutor.math_verifier import SymPyMathVerifier
from app.agents.tutor.realization_contract import (
    MOVE_REALIZATION_CONTRACT,
    MOVE_REALIZATION_PROFILES,
    build_move_realization_instruction,
)
from app.agents.tutor.response_verifier import (
    TEACHING_RESPONSE_AUDIT_SCHEMA,
    TeachingResponseVerifier,
)
from app.clients.gemini import GeminiChatCompatibilityClient
from app.core.config import get_settings
from app.schemas.planner import PlannerOutput
from app.schemas.tutor import TutorInput, TutorMathInput, TutorOutput


math_verifier = SymPyMathVerifier()
logger = logging.getLogger("adaptmath.tutor")


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


VERIFICATION_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "left_expression": {"type": "string"},
        "right_expression": {"type": "string"},
    },
    "required": ["left_expression", "right_expression"],
    "additionalProperties": False,
}


def build_question_preparation_schema(
    include_planner: bool,
) -> dict[str, Any]:
    """One model response containing optional planning and math checks."""
    properties: dict[str, Any] = {
        "tool_actions": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            name
                            for name in TUTOR_ACTION_NAMES
                            if name != "submit_tutor_output"
                        ],
                    },
                    "arguments_json": {"type": "string"},
                },
                "required": ["action", "arguments_json"],
                "additionalProperties": False,
            },
        },
        "verification_checks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": VERIFICATION_CHECK_SCHEMA,
        },
    }
    required = ["tool_actions", "verification_checks"]

    if include_planner:
        properties["planner_output"] = {
            "type": "object",
            "properties": {
                "learning_goal": {"type": "string"},
                "required_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "teaching_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "anticipated_difficulties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "tutor_guidance": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "learning_goal",
                "required_concepts",
                "teaching_sequence",
                "anticipated_difficulties",
                "tutor_guidance",
            ],
            "additionalProperties": False,
        }
        required.append("planner_output")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


RAW_MOVE_LABEL_PREFIX = re.compile(
    r"^\s*(?:\[(?:generic|probing|focus|telling)\]"
    r"|\((?:generic|probing|focus|telling)\)"
    r"|(?:generic|probing|focus|telling)\s*:)\s*",
    flags=re.IGNORECASE,
)

MATHEMATICAL_CLAIM_PATTERN = re.compile(
    r"(?:[=≈≠≤≥]|\d\s*(?:[+\-*/^]|\\(?:times|div))|"
    r"\\(?:frac|sqrt|pi)\b|"
    r"\b(?:equals?|formula|squared|cubed|sum|difference|product|quotient)\b)",
    flags=re.IGNORECASE,
)


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
            "audit": TEACHING_RESPONSE_AUDIT_SCHEMA,
        },
        "required": [
            "selected_move",
            "immediate_target",
            "learner_action",
            "teacher_message",
            "reveals_future_steps",
            "multiple_independent_requests",
            "blends_multiple_pedagogical_actions",
            "audit",
        ],
        "additionalProperties": False,
    }


class DirectGeminiQuestionPreparationModel:
    """Ask Gemini once for optional planning and deterministic math checks."""

    def __init__(
        self,
        client: GeminiChatCompatibilityClient,
        model_name: str,
    ) -> None:
        self.client = client
        self.model_name = model_name

    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        include_planner: bool,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=3200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "adaptmath_question_preparation",
                    "strict": True,
                    "schema": build_question_preparation_schema(
                        include_planner
                    ),
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Gemini returned empty question preparation.")

        outer = json.loads(content)
        if not isinstance(outer, dict):
            raise RuntimeError(
                "Gemini question preparation was not a JSON object."
            )
        return outer


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
            max_retries=settings.gemini_max_retries,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
        self.preparation_model = DirectGeminiQuestionPreparationModel(
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

        self.max_question_preparation_attempts = 2
        self.max_response_verification_attempts = 2
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

REDUCED-CALL FAST PATH

If you can already express the requested final claim as one or more symbolic
verification checks, choose submit_tutor_output immediately. SymPy performs the
actual verification after submission. Do not spend a separate model round
calling evaluate_expression or check_equivalence merely to duplicate those
same final checks. Use the other deterministic tools first only when you need
their result to derive, transform, or disambiguate the final claim. If direct
submission fails deterministic verification, continue the action loop and
repair it with the appropriate tools.


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

        self.question_preparation_prompt = """
You prepare one AdaptMath question before interactive teaching begins.

Return one structured object containing deterministic mathematical work and,
when the schema requires it, an overall interactive teaching plan. This single
response replaces separate Planner and iterative mathematical-action requests.

PLANNING
- If planner_output is required, create a concise interactive plan containing
  the learning goal, required concepts, a teaching sequence, anticipated
  difficulties supported by context, and Tutor guidance.
- Do not select or recommend generic, probing, focus, or telling. MD7/Turn-LinTS
  owns the pedagogical move before every teacher turn.
- Do not write learner-facing teaching text.

MATHEMATICAL PREPARATION
- Solve the actual supplied problem correctly.
- verification_checks must contain the final requested mathematical claim(s)
  as safe Python/SymPy expressions. Use ** for powers and * for multiplication.
- Include a check for every requested final value when representable.
- Python will verify every final check deterministically with SymPy.
- tool_actions is optional supporting work. Include only deterministic tools
  whose results materially support or disambiguate the solution; do not repeat
  a final equivalence check that verification_checks already performs.
- Each tool action uses one valid JSON object encoded in arguments_json.
- Never invent learner history, mathematical conditions, or tool results.

For complex derivations, reason internally and return all needed independent
tool actions in this one batch. If deterministic validation rejects the batch,
you may receive one correction request and must return a repaired full object.
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

DIALOGUE RESPONSIVENESS
- Begin from the learner's latest message, not from a generic restart of the
  problem. Preserve a useful expression, operation, sign, interpretation, or
  partial result the learner has already supplied.
- Make the turn recognizably about THIS problem. Refer to at least one actual
  number, expression, variable, unit, object, or relationship appearing in the
  problem or recent dialogue whenever the context contains one.
- Do not use stock abstractions such as "the whole quantity and the part," "the
  given values," or "the relationship between them" unless those exact ideas
  are genuinely the immediate mathematical target.
- Do not send a learner who has proposed a usable next step back to identifying
  the starting information. Respond to that step at the detail level allowed by
  the selected move.
- Treat short uncertainty messages such as "I don't know" as dialogue context,
  not as content to praise. Orient, probe, focus, or teach according to the
  selected move using a concrete feature of the actual problem.

MATHEMATICAL REQUIREMENTS
- Every mathematical statement you make must be correct.
- Do not invent facts, restrictions, coefficients, learner information, or tool
  results.
- Preserve restrictions from the original problem.
- Do not claim uniqueness or exhaustiveness unless verified evidence supports
  it.
- The structured audit returned with the teacher turn is checked
   deterministically before the turn is shown.

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
- generic: the PRIMARY speech act is context-aware support, acknowledgement,
  orientation, or transition. Make it useful and non-vacuous, normally in 1-3
  sentences. Tie it to a concrete detail from the problem or the learner's
  latest progress. Do not target a specific reasoning gap, introduce a corrective
  hint, explain the method, reveal a worked step, or systematically ask a
  question. A question is optional, not the default. Consult recent dialogue
  to avoid repeating stock praise, including after consecutive generic moves.
- probing: the PRIMARY speech act is eliciting learner reasoning. Normally ask
  ONE clear, context-specific mathematical reasoning question, normally in 1-3
  sentences. Build it from the learner's latest expression or interpretation
  when available. The learner does the cognitive work; do not answer the question,
  provide the missing step, or work the solution before the probe.
- focus: the PRIMARY speech act is directing attention to ONE specific error,
  relationship, clue, representation, or immediate sub-step. Use a targeted
  cue naming the actual sign, term, expression, quantity, or contextual clue,
  normally in 2-4 sentences when explanation
  helps. Be more direct than probing but less explicit than telling, and leave
  meaningful work. Do not automatically rewrite the cue as an open-ended
  question and do not provide the complete explanation.
- telling: the PRIMARY speech act is explicit instruction or explanation. Put
  the needed fact, method, missing step, or worked clarification FIRST and
  actually provide it in clear, problem-connected, step-by-step language,
  normally in 2-5 sentences. If the learner already proposed a usable operation
  or expression, work or explain that exact step. Use equations when helpful. Do not disguise
  telling as leading questions or reduce it to asking what the learner should
  do. A short follow-up invitation may occur only after the explanation.

There is NO global requirement to ask a question or end with a question. The
selected move overrides any general conversational preference for questioning.

LEARNER-AGENCY ESCALATION
When the selected move is telling AND the latest learner message explicitly
asks for the answer, or the latest two learner messages both communicate that
they cannot engage, honor that request instead of asking why they want help.
- For a short, genuinely one-step question, state the verified answer directly
  with at most one brief clarification. Do not turn the response back into a
  question merely to prolong the dialogue.
- For a multi-step problem, explicitly provide one useful next fact or worked
  step and its result, then offer one immediate continuation opportunity. Do
  not reveal the entire remaining solution.

EXPLICIT SCOPE EXAMPLES
These examples define scope only; they are not facts about the current problem.

PASS, generic:
"Good, we're working with the area left after removing the flower bed. Keep
the two circular regions in mind as you continue."

FAIL, generic:
"What numbers would you write down, and what operations like multiplication,
subtraction, and division might you need? Explain all the steps you would take."
Why it fails: it asks for multiple tasks, names several future operations, and
requests the whole solution plan before the next learner turn.

PASS, focus:
"Focus on the sign of the second product. A negative multiplied by a negative
affects that sign before you combine the terms."

FAIL, focus:
"Find the total, subtract the removed amount, then divide what remains."
Why it fails: it exposes several future steps instead of one immediate target.

PASS, probing:
"You chose division for this comparison. What does the quotient represent in
the problem?"

PASS, telling:
"A multiplicative comparison is found by dividing the first quantity by the
second. The quotient tells how many times as large the first quantity is."

PASS, telling after an explicit answer request on a one-step question:
"No. In ordinary arithmetic, 1 + 1 = 2; writing the digits side by side gives
11, but that is not addition."

PASS, generic in an elevation problem:
"Sea level is our zero point, and the diver's starting elevation is -18 meters.
Keep that signed starting value in view as you continue."

PASS, focus after the learner writes "-12":
"Keep the negative sign on the 12: descending 12 meters is a change of -12.
Apply that signed change to the current elevation first."

PASS, probing after the learner writes "add -18 to -12":
"You set up -18 + (-12). What value does that sum give for the elevation after
the descent?"

PASS, telling after the learner writes "add -18 to -12":
"Carry out the signed-number calculation you set up: -18 + (-12) = -30. The
diver is at -30 meters after the descent."

For reteaching, address the diagnosed learner difficulty while still executing
only the newly supplied external move.

Before returning the JSON, inspect your own teacher_message. Set the scope
booleans truthfully. Populate the nested audit object using the supplied inline
audit contract. If the message reveals future steps, asks multiple independent
learner tasks, blends multiple pedagogical actions, or contains a false
mathematical claim, rewrite it before returning.
""".strip()

    def _build_initial_messages(
        self,
        tutor_input: TutorMathInput,
        *,
        include_planner: bool,
        previous_strategies: list[str],
    ) -> list[dict[str, str]]:
        user_context = {
            "question": tutor_input.question,
            "topic": tutor_input.topic,
            "subtopic": tutor_input.subtopic,
            "student_age": tutor_input.student_age,
            "complexity_score": tutor_input.complexity_score,
            "include_interactive_plan": include_planner,
            "previous_errors": tutor_input.previous_errors,
            "previous_strategies": previous_strategies,
            "reteaching": tutor_input.reteaching,
        }

        return [
            {"role": "system", "content": self.question_preparation_prompt},
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
                    + "\n\nReturn the complete question-preparation object."
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

    @staticmethod
    def _contains_mathematical_claim(
        teaching_response: str,
        inline_audit: dict[str, Any],
    ) -> bool:
        claims = inline_audit.get("checkable_claims")
        return bool(
            (isinstance(claims, list) and claims)
            or MATHEMATICAL_CLAIM_PATTERN.search(teaching_response)
        )

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
            build_move_realization_instruction(
                str(tutor_input.pedagogical_move)
            )
            + "\n\nLEARNER / PROBLEM / DIALOGUE CONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, default=str, indent=2)
            + "\n\nVERIFIED MATHEMATICAL EVIDENCE:\n"
            + self._build_evidence_text(evidence)
            + "\n\nINLINE AUDIT CONTRACT:\n"
            + (
                "Apply the following requirements to the nested audit object. "
                "The outer teacher-turn JSON schema remains authoritative; "
                "where the audit instructions say to return JSON, place those "
                "fields inside audit.\n\n"
            )
            + TeachingResponseVerifier.audit_prompt
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
            if RAW_MOVE_LABEL_PREFIX.match(teaching_response):
                correction = (
                    "teacher_message exposed a private pedagogical-move label. "
                    "Remove the label and realize the selected speech act naturally."
                )
                continue

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

            conversation_history = [
                turn.model_dump()
                for turn in tutor_input.conversation_history
            ]
            inline_audit = draft.get("audit")
            if (
                isinstance(inline_audit, dict)
                and hasattr(self.response_verifier, "prepare_inline_payload")
                and hasattr(self.response_verifier, "evaluate_payload")
            ):
                prepared_audit = self.response_verifier.prepare_inline_payload(
                    inline_audit,
                    question=tutor_input.question,
                    verified_evidence=verified_evidence,
                    conversation_history=conversation_history,
                )
                last_audit = self.response_verifier.evaluate_payload(
                    prepared_audit
                )
            else:
                # Backwards-compatible fallback for old checkpoints and test
                # doubles created before inline audits became mandatory.
                last_audit = self.response_verifier.audit(
                    question=tutor_input.question,
                    verified_evidence=verified_evidence,
                    teaching_response=teaching_response,
                    pedagogical_move=tutor_input.pedagogical_move,
                    conversation_history=conversation_history,
                )

            if last_audit.get("approved") is True:
                needs_independent_audit = (
                    isinstance(inline_audit, dict)
                    and self._contains_mathematical_claim(
                        teaching_response,
                        inline_audit,
                    )
                )
                if not needs_independent_audit:
                    return teaching_response

                last_audit = self.response_verifier.audit(
                    question=tutor_input.question,
                    verified_evidence=verified_evidence,
                    teaching_response=teaching_response,
                    pedagogical_move=tutor_input.pedagogical_move,
                    conversation_history=conversation_history,
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
        try:
            return self._finalize_tutor_output(tutor_input, evidence)
        except Exception as exc:
            logger.warning(
                "tutor_generation_fallback exception_type=%s move=%s",
                type(exc).__name__,
                tutor_input.pedagogical_move,
            )
            return TutorOutput(
                teaching_response=self._local_move_fallback(
                    tutor_input,
                    evidence,
                ),
                fallback_used=True,
            )

    @staticmethod
    def _latest_student_message(tutor_input: TutorInput) -> str | None:
        for turn in reversed(tutor_input.conversation_history):
            if turn.role == "student" and turn.content.strip():
                return " ".join(turn.content.split())[:180]
        return None

    @staticmethod
    def _simple_integer_calculation(text: str | None) -> tuple[str, int] | None:
        """Recognize and safely evaluate a small learner-proposed integer step."""

        if not text:
            return None
        patterns = (
            (r"\badd\s+(-?\d+)\s+to\s+(-?\d+)\b", lambda a, b: (b, "+", a, b + a)),
            (r"\bsubtract\s+(-?\d+)\s+from\s+(-?\d+)\b", lambda a, b: (b, "-", a, b - a)),
            (r"(?<!\d)(-?\d+)\s*\+\s*\(?(-?\d+)\)?", lambda a, b: (a, "+", b, a + b)),
            (r"(?<!\d)(-?\d+)\s*-\s*\(?(-?\d+)\)?", lambda a, b: (a, "-", b, a - b)),
        )
        for pattern, operation in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue
            left, symbol, right, result = operation(
                int(match.group(1)), int(match.group(2))
            )
            rendered_right = f"({right})" if right < 0 else str(right)
            return f"{left} {symbol} {rendered_right}", result
        return None

    @staticmethod
    def _is_elevation_problem(question: str) -> bool:
        lowered = question.casefold()
        return "sea level" in lowered and any(
            word in lowered for word in ("elevation", "descend", "rise")
        )

    @staticmethod
    def _signed_integers(text: str | None) -> tuple[int, ...]:
        if not text:
            return ()
        normalized = text.replace("−", "-")
        return tuple(
            int(match.group(0))
            for match in re.finditer(r"(?<!\d)-?\d+(?!\d)", normalized)
        )

    @staticmethod
    def _elevation_problem_values(question: str) -> tuple[int, int, int] | None:
        """Extract a simple start, descent, and rise without evaluating prose."""

        normalized = question.replace("−", "-").replace("**", "")
        start_match = re.search(
            r"\bstarts?\s+at\s+(-?\d+)\s*meters?\b",
            normalized,
            flags=re.IGNORECASE,
        )
        descent_match = re.search(
            r"\bdescend(?:s|ed)?(?:\s+another)?\s+(-?\d+)\s*meters?\b",
            normalized,
            flags=re.IGNORECASE,
        )
        rise_match = re.search(
            r"\b(?:rise|rises|rose)\s+(-?\d+)\s*meters?\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if not all((start_match, descent_match, rise_match)):
            return None
        start = int(start_match.group(1))
        descent = -abs(int(descent_match.group(1)))
        rise = abs(int(rise_match.group(1)))
        return start, descent, rise

    @classmethod
    def _elevation_move_fallback(
        cls,
        tutor_input: TutorInput,
        latest: str | None,
    ) -> str | None:
        values = cls._elevation_problem_values(tutor_input.question)
        if values is None:
            return None

        move = str(tutor_input.pedagogical_move)
        start, descent, rise = values
        after_descent = start + descent
        final_elevation = after_descent + rise
        distance = abs(final_elevation)
        latest_numbers = cls._signed_integers(latest)
        student_number_sets = [
            set(cls._signed_integers(turn.content))
            for turn in tutor_input.conversation_history
            if turn.role == "student"
        ]
        saw_start_and_descent = any(
            start in numbers and descent in numbers
            for numbers in student_number_sets
        )

        if final_elevation in latest_numbers:
            responses = {
                "generic": (
                    f"You reached the signed final elevation, {final_elevation} meters. "
                    "The remaining part is to interpret its distance from sea level."
                ),
                "probing": (
                    f"Your final elevation is {final_elevation} meters. What is "
                    f"|{final_elevation}|, and what does that distance mean here?"
                ),
                "focus": (
                    f"Keep the final elevation {final_elevation} meters signed. "
                    "For absolute value, measure its distance from zero rather than "
                    "changing the elevation itself."
                ),
                "telling": (
                    f"The final elevation is {final_elevation} meters, and "
                    f"|{final_elevation}| = {distance}. The absolute value means the "
                    f"diver is {distance} meters from sea level."
                ),
            }
            return responses[move]

        if distance in latest_numbers and final_elevation not in latest_numbers:
            responses = {
                "generic": (
                    f"The value {distance} meters is a distance, so it is stated "
                    "without a negative sign."
                ),
                "probing": (
                    f"You have the absolute value {distance}. What does being "
                    f"{distance} meters from zero mean for the diver's position?"
                ),
                "focus": (
                    f"Connect the absolute value {distance} to sea level: it gives "
                    "distance from zero, while the negative elevation tells the direction."
                ),
                "telling": (
                    f"An absolute value of {distance} means the diver is {distance} "
                    "meters from sea level; the final negative elevation places the "
                    "diver that far below sea level."
                ),
            }
            return responses[move]

        if after_descent in latest_numbers:
            responses = {
                "generic": (
                    f"You now have {after_descent} meters after the descent. Keep "
                    f"the later rise of {rise} meters as a separate positive change."
                ),
                "probing": (
                    f"The diver is at {after_descent} meters after descending. What "
                    f"does {after_descent} + {rise} equal after the rise?"
                ),
                "focus": (
                    f"A rise of {rise} meters is a positive change. Evaluate "
                    f"{after_descent} + {rise} for the final elevation."
                ),
                "telling": (
                    f"Apply the rise to the intermediate elevation: "
                    f"{after_descent} + {rise} = {final_elevation}. The diver's final "
                    f"elevation is {final_elevation} meters."
                ),
            }
            return responses[move]

        if start in latest_numbers and descent not in latest_numbers:
            responses = {
                "generic": (
                    f"Keep the roles separate: {start} meters is the starting "
                    f"elevation, while {abs(descent)} meters describes the descent."
                ),
                "probing": (
                    f"{start} meters is where the diver starts. What signed change "
                    f"represents descending {abs(descent)} meters?"
                ),
                "focus": (
                    f"{start} is the starting elevation, not the new downward change. "
                    f"Descending {abs(descent)} meters contributes {descent}."
                ),
                "telling": (
                    f"The starting elevation is {start}, and the descent is the "
                    f"separate change {descent}. Combine them as {start} + "
                    f"({descent})."
                ),
            }
            return responses[move]

        if (
            (start in latest_numbers and descent in latest_numbers)
            or (not latest_numbers and saw_start_and_descent)
        ):
            responses = {
                "generic": (
                    f"You have both signed quantities: a starting elevation of "
                    f"{start} meters and a descent change of {descent} meters."
                ),
                "probing": (
                    f"You identified {start} and {descent}. What does {start} + "
                    f"({descent}) equal after the descent?"
                ),
                "focus": (
                    f"Use {start} as the current elevation and {descent} as the "
                    f"change. Evaluate {start} + ({descent}) before handling the rise."
                ),
                "telling": (
                    f"Combine the starting elevation and descent change: {start} + "
                    f"({descent}) = {after_descent}. The diver is at {after_descent} "
                    "meters after descending."
                ),
            }
            return responses[move]

        responses = {
            "generic": (
                f"Use sea level as zero and keep the starting elevation, {start} "
                "meters, signed as you work."
            ),
            "probing": (
                f"The diver starts at {start} meters. What signed change represents "
                f"the descent of {abs(descent)} meters?"
            ),
            "focus": (
                f"Focus on the descent of {abs(descent)} meters: moving farther "
                "below sea level makes that change negative."
            ),
            "telling": (
                f"Descending {abs(descent)} meters is the change {descent}. Add it "
                f"to the starting elevation as {start} + ({descent})."
            ),
        }
        return responses[move]

    @staticmethod
    def _context_label(tutor_input: TutorInput) -> str:
        return (tutor_input.subtopic or tutor_input.topic).strip()

    @classmethod
    def _local_move_fallback(
        cls,
        tutor_input: TutorInput,
        evidence: list[dict[str, str]],
    ) -> str:
        """Return a context-aware deterministic realization of the selected move."""

        del evidence  # Never turn unstructured fallback evidence into a math claim.
        latest = cls._latest_student_message(tutor_input)
        calculation = cls._simple_integer_calculation(latest)
        if calculation is not None:
            expression, result = calculation
            responses = {
                "generic": (
                    f'You have a concrete calculation to work with: {expression}. '
                    "Keep that signed expression connected to the quantity it represents."
                ),
                "probing": (
                    f"You set up {expression}. What value does that calculation "
                    "give, and what does that value represent here?"
                ),
                "focus": (
                    f"Stay with the expression you proposed, {expression}. "
                    "Evaluate that one signed-number step before moving on."
                ),
                "telling": (
                    f"Carry out the calculation you proposed: {expression} = {result}. "
                    "That is the result of this step; now connect it to the "
                    "quantity described in the problem."
                ),
            }
            return responses[str(tutor_input.pedagogical_move)]

        if cls._is_elevation_problem(tutor_input.question):
            elevation_response = cls._elevation_move_fallback(tutor_input, latest)
            if elevation_response is not None:
                return elevation_response

        context = cls._context_label(tutor_input)
        learner_reference = (
            f' your latest idea, "{latest}"' if latest else " the information given"
        )
        responses = {
            "generic": (
                f"Keep{learner_reference} connected to the specific {context} "
                "quantity the problem asks you to find."
            ),
            "probing": (
                f"Looking at{learner_reference}, what single calculation or "
                "relationship would you use next in this problem?"
            ),
            "focus": (
                f"Stay with{learner_reference}. Work on that one {context} step "
                "and state what its result represents before moving on."
            ),
            "telling": (
                f"Use{learner_reference} as the immediate {context} step. Work "
                "that step explicitly, then interpret its result in the problem's "
                "context."
            ),
        }
        return responses[str(tutor_input.pedagogical_move)]

    def _execute_preparation_tools(
        self,
        raw_actions: Any,
    ) -> tuple[list[dict[str, str]], list[str]]:
        evidence: list[dict[str, str]] = []
        errors: list[str] = []
        if not isinstance(raw_actions, list):
            return evidence, ["tool_actions must be an array."]

        for index, item in enumerate(raw_actions, start=1):
            if not isinstance(item, dict):
                errors.append(f"Tool action {index} was not an object.")
                continue
            action = item.get("action")
            arguments_json = item.get("arguments_json")
            selected_tool = self.tool_map.get(action)
            if selected_tool is None:
                errors.append(f"Tool action {index} is not executable: {action!r}.")
                continue
            if not isinstance(arguments_json, str):
                errors.append(f"Tool action {index} arguments_json was not text.")
                continue
            try:
                arguments = json.loads(arguments_json)
            except json.JSONDecodeError as exc:
                errors.append(f"Tool action {index} arguments were invalid: {exc}.")
                continue
            if not isinstance(arguments, dict):
                errors.append(f"Tool action {index} arguments were not an object.")
                continue
            try:
                tool_result = str(selected_tool.invoke(arguments))
            except Exception as exc:
                errors.append(
                    f"Tool action {index} failed with {type(exc).__name__}."
                )
                continue
            evidence.append({"action": str(action), "result": tool_result})

        return evidence, errors

    @staticmethod
    def _verify_preparation_checks(
        raw_checks: Any,
    ) -> tuple[dict[str, str] | None, list[str]]:
        if not isinstance(raw_checks, list) or not raw_checks:
            return None, ["verification_checks must be a non-empty array."]
        if len(raw_checks) > 8:
            return None, ["verification_checks may contain at most 8 items."]

        results: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, check in enumerate(raw_checks, start=1):
            if not isinstance(check, dict):
                errors.append(f"Verification check {index} was not an object.")
                continue
            left = check.get("left_expression")
            right = check.get("right_expression")
            if not all(isinstance(value, str) and value.strip() for value in (left, right)):
                errors.append(f"Verification check {index} had empty expressions.")
                continue
            left = TutorAgent._normalize_verification_expression(left)
            right = TutorAgent._normalize_verification_expression(right)
            try:
                verified = math_verifier.equivalent(left, right)
            except Exception as exc:
                verified = False
                errors.append(
                    f"Verification check {index} failed with {type(exc).__name__}."
                )
            results.append(
                {
                    "check": index,
                    "left_expression": left,
                    "right_expression": right,
                    "verified": verified is True,
                }
            )
            if verified is not True and not any(
                message.startswith(f"Verification check {index} ")
                for message in errors
            ):
                errors.append(f"Verification check {index} was not equivalent.")

        result_text = json.dumps(
            results,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            {"action": "final_claim_verification", "result": result_text},
            errors,
        )

    @staticmethod
    def _normalize_verification_expression(value: str) -> str:
        """Normalize common model math notation into safe SymPy syntax."""
        normalized = value.strip()
        replacements = {
            "\\(": "",
            "\\)": "",
            "\\[": "",
            "\\]": "",
            "$": "",
            "π": "pi",
            "\\pi": "pi",
            "×": "*",
            "\\times": "*",
            "\\cdot": "*",
            "÷": "/",
            "\\div": "/",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return re.sub(r"(?<!\*)\^(?!\*)", "**", normalized)

    @staticmethod
    def _fallback_plan(tutor_input: TutorMathInput) -> PlannerOutput:
        concept = tutor_input.subtopic or tutor_input.topic
        return PlannerOutput(
            learning_goal=(
                f"Develop and check a correct method for this {concept} problem."
            ),
            required_concepts=[concept],
            teaching_sequence=[
                "Interpret the requested quantity from the problem statement.",
                "Work through one mathematical relationship per learner turn.",
                "Check the resulting expression against the original conditions.",
            ],
            anticipated_difficulties=list(tutor_input.previous_errors),
            tutor_guidance=[
                "Keep each teacher turn atomic and preserve the external move."
            ],
        )

    @staticmethod
    def _fallback_preparation_evidence(
        errors: list[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "action": "verification_unavailable_fallback",
                "result": (
                    "Question preparation could not establish reusable final "
                    "claim checks. Avoid unsupported numerical or symbolic "
                    "claims; use a problem-grounded atomic learner step. "
                    f"Validation categories: {len(errors)}."
                ),
            }
        ]

    def prepare_question(
        self,
        tutor_input: TutorMathInput,
        *,
        include_planner: bool,
        previous_strategies: list[str] | None = None,
    ) -> tuple[PlannerOutput | None, list[dict[str, str]]]:
        """Prepare plan and verified mathematics in one usual model call."""
        messages = self._build_initial_messages(
            tutor_input,
            include_planner=include_planner,
            previous_strategies=previous_strategies or [],
        )
        last_errors: list[str] = []

        for attempt in range(self.max_question_preparation_attempts):
            try:
                payload = self.preparation_model.invoke(
                    messages,
                    include_planner=include_planner,
                )
            except Exception as exc:
                last_errors = [
                    f"provider request failed with {type(exc).__name__}."
                ]
                if attempt + 1 < self.max_question_preparation_attempts:
                    continue
                break

            evidence, tool_warnings = self._execute_preparation_tools(
                payload.get("tool_actions")
            )
            final_evidence, check_errors = self._verify_preparation_checks(
                payload.get("verification_checks")
            )
            errors = list(check_errors)

            planner_output: PlannerOutput | None = None
            if include_planner:
                try:
                    planner_output = PlannerOutput.model_validate(
                        payload.get("planner_output")
                    )
                except Exception as exc:
                    errors.append(
                        "planner_output failed validation with "
                        f"{type(exc).__name__}."
                    )

            if not errors and final_evidence is not None:
                if tool_warnings:
                    logger.info(
                        "question_preparation_ignored_tool_warnings count=%d",
                        len(tool_warnings),
                    )
                evidence.append(final_evidence)
                return planner_output, evidence

            last_errors = errors
            if attempt + 1 < self.max_question_preparation_attempts:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Deterministic question-preparation validation "
                                "failed. Return one corrected complete object.\n- "
                                + "\n- ".join(errors)
                            ),
                        },
                    ]
                )

        logger.warning(
            "question_preparation_fallback validation_categories=%d planned=%s",
            len(last_errors),
            include_planner,
        )
        return (
            self._fallback_plan(tutor_input) if include_planner else None,
            self._fallback_preparation_evidence(last_errors),
        )

    def prepare_math_evidence(
        self,
        tutor_input: TutorMathInput,
    ) -> list[dict[str, str]]:
        """Compatibility wrapper for callers that do not request a new plan."""
        _, evidence = self.prepare_question(
            tutor_input,
            include_planner=False,
        )
        return evidence

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
