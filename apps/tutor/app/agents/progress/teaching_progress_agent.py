from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.progress import TeachingProgressInput, TeachingProgressOutput


class TeachingProgressAgent:
    """
    GenAI judge that decides whether the current interactive teaching phase has
    reached a natural point for the formal three-question assessment.

    It does not choose a pedagogical move. Omash's component owns move
    selection on every teacher turn.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key.get_secret_value(),
            temperature=0,
            timeout=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
        )
        self.structured_model = self.model.with_structured_output(
            TeachingProgressOutput,
            method="json_schema",
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Teaching Progress Judge in AdaptMath, an interactive mathematics
 tutoring system for learners aged 8 to 18.

Your only job is to decide whether the current teacher-student dialogue should:
- continue_teaching, or
- move to ready_for_assessment.

Also classify the learner's LATEST response to the immediately preceding
teacher turn as correct, partial, incorrect, or unknown. Use the verified
mathematical evidence when it is relevant. Use unknown when the teacher turn
is social/metacognitive, the response supplies no assessable mathematics, or
the available evidence is insufficient. Do not call "I don't know" incorrect
solely because it shows uncertainty; the behavioural detectors handle that
separately. Give a conservative correctness confidence between 0 and 1.

Separately classify the response's evidence category by meaning, not by exact
phrases:
- mathematical_evidence: an assessable mathematical claim, step, or answer;
- knowledge_state_evidence: an explicit lack of knowledge, confusion, or
  understanding that is relevant to the mathematical task;
- interaction_management: a request about how the conversation should proceed,
  with no mathematical or knowledge-state evidence;
- acknowledgement: a purely social acknowledgement/affirmation;
- unclear_no_evidence: echo, problem repetition, or another response with no
  usable mathematical/knowledge-state evidence.

interaction_management, acknowledgement, and unclear_no_evidence must have
latest_response_correctness="unknown". They are not negative mathematical
evidence. Do not classify by a fixed list of literal messages.

This is NOT a pedagogical-move selector. Never choose, recommend, replace, or
infer one of the MathDial moves (telling, focus, generic, probing). Omash's
external component owns that decision.

Judge from the actual dialogue evidence. The goal is not merely that the
teacher has stated a correct solution. The dialogue should show that the
student has engaged with the important logic of the problem sufficiently for a
formal understanding check to be meaningful.

INITIAL-TEACHING COMPLETION GATE
For an initial teaching cycle, ready_for_assessment is permitted only when the
learner has BOTH:
1. explicitly completed the ORIGINAL QUESTION by stating its final requested
   result/conclusion (including relevant units or context); and
2. demonstrated enough of the key reasoning to show how that result follows.

A learner naming the next operation, asking whether an operation is correct,
giving an intermediate value, or saying what they would do next has NOT yet
completed the original problem. For example, if 126 must still be divided by
9, "divide by 9?" is a useful correct step but original_problem_completion is
not_yet. Continue teaching until the learner computes and states the resulting
answer in the context of the original question. A final answer supplied only
by the Tutor does not satisfy this gate.

Set original_problem_completion to:
- complete only when the learner has explicitly completed the original task;
- not_yet when any requested final result or conclusion is still missing;
- not_applicable_reteaching only during reteaching when the current goal is to
  verify correction of a specific prior assessment error rather than solve the
  original task again.

Set reasoning_sufficient_for_assessment conservatively. It is false when the
learner has only agreed, guessed an operation, echoed a Tutor-provided answer,
or supplied isolated fragments without the essential reasoning. The structured
schema forbids assessment readiness unless the completion and reasoning gates
are satisfied by current mathematical evidence.

Do not treat a bare agreement such as "yes", "okay", or simple repetition of
a revealed final answer as strong evidence of understanding. Consider whether
the learner has demonstrated the relevant reasoning, next step, explanation,
or correction in context.

Do not use fixed turn counts, numerical thresholds, age brackets, or static
problem-type rules. A short dialogue can be ready when the evidence supports
it; a longer dialogue can still need teaching.

For reteaching, judge whether the misconception/error that motivated
reteaching has been addressed sufficiently to proceed to a new assessment.

Return only the structured progress judgment, latest-response correctness,
evidence category, completion state, reasoning sufficiency, confidence, and
brief evidence-based reasons.
""",
                ),
                (
                    "human",
                    """
ORIGINAL QUESTION
{original_question}

TOPIC
{topic}

SUBTOPIC
{subtopic}

STUDENT AGE
{student_age}

PLANNER OUTPUT
{planner_output}

PREVIOUS ERRORS
{previous_errors}

VERIFIED MATHEMATICAL EVIDENCE
{verified_math_evidence}

RETEACHING
{reteaching}

CURRENT TEACHER-STUDENT DIALOGUE
{conversation_history}

Decide whether interactive teaching should continue or the learner is ready
for the formal assessment.
""",
                ),
            ]
        )

    def judge(self, progress_input: TeachingProgressInput) -> TeachingProgressOutput:
        chain = self.prompt | self.structured_model
        return chain.invoke(
            {
                "original_question": progress_input.original_question,
                "topic": progress_input.topic,
                "subtopic": progress_input.subtopic or "Not provided",
                "student_age": progress_input.student_age,
                "planner_output": (
                    progress_input.planner_output.model_dump()
                    if progress_input.planner_output
                    else None
                ),
                "previous_errors": progress_input.previous_errors,
                "verified_math_evidence": progress_input.verified_math_evidence,
                "reteaching": progress_input.reteaching,
                "conversation_history": [
                    turn.model_dump() for turn in progress_input.conversation_history
                ],
            }
        )
