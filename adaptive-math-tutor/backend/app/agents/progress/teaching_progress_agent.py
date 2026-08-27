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
            max_retries=2,
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

This is NOT a pedagogical-move selector. Never choose, recommend, replace, or
infer one of the MathDial moves (telling, focus, generic, probing). Omash's
external component owns that decision.

Judge from the actual dialogue evidence. The goal is not merely that the
teacher has stated a correct solution. The dialogue should show that the
student has engaged with the important logic of the problem sufficiently for a
formal understanding check to be meaningful.

Do not treat a bare agreement such as "yes", "okay", or simple repetition of
a revealed final answer as strong evidence of understanding. Consider whether
the learner has demonstrated the relevant reasoning, next step, explanation,
or correction in context.

Do not use fixed turn counts, numerical thresholds, age brackets, or static
problem-type rules. A short dialogue can be ready when the evidence supports
it; a longer dialogue can still need teaching.

For reteaching, judge whether the misconception/error that motivated
reteaching has been addressed sufficiently to proceed to a new assessment.

Return only the structured status and a brief evidence-based reason.
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
                "reteaching": progress_input.reteaching,
                "conversation_history": [
                    turn.model_dump() for turn in progress_input.conversation_history
                ],
            }
        )
