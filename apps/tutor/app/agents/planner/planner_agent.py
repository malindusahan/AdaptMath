from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.planner import PlannerInput, PlannerOutput


class PlannerAgent:
    """
    GenAI pedagogical planner used only when Router selects planned_tutor.

    The Planner creates an overall instructional plan. It does not select the
    per-turn MathDial pedagogical move; Omash's component owns that decision.
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
            PlannerOutput,
            method="json_schema",
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the planning component of AdaptMath, an interactive mathematics
 tutoring research system for learners aged 8 to 18.

Create an overall pedagogical plan for the Tutor. The Tutor will teach through
multiple short teacher-student turns. A separate external component owned by
Omash selects the MathDial pedagogical move for each teacher turn. Do NOT
select, recommend, predict, or hard-code telling, focus, generic, or probing.

Do not directly teach the learner and do not simply reveal the final answer.
Use only supplied learner information. Do not invent learner weaknesses,
history, or previous strategies.

The complexity score is a continuous problem-level estimate from a trained ML
model. Treat it as contextual evidence; do not convert it into fixed Easy,
Medium, or Hard categories and do not use fixed numerical thresholds.

Produce:
- the learning goal,
- required mathematical concepts,
- a sensible interactive teaching sequence,
- likely difficulties supported by the supplied context,
- guidance that helps the Tutor scaffold reasoning across multiple turns.

The plan should support interaction, not a one-message worked-solution dump.
""",
                ),
                (
                    "human",
                    """
Mathematics question:
{question}

Topic:
{topic}

Subtopic:
{subtopic}

Student age:
{student_age}

Problem complexity score:
{complexity_score}

Previous errors:
{previous_errors}

Previous teaching strategies/moves:
{previous_strategies}

Create the overall interactive tutoring plan.
""",
                ),
            ]
        )

    def create_plan(self, planner_input: PlannerInput) -> PlannerOutput:
        chain = self.prompt | self.structured_model
        return chain.invoke(
            {
                "question": planner_input.question,
                "topic": planner_input.topic,
                "subtopic": planner_input.subtopic or "Not provided",
                "student_age": planner_input.student_age,
                "complexity_score": planner_input.complexity_score,
                "previous_errors": (
                    planner_input.previous_errors
                    if planner_input.previous_errors
                    else ["No previous errors supplied"]
                ),
                "previous_strategies": (
                    planner_input.previous_strategies
                    if planner_input.previous_strategies
                    else ["No previous strategies supplied"]
                ),
            }
        )
