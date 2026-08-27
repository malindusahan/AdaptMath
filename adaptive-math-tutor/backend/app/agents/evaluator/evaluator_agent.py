from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.assessment import (
    AssessmentGenerationInput,
    AssessmentGenerationOutput,
)
from app.schemas.evaluator import (
    EvaluatedAnswer,
    EvaluatorInput,
    EvaluatorOutput,
)


class EvaluationJudgment(BaseModel):
    """Internal LLM judgment before the workflow's reteaching policy is applied."""

    correct_answers: list[EvaluatedAnswer] = Field(default_factory=list)
    wrong_answers: list[EvaluatedAnswer] = Field(default_factory=list)
    identified_errors: list[str] = Field(default_factory=list)
    overall_feedback: str = Field(..., min_length=1)


class EvaluatorAgent:
    """
    GenAI Evaluator for AdaptMath.

    Responsibilities:
    - after interactive teaching is complete, generate exactly three assessment
      questions that check understanding of what was taught;
    - evaluate the learner's three answers;
    - diagnose supported mathematical errors;
    - produce evidence for Hamooth's Memory component and subsequent adaptive
      reteaching.

    It does not select the MathDial pedagogical move. Omash owns that decision.
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
            EvaluationJudgment,
            method="json_schema",
        )
        self.assessment_structured_model = self.model.with_structured_output(
            AssessmentGenerationOutput,
            method="json_schema",
        )

        self.assessment_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the assessment-generation responsibility of the Evaluator Agent in
AdaptMath.

Interactive teacher-student tutoring has just reached a natural assessment
point. Generate EXACTLY THREE short assessment questions that provide useful
evidence of whether the learner understood the mathematics that was taught.

Use the original problem and the actual tutoring dialogue. Do not merely ask
for the same final answer three times. Choose three questions that, together,
make a meaningful understanding check for this specific dialogue. They may
check reasoning, a relevant step, explanation, or nearby application when that
is appropriate, but do not use a fixed question-type template.

Rules:
- age-appropriate wording;
- mathematically objective and unambiguous;
- exactly three questions;
- unique question_id values;
- include a correct expected_answer for internal evaluation;
- never place the expected answer inside the learner-visible question;
- do not introduce concepts that were not established by the original problem
  or the teaching dialogue;
- if this is a reteaching cycle, focus the new assessment on whether the prior
  difficulty has actually been resolved without simply repeating the old
  assessment verbatim.

Do not teach the learner and do not select a pedagogical move.
Return only the structured assessment output.
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

COMPLETED INTERACTIVE TEACHING DIALOGUE
{conversation_history}

Generate the three-question understanding assessment now.
""",
                ),
            ]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Evaluator Agent in AdaptMath, an adaptive mathematics tutoring
research system.

Evaluate every learner answer to the supplied three assessment questions.
Compare the question, learner answer, and internal expected answer, but judge
mathematical meaning rather than simple string equality.

For correct responses, preserve question_id/question/student_answer/
expected_answer, set is_correct=true, give concise feedback, and use
identified_error=null.

For incorrect responses, preserve the same fields, set is_correct=false,
explain the mathematical issue, and provide a concise identified_error when it
is supported by the response. Summarize supported errors in identified_errors.
Do not invent misconceptions.

Every supplied answer must be evaluated exactly once. Do not teach a new
lesson, choose a pedagogical move, route agents, or alter the student's answer.
Return only the structured evaluation judgment.
""",
                ),
                (
                    "human",
                    """
ORIGINAL MATHEMATICS QUESTION
{original_question}

TOPIC
{topic}

SUBTOPIC
{subtopic}

ASSESSMENT QUESTIONS AND INTERNAL REFERENCE ANSWERS
{assessment_questions}

STUDENT ANSWERS
{student_answers}

Evaluate all three learner answers.
""",
                ),
            ]
        )

    def _generate_assessment_once(
        self,
        assessment_input: AssessmentGenerationInput,
    ) -> AssessmentGenerationOutput:
        chain = self.assessment_prompt | self.assessment_structured_model
        return chain.invoke(
            {
                "original_question": assessment_input.original_question,
                "topic": assessment_input.topic,
                "subtopic": assessment_input.subtopic or "Not provided",
                "student_age": assessment_input.student_age,
                "planner_output": (
                    assessment_input.planner_output.model_dump()
                    if assessment_input.planner_output
                    else None
                ),
                "previous_errors": assessment_input.previous_errors,
                "reteaching": assessment_input.reteaching,
                "conversation_history": [
                    turn.model_dump()
                    for turn in assessment_input.conversation_history
                ],
            }
        )

    def generate_assessment(
        self,
        assessment_input: AssessmentGenerationInput,
    ) -> AssessmentGenerationOutput:
        """Generate exactly three valid assessment questions, retrying once."""
        last_error: Exception | None = None
        for _ in range(2):
            try:
                result = self._generate_assessment_once(assessment_input)
                return AssessmentGenerationOutput.model_validate(result)
            except Exception as exc:  # model/schema failure is retried once
                last_error = exc

        raise RuntimeError(
            "Evaluator assessment generation failed after two attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def evaluate(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        assessment_questions = [
            question.model_dump() for question in evaluator_input.assessment_questions
        ]
        student_answers = [
            answer.model_dump() for answer in evaluator_input.student_answers
        ]

        question_ids = {
            question.question_id for question in evaluator_input.assessment_questions
        }
        answer_ids = {answer.question_id for answer in evaluator_input.student_answers}

        if answer_ids != question_ids:
            unknown = answer_ids - question_ids
            missing = question_ids - answer_ids
            raise ValueError(
                "Student answers must match all assessment question IDs exactly. "
                f"Unknown={sorted(unknown)}, missing={sorted(missing)}"
            )

        chain = self.prompt | self.structured_model
        judgment = chain.invoke(
            {
                "original_question": evaluator_input.original_question,
                "topic": evaluator_input.topic,
                "subtopic": evaluator_input.subtopic or "Not provided",
                "assessment_questions": assessment_questions,
                "student_answers": student_answers,
            }
        )

        evaluated_answers = judgment.correct_answers + judgment.wrong_answers
        evaluated_ids = [answer.question_id for answer in evaluated_answers]

        if len(evaluated_ids) != len(set(evaluated_ids)):
            raise RuntimeError("Evaluator returned duplicate question evaluations.")
        if set(evaluated_ids) != answer_ids:
            raise RuntimeError(
                "Evaluator did not evaluate every submitted learner answer exactly once."
            )

        # User-approved workflow policy: any mathematically incorrect assessment
        # answer starts another interactive reteaching cycle.
        needs_reteaching = len(judgment.wrong_answers) > 0

        return EvaluatorOutput(
            correct_answers=judgment.correct_answers,
            wrong_answers=judgment.wrong_answers,
            identified_errors=judgment.identified_errors,
            needs_reteaching=needs_reteaching,
            overall_feedback=judgment.overall_feedback,
        )
