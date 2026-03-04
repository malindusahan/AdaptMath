import logging
import re

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


logger = logging.getLogger(__name__)


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
            timeout=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
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
- assess TRANSFER rather than surface recall: do not copy the original story,
  entities, wording, or numerical values, and do not create the original
  problem again with only one number changed;
- use genuinely different numerical values from every number in the original
  question;
- make the three items meaningfully different from one another. Across the set,
  use varied evidence demands such as a new-context application, explanation
  or justification, error analysis, representation, or reverse reasoning. Do
  not produce three successive fragments of the original solution procedure;
- at least two items must require applying the learned mathematics in a new
  context or structurally different situation, not recalling facts from the
  original story;
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

REJECTION FEEDBACK FROM A PREVIOUS DRAFT
{rejection_feedback}

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
        *,
        rejection_feedback: str | None = None,
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
                "rejection_feedback": rejection_feedback or "None; this is the first draft.",
            }
        )

    @staticmethod
    def _number_literals(text: str) -> set[str]:
        return set(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.lower()))

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "does",
            "each", "for", "from", "had", "has", "have", "how", "if",
            "in", "into", "is", "it", "many", "of", "on", "or", "that",
            "the", "there", "these", "this", "to", "use", "was", "were",
            "what", "when", "which", "why", "will", "with", "would", "you",
            "your",
        }
        return {
            token
            for token in re.findall(r"[a-z]+", text.lower())
            if len(token) > 2 and token not in stopwords
        }

    @classmethod
    def _validate_assessment_variability(
        cls,
        assessment_input: AssessmentGenerationInput,
        result: AssessmentGenerationOutput,
    ) -> None:
        original_numbers = cls._number_literals(assessment_input.original_question)
        original_tokens = cls._content_tokens(assessment_input.original_question)
        normalized_questions: list[str] = []
        question_token_sets: list[set[str]] = []

        for item in result.assessment_questions:
            normalized = " ".join(item.question.lower().split())
            if normalized in normalized_questions:
                raise ValueError("Assessment questions must have distinct wording.")
            normalized_questions.append(normalized)

            reused_numbers = original_numbers & cls._number_literals(item.question)
            if reused_numbers:
                raise ValueError(
                    "Assessment reused original numerical values: "
                    f"{sorted(reused_numbers)}."
                )

            tokens = cls._content_tokens(item.question)
            shared = tokens & original_tokens
            if len(shared) >= 3 and len(shared) / max(1, len(tokens)) >= 0.5:
                raise ValueError(
                    "Assessment question is too close to the original story; "
                    f"shared content={sorted(shared)}."
                )
            question_token_sets.append(tokens)

        for left_index, left in enumerate(question_token_sets):
            for right in question_token_sets[left_index + 1 :]:
                shared = left & right
                smaller = min(len(left), len(right))
                if smaller and len(shared) >= 3 and len(shared) / smaller >= 0.7:
                    raise ValueError(
                        "Assessment questions are too similar to one another; "
                        f"shared content={sorted(shared)}."
                    )

    def generate_assessment(
        self,
        assessment_input: AssessmentGenerationInput,
    ) -> AssessmentGenerationOutput:
        """Generate exactly three valid assessment questions, retrying once.

        Variability is a quality preference, not a reason to strand an otherwise
        complete lesson.  When both structured Gemini drafts are valid but the
        conservative local diversity heuristic still objects, retain the second
        (feedback-informed) draft instead of converting that disagreement into a
        fatal pipeline error.  Provider and schema failures remain fatal when no
        usable three-question assessment was produced.
        """
        last_error: Exception | None = None
        rejection_feedback: str | None = None
        structured_candidates: list[AssessmentGenerationOutput] = []
        for _ in range(2):
            try:
                result = self._generate_assessment_once(
                    assessment_input,
                    rejection_feedback=rejection_feedback,
                )
                validated = AssessmentGenerationOutput.model_validate(result)
                structured_candidates.append(validated)
                self._validate_assessment_variability(
                    assessment_input,
                    validated,
                )
                return validated
            except Exception as exc:  # model/schema failure is retried once
                last_error = exc
                rejection_feedback = (
                    "The previous draft was rejected: "
                    f"{type(exc).__name__}: {exc}. Generate a substantially "
                    "different three-question transfer assessment that obeys "
                    "every variability rule."
                )

        if structured_candidates:
            logger.warning(
                "assessment_variability_fallback candidate_count=%d "
                "last_issue_type=%s",
                len(structured_candidates),
                type(last_error).__name__ if last_error is not None else "unknown",
            )
            return structured_candidates[-1]

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
