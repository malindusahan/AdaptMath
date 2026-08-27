from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.strategy import StrategyInput, TeachingStrategy


class StrategyAgent:
    """
    LEGACY / INACTIVE local strategy selector.

    The active tutoring workflow no longer instantiates this class.
    Pedagogical moves are selected by Omash's external move-selector
    component and supplied to Tutor.

    Historical implementation:

    The agent selects a pedagogical strategy from the current
    learner/problem evidence. It does not use fixed strategy
    lookup tables, complexity thresholds, age brackets, or
    hard-coded mappings from an error type to a strategy.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=(
                settings.gemini_api_key
                .get_secret_value()
            ),
            temperature=0,
            max_retries=2,
        )

        self.structured_model = (
            self.model.with_structured_output(
                TeachingStrategy,
                method="json_schema",
            )
        )

        self.prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are the Strategy Agent in AdaptMath, an adaptive
mathematics tutoring system for learners aged 8 to 18.

Your responsibility is to select a teaching strategy for a
reteaching attempt after the learner has completed an
assessment and the Evaluator has produced evidence of one or
more incorrect answers.

Use the supplied evidence as context:

- the original mathematics question,
- topic and subtopic,
- learner age,
- learned continuous problem-complexity score,
- relevant learner history,
- correct and incorrect assessment evidence,
- identified mathematical errors or misconceptions,
- strategies previously used with the learner.

Choose a strategy that is pedagogically appropriate for this
specific situation and provide concrete guidance that another
Tutor Agent can follow.

The complexity score is continuous evidence produced by a
trained machine-learning model. Do not convert it into fixed
Easy/Medium/Hard categories and do not use numerical routing
thresholds.

Treat age as contextual information only. Do not use fixed
age-bracket strategy rules.

Previous strategies are evidence, not a deterministic rule.
When a previous approach appears not to have resolved the
current difficulty, consider a meaningfully different way of
explaining or representing the mathematics. Do not claim that
a previous strategy failed unless the supplied evidence
supports that conclusion.

Do not use a fixed mapping such as:
"fraction error -> worked example" or
"algebra error -> visual explanation".
Select the strategy from the complete supplied context.

Do not solve the mathematics problem.
Do not evaluate the learner's answers again.
Do not choose a workflow route.
Do not decide whether reteaching is required.
Do not invent learner history, errors, or prior strategies.

Return one structured TeachingStrategy with:
- an optional concise strategy_id,
- a clear strategy_name,
- a short strategy_description,
- actionable teaching_guidance for the Tutor.
""",
                    ),
                    (
                        "human",
                        """
Student ID:
{student_id}

Original mathematics question:
{question}

Topic:
{topic}

Subtopic:
{subtopic}

Student age:
{student_age}

Learned problem complexity score:
{complexity_score}

Relevant learner history:
{relevant_history}

Correct assessment evidence:
{correct_answers}

Incorrect assessment evidence:
{wrong_answers}

Identified mathematical errors:
{identified_errors}

Previous teaching strategies:
{previous_strategies}

Select the most appropriate reteaching strategy for the
Tutor Agent to use next.
""",
                    ),
                ]
            )
        )

    def select_strategy(
        self,
        strategy_input: StrategyInput,
    ) -> TeachingStrategy:
        chain = (
            self.prompt
            | self.structured_model
        )

        result = chain.invoke(
            {
                "student_id":
                    strategy_input.student_id,

                "question":
                    strategy_input.question,

                "topic":
                    strategy_input.topic,

                "subtopic":
                    (
                        strategy_input.subtopic
                        if strategy_input.subtopic
                        else "Not provided"
                    ),

                "student_age":
                    strategy_input.student_age,

                "complexity_score":
                    strategy_input.complexity_score,

                "relevant_history":
                    (
                        strategy_input.relevant_history
                        if strategy_input.relevant_history
                        else [
                            "No relevant history supplied"
                        ]
                    ),

                "correct_answers":
                    (
                        strategy_input.correct_answers
                        if strategy_input.correct_answers
                        else [
                            "No correct-answer evidence supplied"
                        ]
                    ),

                "wrong_answers":
                    strategy_input.wrong_answers,

                "identified_errors":
                    (
                        strategy_input.identified_errors
                        if strategy_input.identified_errors
                        else [
                            "No specific error label supplied"
                        ]
                    ),

                "previous_strategies":
                    (
                        strategy_input.previous_strategies
                        if strategy_input.previous_strategies
                        else [
                            "No previous strategies supplied"
                        ]
                    ),
            }
        )

        return result
