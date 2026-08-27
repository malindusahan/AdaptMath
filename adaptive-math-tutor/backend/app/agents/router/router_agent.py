from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.router import RouterInput, RouterOutput


ROUTER_VERSION = "5.0-development"


class RouterAgent:
    """
    GenAI-driven pre-tutoring workflow router.

    The Router decides only whether a separate Planner stage adds
    value before Tutor. Every tutoring turn is assessed afterwards,
    so assessment is not a Router decision.

    No manually coded numerical complexity thresholds, learner-state
    thresholds, or fixed problem-to-route mappings are used.
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
                RouterOutput,
                method="json_schema",
            )
        )

        self.prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are the intelligent pre-tutoring workflow router for
AdaptMath, an adaptive mathematics tutoring system for learners
aged 8 to 18.

Your only responsibility is to decide whether a separate
pedagogical planning stage is useful before the Tutor responds.

AVAILABLE WORKFLOWS

direct_tutor
- Send the problem directly to the Tutor Agent.
- Appropriate when the problem can reasonably be taught in one
  coherent tutoring interaction without a separate planning stage.

planned_tutor
- Send the problem to the Planner Agent first and then to Tutor.
- Appropriate when explicit pedagogical structuring is likely to
  improve the tutoring interaction, for example by organizing
  concepts, dependencies, intermediate reasoning, likely
  difficulties, or explanation sequence.

IMPORTANT SYSTEM BOUNDARIES

- Every Tutor response is followed by exactly three assessment
  questions and learner evaluation. Assessment is universal and is
  NOT part of your routing decision.
- The pedagogical move (telling, focus, generic, or probing) is
  selected by an external move-selector component. Do not select,
  replace, or infer that move.
- Do not select a reteaching strategy.

DECISION PROCESS

Use both problem-side and learner-side evidence.

Problem-side evidence may include:
- the actual mathematics question,
- topic and subtopic,
- mathematical structure,
- conceptual demands,
- dependencies between reasoning steps,
- whether explicit sequencing would help,
- the learned continuous mathematical complexity score.

Learner-side evidence may include:
- age as contextual information only,
- relevant learner history,
- previous errors or misconceptions,
- previous teaching/move history if supplied.

Treat missing learner history as UNKNOWN. It is not evidence that
the learner is strong, weak, or that planning is unnecessary.

COMPLEXITY SCORE

The complexity_score is a continuous estimate between 0 and 1
produced by a trained machine-learning model.

Use it as continuous evidence together with the actual problem.
Do not create numerical thresholds.
Do not convert it to Easy, Medium, or Hard categories.
Do not describe it as low, moderate, or high unless calibrated
category boundaries are explicitly supplied; none are supplied here.
Do not let the score automatically determine the route.

LEARNER EVIDENCE

Learner difficulty and mathematical problem complexity are
different concepts. Relevant learner evidence may make planning
useful even for a mathematically simple-looking task, while strong
relevant evidence may reduce the value of extra planning. These are
contextual considerations, not deterministic mappings.

GENERAL RESTRICTIONS

Do not solve the mathematics problem.
Do not teach the learner.
Do not create the pedagogical plan yourself.
Do not assess the learner.
Do not select the pedagogical move.
Do not invent learner history, errors, competence, or strategies.

Return a brief reason based only on supplied evidence. Explain why
a separate Planner stage does or does not add useful structure for
this particular problem and learner context.
""",
                    ),
                    (
                        "human",
                        """
Mathematics question:
{question}

Learned problem complexity score:
{complexity_score}

Student age:
{student_age}

Topic:
{topic}

Subtopic:
{subtopic}

Relevant learner history:
{relevant_history}

Previous errors:
{previous_errors}

Previous teaching strategies/moves:
{previous_strategies}

Select exactly one workflow:
- direct_tutor
- planned_tutor
""",
                    ),
                ]
            )
        )

    def route(
        self,
        router_input: RouterInput,
    ) -> RouterOutput:
        profile = router_input.profile
        memory = router_input.memory

        history = [
            item.model_dump()
            for item in memory.relevant_history
        ]

        chain = (
            self.prompt
            | self.structured_model
        )

        result = chain.invoke(
            {
                "question":
                    router_input.question,

                "complexity_score":
                    router_input.complexity_score,

                "student_age":
                    profile.age,

                "topic":
                    memory.topic,

                "subtopic":
                    (
                        memory.subtopic
                        if memory.subtopic
                        else "Not provided"
                    ),

                "relevant_history":
                    (
                        history
                        if history
                        else [
                            "No relevant history supplied"
                        ]
                    ),

                "previous_errors":
                    (
                        memory.previous_errors
                        if memory.previous_errors
                        else [
                            "No previous errors supplied"
                        ]
                    ),

                "previous_strategies":
                    (
                        memory.previous_strategies
                        if memory.previous_strategies
                        else [
                            "No previous strategy/move history supplied"
                        ]
                    ),
            }
        )

        return result
