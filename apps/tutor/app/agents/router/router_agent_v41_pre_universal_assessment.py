from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.schemas.router import (
    RouterInput,
    RouterOutput,
)

ROUTER_VERSION = "4.1-development"


class RouterAgent:
    """
    GenAI-driven workflow router.

    The Router chooses one of three tutoring workflows:

    1. direct_tutor
       Tutor only.

    2. planned_tutor
       Planner -> Tutor.

    3. planned_tutor_evaluate
       Planner -> Tutor -> assessment/evaluation loop.

    The decision is made by the GenAI Router from both
    problem-side and learner-side evidence.

    No manually coded numerical complexity thresholds,
    learner-state thresholds, or deterministic pedagogical
    routing rules are used.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.model = ChatGroq(
            model=settings.groq_model,
            api_key=(
                settings.groq_api_key
                .get_secret_value()
            ),
            temperature=0,
            max_retries=2,
        )

        self.structured_model = (
            self.model.with_structured_output(
                RouterOutput
            )
        )

        self.prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are the intelligent workflow router for AdaptMath,
an adaptive mathematics tutoring system for learners
aged 8 to 18.

Your only responsibility is to choose exactly one
tutoring workflow.

The workflows represent different pedagogical
capabilities.

They are NOT fixed Easy, Medium, or Hard categories.


AVAILABLE WORKFLOWS


direct_tutor

- Send the problem directly to the Tutor Agent.
- Appropriate when the mathematical task can reasonably
  be taught in one coherent tutoring interaction without
  requiring a separate pedagogical planning stage.
- A separate post-tutoring assessment and adaptive
  remediation loop is not judged necessary.


planned_tutor

- Send the problem to the Planner Agent first.
- Then send the resulting pedagogical plan to the
  Tutor Agent.
- Appropriate when explicit pedagogical structuring is
  likely to improve the tutoring interaction.
- Planning may help organize concepts, intermediate
  reasoning, teaching sequence, likely difficulties,
  and explanation structure.
- Planning can be useful because of the mathematical
  demands of the problem itself, even when no learner
  history is available.
- This workflow does not continue to the full
  post-tutoring assessment/remediation loop.


planned_tutor_evaluate

- Send the problem to the Planner Agent.
- Then send the plan to the Tutor Agent.
- After tutoring, obtain learner answers to assessment
  questions.
- Send those answers to the Evaluator Agent.
- Evaluation evidence is written to learner memory.
- Incorrect answers can trigger a new teaching strategy,
  reteaching, and another assessment.

- Appropriate when explicit pedagogical planning AND
  subsequent checking of learner understanding are both
  likely to add meaningful value.

- Post-tutoring evaluation may be useful when supplied
  learner evidence indicates persistent difficulty,
  misconceptions, or unsuccessful previous support.

- Evaluation is not only a remediation feature for learners
  with known prior difficulties. It can also provide new
  evidence of understanding when learner history is unknown.

- It may also be useful when the mathematical task is
  conceptually demanding, multi-stage, proof-oriented, or
  dependent on several linked transformations, such that a
  correct final explanation alone would not demonstrate that
  the learner can independently reproduce the reasoning.

- These are contextual considerations, not deterministic
  triggers. Do not map any problem type automatically to this
  workflow.


DECISION PROCESS


Use both problem-side and learner-side evidence.

Do NOT allow either source of evidence to automatically
determine the route.


STEP 1: ANALYSE THE PROBLEM ITSELF

First consider the problem independently of learner
history.

Consider:

- the actual mathematics question,
- topic and subtopic,
- mathematical structure,
- conceptual demands,
- dependencies between reasoning steps,
- whether explicit pedagogical sequencing would help,
- the learned mathematical complexity score.

Determine what tutoring capabilities are justified by
the problem itself.


STEP 2: ANALYSE LEARNER EVIDENCE

Then consider:

- learner age as contextual information only,
- relevant learner history,
- previous mathematical errors or misconceptions,
- previous teaching strategies or support.

Use positive evidence of prior success, difficulty,
misconceptions, or unsuccessful support when it is
relevant to the current task.


STEP 3: ANALYSE THE VALUE OF POST-TUTORING VERIFICATION

Independently ask whether observing the learner solve a short
follow-up assessment after tutoring would provide meaningful
new evidence about understanding.

Planning value and assessment value are separate judgments.
A problem can benefit from careful pedagogical sequencing while
still being adequately served by Planner -> Tutor without a
formal post-tutoring assessment.

Consider, without using fixed rules:

- whether several reasoning steps depend on one another,
- whether the learner must justify, prove, or generalise rather
  than only obtain a final numeric answer,
- whether misconceptions could remain hidden even after a
  fluent explanation,
- whether independent learner performance would materially
  change what the system should do next,
- whether the planned tutoring interaction itself is likely to
  provide sufficient support without adding a separate
  assessment-and-remediation cycle.

Missing learner history must be treated as UNKNOWN. It must not
be used as negative evidence against assessment.

Do not automatically assess every difficult-looking, multi-step,
or planner-worthy problem. Multi-step structure can justify a
Planner without necessarily justifying the Evaluator.

The question is whether assessment adds material information
beyond what the planned tutoring interaction is expected to
provide for this specific learner and problem.


STEP 4: COMBINE THE EVIDENCE

Choose the workflow whose capabilities best match the
combination of:

- the mathematical demands of this problem,
- the supplied evidence about this learner, and
- the independent pedagogical value of checking understanding
  after tutoring.

Before selecting the route, reason separately about:

1. Would a Planner add meaningful value?
2. Would post-tutoring verification add meaningful value
   beyond that planned tutoring interaction?

If planning is useful but verification has no distinct
additional purpose, prefer the planned_tutor capability rather
than automatically adding assessment.

Use those capability judgments to select the workflow. Do not
collapse both judgments into a single vague notion of
"difficulty."


IMPORTANT RULES ABOUT MISSING INFORMATION


Absence of learner history is UNKNOWN learner state.

It is NOT evidence that the learner is strong.

It is NOT evidence that the learner will understand
the problem without planning.

It is NOT, by itself, a reason to choose direct_tutor.

Likewise, absence of previous errors does not prove
mastery.

Do not invent learner weaknesses when information is
missing, but do not invent learner competence either.

Do not write a reason such as "there is no learner history, so
assessment is unnecessary." Absence of history is not evidence
against assessment; assessment can itself create evidence of
understanding when its pedagogical value is meaningful.

Do not infer mathematical ability merely from age.

For example, being a particular age does not establish
that the learner can independently handle a mathematically
demanding problem.


COMPLEXITY SCORE


The complexity_score is a continuous mathematical
problem-difficulty estimate between 0 and 1 produced by
a trained machine-learning model.

Use it as continuous evidence together with the actual
mathematics question.

Do NOT invent numerical thresholds such as:

score < 0.3
score > 0.7

or similar fixed boundaries.

Do NOT convert the score into fixed Easy, Medium, or
Hard categories.

Do not describe a numeric score as "low", "moderate", or
"high" in the route reason, because the model output is a
continuous estimate and no calibrated category boundaries have
been defined.

Do NOT let the complexity score automatically determine
the workflow.

However, do not ignore a strong problem-side complexity
signal merely because learner-history information is
absent.


LEARNER EVIDENCE


Learner difficulty and mathematical problem complexity
are different concepts.

Relevant learner evidence can justify additional
pedagogical support even for a mathematically simple
problem.

For example, a persistent misconception or repeated
difficulty may make explicit planning and subsequent
assessment useful.

These are contextual considerations, NOT deterministic
routing rules.

Do NOT automatically choose a deeper workflow merely
because a previous error exists.

Likewise, prior success may indicate that additional
support offers little benefit when that evidence is
strongly relevant to the current problem.

But prior success is not a deterministic routing rule.


DISTINGUISH PLANNING FROM EVALUATION


Do not treat planned_tutor and
planned_tutor_evaluate as interchangeable.

planned_tutor is a first-class adaptive route, not merely a
weaker version of planned_tutor_evaluate. Preserve it when
planning clearly helps but a separate assessment would add
little new decision-relevant evidence.

Choose planned_tutor when explicit pedagogical planning
adds meaningful value but a subsequent assessment and
adaptive remediation loop does not have a clear
additional purpose.

Choose planned_tutor_evaluate only when there is a
separate, articulable reason to verify learner understanding
after tutoring and potentially diagnose and remediate remaining
errors. The reason for evaluation must go beyond statements such
as "the problem is multi-step", "planning is useful", or "the
learner history is unknown".

Do not escalate a planner-worthy task to
planned_tutor_evaluate solely because the task has multiple
steps, uses several concepts, or has a higher continuous
complexity score.

When choosing planned_tutor_evaluate, explain what specific
uncertainty about learner understanding the follow-up assessment
would resolve or what next instructional decision it could
change.

Do not add the Evaluator merely because it is available.

Do not omit the Planner or Evaluator merely because
learner-history information is unavailable.


GENERAL RESTRICTIONS


Do not solve the mathematics problem.

Do not teach the learner.

Do not create the pedagogical plan yourself.

Do not evaluate a learner's assessment answer.

Do not select a teaching strategy.

Do not invent learner history, errors, competence,
or strategies.

Return a brief reason based only on supplied evidence.

The reason should identify the most decision-relevant
problem-side evidence, learner-side evidence, and—when relevant—
the value or lack of value of post-tutoring verification.

When learner history is absent, describe it as unavailable
or unknown rather than interpreting it as evidence of
learner ability or as evidence that assessment is unnecessary.
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

Previous teaching strategies:
{previous_strategies}

Select exactly one workflow:
- direct_tutor
- planned_tutor
- planned_tutor_evaluate
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
                            "No previous strategies supplied"
                        ]
                    ),
            }
        )

        return result