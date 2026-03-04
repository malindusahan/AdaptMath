"""Authoritative move-to-language contract for the learner-facing Tutor.

This module owns realization only. It does not inspect learner state, choose a
move, alter a policy probability, or participate in Turn-LinTS updates.
"""

from __future__ import annotations

from typing import Final, TypedDict


class MoveRealizationProfile(TypedDict):
    purpose: str
    question_usage: str
    hint_detail_level: str
    explicit_explanation_level: str
    target_sentences: str
    contract: str


MOVE_REALIZATION_PROFILES: Final[dict[str, MoveRealizationProfile]] = {
    "generic": {
        "purpose": "Support, acknowledge, orient, or transition without diagnosing or teaching the missing method.",
        "question_usage": "Optional and uncommon; never automatically diagnostic.",
        "hint_detail_level": "Broad orientation only; no targeted corrective hint or worked step.",
        "explicit_explanation_level": "Low: do not explain the solution method.",
        "target_sentences": "Usually 1-3 useful sentences.",
        "contract": """
PRIMARY SPEECH ACT: context-aware support, acknowledgement, orientation, or
transition. Produce a useful, non-vacuous statement tied to the current problem
or the learner's progress in 1-3 useful, problem-aware sentences. Refer to an
actual quantity, expression, mathematical object, or contextual feature from
the problem or latest learner message; do not substitute generic talk about
"the values," "the relationship," or "the next step." When the learner has
made concrete progress, acknowledge that exact progress instead of resetting
the dialogue. Do not give a
targeted hint, formula, worked step, solution, or explanation of the missing
method. Avoid vacuous praise. Do not default to interrogating the learner; a
question is optional and must not turn generic into probing. Use the dialogue
history to avoid repeating stock phrases such as "Great", "Okay", or "Let's
continue", including when generic is selected on consecutive turns.
""".strip(),
    },
    "probing": {
        "purpose": "Elicit the learner's mathematical reasoning, interpretation, or chosen method.",
        "question_usage": "One clear, specific main reasoning question.",
        "hint_detail_level": "Minimal setup; the learner supplies the important step.",
        "explicit_explanation_level": "Low: never answer the probe before asking it.",
        "target_sentences": "Usually 1-3 sentences.",
        "contract": """
PRIMARY SPEECH ACT: elicit learner reasoning, interpretation, recall, or chosen
method. Ask exactly ONE clear, specific main mathematical reasoning question;
one short setup sentence is allowed. The learner must supply an important step.
Build the question from the learner's latest expression, operation, sign,
interpretation, or stated uncertainty whenever one is available. Name the
actual mathematical object being discussed; never ask vaguely how "the given
information" connects to "the quantity." 
Do not answer the question, solve or work that step, or explain the missing
method before asking. Avoid vague stock questions and several unrelated or
sequential questions.
""".strip(),
    },
    "focus": {
        "purpose": "Direct attention to the specific relation, error, quantity, representation, or next substep that matters.",
        "question_usage": "Not required; any question is secondary to the targeted cue.",
        "hint_detail_level": "Targeted and more direct than probing, while leaving meaningful work.",
        "explicit_explanation_level": "Medium-low: identify what matters without fully working it out.",
        "target_sentences": "Usually 2-4 sentences when explanation is useful.",
        "contract": """
PRIMARY SPEECH ACT: direct attention to ONE specific mathematical relation,
error, quantity, representation, clue, or immediate sub-step. Give a targeted
cue, observation, or directive that is more direct than probing but less
explicit than telling. Quote or restate the specific sign, term, expression,
quantity, or interpretation that matters now. If the learner proposed a useful
step, preserve it and focus attention on evaluating or interpreting that step
rather than sending them back to the beginning. Identify exactly what matters
now, and leave meaningful mathematical work
for the learner. It does not have to be a question; any short
check is secondary to the targeted cue. Do not fully work out the answer,
become a diagnostic interview, or collapse into generic encouragement.
""".strip(),
    },
    "telling": {
        "purpose": "Explicitly teach the relevant fact, method, explanation, representation, or worked step.",
        "question_usage": "Optional only after the instruction or explanation.",
        "hint_detail_level": "Most direct and instructional; equations are encouraged when helpful.",
        "explicit_explanation_level": "High: supply and connect the needed mathematical information.",
        "target_sentences": "Usually 2-5 sentences; slightly longer only when clarity requires it.",
        "contract": """
PRIMARY SPEECH ACT: explicitly teach the relevant fact, method, representation,
explanation, missing step, or worked clarification. Put understandable
mathematical information FIRST, connect it to the current problem, and use the
actual problem values, equations, or step-by-step wording when useful. Usually
use 2-5 sentences; a slightly longer response is allowed only when clarity
requires it. Teach first; only after the instruction or explanation may you
optionally ask one short follow-up. Never produce only a vague hint or question,
and never use generic substitute text such as "Represent the requested quantity
using the given values and the relationship between them." Do not be terse,
disguise telling as leading questions, or ask what to do without first supplying
the needed instruction. When the learner already supplied a usable expression
or operation, work or explain that exact step instead of restating the general
method.
""".strip(),
    },
}

MOVE_REALIZATION_CONTRACT: Final[dict[str, str]] = {
    move: profile["contract"] for move, profile in MOVE_REALIZATION_PROFILES.items()
}


def _profile(selected_move: str) -> MoveRealizationProfile:
    try:
        return MOVE_REALIZATION_PROFILES[selected_move]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pedagogical move for realization: {selected_move!r}."
        ) from exc


def build_move_realization_instruction(selected_move: str) -> str:
    """Build the private authoritative generation instruction for one move."""

    profile = _profile(selected_move)
    return (
        "AUTHORITATIVE SELECTED-MOVE REALIZATION (PRIVATE)\n"
        f"SELECTED MOVE: {selected_move}\n"
        f"Selected pedagogical move: {selected_move}\n"
        "The selected move is authoritative. Do not reinterpret or change it.\n"
        f"{profile['contract']}\n"
        "The selected move controls the primary pedagogical speech act. "
        "Preserve it throughout generation. Never reveal move labels to the "
        "learner. Do not expose its raw label in teacher_message."
    )


def build_move_audit_instruction(selected_move: str) -> str:
    """Build exact alignment guidance for the post-generation verifier."""

    profile = _profile(selected_move)
    return (
        f"Selected move: {selected_move}\n"
        f"Purpose: {profile['purpose']}\n"
        f"Question usage: {profile['question_usage']}\n"
        f"Hint/detail: {profile['hint_detail_level']}\n"
        f"Explicit explanation: {profile['explicit_explanation_level']}\n"
        f"Length guidance: {profile['target_sentences']}\n"
        "Judge semantic fidelity, not exact wording or punctuation. Treat length "
        "as guidance unless the response is clearly vacuous or needlessly verbose."
    )


__all__ = (
    "MOVE_REALIZATION_CONTRACT",
    "MOVE_REALIZATION_PROFILES",
    "build_move_audit_instruction",
    "build_move_realization_instruction",
)
