"""Reusable validation helpers for public and integration-facing schemas.

These checks are infrastructure safeguards rather than pedagogical rules. They
normalize whitespace, reject control characters that are unsafe for text
processing, and bound request sizes so malformed or unexpectedly large payloads
do not reach the ML/GenAI workflow.
"""

from __future__ import annotations

from typing import Iterable


STUDENT_ID_MAX_LENGTH = 128
QUESTION_MAX_LENGTH = 10_000
TOPIC_MAX_LENGTH = 200
SUBTOPIC_MAX_LENGTH = 200
QUESTION_ID_MAX_LENGTH = 128
ANSWER_MAX_LENGTH = 5_000
CONTEXT_TEXT_MAX_LENGTH = 1_000
EVENT_TYPE_MAX_LENGTH = 200
MAX_CONTEXT_ITEMS = 100
MAX_HISTORY_ITEMS = 100
MAX_ANSWERS_PER_SUBMISSION = 20


def clean_required_text(value: str, *, label: str) -> str:
    """Trim a required text field and reject blank/NUL-containing input."""
    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{label} cannot be blank.")

    if "\x00" in cleaned:
        raise ValueError(f"{label} contains an invalid control character.")

    return cleaned


def clean_optional_text(value: str | None, *, label: str) -> str | None:
    """Trim optional text; normalize whitespace-only values to ``None``."""
    if value is None:
        return None

    cleaned = value.strip()

    if not cleaned:
        return None

    if "\x00" in cleaned:
        raise ValueError(f"{label} contains an invalid control character.")

    return cleaned


def clean_text_list(
    values: Iterable[str],
    *,
    label: str,
    max_item_length: int = CONTEXT_TEXT_MAX_LENGTH,
) -> list[str]:
    """Normalize a list of required text items and enforce an item-size bound."""
    cleaned_values: list[str] = []

    for value in values:
        cleaned = clean_required_text(
            value,
            label=f"{label} item",
        )

        if len(cleaned) > max_item_length:
            raise ValueError(
                f"{label} item must be at most {max_item_length} characters."
            )

        cleaned_values.append(cleaned)

    return cleaned_values
