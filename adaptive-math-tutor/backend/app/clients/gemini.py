"""Gemini clients used by AdaptMath model-facing agents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from google import genai
from google.genai import types


class GeminiChatCompletions:
    """Expose Gemini structured output through the Tutor's chat interface.

    The Tutor response verifier is a frozen component that accepts a client
    exposing ``chat.completions.create``. This adapter preserves that narrow
    interface while sending every model request to the Gemini Developer API.
    """

    def __init__(self, client: genai.Client, *, max_retries: int = 2) -> None:
        self._client = client
        self._max_retries = max_retries

    @staticmethod
    def _structured_schema(response_format: Mapping[str, Any]) -> dict[str, Any]:
        if response_format.get("type") != "json_schema":
            raise ValueError("Gemini Tutor calls require JSON-schema output.")

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, Mapping):
            raise ValueError("Gemini Tutor response_format is missing json_schema.")

        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("Gemini Tutor response_format has no schema object.")
        return schema

    @staticmethod
    def _gemini_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[str | None, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        if not contents:
            raise ValueError("Gemini Tutor request contains no dialogue content.")

        system_instruction = "\n\n".join(system_parts) or None
        return system_instruction, contents

    @staticmethod
    def _thinking_config(
        *,
        model: str,
        reasoning_effort: str | None,
        include_reasoning: bool,
    ) -> tuple[types.ThinkingConfig | None, int]:
        """Translate the Tutor's reasoning contract for Gemini 2.5.

        Gemini 2.5 counts thinking tokens against ``max_output_tokens``.  The
        Tutor's limits describe the JSON answer budget, so reserve the mapped
        thinking budget in addition to that visible-output allowance.  Without
        this translation, the model's default dynamic thinking can consume the
        entire small atomic-turn allowance and return no JSON at all.
        """

        if not model.strip().lower().startswith("gemini-2.5"):
            return None, 0

        normalized_effort = (reasoning_effort or "").strip().lower()
        budgets = {
            "none": 0,
            "minimal": 1024,
            "low": 1024,
            "medium": 8192,
            "high": 24576,
        }
        if normalized_effort not in budgets:
            return None, 0

        budget = budgets[normalized_effort]
        return (
            types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=include_reasoning,
            ),
            budget,
        )

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        max_completion_tokens: int = 1024,
        response_format: Mapping[str, Any],
        reasoning_effort: str | None = None,
        include_reasoning: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        schema = self._structured_schema(response_format)
        system_instruction, contents = self._gemini_messages(messages)
        thinking_config, thinking_budget = self._thinking_config(
            model=model,
            reasoning_effort=reasoning_effort,
            include_reasoning=include_reasoning,
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_completion_tokens + thinking_budget,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=thinking_config,
        )

        response: Any = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception:
                if attempt >= self._max_retries:
                    raise

        content = getattr(response, "text", None)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


class GeminiChatCompatibilityClient:
    """Gemini client with the narrow chat interface required by Tutor."""

    def __init__(self, *, api_key: str, max_retries: int = 2) -> None:
        client = genai.Client(api_key=api_key)
        self.chat = SimpleNamespace(
            completions=GeminiChatCompletions(
                client,
                max_retries=max_retries,
            )
        )


__all__ = (
    "GeminiChatCompatibilityClient",
    "GeminiChatCompletions",
)
