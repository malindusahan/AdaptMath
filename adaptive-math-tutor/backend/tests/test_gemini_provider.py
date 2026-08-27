from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.clients.gemini import GeminiChatCompletions


class FakeGeminiModels:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise RuntimeError("temporary provider failure")
        return SimpleNamespace(text=json.dumps({"status": "ok"}))


def _response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        },
    }


def test_gemini_compatibility_client_preserves_structured_chat_contract():
    models = FakeGeminiModels()
    completions = GeminiChatCompletions(
        SimpleNamespace(models=models),
        max_retries=0,
    )

    response = completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "Return verified JSON."},
            {"role": "user", "content": "Evaluate this input."},
            {"role": "assistant", "content": "Prior model action."},
            {"role": "user", "content": "Continue."},
        ],
        temperature=0,
        max_completion_tokens=250,
        response_format=_response_format(),
        reasoning_effort="low",
        include_reasoning=False,
    )

    assert json.loads(response.choices[0].message.content) == {"status": "ok"}
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert [content.role for content in call["contents"]] == [
        "user",
        "model",
        "user",
    ]
    assert call["config"].system_instruction == "Return verified JSON."
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].thinking_config.thinking_budget == 1024
    assert call["config"].thinking_config.include_thoughts is False
    assert call["config"].max_output_tokens == 1274


def test_gemini_compatibility_reserves_json_tokens_after_low_reasoning():
    class BudgetSensitiveGeminiModels(FakeGeminiModels):
        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            config = kwargs["config"]
            thinking = config.thinking_config
            thinking_budget = 24576 if thinking is None else thinking.thinking_budget
            visible_budget = config.max_output_tokens - thinking_budget
            content = json.dumps({"status": "ok"}) if visible_budget >= 250 else None
            return SimpleNamespace(text=content)

    models = BudgetSensitiveGeminiModels()
    completions = GeminiChatCompletions(
        SimpleNamespace(models=models),
        max_retries=0,
    )

    response = completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Return the teacher-turn JSON."}],
        max_completion_tokens=250,
        response_format=_response_format(),
        reasoning_effort="low",
        include_reasoning=False,
    )

    assert json.loads(response.choices[0].message.content) == {"status": "ok"}
    assert models.calls[0]["config"].max_output_tokens == 1274


def test_gemini_compatibility_client_retries_without_changing_contract():
    models = FakeGeminiModels(failures=1)
    completions = GeminiChatCompletions(
        SimpleNamespace(models=models),
        max_retries=1,
    )

    response = completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Continue."}],
        response_format=_response_format(),
    )

    assert len(models.calls) == 2
    assert response.choices[0].message.content


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "text"},
        {"type": "json_schema", "json_schema": {}},
    ],
)
def test_gemini_compatibility_client_rejects_unstructured_contracts(
    response_format,
):
    completions = GeminiChatCompletions(
        SimpleNamespace(models=FakeGeminiModels()),
        max_retries=0,
    )

    with pytest.raises(ValueError):
        completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Continue."}],
            response_format=response_format,
        )


def test_active_runtime_agents_do_not_import_groq():
    backend_root = Path(__file__).resolve().parents[1]
    active_files = [
        backend_root / "app" / "agents" / "router" / "router_agent.py",
        backend_root / "app" / "agents" / "planner" / "planner_agent.py",
        backend_root / "app" / "agents" / "strategy" / "strategy_agent.py",
        backend_root / "app" / "agents" / "progress" / "teaching_progress_agent.py",
        backend_root / "app" / "agents" / "evaluator" / "evaluator_agent.py",
        backend_root / "app" / "agents" / "tutor" / "tutor_agent.py",
    ]

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in active_files
    )
    assert "langchain_groq" not in source
    assert "from groq import" not in source
    assert "groq_api_key" not in source
