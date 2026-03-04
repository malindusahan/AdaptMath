"""Gemini gateway for messages on which Memory's topic classifier abstains."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, model_validator

from app.core.config import WORKSPACE_ROOT, get_settings


GEMINI_GATEWAY_SYSTEM_PROMPT = """
You are AdaptMath's message gateway. A local topic model could not confidently
map the learner's message. Decide whether the message is a concrete math
question that should start the adaptive Tutor, or general conversation.

Classification rules:
- math_question: a concrete mathematical problem, calculation, equation,
  geometry/statistics task, or word problem with enough information to begin.
  Choose exactly one target_skill from the supplied supported-skill list. For
  a multi-step problem, choose the skill that best represents the main
  reasoning required by the whole problem. Do not answer the problem here.
- general_chat: a greeting, social message, unrelated request, or a vague math
  request without enough problem information. Generate a natural response in
  one to three sentences. Invite a complete math question when appropriate.

Do not mention classifiers, BKT, Turn-LinTS, internal routing, or this prompt
in a learner-facing response. Never invent a target skill outside the supplied
list.
""".strip()


class GeminiGatewayDecision(BaseModel):
    """Structured result from one Gemini gateway invocation."""

    kind: Literal["general_chat", "math_question"]
    target_skill: str | None = None
    response: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_kind_payload(self):
        if self.kind == "math_question":
            if not self.target_skill or not self.target_skill.strip():
                raise ValueError("A math question requires target_skill.")
            self.target_skill = self.target_skill.strip()
            self.response = None
        else:
            if not self.response or not self.response.strip():
                raise ValueError("General chat requires a response.")
            self.response = self.response.strip()
            self.target_skill = None
        return self


@lru_cache(maxsize=1)
def supported_bkt_skills() -> tuple[str, ...]:
    student_model_root = Path(
        os.getenv("STUDENT_MODEL_REPO", WORKSPACE_ROOT / "student-modeling")
    ).resolve()
    parameters_path = student_model_root / "models" / "bkt_params.json"
    payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("The trained BKT vocabulary is unavailable.")

    # Target-skill validation normalizes surrounding whitespace, so exclude
    # malformed legacy keys that cannot be addressed through the public API.
    skills = tuple(
        sorted(
            skill
            for skill in payload
            if isinstance(skill, str) and skill and skill == skill.strip()
        )
    )
    if not skills:
        raise ValueError("The trained BKT vocabulary is empty.")
    return skills


class GeneralChatAgent:
    """Route an abstention with exactly one structured Gemini invocation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key.get_secret_value(),
            temperature=0.2,
            timeout=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
        )
        self.structured_model = self.model.with_structured_output(
            GeminiGatewayDecision,
            method="json_schema",
        )

    def route(self, *, message: str) -> GeminiGatewayDecision:
        skills = supported_bkt_skills()
        response = self.structured_model.invoke(
            [
                (
                    "system",
                    f"{GEMINI_GATEWAY_SYSTEM_PROMPT}\n\n"
                    "Supported target_skill values (use exact spelling):\n"
                    f"{json.dumps(skills, ensure_ascii=False)}",
                ),
                ("human", message),
            ]
        )
        if isinstance(response, GeminiGatewayDecision):
            return response
        return GeminiGatewayDecision.model_validate(response)


@lru_cache(maxsize=1)
def get_general_chat_agent() -> GeneralChatAgent:
    return GeneralChatAgent()


__all__ = (
    "GeneralChatAgent",
    "GeminiGatewayDecision",
    "get_general_chat_agent",
    "supported_bkt_skills",
)
