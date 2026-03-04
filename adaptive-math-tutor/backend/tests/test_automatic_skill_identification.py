from __future__ import annotations

import copy

import pytest
from fastapi import HTTPException

from app.api import tutor as tutor_api
from app.agents.general_chat.general_chat_agent import GeminiGatewayDecision
from app.clients.memory.memory_client import MemoryTopicClassification


class _Coordinator:
    def __init__(self, supported: set[str] | None = None) -> None:
        self.supported = supported or {"Area Circle"}
        self.validated: list[str] = []

    def validate_target_skill(self, value: object) -> str:
        skill = str(value)
        self.validated.append(skill)
        if skill not in self.supported:
            raise KeyError(f"unsupported skill: {skill}")
        return skill

    @staticmethod
    def derive_attempt_id(thread_id: str, index: int) -> str:
        return f"{thread_id}:{index}"

    @staticmethod
    def has_active_attempt(thread_id: str) -> bool:
        del thread_id
        return False


class _Graph:
    def __init__(self) -> None:
        self.initial_state = None

    def invoke(self, state, config):
        del config
        self.initial_state = copy.deepcopy(state)
        return state


class _MemoryClient:
    def __init__(
        self,
        classification: MemoryTopicClassification | None,
    ) -> None:
        self.classification = classification
        self.classify_calls: list[str] = []
        self.context_calls: list[dict] = []

    def classify_question(self, *, question: str):
        self.classify_calls.append(question)
        return self.classification

    def retrieve_tutor_context(self, **kwargs):
        self.context_calls.append(kwargs)
        return None


class _GeneralChatAgent:
    def __init__(
        self,
        decision: GeminiGatewayDecision | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.decision = decision or GeminiGatewayDecision(
            kind="general_chat",
            response="Hi! What complete math question would you like help with?",
            confidence=0.99,
        )

    def route(self, **kwargs) -> GeminiGatewayDecision:
        self.calls.append(kwargs)
        return self.decision


def _request(
    *,
    target_skill: str | None = None,
    question: str | None = None,
) -> tutor_api.TutorStartRequest:
    return tutor_api.TutorStartRequest(
        student_id="student-auto-skill",
        age=15,
        question=question
        or (
            "A circular park has radius 12 meters and a circular flower bed "
            "has radius 5 meters. Find the remaining area."
        ),
        topic="Geometry",
        subtopic="Area of circles",
        target_skill=target_skill,
    )


def _install(monkeypatch, *, memory_client, coordinator=None, general_chat=None):
    graph = _Graph()
    active_coordinator = coordinator or _Coordinator()
    active_general_chat = general_chat or _GeneralChatAgent()
    monkeypatch.setattr(tutor_api, "get_memory_client", lambda: memory_client)
    monkeypatch.setattr(
        tutor_api,
        "get_general_chat_agent",
        lambda: active_general_chat,
    )
    monkeypatch.setattr(tutor_api, "adaptive_coordinator", active_coordinator)
    monkeypatch.setattr(tutor_api, "tutor_graph", graph)
    monkeypatch.setattr(
        tutor_api,
        "_find_active_student_session",
        lambda student_id: None,
    )
    return graph, active_coordinator


def test_omitted_skill_uses_memory_classifier_then_bkt_validation(monkeypatch):
    memory_client = _MemoryClient(
        MemoryTopicClassification(
            topic="Area Circle",
            skill_id="SKILL_39",
            confidence=0.6366,
            is_math=True,
            model_version="phase16-minilm-ft-v2",
        )
    )
    graph, coordinator = _install(monkeypatch, memory_client=memory_client)

    response = tutor_api.start_tutor_session(_request())

    assert coordinator.validated == ["Area Circle"]
    assert memory_client.classify_calls == [_request().question]
    assert memory_client.context_calls[0]["target_skill"] == "Area Circle"
    assert graph.initial_state["target_skill"] == "Area Circle"
    assert response.target_skill == "Area Circle"


def test_explicit_skill_remains_authoritative_and_skips_classifier(monkeypatch):
    memory_client = _MemoryClient(None)
    graph, coordinator = _install(monkeypatch, memory_client=memory_client)

    response = tutor_api.start_tutor_session(
        _request(target_skill="Area Circle")
    )

    assert coordinator.validated == ["Area Circle"]
    assert memory_client.classify_calls == []
    assert graph.initial_state["target_skill"] == "Area Circle"
    assert response.target_skill == "Area Circle"


def test_percent_discount_maps_to_supported_percent_of_bkt_skill(monkeypatch):
    memory_client = _MemoryClient(
        MemoryTopicClassification(
            topic="Percent Discount",
            skill_id="SKILL_203",
            confidence=0.82,
            is_math=True,
            model_version="phase16-minilm-ft-v2",
        )
    )
    coordinator = _Coordinator(supported={"Percent Of"})
    graph, coordinator = _install(
        monkeypatch,
        memory_client=memory_client,
        coordinator=coordinator,
    )

    response = tutor_api.start_tutor_session(_request())

    assert coordinator.validated == ["Percent Of"]
    assert memory_client.context_calls[0]["target_skill"] == "Percent Of"
    assert graph.initial_state["target_skill"] == "Percent Of"
    assert response.target_skill == "Percent Of"


def test_classifier_abstention_uses_gemini_without_starting_adaptive_lesson(
    monkeypatch,
):
    memory_client = _MemoryClient(
        MemoryTopicClassification(
            topic=None,
            skill_id=None,
            confidence=0.1,
            is_math=False,
            model_version="phase16-minilm-ft-v2",
        )
    )
    general_chat = _GeneralChatAgent()
    graph, coordinator = _install(
        monkeypatch,
        memory_client=memory_client,
        general_chat=general_chat,
    )

    request = _request(question="Hi")
    response = tutor_api.start_tutor_session(request)

    assert response.status == "complete"
    assert response.route == "gemini_general_chat"
    assert response.target_skill is None
    assert response.turn_count == 1
    assert response.thread_id.startswith("general-")
    assert response.tutor_response == (
        "Hi! What complete math question would you like help with?"
    )
    assert general_chat.calls == [
        {"message": "Hi"}
    ]
    assert coordinator.validated == []
    assert graph.initial_state is None
    assert memory_client.context_calls == []


def test_classifier_abstention_gemini_math_decision_starts_adaptive_lesson(
    monkeypatch,
):
    memory_client = _MemoryClient(
        MemoryTopicClassification(
            topic=None,
            skill_id=None,
            confidence=0.2589,
            is_math=False,
            model_version="phase16-minilm-ft-v2",
        )
    )
    general_chat = _GeneralChatAgent(
        GeminiGatewayDecision(
            kind="math_question",
            target_skill="Area Circle",
            confidence=0.96,
        )
    )
    graph, coordinator = _install(
        monkeypatch,
        memory_client=memory_client,
        general_chat=general_chat,
    )

    request = _request()
    response = tutor_api.start_tutor_session(request)

    assert general_chat.calls == [{"message": request.question}]
    assert coordinator.validated == ["Area Circle"]
    assert memory_client.context_calls[0]["target_skill"] == "Area Circle"
    assert graph.initial_state["target_skill"] == "Area Circle"
    assert response.target_skill == "Area Circle"


def test_automatic_identification_rejects_non_bkt_ontology_label(monkeypatch):
    memory_client = _MemoryClient(
        MemoryTopicClassification(
            topic="Linear Equations",
            skill_id="SKILL_193",
            confidence=0.9,
            is_math=True,
            model_version="phase16-minilm-ft-v2",
        )
    )
    graph, coordinator = _install(monkeypatch, memory_client=memory_client)

    with pytest.raises(HTTPException) as raised:
        tutor_api.start_tutor_session(_request())

    assert raised.value.status_code == 422
    assert coordinator.validated == ["Linear Equations"]
    assert graph.initial_state is None
    assert memory_client.context_calls == []


def test_automatic_identification_reports_memory_unavailability(monkeypatch):
    memory_client = _MemoryClient(None)
    graph, coordinator = _install(monkeypatch, memory_client=memory_client)

    with pytest.raises(HTTPException) as raised:
        tutor_api.start_tutor_session(_request())

    assert raised.value.status_code == 503
    assert coordinator.validated == []
    assert graph.initial_state is None
    assert memory_client.context_calls == []
