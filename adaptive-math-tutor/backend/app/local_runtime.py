"""Local-development app wrapper with safe cross-component diagnostics.

The production FastAPI app and all integration algorithms remain unchanged.
This module only observes the existing Memory client and compiled Tutor graph
when the Windows local-stack launcher explicitly selects ``app.local_runtime``.
"""

from __future__ import annotations

import json
import logging
import re
from threading import local
from typing import Any, Mapping

from app.api import tutor as tutor_api
from app.clients.memory.memory_client import MemoryClient
from app.main import app


logger = logging.getLogger("adaptmath.local_runtime")
_request_observation = local()
_SAFE_SCORE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_logged_attempts: set[tuple[str, str]] = set()
_logged_turns: set[tuple[str, str, object]] = set()
_logged_completions: set[tuple[str, str]] = set()


def _identifier(value: object, fallback: str = "unknown") -> str:
    if not isinstance(value, str):
        return fallback
    stripped = value.strip()
    if not stripped or len(stripped) > 255 or "\n" in stripped or "\r" in stripped:
        return fallback
    return stripped


def _safe_numeric_scores(value: object) -> str:
    if not isinstance(value, Mapping):
        return "{}"
    scores: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        if len(scores) >= 20 or not isinstance(raw_name, str):
            break
        if not _SAFE_SCORE_NAME.fullmatch(raw_name):
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        scores[raw_name] = float(raw_value)
    return json.dumps(scores, sort_keys=True, separators=(",", ":"))


def _install_memory_observer() -> None:
    original_retrieve = MemoryClient.retrieve_tutor_context
    original_write = MemoryClient.write_completed_attempt

    def observed_retrieve(
        self: MemoryClient,
        *,
        student_id: str,
        target_skill: str,
        limit: int | None = None,
    ):
        result = original_retrieve(
            self,
            student_id=student_id,
            target_skill=target_skill,
            limit=limit,
        )
        _request_observation.memory = {
            "success": result is not None,
            "target_skill": _identifier(target_skill),
            "previous_errors": len(result.misconceptions) if result else 0,
            "relevant_history": len(result.recent_interactions) if result else 0,
        }
        return result

    def observed_write(
        self: MemoryClient,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = original_write(self, payload)
        status = _identifier(result.get("status"))
        logger.info(
            "local_diagnostic component=memory event=completed_attempt_write "
            "thread_id=%s attempt_id=%s target_skill=%s success=%s "
            "status=%s retry_pending=%s",
            _identifier(payload.get("external_session_id")),
            _identifier(payload.get("external_attempt_id")),
            _identifier(payload.get("target_skill")),
            status in {"STORED", "ALREADY_STORED"},
            status,
            bool(result.get("retryable", False)),
        )
        return result

    MemoryClient.retrieve_tutor_context = observed_retrieve
    MemoryClient.write_completed_attempt = observed_write


def _install_graph_observer() -> None:
    original_invoke = tutor_api.tutor_graph.invoke

    def observed_invoke(*args: Any, **kwargs: Any):
        initial_history_count: int | None = None
        if args and isinstance(args[0], Mapping):
            initial_history = args[0].get("conversation_history")
            if isinstance(initial_history, list):
                initial_history_count = len(initial_history)

        result = original_invoke(*args, **kwargs)
        if not isinstance(result, Mapping):
            return result

        config = kwargs.get("config")
        configured_thread = None
        if isinstance(config, Mapping):
            configurable = config.get("configurable")
            if isinstance(configurable, Mapping):
                configured_thread = configurable.get("thread_id")

        thread_id = _identifier(result.get("thread_id") or configured_thread)
        target_skill = _identifier(result.get("target_skill"))
        attempt_id = _identifier(result.get("attempt_id"))

        memory = getattr(_request_observation, "memory", None)
        if isinstance(memory, Mapping):
            logger.info(
                "local_diagnostic component=memory event=retrieval "
                "thread_id=%s attempt_id=%s target_skill=%s success=%s "
                "previous_error_count=%s relevant_history_count=%s "
                "conversation_history_initial_count=%s",
                thread_id,
                attempt_id,
                target_skill,
                bool(memory.get("success", False)),
                int(memory.get("previous_errors", 0)),
                int(memory.get("relevant_history", 0)),
                initial_history_count if initial_history_count is not None else "unknown",
            )
            del _request_observation.memory

        mastery_before = result.get("mastery_before")
        attempt_key = (thread_id, attempt_id)
        if (
            attempt_key not in _logged_attempts
            and isinstance(mastery_before, (int, float))
            and not isinstance(mastery_before, bool)
        ):
            _logged_attempts.add(attempt_key)
            logger.info(
                "local_diagnostic component=bkt event=attempt_active "
                "thread_id=%s attempt_id=%s target_skill=%s mastery_before=%.12g",
                thread_id,
                attempt_id,
                target_skill,
                float(mastery_before),
            )

        diagnostics = result.get("adaptive_turn_diagnostics")
        diagnostic_attempt = (
            _identifier(diagnostics.get("attempt_id"), attempt_id)
            if isinstance(diagnostics, Mapping)
            else attempt_id
        )
        turn_key = (
            thread_id,
            diagnostic_attempt,
            diagnostics.get("turn_index") if isinstance(diagnostics, Mapping) else None,
        )
        if isinstance(diagnostics, Mapping) and turn_key not in _logged_turns:
            _logged_turns.add(turn_key)
            logger.info(
                "local_diagnostic component=adaptive_turn event=completed "
                "thread_id=%s attempt_id=%s target_skill=%s turn_index=%s "
                "final_move=%s tutor_generation_success=%s "
                "mrb1_scoring_success=%s mrb1_scores=%s",
                thread_id,
                diagnostic_attempt,
                target_skill,
                diagnostics.get("turn_index", "unknown"),
                _identifier(diagnostics.get("pedagogical_move")),
                bool(result.get("tutor_response")),
                isinstance(diagnostics.get("mrb1_scores"), Mapping),
                _safe_numeric_scores(diagnostics.get("mrb1_scores")),
            )

        completion = result.get("adaptive_completion_result")
        completed_attempt = _identifier(
            result.get("completed_attempt_id"),
            attempt_id,
        )
        completion_key = (thread_id, completed_attempt)
        if isinstance(completion, Mapping) and completion_key not in _logged_completions:
            _logged_completions.add(completion_key)
            before = completion.get("mastery_before")
            after = completion.get("mastery_after")
            delta = completion.get("delta_mastery")
            if not isinstance(delta, (int, float)):
                delta = completion.get("mastery_delta")
            logger.info(
                "local_diagnostic component=bkt event=attempt_completed "
                "thread_id=%s completed_attempt_id=%s target_skill=%s "
                "mastery_before=%s mastery_after=%s delta_mastery=%s",
                thread_id,
                completed_attempt,
                target_skill,
                before if isinstance(before, (int, float)) else "unknown",
                after if isinstance(after, (int, float)) else "unknown",
                delta if isinstance(delta, (int, float)) else "unknown",
            )

        return result

    tutor_api.tutor_graph.invoke = observed_invoke


_install_memory_observer()
_install_graph_observer()
