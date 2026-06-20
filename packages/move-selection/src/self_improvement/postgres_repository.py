"""PostgreSQL repository for structured adaptive-policy runtime evidence.

The repository only persists values already produced by the frozen scientific
components. It does not select arms, score responses, build contexts, compute
rewards, or update a posterior.
"""

from __future__ import annotations

from collections.abc import Mapping
import atexit
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


C3_FEATURES = (
    "md6_p_generic",
    "md6_p_probing",
    "md6_p_focus",
    "md6_p_telling",
    "running_mistake_identification",
    "running_mistake_location",
    "running_providing_guidance",
    "running_actionability",
    "has_within_attempt_quality",
)
_POSTGRES_POOLS: dict[str, Any] = {}
_POSTGRES_POOL_LOCK = threading.Lock()


def research_persistence_mode() -> str:
    mode = os.getenv("RESEARCH_PERSISTENCE", "file").strip().lower()
    if mode not in {"file", "postgres"}:
        raise ValueError("RESEARCH_PERSISTENCE must be 'file' or 'postgres'.")
    return mode


def postgres_enabled() -> bool:
    return research_persistence_mode() == "postgres"


def policy_key_for_path(path: str | Path) -> str:
    explicit = os.getenv("RESEARCH_POLICY_KEY", "").strip()
    if explicit:
        return explicit
    source = Path(path)
    return f"{source.parent.name}:{source.name}"


def _database_url() -> str:
    value = os.getenv("RESEARCH_DATABASE_URL", os.getenv("ADAPTMATH_DATABASE_URL", "")).strip()
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value.removeprefix("postgresql+psycopg://")
    if not value.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            "RESEARCH_DATABASE_URL (or ADAPTMATH_DATABASE_URL) is required "
            "when RESEARCH_PERSISTENCE=postgres."
        )
    return value


def _database_schema() -> str:
    schema = os.getenv("RESEARCH_DATABASE_SCHEMA", "research").strip()
    if schema != "research":
        raise ValueError("RESEARCH_DATABASE_SCHEMA must be exactly 'research'.")
    return schema


def _connect():
    url = _database_url()
    with _POSTGRES_POOL_LOCK:
        pool = _POSTGRES_POOLS.get(url)
        if pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg-pool is required for research PostgreSQL persistence."
                ) from exc

            def configure(connection: Any) -> None:
                connection.execute(f"SET search_path TO {_database_schema()}, public")
                connection.commit()

            pool = ConnectionPool(
                conninfo=url,
                open=True,
                min_size=1,
                max_size=int(os.getenv("RESEARCH_DB_POOL_SIZE", "8")),
                kwargs={"autocommit": False},
                configure=configure,
                name="adaptmath-research",
            )
            _POSTGRES_POOLS[url] = pool
        return pool.connection()


def _close_postgres_pools() -> None:
    for pool in tuple(_POSTGRES_POOLS.values()):
        pool.close()


atexit.register(_close_postgres_pools)


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonb(value: Mapping[str, object]):
    from psycopg.types.json import Jsonb

    return Jsonb(dict(value))


def _ordered_json(value: Mapping[str, object]):
    from psycopg.types.json import Json

    return Json(dict(value), dumps=lambda item: json.dumps(item, allow_nan=False))


def save_policy_snapshot(
    *,
    policy_key: str,
    payload: Mapping[str, object],
    policy_class: str,
) -> str:
    serialized = _canonical_json(payload)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    inner = payload.get("policy_state")
    state = inner if isinstance(inner, Mapping) else payload
    schema_version = str(payload.get("schema_version", state.get("schema_version", "unknown")))
    context_dimension = state.get("context_dim", state.get("context_dimension"))
    if context_dimension is None and isinstance(state.get("ordered_feature_names"), list):
        context_dimension = len(state["ordered_feature_names"])
    total_updates = state.get("total_updates", state.get("update_count", 0))
    data_mode = state.get("data_mode")
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO research.policy_states
                (policy_key, schema_version, policy_class, data_mode,
                 context_dimension, total_updates, state_sha256, state_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (policy_key) DO UPDATE SET
                schema_version = EXCLUDED.schema_version,
                policy_class = EXCLUDED.policy_class,
                data_mode = EXCLUDED.data_mode,
                context_dimension = EXCLUDED.context_dimension,
                total_updates = EXCLUDED.total_updates,
                state_sha256 = EXCLUDED.state_sha256,
                state_json = EXCLUDED.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                policy_key,
                schema_version,
                policy_class,
                data_mode,
                None if context_dimension is None else int(context_dimension),
                int(total_updates or 0),
                digest,
                _ordered_json(payload),
            ),
        )
        connection.commit()
    return digest


def load_policy_snapshot(policy_key: str) -> dict[str, object]:
    with _connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM research.policy_states WHERE policy_key = %s",
            (policy_key,),
        ).fetchone()
        connection.commit()
    if row is None:
        raise FileNotFoundError(f"PostgreSQL policy state does not exist: {policy_key}.")
    value = row[0]
    if not isinstance(value, dict):
        raise ValueError("PostgreSQL policy state is not a JSON object.")
    return value


def _attempt_thread_id(attempt_id: str) -> str | None:
    head, separator, tail = attempt_id.rpartition(":")
    if separator and tail.isdigit() and head:
        return head
    return None


def _probabilities(record: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("md6_probabilities", "md7_raw_probabilities", "selector_probabilities"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _stage_normalized(connection: Any, record: Mapping[str, object]) -> None:
    action_id = str(record["action_event_id"])
    attempt_id = str(record["attempt_id"])
    turn_index = int(record.get("action_turn_index", record.get("turn_index", 0)))
    probabilities = _probabilities(record)
    canonical = {
        move: probabilities.get(move) for move in ("generic", "probing", "focus", "telling")
    }
    available = {key: float(value) for key, value in canonical.items() if value is not None}
    argmax = max(available, key=available.get) if available else None
    connection.execute(
        """
        INSERT INTO research.selector_decisions
            (action_event_id, attempt_id, turn_index, selector_version,
             p_generic, p_probing, p_focus, p_telling, selector_argmax,
             selected_arm, actual_move, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                CURRENT_TIMESTAMP)
        ON CONFLICT (action_event_id) DO NOTHING
        """,
        (
            action_id,
            attempt_id,
            turn_index,
            record.get("selector_version"),
            canonical["generic"],
            canonical["probing"],
            canonical["focus"],
            canonical["telling"],
            argmax,
            record.get("selected_arm"),
            record.get("actual_move"),
        ),
    )
    context = record.get("policy_context", record.get("context"))
    if not isinstance(context, Mapping):
        return
    vector = context.get("vector")
    names = context.get("feature_names")
    if not isinstance(vector, list) or not isinstance(names, list):
        return
    values = [float(value) for value in vector]
    feature_names = [str(name) for name in names]
    dimension = int(context.get("context_dimension", len(values)))
    connection.execute(
        """
        INSERT INTO research.policy_contexts
            (action_event_id, context_schema_version, context_schema_id,
             context_dimension, feature_names, context_vector, context_payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (action_event_id) DO NOTHING
        """,
        (
            action_id,
            str(context.get("schema_version", "unknown")),
            str(context.get("schema_id", "unknown")),
            dimension,
            feature_names,
            values,
            _jsonb(context),
        ),
    )
    if dimension == 9 and tuple(feature_names) == C3_FEATURES:
        connection.execute(
            """
            INSERT INTO research.c3_contexts
                (action_event_id, context_schema_version, context_dimension,
                 md6_p_generic, md6_p_probing, md6_p_focus, md6_p_telling,
                 mrb1_mistake_identification, mrb1_mistake_location,
                 mrb1_providing_guidance, mrb1_actionability,
                 has_within_attempt_quality, context_vector)
            VALUES (%s, %s, 9, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (action_event_id) DO NOTHING
            """,
            (action_id, str(context.get("schema_version", "C3")), *values, values),
        )


def stage_action(
    record: Mapping[str, object], *, action_id_override: str | None = None
) -> None:
    identity = record.get("identity")
    identity_map = identity if isinstance(identity, Mapping) else {}
    action_id = str(
        action_id_override
        or record.get("action_event_id", identity_map.get("action_event_id", ""))
    ).strip()
    attempt_id = str(
        record.get("attempt_id", identity_map.get("attempt_id", ""))
    ).strip()
    if not action_id or not attempt_id:
        raise ValueError("Research action needs stable action_event_id and attempt_id.")
    payload = dict(record)
    with _connect() as connection:
        existing = connection.execute(
            "SELECT payload FROM research.turn_actions WHERE action_event_id = %s FOR UPDATE",
            (action_id,),
        ).fetchone()
        if existing is not None:
            existing_payload = dict(existing[0])
            if any(existing_payload.get(key) != value for key, value in payload.items()):
                raise RuntimeError("Action identity was replayed with different context.")
            connection.commit()
            return
        connection.execute(
            """
            INSERT INTO research.turn_actions
                (action_event_id, attempt_id, thread_id, student_id,
                 target_skill, turn_index, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                action_id,
                attempt_id,
                record.get("thread_id")
                or identity_map.get("thread_id")
                or _attempt_thread_id(attempt_id),
                record.get("student_id")
                or identity_map.get("student_pseudonymous_id"),
                record.get("target_skill") or identity_map.get("target_skill"),
                int(
                    record.get(
                        "action_turn_index",
                        record.get(
                            "turn_index", identity_map.get("action_turn_index", 0)
                        ),
                    )
                ),
                _jsonb(payload),
            ),
        )
        if (
            action_id_override is None
            and "action_event_id" in record
            and "attempt_id" in record
        ):
            _stage_normalized(connection, payload)
        connection.commit()


def get_action(action_id: str) -> tuple[dict[str, object], str] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload, status FROM research.turn_actions "
            "WHERE action_event_id = %s",
            (action_id,),
        ).fetchone()
        connection.commit()
    if row is None:
        return None
    return dict(row[0]), str(row[1])


def list_pending_actions(schema_version: str) -> list[dict[str, object]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT payload FROM research.turn_actions "
            "WHERE status = 'PENDING' AND payload->>'schema_version' = %s "
            "ORDER BY created_at, action_event_id",
            (schema_version,),
        ).fetchall()
        connection.commit()
    return [dict(row[0]) for row in rows]


def actions_for_attempt(
    attempt_id: str, *, include_pending: bool, schema_version: str | None = None
) -> list[dict[str, object]]:
    status_clause = "" if include_pending else "AND status = 'COMPLETED'"
    schema_clause = (
        "" if schema_version is None else "AND payload->>'schema_version' = %s"
    )
    parameters: tuple[object, ...] = (
        (attempt_id,) if schema_version is None else (attempt_id, schema_version)
    )
    with _connect() as connection:
        rows = connection.execute(
            "SELECT payload FROM research.turn_actions "
            "WHERE attempt_id = %s "
            + status_clause
            + " "
            + schema_clause
            + " ORDER BY turn_index",
            parameters,
        ).fetchall()
        connection.commit()
    return [dict(row[0]) for row in rows]


def patch_pending_action(
    action_id: str, patch: Mapping[str, object]
) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload, status FROM research.turn_actions "
            "WHERE action_event_id = %s FOR UPDATE",
            (action_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("No staged research action exists for patching.")
        payload = dict(row[0])
        for key, value in patch.items():
            if key in payload and payload[key] != value:
                raise RuntimeError("Pending action patch differs on retry.")
            payload[key] = value
        if row[1] != "PENDING":
            if dict(row[0]) != payload:
                raise RuntimeError("Completed action cannot be patched.")
            connection.commit()
            return False
        connection.execute(
            "UPDATE research.turn_actions SET payload = %s "
            "WHERE action_event_id = %s",
            (_jsonb(payload), action_id),
        )
        connection.commit()
        return True


def add_response_quality(
    action_id: str,
    scores: Mapping[str, float],
    tutor_response: str | None,
) -> None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT attempt_id, turn_index, payload FROM research.turn_actions "
            "WHERE action_event_id = %s FOR UPDATE",
            (action_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("No staged research action exists for MRB1.")
        payload = dict(row[2])
        supplied = {key: float(value) for key, value in scores.items()}
        existing = payload.get("current_response_mrb1")
        if existing is not None and existing != supplied:
            raise RuntimeError("MRB1 retry differs for one action identity.")
        payload["current_response_mrb1"] = supplied
        if tutor_response is not None:
            response = str(tutor_response).strip()
            if not response:
                raise ValueError("Tutor response provenance cannot be empty.")
            old_response = payload.get("tutor_response")
            if old_response is not None and old_response != response:
                raise RuntimeError("Tutor response retry differs for one action identity.")
            payload["tutor_response"] = response
        connection.execute(
            "UPDATE research.turn_actions SET payload = %s WHERE action_event_id = %s",
            (_jsonb(payload), action_id),
        )
        connection.execute(
            """
            INSERT INTO research.mrb1_scores
                (action_event_id, attempt_id, turn_index,
                 mistake_identification, mistake_location,
                 providing_guidance, actionability)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (action_event_id) DO UPDATE SET
                mistake_identification = EXCLUDED.mistake_identification,
                mistake_location = EXCLUDED.mistake_location,
                providing_guidance = EXCLUDED.providing_guidance,
                actionability = EXCLUDED.actionability
            """,
            (
                action_id,
                row[0],
                row[1],
                supplied["Mistake_Identification"],
                supplied["Mistake_Location"],
                supplied["Providing_Guidance"],
                supplied["Actionability"],
            ),
        )
        connection.commit()


def complete_action(
    *,
    action_id: str,
    outcome: Mapping[str, object],
    policy_key: str | None = None,
    policy_payload: Mapping[str, object] | None = None,
    policy_class: str = "DirectTurnLinTS",
    experience_event_id: str | None = None,
) -> None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT attempt_id, payload, status FROM research.turn_actions "
            "WHERE action_event_id = %s FOR UPDATE",
            (action_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("No staged research action exists for completion.")
        payload = {**dict(row[1]), **dict(outcome)}
        if row[2] == "COMPLETED":
            if dict(row[1]) != payload:
                raise RuntimeError("Completed action retry differs from ledger.")
            connection.commit()
            return
        connection.execute(
            """
            UPDATE research.turn_actions
            SET payload = %s, status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
            WHERE action_event_id = %s
            """,
            (_jsonb(payload), action_id),
        )
        update_id = payload.get("update_id")
        if update_id and payload.get("posterior_update_occurred"):
            connection.execute(
                """
                INSERT INTO research.policy_updates
                    (update_id, policy_key, action_event_id, attempt_id,
                     selected_arm, reward, sample_weight, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (update_id) DO NOTHING
                """,
                (
                    str(update_id),
                    str(policy_key or "unknown"),
                    action_id,
                    row[0],
                    payload.get("selected_arm"),
                    payload.get("configured_reward_value"),
                    payload.get("posterior_update_weight"),
                    _jsonb(payload),
                ),
            )
        if policy_key is not None and policy_payload is not None:
            serialized = _canonical_json(policy_payload)
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            context_dimension = policy_payload.get(
                "context_dim", policy_payload.get("context_dimension")
            )
            if context_dimension is None and isinstance(
                policy_payload.get("ordered_feature_names"), list
            ):
                context_dimension = len(policy_payload["ordered_feature_names"])
            total_updates = policy_payload.get(
                "total_updates", policy_payload.get("update_count", 0)
            )
            connection.execute(
                """
                INSERT INTO research.policy_states
                    (policy_key, schema_version, policy_class, data_mode,
                     context_dimension, total_updates, state_sha256, state_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (policy_key) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    policy_class = EXCLUDED.policy_class,
                    data_mode = EXCLUDED.data_mode,
                    context_dimension = EXCLUDED.context_dimension,
                    total_updates = EXCLUDED.total_updates,
                    state_sha256 = EXCLUDED.state_sha256,
                    state_json = EXCLUDED.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    policy_key,
                    str(policy_payload.get("schema_version", "unknown")),
                    policy_class,
                    policy_payload.get("data_mode"),
                    None if context_dimension is None else int(context_dimension),
                    int(total_updates or 0),
                    digest,
                    _ordered_json(policy_payload),
                ),
            )
        if experience_event_id is not None:
            _insert_experience(connection, payload, experience_event_id)
        connection.commit()


def _insert_experience(connection: Any, payload: dict[str, object], event_id: str) -> None:
    identity = payload.get("identity")
    identity_map = identity if isinstance(identity, Mapping) else {}
    attempt_id = payload.get(
        "attempt_id", payload.get("episode_id", identity_map.get("attempt_id"))
    )
    learning = payload.get("learning_outcome")
    learning_map = learning if isinstance(learning, Mapping) else {}
    learning_update = payload.get("learning_state_update")
    update_map = learning_update if isinstance(learning_update, Mapping) else {}
    before = payload.get(
        "mastery_before", learning_map.get("mastery_before", update_map.get("mastery_before"))
    )
    after = payload.get(
        "mastery_after", learning_map.get("mastery_after", update_map.get("mastery_after"))
    )
    delta = payload.get(
        "mastery_delta", learning_map.get("delta_mastery", update_map.get("delta_mastery"))
    )
    existing = connection.execute(
        "SELECT payload FROM research.experience_records WHERE event_id = %s",
        (event_id,),
    ).fetchone()
    if existing is not None and existing[0] != payload:
        raise RuntimeError("Experience identity was replayed with different evidence.")
    connection.execute(
        """
        INSERT INTO research.experience_records
            (event_id, schema_version, attempt_id, thread_id, student_id,
             target_skill, mastery_before, mastery_after, delta_mastery,
             reward, payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING
        """,
        (
            event_id,
            str(payload.get("schema_version", "unknown")),
            None if attempt_id is None else str(attempt_id),
            payload.get("thread_id", identity_map.get("thread_id")),
            payload.get(
                "student_id", identity_map.get("student_pseudonymous_id")
            ),
            payload.get(
                "target_skill",
                identity_map.get("target_skill", learning_map.get("skill")),
            ),
            before,
            after,
            delta,
            payload.get("reward", delta),
            _jsonb(payload),
        ),
    )


def append_experience(
    record: Mapping[str, object], *, event_id_override: str | None = None
) -> None:
    payload = dict(record)
    identity = payload.get("identity")
    identity_map = identity if isinstance(identity, Mapping) else {}
    attempt_id = payload.get(
        "attempt_id", payload.get("episode_id", identity_map.get("attempt_id"))
    )
    identity_source = _canonical_json(payload)
    event_id = str(event_id_override or "").strip()
    if not event_id:
        event_id = str(attempt_id).strip() if attempt_id is not None else ""
    if not event_id:
        event_id = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    with _connect() as connection:
        _insert_experience(connection, payload, event_id)
        connection.commit()
