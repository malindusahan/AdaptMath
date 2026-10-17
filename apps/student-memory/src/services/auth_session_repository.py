"""Schema-owned authentication session persistence.

Bearer tokens remain opaque ``mem_sess_*`` values at the API boundary.  Only
their SHA-256 digest is stored in PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
from typing import Any
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session


def hash_session_token(token: str) -> str:
    """Return the stable one-way lookup digest for an opaque bearer token."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthSessionRepository:
    """Transactional access to ``auth.sessions`` and Phase-A account data."""

    def __init__(self) -> None:
        schema = os.getenv("AUTH_DATABASE_SCHEMA", "auth").strip()
        if schema != "auth":
            raise ValueError("AUTH_DATABASE_SCHEMA must be exactly 'auth'.")

    def create(
        self,
        session: Session,
        *,
        token: str,
        user_id: uuid.UUID,
        expires_at: datetime,
    ) -> uuid.UUID:
        session_id = uuid.uuid4()
        session.execute(
            text(
                """
                INSERT INTO auth.sessions
                    (session_id, token_hash, user_id, expires_at)
                VALUES
                    (:session_id, :token_hash, :user_id, :expires_at)
                """
            ),
            {
                "session_id": session_id,
                "token_hash": hash_session_token(token),
                "user_id": user_id,
                "expires_at": expires_at,
            },
        )
        return session_id

    def resolve(self, session: Session, token: str) -> dict[str, Any] | None:
        row = session.execute(
            text(
                """
                SELECT
                    account.user_id,
                    account.username,
                    account.role,
                    account.age,
                    student.external_student_id AS student_id
                FROM auth.sessions auth_session
                JOIN student_memory.user_accounts account
                  ON account.user_id = auth_session.user_id
                LEFT JOIN student_memory.students student
                  ON student.student_id = account.student_id
                WHERE auth_session.token_hash = :token_hash
                  AND auth_session.revoked_at IS NULL
                  AND auth_session.expires_at > CURRENT_TIMESTAMP
                """
            ),
            {"token_hash": hash_session_token(token)},
        ).mappings().one_or_none()
        if row is None:
            return None
        session.execute(
            text(
                """
                UPDATE auth.sessions
                SET last_seen_at = CURRENT_TIMESTAMP
                WHERE token_hash = :token_hash
                """
            ),
            {"token_hash": hash_session_token(token)},
        )
        return {
            "user_id": str(row["user_id"]),
            "username": str(row["username"]),
            "role": str(row["role"]),
            "student_id": (
                None if row["student_id"] is None else str(row["student_id"])
            ),
            "age": row["age"],
        }

    def revoke(self, session: Session, token: str) -> None:
        session.execute(
            text(
                """
                UPDATE auth.sessions
                SET revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP)
                WHERE token_hash = :token_hash
                """
            ),
            {"token_hash": hash_session_token(token)},
        )
