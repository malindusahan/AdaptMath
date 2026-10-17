"""PostgreSQL read/upsert repository for derived memory projections."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models.memory_projection import (
    ConceptMemory,
    LongTermMemory,
    ShortTermMemory,
)
from src.schemas.memory_projection import (
    ConceptMemoryRecord,
    ConceptMemoryUpsert,
    LongTermMemoryRecord,
    LongTermMemoryUpsert,
    ShortTermMemoryRecord,
    ShortTermMemoryUpsert,
)


class MemoryProjectionRepository:
    """Persist current derived summaries without calculating them."""

    def __init__(self, session: Session):
        self.session = session

    def get_short_term_memory(
        self,
        student_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> ShortTermMemoryRecord | None:
        model = self.session.get(ShortTermMemory, (student_id, session_id))
        return ShortTermMemoryRecord.model_validate(model) if model else None

    def upsert_short_term_memory(
        self,
        projection: ShortTermMemoryUpsert,
    ) -> ShortTermMemoryRecord:
        return self._upsert(
            ShortTermMemory,
            projection,
            ["student_id", "session_id"],
            ShortTermMemoryRecord,
        )

    def get_long_term_memory(
        self,
        student_id: uuid.UUID,
    ) -> LongTermMemoryRecord | None:
        model = self.session.get(LongTermMemory, student_id)
        return LongTermMemoryRecord.model_validate(model) if model else None

    def upsert_long_term_memory(
        self,
        projection: LongTermMemoryUpsert,
    ) -> LongTermMemoryRecord:
        return self._upsert(
            LongTermMemory,
            projection,
            ["student_id"],
            LongTermMemoryRecord,
        )

    def get_concept_memory(
        self,
        student_id: uuid.UUID,
        canonical_skill_id: uuid.UUID,
    ) -> ConceptMemoryRecord | None:
        model = self.session.get(
            ConceptMemory,
            (student_id, canonical_skill_id),
        )
        return ConceptMemoryRecord.model_validate(model) if model else None

    def upsert_concept_memory(
        self,
        projection: ConceptMemoryUpsert,
    ) -> ConceptMemoryRecord:
        return self._upsert(
            ConceptMemory,
            projection,
            ["student_id", "canonical_skill_id"],
            ConceptMemoryRecord,
        )

    def _upsert(self, model, projection, key_columns, record_type):
        values = projection.model_dump()
        update_values = {
            key: value for key, value in values.items() if key not in key_columns
        }
        update_values["updated_at"] = func.now()

        statement = (
            insert(model)
            .values(**values)
            .on_conflict_do_update(
                index_elements=key_columns,
                set_=update_values,
            )
            .returning(model)
            .execution_options(populate_existing=True)
        )
        stored = self.session.scalars(statement).one()
        self.session.flush()
        return record_type.model_validate(stored)
