"""Append-only repair outcome repository."""
import uuid
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database.models.supporting_memory import RepairOutcome

class RepairOutcomeRepository:
    def __init__(self, session: Session): self.session=session
    def add(self, *, student_id, session_id, canonical_skill_id, repair_action, outcome, interaction_id=None, score=None, notes=None):
        row=RepairOutcome(repair_outcome_id=uuid.uuid4(),student_id=student_id,session_id=session_id,canonical_skill_id=canonical_skill_id,interaction_id=interaction_id,repair_action=repair_action,outcome=outcome,score=score,notes=notes)
        self.session.add(row); self.session.flush(); return row
    def for_student_skill(self, student_id, canonical_skill_id):
        return list(self.session.scalars(select(RepairOutcome).where(RepairOutcome.student_id==student_id,RepairOutcome.canonical_skill_id==canonical_skill_id).order_by(RepairOutcome.created_at,RepairOutcome.repair_outcome_id)))
    def for_student(self, student_id):
        return list(self.session.scalars(select(RepairOutcome).where(RepairOutcome.student_id==student_id).order_by(RepairOutcome.created_at,RepairOutcome.repair_outcome_id)))
