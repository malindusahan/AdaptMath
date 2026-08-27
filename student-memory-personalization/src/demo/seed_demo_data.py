"""Deterministic and idempotent demo seed script for Student Personalization Memory.

Seeds standard student demo accounts:
1. demo_new (STUDENT): clean cold-start, no prior cognitive history.
2. demo_developing (STUDENT): Algebra / Linear Equations with DEVELOPING state, misconceptions, and interactions.
3. demo_strong (STUDENT): Geometry / Pythagorean Theorem with STRONG state and high accuracy.
4. demo_support (STUDENT): 3+ successful HINT repair outcomes yielding SUPPORTED_BY_HISTORY.

Default Demo Password: DemoPassword123!
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys
import uuid

# Ensure workspace root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os

def _load_env_if_present() -> None:
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_env_if_present()

from sqlalchemy import select

from src.database.models.core import CanonicalSkill, LearningSession, Student
from src.database.models.memory_projection import ConceptMemory, LongTermMemory, ShortTermMemory
from src.database.models.raw_interaction import InteractionLog
from src.database.models.supporting_memory import (
    CurrentLearningState,
    LearningStateSnapshot,
    RepairOutcome,
    StudentMisconception,
)
from src.database.models.user_account import UserAccount
from src.database.postgres_session import SessionFactory, get_session_factory
from src.database.unit_of_work import UnitOfWork
from src.ontology.ontology_seed_service import OntologySeedService, generate_skill_id
from src.services.auth_service import calculate_age, hash_password

DEFAULT_DEMO_PASSWORD = "pass123"


def seed_demo_environment(session_factory: SessionFactory | None = None) -> dict[str, str]:
    """Idempotently populate database with demo students, accounts, and cognitive history."""
    factory = session_factory if session_factory is not None else get_session_factory()

    with UnitOfWork(factory) as uow:
        assert uow.session is not None

        # -------------------------------------------------------------------
        # 1. Canonical Skills
        # -------------------------------------------------------------------
        linear_uuid = generate_skill_id("math :: algebra & functions :: linear equations")
        skill_linear = uow.session.scalar(
            select(CanonicalSkill).where(CanonicalSkill.canonical_name == "math :: algebra & functions :: linear equations")
        )
        if not skill_linear:
            skill_linear = CanonicalSkill(
                skill_id=linear_uuid,
                canonical_name="math :: algebra & functions :: linear equations",
                display_name="Linear Equations",
                description="Solving one-step and multi-step linear equations",
            )
            uow.session.add(skill_linear)
            uow.session.flush()

        pyth_uuid = generate_skill_id("math :: geometry & measurement :: pythagorean theorem")
        skill_pythagorean = uow.session.scalar(
            select(CanonicalSkill).where(CanonicalSkill.canonical_name == "math :: geometry & measurement :: pythagorean theorem")
        )
        if not skill_pythagorean:
            skill_pythagorean = CanonicalSkill(
                skill_id=pyth_uuid,
                canonical_name="math :: geometry & measurement :: pythagorean theorem",
                display_name="Pythagorean Theorem",
                description="Applying a^2 + b^2 = c^2 to right triangles",
            )
            uow.session.add(skill_pythagorean)
            uow.session.flush()

        pw_hash = hash_password(DEFAULT_DEMO_PASSWORD)

        # -------------------------------------------------------------------
        # 3. demo_new (STUDENT - Cold Start)
        # -------------------------------------------------------------------
        stud_new = uow.session.scalar(
            select(Student).where(Student.external_student_id == "demo_new")
        )
        if not stud_new:
            stud_new = Student(
                student_id=uuid.uuid4(),
                external_student_id="demo_new",
                display_name="Demo New Student",
                is_active=True,
            )
            uow.session.add(stud_new)
            uow.session.flush()

        acc_new = uow.session.scalar(
            select(UserAccount).where(UserAccount.username == "demo_new")
        )
        if not acc_new:
            dob_new = date(2008, 5, 14)
            acc_new = UserAccount(
                user_id=uuid.uuid4(),
                username="demo_new",
                password_hash=pw_hash,
                role="STUDENT",
                date_of_birth=dob_new,
                age=calculate_age(dob_new),
                student_id=stud_new.student_id,
            )
            uow.session.add(acc_new)

        # -------------------------------------------------------------------
        # 4. demo_developing (STUDENT - Returning with Linear Equations history)
        # -------------------------------------------------------------------
        stud_dev = uow.session.scalar(
            select(Student).where(Student.external_student_id == "demo_developing")
        )
        if not stud_dev:
            stud_dev = Student(
                student_id=uuid.uuid4(),
                external_student_id="demo_developing",
                display_name="Demo Developing Student",
                is_active=True,
            )
            uow.session.add(stud_dev)
            uow.session.flush()

        acc_dev = uow.session.scalar(
            select(UserAccount).where(UserAccount.username == "demo_developing")
        )
        if not acc_dev:
            dob_dev = date(2006, 3, 22)
            acc_dev = UserAccount(
                user_id=uuid.uuid4(),
                username="demo_developing",
                password_hash=pw_hash,
                role="STUDENT",
                date_of_birth=dob_dev,
                age=calculate_age(dob_dev),
                student_id=stud_dev.student_id,
            )
            uow.session.add(acc_dev)

        # Ensure session for demo_developing
        sess_dev = uow.session.scalar(
            select(LearningSession).where(
                LearningSession.student_id == stud_dev.student_id,
                LearningSession.external_session_id == "session_dev_01",
            )
        )
        if not sess_dev:
            sess_dev = LearningSession(
                session_id=uuid.uuid4(),
                student_id=stud_dev.student_id,
                external_session_id="session_dev_01",
                status="ACTIVE",
            )
            uow.session.add(sess_dev)
            uow.session.flush()

        # Check if interactions already exist for demo_developing
        # Add 2 interaction logs for demo_developing
        if not uow.session.scalar(
            select(InteractionLog).where(
                InteractionLog.source == "EVALUATOR",
                InteractionLog.external_interaction_id == "dev_act_01",
            )
        ):
            uow.session.add(InteractionLog(
                interaction_id=uuid.uuid4(),
                student_id=stud_dev.student_id,
                session_id=sess_dev.session_id,
                canonical_skill_id=skill_linear.skill_id,
                source="EVALUATOR",
                external_interaction_id="dev_act_01",
                student_utterance="3x = 12, so x = 9",
                tutor_response="Remember to divide both sides by 3.",
                is_correct=False,
                identified_error="Subtracted coefficient instead of dividing",
                attempt_count=2,
                hint_count=1,
                response_time_ms=6200.0,
            ))
        if not uow.session.scalar(
            select(InteractionLog).where(
                InteractionLog.source == "EVALUATOR",
                InteractionLog.external_interaction_id == "dev_act_02",
            )
        ):
            uow.session.add(InteractionLog(
                interaction_id=uuid.uuid4(),
                student_id=stud_dev.student_id,
                session_id=sess_dev.session_id,
                canonical_skill_id=skill_linear.skill_id,
                source="EVALUATOR",
                external_interaction_id="dev_act_02",
                student_utterance="2x + 4 = 10, 2x = 6, x = 3",
                tutor_response="Excellent work!",
                is_correct=True,
                attempt_count=1,
                hint_count=0,
                response_time_ms=4500.0,
            ))

            # Concept Memory
            if not uow.session.scalar(
                select(ConceptMemory).where(
                    ConceptMemory.student_id == stud_dev.student_id,
                    ConceptMemory.canonical_skill_id == skill_linear.skill_id,
                )
            ):
                uow.session.add(ConceptMemory(
                    student_id=stud_dev.student_id,
                    canonical_skill_id=skill_linear.skill_id,
                    interaction_count=2,
                    correct_count=1,
                    incorrect_count=1,
                    accuracy=0.50,
                    attempt_observation_count=2,
                    attempt_sum=3,
                    hint_observation_count=2,
                    hint_sum=1,
                    response_time_observation_count=2,
                    response_time_sum_ms=10700.0,
                ))

            # Short Term Memory
            if not uow.session.scalar(
                select(ShortTermMemory).where(
                    ShortTermMemory.student_id == stud_dev.student_id,
                    ShortTermMemory.session_id == sess_dev.session_id,
                )
            ):
                uow.session.add(ShortTermMemory(
                    session_id=sess_dev.session_id,
                    student_id=stud_dev.student_id,
                    interaction_count=2,
                    correct_count=1,
                    incorrect_count=1,
                    recent_accuracy=0.50,
                    attempt_observation_count=2,
                    attempt_sum=3,
                    hint_observation_count=2,
                    hint_sum=1,
                    response_time_observation_count=2,
                    response_time_sum_ms=10700.0,
                ))

            # Long Term Memory
            if not uow.session.scalar(
                select(LongTermMemory).where(
                    LongTermMemory.student_id == stud_dev.student_id,
                )
            ):
                uow.session.add(LongTermMemory(
                    student_id=stud_dev.student_id,
                    total_sessions=1,
                    concept_count=1,
                    interaction_count=2,
                    correct_count=1,
                    incorrect_count=1,
                    overall_accuracy=0.50,
                    attempt_observation_count=2,
                    attempt_sum=3,
                    hint_observation_count=2,
                    hint_sum=1,
                    response_time_observation_count=2,
                    response_time_sum_ms=10700.0,
                ))

            # Misconception
            if not uow.session.scalar(
                select(StudentMisconception).where(
                    StudentMisconception.student_id == stud_dev.student_id,
                    StudentMisconception.canonical_skill_id == skill_linear.skill_id,
                )
            ):
                uow.session.add(StudentMisconception(
                    misconception_id=uuid.uuid4(),
                    student_id=stud_dev.student_id,
                    canonical_skill_id=skill_linear.skill_id,
                    normalized_error="subtracted_coefficient_instead_of_dividing",
                    display_error="Subtracted coefficient instead of dividing",
                    occurrence_count=1,
                ))

            # Snapshot & Current Learning State
            snap_dev = uow.session.scalar(
                select(LearningStateSnapshot).where(
                    LearningStateSnapshot.student_id == stud_dev.student_id,
                    LearningStateSnapshot.canonical_skill_id == skill_linear.skill_id,
                )
            )
            if not snap_dev:
                snap_dev = LearningStateSnapshot(
                    student_id=stud_dev.student_id,
                    session_id=sess_dev.session_id,
                    canonical_skill_id=skill_linear.skill_id,
                    learning_state="DEVELOPING",
                    evidence_level="FULL_SKILL",
                    evidence_strength="MEDIUM",
                    behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                    model_used=True,
                    previous_interaction_count=1,
                    previous_skill_interaction_count=1,
                    recent_interaction_count=2,
                    attempt_observation_count=2,
                    hint_observation_count=2,
                    response_time_observation_count=2,
                )
                uow.session.add(snap_dev)
                uow.session.flush()

            if not uow.session.scalar(
                select(CurrentLearningState).where(
                    CurrentLearningState.student_id == stud_dev.student_id,
                    CurrentLearningState.canonical_skill_id == skill_linear.skill_id,
                )
            ):
                uow.session.add(CurrentLearningState(
                    student_id=stud_dev.student_id,
                    canonical_skill_id=skill_linear.skill_id,
                    learning_state="DEVELOPING",
                    evidence_level="FULL_SKILL",
                    evidence_strength="MEDIUM",
                    behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                    model_used=True,
                    recent_interaction_count=2,
                    attempt_observation_count=2,
                    hint_observation_count=2,
                    response_time_observation_count=2,
                    last_snapshot_id=snap_dev.snapshot_id if snap_dev else None,
                ))

        # -------------------------------------------------------------------
        # 5. demo_strong (STUDENT - Strong with Pythagorean Theorem)
        # -------------------------------------------------------------------
        stud_str = uow.session.scalar(
            select(Student).where(Student.external_student_id == "demo_strong")
        )
        if not stud_str:
            stud_str = Student(
                student_id=uuid.uuid4(),
                external_student_id="demo_strong",
                display_name="Demo Strong Student",
                is_active=True,
            )
            uow.session.add(stud_str)
            uow.session.flush()

        acc_str = uow.session.scalar(
            select(UserAccount).where(UserAccount.username == "demo_strong")
        )
        if not acc_str:
            dob_str = date(2005, 11, 8)
            acc_str = UserAccount(
                user_id=uuid.uuid4(),
                username="demo_strong",
                password_hash=pw_hash,
                role="STUDENT",
                date_of_birth=dob_str,
                age=calculate_age(dob_str),
                student_id=stud_str.student_id,
            )
            uow.session.add(acc_str)

        sess_str = uow.session.scalar(
            select(LearningSession).where(
                LearningSession.student_id == stud_str.student_id,
                LearningSession.external_session_id == "session_str_01",
            )
        )
        if not sess_str:
            sess_str = LearningSession(
                session_id=uuid.uuid4(),
                student_id=stud_str.student_id,
                external_session_id="session_str_01",
                status="ACTIVE",
            )
            uow.session.add(sess_str)
            uow.session.flush()

        cm_str = uow.session.scalar(
            select(ConceptMemory).where(
                ConceptMemory.student_id == stud_str.student_id,
                ConceptMemory.canonical_skill_id == skill_pythagorean.skill_id,
            )
        )
        if not cm_str:
            if not uow.session.scalar(
                select(InteractionLog).where(
                    InteractionLog.source == "EVALUATOR",
                    InteractionLog.external_interaction_id == "str_act_01",
                )
            ):
                uow.session.add(InteractionLog(
                    interaction_id=uuid.uuid4(),
                    student_id=stud_str.student_id,
                    session_id=sess_str.session_id,
                    canonical_skill_id=skill_pythagorean.skill_id,
                    source="EVALUATOR",
                    external_interaction_id="str_act_01",
                    student_utterance="3^2 + 4^2 = 9 + 16 = 25, so hypotenuse is 5",
                    tutor_response="Correct!",
                    is_correct=True,
                    attempt_count=1,
                    hint_count=0,
                    response_time_ms=3100.0,
                ))
            if not uow.session.scalar(
                select(ConceptMemory).where(
                    ConceptMemory.student_id == stud_str.student_id,
                    ConceptMemory.canonical_skill_id == skill_pythagorean.skill_id,
                )
            ):
                uow.session.add(ConceptMemory(
                    student_id=stud_str.student_id,
                    canonical_skill_id=skill_pythagorean.skill_id,
                    interaction_count=3,
                    correct_count=3,
                    incorrect_count=0,
                    accuracy=1.0,
                    attempt_observation_count=3,
                    attempt_sum=3,
                    hint_observation_count=3,
                    hint_sum=0,
                    response_time_observation_count=3,
                    response_time_sum_ms=9000.0,
                ))

            if not uow.session.scalar(
                select(ShortTermMemory).where(
                    ShortTermMemory.student_id == stud_str.student_id,
                    ShortTermMemory.session_id == sess_str.session_id,
                )
            ):
                uow.session.add(ShortTermMemory(
                    session_id=sess_str.session_id,
                    student_id=stud_str.student_id,
                    interaction_count=3,
                    correct_count=3,
                    incorrect_count=0,
                    recent_accuracy=1.0,
                    attempt_observation_count=3,
                    attempt_sum=3,
                    hint_observation_count=3,
                    hint_sum=0,
                    response_time_observation_count=3,
                    response_time_sum_ms=9000.0,
                ))

            if not uow.session.scalar(
                select(LongTermMemory).where(
                    LongTermMemory.student_id == stud_str.student_id,
                )
            ):
                uow.session.add(LongTermMemory(
                    student_id=stud_str.student_id,
                    total_sessions=1,
                    concept_count=1,
                    interaction_count=3,
                    correct_count=3,
                    incorrect_count=0,
                    overall_accuracy=1.0,
                    attempt_observation_count=3,
                    attempt_sum=3,
                    hint_observation_count=3,
                    hint_sum=0,
                    response_time_observation_count=3,
                    response_time_sum_ms=9000.0,
                ))

            snap_str = uow.session.scalar(
                select(LearningStateSnapshot).where(
                    LearningStateSnapshot.student_id == stud_str.student_id,
                    LearningStateSnapshot.canonical_skill_id == skill_pythagorean.skill_id,
                )
            )
            if not snap_str:
                snap_str = LearningStateSnapshot(
                    student_id=stud_str.student_id,
                    session_id=sess_str.session_id,
                    canonical_skill_id=skill_pythagorean.skill_id,
                    learning_state="STRONG",
                    evidence_level="FULL_SKILL",
                    evidence_strength="HIGH",
                    behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                    model_used=True,
                    previous_interaction_count=2,
                    previous_skill_interaction_count=2,
                    recent_interaction_count=3,
                    attempt_observation_count=3,
                    hint_observation_count=3,
                    response_time_observation_count=3,
                )
                uow.session.add(snap_str)
                uow.session.flush()

            if not uow.session.scalar(
                select(CurrentLearningState).where(
                    CurrentLearningState.student_id == stud_str.student_id,
                    CurrentLearningState.canonical_skill_id == skill_pythagorean.skill_id,
                )
            ):
                uow.session.add(CurrentLearningState(
                    student_id=stud_str.student_id,
                    canonical_skill_id=skill_pythagorean.skill_id,
                    learning_state="STRONG",
                    evidence_level="FULL_SKILL",
                    evidence_strength="HIGH",
                    behavioural_coverage="FULL_BEHAVIOURAL_COVERAGE",
                    model_used=True,
                    recent_interaction_count=3,
                    attempt_observation_count=3,
                    hint_observation_count=3,
                    response_time_observation_count=3,
                    last_snapshot_id=snap_str.snapshot_id if snap_str else None,
                ))

        # -------------------------------------------------------------------
        # 6. demo_support (STUDENT - 3+ Successful HINT Repair Outcomes)
        # -------------------------------------------------------------------
        stud_sup = uow.session.scalar(
            select(Student).where(Student.external_student_id == "demo_support")
        )
        if not stud_sup:
            stud_sup = Student(
                student_id=uuid.uuid4(),
                external_student_id="demo_support",
                display_name="Demo Support Student",
                is_active=True,
            )
            uow.session.add(stud_sup)
            uow.session.flush()

        acc_sup = uow.session.scalar(
            select(UserAccount).where(UserAccount.username == "demo_support")
        )
        if not acc_sup:
            dob_sup = date(2007, 9, 30)
            acc_sup = UserAccount(
                user_id=uuid.uuid4(),
                username="demo_support",
                password_hash=pw_hash,
                role="STUDENT",
                date_of_birth=dob_sup,
                age=calculate_age(dob_sup),
                student_id=stud_sup.student_id,
            )
            uow.session.add(acc_sup)

        # 3 sessions for distinct repairs
        sess_sup_list = []
        for i in range(1, 4):
            sess_ext = f"session_sup_0{i}"
            s_obj = uow.session.scalar(
                select(LearningSession).where(
                    LearningSession.student_id == stud_sup.student_id,
                    LearningSession.external_session_id == sess_ext,
                )
            )
            if not s_obj:
                s_obj = LearningSession(
                    session_id=uuid.uuid4(),
                    student_id=stud_sup.student_id,
                    external_session_id=sess_ext,
                    status="ACTIVE",
                )
                uow.session.add(s_obj)
                uow.session.flush()
            sess_sup_list.append(s_obj)

        repairs_sup = uow.session.scalars(
            select(RepairOutcome).where(RepairOutcome.student_id == stud_sup.student_id)
        ).all()
        if len(repairs_sup) < 3:
            for idx, s_obj in enumerate(sess_sup_list):
                uow.session.add(RepairOutcome(
                    repair_outcome_id=uuid.uuid4(),
                    student_id=stud_sup.student_id,
                    session_id=s_obj.session_id,
                    canonical_skill_id=skill_linear.skill_id,
                    repair_action="HINT",
                    outcome="RESOLVED",
                    score=1.0,
                    notes=f"Scaffolded hint #{idx+1} enabled student to self-correct linear equation.",
                ))

        # -------------------------------------------------------------------
        # 7. student01 / Student01 (STUDENT)
        # -------------------------------------------------------------------
        for uname in ["student01", "Student01"]:
            stud_01 = uow.session.scalar(
                select(Student).where(Student.external_student_id == uname)
            )
            if not stud_01:
                stud_01 = Student(
                    student_id=uuid.uuid4(),
                    external_student_id=uname,
                    display_name=uname,
                    is_active=True,
                )
                uow.session.add(stud_01)
                uow.session.flush()

            acc_01 = uow.session.scalar(
                select(UserAccount).where(UserAccount.username == uname)
            )
            if not acc_01:
                dob_01 = date(2007, 1, 15)
                acc_01 = UserAccount(
                    user_id=uuid.uuid4(),
                    username=uname,
                    password_hash=pw_hash,
                    role="STUDENT",
                    date_of_birth=dob_01,
                    age=calculate_age(dob_01),
                    student_id=stud_01.student_id,
                )
                uow.session.add(acc_01)

        uow.session.commit()

    return {
        "demo_new": "demo_new",
        "demo_developing": "demo_developing",
        "demo_strong": "demo_strong",
        "demo_support": "demo_support",
        "student01": "student01",
        "password": DEFAULT_DEMO_PASSWORD,
    }


if __name__ == "__main__":
    print("Seeding deterministic demo data...")
    res = seed_demo_environment()
    print("Demo data seeded successfully:")
    for k, v in res.items():
        print(f"  {k}: {v}")
