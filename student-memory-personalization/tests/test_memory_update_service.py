import pytest

from src.database.connection import get_connection
from src.database.misconception_repository import get_student_misconceptions
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
    MemoryUpdateResponse,
)
from src.services.memory_update_service import (
    MemoryUpdateService,
    MemoryUpdateServiceError,
    get_memory_update_service,
    update_student_memory,
)


@pytest.fixture
def service():
    return MemoryUpdateService()


def make_request(
    *,
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
    prefix="q",
    correctness=None,
    identified_errors=None,
    overall_feedback=None,
    attempts=None,
    hints=None,
    hint_totals=None,
    response_times=None,
):
    correctness = [True, False, True] if correctness is None else correctness
    questions = []
    for index, correct in enumerate(correctness, start=1):
        position = index - 1
        question_error = identified_errors[0] if identified_errors and not correct else None
        questions.append(
            AssessmentQuestionResult(
                question_id=f"{prefix}{index}",
                question=f"Question {index}",
                student_answer="student answer",
                expected_answer="expected answer",
                is_correct=correct,
                identified_error=question_error,
                attempt_count=None if attempts is None else attempts[position],
                hint_count=None if hints is None else hints[position],
                hint_total=None if hint_totals is None else hint_totals[position],
                response_time_ms=(
                    None if response_times is None else response_times[position]
                ),
            )
        )
    return AssessmentMemoryUpdateRequest(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        assessment_questions=questions,
        identified_errors=[] if identified_errors is None else identified_errors,
        overall_feedback=overall_feedback,
    )


def test_complete_memory_update_returns_schema(service, tmp_path):
    result = service.update_memory(make_request(), tmp_path / "memory.db")
    assert isinstance(result, MemoryUpdateResponse)
    assert result.student_id == "s1"
    assert result.topic == "Algebra"
    assert result.subtopic == "Linear equations"
    assert result.assessment_id > 0
    assert result.snapshot_id > 0
    assert result.memory_updated is True


def test_first_assessment_is_included_in_updated_state(service, tmp_path):
    result = service.update_memory(
        make_request(correctness=[True, False, True]), tmp_path / "memory.db"
    )
    assert result.previous_interaction_count == 3
    assert result.previous_skill_interaction_count == 3
    assert result.learning_state in {"NEEDS_SUPPORT", "DEVELOPING", "STRONG"}
    assert result.model_used is True
    assert result.evidence_level == "PARTIAL_SKILL"
    assert result.evidence_strength == "MEDIUM"


def test_correctness_only_coverage_is_preserved(service, tmp_path):
    result = service.update_memory(make_request(), tmp_path / "memory.db")
    assert result.behavioural_coverage == "CORRECTNESS_ONLY_COVERAGE"
    assert result.attempt_observation_count == 0
    assert result.hint_observation_count == 0
    assert result.response_time_observation_count == 0


def test_full_behavioural_coverage_flows_through(service, tmp_path):
    result = service.update_memory(
        make_request(
            correctness=[True, False, True, True, False],
            attempts=[1, 2, 1, 3, 1],
            hints=[0, 1, 0, 1, 0],
            hint_totals=[2, 2, 2, 2, 2],
            response_times=[1000, 2000, 1500, 2500, 1800],
        ),
        tmp_path / "memory.db",
    )
    assert result.behavioural_coverage == "FULL_BEHAVIOURAL_COVERAGE"
    assert result.attempt_observation_count == 5
    assert result.hint_observation_count == 5
    assert result.response_time_observation_count == 5


def test_assessment_is_persisted(service, tmp_path):
    database_path = tmp_path / "memory.db"
    result = service.update_memory(make_request(), database_path)
    with get_connection(database_path) as connection:
        assessment = connection.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?",
            (result.assessment_id,),
        ).fetchone()
        interactions = connection.execute(
            "SELECT COUNT(*) FROM assessment_interactions WHERE assessment_id = ?",
            (result.assessment_id,),
        ).fetchone()[0]
    assert assessment is not None
    assert interactions == 3


def test_snapshot_and_current_memory_are_persisted(service, tmp_path):
    database_path = tmp_path / "memory.db"
    result = service.update_memory(make_request(), database_path)
    with get_connection(database_path) as connection:
        snapshot = connection.execute(
            "SELECT * FROM learning_state_snapshots WHERE snapshot_id = ?",
            (result.snapshot_id,),
        ).fetchone()
        memory = connection.execute(
            """
            SELECT * FROM student_memory
            WHERE student_id = ? AND topic = ? AND subtopic = ?
            """,
            ("s1", "Algebra", "Linear equations"),
        ).fetchone()
    assert snapshot["learning_state"] == result.learning_state
    assert memory["current_learning_state"] == result.learning_state
    assert memory["last_assessment_id"] == result.assessment_id
    assert memory["last_snapshot_id"] == result.snapshot_id


def test_misconceptions_are_updated(service, tmp_path):
    database_path = tmp_path / "memory.db"
    result = service.update_memory(
        make_request(correctness=[True, False], identified_errors=["addition error"]),
        database_path,
    )
    misconceptions = get_student_misconceptions(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert len(misconceptions) == 1
    assert result.misconception_count == 1
    assert misconceptions[0].misconception == "addition error"
    assert misconceptions[0].occurrence_count == 1


def test_repeated_misconception_increments_memory(service, tmp_path):
    database_path = tmp_path / "memory.db"
    service.update_memory(
        make_request(
            prefix="first",
            correctness=[False],
            identified_errors=["addition error"],
        ),
        database_path,
    )
    result = service.update_memory(
        make_request(
            prefix="second",
            correctness=[False],
            identified_errors=["Addition Error"],
        ),
        database_path,
    )
    misconceptions = get_student_misconceptions(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert len(misconceptions) == 1
    assert result.misconception_count == 1
    assert misconceptions[0].occurrence_count == 2


def test_second_assessment_advances_history_and_snapshot(service, tmp_path):
    database_path = tmp_path / "memory.db"
    first = service.update_memory(
        make_request(prefix="first", correctness=[True, False, True]), database_path
    )
    second = service.update_memory(
        make_request(prefix="second", correctness=[True, True]), database_path
    )
    assert first.previous_interaction_count == 3
    assert second.previous_interaction_count == 5
    assert second.previous_skill_interaction_count == 5
    assert second.evidence_level == "FULL_SKILL"
    assert second.evidence_strength == "HIGH"
    assert first.snapshot_id != second.snapshot_id
    assert first.assessment_id != second.assessment_id
    with get_connection(database_path) as connection:
        snapshots = connection.execute(
            "SELECT COUNT(*) FROM learning_state_snapshots WHERE student_id = ?",
            ("s1",),
        ).fetchone()[0]
        memories = connection.execute(
            "SELECT COUNT(*) FROM student_memory WHERE student_id = ?", ("s1",)
        ).fetchone()[0]
    assert snapshots == 2
    assert memories == 1


def test_topics_have_independent_current_memory(service, tmp_path):
    database_path = tmp_path / "memory.db"
    service.update_memory(
        make_request(prefix="alg", topic="Algebra", subtopic="Linear equations"),
        database_path,
    )
    service.update_memory(
        make_request(prefix="geo", topic="Geometry", subtopic="Angles"),
        database_path,
    )
    with get_connection(database_path) as connection:
        memories = connection.execute(
            "SELECT * FROM student_memory WHERE student_id = ? ORDER BY topic",
            ("s1",),
        ).fetchall()
    assert len(memories) == 2
    assert {row["topic"] for row in memories} == {"Algebra", "Geometry"}


def test_null_subtopic_supported_end_to_end(service, tmp_path):
    database_path = tmp_path / "memory.db"
    result = service.update_memory(make_request(subtopic=None), database_path)
    assert result.subtopic is None
    with get_connection(database_path) as connection:
        snapshot = connection.execute(
            "SELECT subtopic FROM learning_state_snapshots WHERE snapshot_id = ?",
            (result.snapshot_id,),
        ).fetchone()
        memory = connection.execute(
            "SELECT subtopic FROM student_memory WHERE student_id = ? AND topic = ?",
            ("s1", "Algebra"),
        ).fetchone()
    assert snapshot["subtopic"] is None
    assert memory["subtopic"] == ""


def test_response_serializes_cleanly(service, tmp_path):
    payload = service.update_memory(
        make_request(), tmp_path / "memory.db"
    ).model_dump()
    assert set(payload) == {
        "student_id", "topic", "subtopic", "assessment_id", "snapshot_id",
        "learning_state", "evidence_level", "evidence_strength",
        "behavioural_coverage", "model_used", "previous_interaction_count",
        "previous_skill_interaction_count", "recent_interaction_count",
        "attempt_observation_count", "hint_observation_count",
        "response_time_observation_count", "misconception_count",
        "misconceptions",
        "memory_updated",
    }


def test_convenience_interface(tmp_path):
    result = update_student_memory(make_request(), tmp_path / "memory.db")
    assert isinstance(result, MemoryUpdateResponse)
    assert result.memory_updated is True


def test_service_singleton_reused():
    assert get_memory_update_service() is get_memory_update_service()


def test_invalid_request_type_rejected(service, tmp_path):
    with pytest.raises(MemoryUpdateServiceError):
        service.update_memory({"student_id": "s1"}, tmp_path / "memory.db")
