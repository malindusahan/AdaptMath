import pytest

from src.database.assessment_repository import store_assessment
from src.database.connection import get_connection, initialize_database
from src.database.state_repository import (
    AssessmentContextMismatchError,
    StateRepositoryError,
    persist_learning_state,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
    StoredLearningStateSnapshot,
    StudentHistoryPredictionResponse,
)


def make_assessment(
    student_id="s1", topic="Algebra", subtopic="Linear equations"
):
    return AssessmentMemoryUpdateRequest(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        assessment_questions=[
            AssessmentQuestionResult(question_id="q1", is_correct=True)
        ],
    )


def make_prediction(
    *,
    student_id="s1",
    topic="Algebra",
    subtopic="Linear equations",
    learning_state="DEVELOPING",
    evidence_level="FULL_SKILL",
    evidence_strength="HIGH",
    behavioural_coverage="CORRECTNESS_ONLY_COVERAGE",
    model_used=True,
    previous_interaction_count=10,
    previous_skill_interaction_count=6,
    recent_interaction_count=10,
    attempt_observation_count=0,
    hint_observation_count=0,
    response_time_observation_count=0,
):
    return StudentHistoryPredictionResponse(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        learning_state=learning_state,
        evidence_level=evidence_level,
        evidence_strength=evidence_strength,
        behavioural_coverage=behavioural_coverage,
        model_used=model_used,
        previous_interaction_count=previous_interaction_count,
        previous_skill_interaction_count=previous_skill_interaction_count,
        recent_interaction_count=recent_interaction_count,
        attempt_observation_count=attempt_observation_count,
        hint_observation_count=hint_observation_count,
        response_time_observation_count=response_time_observation_count,
    )


def test_extended_schema_contains_coverage_columns(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    expected = {
        "behavioural_coverage",
        "recent_interaction_count",
        "attempt_observation_count",
        "hint_observation_count",
        "response_time_observation_count",
    }
    with get_connection(database_path) as connection:
        snapshot_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(learning_state_snapshots)"
            ).fetchall()
        }
        memory_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(student_memory)").fetchall()
        }
    assert expected.issubset(snapshot_columns)
    assert expected.issubset(memory_columns)


def test_persist_state_creates_snapshot(tmp_path):
    database_path = tmp_path / "memory.db"
    assessment = store_assessment(make_assessment(), database_path)
    result = persist_learning_state(
        make_prediction(), assessment.assessment_id, database_path
    )
    assert isinstance(result, StoredLearningStateSnapshot)
    assert result.snapshot_id > 0
    assert result.assessment_id == assessment.assessment_id
    assert result.learning_state == "DEVELOPING"
    assert result.persisted is True
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM learning_state_snapshots WHERE snapshot_id = ?",
            (result.snapshot_id,),
        ).fetchone()
    assert row["learning_state"] == "DEVELOPING"
    assert row["evidence_level"] == "FULL_SKILL"
    assert row["behavioural_coverage"] == "CORRECTNESS_ONLY_COVERAGE"


def test_persist_state_creates_current_memory(tmp_path):
    database_path = tmp_path / "memory.db"
    assessment = store_assessment(make_assessment(), database_path)
    result = persist_learning_state(
        make_prediction(), assessment.assessment_id, database_path
    )
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT * FROM student_memory
            WHERE student_id = ? AND topic = ? AND subtopic = ?
            """,
            ("s1", "Algebra", "Linear equations"),
        ).fetchone()
    assert row["current_learning_state"] == "DEVELOPING"
    assert row["last_assessment_id"] == assessment.assessment_id
    assert row["last_snapshot_id"] == result.snapshot_id


def test_second_snapshot_updates_memory_not_history(tmp_path):
    database_path = tmp_path / "memory.db"
    first_assessment = store_assessment(make_assessment(), database_path)
    first = persist_learning_state(
        make_prediction(learning_state="DEVELOPING"),
        first_assessment.assessment_id,
        database_path,
    )
    second_assessment = store_assessment(
        AssessmentMemoryUpdateRequest(
            student_id="s1",
            topic="Algebra",
            subtopic="Linear equations",
            assessment_questions=[
                AssessmentQuestionResult(question_id="q2", is_correct=False)
            ],
        ),
        database_path,
    )
    second = persist_learning_state(
        make_prediction(
            learning_state="NEEDS_SUPPORT",
            previous_interaction_count=11,
            previous_skill_interaction_count=7,
        ),
        second_assessment.assessment_id,
        database_path,
    )
    with get_connection(database_path) as connection:
        snapshot_count = connection.execute(
            "SELECT COUNT(*) FROM learning_state_snapshots WHERE student_id = ?",
            ("s1",),
        ).fetchone()[0]
        memory_count = connection.execute(
            "SELECT COUNT(*) FROM student_memory WHERE student_id = ?", ("s1",)
        ).fetchone()[0]
        memory = connection.execute(
            """
            SELECT * FROM student_memory
            WHERE student_id = ? AND topic = ? AND subtopic = ?
            """,
            ("s1", "Algebra", "Linear equations"),
        ).fetchone()
    assert snapshot_count == 2
    assert memory_count == 1
    assert first.snapshot_id != second.snapshot_id
    assert memory["current_learning_state"] == "NEEDS_SUPPORT"
    assert memory["last_snapshot_id"] == second.snapshot_id


def test_context_mismatch_rolls_back_everything(tmp_path):
    database_path = tmp_path / "memory.db"
    assessment = store_assessment(make_assessment(student_id="s1"), database_path)
    with pytest.raises(AssessmentContextMismatchError):
        persist_learning_state(
            make_prediction(student_id="s2"),
            assessment.assessment_id,
            database_path,
        )
    with get_connection(database_path) as connection:
        snapshots = connection.execute(
            "SELECT COUNT(*) FROM learning_state_snapshots"
        ).fetchone()[0]
        memories = connection.execute(
            "SELECT COUNT(*) FROM student_memory"
        ).fetchone()[0]
    assert snapshots == 0
    assert memories == 0


def test_nonexistent_assessment_rejected(tmp_path):
    with pytest.raises(AssessmentContextMismatchError):
        persist_learning_state(
            make_prediction(), 999, tmp_path / "memory.db"
        )


def test_null_subtopic_is_normalized_for_current_memory(tmp_path):
    database_path = tmp_path / "memory.db"
    assessment = store_assessment(make_assessment(subtopic=None), database_path)
    result = persist_learning_state(
        make_prediction(subtopic=None), assessment.assessment_id, database_path
    )
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


def test_observation_counts_are_preserved(tmp_path):
    database_path = tmp_path / "memory.db"
    assessment = store_assessment(make_assessment(), database_path)
    prediction = make_prediction(
        behavioural_coverage="PARTIAL_BEHAVIOURAL_COVERAGE",
        attempt_observation_count=7,
        hint_observation_count=2,
        response_time_observation_count=0,
    )
    result = persist_learning_state(
        prediction, assessment.assessment_id, database_path
    )
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM learning_state_snapshots WHERE snapshot_id = ?",
            (result.snapshot_id,),
        ).fetchone()
    assert row["attempt_observation_count"] == 7
    assert row["hint_observation_count"] == 2
    assert row["response_time_observation_count"] == 0
    assert row["behavioural_coverage"] == "PARTIAL_BEHAVIOURAL_COVERAGE"


def test_cold_start_snapshot_can_be_persisted_without_assessment(tmp_path):
    prediction = make_prediction(
        learning_state="UNAVAILABLE",
        evidence_level="COLD_START",
        evidence_strength="NONE",
        behavioural_coverage="CORRECTNESS_ONLY_COVERAGE",
        model_used=False,
        previous_interaction_count=0,
        previous_skill_interaction_count=0,
        recent_interaction_count=0,
    )
    result = persist_learning_state(
        prediction, None, tmp_path / "memory.db"
    )
    assert result.learning_state == "UNAVAILABLE"
    assert result.assessment_id is None
    assert result.model_used is False


@pytest.mark.parametrize("assessment_id", [0, -1, 1.5, True])
def test_invalid_assessment_id_rejected(tmp_path, assessment_id):
    with pytest.raises(StateRepositoryError):
        persist_learning_state(
            make_prediction(), assessment_id, tmp_path / "memory.db"
        )
