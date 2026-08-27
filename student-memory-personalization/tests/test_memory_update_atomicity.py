import pytest

from src.database.connection import get_connection, initialize_database
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)
from src.services.memory_update_service import (
    MemoryUpdateService,
    MemoryUpdateServiceError,
)
from src.services.student_state_service import StudentStateService


def make_request():
    return AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q1",
                is_correct=False,
                identified_error="addition error",
            ),
            AssessmentQuestionResult(question_id="q2", is_correct=True),
        ],
        identified_errors=["addition error"],
    )


def get_counts(database_path):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        return {
            "assessments": connection.execute(
                "SELECT COUNT(*) FROM assessments"
            ).fetchone()[0],
            "interactions": connection.execute(
                "SELECT COUNT(*) FROM assessment_interactions"
            ).fetchone()[0],
            "misconceptions": connection.execute(
                "SELECT COUNT(*) FROM student_misconceptions"
            ).fetchone()[0],
            "snapshots": connection.execute(
                "SELECT COUNT(*) FROM learning_state_snapshots"
            ).fetchone()[0],
            "memory": connection.execute(
                "SELECT COUNT(*) FROM student_memory"
            ).fetchone()[0],
        }


def test_successful_update_commits_everything(tmp_path):
    database_path = tmp_path / "memory.db"
    result = MemoryUpdateService().update_memory(make_request(), database_path)
    assert result.memory_updated is True
    assert get_counts(database_path) == {
        "assessments": 1,
        "interactions": 2,
        "misconceptions": 1,
        "snapshots": 1,
        "memory": 1,
    }


class FailingStateService(StudentStateService):
    def predict_from_stored_history(self, *args, **kwargs):
        raise RuntimeError("Simulated prediction failure")


def test_prediction_failure_rolls_back_everything(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService(state_service=FailingStateService())
    with pytest.raises(MemoryUpdateServiceError):
        service.update_memory(make_request(), database_path)
    assert get_counts(database_path) == {
        "assessments": 0,
        "interactions": 0,
        "misconceptions": 0,
        "snapshots": 0,
        "memory": 0,
    }


def test_uncommitted_assessment_is_visible_to_prediction(tmp_path):
    result = MemoryUpdateService().update_memory(
        make_request(), tmp_path / "memory.db"
    )
    assert result.previous_interaction_count == 2
    assert result.previous_skill_interaction_count == 2


def test_existing_committed_data_survives_later_failed_update(tmp_path):
    database_path = tmp_path / "memory.db"
    MemoryUpdateService().update_memory(make_request(), database_path)
    before = get_counts(database_path)

    second_request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="later-q1",
                is_correct=False,
                identified_error="sign error",
            )
        ],
        identified_errors=["sign error"],
    )
    failing_service = MemoryUpdateService(state_service=FailingStateService())
    with pytest.raises(MemoryUpdateServiceError):
        failing_service.update_memory(second_request, database_path)

    assert get_counts(database_path) == before
