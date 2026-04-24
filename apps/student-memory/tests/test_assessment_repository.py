import pytest

from src.database.assessment_repository import (
    AssessmentRepositoryError,
    DuplicateAssessmentQuestionError,
    store_assessment,
)
from src.database.connection import get_connection, initialize_database
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
    StoredAssessmentResult,
)


def make_request():
    return AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q1",
                question="What is x if 3x = 15?",
                student_answer="5",
                expected_answer="5",
                is_correct=True,
            ),
            AssessmentQuestionResult(
                question_id="q2",
                question="Solve x - 2 = 7.",
                student_answer="x = 8",
                expected_answer="x = 9",
                is_correct=False,
                identified_error="addition error",
            ),
        ],
        identified_errors=["addition error"],
        overall_feedback="Mostly correct, but review addition in linear equations.",
    )


def test_store_assessment_success(tmp_path):
    result = store_assessment(make_request(), tmp_path / "memory.db")

    assert isinstance(result, StoredAssessmentResult)
    assert result.assessment_id > 0
    assert result.student_id == "s1"
    assert result.topic == "Algebra"
    assert result.total_questions == 2
    assert result.correct_count == 1
    assert result.wrong_count == 1
    assert result.stored is True


def test_assessment_and_interactions_are_written(tmp_path):
    database_path = tmp_path / "memory.db"
    result = store_assessment(make_request(), database_path)

    with get_connection(database_path) as connection:
        assessment = connection.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?",
            (result.assessment_id,),
        ).fetchone()
        interactions = connection.execute(
            """
            SELECT * FROM assessment_interactions
            WHERE assessment_id = ?
            ORDER BY interaction_id
            """,
            (result.assessment_id,),
        ).fetchall()

    assert assessment is not None
    assert assessment["student_id"] == "s1"
    assert assessment["topic"] == "Algebra"
    assert len(interactions) == 2
    assert interactions[0]["question_id"] == "q1"
    assert interactions[0]["is_correct"] == 1
    assert interactions[1]["question_id"] == "q2"
    assert interactions[1]["is_correct"] == 0
    assert interactions[1]["identified_error"] == "addition error"


def test_missing_behaviour_is_preserved_as_null(tmp_path):
    database_path = tmp_path / "memory.db"
    result = store_assessment(make_request(), database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT attempt_count, hint_count, hint_total, response_time_ms,
                   attempt_data_available, hint_data_available,
                   response_time_available
            FROM assessment_interactions
            WHERE assessment_id = ? AND question_id = ?
            """,
            (result.assessment_id, "q1"),
        ).fetchone()

    assert row["attempt_count"] is None
    assert row["hint_count"] is None
    assert row["hint_total"] is None
    assert row["response_time_ms"] is None
    assert row["attempt_data_available"] == 0
    assert row["hint_data_available"] == 0
    assert row["response_time_available"] == 0


def test_supplied_behaviour_sets_availability_flags(tmp_path):
    database_path = tmp_path / "memory.db"
    request = AssessmentMemoryUpdateRequest(
        student_id="s2",
        topic="Fractions",
        subtopic="Addition",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q1",
                question="1/2 + 1/2?",
                student_answer="1",
                expected_answer="1",
                is_correct=True,
                attempt_count=2,
                hint_count=1,
                hint_total=3,
                response_time_ms=12500,
            )
        ],
    )
    result = store_assessment(request, database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM assessment_interactions WHERE assessment_id = ?",
            (result.assessment_id,),
        ).fetchone()

    assert row["attempt_count"] == 2
    assert row["hint_count"] == 1
    assert row["hint_total"] == 3
    assert row["response_time_ms"] == 12500
    assert row["attempt_data_available"] == 1
    assert row["hint_data_available"] == 1
    assert row["response_time_available"] == 1


def test_duplicate_question_ids_are_rejected_before_write(tmp_path):
    database_path = tmp_path / "memory.db"
    request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        assessment_questions=[
            AssessmentQuestionResult(question_id="q1", is_correct=True),
            AssessmentQuestionResult(question_id="q1", is_correct=False),
        ],
    )

    with pytest.raises(DuplicateAssessmentQuestionError):
        store_assessment(request, database_path)

    initialize_database(database_path)
    with get_connection(database_path) as connection:
        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM assessments"
        ).fetchone()[0]
        interaction_count = connection.execute(
            "SELECT COUNT(*) FROM assessment_interactions"
        ).fetchone()[0]

    assert assessment_count == 0
    assert interaction_count == 0


def test_transaction_rolls_back_if_interaction_insert_fails(tmp_path):
    database_path = tmp_path / "memory.db"
    request = make_request()
    broken_question = AssessmentQuestionResult.model_construct(
        question_id="broken",
        question=None,
        student_answer=None,
        expected_answer=None,
        is_correct=True,
        identified_error=None,
        attempt_count=-10,
        hint_count=None,
        hint_total=None,
        response_time_ms=None,
    )
    request.assessment_questions.append(broken_question)

    with pytest.raises(AssessmentRepositoryError):
        store_assessment(request, database_path)

    with get_connection(database_path) as connection:
        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM assessments"
        ).fetchone()[0]
        interaction_count = connection.execute(
            "SELECT COUNT(*) FROM assessment_interactions"
        ).fetchone()[0]

    assert assessment_count == 0
    assert interaction_count == 0


def test_multiple_assessments_for_same_student_are_allowed(tmp_path):
    database_path = tmp_path / "memory.db"
    first = store_assessment(make_request(), database_path)
    second = store_assessment(make_request(), database_path)

    assert second.assessment_id != first.assessment_id

    with get_connection(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM assessments WHERE student_id = ?", ("s1",)
        ).fetchone()[0]

    assert count == 2


def test_question_result_schema_rejects_negative_behaviour():
    with pytest.raises(Exception):
        AssessmentQuestionResult(
            question_id="q1",
            is_correct=True,
            attempt_count=-1,
        )


def test_request_requires_at_least_one_question():
    with pytest.raises(Exception):
        AssessmentMemoryUpdateRequest(
            student_id="s1",
            topic="Algebra",
            assessment_questions=[],
        )
