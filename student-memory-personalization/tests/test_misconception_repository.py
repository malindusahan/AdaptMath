import pytest

from src.database.connection import get_connection
from src.database.misconception_repository import (
    MisconceptionRepositoryError,
    collect_unique_misconceptions,
    get_student_misconceptions,
    normalize_misconception,
    update_misconception_memory,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)


def make_request():
    return AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(question_id="q1", is_correct=True),
            AssessmentQuestionResult(
                question_id="q2",
                is_correct=False,
                identified_error="Addition Error",
            ),
            AssessmentQuestionResult(
                question_id="q3",
                is_correct=False,
                identified_error="sign error",
            ),
        ],
        identified_errors=[" addition   error ", "Sign Error"],
    )


def test_normalize_misconception():
    assert normalize_misconception("  Addition   Error ") == "addition error"


def test_empty_misconception_rejected():
    with pytest.raises(MisconceptionRepositoryError):
        normalize_misconception("   ")


def test_collect_deduplicates_top_and_question_errors():
    assert collect_unique_misconceptions(make_request()) == [
        "addition error",
        "sign error",
    ]


def test_first_assessment_creates_misconceptions(tmp_path):
    result = update_misconception_memory(make_request(), tmp_path / "memory.db")
    assert len(result) == 2
    by_name = {item.misconception: item for item in result}
    assert by_name["addition error"].occurrence_count == 1
    assert by_name["sign error"].occurrence_count == 1


def test_repeated_assessment_increments_once_per_error(tmp_path):
    database_path = tmp_path / "memory.db"
    request = make_request()
    update_misconception_memory(request, database_path)
    result = update_misconception_memory(request, database_path)
    by_name = {item.misconception: item for item in result}
    assert by_name["addition error"].occurrence_count == 2
    assert by_name["sign error"].occurrence_count == 2


def test_different_misconceptions_have_independent_counts(tmp_path):
    database_path = tmp_path / "memory.db"
    update_misconception_memory(make_request(), database_path)
    second_request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q10",
                is_correct=False,
                identified_error="addition error",
            )
        ],
    )
    result = update_misconception_memory(second_request, database_path)
    by_name = {item.misconception: item for item in result}
    assert by_name["addition error"].occurrence_count == 2
    assert by_name["sign error"].occurrence_count == 1


def test_memory_is_separated_by_student(tmp_path):
    database_path = tmp_path / "memory.db"
    first = make_request()
    update_misconception_memory(first, database_path)
    second = make_request()
    second.student_id = "s2"
    update_misconception_memory(second, database_path)

    s1 = get_student_misconceptions(
        "s1", "Algebra", "Linear equations", database_path
    )
    s2 = get_student_misconceptions(
        "s2", "Algebra", "Linear equations", database_path
    )
    assert len(s1) == 2
    assert len(s2) == 2
    assert all(item.student_id == "s1" for item in s1)
    assert all(item.student_id == "s2" for item in s2)


def test_memory_is_separated_by_topic(tmp_path):
    database_path = tmp_path / "memory.db"
    update_misconception_memory(make_request(), database_path)
    second = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Geometry",
        subtopic="Angles",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q20",
                is_correct=False,
                identified_error="addition error",
            )
        ],
    )
    update_misconception_memory(second, database_path)

    algebra = get_student_misconceptions(
        "s1", "Algebra", "Linear equations", database_path
    )
    geometry = get_student_misconceptions(
        "s1", "Geometry", "Angles", database_path
    )
    assert len(algebra) == 2
    assert len(geometry) == 1
    assert geometry[0].occurrence_count == 1


def test_null_subtopic_supported(tmp_path):
    database_path = tmp_path / "memory.db"
    request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic=None,
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q1",
                is_correct=False,
                identified_error="sign error",
            )
        ],
    )
    update_misconception_memory(request, database_path)
    result = get_student_misconceptions("s1", "Algebra", None, database_path)
    assert len(result) == 1
    assert result[0].misconception == "sign error"


def test_no_errors_creates_no_memory(tmp_path):
    database_path = tmp_path / "memory.db"
    request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        assessment_questions=[
            AssessmentQuestionResult(question_id="q1", is_correct=True)
        ],
    )
    assert update_misconception_memory(request, database_path) == []
    assert get_student_misconceptions(
        "s1", "Algebra", None, database_path
    ) == []


def test_database_contains_only_unique_rows(tmp_path):
    database_path = tmp_path / "memory.db"
    request = make_request()
    update_misconception_memory(request, database_path)
    update_misconception_memory(request, database_path)

    with get_connection(database_path) as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM student_misconceptions
            WHERE student_id = ? AND topic = ? AND subtopic IS ?
            """,
            ("s1", "Algebra", "Linear equations"),
        ).fetchone()[0]

    assert count == 2
