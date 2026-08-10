from src.database.assessment_repository import store_assessment
from src.database.history_repository import (
    get_assessment_interactions,
    get_student_history,
    get_student_topic_history,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
)


def make_assessment(student_id, topic, subtopic, question_prefix, results):
    questions = [
        AssessmentQuestionResult(
            question_id=f"{question_prefix}{index}",
            is_correct=is_correct,
        )
        for index, is_correct in enumerate(results, start=1)
    ]
    return AssessmentMemoryUpdateRequest(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        assessment_questions=questions,
    )


def seed_history(database_path):
    first = store_assessment(
        make_assessment(
            "s1", "Algebra", "Linear equations", "a", [True, False]
        ),
        database_path,
    )
    second = store_assessment(
        make_assessment("s1", "Geometry", "Angles", "b", [True]),
        database_path,
    )
    third = store_assessment(
        make_assessment(
            "s1", "Algebra", "Linear equations", "c", [False, True, True]
        ),
        database_path,
    )
    return first, second, third


def test_student_history_returns_all_interactions(tmp_path):
    database_path = tmp_path / "memory.db"
    seed_history(database_path)
    history = get_student_history("s1", database_path)
    assert len(history) == 6
    assert [item.question_id for item in history] == [
        "a1", "a2", "b1", "c1", "c2", "c3"
    ]


def test_student_history_is_chronological(tmp_path):
    database_path = tmp_path / "memory.db"
    seed_history(database_path)
    history = get_student_history("s1", database_path)
    keys = [(item.assessment_id, item.interaction_id) for item in history]
    assert keys == sorted(keys)


def test_topic_history_filters_learning_area(tmp_path):
    database_path = tmp_path / "memory.db"
    seed_history(database_path)
    history = get_student_topic_history(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        database_path=database_path,
    )
    assert len(history) == 5
    assert all(item.topic == "Algebra" for item in history)
    assert all(item.subtopic == "Linear equations" for item in history)
    assert [item.question_id for item in history] == [
        "a1", "a2", "c1", "c2", "c3"
    ]


def test_before_assessment_cutoff_excludes_current_and_later(tmp_path):
    database_path = tmp_path / "memory.db"
    _, _, third = seed_history(database_path)
    history = get_student_history(
        student_id="s1",
        database_path=database_path,
        before_assessment_id=third.assessment_id,
    )
    assert [item.question_id for item in history] == ["a1", "a2", "b1"]
    assert all(item.assessment_id < third.assessment_id for item in history)


def test_topic_history_cutoff_is_leakage_safe(tmp_path):
    database_path = tmp_path / "memory.db"
    _, _, third = seed_history(database_path)
    history = get_student_topic_history(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        database_path=database_path,
        before_assessment_id=third.assessment_id,
    )
    assert [item.question_id for item in history] == ["a1", "a2"]


def test_assessment_interactions_returns_only_requested_assessment(tmp_path):
    database_path = tmp_path / "memory.db"
    _, _, third = seed_history(database_path)
    interactions = get_assessment_interactions(third.assessment_id, database_path)
    assert len(interactions) == 3
    assert [item.question_id for item in interactions] == ["c1", "c2", "c3"]
    assert all(item.assessment_id == third.assessment_id for item in interactions)


def test_student_isolation(tmp_path):
    database_path = tmp_path / "memory.db"
    seed_history(database_path)
    store_assessment(
        make_assessment(
            "s2", "Algebra", "Linear equations", "other", [True, True]
        ),
        database_path,
    )
    s1_history = get_student_history("s1", database_path)
    s2_history = get_student_history("s2", database_path)
    assert len(s1_history) == 6
    assert len(s2_history) == 2
    assert all(item.student_id == "s1" for item in s1_history)
    assert all(item.student_id == "s2" for item in s2_history)


def test_null_subtopic_history_supported(tmp_path):
    database_path = tmp_path / "memory.db"
    store_assessment(
        make_assessment("s1", "Algebra", None, "n", [True, False]),
        database_path,
    )
    history = get_student_topic_history(
        student_id="s1",
        topic="Algebra",
        subtopic=None,
        database_path=database_path,
    )
    assert len(history) == 2
    assert all(item.subtopic is None for item in history)


def test_unknown_student_returns_empty_history(tmp_path):
    database_path = tmp_path / "memory.db"
    seed_history(database_path)
    assert get_student_history("does-not-exist", database_path) == []


def test_behavioural_nulls_remain_none(tmp_path):
    database_path = tmp_path / "memory.db"
    result = store_assessment(
        make_assessment(
            "s1", "Algebra", "Linear equations", "q", [True]
        ),
        database_path,
    )
    interaction = get_assessment_interactions(
        result.assessment_id, database_path
    )[0]
    assert interaction.attempt_count is None
    assert interaction.hint_count is None
    assert interaction.hint_total is None
    assert interaction.response_time_ms is None
    assert interaction.attempt_data_available is False
    assert interaction.hint_data_available is False
    assert interaction.response_time_available is False


def test_supplied_behaviour_is_preserved(tmp_path):
    database_path = tmp_path / "memory.db"
    request = AssessmentMemoryUpdateRequest(
        student_id="s1",
        topic="Algebra",
        subtopic="Linear equations",
        assessment_questions=[
            AssessmentQuestionResult(
                question_id="q1",
                is_correct=True,
                attempt_count=2,
                hint_count=1,
                hint_total=4,
                response_time_ms=15000,
            )
        ],
    )
    result = store_assessment(request, database_path)
    interaction = get_assessment_interactions(
        result.assessment_id, database_path
    )[0]
    assert interaction.attempt_count == 2
    assert interaction.hint_count == 1
    assert interaction.hint_total == 4
    assert interaction.response_time_ms == 15000
    assert interaction.attempt_data_available is True
    assert interaction.hint_data_available is True
    assert interaction.response_time_available is True
