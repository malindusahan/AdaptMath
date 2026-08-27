from src.database.state_repository import (
    get_current_student_memory,
    get_learning_state_history,
)
from src.schemas.interaction import (
    AssessmentMemoryUpdateRequest,
    AssessmentQuestionResult,
    CurrentStudentMemory,
    LearningStateHistoryItem,
)
from src.services.memory_update_service import MemoryUpdateService


def make_request(
    *, student_id="s1", topic="Algebra", subtopic="Linear equations",
    question_id="q1", is_correct=True
):
    return AssessmentMemoryUpdateRequest(
        student_id=student_id,
        topic=topic,
        subtopic=subtopic,
        assessment_questions=[
            AssessmentQuestionResult(question_id=question_id, is_correct=is_correct)
        ],
    )


def test_unknown_current_memory_returns_none(tmp_path):
    assert get_current_student_memory(
        "unknown", "Algebra", "Linear equations", tmp_path / "memory.db"
    ) is None


def test_unknown_history_returns_empty_list(tmp_path):
    assert get_learning_state_history(
        "unknown", "Algebra", "Linear equations", tmp_path / "memory.db"
    ) == []


def test_current_memory_matches_latest_update(tmp_path):
    database_path = tmp_path / "memory.db"
    update = MemoryUpdateService().update_memory(make_request(), database_path)
    current = get_current_student_memory(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert isinstance(current, CurrentStudentMemory)
    assert current.learning_state == update.learning_state
    assert current.evidence_level == update.evidence_level
    assert current.evidence_strength == update.evidence_strength
    assert current.behavioural_coverage == update.behavioural_coverage
    assert current.last_assessment_id == update.assessment_id
    assert current.last_snapshot_id == update.snapshot_id


def test_history_contains_snapshot(tmp_path):
    database_path = tmp_path / "memory.db"
    update = MemoryUpdateService().update_memory(make_request(), database_path)
    history = get_learning_state_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert len(history) == 1
    assert isinstance(history[0], LearningStateHistoryItem)
    assert history[0].snapshot_id == update.snapshot_id
    assert history[0].assessment_id == update.assessment_id
    assert history[0].learning_state == update.learning_state


def test_multiple_snapshots_return_chronologically(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService()
    first = service.update_memory(
        make_request(question_id="q1", is_correct=False), database_path
    )
    second = service.update_memory(
        make_request(question_id="q2", is_correct=True), database_path
    )
    third = service.update_memory(
        make_request(question_id="q3", is_correct=True), database_path
    )
    history = get_learning_state_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert [item.snapshot_id for item in history] == [
        first.snapshot_id, second.snapshot_id, third.snapshot_id
    ]
    assert [item.assessment_id for item in history] == [
        first.assessment_id, second.assessment_id, third.assessment_id
    ]


def test_current_memory_points_to_latest_snapshot(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService()
    service.update_memory(make_request(question_id="q1", is_correct=False), database_path)
    latest = service.update_memory(
        make_request(question_id="q2", is_correct=True), database_path
    )
    current = get_current_student_memory(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert current.last_snapshot_id == latest.snapshot_id
    assert current.last_assessment_id == latest.assessment_id
    assert current.learning_state == latest.learning_state


def test_student_isolation(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService()
    service.update_memory(make_request(student_id="s1", question_id="s1-q1"), database_path)
    service.update_memory(make_request(student_id="s2", question_id="s2-q1"), database_path)
    s1 = get_learning_state_history("s1", "Algebra", "Linear equations", database_path)
    s2 = get_learning_state_history("s2", "Algebra", "Linear equations", database_path)
    assert len(s1) == len(s2) == 1
    assert s1[0].student_id == "s1"
    assert s2[0].student_id == "s2"


def test_topic_isolation(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService()
    service.update_memory(
        make_request(topic="Algebra", subtopic="Linear equations", question_id="alg-q1"),
        database_path,
    )
    service.update_memory(
        make_request(topic="Geometry", subtopic="Angles", question_id="geo-q1"),
        database_path,
    )
    algebra = get_learning_state_history("s1", "Algebra", "Linear equations", database_path)
    geometry = get_learning_state_history("s1", "Geometry", "Angles", database_path)
    assert len(algebra) == len(geometry) == 1
    assert algebra[0].topic == "Algebra"
    assert geometry[0].topic == "Geometry"


def test_subtopic_isolation(tmp_path):
    database_path = tmp_path / "memory.db"
    service = MemoryUpdateService()
    service.update_memory(
        make_request(subtopic="Linear equations", question_id="linear-q1"), database_path
    )
    service.update_memory(
        make_request(subtopic="Quadratic equations", question_id="quad-q1"), database_path
    )
    linear = get_learning_state_history("s1", "Algebra", "Linear equations", database_path)
    quadratic = get_learning_state_history("s1", "Algebra", "Quadratic equations", database_path)
    assert len(linear) == len(quadratic) == 1
    assert linear[0].subtopic == "Linear equations"
    assert quadratic[0].subtopic == "Quadratic equations"


def test_null_subtopic_round_trip(tmp_path):
    database_path = tmp_path / "memory.db"
    update = MemoryUpdateService().update_memory(make_request(subtopic=None), database_path)
    current = get_current_student_memory("s1", "Algebra", None, database_path)
    history = get_learning_state_history("s1", "Algebra", None, database_path)
    assert current is not None
    assert current.subtopic is None
    assert len(history) == 1
    assert history[0].subtopic is None
    assert current.last_snapshot_id == update.snapshot_id


def test_retrieval_preserves_coverage_counts(tmp_path):
    database_path = tmp_path / "memory.db"
    update = MemoryUpdateService().update_memory(make_request(), database_path)
    current = get_current_student_memory(
        "s1", "Algebra", "Linear equations", database_path
    )
    history = get_learning_state_history(
        "s1", "Algebra", "Linear equations", database_path
    )
    assert current.recent_interaction_count == update.recent_interaction_count
    assert current.attempt_observation_count == update.attempt_observation_count
    assert current.hint_observation_count == update.hint_observation_count
    assert current.response_time_observation_count == update.response_time_observation_count
    assert history[0].behavioural_coverage == update.behavioural_coverage


def test_retrieval_models_serialize_cleanly(tmp_path):
    database_path = tmp_path / "memory.db"
    MemoryUpdateService().update_memory(make_request(), database_path)
    current = get_current_student_memory("s1", "Algebra", "Linear equations", database_path)
    history = get_learning_state_history("s1", "Algebra", "Linear equations", database_path)
    current_payload = current.model_dump()
    history_payload = [item.model_dump() for item in history]
    assert current_payload["student_id"] == "s1"
    assert len(history_payload) == 1
    assert history_payload[0]["snapshot_id"] == current.last_snapshot_id
