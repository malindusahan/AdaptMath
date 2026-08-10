import sqlite3

from src.database.connection import get_connection, initialize_database


EXPECTED_TABLES = {
    "assessments",
    "assessment_interactions",
    "student_misconceptions",
    "learning_state_snapshots",
    "student_memory",
}


def get_table_names(connection):
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()

    return {row["name"] for row in rows if not row["name"].startswith("sqlite_")}


def test_database_initialization_creates_database(tmp_path):
    database_path = tmp_path / "test_memory.db"

    result = initialize_database(database_path)

    assert result == database_path
    assert database_path.exists()


def test_all_expected_tables_created(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        tables = get_table_names(connection)

    assert EXPECTED_TABLES.issubset(tables)


def test_database_initialization_is_repeatable(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        tables = get_table_names(connection)

    assert EXPECTED_TABLES.issubset(tables)


def test_foreign_keys_are_enabled(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        value = connection.execute("PRAGMA foreign_keys;").fetchone()[0]

    assert value == 1


def test_interaction_behaviour_defaults_do_not_fabricate_measurements(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO assessments (student_id, topic, subtopic)
            VALUES (?, ?, ?)
            """,
            ("s1", "Algebra", "Linear equations"),
        )
        assessment_id = cursor.lastrowid

        connection.execute(
            """
            INSERT INTO assessment_interactions (
                assessment_id,
                student_id,
                question_id,
                topic,
                subtopic,
                question_text,
                student_answer,
                expected_answer,
                is_correct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                "s1",
                "q1",
                "Algebra",
                "Linear equations",
                "What is x if 3x = 15?",
                "5",
                "5",
                1,
            ),
        )

        row = connection.execute(
            """
            SELECT
                attempt_count,
                hint_count,
                hint_total,
                response_time_ms,
                attempt_data_available,
                hint_data_available,
                response_time_available
            FROM assessment_interactions
            WHERE assessment_id = ?
            """,
            (assessment_id,),
        ).fetchone()

    assert row["attempt_count"] is None
    assert row["hint_count"] is None
    assert row["hint_total"] is None
    assert row["response_time_ms"] is None
    assert row["attempt_data_available"] == 0
    assert row["hint_data_available"] == 0
    assert row["response_time_available"] == 0


def test_invalid_learning_state_rejected(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO learning_state_snapshots (
                    student_id,
                    topic,
                    learning_state,
                    evidence_level,
                    evidence_strength,
                    model_used
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("s1", "Algebra", "INVALID_STATE", "COLD_START", "NONE", 0),
            )
            connection.commit()
            raised = False
        except sqlite3.IntegrityError:
            raised = True

    assert raised is True


def test_assessment_delete_cascades_interactions(tmp_path):
    database_path = tmp_path / "test_memory.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO assessments (student_id, topic)
            VALUES (?, ?)
            """,
            ("s1", "Algebra"),
        )
        assessment_id = cursor.lastrowid

        connection.execute(
            """
            INSERT INTO assessment_interactions (
                assessment_id,
                student_id,
                question_id,
                topic,
                is_correct
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (assessment_id, "s1", "q1", "Algebra", 1),
        )
        connection.commit()

        connection.execute(
            """
            DELETE FROM assessments
            WHERE assessment_id = ?
            """,
            (assessment_id,),
        )
        connection.commit()

        remaining = connection.execute(
            """
            SELECT COUNT(*)
            FROM assessment_interactions
            WHERE assessment_id = ?
            """,
            (assessment_id,),
        ).fetchone()[0]

    assert remaining == 0
