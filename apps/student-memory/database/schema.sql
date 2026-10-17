-- Student Personalization Memory SQLite Schema
-- Synchronized from src/database/connection.py SCHEMA_SQL
-- Runtime database connections enable PRAGMA foreign_keys = ON.

CREATE TABLE IF NOT EXISTS assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    subtopic TEXT,
    overall_feedback TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assessment_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    student_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    subtopic TEXT,
    question_text TEXT,
    student_answer TEXT,
    expected_answer TEXT,
    is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    identified_error TEXT,

    /* Optional measurements remain NULL when unavailable. */
    attempt_count INTEGER CHECK (attempt_count IS NULL OR attempt_count >= 0),
    hint_count INTEGER CHECK (hint_count IS NULL OR hint_count >= 0),
    hint_total INTEGER CHECK (hint_total IS NULL OR hint_total >= 0),
    response_time_ms REAL CHECK (
        response_time_ms IS NULL OR response_time_ms >= 0
    ),

    /* Explicit measurement-availability indicators. */
    attempt_data_available INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_data_available IN (0, 1)),
    hint_data_available INTEGER NOT NULL DEFAULT 0
        CHECK (hint_data_available IN (0, 1)),
    response_time_available INTEGER NOT NULL DEFAULT 0
        CHECK (response_time_available IN (0, 1)),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (assessment_id)
        REFERENCES assessments (assessment_id)
        ON DELETE CASCADE,
    UNIQUE (assessment_id, question_id)
);

CREATE TABLE IF NOT EXISTS student_misconceptions (
    misconception_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    subtopic TEXT,
    misconception TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1
        CHECK (occurrence_count >= 1),
    first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (student_id, topic, subtopic, misconception)
);

CREATE TABLE IF NOT EXISTS learning_state_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    subtopic TEXT,
    assessment_id INTEGER,
    learning_state TEXT NOT NULL CHECK (
        learning_state IN (
            'NEEDS_SUPPORT', 'DEVELOPING', 'STRONG', 'UNAVAILABLE'
        )
    ),
    evidence_level TEXT NOT NULL CHECK (
        evidence_level IN (
            'COLD_START', 'OVERALL_ONLY', 'PARTIAL_SKILL', 'FULL_SKILL'
        )
    ),
    evidence_strength TEXT NOT NULL CHECK (
        evidence_strength IN ('NONE', 'LOW', 'MEDIUM', 'HIGH')
    ),
    model_used INTEGER NOT NULL CHECK (model_used IN (0, 1)),
    previous_interaction_count INTEGER NOT NULL DEFAULT 0
        CHECK (previous_interaction_count >= 0),
    previous_skill_interaction_count INTEGER NOT NULL DEFAULT 0
        CHECK (previous_skill_interaction_count >= 0),
    behavioural_coverage TEXT NOT NULL CHECK (
        behavioural_coverage IN (
            'FULL_BEHAVIOURAL_COVERAGE',
            'PARTIAL_BEHAVIOURAL_COVERAGE',
            'CORRECTNESS_ONLY_COVERAGE'
        )
    ),
    recent_interaction_count INTEGER NOT NULL DEFAULT 0
        CHECK (recent_interaction_count >= 0),
    attempt_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_observation_count >= 0),
    hint_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (hint_observation_count >= 0),
    response_time_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (response_time_observation_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id)
        REFERENCES assessments (assessment_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS student_memory (
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    subtopic TEXT NOT NULL DEFAULT '',
    current_learning_state TEXT NOT NULL CHECK (
        current_learning_state IN (
            'NEEDS_SUPPORT', 'DEVELOPING', 'STRONG', 'UNAVAILABLE'
        )
    ),
    evidence_level TEXT NOT NULL CHECK (
        evidence_level IN (
            'COLD_START', 'OVERALL_ONLY', 'PARTIAL_SKILL', 'FULL_SKILL'
        )
    ),
    evidence_strength TEXT NOT NULL CHECK (
        evidence_strength IN ('NONE', 'LOW', 'MEDIUM', 'HIGH')
    ),
    model_used INTEGER NOT NULL CHECK (model_used IN (0, 1)),
    behavioural_coverage TEXT NOT NULL CHECK (
        behavioural_coverage IN (
            'FULL_BEHAVIOURAL_COVERAGE',
            'PARTIAL_BEHAVIOURAL_COVERAGE',
            'CORRECTNESS_ONLY_COVERAGE'
        )
    ),
    recent_interaction_count INTEGER NOT NULL DEFAULT 0
        CHECK (recent_interaction_count >= 0),
    attempt_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_observation_count >= 0),
    hint_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (hint_observation_count >= 0),
    response_time_observation_count INTEGER NOT NULL DEFAULT 0
        CHECK (response_time_observation_count >= 0),
    last_assessment_id INTEGER,
    last_snapshot_id INTEGER,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, topic, subtopic),
    FOREIGN KEY (last_assessment_id)
        REFERENCES assessments (assessment_id)
        ON DELETE SET NULL,
    FOREIGN KEY (last_snapshot_id)
        REFERENCES learning_state_snapshots (snapshot_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_assessments_student
ON assessments (student_id);

CREATE INDEX IF NOT EXISTS idx_assessments_student_topic
ON assessments (student_id, topic, subtopic);

CREATE INDEX IF NOT EXISTS idx_interactions_student
ON assessment_interactions (student_id);

CREATE INDEX IF NOT EXISTS idx_interactions_student_topic
ON assessment_interactions (student_id, topic, subtopic);

CREATE INDEX IF NOT EXISTS idx_interactions_assessment
ON assessment_interactions (assessment_id);

CREATE INDEX IF NOT EXISTS idx_snapshots_student
ON learning_state_snapshots (student_id);

CREATE INDEX IF NOT EXISTS idx_snapshots_student_topic
ON learning_state_snapshots (student_id, topic, subtopic);

CREATE INDEX IF NOT EXISTS idx_misconceptions_student_topic
ON student_misconceptions (student_id, topic, subtopic);

-- Runtime data policy:
-- student_memory.db contains mutable application data and is not required
-- to reproduce the application schema.
--
-- A fresh database is initialized by the application using SCHEMA_SQL in
-- src/database/connection.py.
--
-- The development database may contain controlled integration-test data.
-- Runtime/test records should not be treated as model-training data.
