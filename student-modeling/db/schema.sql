-- Meta-Agent SQLite schema
-- Used during development. In production, this storage layer will be
-- backed by the Memory component's persistence layer.

-- Students table
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    profile_json TEXT
);

-- Per-(student, skill) mastery state — the knowledge graph
CREATE TABLE IF NOT EXISTS mastery (
    student_id TEXT,
    skill_name TEXT,
    mastery_probability REAL NOT NULL,
    mastery_label TEXT NOT NULL,
    previous_mastery_probability REAL,  -- NEW
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, skill_name),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Immutable BKT initialization for each (student, skill).
--
-- This is intentionally separate from `mastery`: an adaptive attempt may
-- need to snapshot the exact cold-start prior before the first observation,
-- and doing so must not create or change a mastery row.  Existing databases
-- migrate safely through CREATE TABLE IF NOT EXISTS. Historical mastery and
-- attempts-only pairs are marked `legacy_unknown`; their original transferred
-- prior is never guessed during migration.
CREATE TABLE IF NOT EXISTS bkt_initial_priors (
    student_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    effective_initial_prior REAL,
    prior_source TEXT NOT NULL CHECK (
        prior_source IN (
            'population',
            'transfer',
            'historical_population_rebase',
            'legacy_unknown'
        )
    ),
    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, skill_name),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    CHECK (
        (
            prior_source = 'legacy_unknown'
            AND effective_initial_prior IS NULL
        )
        OR
        (
            prior_source != 'legacy_unknown'
            AND effective_initial_prior IS NOT NULL
            AND effective_initial_prior >= 0.0
            AND effective_initial_prior <= 1.0
        )
    )
);

-- Every attempt a student has made
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    confidence REAL NOT NULL DEFAULT 1.0,
    signal_type TEXT,
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Session audit log
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concept_count INTEGER,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_attempts_student_skill 
    ON attempts(student_id, skill_name, created_at);
CREATE INDEX IF NOT EXISTS idx_mastery_student 
    ON mastery(student_id);
