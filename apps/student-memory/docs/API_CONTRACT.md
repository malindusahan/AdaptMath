# Student Personalization Memory - API Contract Specification

**Version**: `1.0.0` (Production Frozen)  
**Protocol**: HTTP/1.1 REST + JSON  
**Authentication**: `X-Service-Key` header (Machine-to-Machine) or Bearer JWT (Student Web UI)
**Standard Error Format**: RFC-7807 structured JSON

---

## 1. Authentication & Headers

All integration requests between backend components (Tutor, Evaluator, Planner, Orchestrator) and this Memory Service must supply:

```http
Content-Type: application/json
X-Service-Key: <configured-service-key>
```

---

## 2. Topic Classification Endpoint

### `POST /topic/classify`
Classifies student input text into one of the **111 Canonical Topics** using the fine-tuned Sentence Transformer neural model.

#### Request Body
```json
{
  "question": "Solve 3x + 5 = 20."
}
```

#### Math Question Response (`200 OK`)
```json
{
  "topic": "Linear Equations",
  "skill_id": "SKILL_193",
  "confidence": 0.9412,
  "is_math": true,
  "model_version": "phase16-minilm-ft-v2"
}
```

#### Non-Math Greeting Response (`200 OK`)
```json
{
  "topic": null,
  "skill_id": null,
  "confidence": 0.1145,
  "is_math": false,
  "model_version": "phase16-minilm-ft-v2"
}
```

---

## 3. Memory Write Endpoints

### `POST /memory/update`
Invoked by the **Evaluator** after a practice or assessment activity completes. Updates raw interaction logs, concept memory stability/retrieval strength, misconceptions, and dynamic learning state.

#### Request Body
```json
{
  "student_id": "s1",
  "topic": "Linear Equations",
  "assessment_questions": [
    {
      "question_id": "q1",
      "question": "What is x if 3x = 15?",
      "student_answer": "5",
      "expected_answer": "5",
      "is_correct": true,
      "identified_error": null,
      "attempt_count": 1,
      "hint_count": 0,
      "hint_total": 2,
      "response_time_ms": 3500.0
    },
    {
      "question_id": "q2",
      "question": "Solve x - 2 = 7.",
      "student_answer": "x = 8",
      "expected_answer": "x = 9",
      "is_correct": false,
      "identified_error": "addition sign error",
      "attempt_count": 2,
      "hint_count": 1,
      "hint_total": 2,
      "response_time_ms": 6200.0
    }
  ],
  "identified_errors": ["addition sign error"],
  "overall_feedback": "Student solved division easily but made a sign error during addition."
}
```

#### Response Body (`200 OK`)
```json
{
  "student_id": "s1",
  "topic": "Linear Equations",
  "subtopic": null,
  "assessment_id": 42,
  "snapshot_id": 108,
  "learning_state": "DEVELOPING",
  "evidence_level": "FULL_SKILL",
  "evidence_strength": "MEDIUM",
  "behavioural_coverage": "FULL_BEHAVIOURAL_COVERAGE",
  "model_used": true,
  "previous_interaction_count": 8,
  "previous_skill_interaction_count": 4,
  "recent_interaction_count": 2,
  "attempt_observation_count": 2,
  "hint_observation_count": 2,
  "response_time_observation_count": 2,
  "misconception_count": 1,
  "misconceptions": [
    "addition sign error"
  ],
  "memory_updated": true
}
```

---

### `POST /memory/repair-outcome`
Records pedagogical repair interventions and their effectiveness.

#### Request Body
```json
{
  "student_id": "s1",
  "topic": "Linear Equations",
  "repair_action": "HINT_SIMPLIFICATION",
  "outcome": "RESOLVED",
  "score": 0.85,
  "notes": "Student corrected misconception after step-by-step subtraction hint."
}
```

#### Response Body (`200 OK`)
```json
{
  "repair_outcome_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "student_id": "s1",
  "topic": "Linear Equations",
  "repair_action": "HINT_SIMPLIFICATION",
  "outcome": "RESOLVED",
  "score": 0.85,
  "created_at": "2026-08-19T09:30:00Z"
}
```

---

## 4. Memory Read Context Endpoints

### `GET /memory/{student_id}/tutor-context`
Retrieves compact, high-signal personalization context tailored for the Tutor component.

#### Response Body (`200 OK`)
```json
{
  "student_id": "s1",
  "session_id": "session_5001",
  "skill_id": "SKILL_193",
  "current_learning_state": "DEVELOPING",
  "evidence_strength": "MEDIUM",
  "behavioural_coverage": "FULL_BEHAVIOURAL_COVERAGE",
  "recent_accuracy": 0.50,
  "recent_correct_count": 1,
  "recent_incorrect_count": 1,
  "attempt_evidence": {
    "observation_count": 2,
    "total_value": 3.0,
    "average_value": 1.5
  },
  "hint_evidence": {
    "observation_count": 2,
    "total_value": 1.0,
    "average_value": 0.5
  },
  "response_time_evidence": {
    "observation_count": 2,
    "total_value": 9700.0,
    "average_value": 4850.0
  },
  "misconceptions": [
    {
      "misconception_id": "m_1",
      "normalized_error": "addition sign error",
      "display_error": "addition sign error",
      "occurrence_count": 2,
      "last_seen_at": "2026-08-19T09:20:00Z"
    }
  ],
  "recent_interactions": [],
  "recent_repairs": []
}
```

### `GET /memory/{student_id}/planner-context`
Retrieves curriculum-planning personalization context tailored for the Planner component.

---

## 5. System Health Endpoints

### `GET /health`
Liveness probe.
```json
{
  "status": "healthy",
  "service": "student_personalization_memory",
  "version": "1.0.0"
}
```

### `GET /ready`
Readiness probe checking database connectivity and topic model status.
```json
{
  "ready": true,
  "database": "connected",
  "model": "loaded"
}
```
