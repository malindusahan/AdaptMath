# cURL Integration Examples

All requests to the Student Personalization Memory service should include the `X-Service-Key` header (or student session JWT where applicable).

---

## 1. Classify Math Question (`POST /topic/classify`)

```bash
curl -X POST "http://localhost:8000/topic/classify" \
  -H "Content-Type: application/json" \
  -H "X-Service-Key: development-secret-key" \
  -d '{
    "question": "Solve 3x + 5 = 20."
  }'
```

### Math Response:
```json
{
  "topic": "Linear Equations",
  "skill_id": "SKILL_193",
  "confidence": 0.9412,
  "is_math": true,
  "model_version": "phase16-minilm-ft-v2"
}
```

### Non-Math Greeting Response:
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

## 2. Update Memory with Evaluator Assessment (`POST /memory/update`)

```bash
curl -X POST "http://localhost:8000/memory/update" \
  -H "Content-Type: application/json" \
  -H "X-Service-Key: development-secret-key" \
  -d '{
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
    "overall_feedback": "The student understands basic division in linear equations but made a sign error during addition."
  }'
```

---

## 3. Retrieve Tutor Personalization Context (`GET /memory/{student_id}/tutor-context`)

```bash
curl -X GET "http://localhost:8000/memory/s1/tutor-context" \
  -H "X-Service-Key: development-secret-key"
```

---

## 4. Retrieve Planner Personalization Context (`GET /memory/{student_id}/planner-context`)

```bash
curl -X GET "http://localhost:8000/memory/s1/planner-context" \
  -H "X-Service-Key: development-secret-key"
```

---

## 5. Check Health & Readiness

```bash
curl -X GET "http://localhost:8000/health"
curl -X GET "http://localhost:8000/ready"
```
