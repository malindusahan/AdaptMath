# Client Integration Guide

Welcome to the **Student Personalization Memory Service**. This guide explains how external components (**Tutor**, **Evaluator**, **Planner**, and **Router**) interact with this memory subsystem.

---

## 1. System Overview

```
                          ┌────────────────────────┐
                          │  Student Math Prompt   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                      POST /topic/classify (Topic API)
                                      │
                       Returns 1 of 111 Canonical Topics
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  Tutor Agent │              │ Planner Agt  │              │ Evaluator    │
│  (Generation)│              │ (Curriculum) │              │ (Scoring)    │
└───────┬──────┘              └──────┬───────┘              └──────┬───────┘
        │                             │                            │
        │ GET /memory/{id}/           │ GET /memory/{id}/          │ POST /memory/
        │ tutor-context               │ planner-context            │ update
        ▼                             ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             STUDENT PERSONALIZATION MEMORY SUBSYSTEM                    │
│   • Multi-tier Projections (STM, LTM, Concept Memory Stability)          │
│   • Neural Knowledge-Tracing Dynamic Learning State Engine               │
│   • Active Misconceptions Tracker & Repair Outcome History              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Integration Rules

1. **Single-Level Topic Contract**:
   There is **only one classification level: `topic`**. We have eliminated `subtopic` from all integration payloads. The topic returned and received maps strictly to the **111 Canonical Topics** defined in [TOPIC_ONTOLOGY.md](./TOPIC_ONTOLOGY.md).
2. **Transparent Student Identifiers**:
   Your components can pass any student identifier string (e.g., `"s1"`, `"999001"`, `"STU-12345"`). The Memory Subsystem resolves internal database records and provisions new students automatically.
3. **Machine-to-Machine Authentication**:
   Include the `X-Service-Key` header with all HTTP requests.

---

## 3. Step-by-Step Flow

### Step A: Classifying a Student Question
When a student inputs a question:
```http
POST /topic/classify HTTP/1.1
Host: localhost:8000
Content-Type: application/json
X-Service-Key: development-secret-key

{
  "question": "Solve 3x + 5 = 20."
}
```
**Response**:
```json
{
  "topic": "Linear Equations",
  "skill_id": "SKILL_193",
  "confidence": 0.9412,
  "is_math": true
}
```

### Step B: Fetching Personalization Context for the Tutor
Before generating a response or instructional step:
```http
GET /memory/s1/tutor-context HTTP/1.1
Host: localhost:8000
X-Service-Key: development-secret-key
```
**Response provides**:
- Student's current dynamic learning state (`NEEDS_SUPPORT`, `DEVELOPING`, `STRONG`)
- Active misconceptions (e.g. `"addition sign error"`)
- Average response time & hint usage patterns

### Step C: Persisting Evaluator Assessment Results
When a student completes an assessment or practice exercise:
```http
POST /memory/update HTTP/1.1
Host: localhost:8000
Content-Type: application/json
X-Service-Key: development-secret-key

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
      "identified_error": null
    }
  ],
  "identified_errors": [],
  "overall_feedback": "Perfect solving."
}
```
**Response**:
```json
{
  "student_id": "s1",
  "topic": "Linear Equations",
  "learning_state": "STRONG",
  "evidence_level": "FULL_SKILL",
  "evidence_strength": "MEDIUM",
  "misconceptions": [],
  "memory_updated": true
}
```

---

## 4. Quickstart SDKs

Ready-to-use client libraries are included in `examples/`:
- **Python**: [`examples/python_client.py`](../examples/python_client.py)
- **JavaScript**: [`examples/javascript_client.js`](../examples/javascript_client.js)
- **cURL**: [`examples/curl_examples.md`](../examples/curl_examples.md)
