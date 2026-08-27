# Safe local adaptive Tutor demo

This procedure uses synthetic identities and dedicated demo persistence. Never
point it at real policy, experience-log, checkpoint, or student-model state.
The old `backend/runtime/adaptmath_checkpoints.sqlite3` is not used.

## 1. Create and verify the backend environment

From `adaptive-math-tutor` in PowerShell:

```powershell
python -m venv .venv-integration
.\.venv-integration\Scripts\Activate.ps1
python -m pip install -r backend\requirements-lock.txt
python -m pip install -r backend\requirements-test.txt
cd backend
python scripts\prepare_local_demo.py
Copy-Item .env.integration-demo.example .env.integration-demo.local
```

Edit only `.env.integration-demo.local` and replace the placeholder
`GEMINI_API_KEY` is shared from `../student-modeling/.env`. Do not commit that
file or print the key. Verify the environment
without making a network model request:

```powershell
dotenv -f .env.integration-demo.local run -- python scripts\verify_environment.py
dotenv -f .env.integration-demo.local run -- python scripts\verify_local_demo.py
```

Safe example BKT skill identifiers include:

- `Equation Solving Two or Fewer Steps`
- `Percent Of`
- `Equivalent Fractions`
- `Pythagorean Theorem`

The start request must contain one exact `target_skill`. Topic and subtopic are
not skill aliases.

Set `MEMORY_API_URL` to the local Memory backend and keep
`MEMORY_SERVICE_API_KEY` identical in the two server-side environments. Never
place that key in frontend configuration. A Memory outage is fail-open for
tutoring; transient completed-attempt writes remain in the LangGraph checkpoint
queue with their original attempt IDs.

## 2. Start the backend and frontend

Backend, from `adaptive-math-tutor/backend`:

```powershell
dotenv -f .env.integration-demo.local run -- python -m uvicorn app.main:app --reload
```

Frontend, from `adaptive-math-tutor/frontend` in a second terminal:

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, enter a synthetic learner identifier, a problem,
and an exact valid target skill. Start the session, answer each Tutor turn, and
complete all three assessment questions. A failed assessment completes the
current adaptive attempt and starts a new indexed attempt before reteaching.

## 3. Demo storage and safety behavior

The preparation command creates only:

```text
backend/runtime/adaptive_demo/
    checkpoints.sqlite3
    policy_state.json          # official fresh synthetic LinTS state
    attempts.jsonl             # synthetic completed-attempt records
    student_model/data/meta_agent.db
```

Current research runtime supports one active adaptive attempt per backend
process. Distinct Tutor sessions are isolated, but adaptive episodes execute
sequentially. Starting a second active episode returns HTTP 409; retry after the
first completes. An active episode interrupted by a backend restart also
returns HTTP 409 and must be replaced by a new Tutor session.
