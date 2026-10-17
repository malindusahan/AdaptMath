# Tutor component

The Tutor is the main FastAPI orchestration service. It owns the conversation
workflow, routing, planning, tutoring, evaluation, assessment lifecycle, and
durable checkpoint integration.

## Entry points

- Application: `app.local_runtime:app`
- Health: `GET /health`
- Readiness: `GET /ready`
- Tutor routes: `app/api/tutor.py`
- Workflow: `app/graph/workflow.py`
- Adaptive integration: `app/integrations/adaptive_component_coordinator.py`

## Component dependencies

The Tutor imports the scientific components from:

- `packages/student-modeling`
- `packages/move-selection`

It calls `apps/student-memory` over HTTP and persists Tutor, student-model, and
research state in their dedicated PostgreSQL schemas.

## Local verification

From the repository root:

```powershell
.\apps\tutor\.venv-integration\Scripts\python.exe -m pytest apps\tutor\tests
```

Use the repository launcher for a complete interactive session.
