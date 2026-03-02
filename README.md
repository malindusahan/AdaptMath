# AdaptMath

AdaptMath is a research-oriented adaptive mathematics tutoring system maintained
as a shared monorepo. The deployable applications are separated from the two
scientific Python packages so each research contribution can be inspected and
tested independently while the integrated Tutor can import them directly.

## Repository layout

```text
apps/
  web/                    React and Vite learner interface
  tutor/                  FastAPI adaptive Tutor service
  student-memory/         Memory, authentication, topic classification, Alembic
packages/
  student-modeling/       BKT, mastery, regressions, and learning paths
  move-selection/         MD7, Turn-LinTS, MRB1, and policy persistence
docs/
  components/             Panel-facing component documentation
  architecture/           Cross-component persistence and architecture records
research/                 Offline experiments, datasets, and reports
scripts/                  Cross-component migration and verification utilities
infrastructure/           Railway and local-container deployment definitions
tests/integration/        Whole-system verification entry points
```

## Component ownership

| Component | Source | Runtime role |
|---|---|---|
| Tutor | `apps/tutor` | Orchestrates tutoring conversations and assessments |
| Student Memory | `apps/student-memory` | Authentication, memory, ontology, and database migrations |
| Student Modeling | `packages/student-modeling` | BKT mastery and personalized learning-state calculations |
| Move Selection | `packages/move-selection` | Pedagogical policy, contextual bandit, and response-quality logic |

Detailed panel notes are in [`docs/components`](docs/components), with the
integrated walkthrough in [`docs/panel-demo.md`](docs/panel-demo.md).

## Local development

The existing Windows launcher remains the supported integrated development
entry point:

```powershell
.\start-adaptmath-local.ps1 -FinalSKLLive
```

Check or stop the stack with:

```powershell
.\status-adaptmath-local.ps1
.\stop-adaptmath-local.ps1
```

Local environments, secrets, databases, logs, backups, and generated runtime
state are ignored. Never commit real `.env` files or API keys.

## Deployment

Railway deployment will use four services: Web, Tutor API, Student Memory API,
and PostgreSQL. The Tutor image is built from the repository root because it
depends on both packages. Production deployment must not use the local
PowerShell launcher.
