# Deployment & Operations Guide

This guide covers running the **Student Personalization Memory Service** in development, testing, and production containerized environments.

---

## 1. Environment Configuration

Copy the template `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `MEMORY_DATABASE_URL` | No production default | PostgreSQL URI using `postgresql+psycopg` |
| `MEMORY_DATABASE_SCHEMA` | `student_memory` | Target PostgreSQL schema |
| `MEMORY_AUTH_ENABLED` | `true` | Require authenticated service/UI access |
| `MEMORY_SERVICE_API_KEY` | No production default | Machine secret for `X-Service-Key`; never send to a browser |
| `JWT_SECRET_KEY` | `jwt-development-secret-key-min-32-chars` | Student Web UI JWT signing key |
| `PORT` | `8000` | HTTP port |

---

## 2. Running with Docker Compose (Recommended)

To launch a dedicated PostgreSQL integration database, run Alembic/ontology
provisioning explicitly, and then launch the Memory Service:

```bash
docker-compose up --build -d
```

Check health:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## 3. Running Locally for Development

### A. Python Backend
```bash
# 1. Activate environment
.\.venv\Scripts\activate

# 2. Run database migrations
alembic upgrade head

# 3. Start FastAPI server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### B. React / TypeScript Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 4. Health & Monitoring Probes

- **Liveness**: `GET /health` returns `200 OK` when the process is up.
- **Readiness**: `GET /ready` returns `200 OK` only when PostgreSQL is reachable,
  Alembic is at `0011_adaptmath_ingestion`, required tables exist, and runtime
  model artifacts are present.

Normal application startup does not create schemas or tables. For a host-run
service, execute `python scripts/provision_local_database.py` explicitly before
starting Uvicorn. This command reports only the database name, schema, revision,
and table names; it never prints credentials.
