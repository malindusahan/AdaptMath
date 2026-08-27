# 🧠 Student Personalization Memory Subsystem

A production-grade, persistent student-memory and neural topic-classification subsystem designed for multi-agent educational architectures (Tutor, Evaluator, Planner, and Router).

Developed as a core component of the **TEA CUPS Student Personalization System**.

---

## 📑 Table of Contents
1. [System Architecture & 3-Tier Memory](#1-system-architecture--3-tier-memory)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-Step Installation & Setup from Scratch](#3-step-by-step-installation--setup-from-scratch)
4. [Database Configuration & Team Sharing](#4-database-configuration--team-sharing)
5. [Viewing Tables & Data in pgAdmin / DBeaver](#5-viewing-tables--data-in-pgadmin--dbeaver)
6. [Starting the Application (Backend & Frontend)](#6-starting-the-application-backend--frontend)
7. [Core API Endpoints for Other Agents](#7-core-api-endpoints-for-other-agents)
8. [Running Automated Tests](#8-running-automated-tests)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. System Architecture & 3-Tier Memory

```text
Student Free-Text Math Question
              │
              ▼
    POST /topic/classify  ───────────────► Fine-tuned MiniLM classifies into 1 of 111 Canonical Topics
            .\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
   ┌──────────┴────────────────────────┬────────────────────────────────┐

            Run this command from the repository root. If the API is already running on
            port `8000`, do not start a second copy; use the existing server at
            `http://127.0.0.1:8000`.
   ▼                                   ▼                                ▼
Tutor Agent                     Planner Agent                    Evaluator Agent
GET /memory/{id}/tutor-context  GET /memory/{id}/planner-context POST /memory/update
   │                                   │                                │
   └───────────────────────────────────┼────────────────────────────────┘
                                       │
                                       ▼
             POSTGRESQL MULTI-TIER MEMORY ENGINE (`student_memory` schema)
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. Short-Term Memory: Active session progress, recent accuracy & hints  │
    │ 2. Long-Term Memory: Lifetime student profile across all past sessions  │
    │ 3. Concept Memory: All historical data for a specific math concept      │
    │ 4. Legacy Learning State: Memory-only projection, never BKT mastery     │
    │ 5. Misconception Tracker: Active recurring student errors per topic     │
    │ 6. Interaction Logs: Permanent source-of-truth raw event stream         │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites

Ensure the following are installed on your machine:
* **Python 3.11+**: [Download Python](https://www.python.org/downloads/) (Check: `python --version`)
* **Node.js 18+ & npm**: [Download Node.js](https://nodejs.org/) (Check: `node -v` and `npm -v`)
* **PostgreSQL 15+** *(Optional if using shared cloud database)*: [Download PostgreSQL](https://www.postgresql.org/download/)
* **pgAdmin 4** or **DBeaver**: For graphical database table inspection.
* **Git**: [Download Git](https://git-scm.com/)

---

## 3. Step-by-Step Installation & Setup from Scratch

Follow these exact steps to get the entire backend, frontend, and database running on your laptop.

### Step 3.1: Clone the Repository
```bash
git clone https://github.com/Kugenthiran-Mathusan/student_personalization_memory.git
cd student_personalization_memory
```

### Step 3.2: Create and Activate Python Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3.3: Install Python Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 4. Database Configuration & Team Sharing

### Option A: Shared Cloud Database (Recommended for Team Collaboration)
If your team uses a shared cloud database (e.g. **Neon.tech** or **Supabase**), you do **NOT** need to install PostgreSQL locally. Everyone connects to the exact same database.

1. Create a `.env` file in the project root:
   ```env
   MEMORY_API_HOST=127.0.0.1
   MEMORY_API_PORT=8000

   MEMORY_DATABASE_URL=postgresql+psycopg://<username>:<password>@<cloud-host>:5432/<dbname>?sslmode=require
   MEMORY_DATABASE_SCHEMA=student_memory
   ```

---

### Option B: Local PostgreSQL Database Setup

The canonical local setup is the dedicated Docker Compose database. It uses
database `adaptmath_memory_integration`, schema `student_memory`, host port
`5433`, and a project-specific named volume; it does not reuse or alter another
local PostgreSQL database.

1. Create your `.env` file in the project root:
   ```env
   MEMORY_API_HOST=127.0.0.1
   MEMORY_API_PORT=8000

   MEMORY_DATABASE_URL=postgresql+psycopg://memory_app:<password>@127.0.0.1:5433/adaptmath_memory_integration
   MEMORY_DATABASE_SCHEMA=student_memory
   ```
2. Run `docker compose up --build`. The one-shot `memory_provision` service
   applies Alembic to the explicit head and seeds the ontology before the API
   starts.

---

### Step 4.1: Run Database Migrations & Initialize Tables

Run Alembic to create all schemas, tables, and presentation views:

**On Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\alembic.exe upgrade head
```

**On Linux / macOS:**
```bash
PYTHONPATH="." alembic upgrade head
```

### Step 4.2: Seed Canonical Topics Ontology (111 Skills)

Run the clean/seed script to verify and seed the 111 ASSISTments canonical skills:

**On Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe scripts\clean_database.py
```

**On Linux / macOS:**
```bash
PYTHONPATH="." python scripts/clean_database.py
```

---

## 5. Viewing Tables & Data in pgAdmin / DBeaver

During viva evaluations or team testing, you can open and inspect the database directly in **pgAdmin 4**:

### How to Connect pgAdmin:
1. Open pgAdmin ➔ Right click **Servers** ➔ **Register** ➔ **Server...**
2. **General Tab**: Name = `Student Memory Local` (or `Cloud DB`).
3. **Connection Tab**:
   - **Host name / address**: `127.0.0.1` (or your cloud host)
   - **Port**: `5433` (Docker PostgreSQL; use `5432` only for a native PostgreSQL installation)
   - **Maintenance database**: `adaptmath_memory_integration`
   - **Username**: `memory_app`
   - **Password**: the value of `MEMORY_POSTGRES_PASSWORD`
4. Click **Save**.

To see the readable skill columns, right-click the `student_memory` schema and
choose **Refresh**, then open `v_short_term_memory`, `v_concept_memory`, and
`v_long_term_memory` under **Views**. Short-term and concept views show
`skill_name`; the long-term view shows `skill_names`, a list of all skills
contributing to that student's aggregate memory.

### Where to Find the Tables:
In the left sidebar, navigate to:
`adaptmath_memory_integration` ➔ `Schemas` ➔ `student_memory` ➔ `Tables` / `Views`.

All memory tables are designed with **human-readable identifiers at the very front**:

| Table | First Columns | Description |
| :--- | :--- | :--- |
| **`concept_memory`** | `student_id`, **`student_external_id`**, `canonical_skill_id`, **`skill_name`**, `accuracy`... | Full historical performance on each specific math topic. |
| **`short_term_memory`** | `student_id`, **`student_external_id`**, `session_id`, `current_skill_id`, **`skill_name`**... | Progress and friction within the current study session. |
| **`long_term_memory`** | `student_id`, **`student_external_id`**, `total_sessions`, `concept_count`, `overall_accuracy`... | Cross-session lifetime student profile. |
| **`current_learning_state`** | `student_id`, **`student_external_id`**, `canonical_skill_id`, **`skill_name`**, `learning_state`... | Dynamic ML prediction (`STRONG`, `DEVELOPING`, `NEEDS_SUPPORT`). |
| **`student_misconceptions`** | `misconception_id`, `student_id`, **`student_external_id`**, `skill_name`, `display_error`... | Specific active errors (e.g. `"addition sign error"`). |

### Quick SQL Presentation Views:
You can also run instant queries using the pre-built presentation views:
```sql
SELECT * FROM student_memory.v_concept_memory;
SELECT * FROM student_memory.v_short_term_memory;
SELECT * FROM student_memory.v_long_term_memory;
SELECT * FROM student_memory.v_current_learning_state;
SELECT * FROM student_memory.v_student_misconceptions;
```

---

## 6. Starting the Application (Backend & Frontend)

### 6.1 Start the FastAPI Backend Server
In your first terminal (with `.venv` activated):

```powershell
$env:PYTHONPATH="."
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

* **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 6.2 Start the React + Vite Frontend Dashboard
In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at: **[http://localhost:5173](http://localhost:5173)**
* **Student Problem-Solving Chat**: Interact with questions and observe live memory updates.
* **Memory Inspector**: View real-time Short-Term, Long-Term, and Concept Memory breakdown.

---

## 7. Core API Endpoints for Other Agents

### 1. Neural Topic Extraction (`POST /topic/classify`)
Classifies free-text math questions into one of 111 canonical skills:
```bash
curl -X POST "http://127.0.0.1:8000/topic/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Solve 3x + 5 = 20 for x"}'
```

### 2. Update Student Memory (`POST /memory/update`)
Submitted by the Evaluator after grading an assessment:
```json
POST /memory/update
{
  "student_id": "alex",
  "topic": "Linear Equations",
  "assessment_questions": [
    {
      "question_id": "q1",
      "question": "Solve 3x = 15",
      "student_answer": "5",
      "expected_answer": "5",
      "is_correct": true,
      "attempt_count": 1,
      "hint_count": 0,
      "response_time_ms": 12000
    }
  ]
}
```

### 3. Dedicated Agent Context Endpoints
* **Tutor Agent**: `GET /memory/{student_id}/tutor-context`
* **Planner Agent**: `GET /memory/{student_id}/planner-context`
* **FAPR-LB Struggle Detector**: `GET /memory/{student_id}/fapr-context`
* **Unified Context**: `GET /memory/{student_id}/context`

---

## 8. Running Automated Tests

Run the complete test suite (all integration, contract, and migration tests):

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 9. Troubleshooting & FAQs

#### Q1: Alembic says `relation "student_memory.xxx" does not exist`
* Make sure your `.env` contains `MEMORY_DATABASE_SCHEMA=student_memory` and run `$env:PYTHONPATH="."; .\.venv\Scripts\alembic.exe upgrade head`.

#### Q2: How do I completely wipe and start fresh data?
* Run `$env:PYTHONPATH="."; .\.venv\Scripts\python.exe scripts\clean_database.py`. This resets all student records to 0 while keeping the schema and 111 topics intact.

#### Q3: Do my teammates need PostgreSQL installed if we use cloud DB?
* **No!** If using Neon or Supabase, team members only need Python and Node.js.
