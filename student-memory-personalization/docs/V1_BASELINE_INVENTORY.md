# V1 Frozen Baseline Inventory

## Purpose

This document protects the completed Version 1 system before Version 2 development begins.

```text
Version: V1 — Validated Learning-State Memory Core
Status:  COMPLETE AND FROZEN
Date:    2026-08-17
```

V2 work must extend the repository without silently changing V1 evidence, artifacts, metrics, or API claims.

The machine-readable checksum inventory is:

```text
artifacts/v1_baseline_inventory.json
```

Inventory SHA-256 at creation:

```text
039cd805325491541936f1dcfc4698dff2b92b59d4ee266f164310219650b8a7
```

## Frozen Validation Evidence

```text
Full automated test suite: 234 passed
API tests:                  61 passed
pip check:                  PASS
Artifact checksum checks:   PASS
Clean-state API smoke test: PASS
Runtime database at freeze: absent
Port 8001 at freeze:        not listening
```

The remaining FastAPI/Starlette `TestClient` deprecation warning was documented as non-blocking.

## Frozen Runtime Contract

```text
Python:     3.11.0
Entry point: src.api.app:app
Base URL:   http://127.0.0.1:8001
Database:   SQLite, generated at database/student_memory.db when required
```

Frozen endpoints:

```text
GET  /health
POST /memory/update
GET  /memory/{student_id}/current
GET  /memory/{student_id}/history
```

## Frozen Learning-State Model

```text
Model:              DecisionTreeClassifier
Predictors:         12 behavioural/history features
Held-out Macro F1:  0.8717
Model artifact:     artifacts/learning_state_model.joblib
Imputer artifact:   artifacts/learning_state_imputer.joblib
Metadata artifact:  artifacts/learning_state_model_metadata.json
Runtime manifest:   artifacts/runtime_manifest.json
```

Frozen artifact SHA-256 values:

```text
Model:    77a54e638bae449b8878e5957b6eeb007a43137ae1d2f7b6cd613532d7dcf3f9
Imputer:  fc88fd4625a573182f533896107e929782321ac3bb80bda23626beda1ed9de62
Metadata: 2462536e06dbe386fa45412cbe2dcae9f3cd4ce3077d919131ff08ac1822931b
```

The labels are derived research labels rather than externally annotated ground truth. The behavioural model excludes the direct historical-accuracy fields used to derive those labels.

## Baseline File Inventory

The checksum inventory records 66 baseline files:

```text
Source files:             23
Test files:               21
Frozen artifacts:          4
Research evidence files:   9
Documentation files:       2
Configuration files:       7
```

The inventory covers:

- runtime source and schemas;
- all V1 automated tests;
- model, imputer, metadata, and runtime manifest;
- SQLite schema reference;
- pinned dependency files and environment examples;
- README and development log at the V1 freeze point;
- notebooks;
- raw and processed research datasets.

Excluded intentionally:

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
database/student_memory.db
```

The runtime database is mutable application state and is generated on demand. It is not part of the frozen V1 distribution.

## V1 Research Evidence

The preserved V1 evidence includes:

```text
notebooks/01_dataset_exploration.ipynb
notebooks/02_data_preprocessing.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/02_model_development.ipynb

data/raw/skill_builder_data.csv
data/processed/base_interactions.csv
data/processed/concept_interactions.csv
data/processed/learning_state_features.csv
data/processed/learning_state_labelled.csv
```

Each file has an individual SHA-256 value in the machine-readable inventory.

## Change-Control Rules for V2

1. V1 artifact files must not be overwritten by V2 training.
2. V1 hashes and metrics must remain reported as V1 results.
3. V2 database migrations must preserve or explicitly version compatibility boundaries.
4. Existing V1 API contracts must remain available or receive a documented migration path.
5. New experiments require durable notebooks/reports and saved evaluation outputs.
6. New thresholds and derived fields require definitions, evidence sources, validation methods, and tests.
7. `docs/PROJECT_DEVELOPMENT_LOG.md` must be updated after every completed V2 phase.
8. V2 results must not be described as V1 results, and V1 validation must not be retroactively reinterpreted.

## Source-Control Status

At the moment this inventory was created, the project directory was not yet initialized as a Git repository.

Git initialization and the first traceable V2 commit/tag are separate Phase 11 actions. This inventory was deliberately created first so the exact pre-V2 state is documented before source-control metadata is introduced.

## Baseline Acceptance

```text
V1 BASELINE INVENTORIED
V1 EVIDENCE PRESERVED
V2 SOURCE CHANGES NOT YET STARTED
```
