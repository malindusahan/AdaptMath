# Conservative cleanup plan

Inventory timestamp: 2026-08-29T04:30:04.858316Z

- Project size before: 17584105703 bytes.
- Files/directories inventoried: 253355 / 28959.
- Exact notebooks retained: 36.
- Dataset-like retained assets: 1124.
- Model lineages indexed: 46.
- Ambiguous entries retained: 1594.
- Verified safe candidates: 79458 files and 10504 directory entries, totaling 1584443300 file bytes.

Deletion allowlist is limited to `__pycache__`, pytest/mypy/ruff/matplotlib/coverage caches, `htmlcov`, `.pyc`, and `.pyo`. Every target must resolve beneath the workspace. Notebooks, datasets, models, results, source, dependencies, runtime state/events, manifests, logs, and ambiguous paths are ineligible.
