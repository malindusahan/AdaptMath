"""Script to export OpenAPI 3.1 schema to docs/openapi.json."""

import json
from pathlib import Path

from src.api.app import app

docs_dir = Path("docs")
docs_dir.mkdir(parents=True, exist_ok=True)
openapi_path = docs_dir / "openapi.json"

with open(openapi_path, "w", encoding="utf-8") as f:
    json.dump(app.openapi(), f, indent=2)

print(f"Exported OpenAPI schema to {openapi_path}")
