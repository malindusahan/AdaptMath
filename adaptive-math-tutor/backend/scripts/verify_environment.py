from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path


# When this file is executed as ``python scripts/verify_environment.py``, Python
# puts the ``scripts`` directory (rather than the backend project root) on
# sys.path. Add the backend root explicitly so project modules such as ``app``
# can be imported reliably regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_PYTHON = (3, 13, 7)
EXPECTED_PACKAGES = {
    "fastapi": "0.141.1",
    "uvicorn": "0.52.3",
    "google-genai": "2.19.0",
    "langchain-core": "1.5.5",
    "langchain-google-genai": "4.3.4",
    "langgraph": "1.2.11",
    "langgraph-checkpoint": "4.2.0",
    "langgraph-checkpoint-sqlite": "3.1.1",
    "pydantic": "2.13.4",
    "pydantic-settings": "2.15.0",
    "python-dotenv": "1.2.2",
    "joblib": "1.5.3",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "sympy": "1.14.0",
    "scikit-learn": "1.9.0",
    "torch": "2.13.0",
    "transformers": "5.15.0",
}

IMPORT_CHECKS = {
    "FastAPI": "fastapi",
    "LangGraph": "langgraph",
    "Pydantic": "pydantic",
    "Google GenAI client": "google.genai",
    "LangChain Google GenAI": "langchain_google_genai",
    "SymPy": "sympy",
    "joblib": "joblib",
    "NumPy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyTorch": "torch",
    "Transformers": "transformers",
}


def main() -> int:
    failures: list[str] = []

    actual_python = sys.version_info[:3]
    print(f"Python: {'.'.join(map(str, actual_python))}")
    if actual_python != EXPECTED_PYTHON:
        failures.append(
            "Python version mismatch: expected "
            f"{'.'.join(map(str, EXPECTED_PYTHON))}, got "
            f"{'.'.join(map(str, actual_python))}."
        )

    print("\nPinned packages:")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "NOT INSTALLED"

        marker = "OK" if actual == expected else "MISMATCH"
        print(f"  {marker:8} {package}=={actual}")
        if actual != expected:
            failures.append(f"{package}: expected {expected}, got {actual}.")

    print("\nRunning pip check...")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    pip_output = (completed.stdout or completed.stderr).strip()
    if pip_output:
        print(pip_output)
    if completed.returncode != 0:
        failures.append("pip check reported dependency conflicts.")

    print("\nRequired imports:")
    for label, module_name in IMPORT_CHECKS.items():
        try:
            import_module(module_name)
            print(f"  OK       {label}")
        except Exception as exc:
            print(f"  FAILED   {label}: {type(exc).__name__}: {exc}")
            failures.append(f"{label} import failed: {type(exc).__name__}: {exc}")

    workspace_root = PROJECT_ROOT.parents[1]
    student_root = Path(
        os.getenv("STUDENT_MODEL_REPO", workspace_root / "student-modeling")
    ).resolve()
    move_root = Path(
        os.getenv(
            "PEDAGOGICAL_MOVE_REPO",
            workspace_root / "pedagogical-move-selection",
        )
    ).resolve()
    for root in (student_root, move_root):
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    print("\nLocal integration imports:")
    local_imports = {
        "student BKT": "bkt.predict",
        "student knowledge graph": "core.knowledge_graph",
        "student cross-session pipeline": "core.cross_session_pipeline",
        "student database": "db.database",
        "completion bridge": "src.integration.student_model_v3_bridge",
        "frozen Tutor pipeline": "src.self_improvement.adaptive_tutor_pipeline",
    }
    for label, module_name in local_imports.items():
        try:
            import_module(module_name)
            print(f"  OK       {label}")
        except Exception as exc:
            print(f"  FAILED   {label}: {type(exc).__name__}: {exc}")
            failures.append(f"{label} import failed: {type(exc).__name__}: {exc}")

    print("\nGemini credential:")
    try:
        from app.core.config import get_settings

        present = bool(get_settings().gemini_api_key.get_secret_value().strip())
    except Exception:
        present = False
    print("  PRESENT" if present else "  ABSENT")
    if not present:
        failures.append("GEMINI_API_KEY is absent; a manual Tutor demo requires it.")

    print("\nLoading the serialized complexity model...")
    try:
        from app.agents.complexity.service import get_complexity_service

        service = get_complexity_service()
        if service.model is None:
            raise RuntimeError("Complexity model resolved to None.")
        print("  OK       complexity_model.joblib loaded")
    except Exception as exc:  # explicit verification output
        print(f"  FAILED   {type(exc).__name__}: {exc}")
        failures.append(
            f"Complexity model load failed: {type(exc).__name__}: {exc}"
        )

    if failures:
        print("\nENVIRONMENT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nENVIRONMENT VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
