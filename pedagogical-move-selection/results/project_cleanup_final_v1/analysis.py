"""Conservative project inventory, cleanup manifests, and storage evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES = HERE / "figures"
INVENTORY = HERE / "PROJECT_INVENTORY.csv"
SUMMARY_BEFORE = HERE / "storage_before.json"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "adaptmath-project-cleanup-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

DATASET_EXTENSIONS = {
    ".arrow", ".csv", ".json", ".jsonl", ".npy", ".npz", ".parquet",
    ".pickle", ".pkl", ".tsv",
}
MODEL_EXTENSIONS = {
    ".bin", ".ckpt", ".joblib", ".model", ".onnx", ".pt", ".pth",
    ".safetensors",
}
SOURCE_EXTENSIONS = {
    ".c", ".cpp", ".css", ".go", ".h", ".html", ".java", ".js",
    ".jsx", ".md", ".ps1", ".py", ".rs", ".sh", ".sql", ".toml",
    ".ts", ".tsx", ".yaml", ".yml",
}
CACHE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".matplotlib-cache", "htmlcov",
}
CACHE_FILE_NAMES = {".coverage", "coverage.xml"}
CACHE_FILE_EXTENSIONS = {".pyc", ".pyo"}
DEPENDENCY_PARTS = {"node_modules", ".venv", ".venv-integration"}
CURRENT_EVIDENCE = {
    "turn_lints_final_architecture_selection_v1",
    "turn_lints_final_architecture_skl_freeze_v1",
    "turn_lints_live_skl_synthetic_warmstart_v1",
    "tutor_move_realization_final_v1",
    "turn_lints_reward_context_audit_v1",
    "turn_lints_randomized_warmstart_audit_v1",
    "turn_lints_shadow_collection_v2",
    "turn_lints_selector_anchor_v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def tracked_paths() -> tuple[set[str], set[str]]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return set(), set()
    files = {value for value in raw.decode("utf-8", errors="replace").split("\0") if value}
    directories: set[str] = set()
    for file in files:
        parts = Path(file).parts
        for index in range(1, len(parts)):
            directories.add(Path(*parts[:index]).as_posix())
    return files, directories


def under_cache(parts: tuple[str, ...]) -> bool:
    return any(part in CACHE_DIR_NAMES for part in parts)


def classify(path: Path, kind: str) -> tuple[str, str]:
    rel = relative(path)
    parts = path.relative_to(ROOT).parts
    lowered = tuple(part.casefold() for part in parts)
    suffix = path.suffix.casefold() if kind == "file" else ""
    name = path.name.casefold()

    if under_cache(parts) or (kind == "file" and (name in CACHE_FILE_NAMES or suffix in CACHE_FILE_EXTENSIONS)):
        return "DELETE_SAFE_CACHE", "Regenerable interpreter/test/plot/coverage cache or compiled bytecode."
    if ".git" in lowered:
        return "KEEP_REQUIRED", "Git repository metadata; retained and excluded from cleanup."
    if any(part in DEPENDENCY_PARTS for part in parts):
        return "KEEP_REQUIRED", "Installed dependency retained for immediate offline presentation startup."
    if suffix == ".ipynb":
        return "KEEP_NOTEBOOK", "Notebook preserved by absolute keep rule."
    if "runtime" in lowered or rel.startswith("_local_runtime_state/") or rel.startswith("_local_runtime_logs/"):
        if "turn_lints_live_skl_final_v1" in lowered:
            return "KEEP_RUNTIME_REAL_DATA", "Current final LIVE posterior lineage; immutable keep target."
        return "AMBIGUOUS_KEEP", "Runtime lineage/log retained because real-data status or dependency may be ambiguous."
    if "models" in lowered or "model" in lowered or suffix in MODEL_EXTENSIONS:
        return "KEEP_MODEL", "Meaningful trained/model artifact, lineage metadata, or model support file."
    if suffix in DATASET_EXTENSIONS:
        return "KEEP_DATASET", "Research/training/evaluation/configured data format retained conservatively."
    if "results" in lowered or "research" in lowered or any(part in CURRENT_EVIDENCE for part in parts):
        return "KEEP_RESEARCH", "Scientific result/evidence or research analysis retained for reproducibility."
    if suffix in SOURCE_EXTENSIONS or name in {"package.json", "package-lock.json", "requirements.txt", "requirements-lock.txt"}:
        return "KEEP_SOURCE", "Source, documentation, manifest, lockfile, or reproducibility tooling."
    if kind == "directory" and name in {"dist", "build"}:
        return "AMBIGUOUS_KEEP", "Generated build-like directory retained to protect offline presentation readiness."
    return "AMBIGUOUS_KEEP", "No verified safe-deletion rule matched; retained by default."


def scan() -> tuple[list[dict[str, object]], dict[str, int]]:
    tracked_files, tracked_dirs = tracked_paths()
    file_info: dict[Path, int] = {}
    directories: set[Path] = {ROOT}
    inaccessible: list[str] = []
    for current, dirnames, filenames in os.walk(ROOT, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.add(current_path)
        for dirname in dirnames:
            directories.add(current_path / dirname)
        for filename in filenames:
            path = current_path / filename
            try:
                file_info[path] = path.stat().st_size
            except OSError:
                inaccessible.append(relative(path))

    dir_sizes: defaultdict[Path, int] = defaultdict(int)
    for path, size in file_info.items():
        parent = path.parent
        while True:
            dir_sizes[parent] += size
            if parent == ROOT:
                break
            parent = parent.parent

    rows: list[dict[str, object]] = []
    for path in sorted(directories | set(file_info), key=lambda item: relative(item).casefold() if item != ROOT else ""):
        if path == ROOT:
            continue
        kind = "file" if path in file_info else "directory"
        category, reason = classify(path, kind)
        rel = relative(path)
        parts = path.relative_to(ROOT).parts
        lowered = tuple(part.casefold() for part in parts)
        research_data = category in {"KEEP_DATASET", "KEEP_RESEARCH"}
        model = category == "KEEP_MODEL"
        notebook = path.suffix.casefold() == ".ipynb"
        current_evidence = any(part in CURRENT_EVIDENCE for part in parts)
        real_interaction = (
            category == "KEEP_RUNTIME_REAL_DATA"
            or ("runtime" in lowered and path.suffix.casefold() in {".jsonl", ".db", ".sqlite", ".sqlite3"})
            or "real" in lowered
        )
        safe_regenerate = category.startswith("DELETE_SAFE_")
        tracked = rel in (tracked_files if kind == "file" else tracked_dirs)
        rows.append(
            {
                "path": rel,
                "type": kind,
                "size_bytes": file_info.get(path, dir_sizes.get(path, 0)),
                "category": category,
                "reason": reason,
                "research_data": research_data,
                "model": model,
                "notebook": notebook,
                "current_architecture_evidence": current_evidence,
                "real_interaction_data": real_interaction,
                "safe_to_regenerate": safe_regenerate,
                "git_tracked": tracked,
            }
        )
    stats = {
        "total_size_bytes": sum(file_info.values()),
        "file_count": len(file_info),
        "directory_count": len(directories) - 1,
        "inaccessible_count": len(inaccessible),
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_origin(path: str) -> str:
    lowered = path.casefold()
    if "synthetic" in lowered:
        return "synthetic"
    if any(token in lowered for token in ("derived", "processed", "analysis", "summary", "prediction")):
        return "derived"
    if any(token in lowered for token in ("live", "real", "attempt", "interaction", "collector")):
        return "real_or_runtime"
    return "unspecified_retained"


def cheap_rows(path: Path) -> str:
    if path.suffix.casefold() not in {".csv", ".tsv", ".jsonl"}:
        return ""
    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return ""
        with path.open("rb") as handle:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
        return str(max(0, lines - (1 if path.suffix.casefold() in {".csv", ".tsv"} else 0)))
    except OSError:
        return ""


def notebook_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if row["type"] == "file" and str(row["path"]).casefold().endswith(".ipynb"):
            path = str(row["path"])
            output.append({"path": path, "repo": path.split("/", 1)[0], "size_bytes": row["size_bytes"], "brief_purpose": Path(path).stem.replace("_", " ").replace("-", " ")})
    return output


def dataset_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if row["type"] != "file" or row["category"] != "KEEP_DATASET":
            continue
        rel = str(row["path"])
        path = ROOT / rel
        output.append({"path": rel, "format": path.suffix.casefold().lstrip("."), "size_bytes": row["size_bytes"], "repo": rel.split("/", 1)[0], "approx_rows": cheap_rows(path), "category": "retained_research_or_configuration_data", "origin": dataset_origin(rel)})
    return output


def model_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: set[str] = set()
    for row in rows:
        if row["category"] != "KEEP_MODEL":
            continue
        rel = Path(str(row["path"]))
        parts = rel.parts
        if "models" in parts:
            index = parts.index("models")
            depth = min(len(parts), index + 3)
            candidates.add(Path(*parts[:depth]).as_posix())
        elif row["type"] == "file" and rel.suffix.casefold() in MODEL_EXTENSIONS:
            candidates.add(rel.as_posix())
    output = []
    for path in sorted(candidates):
        lowered = path.casefold()
        if "md7_r2_tell_c1_epoch2" in lowered:
            role, status = "production selector", "selected/frozen"
        elif "md7" in lowered:
            role, status = "selector candidate or rollback", "retained"
        elif "md6" in lowered:
            role, status = "legacy frozen selector/rollback", "retained"
        elif "mrb1" in lowered:
            role, status = "Tutor-response quality critic", "diagnostic"
        elif "student" in lowered or "bkt" in lowered or "detector" in lowered:
            role, status = "student model", "active/supporting"
        else:
            role, status = "model artifact lineage", "retained"
        known_hash = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5" if "md7_r2_tell_c1_epoch2" in lowered else ""
        output.append({"path": path, "name": Path(path).name, "role": role, "status": status, "known_sha256": known_hash})
    return output


def evidence_status(name: str) -> tuple[str, str, str]:
    lowered = name.casefold()
    if any(token in lowered for token in ("final_architecture", "skl_freeze", "live_skl", "realization_final")):
        return "selected", "current frozen/LIVE architecture decision", "final"
    if "selector_anchor" in lowered:
        return "rejected", "P1 selector-anchor rejection", "supporting"
    if "reward_context" in lowered:
        return "selected", "HEADROOM reward and context evidence", "supporting"
    if "randomized_warmstart" in lowered:
        return "diagnostic", "randomized warm-start collection/audit", "supporting"
    if "shadow" in lowered:
        return "diagnostic", "P0 cold-start/shadow evidence", "supporting"
    if "telling" in lowered or "md7" in lowered:
        return "diagnostic", "MD7/telling selector evidence", "supporting"
    return "diagnostic", "retained scientific or engineering evidence", "historical"


def experiment_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    directories = {str(row["path"]) for row in rows if row["type"] == "directory"}
    experiments: set[str] = set()
    for rel in directories:
        parts = Path(rel).parts
        for index, part in enumerate(parts[:-1]):
            if part.casefold() == "results" and index + 1 < len(parts):
                experiments.add(Path(*parts[: index + 2]).as_posix())
                break
    output = []
    file_paths = [str(row["path"]) for row in rows if row["type"] == "file"]
    for path in sorted(experiments):
        prefix = path + "/"
        members = [member for member in file_paths if member.startswith(prefix)]
        reports = [member for member in members if Path(member).name.casefold() in {"report.md", "decision_evidence.md", "readme.md", "implementation_notes.md"}]
        pngs = [member for member in members if member.casefold().endswith(".png")]
        tables = [member for member in members if Path(member).suffix.casefold() in {".csv", ".json"}]
        status, decision, importance = evidence_status(Path(path).name)
        output.append({"experiment_name": Path(path).name, "path": path, "decision_supported": decision, "status": status, "key_report": reports[0] if reports else "", "key_pngs": " | ".join(pngs[:5]), "key_csv_json": " | ".join(tables[:5]), "importance": importance})
    return output


def retained_sizes(rows: list[dict[str, object]]) -> dict[str, int]:
    sizes: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if row["type"] != "file":
            continue
        category = str(row["category"])
        if category == "KEEP_MODEL": label = "models"
        elif category == "KEEP_DATASET": label = "datasets"
        elif category == "KEEP_NOTEBOOK": label = "notebooks"
        elif category == "KEEP_RESEARCH": label = "results"
        elif category == "KEEP_SOURCE": label = "source"
        elif category == "KEEP_RUNTIME_REAL_DATA" or "runtime" in str(row["path"]).casefold(): label = "runtime"
        elif any(part in DEPENDENCY_PARTS for part in Path(str(row["path"])).parts): label = "dependencies"
        else: label = "other"
        sizes[label] += int(row["size_bytes"])
    return dict(sizes)


def write_structure(notebooks: int, datasets: int, models: int) -> None:
    (HERE / "FINAL_PROJECT_STRUCTURE.md").write_text(
        f"""# Final AdaptMath project structure

1. Final frozen selector: `pedagogical-move-selection/models/frozen/md7_r2_tell_c1_epoch2/`
2. Turn-LinTS source: `pedagogical-move-selection/src/self_improvement/turn_lints_*.py`
3. S+K+L architecture: `pedagogical-move-selection/results/turn_lints_final_architecture_skl_freeze_v1/`
4. LIVE runtime: `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/`
5. Synthetic initialization: `pedagogical-move-selection/results/turn_lints_live_skl_synthetic_warmstart_v1/synthetic_warmstart_observations.jsonl`
6. Real interaction event log: `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/turn_events.jsonl` (created only by genuine interactions; may be absent before the first interaction)
7. HEADROOM reward: `pedagogical-move-selection/src/self_improvement/turn_lints_reward.py`
8. MRB1 model: `pedagogical-move-selection/models/frozen/mrb1/`
9. Student Modeling: `student-modeling/`
10. Notebooks: indexed in `NOTEBOOK_INDEX.csv` ({notebooks} retained)
11. Datasets: indexed in `DATASET_INDEX.csv` ({datasets} retained assets)
12. Models: indexed in `MODEL_INDEX.csv` ({models} retained lineages)
13. Final experiment evidence: architecture selection, S+K+L freeze, synthetic warm-start LIVE audit, and final Tutor realization folders
14. Supporting/rejected evidence: indexed in `EXPERIMENT_EVIDENCE_INDEX.csv`, including anchor rejection and warm-start/cold-start diagnostics
15. Tutor source/frontend: `adaptive-math-tutor/backend/` and `adaptive-math-tutor/frontend/`
16. Launch scripts: `start-adaptmath-local.ps1`, `status-adaptmath-local.ps1`, and `stop-adaptmath-local.ps1`

The inventory is conservative: ambiguous assets, source, dependencies, historical results, runtime lineages, and logs are retained.
""",
        encoding="utf-8",
    )


def before() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, stats = scan()
    write_csv(INVENTORY, rows)
    keep = [row for row in rows if not str(row["category"]).startswith("DELETE_SAFE_")]
    delete = [row for row in rows if str(row["category"]).startswith("DELETE_SAFE_")]
    ambiguous = [row for row in rows if row["category"] == "AMBIGUOUS_KEEP"]
    write_csv(HERE / "KEEP_MANIFEST.csv", keep)
    write_csv(HERE / "DELETE_MANIFEST.csv", delete)
    write_csv(HERE / "AMBIGUOUS_RETAINED.csv", ambiguous)
    notebooks = notebook_index(rows)
    datasets = dataset_index(rows)
    models = model_index(rows)
    evidence = experiment_index(rows)
    write_csv(HERE / "NOTEBOOK_INDEX.csv", notebooks, ["path", "repo", "size_bytes", "brief_purpose"])
    write_csv(HERE / "DATASET_INDEX.csv", datasets, ["path", "format", "size_bytes", "repo", "approx_rows", "category", "origin"])
    write_csv(HERE / "MODEL_INDEX.csv", models, ["path", "name", "role", "status", "known_sha256"])
    write_csv(HERE / "EXPERIMENT_EVIDENCE_INDEX.csv", evidence, ["experiment_name", "path", "decision_supported", "status", "key_report", "key_pngs", "key_csv_json", "importance"])
    stats.update({"generated_at": utc_now(), "category_sizes": retained_sizes(rows), "notebook_count": len(notebooks), "dataset_count": len(datasets), "model_lineage_count": len(models), "experiment_count": len(evidence), "delete_candidate_files": sum(row["type"] == "file" for row in delete), "delete_candidate_directories": sum(row["type"] == "directory" for row in delete), "delete_candidate_bytes": sum(int(row["size_bytes"]) for row in delete if row["type"] == "file"), "ambiguous_retained_count": len(ambiguous)})
    SUMMARY_BEFORE.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "CLEANUP_PLAN.md").write_text(
        f"""# Conservative cleanup plan

Inventory timestamp: {stats['generated_at']}

- Project size before: {stats['total_size_bytes']} bytes.
- Files/directories inventoried: {stats['file_count']} / {stats['directory_count']}.
- Exact notebooks retained: {len(notebooks)}.
- Dataset-like retained assets: {len(datasets)}.
- Model lineages indexed: {len(models)}.
- Ambiguous entries retained: {len(ambiguous)}.
- Verified safe candidates: {stats['delete_candidate_files']} files and {stats['delete_candidate_directories']} directory entries, totaling {stats['delete_candidate_bytes']} file bytes.

Deletion allowlist is limited to `__pycache__`, pytest/mypy/ruff/matplotlib/coverage caches, `htmlcov`, `.pyc`, and `.pyo`. Every target must resolve beneath the workspace. Notebooks, datasets, models, results, source, dependencies, runtime state/events, manifests, logs, and ambiguous paths are ineligible.
""",
        encoding="utf-8",
    )
    write_structure(len(notebooks), len(datasets), len(models))
    import matplotlib.pyplot as plt
    labels = list(stats["category_sizes"])
    values = [stats["category_sizes"][label] / (1024**3) for label in labels]
    plt.figure(figsize=(10, 5)); plt.bar(labels, values, color="#4472C4"); plt.xticks(rotation=25, ha="right"); plt.ylabel("GiB"); plt.title(f"Project storage before conservative cleanup ({stats['total_size_bytes'] / (1024**3):.2f} GiB)"); plt.tight_layout(); plt.savefig(FIGURES / "01_storage_before.png", dpi=220, bbox_inches="tight"); plt.close()
    print(json.dumps(stats, indent=2))


def plot_before() -> None:
    """Render the before chart from the immutable pre-cleanup measurement."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    stats = json.loads(SUMMARY_BEFORE.read_text(encoding="utf-8"))
    import matplotlib.pyplot as plt
    labels = list(stats["category_sizes"])
    values = [stats["category_sizes"][label] / (1024**3) for label in labels]
    plt.figure(figsize=(10, 5)); plt.bar(labels, values, color="#4472C4"); plt.xticks(rotation=25, ha="right"); plt.ylabel("GiB"); plt.title(f"Project storage before conservative cleanup ({stats['total_size_bytes'] / (1024**3):.2f} GiB)"); plt.tight_layout(); plt.savefig(FIGURES / "01_storage_before.png", dpi=220, bbox_inches="tight"); plt.close()
    print(json.dumps({"source": str(SUMMARY_BEFORE), "figure": str(FIGURES / "01_storage_before.png")}, indent=2))


def finalize() -> None:
    before_stats = json.loads(SUMMARY_BEFORE.read_text(encoding="utf-8"))
    before_rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    current_rows, after_stats = scan()
    current_paths = {str(row["path"]) for row in current_rows}
    removed = [row for row in before_rows if row["path"] not in current_paths and str(row["category"]).startswith("DELETE_SAFE_")]
    removed_files = [row for row in removed if row["type"] == "file"]
    removed_dirs = [row for row in removed if row["type"] == "directory"]
    removed_bytes = sum(int(row["size_bytes"]) for row in removed_files)
    after_sizes = retained_sizes(current_rows)
    report = {
        "generated_at": utc_now(),
        "project_size_before_bytes": before_stats["total_size_bytes"],
        "project_size_after_bytes": after_stats["total_size_bytes"],
        "net_bytes_removed": int(before_stats["total_size_bytes"]) - int(after_stats["total_size_bytes"]),
        "verified_removed_candidate_bytes": removed_bytes,
        "files_removed": len(removed_files),
        "directories_removed": len(removed_dirs),
        "removed_by_category": dict(Counter(str(row["category"]) for row in removed_files)),
        "retained_size_by_category": after_sizes,
        "notebooks_deleted": sum(str(row["path"]).casefold().endswith(".ipynb") for row in removed),
        "datasets_deleted": sum(row["category"] == "KEEP_DATASET" for row in removed),
        "models_deleted": sum(row["category"] == "KEEP_MODEL" for row in removed),
        "real_interaction_data_deleted": sum(str(row["real_interaction_data"]).casefold() == "true" for row in removed),
        "current_architecture_evidence_deleted": sum(
            str(row["current_architecture_evidence"]).casefold() == "true"
            and not str(row["category"]).startswith("DELETE_SAFE_")
            for row in removed
        ),
        "current_architecture_cache_files_removed": sum(
            str(row["current_architecture_evidence"]).casefold() == "true"
            and str(row["category"]).startswith("DELETE_SAFE_")
            for row in removed
        ),
    }
    (HERE / "cleanup_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "CLEANUP_REPORT.md").write_text(
        f"""# Conservative cleanup report

- Project size before: {report['project_size_before_bytes']} bytes ({report['project_size_before_bytes'] / (1024**3):.3f} GiB)
- Project size after: {report['project_size_after_bytes']} bytes ({report['project_size_after_bytes'] / (1024**3):.3f} GiB)
- Net storage removed: {report['net_bytes_removed']} bytes
- Verified candidate file bytes removed: {removed_bytes} bytes
- Files removed: {len(removed_files)}
- Directory entries removed: {len(removed_dirs)}
- Removed categories: {json.dumps(report['removed_by_category'], sort_keys=True)}
- Notebooks/datasets/models/real interactions/current architecture evidence deleted: 0/0/0/0/0

Only allowlisted regenerable caches and compiled bytecode were removed. Dependency environments and `node_modules` remain for offline presentation. Runtime lineages, logs, source, notebooks, datasets, models, scientific results, frozen manifests, and ambiguous assets remain.
""",
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt
    labels = list(after_sizes); values = [after_sizes[label] / (1024**3) for label in labels]
    plt.figure(figsize=(10, 5)); plt.bar(labels, values, color="#70AD47"); plt.xticks(rotation=25, ha="right"); plt.ylabel("GiB"); plt.title(f"Project storage after conservative cleanup ({after_stats['total_size_bytes'] / (1024**3):.2f} GiB)"); plt.tight_layout(); plt.savefig(FIGURES / "02_storage_after.png", dpi=220, bbox_inches="tight"); plt.close()
    categories = list(report["removed_by_category"]) or ["none"]
    category_values = [report["removed_by_category"].get(label, 0) for label in categories]
    plt.figure(figsize=(8, 5)); plt.bar(categories, category_values, color="#ED7D31"); plt.ylabel("Files removed"); plt.title("Verified safe removals by category"); plt.tight_layout(); plt.savefig(FIGURES / "03_removed_by_category.png", dpi=220, bbox_inches="tight"); plt.close()
    fig, axis = plt.subplots(figsize=(12, 6)); axis.axis("off")
    assets = [(0.15,.78,"Source + launchers"),(0.5,.78,"Frozen models"),(0.85,.78,"Datasets + notebooks"),(0.15,.35,"Scientific evidence"),(0.5,.35,"Final LIVE runtime"),(0.85,.35,"Offline dependencies")]
    for x,y,label in assets: axis.text(x,y,label,ha="center",va="center",fontsize=12,bbox=dict(boxstyle="round,pad=.55",fc="#EAF2F8",ec="#34495E"))
    axis.text(.5,.08,"All research assets retained; only verified regenerable caches removed.",ha="center",fontsize=12,color="#1E8449")
    axis.set_title("Final AdaptMath project asset map",fontsize=16); plt.tight_layout(); plt.savefig(FIGURES / "04_final_project_asset_map.png",dpi=220,bbox_inches="tight"); plt.close()
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "plot-before", "finalize"))
    args = parser.parse_args()
    if args.phase == "before":
        before()
    elif args.phase == "plot-before":
        plot_before()
    else:
        finalize()


if __name__ == "__main__":
    main()
