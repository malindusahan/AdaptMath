from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
EXPERIMENTS_ROOT = PROJECT_ROOT / "research" / "experiments"

sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

import evaluate_router as base  # noqa: E402
from app.agents.complexity.service import get_complexity_service  # noqa: E402
from app.agents.router.router_agent import RouterAgent  # noqa: E402
from app.core.config import get_settings  # noqa: E402


RESUME_PATH = EXPERIMENTS_ROOT / "router_evaluation_v5_resume.json"
ROUTER_SOURCE_PATH = BACKEND_ROOT / "app" / "agents" / "router" / "router_agent.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_fingerprint() -> str:
    payload = {
        "repeats_per_case": base.REPEATS_PER_CASE,
        "test_cases": base.TEST_CASES,
        "context_pairs": base.CONTEXT_PAIRS,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def new_resume_state(model: str) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "router_source_sha256": sha256_file(ROUTER_SOURCE_PATH),
        "benchmark_sha256": benchmark_fingerprint(),
        "repeats_per_case": base.REPEATS_PER_CASE,
        "calls": {},
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    RESUME_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_or_create_state(model: str, reset: bool) -> dict[str, Any]:
    if reset and RESUME_PATH.exists():
        RESUME_PATH.unlink()

    if not RESUME_PATH.exists():
        state = new_resume_state(model)
        save_state(state)
        return state

    state = json.loads(RESUME_PATH.read_text(encoding="utf-8"))

    expected_router_hash = sha256_file(ROUTER_SOURCE_PATH)
    expected_benchmark_hash = benchmark_fingerprint()

    mismatches: list[str] = []
    if state.get("model") != model:
        mismatches.append("model")
    if state.get("router_source_sha256") != expected_router_hash:
        mismatches.append("router source hash")
    if state.get("benchmark_sha256") != expected_benchmark_hash:
        mismatches.append("benchmark hash")
    if state.get("repeats_per_case") != base.REPEATS_PER_CASE:
        mismatches.append("repeat count")

    if mismatches:
        raise RuntimeError(
            "Existing resume file does not match the current evaluation "
            f"configuration ({', '.join(mismatches)}). Run again with "
            "--reset only if you intentionally want a fresh benchmark."
        )

    return state


def get_saved_call(
    state: dict[str, Any],
    case_id: str,
    repeat_index: int,
) -> dict[str, Any] | None:
    result = (
        state.get("calls", {})
        .get(case_id, {})
        .get(str(repeat_index))
    )
    if not result:
        return None
    if result.get("route") not in base.VALID_ROUTES:
        return None
    if not result.get("reason"):
        return None
    return result


def store_call(
    state: dict[str, Any],
    case_id: str,
    repeat_index: int,
    route: str,
    reason: str,
    latency_seconds: float,
    source: str,
) -> None:
    state.setdefault("calls", {}).setdefault(case_id, {})[
        str(repeat_index)
    ] = {
        "route": route,
        "reason": reason,
        "latency_seconds": latency_seconds,
        "source": source,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)


def seed_from_current_results(state: dict[str, Any]) -> int:
    """
    Import only successful/valid calls from the current standard result file.

    Use this only when that result file came from the SAME Router source and
    SAME benchmark definition as the current interrupted run. The resume file
    itself is hash-locked after seeding.
    """
    if not base.OUTPUT_PATH.exists():
        return 0

    data = json.loads(base.OUTPUT_PATH.read_text(encoding="utf-8"))
    imported = 0

    expected_case_ids = {case["case_id"] for case in base.TEST_CASES}

    for case in data.get("cases", []):
        case_id = case.get("case_id")
        if case_id not in expected_case_ids:
            continue

        for run in case.get("runs", []):
            repeat_index = run.get("repeat")
            route = run.get("route")
            reason = run.get("reason")
            latency = run.get("latency_seconds")
            error = run.get("error")

            if (
                not isinstance(repeat_index, int)
                or route not in base.VALID_ROUTES
                or not isinstance(reason, str)
                or not reason.strip()
                or not isinstance(latency, (int, float))
                or error is not None
            ):
                continue

            if get_saved_call(state, case_id, repeat_index) is not None:
                continue

            store_call(
                state,
                case_id,
                repeat_index,
                route,
                reason,
                float(latency),
                "seeded_from_current_results",
            )
            imported += 1

    return imported


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        type(exc).__name__ == "RateLimitError"
        or "rate_limit_exceeded" in text
        or "rate limit reached" in text
    )


def count_complete_calls(state: dict[str, Any]) -> int:
    total = 0
    for case in base.TEST_CASES:
        for repeat_index in range(1, base.REPEATS_PER_CASE + 1):
            if get_saved_call(state, case["case_id"], repeat_index):
                total += 1
    return total


def collect_missing_calls(state: dict[str, Any]) -> bool:
    settings = get_settings()
    complexity_service = get_complexity_service()
    router = RouterAgent()

    total_calls = len(base.TEST_CASES) * base.REPEATS_PER_CASE
    completed = count_complete_calls(state)

    print("\n=== AdaptMath Router Resumable Evaluation ===\n")
    print("Router source SHA-256:", state["router_source_sha256"])
    print("Benchmark SHA-256:", state["benchmark_sha256"])
    print("Model:", settings.groq_model)
    print("Already completed valid calls:", completed, "/", total_calls)
    print("Resume file:", RESUME_PATH)
    print()

    for case_index, case in enumerate(base.TEST_CASES, start=1):
        complexity_score = float(complexity_service.predict(case["question"]))
        router_input = base.build_router_input(case, complexity_score)

        print(f"Case {case_index}/{len(base.TEST_CASES)}: {case['case_id']}")

        for repeat_index in range(1, base.REPEATS_PER_CASE + 1):
            cached = get_saved_call(state, case["case_id"], repeat_index)
            if cached is not None:
                print(
                    f"  Run {repeat_index}: cached -> {cached['route']}"
                )
                continue

            start = time.perf_counter()
            try:
                output = router.route(router_input)
                latency = time.perf_counter() - start

                if output.route not in base.VALID_ROUTES:
                    raise RuntimeError(
                        f"Router returned invalid route: {output.route!r}"
                    )
                if not output.reason or not output.reason.strip():
                    raise RuntimeError("Router returned an empty reason.")

                store_call(
                    state,
                    case["case_id"],
                    repeat_index,
                    output.route,
                    output.reason,
                    latency,
                    "live_call",
                )
                completed += 1
                print(
                    f"  Run {repeat_index}: {output.route} "
                    f"({completed}/{total_calls})"
                )

            except Exception as exc:
                save_state(state)
                print(
                    f"  Run {repeat_index}: STOPPED - "
                    f"{type(exc).__name__}: {exc}"
                )
                if is_rate_limit_error(exc):
                    print(
                        "\nProvider rate limit reached. No failed call is "
                        "counted as benchmark evidence. Wait for quota to "
                        "recover, then run this SAME command again; completed "
                        "valid calls will be reused automatically."
                    )
                else:
                    print(
                        "\nEvaluation stopped on an unexpected error. The "
                        "completed valid calls remain saved for diagnosis."
                    )
                return False

            if completed < total_calls:
                time.sleep(base.INTER_CALL_DELAY_SECONDS)

        print()

    return True


class ReplayRouter:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.index = 0

    def route(self, _router_input: Any) -> SimpleNamespace:
        if self.index >= len(self.records):
            raise RuntimeError("ReplayRouter exhausted saved calls.")
        record = self.records[self.index]
        self.index += 1
        return SimpleNamespace(
            route=record["route"],
            reason=record["reason"],
        )


class ReplayPerfCounter:
    """Return start/end timestamps that reproduce saved call latencies."""

    def __init__(self, latencies: list[float]) -> None:
        self.values: list[float] = []
        current = 1000.0
        for latency in latencies:
            self.values.append(current)
            current += max(float(latency), 0.0)
            self.values.append(current)
            current += 0.001
        self.index = 0

    def __call__(self) -> float:
        if self.index >= len(self.values):
            return self.values[-1] if self.values else 1000.0
        value = self.values[self.index]
        self.index += 1
        return value


def finalize_standard_results(state: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []

    for case in base.TEST_CASES:
        for repeat_index in range(1, base.REPEATS_PER_CASE + 1):
            record = get_saved_call(state, case["case_id"], repeat_index)
            if record is None:
                raise RuntimeError(
                    "Cannot finalize because at least one valid call is missing: "
                    f"{case['case_id']} repeat {repeat_index}."
                )
            records.append(record)

    replay_router = ReplayRouter(records)
    replay_clock = ReplayPerfCounter(
        [float(record["latency_seconds"]) for record in records]
    )

    original_router_agent = base.RouterAgent
    original_delay = base.INTER_CALL_DELAY_SECONDS
    original_perf_counter = base.time.perf_counter

    try:
        base.RouterAgent = lambda: replay_router  # type: ignore[assignment]
        base.INTER_CALL_DELAY_SECONDS = 0.0
        base.time.perf_counter = replay_clock  # type: ignore[assignment]
        print(
            "\nAll 36 valid calls are available. Replaying saved calls "
            "locally to generate the standard metrics/result JSON...\n"
        )
        base.main()
    finally:
        base.RouterAgent = original_router_agent
        base.INTER_CALL_DELAY_SECONDS = original_delay
        base.time.perf_counter = original_perf_counter

    print("\nFINALIZATION COMPLETE")
    print("Standard result file:", base.OUTPUT_PATH)
    print("Raw resumable-call evidence:", RESUME_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the existing AdaptMath Router development benchmark "
            "without counting provider rate-limit failures as model results."
        )
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete saved valid calls and start a fresh benchmark.",
    )
    parser.add_argument(
        "--seed-from-current-results",
        action="store_true",
        help=(
            "Import successful valid calls from the current "
            "router_evaluation_results_v5.json before resuming. Use only when "
            "that file came from the same Router version and benchmark."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    state = load_or_create_state(settings.groq_model, args.reset)

    if args.seed_from_current_results:
        imported = seed_from_current_results(state)
        print("Imported valid calls from current results:", imported)

    complete = collect_missing_calls(state)
    if complete:
        finalize_standard_results(state)


if __name__ == "__main__":
    main()
