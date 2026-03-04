"""Import legacy attempt/turn JSONL evidence with stable idempotency keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pedagogical-move-selection"))

from src.self_improvement.postgres_repository import (  # noqa: E402
    add_response_quality,
    append_experience,
    complete_action,
    stage_action,
)


OUTCOME_KEYS = {
    "resolver_event_id", "mastery_before", "mastery_after", "raw_delta",
    "headroom_normalized_delta", "configured_reward_mode",
    "configured_reward_value", "reward_attributed_to_move",
    "reward_attributed_to_shadow_hypothetical", "learner_signals_for_next_action",
    "posterior_update_occurred", "posterior_updated", "posterior_update_origin",
    "posterior_update_weight", "agency_forced_no_policy_update", "update_id",
    "update_timestamp", "outcome_observed", "resolution_status", "outcome_status",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.database_url:
        os.environ["RESEARCH_DATABASE_URL"] = args.database_url
    os.environ["RESEARCH_PERSISTENCE"] = "postgres"
    for path in args.paths:
        count = 0
        digest = hashlib.sha256()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            count += 1
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            if args.dry_run:
                continue
            action_id = record.get("action_event_id")
            if action_id and record.get("attempt_id"):
                stage = {key: value for key, value in record.items() if key not in OUTCOME_KEYS}
                scores = stage.pop("current_response_mrb1", None)
                tutor_response = stage.pop("tutor_response", None)
                stage_action(stage)
                if isinstance(scores, dict):
                    add_response_quality(str(action_id), scores, tutor_response)
                outcome = {key: record[key] for key in OUTCOME_KEYS if key in record}
                complete_action(action_id=str(action_id), outcome=outcome)
            else:
                append_experience(
                    record,
                    event_id_override=f"legacy:{path.parent.name}:{path.name}:{line_number}",
                )
        print(
            f"path={path.resolve()} source={count} sha256={digest.hexdigest()} "
            f"dry_run={str(args.dry_run).lower()}"
        )


if __name__ == "__main__":
    main()

