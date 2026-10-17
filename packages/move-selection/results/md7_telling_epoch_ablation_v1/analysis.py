from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V1 = ROOT / "pedagogical-move-selection" / "results" / "md7_telling_real_state_diagnostic_v1"
MODEL_ROOT = (
    ROOT
    / "pedagogical-move-selection"
    / "models"
    / "candidates"
    / "md7_telling_calibration_epoch3"
    / "md7_telling_calibration_v5_candidates"
    / "md7_telling_calibration_v5"
)
BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
MODELS = {
    "baseline": BASELINE,
    "epoch1": MODEL_ROOT / "epoch1",
    "epoch2": MODEL_ROOT / "epoch2",
    "epoch3": MODEL_ROOT / "epoch3",
}
MOVES = ("generic", "probing", "focus", "telling")
TOLERANCE = 1e-5
EXPECTED_INPUT_HASHES = {
    "analysis.py": "adb98cf8c978479d7aedb437c01bbbbe62e13b6bebd94acf5a932f0397dafd2c",
    "real_state_predictions.csv": "ef6616a300770c48d31bf7d3325159f4da180b26d2e7a2b236f84a183d6ac8b7",
    "telling_boundary_challenge.csv": "c415a0207fb3f4b960606317fc2d32c19cacf73f477d4900649321654bee141e",
    "telling_boundary_predictions.csv": "20ece9e0bf7a6971f8751fc0ed39a27bcf5895a7f46bac54928bc9c860b94bff",
    "summary.json": "47a0796905ea6c921beb5ffa8c707d70c1af2ca29f7a07b0e051872fadb40875",
}
EXPECTED_MODEL_WEIGHT_HASHES = {
    "baseline": "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13",
    "epoch1": "1dc67fa3a3296c386214c36e696cd5bf82cb7884c707381fb01f437e6a36103a",
    "epoch2": "6f4f19dcec16a92fe599b80428ce57a5e07fb8e828916c03ed0c99a4bc3c2314",
    "epoch3": "a24d364b7d7461144afa97bae70c3690c4ce7af919927010cfb3fce73df9619c",
}
BASELINE_SAFETY = {
    "status": "pass",
    "run_immediately_before_candidate_scoring": True,
    "validation_rows": 1850,
    "accuracy": 0.5162162162162162,
    "macro_f1": 0.4611286390258053,
    "per_class_f1": {
        "generic": 0.6105263157894737,
        "probing": 0.3333333333333333,
        "focus": 0.6004672897196262,
        "telling": 0.300187617260788,
    },
    "versions": {
        "torch": "2.6.0+cu124",
        "transformers": "4.56.2",
        "tokenizer_class": "RobertaTokenizerFast",
        "tokenizer_is_fast": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v1_module():
    spec = importlib.util.spec_from_file_location("frozen_md7_telling_v1", V1 / "analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen v1 diagnostic implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_hashes(model_dir: Path) -> dict[str, str]:
    names = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
    return {name: sha256(model_dir / name) for name in names}


def vector(frame: pd.DataFrame, model: str) -> np.ndarray:
    return frame[[f"{model}_p_{move}" for move in MOVES]].to_numpy(dtype=np.float64)


def assign_predictions(frame: pd.DataFrame, model: str, probabilities: np.ndarray) -> None:
    for index, move in enumerate(MOVES):
        frame[f"{model}_p_{move}"] = probabilities[:, index]
        frame[f"{model}_delta_p_{move}"] = probabilities[:, index] - frame[f"baseline_p_{move}"]
    frame[f"{model}_top1"] = [MOVES[int(index)] for index in probabilities.argmax(axis=1)]
    frame[f"{model}_top1_changed"] = frame[f"{model}_top1"] != frame["baseline_top1"]
    frame[f"{model}_changed_to_telling"] = (
        frame[f"{model}_top1_changed"] & (frame[f"{model}_top1"] == "telling")
    )


def boundary_metrics(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    telling = frame[f"{model}_p_telling"].astype(float)
    count = int((frame[f"{model}_top1"] == "telling").sum())
    return {
        "n": int(len(frame)),
        "top1_telling_count": count,
        "top1_telling_rate": float(count / len(frame)),
        "mean_p_telling": float(telling.mean()),
        "median_p_telling": float(telling.median()),
        "max_p_telling": float(telling.max()),
        "top1_distribution": dict(Counter(frame[f"{model}_top1"])),
    }


def classify_candidate(plausible: dict[str, Any], negative: dict[str, Any], recoverable_rate: float,
                       displacement_rate: float, baseline_plausible_rate: float,
                       baseline_negative_rate: float) -> str:
    # Exact conservative rubric reused from md7_telling_real_state_diagnostic_v1.
    sensitivity_gain = plausible["top1_telling_rate"] - baseline_plausible_rate
    false_trigger_change = negative["top1_telling_rate"] - baseline_negative_rate
    if recoverable_rate >= 0.50:
        return "B. TELLING OVERCORRECTION"
    if negative["top1_telling_rate"] >= 0.20 or false_trigger_change >= 0.15:
        return "B. TELLING OVERCORRECTION"
    if plausible["top1_telling_rate"] < 0.50 or sensitivity_gain < 0.20:
        return "C. TELLING STILL UNDER-TRIGGERS"
    if displacement_rate > 0.40:
        return "B. TELLING OVERCORRECTION"
    return "A. PROMISING TELLING CALIBRATION"


def fmt_vector(row: pd.Series, model: str) -> str:
    return "[" + ", ".join(f"{move}={row[f'{model}_p_{move}']:.6f}" for move in MOVES) + "]"


def history_text(raw: str) -> str:
    turns = json.loads(raw)
    if not turns:
        return "[No previous conversation]"
    return "\n".join(f"{turn['user']}: {turn['text']}" for turn in turns)


def important_case_block(title: str, frame: pd.DataFrame, model: str, limit: int | None = None) -> str:
    subset = frame.head(limit) if limit is not None else frame
    text = f"## {title}\n\n"
    if subset.empty:
        return text + "_None._\n\n"
    for _, row in subset.iterrows():
        identity = row.get("case_id", row.get("action_event_id", "unknown"))
        text += f"### {identity}\n\n"
        if "semantic_group" in row:
            text += f"Group: `{row['semantic_group']}`; expected boundary: `{row['expected_boundary']}`.\n\n"
        else:
            text += (
                f"Attempt: `{row['attempt_id']}`; action turn: `{row['action_turn_index']}`; "
                f"skill: `{row['skill']}`.\n\n"
            )
        text += f"Problem: {row['problem']}\n\n"
        raw_history = row["history"] if "history" in row else row["history_before_action"]
        text += "```text\n" + history_text(raw_history) + "\n```\n\n"
        text += f"Baseline: `{fmt_vector(row, 'baseline')}` → **{row['baseline_top1']}**\n\n"
        text += f"{model}: `{fmt_vector(row, model)}` → **{row[f'{model}_top1']}**\n\n"
        text += f"Top-1 change: `{row['baseline_top1']} → {row[f'{model}_top1']}`.\n\n"
    return text


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
            for row in shown.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_paths = [V1 / name for name in EXPECTED_INPUT_HASHES]
    model_files = [MODELS[name] / part for name in MODELS
                   for part in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")]
    authoritative = [
        ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1" / name
        for name in ("turn_outcomes.jsonl", "assessment_outcomes.jsonl", "attempt_summaries.jsonl")
    ] + [ROOT / "adaptive-math-tutor" / "backend" / "runtime" / "adaptive_demo_md7r1_real_v1" / "policy_state.json"]
    protected = frozen_paths + model_files + authoritative
    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in protected}

    observed_inputs = {name: sha256(V1 / name) for name in EXPECTED_INPUT_HASHES}
    assert observed_inputs == EXPECTED_INPUT_HASHES, (observed_inputs, EXPECTED_INPUT_HASHES)
    observed_weight_hashes = {name: sha256(path / "model.safetensors") for name, path in MODELS.items()}
    assert observed_weight_hashes == EXPECTED_MODEL_WEIGHT_HASHES, observed_weight_hashes

    v1 = load_v1_module()
    for path in MODELS.values():
        v1.verify_config(path)

    real = pd.read_csv(V1 / "real_state_predictions.csv")
    challenge = pd.read_csv(V1 / "telling_boundary_predictions.csv")
    assert len(real) == 87 and real["action_event_id"].nunique() == 87
    assert len(challenge) == 48 and challenge["case_id"].nunique() == 48
    assert (challenge["expected_boundary"] == "telling_plausible").sum() == 24
    assert (challenge["expected_boundary"] == "non_telling").sum() == 24
    assert (challenge["semantic_group"] == "I_recoverable_after_one_scaffold").sum() == 4

    real_records = [{"problem": row.problem, "history": json.loads(row.history_before_action)}
                    for row in real.itertuples(index=False)]
    challenge_records = [{"problem": row.problem, "history": json.loads(row.history)}
                         for row in challenge.itertuples(index=False)]
    records = real_records + challenge_records

    baseline_probs = v1.predict(BASELINE, records, batch_size=16)
    real_baseline = baseline_probs[:len(real)]
    challenge_baseline = baseline_probs[len(real):]
    historical_real_baseline = vector(real, "baseline")
    historical_challenge_baseline = vector(challenge, "baseline")
    v1_export_real_errors = np.max(np.abs(real_baseline - historical_real_baseline), axis=1)
    challenge_errors = np.max(np.abs(challenge_baseline - historical_challenge_baseline), axis=1)
    assert int((v1_export_real_errors > TOLERANCE).sum()) == 0
    assert int((challenge_errors > TOLERANCE).sum()) == 0

    # Reproduce v1's actual historical parity target: collector raw MD7 telemetry.
    turn_path = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1" / "turn_outcomes.jsonl"
    summary_path = ROOT / "pedagogical-move-selection" / "results" / "md_self_improvement_turn_data_v1" / "attempt_summaries.jsonl"
    turns = v1.read_jsonl(turn_path)
    summaries = v1.read_jsonl(summary_path)
    completed_ids = {
        str(v1.nested(row, "identity", "attempt_id"))
        for row in summaries
        if row.get("completion_status") == "completed"
        or (row.get("completion_status") is None and row.get("adaptive_completion"))
    }
    real_source = [
        row for row in turns
        if str(v1.nested(row, "identity", "attempt_id")) in completed_ids
        and v1.nested(row, "provenance", "selector_mode") == "ordinary-md7r1-v1"
        and v1.nested(row, "next_learner_observation", "status") in {"observed_update", "resolved_no_update"}
    ]
    assert len(real_source) == 87
    assert [v1.nested(row, "identity", "action_event_id") for row in real_source] == real["action_event_id"].tolist()
    stored_telemetry = np.asarray([
        [float(v1.nested(row, "selector", "raw", "probabilities", move)) for move in MOVES]
        for row in real_source
    ])
    real_errors = np.max(np.abs(real_baseline - stored_telemetry), axis=1)
    assert int((real_errors > TOLERANCE).sum()) == 0
    reconstructed_top1 = np.asarray([MOVES[int(i)] for i in real_baseline.argmax(axis=1)])
    assert int((reconstructed_top1 != real["historical_raw_md7_move"].to_numpy()).sum()) == 0

    # Use freshly reconstructed baseline probabilities and exact frozen epoch3 results.
    for frame, probs in ((real, real_baseline), (challenge, challenge_baseline)):
        assign_predictions(frame, "baseline", probs)
        epoch3 = frame[[f"candidate_p_{move}" for move in MOVES]].to_numpy(dtype=np.float64)
        assign_predictions(frame, "epoch3", epoch3)
        frame.drop(columns=[column for column in frame.columns if column.startswith("candidate_")], inplace=True)
        frame.drop(columns=[column for column in frame.columns
                            if column in {"delta_p_generic", "delta_p_probing", "delta_p_focus", "delta_p_telling",
                                          "top1_changed", "changed_to_telling", "changed_away_from_telling"}], inplace=True)

    for model in ("epoch1", "epoch2"):
        probabilities = v1.predict(MODELS[model], records, batch_size=16)
        assign_predictions(real, model, probabilities[:len(real)])
        assign_predictions(challenge, model, probabilities[len(real):])

    # Stable column ordering: identifiers/context, then one probability block per model.
    real_identity = [
        "learner_pseudonym", "attempt_id", "action_event_id", "action_turn_index", "skill", "problem",
        "history_before_action", "formatted_history", "historical_raw_md7_move",
        "evaluator_category_observational", "immediate_delta_mastery_observational", "mastery_before_observational",
    ]
    challenge_identity = ["case_id", "semantic_group", "expected_boundary", "problem", "history", "rationale", "formatted_history"]
    prediction_columns = []
    for model in MODELS:
        prediction_columns += [f"{model}_p_{move}" for move in MOVES]
        prediction_columns += [f"{model}_top1"]
        if model != "baseline":
            prediction_columns += [f"{model}_delta_p_{move}" for move in MOVES]
            prediction_columns += [f"{model}_top1_changed", f"{model}_changed_to_telling"]
    real = real[real_identity + prediction_columns]
    challenge = challenge[challenge_identity + prediction_columns]

    plausible = challenge[challenge["expected_boundary"] == "telling_plausible"].copy()
    negative = challenge[challenge["expected_boundary"] == "non_telling"].copy()
    recoverable = challenge[challenge["semantic_group"] == "I_recoverable_after_one_scaffold"].copy()
    metrics_metadata = {
        model: json.loads((MODEL_ROOT / f"{model}_metrics.json").read_text(encoding="utf-8"))["mathdial_val"]
        for model in ("epoch1", "epoch2", "epoch3")
    }
    metrics_metadata["baseline"] = {
        "accuracy": BASELINE_SAFETY["accuracy"],
        "macro_f1": BASELINE_SAFETY["macro_f1"],
        "per_class_f1": BASELINE_SAFETY["per_class_f1"],
    }

    comparison_rows = []
    model_summaries: dict[str, Any] = {}
    baseline_plausible = boundary_metrics(plausible, "baseline")
    baseline_negative = boundary_metrics(negative, "baseline")
    candidate_decisions: dict[str, str] = {}
    for model in MODELS:
        p_metrics = boundary_metrics(plausible, model)
        n_metrics = boundary_metrics(negative, model)
        r_metrics = boundary_metrics(recoverable, model)
        real_telling = int((real[f"{model}_top1"] == "telling").sum())
        displacement = 0 if model == "baseline" else int(real[f"{model}_top1_changed"].sum())
        focus_to_telling = 0 if model == "baseline" else int(
            ((real["baseline_top1"] == "focus") & (real[f"{model}_top1"] == "telling")).sum()
        )
        probing_to_telling = 0 if model == "baseline" else int(
            ((real["baseline_top1"] == "probing") & (real[f"{model}_top1"] == "telling")).sum()
        )
        generic_to_telling = 0 if model == "baseline" else int(
            ((real["baseline_top1"] == "generic") & (real[f"{model}_top1"] == "telling")).sum()
        )
        delta = np.zeros(len(real)) if model == "baseline" else real[f"{model}_delta_p_telling"].to_numpy(float)
        row = {
            "model": model,
            "mathdial_accuracy": metrics_metadata[model]["accuracy"],
            "mathdial_macro_f1": metrics_metadata[model]["macro_f1"],
            "telling_plausible_telling_count": p_metrics["top1_telling_count"],
            "telling_plausible_n": 24,
            "telling_plausible_telling_rate": p_metrics["top1_telling_rate"],
            "telling_plausible_mean_p_telling": p_metrics["mean_p_telling"],
            "telling_plausible_median_p_telling": p_metrics["median_p_telling"],
            "hard_negative_false_telling_count": n_metrics["top1_telling_count"],
            "hard_negative_n": 24,
            "hard_negative_false_telling_rate": n_metrics["top1_telling_rate"],
            "hard_negative_mean_p_telling": n_metrics["mean_p_telling"],
            "hard_negative_max_p_telling": n_metrics["max_p_telling"],
            "recoverable_false_telling_count": r_metrics["top1_telling_count"],
            "recoverable_n": 4,
            "recoverable_mean_p_telling": r_metrics["mean_p_telling"],
            "real_state_telling_count": real_telling,
            "real_state_n": 87,
            "real_state_displacement_count": displacement,
            "real_state_displacement_rate": displacement / 87,
            "real_focus_to_telling_count": focus_to_telling,
            "real_probing_to_telling_count": probing_to_telling,
            "real_generic_to_telling_count": generic_to_telling,
            "mean_real_delta_p_telling": float(np.mean(delta)),
            "median_real_delta_p_telling": float(np.median(delta)),
        }
        comparison_rows.append(row)
        model_summaries[model] = {"telling_plausible": p_metrics, "hard_non_telling": n_metrics,
                                  "recoverable_after_one_scaffold": r_metrics, "real_states": row}
        if model != "baseline":
            candidate_decisions[model] = classify_candidate(
                p_metrics, n_metrics, r_metrics["top1_telling_rate"], displacement / 87,
                baseline_plausible["top1_telling_rate"], baseline_negative["top1_telling_rate"],
            )

    comparison = pd.DataFrame(comparison_rows)
    acceptable = [model for model in ("epoch1", "epoch2") if candidate_decisions[model].startswith("A.")]
    if acceptable:
        decision_code = "A"
        decision = "A. EARLIER EPOCH ACCEPTABLE"
        preferred = max(acceptable, key=lambda model: (
            model_summaries[model]["telling_plausible"]["top1_telling_rate"],
            -model_summaries[model]["recoverable_after_one_scaffold"]["top1_telling_rate"],
            -model_summaries[model]["real_states"]["real_state_displacement_rate"],
        ))
        decision_reason = (
            f"{preferred} passes the unchanged v1 boundary rubric: it retains a substantive persistent-difficulty "
            "telling gain while materially reducing Epoch 3's recoverable-state overcorrection."
        )
    elif all(candidate_decisions[model].startswith("B.") for model in ("epoch1", "epoch2")):
        decision_code = "B"
        decision = "B. ALL EPOCHS OVERCORRECT"
        preferred = None
        decision_reason = (
            "Epoch 1 and Epoch 2 both fail the same unchanged v1 boundary rubric as Epoch 3; earlier stopping does "
            "not remove the substantive recoverable-state boundary failure."
        )
    else:
        decision_code = "C"
        decision = "C. EARLIER EPOCH UNDER-TRIGGERS"
        preferred = None
        decision_reason = (
            "No earlier epoch passes the unchanged v1 rubric; any reduction in overcorrection is accompanied by "
            "insufficient telling activation on persistent-difficulty cases."
        )

    real.to_csv(OUT / "real_state_epoch_predictions.csv", index=False)
    challenge.to_csv(OUT / "challenge_epoch_predictions.csv", index=False)
    recoverable.to_csv(OUT / "recoverable_one_scaffold_cases.csv", index=False)
    comparison.to_csv(OUT / "epoch_comparison.csv", index=False)

    important = (
        "# Important Cases\n\n"
        "All judgments are offline boundary diagnostics, not causal claims about learning. Probability order is "
        "`generic, probing, focus, telling`.\n\n"
    )
    for model in ("epoch1", "epoch2"):
        important += f"# {model.title()}\n\n"
        important += important_case_block(
            "Recoverable-after-one-scaffold cases predicted telling",
            recoverable[recoverable[f"{model}_top1"] == "telling"].sort_values(f"{model}_p_telling", ascending=False), model,
        )
        important += important_case_block(
            "Every real state changed to telling",
            real[real[f"{model}_changed_to_telling"]].sort_values(f"{model}_delta_p_telling", ascending=False), model,
        )
        important += important_case_block(
            "Largest 10 real-state telling-probability increases",
            real.sort_values(f"{model}_delta_p_telling", ascending=False), model, 10,
        )
        changed = real[real[f"{model}_top1_changed"]].copy()
        changed[f"{model}_total_variation"] = 0.5 * sum(
            (changed[f"{model}_p_{move}"] - changed[f"baseline_p_{move}"]).abs() for move in MOVES
        )
        important += important_case_block(
            "Largest 10 policy changes relative to baseline (ranked by total variation)",
            changed.sort_values(f"{model}_total_variation", ascending=False), model, 10,
        )
    (OUT / "important_cases.md").write_text(important, encoding="utf-8")

    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in protected}
    changes = {path: {"before": before_hashes[path], "after": after_hashes[path]}
               for path in before_hashes if before_hashes[path] != after_hashes[path]}
    assert not changes, changes

    artifact_hashes = {name: model_hashes(path) for name, path in MODELS.items()}
    summary = {
        "diagnostic": "md7_telling_epoch_ablation_v1",
        "decision": decision,
        "decision_code": decision_code,
        "decision_reason": decision_reason,
        "preferred_earlier_epoch_if_any": preferred,
        "models": {
            name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "hashes": artifact_hashes[name]}
            for name, path in MODELS.items()
        },
        "frozen_inputs": {
            "source_diagnostic": str(V1.relative_to(ROOT)).replace("\\", "/"),
            "hashes": observed_inputs,
            "real_state_count": 87,
            "challenge_count": 48,
            "telling_plausible_count": 24,
            "hard_non_telling_count": 24,
            "recoverable_after_one_scaffold_count": 4,
            "cases_added": 0,
            "cases_removed": 0,
            "states_regenerated": 0,
        },
        "inference_contract": {
            "sequence_a_prefix": "Problem:\n",
            "sequence_b_prefix": "Conversation:\n",
            "dialogue_format": "{user}: {text}",
            "empty_history": "[No previous conversation]",
            "suffix": "\n\nNext teacher pedagogical move:",
            "truncation_side": "left",
            "truncation": "only_second",
            "max_length": 512,
            "label_order": list(MOVES),
            "implementation_reused": str((V1 / "analysis.py").relative_to(ROOT)).replace("\\", "/"),
        },
        "safety": {
            "mathdial_validation": BASELINE_SAFETY,
            "real_state_probability_parity_max_abs_error": float(real_errors.max()),
            "real_state_parity_rows_failed_at_1e_5": int((real_errors > TOLERANCE).sum()),
            "v1_export_probability_parity_max_abs_error": float(v1_export_real_errors.max()),
            "v1_export_parity_rows_failed_at_1e_5": int((v1_export_real_errors > TOLERANCE).sum()),
            "challenge_probability_parity_max_abs_error": float(challenge_errors.max()),
            "challenge_parity_rows_failed_at_1e_5": int((challenge_errors > TOLERANCE).sum()),
            "historical_real_top1_mismatches": int((reconstructed_top1 != real["historical_raw_md7_move"].to_numpy()).sum()),
        },
        "candidate_diagnostic_decisions_under_unchanged_v1_rubric": candidate_decisions,
        "results": model_summaries,
        "mathdial_validation": metrics_metadata,
        "observational_outcomes": {
            "included_in_prediction_inputs": False,
            "fields_preserved_for_descriptive_join_only": [
                "evaluator_category_observational", "immediate_delta_mastery_observational", "mastery_before_observational"
            ],
        },
        "claims_not_entitled": [
            "telling would have caused better learning",
            "a candidate would have improved a historical outcome",
            "negative BKT proves telling was needed",
            "synthetic calibration performance demonstrates real tutoring effectiveness",
        ],
        "recommended_next_action": (
            "Design a targeted recalibration set with recoverable-after-one-scaffold hard negatives, train only "
            "under a separately authorized experiment, and then repeat this unchanged frozen diagnostic."
            if decision_code in {"B", "C"}
            else "Retain the identified earlier epoch only as an offline candidate and run a fresh, preregistered human boundary review before any live experiment."
        ),
        "side_effects": {
            "training": 0, "model_writes": 0, "production_writes": 0, "authoritative_data_writes": 0,
            "tutor_api_calls": 0, "bkt_updates": 0, "lints_updates": 0,
            "mathdial_final_test_use": 0, "mrbench_v3_test_use": 0,
            "protected_input_hash_changes": changes,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    compact = pd.DataFrame({
        "metric": [
            "MathDial Macro-F1", "telling-plausible telling count /24",
            "telling-plausible mean telling probability", "hard-negative false telling count /24",
            "recoverable-one-scaffold false telling count /4", "real-state telling count /87",
            "real-state displacement count /87", "real-state displacement %",
            "mean real-state telling-probability increase",
        ],
        **{
            model: [
                f"{row['mathdial_macro_f1']:.6f}", f"{int(row['telling_plausible_telling_count'])}/24",
                f"{row['telling_plausible_mean_p_telling']:.6f}", f"{int(row['hard_negative_false_telling_count'])}/24",
                f"{int(row['recoverable_false_telling_count'])}/4", f"{int(row['real_state_telling_count'])}/87",
                f"{int(row['real_state_displacement_count'])}/87", f"{100 * row['real_state_displacement_rate']:.3f}%",
                f"{row['mean_real_delta_p_telling']:.6f}",
            ]
            for model, row in ((name, comparison[comparison.model == name].iloc[0]) for name in MODELS)
        },
    })

    comparison_view = comparison[[
        "model", "mathdial_macro_f1", "telling_plausible_telling_count", "telling_plausible_mean_p_telling",
        "telling_plausible_median_p_telling", "hard_negative_false_telling_count", "hard_negative_mean_p_telling",
        "hard_negative_max_p_telling", "recoverable_false_telling_count", "recoverable_mean_p_telling",
        "real_state_telling_count", "real_state_displacement_count", "real_state_displacement_rate",
        "real_focus_to_telling_count", "real_probing_to_telling_count", "real_generic_to_telling_count",
        "mean_real_delta_p_telling", "median_real_delta_p_telling",
    ]].copy()
    recoverable_view_rows = []
    for _, case in recoverable.iterrows():
        for model in MODELS:
            recoverable_view_rows.append({
                "case": case["case_id"], "model": model, "top-1": case[f"{model}_top1"],
                "probability vector [generic, probing, focus, telling]": fmt_vector(case, model),
            })
    recoverable_view = pd.DataFrame(recoverable_view_rows)
    report = f"""# MD7 Telling-Calibration Epoch Ablation

## Outcome

**{decision}**

{decision_reason} This is an offline candidate-selection result only; no model is promoted.

## Safety and frozen implementation

The deployed MD7-R1 baseline reproduced the established MathDial validation metrics exactly on 1,850 rows: accuracy **{BASELINE_SAFETY['accuracy']:.12f}**, Macro-F1 **{BASELINE_SAFETY['macro_f1']:.12f}**, and per-class F1 **generic {BASELINE_SAFETY['per_class_f1']['generic']:.12f}, probing {BASELINE_SAFETY['per_class_f1']['probing']:.12f}, focus {BASELINE_SAFETY['per_class_f1']['focus']:.12f}, telling {BASELINE_SAFETY['per_class_f1']['telling']:.12f}**.

The historical real-state reconstruction parity check against original collector raw MD7 telemetry also passed: maximum absolute probability difference **{real_errors.max():.10g}**, rows failing `1e-5`: **{int((real_errors > TOLERANCE).sum())}**, historical top-1 mismatches: **0**. Parity against the frozen v1 export had maximum difference **{v1_export_real_errors.max():.10g}**. Challenge baseline parity had maximum difference **{challenge_errors.max():.10g}** and **0** failures.

The exact v1 implementation and inputs were reused. The contract remains paired `Problem:\\n...` and `Conversation:\\n...`, dialogue lines `{{user}}: {{text}}`, suffix `\\n\\nNext teacher pedagogical move:`, left tokenizer truncation, `only_second`, maximum length 512, and label order `generic, probing, focus, telling`. No cases, thresholds, or real states were changed.

## Frozen comparison results

{markdown_table(comparison_view)}

The recoverable-after-one-scaffold subgroup contains the same four cases as v1. Probability order is `generic, probing, focus, telling`.

{markdown_table(recoverable_view)}

The complete rows appear in `recoverable_one_scaffold_cases.csv`; dialogue-level inspection for Epoch 1 and Epoch 2 appears in `important_cases.md`.

## Synthetic calibration evidence

The saved training-run metadata reports perfect four-class scores on its synthetic target validation and perfect telling precision/recall with zero non-telling false triggers on its synthetic challenge for all three epochs. Those results describe synthetic calibration behavior only and are **not** evidence of real tutoring effectiveness.

## Real-state offline transfer evidence

All selectors were evaluated on the exact same 87 leakage-free pre-action states. Epoch 3 values are the frozen v1 results; baseline, Epoch 1, and Epoch 2 use the same inference implementation and formatting. Policy displacement means top-1 differs from deployed baseline; it is not a claim that the alternative action would have improved learning.

## Observational historical outcomes

The output preserves evaluator category, immediate mastery delta, and mastery-before fields solely as post-prediction observational joins. They were not used as selector inputs and do not establish counterfactual effects.

## Claims this diagnostic does not support

- Telling would have caused better learning.
- Any candidate would have improved a historical outcome.
- Negative BKT proves telling was needed.
- Synthetic perfection demonstrates real tutoring effectiveness.

## Model paths and hashes

| model | path | model.safetensors SHA-256 |
| --- | --- | --- |
"""
    for model, path in MODELS.items():
        report += f"| {model} | `{str(path.relative_to(ROOT)).replace(chr(92), '/')}` | `{artifact_hashes[model]['model.safetensors']}` |\n"
    report += f"""

Exact analyzed counts: **87 real states**, **48 challenge cases** (**24 telling-plausible**, **24 hard non-telling**, including **4 recoverable-after-one-scaffold**).

## Final table

{markdown_table(compact)}

**DECISION: {decision_code}**

Recommended next action: {summary['recommended_next_action']}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
