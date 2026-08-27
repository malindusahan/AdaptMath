from pathlib import Path
import re


PROJECT_ROOT = Path(".").resolve()

OUTPUT_DIR = Path(
    "results/model_recovery"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    /
    "si9a_training_asset_audit.txt"
)


SEARCH_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".ps1",
    ".yaml",
    ".yml",
}


SEARCH_TERMS = [
    # Models / experiments
    "answerdotai/ModernBERT-base",
    "ModernBERT",
    "md3",
    "MD3",
    "distilbert/distilroberta-base",
    "DistilRoBERTa",
    "distilroberta",
    "md4",
    "MD4",

    # Hugging Face training
    "TrainingArguments",
    "Trainer",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "weight_decay",
    "warmup_ratio",
    "warmup_steps",
    "lr_scheduler_type",
    "num_train_epochs",
    "eval_strategy",
    "evaluation_strategy",
    "save_strategy",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "save_total_limit",
    "fp16",
    "bf16",
    "gradient_checkpointing",

    # Reproducibility
    "seed",
    "data_seed",
    "set_seed",
    "manual_seed",
    "deterministic",

    # Loss / labels
    "CrossEntropyLoss",
    "class_weights",
    "class_weight",
    "inverse_frequency",
    "weighted_ce",
    "label2id",
    "id2label",

    # Input construction
    "max_length",
    "truncation",
    "padding",
    "tokenizer",
    "history",
    "problem",
    "conversation",
]


SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}


CANDIDATE_NAME_PATTERNS = [
    r"md3",
    r"md4",
    r"modernbert",
    r"distilroberta",
    r"selector",
    r"train",
    r"split",
    r"preprocess",
    r"prediction",
    r"validation",
]


def should_skip(path):
    return any(
        part in SKIP_DIR_NAMES
        for part in path.parts
    )


def read_text_safely(path):
    try:
        return path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        try:
            return path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return None
    except Exception:
        return None


def find_candidate_files():

    candidates = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if should_skip(path):
            continue

        relative = path.relative_to(
            PROJECT_ROOT
        )

        relative_text = str(
            relative
        ).lower()

        if any(
            re.search(
                pattern,
                relative_text,
                flags=re.IGNORECASE,
            )
            for pattern
            in CANDIDATE_NAME_PATTERNS
        ):
            candidates.append(
                relative
            )

    return sorted(
        candidates,
        key=lambda x:
            str(x).lower(),
    )


def search_source_files():

    matches = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if should_skip(path):
            continue

        if (
            path.suffix.lower()
            not in
            SEARCH_EXTENSIONS
        ):
            continue

        text = read_text_safely(
            path
        )

        if text is None:
            continue

        lines = text.splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            matched_terms = [
                term
                for term
                in SEARCH_TERMS
                if term.lower()
                in line.lower()
            ]

            if not matched_terms:
                continue

            matches.append(
                {
                    "path":
                        path.relative_to(
                            PROJECT_ROOT
                        ),

                    "line_number":
                        line_number,

                    "matched_terms":
                        matched_terms,

                    "line":
                        line.strip(),
                }
            )

    return matches


def inspect_known_artifacts():

    paths = [
        Path(
            "artifacts/frozen_selector/md3/"
            "md3_modernbert_predictions.jsonl"
        ),

        Path(
            "artifacts/frozen_selector/md3/"
            "md3_modernbert_validation.json"
        ),

        Path(
            "artifacts/frozen_selector/md4/"
            "md4_weighted_distilroberta_predictions.jsonl"
        ),

        Path(
            "artifacts/frozen_selector/md4/"
            "md4_weighted_distilroberta_validation.json"
        ),

        Path(
            "artifacts/frozen_selector/"
            "recovered_context_routed_hybrid_validation_predictions.jsonl"
        ),

        Path(
            "results/self_improvement/"
            "frozen_adaptive_policy_spec.json"
        ),
    ]

    result = []

    for path in paths:

        result.append(
            (
                path,
                path.exists(),
                (
                    path.stat().st_size
                    if path.exists()
                    else None
                ),
            )
        )

    return result


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_status = (
        inspect_known_artifacts()
    )

    candidate_files = (
        find_candidate_files()
    )

    source_matches = (
        search_source_files()
    )

    output_lines = []

    output_lines.append(
        "=" * 80
    )

    output_lines.append(
        "SI9-A FROZEN MODEL REPRODUCTION ASSET AUDIT"
    )

    output_lines.append(
        "=" * 80
    )

    output_lines.append("")

    # --------------------------------------------------
    # Recovered artifacts
    # --------------------------------------------------

    output_lines.append(
        "KNOWN RECOVERED ARTIFACTS"
    )

    output_lines.append(
        "-" * 80
    )

    for (
        path,
        exists,
        size,
    ) in artifact_status:

        if exists:

            output_lines.append(
                f"[FOUND] {path} "
                f"({size:,} bytes)"
            )

        else:

            output_lines.append(
                f"[MISSING] {path}"
            )

    output_lines.append("")

    # --------------------------------------------------
    # Candidate filenames
    # --------------------------------------------------

    output_lines.append(
        "CANDIDATE PROJECT FILES"
    )

    output_lines.append(
        "-" * 80
    )

    if candidate_files:

        for path in candidate_files:
            output_lines.append(
                str(path)
            )

    else:

        output_lines.append(
            "No candidate files found."
        )

    output_lines.append("")

    # --------------------------------------------------
    # Training config matches
    # --------------------------------------------------

    output_lines.append(
        "TRAINING / PREPROCESSING CONFIG MATCHES"
    )

    output_lines.append(
        "-" * 80
    )

    if source_matches:

        for match in source_matches:

            output_lines.append(
                (
                    f"{match['path']}:"
                    f"{match['line_number']}"
                )
            )

            output_lines.append(
                "  terms: "
                +
                ", ".join(
                    match[
                        "matched_terms"
                    ]
                )
            )

            output_lines.append(
                "  "
                +
                match[
                    "line"
                ]
            )

            output_lines.append("")

    else:

        output_lines.append(
            "No matching source/config lines found."
        )

    output_lines.append("")

    output_lines.append(
        "=" * 80
    )

    output_lines.append(
        "END OF AUDIT"
    )

    output_lines.append(
        "=" * 80
    )

    output_text = "\n".join(
        output_lines
    )

    OUTPUT_PATH.write_text(
        output_text,
        encoding="utf-8",
    )

    print("=" * 80)

    print(
        "SI9-A FROZEN MODEL REPRODUCTION ASSET AUDIT"
    )

    print("=" * 80)

    print()

    found_artifacts = sum(
        1
        for _, exists, _
        in artifact_status
        if exists
    )

    print(
        "Recovered artifacts found:",
        found_artifacts,
        "/",
        len(
            artifact_status
        ),
    )

    print(
        "Candidate project files:",
        len(
            candidate_files
        ),
    )

    print(
        "Training/config matches:",
        len(
            source_matches
        ),
    )

    print()

    print(
        "Full report:"
    )

    print(
        " ",
        OUTPUT_PATH
    )

    print()

    print("=" * 80)


if __name__ == "__main__":
    main()