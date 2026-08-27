"""Offline production inference wrapper for the frozen MD6 selector."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Final

from .context_builder import MOVE_ORDER


MAX_LENGTH: Final[int] = 512
EXPECTED_ID2LABEL: Final[dict[int, str]] = dict(enumerate(MOVE_ORDER))
EXPECTED_LABEL2ID: Final[dict[str, int]] = {
    label: index for index, label in enumerate(MOVE_ORDER)
}


def format_conversation_history(
    conversation_history: Sequence[Mapping[str, object]],
) -> str:
    """Format host-memory turns exactly as the validated MD6 audit does."""

    if isinstance(conversation_history, (str, bytes)) or not isinstance(
        conversation_history,
        Sequence,
    ):
        raise TypeError("conversation_history must be a sequence of turn mappings.")
    if not conversation_history:
        return "[No previous conversation]"

    lines: list[str] = []
    for index, turn in enumerate(conversation_history):
        if not isinstance(turn, Mapping):
            raise TypeError(f"conversation_history[{index}] must be a mapping.")
        if "user" not in turn or "text" not in turn:
            raise ValueError(
                f"conversation_history[{index}] requires 'user' and 'text'."
            )
        user = turn["user"]
        text = turn["text"]
        if not isinstance(user, str) or not user.strip():
            raise ValueError(
                f"conversation_history[{index}].user must be a non-empty string."
            )
        if not isinstance(text, str):
            raise TypeError(
                f"conversation_history[{index}].text must be a string."
            )
        lines.append(f"{user}: {text}")
    return "\n".join(lines)


def build_paired_input(
    problem: str,
    conversation_history: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    """Build the exact validated frozen-MD6 pair without changing text."""

    if not isinstance(problem, str):
        raise TypeError("problem must be a string.")
    if not problem.strip():
        raise ValueError("problem cannot be empty or whitespace.")

    sequence_a = f"Problem:\n{problem}"
    sequence_b = (
        "Conversation:\n"
        f"{format_conversation_history(conversation_history)}"
        "\n\n"
        "Next teacher pedagogical move:"
    )
    return sequence_a, sequence_b


class FrozenMD6Inference:
    """Load frozen MD6 locally and return one four-class probability vector."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.model_dir = (
            project_root / "models" / "frozen" / "md6"
            if model_dir is None
            else Path(model_dir)
        ).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Frozen MD6 directory not found: {self.model_dir}")

        config_path = self.model_dir / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Frozen MD6 config not found: {config_path}"
            ) from exc
        observed_id2label = {
            int(index): label for index, label in config.get("id2label", {}).items()
        }
        if observed_id2label != EXPECTED_ID2LABEL:
            raise ValueError(
                f"Frozen MD6 class order mismatch: {observed_id2label!r}."
            )
        if config.get("label2id") != EXPECTED_LABEL2ID:
            raise ValueError("Frozen MD6 label2id mapping is incompatible.")
        if config.get("architectures") != ["RobertaForSequenceClassification"]:
            raise ValueError("Frozen MD6 architecture is incompatible.")

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.to("cpu")
        self.model.eval()
        if self.model.config.id2label != EXPECTED_ID2LABEL:
            raise ValueError("Loaded MD6 model class order is incompatible.")

    def predict_probabilities(
        self,
        problem: str,
        conversation_history: Sequence[Mapping[str, object]],
    ) -> dict[str, float]:
        """Run one no-gradient CPU prediction using only frozen local assets."""

        sequence_a, sequence_b = build_paired_input(
            problem,
            conversation_history,
        )
        encoded = self.tokenizer(
            sequence_a,
            sequence_b,
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            output = self.model(**encoded)
            vector = self._torch.softmax(output.logits, dim=-1)[0]
        values = [float(value) for value in vector.detach().cpu().tolist()]
        if len(values) != len(MOVE_ORDER):
            raise RuntimeError(f"Frozen MD6 returned {len(values)} probabilities.")
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise RuntimeError("Frozen MD6 returned invalid probabilities.")
        if abs(sum(values) - 1.0) > 1e-5:
            raise RuntimeError("Frozen MD6 probabilities do not sum to one.")
        return dict(zip(MOVE_ORDER, values, strict=True))


__all__ = (
    "FrozenMD6Inference",
    "MAX_LENGTH",
    "build_paired_input",
    "format_conversation_history",
)
