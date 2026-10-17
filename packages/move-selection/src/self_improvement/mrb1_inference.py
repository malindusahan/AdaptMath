"""Offline production inference wrapper for the frozen MRB1 critic."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Final

from .md6_inference import format_conversation_history
from .turn_context_builder import MRB1_TASKS


MAX_LENGTH: Final[int] = 512
MAX_RESPONSE_TOKENS: Final[int] = 256
MRB1_LABELS: Final[tuple[str, ...]] = ("No", "To some extent", "Yes")


def build_mrb1_text_pair(
    conversation_history: Sequence[Mapping[str, object]],
    tutor_response: str,
) -> tuple[str, str]:
    """Apply the exact validated MRB1 prompt text from the training notebook."""

    if not isinstance(tutor_response, str):
        raise TypeError("tutor_response must be a string.")
    if not tutor_response.strip():
        raise ValueError("tutor_response cannot be empty or whitespace.")
    conversation_text = (
        "Conversation history:\n"
        + format_conversation_history(conversation_history)
    )
    response_text = (
        "Candidate tutor response:\n"
        + tutor_response
        + "\n\nEvaluate pedagogical quality:"
    )
    return conversation_text, response_text


def _load_model_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "frozen_mrb1_multitask_model",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen MRB1 model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenMRB1Inference:
    """Load frozen MRB1 locally and return four continuous expected scores."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.model_dir = (
            project_root / "models" / "frozen" / "mrb1"
            if model_dir is None
            else Path(model_dir)
        ).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Frozen MRB1 directory not found: {self.model_dir}")

        config_path = self.model_dir / "config.json"
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        if tuple(config_json.get("mrb_tasks", ())) != MRB1_TASKS:
            raise ValueError("Frozen MRB1 task order is incompatible.")
        if tuple(config_json.get("mrb_labels", ())) != MRB1_LABELS:
            raise ValueError("Frozen MRB1 label order is incompatible.")
        if config_json.get("architectures") != ["MultiTaskRoberta"]:
            raise ValueError("Frozen MRB1 architecture is incompatible.")

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer, RobertaConfig

        model_module = _load_model_module(
            self.model_dir / "mrb1_multitask_model.py"
        )
        model_class = getattr(model_module, "MultiTaskRoberta", None)
        if model_class is None:
            raise ImportError("Frozen MRB1 module does not define MultiTaskRoberta.")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        config = RobertaConfig.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = model_class.from_pretrained(
            self.model_dir,
            config=config,
            local_files_only=True,
        )
        self.model.to("cpu")
        self.model.eval()
        if int(self.model.num_tasks) != len(MRB1_TASKS):
            raise ValueError("Loaded MRB1 model has the wrong number of tasks.")
        if int(self.model.num_classes) != len(MRB1_LABELS):
            raise ValueError("Loaded MRB1 model has the wrong number of labels.")

        self._pair_special_tokens = self.tokenizer.num_special_tokens_to_add(
            pair=True
        )
        if self._pair_special_tokens != 4:
            raise ValueError("Frozen MRB1 tokenizer must use four pair special tokens.")
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
            raise ValueError("Frozen MRB1 tokenizer is missing BOS/EOS tokens.")

    def _encode_no_special(self, text: str) -> list[int]:
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        return [int(value) for value in result]

    @staticmethod
    def _head_tail(token_ids: list[int], max_tokens: int) -> list[int]:
        if len(token_ids) <= max_tokens:
            return token_ids
        head = max_tokens // 2
        tail = max_tokens - head
        return token_ids[:head] + token_ids[-tail:]

    def _encode_pair(
        self,
        conversation_text: str,
        response_text: str,
    ) -> dict[str, object]:
        content_budget = MAX_LENGTH - self._pair_special_tokens
        history_ids = self._encode_no_special(conversation_text)
        full_response_ids = self._encode_no_special(response_text)
        response_ids = self._head_tail(
            full_response_ids,
            min(MAX_RESPONSE_TOKENS, content_budget),
        )
        history_budget = content_budget - len(response_ids)
        history_ids = history_ids[-history_budget:] if history_budget > 0 else []
        bos_id = int(self.tokenizer.bos_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        input_ids = (
            [bos_id]
            + history_ids
            + [eos_id, eos_id]
            + response_ids
            + [eos_id]
        )
        if len(input_ids) > MAX_LENGTH:
            raise RuntimeError("Internal MRB1 token packing exceeded max_length.")
        return {
            "input_ids": self._torch.tensor([input_ids], dtype=self._torch.long),
            "attention_mask": self._torch.ones(
                (1, len(input_ids)),
                dtype=self._torch.long,
            ),
        }

    def score_response(
        self,
        conversation_history: Sequence[Mapping[str, object]],
        tutor_response: str,
    ) -> dict[str, float]:
        """Score one completed tutor response with the frozen four-head critic."""

        conversation_text, response_text = build_mrb1_text_pair(
            conversation_history,
            tutor_response,
        )
        encoded = self._encode_pair(conversation_text, response_text)
        with self._torch.inference_mode():
            output = self.model(**encoded)
            logits = output.logits
            if tuple(logits.shape) != (1, len(MRB1_TASKS), len(MRB1_LABELS)):
                raise RuntimeError(
                    f"Frozen MRB1 returned logits with shape {tuple(logits.shape)}."
                )
            probabilities = self._torch.softmax(logits, dim=-1)[0]
            continuous = 0.5 * probabilities[:, 1] + probabilities[:, 2]

        values = [float(value) for value in continuous.detach().cpu().tolist()]
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise RuntimeError("Frozen MRB1 returned invalid continuous scores.")
        return dict(zip(MRB1_TASKS, values, strict=True))


__all__ = (
    "FrozenMRB1Inference",
    "MAX_LENGTH",
    "MAX_RESPONSE_TOKENS",
    "MRB1_LABELS",
    "build_mrb1_text_pair",
)
