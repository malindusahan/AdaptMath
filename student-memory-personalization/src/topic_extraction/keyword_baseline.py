"""Keyword and Rule-based Baseline Classifier for Canonical Topic Extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    normalize_token,
)


@dataclass(frozen=True)
class KeywordPrediction:
    """Prediction result from the keyword baseline classifier."""

    skill_id: str | None
    skill_code: str | None
    canonical_name: str | None
    display_name: str | None
    is_abstain: bool
    matched_alias: str | None
    confidence: float


class KeywordBaselineClassifier:
    """
    Deterministic rule and alias-matching classifier using canonical skill ontology.
    Extracts explicit mentions of skills, aliases, and mathematical surface forms.
    Abstains if no keywords match or if multiple distinct skills conflict.
    """

    def __init__(self, ontology_path: Path | str | None = None):
        self.ontology_path = Path(
            ontology_path if ontology_path is not None else DEFAULT_ONTOLOGY_JSON_PATH
        )
        self.skill_patterns: list[tuple[dict[str, Any], str, re.Pattern]] = []
        self._load_and_compile_patterns()

    def _load_and_compile_patterns(self):
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        compiled_list = []
        # Exclude overly generic single words from standalone trigger matching
        stop_aliases = {
            "table",
            "rate",
            "range",
            "mode",
            "mean",
            "median",
            "fraction of",
            "percent of",
        }

        for skill in ontology:
            candidates = set()
            candidates.add(skill["display_name"])
            for a in skill.get("aliases", []):
                # Skip technical database codes from natural phrase matching
                if (
                    a.lower().startswith("skill_")
                    or a.lower().startswith("assistments")
                    or a.isdigit()
                    or "::" in a
                    or "/" in a
                ):
                    continue
                candidates.add(a)

            # Sort candidate phrases by length descending (longest match priority)
            for cand in candidates:
                norm_cand = normalize_token(cand)
                if not norm_cand or norm_cand in stop_aliases:
                    continue
                # Build regex word-boundary pattern
                escaped = re.escape(norm_cand)
                pat = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                compiled_list.append((skill, norm_cand, pat))

        # Sort all global patterns by phrase length descending (longest phrase priority)
        compiled_list.sort(key=lambda x: len(x[1]), reverse=True)
        self.skill_patterns = compiled_list

    def predict(self, text: str) -> KeywordPrediction:
        """Predict canonical skill for input text or abstain."""
        if not text or not text.strip():
            return KeywordPrediction(
                skill_id=None,
                skill_code=None,
                canonical_name=None,
                display_name=None,
                is_abstain=True,
                matched_alias=None,
                confidence=0.0,
            )

        norm_text = normalize_token(text)
        matched_skills: dict[str, tuple[dict[str, Any], str]] = {}

        for skill, phrase_str, pattern in self.skill_patterns:
            if pattern.search(norm_text):
                c_name = skill["canonical_name"]
                if c_name not in matched_skills:
                    matched_skills[c_name] = (skill, phrase_str)

        # Unambiguous single skill match
        if len(matched_skills) == 1:
            skill, matched_phrase = next(iter(matched_skills.values()))
            return KeywordPrediction(
                skill_id=skill.get("skill_id"),
                skill_code=skill["skill_code"],
                canonical_name=skill["canonical_name"],
                display_name=skill["display_name"],
                is_abstain=False,
                matched_alias=matched_phrase,
                confidence=1.0,
            )

        # Disambiguate parent/child phrase containment (e.g. Absolute Value Advanced vs Absolute Value)
        if len(matched_skills) > 1:
            # Check if one match strictly subsumes all others in phrase length
            sorted_matches = sorted(
                matched_skills.values(), key=lambda x: len(x[1]), reverse=True
            )
            top_skill, top_phrase = sorted_matches[0]
            second_skill, second_phrase = sorted_matches[1]

            if top_phrase != second_phrase and second_phrase in top_phrase:
                return KeywordPrediction(
                    skill_id=top_skill.get("skill_id"),
                    skill_code=top_skill["skill_code"],
                    canonical_name=top_skill["canonical_name"],
                    display_name=top_skill["display_name"],
                    is_abstain=False,
                    matched_alias=top_phrase,
                    confidence=1.0,
                )

        # Multiple ambiguous matches or zero matches -> Abstain
        return KeywordPrediction(
            skill_id=None,
            skill_code=None,
            canonical_name=None,
            display_name=None,
            is_abstain=True,
            matched_alias=None,
            confidence=0.0,
        )

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Evaluate classifier performance metrics on a dataset split."""
        predictions = []
        y_true = df["canonical_name"].tolist()
        y_pred = []
        covered_indices = []

        for idx, row in df.iterrows():
            pred = self.predict(str(row["text"]))
            predictions.append(pred)
            if not pred.is_abstain:
                y_pred.append(pred.canonical_name)
                covered_indices.append(idx)
            else:
                y_pred.append("__ABSTAIN__")

        total_samples = len(df)
        covered_count = len(covered_indices)
        coverage = covered_count / total_samples if total_samples > 0 else 0.0
        abstention_rate = 1.0 - coverage

        # Overall accuracy (abstention counts as wrong)
        correct_total = sum(
            1 for yt, yp in zip(y_true, y_pred) if yt == yp and yp != "__ABSTAIN__"
        )
        accuracy = correct_total / total_samples if total_samples > 0 else 0.0

        # Covered-example accuracy
        if covered_count > 0:
            covered_true = [y_true[i] for i in covered_indices]
            covered_pred = [y_pred[i] for i in covered_indices]
            covered_accuracy = sum(
                1 for yt, yp in zip(covered_true, covered_pred) if yt == yp
            ) / covered_count
        else:
            covered_accuracy = 0.0

        # Macro F1 across all classes in ground truth
        unique_classes = sorted(list(set(y_true)))
        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=unique_classes,
                average="macro",
                zero_division=0,
            )
        )

        return {
            "total_samples": float(total_samples),
            "covered_samples": float(covered_count),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "coverage": float(coverage),
            "abstention_rate": float(abstention_rate),
            "covered_accuracy": float(covered_accuracy),
        }


def main():
    val_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "validation.csv"
    val_df = pd.read_csv(val_csv)

    classifier = KeywordBaselineClassifier()
    metrics = classifier.evaluate(val_df)

    print("=" * 60)
    print("Keyword / Rule Baseline Validation Results")
    print("=" * 60)
    print(f"Validation accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1:                  {metrics['macro_f1']:.4f}")
    print(f"Coverage:                  {metrics['coverage']:.4f} ({metrics['coverage']*100:.2f}%)")
    print(f"Abstention rate:           {metrics['abstention_rate']:.4f} ({metrics['abstention_rate']*100:.2f}%)")
    print(f"Covered-example accuracy:  {metrics['covered_accuracy']:.4f} ({metrics['covered_accuracy']*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
