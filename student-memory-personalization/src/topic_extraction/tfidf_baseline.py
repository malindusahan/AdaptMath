"""TF-IDF + Cosine Similarity Baseline Classifier for Canonical Topic Extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    normalize_token,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
CENTROIDS_PATH = ARTIFACTS_DIR / "tfidf_skill_centroids.joblib"


@dataclass(frozen=True)
class TfidfCandidate:
    """Candidate match with similarity score."""

    skill_code: str
    canonical_name: str
    display_name: str
    similarity: float


@dataclass(frozen=True)
class TfidfPrediction:
    """Prediction result from the TF-IDF cosine similarity classifier."""

    skill_id: str | None
    skill_code: str
    canonical_name: str
    display_name: str
    similarity: float
    margin: float
    top_candidates: list[TfidfCandidate]


class TfidfBaselineClassifier:
    """
    TF-IDF + Cosine Centroid Classifier.
    Builds L2-normalized class prototypes across all 111 canonical skills
    and predicts skills based on cosine similarity against prototypes.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            preprocessor=normalize_token,
            norm="l2",
        )
        self.skill_metadata: list[dict[str, Any]] = []
        self.skill_centroids: np.ndarray | None = None
        self.skill_name_to_idx: dict[str, int] = {}
        self.idx_to_skill_name: list[str] = []

    def fit(self, train_df: pd.DataFrame, ontology_path: Path | str | None = None):
        """Fit TF-IDF on training corpus and construct L2-normalized centroid vectors per skill."""
        # 1. Load ontology reference
        ont_path = Path(
            ontology_path if ontology_path is not None else DEFAULT_ONTOLOGY_JSON_PATH
        )
        with open(ont_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        self.skill_metadata = ontology
        self.idx_to_skill_name = [s["canonical_name"] for s in ontology]
        self.skill_name_to_idx = {
            s["canonical_name"]: i for i, s in enumerate(ontology)
        }

        # 2. Fit TF-IDF on train texts
        texts = train_df["text"].astype(str).tolist()
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # 3. Compute skill centroid vectors
        n_skills = len(ontology)
        n_features = tfidf_matrix.shape[1]
        raw_centroids = np.zeros((n_skills, n_features), dtype=np.float32)

        for skill_name, group in train_df.groupby("canonical_name"):
            if skill_name in self.skill_name_to_idx:
                skill_idx = self.skill_name_to_idx[skill_name]
                indices = group.index.tolist()
                skill_vectors = tfidf_matrix[indices]
                mean_vec = np.asarray(skill_vectors.mean(axis=0)).flatten()
                raw_centroids[skill_idx] = mean_vec

        # 4. L2-normalize centroids for direct dot-product cosine similarity
        self.skill_centroids = normalize(raw_centroids, norm="l2", axis=1)

    def predict(self, text: str, top_k: int = 5) -> TfidfPrediction:
        """Predict canonical skill and top candidate ranking for an input text."""
        if self.skill_centroids is None:
            raise RuntimeError("Classifier must be fitted before predict() is called.")

        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.skill_centroids)[0]

        ranked_indices = np.argsort(sims)[::-1]
        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        top_skill = self.skill_metadata[top_idx]
        top_sim = float(sims[top_idx])
        second_sim = float(sims[second_idx])
        margin = float(top_sim - second_sim)

        candidates = []
        for i in ranked_indices[:top_k]:
            sk = self.skill_metadata[i]
            candidates.append(
                TfidfCandidate(
                    skill_code=sk["skill_code"],
                    canonical_name=sk["canonical_name"],
                    display_name=sk["display_name"],
                    similarity=float(sims[i]),
                )
            )

        return TfidfPrediction(
            skill_id=top_skill.get("skill_id"),
            skill_code=top_skill["skill_code"],
            canonical_name=top_skill["canonical_name"],
            display_name=top_skill["display_name"],
            similarity=top_sim,
            margin=margin,
            top_candidates=candidates,
        )

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate accuracy, top-3 accuracy, macro/weighted F1, and similarity margins."""
        y_true = df["canonical_name"].tolist()
        y_pred = []
        top3_correct = 0
        top1_sims = []
        margins = []

        for _, row in df.iterrows():
            pred = self.predict(str(row["text"]), top_k=3)
            y_pred.append(pred.canonical_name)
            top1_sims.append(pred.similarity)
            margins.append(pred.margin)

            candidate_names = [c.canonical_name for c in pred.top_candidates]
            if row["canonical_name"] in candidate_names:
                top3_correct += 1

        total_samples = len(df)
        accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total_samples
        top3_accuracy = top3_correct / total_samples

        all_classes = sorted(list(self.skill_name_to_idx.keys()))
        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=all_classes,
                average="macro",
                zero_division=0,
            )
        )
        weighted_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=all_classes,
                average="weighted",
                zero_division=0,
            )
        )

        # Per-skill F1 scores
        precision, recall, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_classes, zero_division=0
        )
        per_skill_f1 = {
            cls: float(f) for cls, f in zip(all_classes, f1s)
        }

        return {
            "total_samples": total_samples,
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "mean_top1_similarity": float(np.mean(top1_sims)),
            "mean_margin": float(np.mean(margins)),
            "per_skill_f1": per_skill_f1,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save_artifacts(self, directory: Path | str | None = None):
        """Save vectorizer and centroid prototypes to disk."""
        target_dir = Path(directory if directory is not None else ARTIFACTS_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, target_dir / "tfidf_vectorizer.joblib")
        joblib.dump(
            {
                "centroids": self.skill_centroids,
                "metadata": self.skill_metadata,
                "name_to_idx": self.skill_name_to_idx,
            },
            target_dir / "tfidf_skill_centroids.joblib",
        )

    @classmethod
    def load_artifacts(
        cls, directory: Path | str | None = None
    ) -> TfidfBaselineClassifier:
        """Load fitted vectorizer and centroid prototypes from disk."""
        target_dir = Path(directory if directory is not None else ARTIFACTS_DIR)
        vectorizer = joblib.load(target_dir / "tfidf_vectorizer.joblib")
        centroid_data = joblib.load(target_dir / "tfidf_skill_centroids.joblib")

        clf = cls()
        clf.vectorizer = vectorizer
        clf.skill_centroids = centroid_data["centroids"]
        clf.skill_metadata = centroid_data["metadata"]
        clf.skill_name_to_idx = centroid_data["name_to_idx"]
        clf.idx_to_skill_name = [s["canonical_name"] for s in clf.skill_metadata]
        return clf


def main():
    train_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "validation.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("Fitting TF-IDF baseline on train split (N=3,186)...")
    clf = TfidfBaselineClassifier()
    clf.fit(train_df)

    print("Saving fitted artifacts to artifacts/topic_extractor/...")
    clf.save_artifacts()

    print("Evaluating on frozen validation split (N=641)...")
    results = clf.evaluate(val_df)

    print("=" * 60)
    print("TF-IDF + Cosine Similarity Validation Results")
    print("=" * 60)
    print(f"TF-IDF validation accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Macro F1:                   {results['macro_f1']:.4f}")
    print(f"Weighted F1:                {results['weighted_f1']:.4f}")
    print(f"Top-3 accuracy:             {results['top3_accuracy']:.4f} ({results['top3_accuracy']*100:.2f}%)")
    print(f"Mean top-1 similarity:      {results['mean_top1_similarity']:.4f}")
    print(f"Mean top-1/top-2 margin:    {results['mean_margin']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
