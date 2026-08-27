"""Pretrained MiniLM (all-MiniLM-L6-v2) Embedding Centroid Baseline Classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    normalize_token,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
PRETRAINED_CENTROIDS_NPZ = (
    ARTIFACTS_DIR / "minilm_pretrained_skill_centroids.npz"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class MiniLMCandidate:
    """Candidate match with cosine similarity score."""

    skill_code: str
    canonical_name: str
    display_name: str
    similarity: float


@dataclass(frozen=True)
class MiniLMPrediction:
    """Prediction result from MiniLM embedding centroid classifier."""

    skill_id: str | None
    skill_code: str
    canonical_name: str
    display_name: str
    similarity: float
    margin: float
    top_candidates: list[MiniLMCandidate]


class PretrainedMiniLMBaseline:
    """
    Pretrained sentence embedding classifier using all-MiniLM-L6-v2.
    Embeds student queries into 384-dimensional dense semantic space and matches
    against L2-normalized skill centroid prototypes.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.skill_metadata: list[dict[str, Any]] = []
        self.skill_centroids: np.ndarray | None = None
        self.idx_to_skill_name: list[str] = []
        self.skill_name_to_idx: dict[str, int] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        ontology_path: Path | str | None = None,
        batch_size: int = 64,
    ):
        """Encode training corpus and compute L2-normalized 384-dim skill centroids."""
        # 1. Load ontology
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

        # 2. Encode all training texts
        texts = train_df["text"].astype(str).tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # 3. Compute centroid per canonical skill
        n_skills = len(ontology)
        dim = embeddings.shape[1]
        raw_centroids = np.zeros((n_skills, dim), dtype=np.float32)

        for skill_name, group in train_df.groupby("canonical_name"):
            if skill_name in self.skill_name_to_idx:
                skill_idx = self.skill_name_to_idx[skill_name]
                indices = group.index.tolist()
                skill_vecs = embeddings[indices]
                mean_vec = np.mean(skill_vecs, axis=0)
                raw_centroids[skill_idx] = mean_vec

        # 4. L2-normalize centroids
        self.skill_centroids = normalize(raw_centroids, norm="l2", axis=1)

    def predict(self, text: str, top_k: int = 5) -> MiniLMPrediction:
        """Predict canonical skill for input text using cosine similarity."""
        if self.skill_centroids is None:
            raise RuntimeError("Classifier must be fitted or centroids loaded first.")

        query_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sims = np.dot(query_emb, self.skill_centroids.T)[0]
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
                MiniLMCandidate(
                    skill_code=sk["skill_code"],
                    canonical_name=sk["canonical_name"],
                    display_name=sk["display_name"],
                    similarity=float(sims[i]),
                )
            )

        return MiniLMPrediction(
            skill_id=top_skill.get("skill_id"),
            skill_code=top_skill["skill_code"],
            canonical_name=top_skill["canonical_name"],
            display_name=top_skill["display_name"],
            similarity=top_sim,
            margin=margin,
            top_candidates=candidates,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Evaluate accuracy, top-3 accuracy, macro/weighted F1, and similarity margins."""
        if self.skill_centroids is None:
            raise RuntimeError("Classifier must be fitted before evaluate().")

        texts = df["text"].astype(str).tolist()
        query_embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Cosine similarity matrix: (N_samples, 111)
        sim_matrix = np.dot(query_embs, self.skill_centroids.T)

        y_true = df["canonical_name"].tolist()
        y_pred = []
        top3_correct = 0
        top1_sims = []
        margins = []

        for i, row_true in enumerate(y_true):
            sims = sim_matrix[i]
            ranked = np.argsort(sims)[::-1]
            top_idx = ranked[0]
            second_idx = ranked[1] if len(ranked) > 1 else top_idx

            pred_name = self.idx_to_skill_name[top_idx]
            y_pred.append(pred_name)

            top1_sim = float(sims[top_idx])
            second_sim = float(sims[second_idx])
            top1_sims.append(top1_sim)
            margins.append(top1_sim - second_sim)

            top3_names = [self.idx_to_skill_name[idx] for idx in ranked[:3]]
            if row_true in top3_names:
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
        per_skill_f1 = {cls: float(f) for cls, f in zip(all_classes, f1s)}

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

    def save_centroids(self, filepath: Path | str | None = None):
        """Save computed centroids and metadata to .npz file."""
        target_path = Path(
            filepath if filepath is not None else PRETRAINED_CENTROIDS_NPZ
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_path,
            centroids=self.skill_centroids,
            canonical_names=np.array(self.idx_to_skill_name),
            metadata_json=json.dumps(self.skill_metadata),
        )

    def load_centroids(self, filepath: Path | str | None = None):
        """Load centroids and metadata from .npz file."""
        target_path = Path(
            filepath if filepath is not None else PRETRAINED_CENTROIDS_NPZ
        )
        data = np.load(target_path)
        self.skill_centroids = data["centroids"]
        self.idx_to_skill_name = list(data["canonical_names"])
        self.skill_metadata = json.loads(str(data["metadata_json"]))
        self.skill_name_to_idx = {
            name: i for i, name in enumerate(self.idx_to_skill_name)
        }


def main():
    train_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "validation.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Loading pretrained model: {MODEL_NAME}...")
    clf = PretrainedMiniLMBaseline()

    print(f"Encoding training split (N={len(train_df)}) and computing skill centroids...")
    clf.fit(train_df)

    print("Saving pretrained centroids artifact to artifacts/topic_extractor/...")
    clf.save_centroids()

    print(f"Evaluating on frozen validation split (N={len(val_df)})...")
    results = clf.evaluate(val_df)

    print("=" * 60)
    print("Pretrained MiniLM Validation Results")
    print("=" * 60)
    print(f"Pretrained MiniLM validation accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Macro F1:                              {results['macro_f1']:.4f}")
    print(f"Weighted F1:                           {results['weighted_f1']:.4f}")
    print(f"Top-3 accuracy:                        {results['top3_accuracy']:.4f} ({results['top3_accuracy']*100:.2f}%)")
    print(f"Mean top-1 similarity:                 {results['mean_top1_similarity']:.4f}")
    print(f"Mean top-1/top-2 margin:               {results['mean_margin']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
