"""Fine-tune SentenceTransformer all-MiniLM-L6-v2 on Educational Skill Pairs."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import DEFAULT_ONTOLOGY_JSON_PATH

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor"
FINETUNED_MODEL_DIR = ARTIFACTS_DIR / "minilm_finetuned"
FINETUNED_CENTROIDS_NPZ = (
    ARTIFACTS_DIR / "minilm_finetuned_skill_centroids.npz"
)
FINETUNE_HISTORY_JSON = ARTIFACTS_DIR / "finetune_history.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_skill_representations(
    ontology_path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate concise, informative skill description strings for contrastive targets."""
    ont_path = Path(
        ontology_path if ontology_path is not None else DEFAULT_ONTOLOGY_JSON_PATH
    )
    with open(ont_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    rep_map = {}
    for skill in ontology:
        name = skill["display_name"]
        cat = skill.get("category", "")
        desc = skill.get("description", "")
        # Build representation target text
        rep_text = f"{name} ({cat}): {desc}" if desc else f"{name} ({cat})"
        rep_map[skill["canonical_name"]] = {
            "metadata": skill,
            "rep_text": rep_text,
        }
    return rep_map


def compute_skill_centroids(
    model: SentenceTransformer,
    train_df: pd.DataFrame,
    ontology: list[dict[str, Any]],
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Compute L2-normalized 384-dimensional centroid embeddings for all 111 canonical skills."""
    canonical_names = [s["canonical_name"] for s in ontology]
    name_to_idx = {name: i for i, name in enumerate(canonical_names)}

    texts = train_df["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    n_skills = len(ontology)
    dim = embeddings.shape[1]
    raw_centroids = np.zeros((n_skills, dim), dtype=np.float32)

    for skill_name, group in train_df.groupby("canonical_name"):
        if skill_name in name_to_idx:
            idx = name_to_idx[skill_name]
            skill_vecs = embeddings[group.index.tolist()]
            raw_centroids[idx] = np.mean(skill_vecs, axis=0)

    norm_centroids = normalize(raw_centroids, norm="l2", axis=1)
    return norm_centroids, canonical_names


def evaluate_model(
    model: SentenceTransformer,
    centroids: np.ndarray,
    canonical_names: list[str],
    val_df: pd.DataFrame,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate accuracy, top-3 accuracy, macro/weighted F1 on validation split."""
    val_texts = val_df["text"].astype(str).tolist()
    query_embs = model.encode(
        val_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    sim_matrix = np.dot(query_embs, centroids.T)
    y_true = val_df["canonical_name"].tolist()
    y_pred = []
    top3_correct = 0
    top1_sims = []
    margins = []

    for i, row_true in enumerate(y_true):
        sims = sim_matrix[i]
        ranked = np.argsort(sims)[::-1]
        top_idx = ranked[0]
        second_idx = ranked[1] if len(ranked) > 1 else top_idx

        pred_name = canonical_names[top_idx]
        y_pred.append(pred_name)

        top1_sim = float(sims[top_idx])
        second_sim = float(sims[second_idx])
        top1_sims.append(top1_sim)
        margins.append(top1_sim - second_sim)

        top3_names = [canonical_names[idx] for idx in ranked[:3]]
        if row_true in top3_names:
            top3_correct += 1

    total_samples = len(val_df)
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total_samples
    top3_accuracy = top3_correct / total_samples

    all_classes = sorted(canonical_names)
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

    return {
        "total_samples": total_samples,
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mean_top1_similarity": float(np.mean(top1_sims)),
        "mean_margin": float(np.mean(margins)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_and_select_model(
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train sentence-transformers/all-MiniLM-L6-v2 and select best epoch checkpoint."""
    set_seed(seed)

    train_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "validation.csv"
    ont_path = DEFAULT_ONTOLOGY_JSON_PATH

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    with open(ont_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    skill_rep_map = build_skill_representations(ont_path)

    # Prepare InputExamples
    train_examples = []
    for _, row in train_df.iterrows():
        c_name = row["canonical_name"]
        target_text = skill_rep_map[c_name]["rep_text"]
        train_examples.append(
            InputExample(texts=[str(row["text"]), str(target_text)])
        )

    print(f"Loaded {len(train_examples)} training pairs across {len(ontology)} skills.")
    print("Initializing base model: sentence-transformers/all-MiniLM-L6-v2...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = MultipleNegativesRankingLoss(model)

    best_epoch = 0
    best_macro_f1 = -1.0
    best_metrics = {}
    history = []

    # Initial epoch 0 (pretrained) evaluation
    p_centroids, p_names = compute_skill_centroids(model, train_df, ontology)
    init_eval = evaluate_model(model, p_centroids, p_names, val_df)
    print(
        f"[Pretrained Epoch 0] Val Acc: {init_eval['accuracy']*100:.2f}%, "
        f"Macro F1: {init_eval['macro_f1']:.4f}, Top-3: {init_eval['top3_accuracy']*100:.2f}%"
    )
    history.append({"epoch": 0, "metrics": init_eval})

    # Epoch-by-epoch fine-tuning
    warmup_steps = int(len(train_dataloader) * 0.1)

    for epoch in range(1, epochs + 1):
        print(f"\n--- Training Epoch {epoch} / {epochs} ---")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=False,
        )

        centroids, names = compute_skill_centroids(model, train_df, ontology)
        epoch_eval = evaluate_model(model, centroids, names, val_df)
        print(
            f"[Epoch {epoch}] Val Acc: {epoch_eval['accuracy']*100:.2f}%, "
            f"Macro F1: {epoch_eval['macro_f1']:.4f}, Top-3: {epoch_eval['top3_accuracy']*100:.2f}%, "
            f"Margin: {epoch_eval['mean_margin']:.4f}"
        )

        history.append({"epoch": epoch, "metrics": epoch_eval})

        if epoch_eval["macro_f1"] > best_macro_f1:
            best_macro_f1 = epoch_eval["macro_f1"]
            best_epoch = epoch
            best_metrics = epoch_eval

            # Save best model checkpoint
            print(f"--> Best Macro F1 ({best_macro_f1:.4f}) achieved at Epoch {epoch}! Saving checkpoint...")
            FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(str(FINETUNED_MODEL_DIR))

            # Save best centroids
            np.savez_compressed(
                FINETUNED_CENTROIDS_NPZ,
                centroids=centroids,
                canonical_names=np.array(names),
                metadata_json=json.dumps(ontology),
            )

    # Save training history
    with open(FINETUNE_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "history": [
                    {
                        "epoch": h["epoch"],
                        "accuracy": h["metrics"]["accuracy"],
                        "macro_f1": h["metrics"]["macro_f1"],
                        "weighted_f1": h["metrics"]["weighted_f1"],
                        "top3_accuracy": h["metrics"]["top3_accuracy"],
                        "mean_top1_similarity": h["metrics"]["mean_top1_similarity"],
                        "mean_margin": h["metrics"]["mean_margin"],
                    }
                    for h in history
                ],
            },
            f,
            indent=2,
        )

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
    }


def main():
    results = train_and_select_model(
        epochs=4,
        batch_size=32,
        learning_rate=2e-5,
        seed=42,
    )

    best_m = results["best_metrics"]
    best_ep = results["best_epoch"]

    print("\n" + "=" * 60)
    print("Fine-Tuned MiniLM Validation Results")
    print("=" * 60)
    print(f"Fine-tuned MiniLM validation accuracy: {best_m['accuracy']:.4f} ({best_m['accuracy']*100:.2f}%)")
    print(f"Macro F1:                              {best_m['macro_f1']:.4f}")
    print(f"Weighted F1:                           {best_m['weighted_f1']:.4f}")
    print(f"Top-3 accuracy:                        {best_m['top3_accuracy']:.4f} ({best_m['top3_accuracy']*100:.2f}%)")
    print(f"Mean top-1 similarity:                 {best_m['mean_top1_similarity']:.4f}")
    print(f"Mean margin:                           {best_m['mean_margin']:.4f}")
    print(f"Best epoch:                            {best_ep}")
    print("=" * 60)


if __name__ == "__main__":
    main()
