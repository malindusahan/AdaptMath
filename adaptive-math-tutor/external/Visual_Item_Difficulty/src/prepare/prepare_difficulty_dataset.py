#!/usr/bin/env python3
"""Build the item-level difficulty dataset from Eedi Rasch item parameters.

Needs two files from the original NeurIPS 2020 Education Challenge release that
are not redistributed here (see data/README.md):

  <data-dir>/ori_data/train_data/train_task_3_4.csv   student response records
  $VDIFF_IMAGES (default <data-dir>/images)           question images

Writes item_manifest.csv, difficulty_minimal.csv, train.csv, test.csv and
split_summary.csv into <data-dir>.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-frac", default=0.8, type=float)
    parser.add_argument("--test-frac", default=0.2, type=float)
    return parser.parse_args()


def check_fractions(train_frac: float, test_frac: float) -> None:
    total = train_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")


def make_manifest(data_dir: Path) -> pd.DataFrame:
    rasch_path = data_dir / "rasch_item_params.csv"
    train_path = data_dir / "ori_data" / "train_data" / "train_task_3_4.csv"
    image_dir = Path(os.environ.get("VDIFF_IMAGES", data_dir / "images"))

    rasch = pd.read_csv(rasch_path)
    rasch["QuestionId"] = rasch["item"].str.extract(r"QuestionId_(\d+)").astype(int)
    rasch = rasch.rename(columns={"beta": "difficulty"})

    answers = pd.read_csv(train_path, usecols=["QuestionId", "IsCorrect"])
    stats = (
        answers.groupby("QuestionId")
        .agg(n_answers=("IsCorrect", "size"), correct_rate=("IsCorrect", "mean"))
        .reset_index()
    )
    stats["error_rate"] = 1.0 - stats["correct_rate"]

    manifest = rasch.merge(stats, on="QuestionId", how="left")
    manifest["image_path"] = manifest["QuestionId"].map(
        lambda qid: f"data/ori_data/images/{qid}.jpg"
    )

    missing_stats = manifest[manifest["n_answers"].isna()]
    if len(missing_stats) > 0:
        raise ValueError(f"{len(missing_stats)} Rasch items are missing answer stats")

    missing_images = [
        path for path in manifest["image_path"] if not (data_dir.parent / path).exists()
    ]
    if missing_images:
        raise ValueError(f"{len(missing_images)} manifest images are missing")

    if (manifest["n_answers"] < 200).any():
        raise ValueError("Rasch manifest contains items with fewer than 200 answers")

    columns = [
        "QuestionId",
        "image_path",
        "difficulty",
        "SE_beta",
        "n_answers",
        "correct_rate",
        "error_rate",
        "item",
    ]
    return manifest[columns].sort_values("QuestionId").reset_index(drop=True)


def add_splits(
    manifest: pd.DataFrame,
    seed: int,
    train_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    manifest = manifest.copy()
    manifest["difficulty_bin"] = pd.qcut(
        manifest["difficulty"], q=10, labels=False, duplicates="drop"
    )
    manifest["split"] = ""

    for _, group in manifest.groupby("difficulty_bin", sort=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_frac))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        manifest.loc[train_idx, "split"] = "train"
        manifest.loc[test_idx, "split"] = "test"

    return manifest.drop(columns=["difficulty_bin"]).sort_values("QuestionId")


def write_outputs(manifest: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "item_manifest.csv", index=False)
    manifest[["image_path", "difficulty"]].to_csv(
        out_dir / "difficulty_minimal.csv", index=False
    )
    for split in ["train", "test"]:
        split_df = manifest[manifest["split"] == split]
        split_df.to_csv(out_dir / f"{split}.csv", index=False)

    summary = manifest.groupby("split").agg(
        n=("QuestionId", "size"),
        difficulty_mean=("difficulty", "mean"),
        difficulty_std=("difficulty", "std"),
        difficulty_min=("difficulty", "min"),
        difficulty_max=("difficulty", "max"),
        n_answers_min=("n_answers", "min"),
        n_answers_median=("n_answers", "median"),
    )
    summary.to_csv(out_dir / "split_summary.csv")

    # (data/README.md is maintained by hand and deliberately not written here)


def main() -> None:
    args = parse_args()
    check_fractions(args.train_frac, args.test_frac)
    manifest = make_manifest(args.data_dir)
    manifest = add_splits(
        manifest,
        args.seed,
        args.train_frac,
        args.test_frac,
    )
    out_dir = args.data_dir
    write_outputs(manifest, out_dir)
    print(f"wrote {len(manifest)} rows to {out_dir / 'item_manifest.csv'}")
    print(manifest["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
