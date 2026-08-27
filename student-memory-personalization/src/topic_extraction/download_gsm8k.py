"""Download and persist the raw GSM8K dataset."""

from __future__ import annotations

from pathlib import Path
from datasets import load_dataset
import pandas as pd


OUTPUT_DIR = Path("data/topic_extraction/raw/gsm8k")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/gsm8k", "main")
    for split_name in ("train", "test"):
        df = pd.DataFrame(dataset[split_name])
        output_path = OUTPUT_DIR / f"{split_name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"{split_name}: {len(df)} rows -> {output_path}")
    print("GSM8K download complete.")


if __name__ == "__main__":
    main()
