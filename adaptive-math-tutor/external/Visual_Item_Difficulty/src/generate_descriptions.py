#!/usr/bin/env python3
"""Visual textualization: generate the figure description D with an open VLM.

Captions only the items that carry an additional visual component (non-empty
`visual_description` in items.csv) and writes {QuestionId: description} to
data/descriptions/<model_slug>.json. That file is then passed to
llm_lora_regression.py via --vd_json to build the Q+D input.

Usage:
  python generate_descriptions.py --model_name Qwen/Qwen2.5-VL-7B-Instruct
  python generate_descriptions.py --model_name OpenGVLab/InternVL2_5-4B --max_items 2  # smoke
"""
import os, sys, json, argparse
# Set HF_HOME in your shell to control where model weights are cached.
import torch
from PIL import Image

from vdiff_utils import load_data, DATA_DIR, resolve_image
from vlm_loaders import (detect_model_family, load_model_and_processor,
                              generate_prediction)

CAPTION_PROMPT = (
    "This image is from a math assessment item. In 1-3 sentences, describe the "
    "figure/diagram only: the visual elements (shapes, graphs, axes, geometry, "
    "labels, numbers) that a student would need to read to answer. Be concise and "
    "factual. Do NOT solve the question and do NOT restate the question text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_items", type=int, default=0, help=">0 = smoke on first N figure items")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = detect_model_family(args.model_name)
    slug = args.model_name.replace("/", "_")
    print(f"[describe] {args.model_name} family={family} device={device}", flush=True)

    train_df, test_df = load_data()
    import pandas as pd
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    # figure items = original (GPT-5.5) visual_description non-empty
    has_fig = all_df["visual_description"].fillna("").astype(str).str.strip() != ""
    fig_df = all_df[has_fig].reset_index(drop=True)
    if args.max_items > 0:
        fig_df = fig_df.iloc[: args.max_items].reset_index(drop=True)
    print(f"[describe] captioning {len(fig_df)} figure items "
          f"({'SMOKE' if args.max_items else 'full'})", flush=True)

    model, processor = load_model_and_processor(args.model_name, family)
    model.eval()

    out = {}
    for i, row in fig_df.iterrows():
        img = Image.open(resolve_image(row["image_path"])).convert("RGB")
        try:
            cap = generate_prediction(model, processor, family, img, CAPTION_PROMPT,
                                      device, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            cap = ""
            print(f"  [warn] item {row['QuestionId']} failed: {e}", flush=True)
        out[int(row["QuestionId"])] = cap.strip()
        if i < 3 or i % 50 == 0:
            print(f"  [{i+1}/{len(fig_df)}] Qid={row['QuestionId']}: {cap.strip()[:120]}", flush=True)

    out_dir = os.path.join(DATA_DIR, "descriptions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{slug}{args.out_suffix}.json")
    nonempty = sum(1 for v in out.values() if v)
    payload = {"model": args.model_name, "family": family,
               "n_figure_items": len(out), "n_nonempty": nonempty,
               "max_new_tokens": args.max_new_tokens, "descriptions": out}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[describe] saved {nonempty}/{len(out)} non-empty -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
