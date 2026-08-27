# Representing Visual Evidence for Item Difficulty Prediction

Code, labels, and per-item predictions for our study of **how visual evidence should be
represented when predicting the difficulty of mathematics assessment items**.

Given a math item that contains a diagram, you can give a model the question text alone
(**Q**), a generated language description of the visual component (**visual
textualization**, Q+D), or the original item image (**image-native modeling**, I+Q). We
compare all three against the same response-calibrated target, item split, and evaluation
protocol.

Target: the Rasch difficulty parameter β of 725 Eedi items, estimated from real student
responses. Fixed split: 580 train / 145 test.

| Interface | Best system | Test RMSE |
|---|---|---|
| Question text (Q) | Llama-3.1-8B, LoRA attn+MLP | 0.517 |
| Visual textualization (Q+D) | Llama-3.1-8B, LoRA attn, D from Qwen2.5-VL-7B | 0.506 |
| Image-native (I+Q) | Qwen2.5-VL-7B, LoRA attn+MLP | 0.497 |

RMSE is the mean over three seeds {17, 42, 2026}. The paired bootstrap intervals between
these three leaders all include zero — they are a competitive group, not a ranking. The
matched, within-model comparisons are where the consistent effects are.

## What is here

```
data/          items, Rasch labels, fixed split, CV folds, Q transcriptions, D descriptions
src/           training and inference code for the two predictor families
analysis/      the scripts that turn predictions into the paper's tables
results/       per-item test predictions for every run, plus the derived analyses
```

Trained checkpoints (~9 GB of LoRA adapters and regression heads) and the Eedi question
images are **not** redistributed. See [data/README.md](data/README.md) for how to obtain
the images.

## Reproducing the numbers without a GPU

Every result in the paper is recomputable from the shipped per-item predictions. No model
weights, no images, no GPU.

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src

python analysis/aggregate_results.py            # per-seed + seed-ensemble RMSE, paired bootstrap CIs
python analysis/analyze_visual_taxonomy.py      # item-type breakdown of the three headline systems
python analysis/analyze_interface_tradeoffs.py  # Q+D vs I+Q win/loss, subgroups, workflow cost
```

Or, directly:

```python
import glob, numpy as np, pandas as pd
runs = sorted(glob.glob("results/main/vlm/Qwen_Qwen2.5-VL-7B-Instruct_IQ_ablMlp/seed*/test_predictions.csv"))
r = [np.sqrt(((d.pred_beta - d.gold_beta) ** 2).mean()) for d in map(pd.read_csv, runs)]
print(np.mean(r))   # 0.4966
```

Each `test_predictions.csv` has one row per test item with `item_id`, `gold_beta`,
`pred_beta`, residual and absolute/squared error, the `has_figure` flag, Q and D character
lengths, plus SHA256 hashes of the split and of each item's input text so a rerun can be
verified against these files.

## Reproducing the training runs

The two predictor scripts share the same staged protocol:

- `--run_mode cv` — evaluate one recipe on the frozen 5-fold split of the 580 training
  items. Recipe selection happens here and never touches the test set. The fold file
  carries a SHA256 of the split and the script aborts on a mismatch.
- `--run_mode final` — retrain on all 580 items for a fixed epoch count, predict the 145
  test items, save predictions and weights. Run once per seed in {17, 42, 2026}.

The exact configuration behind each shipped run is recorded in its
`results.json` under `provenance.args`. The three headline systems:

```bash
export PYTHONPATH=$PWD/src
export VDIFF_IMAGES=/path/to/eedi/images      # image-native route only
export HF_TOKEN=...                           # gated checkpoints (Llama)

# Q — question text alone
CUDA_VISIBLE_DEVICES=0 python src/llm_lora_regression.py \
    --model_name meta-llama/Llama-3.1-8B --input_mode text \
    --lora_targets attn_mlp --out_suffix _ablMlp \
    --run_mode final --lora_r 16 --lr 1e-4 --epochs 6 --seed 42

# Q+D — visual textualization
CUDA_VISIBLE_DEVICES=0 python src/llm_lora_regression.py \
    --model_name meta-llama/Llama-3.1-8B --input_mode text_vd \
    --vd_json data/descriptions/Qwen_Qwen2.5-VL-7B-Instruct.json --vd_tag qwen25vl7b \
    --run_mode final --lora_r 16 --lr 1e-4 --epochs 6 --seed 42

# I+Q — image-native
CUDA_VISIBLE_DEVICES=0 python src/vlm_lora_regression.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct --input_mode image_text --text_field text \
    --lora_targets attn_mlp --out_suffix _ablMlp \
    --run_mode final --lora_r 8 --lr 5e-5 --epochs 8 --seed 42
```

Regenerating the visual textualizations, and the test-time image interventions:

```bash
# D from an open VLM -> data/descriptions/<model_slug>.json
CUDA_VISIBLE_DEVICES=0 python src/generate_descriptions.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct

# blank / shuffled image at test time, reusing a trained checkpoint
python src/eval_image_intervention.py \
    --run_dir results/main/vlm/Qwen_Qwen2.5-VL-7B-Instruct_IQ_ablMlp/seed42 \
    --intervention blank
```

A single A100-80GB fits every model used here. The image-native runs are the expensive
ones: ~1,600 s per final run against ~190 s for Q+D, because descriptions are generated
once and cached while images are reprocessed every epoch.

## Layout of `results/`

| Path | Contents |
|---|---|
| `main/llm`, `main/vlm` | Final three-seed runs. One directory per system, `seed{17,42,2026}/` inside, each with `results.json` (metrics + full provenance) and `test_predictions.csv`. The image-intervention runs also carry `intervention_{blank,shuffle}.csv`. |
| `model_selection/llm`, `model_selection/vlm` | Frozen-fold CV results for every recipe considered, i.e. the search that picked the final configurations. |
| `analysis/` | Derived tables: item-type taxonomy, Q+D vs I+Q win/loss and subgroups, workflow cost. Regenerated by the scripts in `analysis/`. |
| `run_timings.csv` | Wall-clock duration and exit code of every queued training job, backing the workflow-cost comparison. |
| `adapter_param_counts.json` | Trainable-parameter counts precomputed from the checkpoints, so the cost table works without them. |
| `other_baselines/` | See below. |

Run directories are named `<model>_<route>[_abl*]`, where the route suffix is `text` /
`text_vd` / `vd_<textualizer>` for text models and `I` / `IQ` / `IQD` for VLMs;
`_ablMlp` marks attn+MLP LoRA, `_ablVis` also adapting the vision encoder, `_ablPoolT`
pooling over text tokens only, `_ep10` a longer schedule.

### `results/other_baselines/`

The paper also maps a broader design space: fine-tuned text encoders, frozen text/vision/VLM
features, scalar generation, full fine-tuning, and late fusion. Those metrics are included
here for completeness, but **the code that produced them is not part of this release** —
this repository ships only the LLM and VLM regression pipelines that back the headline
comparison. Treat those files as a record, not as something you can rerun from this repo.

## Notes on the setup

- **Q is not raw OCR.** It is a GPT-5.5 two-pass extraction from the item image that was
  then manually verified, so the question-text baseline is deliberately a strong one. D,
  by contrast, is never manually verified — it is the experimental variable.
- **D is not a caption.** The prompt asks for the problem-relevant notation and spatial
  relations a student would need, and explicitly forbids solving or restating the question.
- **The image-native predictor really does use its image.** Replacing it with a blank
  image at test time moves RMSE 0.497 → 0.945; pairing each item with another item's
  image gives 0.964 and drops Spearman from 0.785 to 0.112. This shows dependence on the
  full-item image, which also carries rendered text and layout — it does not isolate
  dependence on the additional visual component alone.
- **Recipes were selected on training data only**, via the frozen folds in
  `data/cv_folds.json`, and fixed before the three-seed test evaluation.

## Citation and licensing

Citation details will be added on publication.

Code is released for research use. The labels, splits, and predictions are our own output.
The Q transcriptions and D descriptions are derived from Eedi item images and are governed
by the terms of the original NeurIPS 2020 Education Challenge release
([Wang et al., 2020](https://arxiv.org/abs/2007.12061)); the images themselves are not
redistributed here.

Content-based difficulty estimates are meant as provisional support for item development
and cold-start decisions. They do not replace calibration from real student responses or
expert review, and they should not be assumed to transfer to other curricula, languages,
or learner populations.
