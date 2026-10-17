# MD7-R2-TELL-C1 preparation

Status: **prepared only — not trained, selected, promoted, or deployed**.

## Purpose

Teach the boundary that continued difficulty after exactly one meaningful scaffold normally remains `focus`, while persistent unresolved difficulty after two distinct meaningful scaffolds may become `telling`.

## Correction dataset audit

- Rows: **2,400** (`focus=1,200`, `telling=1,200`), all `split=train`.
- Matched families: **1,200**, each containing one one-scaffold hard negative and one two-scaffold positive on the same problem.
- Exact problem+history duplicates: **0**.
- Exact history duplicates: **0**.
- Exact overlap with frozen 48-case diagnostic: **0**.
- Exact overlap with original synthetic validation/final-test/challenge: **0**.
- Focus rows containing a direct-answer request: **0**.
- Model-text artificial label-token occurrences: **0**.
- Real learner identifiers: **0**.

The frozen diagnostic was used only after generation for exact-overlap auditing. Its cases were not copied, individually paraphrased, or mechanically transformed.

## Exact training pool

| component | rows | fraction |
| --- | ---: | ---: |
| Original telling-calibration synthetic train | 6,000 | 50% |
| New matched correction train | 2,400 | 20% |
| MathDial train replay | 3,600 | 30% |
| Total | 12,000 | 100% |

Overall synthetic/MathDial ratio is exactly **70%/30%**. MathDial replay quota is `generic=864, probing=824, focus=1,315, telling=597`. Resulting class counts are `generic=2,364, probing=2,324, focus=4,015, telling=3,297`.

## Kaggle execution

1. Create/attach the `md7-r2-tell-c1-preparation-v1` dataset containing the six root files listed in `kaggle_upload_manifest.json`.
2. Separately attach `mathdial-move-selection`, containing `train.jsonl` and `validation.jsonl`. Its mounted `test.jsonl` is never opened or used.
3. Separately attach the `MD7-R1 - AdaptMath` notebook output containing the complete frozen epoch-3 Hugging Face export (`model.safetensors`, config, and tokenizer files).
4. Open `MD7-R2-TELL-C1-kaggle.ipynb`, enable a GPU, and run top to bottom. The notebook discovers these inputs by exact SHA-256 rather than fixed Kaggle owner paths.
5. The notebook blocks training unless frozen MD7-R1 reproduces the exact MathDial validation metrics.
6. Download `md7_r2_tell_c1_epoch_candidates.zip` after all three epochs finish.
7. Do not choose or promote an epoch in the notebook. Run the unchanged frozen offline diagnostic on all three exports.

The package intentionally excludes MathDial test, MRBench V3 test, the original final synthetic test, and all telling-calibration Epoch 1/2/3 models.

## Reproduction

`generate_package.py` deterministically regenerates the correction data and derived package from seed 42. It also rechecks all source and protected-artifact hashes before and after generation.
