# MD0 to MD13 research map

Audit date: 2026-10-16

Scope: read-only dependency and research audit of `packages/move-selection`. No file was deleted or moved. A path marked `NOT FOUND` was not found anywhere under `packages/move-selection`; a Kaggle path is reported only when it is recorded in an executed notebook or result artifact and is not evidence that the checkpoint exists locally.

## Authoritative datasets

All MD0-MD13 supervised experiments use the same leakage-checked processed MathDial split unless a row below says otherwise:

| Use | Local path | Rows | SHA-256 |
|---|---|---:|---|
| Train | `data/processed/mathdial/train.jsonl` | 14,905 | `2d0c9605d86d99c956b62466fcea6cfc0dbfed7d4ebfe2d39229689a5c10af76` |
| Validation/model selection | `data/processed/mathdial/validation.jsonl` | 1,850 | `225e6b8f671bf1e335af175343722aa9ac2a603a96d67bcb5220be1e92c0b2b6` |
| Protected held-out test | `data/processed/mathdial/test.jsonl` | 1,852 | `3626d88a2df58abd96108a50078c75650bfdcf20bc08710c07b2813e4a477847` |
| Combined processed corpus | `data/processed/mathdial/all_examples.jsonl` | 18,607 | `ecb3ea5a5a5d13271fb97f92228ff1f578e7f29f16e817e27134f59831c12ba9` |

The processed data was produced from `data/raw/mathdial/train.jsonl` (2,262 dialogues) and `data/raw/mathdial/test.jsonl` (599 dialogues). The notebooks reference the Kaggle copies `/kaggle/input/datasets/malindus/mathdial-move-selection/{train,validation,test}.jsonl`; those correspond to the local processed files above. MD0-MD5 and MD7-MD13 did not use the protected test split. MD6 is the only model with a recorded one-time held-out test evaluation.

## Model-by-model inventory

### MD0 — majority-class baseline

- TRAINING_NOTEBOOK: `NOT FOUND`
- TRAINING_SCRIPT: `src/baselines/mathdial_baselines.py`
- EVALUATION_NOTEBOOK: `NOT FOUND`
- EVALUATION_SCRIPT: `src/baselines/mathdial_baselines.py`
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: validation accuracy `0.3664864865`; Macro-F1 `0.1340981013`; majority class `focus`
- FINAL_CHECKPOINT: not applicable; this is a deterministic count baseline
- RELATED_RESULT_FOLDER: `results/baselines/mathdial_majority_validation.json`
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD1 — TF-IDF + logistic regression

- TRAINING_NOTEBOOK: `NOT FOUND`
- TRAINING_SCRIPT: `src/baselines/mathdial_baselines.py`
- EVALUATION_NOTEBOOK: `NOT FOUND`
- EVALUATION_SCRIPT: `src/baselines/mathdial_baselines.py`
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: validation accuracy `0.4745945946`; Macro-F1 `0.4057544666`
- FINAL_CHECKPOINT: `NOT FOUND`; the script saves metrics, not a serialized estimator
- RELATED_RESULT_FOLDER: `results/baselines/mathdial_tfidf_logreg_validation.json`
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD2 — DistilRoBERTa, ordinary cross-entropy

- TRAINING_NOTEBOOK: `notebooks/supervised/md2-distilroberta-move-selector (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND` (training code is in the notebook)
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `src/baselines/analyze_md2_md3.py` and `src/baselines/analyze_md2_md3_md4.py` provide later comparisons; no separate primary evaluator was found
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: epoch 3 validation accuracy `0.5081081081`; Macro-F1 `0.4669758202`
- FINAL_CHECKPOINT: `NOT FOUND`; notebook uses `save_strategy="no"` and evaluates the final in-memory model
- RELATED_RESULT_FOLDER: `results/baselines/md2_distilroberta_validation.json` and `md2_validation_predictions.jsonl`
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD3 — ModernBERT long-context classifier

- TRAINING_NOTEBOOK: `NOT FOUND` (verified)
- TRAINING_SCRIPT: `NOT FOUND` (verified)
- EVALUATION_NOTEBOOK: `NOT FOUND`
- EVALUATION_SCRIPT: `src/baselines/analyze_md2_md3.py`, `src/baselines/analyze_md2_md3_md4.py`, and historical selector-recovery validators consume saved predictions
- TRAINING_DATASET: the surviving result declares the same processed MathDial train split
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: validation accuracy `0.4843243243`; Macro-F1 `0.4560921971`; long-context (>512) Macro-F1 `0.3999735543`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; the result JSON records `/kaggle/working/md3_modernbert_checkpoints/checkpoint-2796`
- RELATED_RESULT_FOLDER: `results/baselines/md3_modernbert_validation.json`, `md3_modernbert_predictions.jsonl`; duplicate historical evidence also exists in `artifacts/historical_selector_outputs/md3/`
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes; preserve the result and predictions because the actual training notebook is missing

### MD4 — DistilRoBERTa, inverse-frequency weighted cross-entropy

- TRAINING_NOTEBOOK: `notebooks/supervised/md4-distilroberta-weighted-cross-entropy (2).ipynb`
- TRAINING_SCRIPT: `NOT FOUND` (training code is in the notebook)
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `src/baselines/analyze_md2_md3_md4.py` plus the notebook's three-checkpoint recovery evaluation
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: recovered epoch 3 accuracy `0.4897297297`; Macro-F1 `0.4832109213`. The older saved baseline JSON records accuracy `0.4772972973`, Macro-F1 `0.4720959135`; the notebook explicitly audits this difference.
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md4_weighted_distilroberta/checkpoint-2796` and recovered archives under `/kaggle/working/md4_recovery/`
- RELATED_RESULT_FOLDER: `results/baselines/`; MD4 duplicate historical predictions also exist in `artifacts/historical_selector_outputs/md4/`
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD5 — DistilRoBERTa, softened weighted CE (alpha 0.50)

- TRAINING_NOTEBOOK: `notebooks/supervised/md5-distilroberta-soft-weighted-ce.ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: best epoch 3 accuracy `0.4978378378`; Macro-F1 `0.4792721544`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/tune01_soft_weights_alpha050/checkpoint-2796` and `/kaggle/working/tune01_results/tune01_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no local standalone MD5 result folder; executed notebook output is the authoritative evidence
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD6 — DistilRoBERTa, softened weighted CE (alpha 0.75)

- TRAINING_NOTEBOOK: `notebooks/supervised/md6-distilroberta-soft-weighted-ce-alpha075 (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: training notebook plus `notebooks/evaluation/md6-held-out-test.ipynb`
- EVALUATION_SCRIPT: MD6 diagnostic scripts under `experiments/` are auxiliary; the primary validation/test evaluation is in the notebooks
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: processed MathDial held-out test, 1,852 examples, consumed once by the held-out-test notebook
- FINAL_METRICS: validation accuracy `0.5016216216`, Macro-F1 `0.4876845727`; held-out test accuracy `0.5194384449`, Macro-F1 `0.5093889929`, weighted-F1 `0.5222339194`
- FINAL_CHECKPOINT: `models/frozen/md6/` (not the normal live selector); notebook source checkpoint `/kaggle/working/md6_soft_weights_alpha075/checkpoint-1864`
- RELATED_RESULT_FOLDER: `results/self_improvement/md6_*` contains later diagnostics, not primary MD6 training evidence
- REQUIRED_FOR_RUNTIME: rollback-only (`-MD6Rollback`), not the normal live path
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD7 — DistilRoBERTa, targeted telling weight

- TRAINING_NOTEBOOK: `notebooks/supervised/md7-distilroberta-targeted-telling-weight (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET: processed MathDial train, 14,905 examples
- VALIDATION_DATASET: processed MathDial validation, 1,850 examples
- TEST_DATASET: `NOT USED`
- FINAL_METRICS: best epoch 3 accuracy `0.4859459459`; Macro-F1 `0.4751072388`
- FINAL_CHECKPOINT: original MD7 local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md7_targeted_telling_weight/checkpoint-2796` and `/kaggle/working/md7_results/md7_best_model.tar.gz`
- RELATED_RESULT_FOLDER: executed notebook is primary evidence. The current runtime's later MD7-R2-TELL-C1 derivation is documented separately below and is not the original MD7 checkpoint.
- REQUIRED_FOR_RUNTIME: original MD7 no; the derived MD7-R2-TELL-C1 Epoch 2 model is the current runtime selector
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD8 — DistilRoBERTa, alpha 0.75, learning rate 1.5e-5

- TRAINING_NOTEBOOK: `notebooks/supervised/md8-distilroberta-alpha075-lr1p5e5 (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: best epoch 3 accuracy `0.4994594595`; Macro-F1 `0.4874127108`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md8_alpha075_lr1p5e5/checkpoint-2796` and `/kaggle/working/md8_results/md8_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD9 — MD8 recipe, four epochs

- TRAINING_NOTEBOOK: `notebooks/supervised/md9-distilroberta-alpha075-lr1p5e5-4epochs (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: selected epoch 4 accuracy `0.4908108108`; Macro-F1 `0.4764269902`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md9_alpha075_lr1p5e5_4epochs/checkpoint-3728` and `/kaggle/working/md9_results/md9_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD10 — ModernBERT + softened weighted CE

- TRAINING_NOTEBOOK: `notebooks/supervised/md10-modernbert-weighted-ce-alpha075 (2).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: best epoch 2 accuracy `0.4951351351`; Macro-F1 `0.4694130591`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md10_modernbert_weighted_alpha075/checkpoint-1864` and `/kaggle/working/md10_results/md10_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD11 — RoBERTa-base + softened weighted CE

- TRAINING_NOTEBOOK: `notebooks/supervised/md11-robertabase-weighted-ce-alpha075 (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: best epoch 3 accuracy `0.5021621622`; Macro-F1 `0.4842870850`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md11_roberta_base_weighted_alpha075/checkpoint-2796` and `/kaggle/working/md11_results/md11_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD12 — RoBERTa-base, four epochs

- TRAINING_NOTEBOOK: `notebooks/supervised/md12-robertabase-weighted-ce-alpha075-4epochs (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: selected epoch 3 accuracy `0.5005405405`; Macro-F1 `0.4817452965`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md12_roberta_base_weighted_alpha075_4epochs/checkpoint-2796` and `/kaggle/working/md12_results/md12_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

### MD13 — DeBERTa-v3-base + softened weighted CE

- TRAINING_NOTEBOOK: `notebooks/supervised/md13-debertav3base-weighted-ce-alpha075 (1).ipynb`
- TRAINING_SCRIPT: `NOT FOUND`
- EVALUATION_NOTEBOOK: same executed notebook
- EVALUATION_SCRIPT: `NOT FOUND` outside the notebook
- TRAINING_DATASET / VALIDATION_DATASET / TEST_DATASET: processed MathDial train / validation / `NOT USED`
- FINAL_METRICS: best epoch 3 accuracy `0.4221621622`; Macro-F1 `0.4257923789`
- FINAL_CHECKPOINT: local checkpoint `NOT FOUND`; notebook records `/kaggle/working/md13_debertav3_base_weighted_alpha075/checkpoint-2796` and `/kaggle/working/md13_results/md13_best_model.tar.gz`
- RELATED_RESULT_FOLDER: no standalone local folder; executed notebook output is authoritative
- REQUIRED_FOR_RUNTIME: no
- REQUIRED_FOR_RESEARCH_HISTORY: yes

## Current supervised runtime model

The normal current runtime does **not** use MD6. It uses the later MD7-R2-TELL-C1 Epoch 2 derivative:

- CURRENT_SUPERVISED_MODEL: `MD7-R2-TELL-C1 Epoch 2`
- EXACT_CHECKPOINT_PATH: `models/frozen/md7_r2_tell_c1_epoch2/`
- WEIGHT_SHA256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- MODEL_LOADER: `src/self_improvement/md6_inference.py::FrozenMD6Inference`; the class name is historical, but normal mode passes the MD7 directory explicitly
- CALL_SITE: `apps/tutor/app/integrations/adaptive_component_coordinator.py::_ProductionAdaptiveResources._build_base_selector()` and `_ensure_shared()`
- OUTPUT_CLASSES: `generic`, `probing`, `focus`, `telling`
- WHERE_PROBABILITIES_ENTER_CONTEXT: `src/self_improvement/turn_lints_runtime.py::DirectTurnLinTSPipeline.run_tutor_turn()` calls `predict_probabilities()`, then `DirectTurnController.select_turn()` calls `build_turn_lints_context()`; the four probabilities are block `S`, positions 1-4 of the live 11-dimensional S+K+L vector
- WHY_MD6_IS_STILL_PRESENT: explicit rollback mode, historical comparison/evaluation evidence, and compatibility class/module naming
- WHY_MD7_IS_STILL_PRESENT: the frozen MD7-R2 checkpoint is the normal production probability generator; MD7-R1 remains only as a manual rollback path and is a deletion candidate once rollback support is intentionally retired
- BOTH_USED_NORMALLY: no. Exactly one supervised selector is loaded for the selected mode. MD6 is only used when explicit rollback mode is requested.

## Preservation and proposed model-history layout

Preserve all MD notebooks/scripts and the compact evaluation artifacts. The minimal later reorganization (not performed) is:

```text
notebooks/model-development/
  md0/  -> pointer/readme to src/baselines/mathdial_baselines.py
  md1/  -> pointer/readme to src/baselines/mathdial_baselines.py
  md2/  -> md2 notebook
  md3/  -> README explicitly recording TRAINING NOTEBOOK NOT FOUND
  md4/ ... md13/ -> one executed notebook per model

results/model-development/
  md0-md4/ -> current results/baselines compact JSON/JSONL evidence
  md3/     -> preserve validation JSON and predictions because training source is absent
  md5-md13/ -> extracted compact metrics/manifests if created later; do not invent them now
```

No file was moved in this audit.
