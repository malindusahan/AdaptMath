# Final/current LinTS research map

Audit date: 2026-10-16

Scope: read-only audit. No file was deleted or moved. “Current code reference” distinguishes the normal request path from research scripts/tests where practical.

## Actual current Tutor execution path

```text
POST /tutor/start or tutoring turn
  -> apps/tutor/app/api/tutor.py
  -> apps/tutor/app/graph/workflow.py
  -> AdaptiveComponentCoordinator.start_attempt()/run_tutor_turn()
  -> _ProductionAdaptiveResources._build_base_selector(FrozenMD6Inference)
       normal selector mode passes models/frozen/md7_r2_tell_c1_epoch2
  -> FrozenMD6Inference.predict_probabilities(problem, history)
       returns generic/probing/focus/telling probabilities
  -> DirectTurnLinTSPipeline.run_tutor_turn()
  -> DirectTurnController.select_turn()
  -> build_turn_lints_context(..., blocks=S+K+L)
  -> DirectTurnLinTS.select_arm()
  -> TutorAgent.generate(... selected_move ...)
  -> FrozenMRB1Inference.score_response() [diagnostic scores]
  -> linked BKT outcome
  -> turn_rewards()/headroom_normalized
  -> DirectTurnController.complete_outcome()
  -> DirectTurnLinTS.update_once(selected_arm only)
  -> PostgreSQL policy snapshot + action/update evidence
```

### Supervised selector finding

- CURRENT_SUPERVISED_MODEL: `MD7-R2-TELL-C1 Epoch 2`
- EXACT_CHECKPOINT_PATH: `packages/move-selection/models/frozen/md7_r2_tell_c1_epoch2/`
- MODEL_WEIGHT_SHA256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- MODEL_LOADER: `packages/move-selection/src/self_improvement/md6_inference.py::FrozenMD6Inference`
- CALL_SITE: `apps/tutor/app/integrations/adaptive_component_coordinator.py::_ProductionAdaptiveResources._build_base_selector()`; normal mode calls `model_loader(self.md7_model_dir)`
- OUTPUT_CLASSES: `generic`, `probing`, `focus`, `telling`, in that exact order
- WHERE_PROBABILITIES_ENTER_CONTEXT: `DirectTurnLinTSPipeline.run_tutor_turn()` obtains the probability mapping; `DirectTurnController.select_turn()` passes it to `build_turn_lints_context()`. The four probabilities are block `S` of the 11-dimensional live vector.
- MD6_CURRENT_ROLE: explicit `MD6_ROLLBACK` mode, smoke/comparison research, and compatibility names. It is not simultaneously consulted in normal mode.
- MD7_CURRENT_ROLE: the frozen MD7-R2 checkpoint is the live four-class probability producer. MD7-R1 is only a separately selected rollback/manual-controlled branch.
- WHY_BOTH_EXIST: rollback and historical reproducibility. Normal execution loads one base selector, not both.

## MRB1 finding

- MRB1_CURRENT_RUNTIME_USE: **YES**
- MRB1_CHECKPOINT: `packages/move-selection/models/frozen/mrb1/` (`model.safetensors` is 496,281,464 bytes)
- LOADER: `packages/move-selection/src/self_improvement/mrb1_inference.py::FrozenMRB1Inference`
- PRIMARY_IMPORT/INSTANTIATION: `apps/tutor/app/integrations/adaptive_component_coordinator.py` imports and constructs `FrozenMRB1Inference()` in `_ensure_shared()`
- CALL_SITE: `packages/move-selection/src/self_improvement/turn_lints_runtime.py::DirectTurnLinTSPipeline.run_tutor_turn()` calls `score_response()` after Tutor generation; `record_mrb1_scores()` persists the four heads
- FOUR_HEADS: `Mistake_Identification`, `Mistake_Location`, `Providing_Guidance`, `Actionability`
- PURPOSE: response-quality diagnostic/evaluation and research provenance
- AFFECTS_MOVE_SELECTION: no in the final S+K+L policy. MRB1 is block `Q`, which is excluded from the live 11-dimensional vector.
- AFFECTS_REWARD: no. Reward comes only from the linked BKT mastery change.
- AFFECTS_POLICY_UPDATE: no. Only the selected arm, pre-action S+K+L vector, configured reward, and sample weight update the posterior.
- AFFECTS_EVALUATION: yes; scores are logged and analyzed as auxiliary quality signals.
- OTHER_IMPORTS: old controlled-live experiment, `full_attempt_demo.py`, the older `adaptive_tutor_pipeline.py`, and validation code also import the loader.
- TEST_REFERENCES: multiple Tutor integration, Turn-LinTS, warm-start, manual-control, trace, and lifecycle tests refer to MRB1; `packages/move-selection/tests/test_student_model_v3_bridge.py` also references it.
- TRAINING_EVIDENCE: `notebooks/critic/mrb1-robertabase-multitask-critic (1).ipynb`; best internal validation mean four-task Macro-F1 `0.5349478829` at epoch 4. The official unlabeled MRBench V3 test was not used for selection.
- CAN_MODEL_BE_REMOVED: **NO** without first changing the current pipeline and its diagnostic persistence contract.

## Authoritative final LinTS evidence

### Dataset and integrity

- AUTHORITATIVE_DATASET: `results/turn_lints_final_architecture_selection_v1/derived_analysis_dataset.csv`
- SHA-256 used by final evaluation: `da6ee0dde545be3d5bff28f179751bc041061defa5381e2d809e18f926713914`
- INCLUDED_TURNS: `713`
- LEARNERS: `31`
- ATTEMPTS: `92`
- SKILLS: `22`
- COLUMNS: `67`
- L coverage: `655/713` (`91.865%`); 58 rows use the explicit missingness representation
- ACTION_COUNTS: generic 168, probing 185, focus 212, telling 148
- PREFLIGHT: `results/adaptive_policy_final_evaluation_v1/preflight_audit.json` is `PASS`
- LEAKAGE/TEMPORAL CHECKS: zero duplicate IDs/rows, no current/post-action features, zero previous-delta/Q/move mismatches, zero turn-order/identity mismatches, finite complete contexts, and exact reward-equation checks

The canonical CSV lives in the earlier architecture-selection directory but the authoritative 713-row evaluation is `results/adaptive_policy_final_evaluation_v1/`. Its `final_results.json`, `preflight_audit.json`, and input hashes supersede the earlier 84-row selection summary for quantitative claims.

### Context selection / ablation

Experiment: `results/adaptive_policy_final_evaluation_v1/`

- Split: five-fold learner-grouped cross-validation (learners are never split between train and validation)
- Model: standardized Ridge reward prediction with direct action-by-context interactions
- Random turn split: false
- Final context: **S+K+L**, 11 dimensions
- S: four MD7 supervised probabilities
- K: `mastery_before`, `previous_mastery_delta`, `previous_mastery_delta_missing`
- L: previous learner reasoning, uncertainty, clarification probabilities, plus `previous_learner_signals_missing`

| Context | Dim | MAE mean | MAE SE | RMSE mean | RMSE SE | R² mean | R² SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 4 | 0.106082 | 0.004955 | 0.137112 | 0.004885 | -0.008733 | 0.008971 |
| S+K | 7 | 0.093327 | 0.005300 | 0.117767 | 0.005131 | 0.254411 | 0.036275 |
| S+L | 8 | 0.074650 | 0.002417 | 0.095109 | 0.002828 | 0.513326 | 0.012626 |
| **S+K+L** | **11** | **0.070489** | **0.002243** | **0.089850** | **0.002722** | **0.564885** | **0.017880** |

The earlier freeze/amendment package, `results/turn_lints_final_architecture_skl_freeze_v1/`, establishes the exact feature order and schema `turn_lints_context_v1:S+K+L:fd7363f6eee1`; the 713-row final evaluation supplies the stronger final empirical result.

### Reward selection

Evidence paths:

- `results/turn_lints_reward_context_audit_v1/` — initial real-data reward/context audit and explicit MRB1-not-reward conclusion
- `results/turn_lints_final_architecture_selection_v1/REWARD_DECISION.md` — frozen decision rationale
- `results/adaptive_policy_final_evaluation_v1/reward_summary.csv` and `reward_summary.json` — authoritative 713-row confirmation
- Implementation: `src/self_improvement/turn_lints_reward.py`

Compared rewards:

- raw mastery delta: `mastery_after - mastery_before`
- symmetric available-headroom normalization: positive delta divided by `1 - mastery_before`; negative delta divided by `mastery_before`; zero remains zero
- implementation asserts and clamps only numerical spillover to `[-1, 1]`; it does not use a tuned clipping threshold

On the 713-row final dataset, Spearman dependence between **absolute reward magnitude** and starting mastery was `rho=-0.441072` for raw delta and `rho=-0.250252` for headroom-normalized reward. The initial 96-row audit found `rho=-0.703475` versus `rho=-0.025773`. Both audits therefore favor headroom normalization; the larger audit shows reduction, not complete removal, of starting-mastery dependence.

### Grouped CV, bootstrap, and preflight

- GROUPED_CV: `results/adaptive_policy_final_evaluation_v1/ablation_fold_results.csv` and `ablation_summary.csv`
- OOF_PREDICTIONS: `ablation_oof_predictions.csv`
- BOOTSTRAP: `bootstrap_replicates.csv` and `bootstrap_summary.csv`
- BOOTSTRAP_REPLICATES: 5,000 learner-level cluster replicates
- S+K+L minus S MAE: point delta `-0.035539`; 95% CI `[-0.040968, -0.029674]`; improvement probability `1.0`
- S+K+L minus S RMSE: point delta `-0.047376`; 95% CI `[-0.054198, -0.040382]`; improvement probability `1.0`
- PREFLIGHT/LEAKAGE_AUDIT: `preflight_audit.py` and `preflight_audit.json`, status `PASS`

### Direct disjoint LinTS implementation and persistence

- POLICY: `src/self_improvement/turn_lints_policy.py::DirectTurnLinTS`
- CONTROLLER/PIPELINE: `src/self_improvement/turn_lints_runtime.py`
- CONTEXT: `src/self_improvement/turn_lints_context.py`
- REWARD: `src/self_improvement/turn_lints_reward.py`
- POSTGRES REPOSITORY: `src/self_improvement/postgres_repository.py`
- ARMS: generic, probing, focus, telling
- ARM_STATE: one `A` matrix and `b` vector per arm
- SAMPLING: for each arm, solve `A^-1 b` for posterior mean; sample a coefficient vector from a multivariate normal with covariance scaled from `A^-1`; score `x @ theta`
- ACTION_SELECTION: choose the maximum sampled score. Current `anchor_mode=none`, `anchor_gamma=0`, so no selector anchor changes the LinTS decision.
- UPDATE: `update_once()` changes only the selected arm using `A += w xx^T` and `b += w r x`; update IDs prevent duplicate credit.
- PERSISTENCE: every eligible completion saves the policy snapshot. Current local JSON is a bootstrap/fallback, while PostgreSQL is authoritative when enabled.

Read-only PostgreSQL inspection on 2026-10-16 found:

- POLICY_KEY: `turn_lints_live_skl_final_v1:policy_state.json`
- ALGORITHM: `direct_disjoint_turn_lints_v1`
- SCHEMA: `adaptmath_turn_lints_state_v2`
- CONTEXT: S+K+L, dimension 11
- REWARD: `headroom_normalized`
- SELECTOR: MD7-R2-TELL-C1 Epoch 2 with the expected SHA-256
- HYPERPARAMETERS: ridge lambda 1.0, exploration scale 0.2, seed 42
- DATA_MODE: real
- UPDATE_COUNT: 549
- ARM_UPDATE_COUNTS: generic 130, probing 201, focus 187, telling 31
- STATE_SHA256: `550cb9358008a88f280eca1f71a111c29f2922b35795147429d8ba9614c79e6a`

The current lineage retains `results/turn_lints_live_skl_synthetic_warmstart_v1/` because the live posterior was initialized with 500 synthetic observations and the runtime manifest cites this package. That package is therefore an exception to the general “remove warm-start research” rule.

## Proposed minimal final structure (not performed)

```text
packages/move-selection/
├── notebooks/
│   ├── model-development/md0 ... md13/
│   └── critic/mrb1/                    # current diagnostic model provenance
├── data/
│   ├── mathdial/raw-and-processed/
│   └── adaptive-policy/final-713-turn-dataset/
├── models/runtime/
│   ├── md7-r2-tell-c1-epoch2/
│   ├── mrb1/
│   └── md6-rollback/                   # retain only while rollback is supported
├── experiments/final-lints/
│   ├── context-ablation/
│   ├── reward-selection/
│   ├── grouped-cv-bootstrap-preflight/
│   └── live-initialization/
├── results/
│   ├── model-development/
│   └── final-lints/
├── src/                                # current selector, context, policy, reward, persistence
├── tests/
├── README.md
└── requirements.txt
```

Unique notebooks/evidence should be moved only in a later, separately approved operation with a reference-rewrite and hash manifest. No move occurred in this audit.

## Storage estimate

- CURRENT MOVE-SELECTION SIZE: **5,407,939,482 bytes** (5.408 GB / 5.037 GiB), including the three audit reports created in this pass
- ESTIMATED SPACE RECOVERED: **3,778,420,572 bytes** (3.778 GB / 3.519 GiB)
- ESTIMATED SIZE AFTER REMOVING SAFE OLD-DIRECTION CANDIDATES: **1,629,518,910 bytes** (1.630 GB / 1.518 GiB)

The approved deletion total was 3,778,420,572 bytes across 39 non-overlapping paths. Completion is recorded in `docs/maintenance/FINAL_OLD_RESEARCH_DELETION_LOG.csv`. The cleanup deliberately excluded the 332,076,874-byte `models/candidates/md7r1_epoch3` directory because current code still exposes that exact directory as the explicit MD7-R1 rollback checkpoint. Actual filesystem allocation may differ from logical byte totals.

## Complete `results/` folder classification

Sizes are recursive exact byte counts as of 2026-10-16. A parent and its child both show recursive size by design; deletion estimates do not double-count nested paths.

| Path | SizeBytes | Classification | Reason | Unique files of value | Current code reference | Final research reference | Safe to delete |
|---|---:|---|---|---|---|---|---|
| `results/adaptive_policy_final_evaluation_v1` | 2,222,210 | KEEP_FINAL_LINTS_RESEARCH | Authoritative 713-turn final evaluation | final_results, ablations, bootstrap, preflight | Validation/research scripts | Yes | No |
| `results/adaptive_policy_final_evaluation_v1/figures` | 300,148 | KEEP_FINAL_LINTS_RESEARCH | Final-result visual evidence | six PNG/PDF figures | No | Yes | No |
| `results/adaptive_policy_final_evaluation_v1/input_backup` | 722,830 | DELETE_DUPLICATE | Pre-update copies superseded by hash-verified canonical CSV | two backup CSVs | No | No | Yes |
| `results/baselines` | 1,962,557 | KEEP_MD0_MD13_MODEL_RESEARCH | MD0-MD4 primary/irreplaceable evidence, especially missing-source MD3 | baseline JSON and prediction JSONL | Baseline/recovery scripts | No | No |
| `results/md_self_improvement_expanded_analysis_v1` | 1,035,215 | DELETE_OLD_RESEARCH_DIRECTION | Superseded observational self-improvement analysis | report, tables, figures | No | No | Yes |
| `results/md_self_improvement_pilot_analysis_v1` | 437,811 | DELETE_OLD_RESEARCH_DIRECTION | Superseded pilot analysis | report and pilot tables | No | No | Yes |
| `results/md_self_improvement_turn_data_v1` | 1,813,303 | KEEP_UNIQUE_DATASET | Durable raw turn/assessment/BKT evidence and collector fallback | turn_outcomes, BKT activity, summaries | Yes: collector/coordinator | Source provenance | No |
| `results/md_self_improvement_turn_data_v1/pending_actions` | 8,751 | KEEP_CURRENT_RUNTIME | Pending-action recovery area with one record | pending action JSON | Yes: collector | No | No |
| `results/md_self_improvement_turn_data_v1/pending_processing_failures` | 0 | KEEP_CURRENT_RUNTIME | Expected failure-recovery directory | empty | Yes: collector | No | No |
| `results/md7_r2_tell_c1_blinded_review_v1` | 166,813 | DELETE_OLD_RESEARCH_DIRECTION | Superseded MD7-R2 blinded review v1 | completed review/report | No | No | Yes |
| `results/md7_r2_tell_c1_blinded_review_v2` | 262,003 | DELETE_OLD_RESEARCH_DIRECTION | Abandoned MD7-R2 review machinery | v2 forms/states | No | No | Yes |
| `results/md7_r2_tell_c1_epoch2_diagnostic_v1` | 406,184 | KEEP_CURRENT_RUNTIME | Compact diagnostic evidence for exact live checkpoint | summary, real-state/challenge predictions | No normal call | Selector provenance | No |
| `results/md7_r2_tell_c1_preparation_v1` | 368,200,105 | KEEP_CURRENT_RUNTIME | Contains unique training notebook/data for current live selector; child duplicates can go | MD7-R2 Kaggle notebook, calibration manifests/data | No normal call | Selector provenance | No as a whole |
| `results/md7_r2_tell_c1_preparation_v1/mathdial` | 27,920,313 | DELETE_DUPLICATE | Duplicate MathDial train/validation copies | no unique files | No | No | Yes |
| `results/md7_r2_tell_c1_preparation_v1/md7r1_epoch3` | 332,058,388 | DELETE_DUPLICATE | Old MD7-R1 checkpoint duplicated elsewhere and not current | model/tokenizer only | No | No | Yes |
| `results/md7_r2_turn_lints_implementation_v1` | 6,610 | DELETE_OLD_RESEARCH_DIRECTION | Superseded handoff document | IMPLEMENTATION_HANDOFF.md | No | No | Yes |
| `results/md7_telling_epoch_ablation_v1` | 458,890 | KEEP_CURRENT_RUNTIME | Compact checkpoint-selection provenance for current derivative | epoch comparison, summary | No normal call | Selector provenance | No |
| `results/md7_telling_real_state_diagnostic_v1` | 581,847 | KEEP_CURRENT_RUNTIME | Compact live-selector calibration provenance | summary, safety check, predictions | No normal call | Selector provenance | No |
| `results/md7r1_manual_controlled_live_v1` | 241,710 | DELETE_OLD_RESEARCH_DIRECTION | Obsolete MD7-R1 controlled-live session | session logs | Rollback validator only | No | Yes |
| `results/md7r1_promotion_v1` | 15,722 | DELETE_OLD_RESEARCH_DIRECTION | Obsolete promotion manifest | promotion_manifest.json | Legacy validator | No | Yes |
| `results/md7r1_real_diagnostic_v1` | 95,465 | DELETE_OLD_RESEARCH_DIRECTION | Obsolete MD6-vs-MD7R1 diagnostic | case-level comparison | Old experiment only | No | Yes |
| `results/model_recovery` | 324,712 | KEEP_MD0_MD13_MODEL_RESEARCH | Records recovery audit for MD3/MD4 evidence; MD3 training source is missing | si9a_training_asset_audit.txt | Recovery audit | No | No |
| `results/move_selector_architecture_io_view_v1` | 106,383 | KEEP_CURRENT_RUNTIME | Compact audit of live 11D I/O and posterior accounting | workbook, builder, summary | No normal call | Yes | No |
| `results/preprocessing` | 82,331 | KEEP_MD0_MD13_MODEL_RESEARCH | Dataset construction/EDA evidence used across MD history | preprocessing summaries/figures | Preprocess scripts | No | No |
| `results/project_cleanup_final_v1` | 138,975,684 | DELETE_MAINTENANCE | Cleanup inventories/reports, not research evidence | cleanup manifests only | No | No | Yes |
| `results/project_cleanup_final_v1/figures` | 248,325 | DELETE_MAINTENANCE | Cleanup-report figures | four figures | No | No | Yes with parent |
| `results/selection` | 2,244 | KEEP_MD0_MD13_MODEL_RESEARCH | Compact MD3/MD4 historical selector decision | frozen_supervised_selector.json | Historical recovery only | No | No |
| `results/self_improvement` | 889,516,557 | DELETE_OLD_RESEARCH_DIRECTION | Superseded attempt-level/threshold/value/self-improvement program | many SI manifests and results | Archived/legacy code only | No | Yes |
| `results/self_improvement/context_ablation_v1` | 406,205,075 | DELETE_OLD_RESEARCH_DIRECTION | Synthetic/old context ablation, not final 713-turn study | raw metrics, summaries | Legacy experiment | No | Yes with parent |
| `results/self_improvement/context_ablation_v1/e2_mastery_diagnostic` | 7,614,532 | DELETE_OLD_RESEARCH_DIRECTION | Nested obsolete diagnostic | mastery sweep/results | Legacy analysis | No | Yes with parent |
| `results/self_improvement/context_ablation_v1/e2_mastery_diagnostic/figures` | 844,430 | DELETE_OLD_RESEARCH_DIRECTION | Figures for obsolete diagnostic | ten figures | No | No | Yes with parent |
| `results/self_improvement/context_ablation_v1/figures` | 1,931,675 | DELETE_OLD_RESEARCH_DIRECTION | Figures for obsolete ablation | ten figures | No | No | Yes with parent |
| `results/self_improvement/final_candidate_policy` | 3,715 | DELETE_OLD_RESEARCH_DIRECTION | Superseded attempt-level candidate | final_candidate_manifest.json | Legacy only | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1` | 97,604,076 | DELETE_OLD_RESEARCH_DIRECTION | Superseded attempt-level sensitivity suite | experiment manifest and branches | Legacy only | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/credit` | 21,164,885 | DELETE_OLD_RESEARCH_DIRECTION | Old credit sensitivity | raw metrics/summary | Legacy only | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/credit/figures` | 390,934 | DELETE_OLD_RESEARCH_DIRECTION | Old credit figures | two figures | No | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/lints` | 54,764,341 | DELETE_OLD_RESEARCH_DIRECTION | Superseded attempt-level LinTS sensitivity | raw metrics/summary | Legacy only | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/lints/figures` | 418,516 | DELETE_OLD_RESEARCH_DIRECTION | Old LinTS figures | two figures | No | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/reward` | 21,668,213 | DELETE_OLD_RESEARCH_DIRECTION | Old simulated reward comparison | raw metrics/summary | Legacy only | No | Yes with parent |
| `results/self_improvement/final_sensitivity_suite_v1/reward/figures` | 421,004 | DELETE_OLD_RESEARCH_DIRECTION | Old reward figures | two figures | No | No | Yes with parent |
| `results/self_improvement/md6_dialogue_trajectory_v1` | 219,912 | DELETE_OLD_RESEARCH_DIRECTION | Auxiliary MD6 diagnostic, not primary MD6 training evidence | predictions/summary | Old experiment | No | Yes with parent |
| `results/self_improvement/md6_dialogue_trajectory_v1/figures` | 209,508 | DELETE_OLD_RESEARCH_DIRECTION | Diagnostic figures | three figures | No | No | Yes with parent |
| `results/self_improvement/md6_gap_threshold_audit` | 1,877,065 | DELETE_OLD_RESEARCH_DIRECTION | Threshold-gated policy research superseded by direct LinTS | threshold tables | Legacy analysis | No | Yes with parent |
| `results/self_improvement/md6_gap_threshold_audit/figures` | 963,043 | DELETE_OLD_RESEARCH_DIRECTION | Threshold figures | ten figures | No | No | Yes with parent |
| `results/self_improvement/md6_history_ablation_v1` | 656,740 | DELETE_OLD_RESEARCH_DIRECTION | Auxiliary MD6 behavior study | compressed predictions/summary | Old experiment | No | Yes with parent |
| `results/self_improvement/md6_history_ablation_v1/figures` | 269,991 | DELETE_OLD_RESEARCH_DIRECTION | History-ablation figures | five figures | No | No | Yes with parent |
| `results/self_improvement/md6_move_behavior_v1` | 151,272 | DELETE_OLD_RESEARCH_DIRECTION | Auxiliary MD6 behavior study | cases/summary | Old experiment | No | Yes with parent |
| `results/self_improvement/md6_move_behavior_v1/figures` | 135,707 | DELETE_OLD_RESEARCH_DIRECTION | Move-behavior figures | three figures | No | No | Yes with parent |
| `results/self_improvement/md6_opportunity_by_history_depth_v1` | 451,233 | DELETE_OLD_RESEARCH_DIRECTION | Old eligibility/opportunity study | depth tables/summary | Old experiment | No | Yes with parent |
| `results/self_improvement/md6_opportunity_by_history_depth_v1/figures` | 277,819 | DELETE_OLD_RESEARCH_DIRECTION | Opportunity figures | four figures | No | No | Yes with parent |
| `results/self_improvement/md6_validation_vs_demo_v1` | 612,409 | DELETE_OLD_RESEARCH_DIRECTION | Auxiliary MD6 distribution diagnostic | predictions/summary | Old experiment | No | Yes with parent |
| `results/self_improvement/md6_validation_vs_demo_v1/figures` | 272,131 | DELETE_OLD_RESEARCH_DIRECTION | Diagnostic figures | six figures | No | No | Yes with parent |
| `results/self_improvement/personalized_behavior_context_v1` | 56,097,893 | DELETE_OLD_RESEARCH_DIRECTION | Superseded simulated personalization study | compressed raw results | Old experiment | No | Yes with parent |
| `results/self_improvement/personalized_behavior_context_v1/figures` | 92,650 | DELETE_OLD_RESEARCH_DIRECTION | Study figures | two figures | No | No | Yes with parent |
| `results/self_improvement/personalized_context_v1` | 60,154,933 | DELETE_OLD_RESEARCH_DIRECTION | Superseded simulated personalization study | compressed raw results | Old experiment | No | Yes with parent |
| `results/self_improvement/personalized_context_v1/figures` | 90,266 | DELETE_OLD_RESEARCH_DIRECTION | Study figures | two figures | No | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v1` | 0 | DELETE_OLD_RESEARCH_DIRECTION | Empty obsolete directory | none | Old experiment path | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v1/figures` | 0 | DELETE_OLD_RESEARCH_DIRECTION | Empty obsolete directory | none | No | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v2` | 14,626,328 | DELETE_OLD_RESEARCH_DIRECTION | Superseded simulated reward study | compressed raw results | Old experiment | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v2/figures` | 159,518 | DELETE_OLD_RESEARCH_DIRECTION | Old reward figures | three figures | No | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v3_signed` | 14,611,494 | DELETE_OLD_RESEARCH_DIRECTION | Superseded simulated signed-reward study | compressed raw results | Old experiment | No | Yes with parent |
| `results/self_improvement/reward_function_comparison_v3_signed/figures` | 158,262 | DELETE_OLD_RESEARCH_DIRECTION | Old reward figures | three figures | No | No | Yes with parent |
| `results/self_improvement/threshold_sensitivity_v1` | 236,156,033 | DELETE_OLD_RESEARCH_DIRECTION | Threshold policy superseded by direct, ungated LinTS | raw metrics/summary | Legacy experiment | No | Yes with parent |
| `results/self_improvement/threshold_sensitivity_v1/figures` | 1,428,723 | DELETE_OLD_RESEARCH_DIRECTION | Threshold sensitivity figures | fourteen figures | No | No | Yes with parent |
| `results/turn_lints_final_architecture_selection_v1` | 2,123,478 | KEEP_FINAL_LINTS_RESEARCH | Contains canonical dataset, initial reward decision, and architecture provenance | canonical CSV, decisions, ablations | Architecture validator | Yes | No |
| `results/turn_lints_final_architecture_selection_v1/figures` | 1,361,782 | KEEP_FINAL_LINTS_RESEARCH | Final-lineage visual evidence | fourteen figures | No | Yes | No |
| `results/turn_lints_final_architecture_skl_freeze_v1` | 698,892 | KEEP_FINAL_LINTS_RESEARCH | Freezes 11D S+K+L schema used live | manifest/amendment/notes | Runtime lineage references | Yes | No |
| `results/turn_lints_final_architecture_skl_freeze_v1/figures` | 663,213 | KEEP_FINAL_LINTS_RESEARCH | Frozen architecture figures | seven figures | No | Yes | No |
| `results/turn_lints_live_skl_synthetic_warmstart_v1` | 4,463,168 | KEEP_CURRENT_RUNTIME | Current live posterior initialization and activation evidence | 500-observation package, posterior before/after | Yes: activation, lineage, tests | Yes | No |
| `results/turn_lints_live_skl_synthetic_warmstart_v1/figures` | 2,790,605 | KEEP_CURRENT_RUNTIME | Visual evidence for current initialization | 28 figures | No normal call | Yes | No |
| `results/turn_lints_randomized_warmstart_audit_v1` | 131,355 | DELETE_OLD_RESEARCH_DIRECTION | Randomized warm-start proposal superseded by final synthetic initialization | audit/report | No | No | Yes |
| `results/turn_lints_randomized_warmstart_collection_v1` | 382,346 | DELETE_OLD_RESEARCH_DIRECTION | Abandoned randomized collection machinery | handoff/readiness scripts | No | No | Yes |
| `results/turn_lints_randomized_warmstart_collection_v1/figures` | 350,059 | DELETE_OLD_RESEARCH_DIRECTION | Figures for abandoned collection | four figures | No | No | Yes with parent |
| `results/turn_lints_reward_context_audit_v1` | 200,295 | KEEP_FINAL_LINTS_RESEARCH | Initial real-data justification for headroom reward and MRB1 exclusion | report, summary, reward table | No normal call | Yes | No |
| `results/turn_lints_selector_anchor_v1` | 61,406 | KEEP_FINAL_LINTS_RESEARCH | Negative/selection evidence for final anchor=none, gamma=0 | gamma calibration and report | Tests only | Yes | No |
| `results/turn_lints_shadow_collection_v2` | 67,155 | DELETE_OLD_RESEARCH_DIRECTION | Superseded shadow collection plan | schema/readiness docs | No | No | Yes |
| `results/tutor_move_realization_final_v1` | 353,361 | KEEP_CURRENT_RUNTIME | Final four-move Tutor realization contract | contract, decision evidence, cases | Tests validate it | Yes | No |
| `results/tutor_move_realization_final_v1/figures` | 312,535 | KEEP_CURRENT_RUNTIME | Current contract figures | four figures | No | Yes | No |
| `results/tutor_move_realization_v1` | 31,159 | DELETE_OLD_RESEARCH_DIRECTION | Superseded by final realization v1 package | old smoke report/contract | No | No | Yes |

Coverage check: 79 directories are present under `results/`, and this table contains 79 classified rows. No directory is left `UNKNOWN`.
