# Move-Selector Research Artefact Map

Audit date: 2026-10-16  
Scope: move-selection-owned artefacts. Sizes are exact bytes for files or recursive bytes for directories as observed before cleanup. `Yes` means preserve; `No` means not required for that particular role. Nothing listed here is approved for deletion.

## 1. MD Model Development

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Baseline and supervised-selector analysis/freezing code | `src\baselines` | 67,863 B | No | Yes | Yes |
| Complete supervised experiment notebooks through MD13 | `notebooks\supervised` | 425,026 B | No | Yes | Yes |
| Historical recovered MD3/MD4 outputs | `artifacts\historical_selector_outputs` | 2,062,148 B | No | Yes | Yes |
| Baseline validation/prediction and bootstrap comparisons | `results\baselines` | 1,962,557 B | No | Yes | Yes |
| Later model-selection, diagnostic, promotion, blinded-review, and ablation results | `results\md7*` directories listed below | 370,435,349 B | No | Yes | Yes |

## 2. MD0–Highest MD Training Code

The highest numbered MD notebook found is MD13.

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| MD0 majority baseline and MD1 TF-IDF/logistic-regression baseline code | `src\baselines\mathdial_baselines.py` | 8,100 B | No | Yes | Yes |
| MD0/MD1 validation outputs | `results\baselines\mathdial_majority_validation.json`; `results\baselines\mathdial_tfidf_logreg_validation.json` | 1,254 B | No | Yes | Yes |
| MD2 DistilRoBERTa training notebook | `notebooks\supervised\md2-distilroberta-move-selector (1).ipynb` | 55,999 B | No | Yes | Yes |
| MD2 validation metrics/predictions | `results\baselines\md2_distilroberta_validation.json`; `results\baselines\md2_validation_predictions.jsonl` | 680,352 B | No | Yes | Yes |
| MD3 comparison/recovery code | `src\baselines\analyze_md2_md3.py`; `src\baselines\analyze_md2_md3_md4.py` | 22,764 B | No | Yes | Yes |
| MD3 validation/predictions and historical recovery copy | `results\baselines\md3_*`; `artifacts\historical_selector_outputs\md3` | 1,273,862 B | No | Yes | Yes |
| MD4 weighted-loss notebook | `notebooks\supervised\md4-distilroberta-weighted-cross-entropy (2).ipynb` | 57,933 B | No | Yes | Yes |
| MD4 validation/predictions and historical recovery copy | `results\baselines\md4_*`; `artifacts\historical_selector_outputs\md4` | 1,273,462 B | No | Yes | Yes |
| MD5 soft-weighted-CE notebook | `notebooks\supervised\md5-distilroberta-soft-weighted-ce.ipynb` | 32,613 B | No | Yes | Yes |
| MD6 training and held-out-test notebooks | `notebooks\supervised\md6-distilroberta-soft-weighted-ce-alpha075 (1).ipynb`; `notebooks\md6-held-out-test.ipynb` | 55,286 B | No | Yes | Yes |
| MD6 frozen inference loader | `src\self_improvement\md6_inference.py` | 6,037 B | Yes | Yes | Yes |
| MD7 targeted-telling notebook | `notebooks\supervised\md7-distilroberta-targeted-telling-weight (1).ipynb` | 32,744 B | No | Yes | Yes |
| MD8–MD13 later-model notebooks | `notebooks\supervised\md8-*` through `md13-*` | 212,849 B | No | Yes | Yes |
| MD7/MD7r1 preparation, calibration, and comparison evidence | `results\md7_r2_tell_c1_preparation_v1`; `md7_r2_tell_c1_blinded_review_v1`; `md7_r2_tell_c1_blinded_review_v2`; `md7_r2_tell_c1_epoch2_diagnostic_v1`; `md7_r2_turn_lints_implementation_v1`; `md7_telling_epoch_ablation_v1`; `md7_telling_real_state_diagnostic_v1`; `md7r1_manual_controlled_live_v1`; `md7r1_promotion_v1`; `md7r1_real_diagnostic_v1` | 370,435,349 B | No | Yes | Yes |

No file named as an MD3 training notebook was found. The surviving MD3 analysis code, validation metrics, full predictions, and recovered historical output are retained; the missing notebook is not recreated or inferred.

## 3. MathDial Datasets

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Raw MathDial train/test source data | `data\raw\mathdial` | 9,039,609 B | No | Yes | Yes |
| Cleaned all/train/validation/test splits | `data\processed\mathdial` | 61,929,438 B | No | Yes | Yes |
| Profile train/validation splits | `data\processed\mathdial_profile` | 30,499,724 B | No | Yes | Yes |
| External/raw supporting inputs | `data\external` | 5,054,215 B | No | Yes | Yes |
| Preprocessing, audit, and split construction code | `src\data`; `src\baselines` | 120,092 B | No | Yes | Yes |

## 4. Final MD6 Runtime Model

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Complete frozen MD6 package: config, weights, tokenizer, tokenizer config | `models\frozen\md6` | 332,058,389 B | Yes (rollback/smoke path) | Yes | Yes |
| Offline production loader/preprocessor | `src\self_improvement\md6_inference.py` | 6,037 B | Yes | Yes | Yes |
| Frozen-model checksums | `models\frozen\SHA256SUMS.txt` | 1,187 B | Yes | Yes | Yes |

## 5. Later Candidate Models

| Classification | Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---|---:|---|---|---|
| CURRENT_RUNTIME | Frozen MD7-R2-TELL-C1 epoch 2 selected by current Tutor configuration | `models\frozen\md7_r2_tell_c1_epoch2` | 332,058,437 B | Yes | Yes | Yes |
| IMPORTANT_RESEARCH_CANDIDATE | Same MD7-R2 epoch checkpoint at candidate lineage path | `models\candidates\md7_r2_tell_c1` | 332,058,437 B | No | Yes | Yes |
| IMPORTANT_RESEARCH_CANDIDATE | MD7r1 epoch 3 supported rollback/manual mode | `models\candidates\md7r1_epoch3` | 332,076,874 B | Conditional | Yes | Yes |
| IMPORTANT_RESEARCH_CANDIDATE | MD7 telling-calibration epoch candidates and associated package | `models\candidates\md7_telling_calibration_epoch3` | 1,913,858,197 B | No | Yes | Yes |

No candidate was classified safe as an obsolete intermediate because reported comparisons or runtime rollback paths may depend on it.

## 6. ~713-Turn Adaptive-Policy Dataset

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Authoritative derived analysis dataset: 713 turns, 92 attempts, 31 learners, 22 skills, 67 columns | `results\turn_lints_final_architecture_selection_v1\derived_analysis_dataset.csv` | 588,172 B | No | Yes | Yes |
| Provenance mapping for the derived rows | `results\turn_lints_final_architecture_selection_v1\data_provenance.csv` | 3,566 B | No | Yes | Yes |
| Collected/linked turn-level raw evidence | `results\md_self_improvement_turn_data_v1` | 1,813,303 B | No | Yes | Yes |
| Derived self-improvement transitions, rewards, context and contrastive data | `data\processed\self_improvement` | 121,295,775 B | No | Yes | Yes |

The authoritative dataset SHA-256 is `da6ee0dde545be3d5bff28f179751bc041061defa5381e2d809e18f926713914`.

## 7. Reward Design Experiments

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Runtime raw/headroom reward implementation and clipping | `src\self_improvement\turn_lints_reward.py` | 1,780 B | Yes | Yes | Yes |
| Reward-context audit, linked turn table, mastery-band and turn-position analyses | `results\turn_lints_reward_context_audit_v1` | 200,295 B | No | Yes | Yes |
| Reward formulation comparison v1–v3 (raw, normalized, signed) | `results\self_improvement\reward_function_comparison_v1`; `reward_function_comparison_v2`; `reward_function_comparison_v3_signed` | 29,237,822 B | No | Yes | Yes |
| Final architecture reward decision and summary | `results\turn_lints_final_architecture_selection_v1\REWARD_DECISION.md`; `reward_experiment_results.csv`; `reward_summary.json` | 10,453 B | No | Yes | Yes |

## 8. Context Selection / Ablation

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Runtime pre-action block definitions, missingness, scaling and exact S+K+L feature order | `src\self_improvement\turn_lints_context.py` | 8,782 B | Yes | Yes | Yes |
| Final context-selection package: S, S+K, S+L, S+K+L and larger variants | `results\turn_lints_final_architecture_selection_v1` | 2,243,042 B | No | Yes | Yes |
| Frozen S+K+L amendment and 11-dimensional manifest | `results\turn_lints_final_architecture_skl_freeze_v1` | 698,892 B | Yes (lineage/config evidence) | Yes | Yes |
| Full raw context-ablation experiment results | `results\self_improvement\context_ablation_v1` | 406,205,075 B | No | Yes | Yes |

## 9. Learner-Grouped Evaluation

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| GroupKFold, Ridge, MAE, RMSE, R-squared and out-of-fold final evaluation | `results\adaptive_policy_final_evaluation_v1` | 2,272,636 B | No | Yes | Yes |
| Per-context grouped fold outputs | `results\turn_lints_final_architecture_selection_v1\context_fold_results.csv` | 14,437 B | No | Yes | Yes |

## 10. Bootstrap Analysis

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Learner-level 5,000-replicate bootstrap implementation/config and final metrics | `results\adaptive_policy_final_evaluation_v1\evaluate_recovered_dataset.py`; `bootstrap_replicates.csv`; `bootstrap_summary.csv`; `final_results.json` | 337,254 B | No | Yes | Yes |
| Final context bootstrap comparisons | `results\turn_lints_final_architecture_selection_v1\context_bootstrap_comparisons.csv` | 1,295 B | No | Yes | Yes |
| Supervised selector bootstrap scripts/results | `src\baselines\bootstrap_selector_comparison.py`; `cluster_bootstrap_selector_comparison.py`; `results\baselines\selector_*bootstrap_comparison.json` | 37,968 B | No | Yes | Yes |

## 11. Preflight / Leakage Audit

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Row/ID/count/missingness/action/temporal/leakage preflight script and PASS audit | `results\adaptive_policy_final_evaluation_v1\preflight_audit.py`; `preflight_audit.json` | 17,353 B | No | Yes | Yes |
| Context coverage and provenance checks | `results\turn_lints_final_architecture_selection_v1\context_feature_coverage.csv`; `data_provenance.csv` | 5,863 B | No | Yes | Yes |
| Historical SI numeric/reward/schema validation | Superseded validators were removed on 2026-10-17 after their old source-data and result trees had already been removed. The retained final validation is `results\adaptive_policy_final_evaluation_v1\preflight_audit.py` with `preflight_audit.json`. | 0 B | No | No | No |

The preflight audit records PASS with no duplicate IDs/rows and no detected future/post-action leakage.

## 12. LinTS

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Current direct disjoint per-arm A/b policy, sampling, update and state I/O | `src\self_improvement\turn_lints_policy.py` | 20,595 B | Yes | Yes | Yes |
| Runtime selection modes, ledgers and completion/update flow | `src\self_improvement\turn_lints_runtime.py` | 37,465 B | Yes | Yes | Yes |
| PostgreSQL-backed policy/action persistence adapter | `src\self_improvement\postgres_repository.py` | 24,722 B | Yes | Yes | Yes |
| Live lineage validation | `src\self_improvement\turn_lints_live_lineage.py` | 4,744 B | Yes | Yes | Yes |
| Audited S+K+L synthetic warm-start observations and posterior before/after | `results\turn_lints_live_skl_synthetic_warmstart_v1` | 4,463,168 B | Yes (lineage) | Yes | Yes |
| Historical LinTS/controller implementations and experiment code | `src\self_improvement\lints_policy.py`; `turn_level_controller.py`; `turn_lints_architecture_v1.py`; related `experiments` | 501,532 B | No | Yes | Yes |

The live policy state itself is physically owned by Tutor at `apps\tutor\runtime\turn_lints_live_skl_final_v1` and current authoritative persistence may be PostgreSQL-backed. Those integration artefacts were inspected read-only and were not changed.

## 13. Runtime Integration

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Active frozen MD7 selector package | `models\frozen\md7_r2_tell_c1_epoch2` | 332,058,437 B | Yes | Yes | Yes |
| Frozen MRB1 critic package and custom model class | `models\frozen\mrb1` | 499,843,480 B excluding disposable bytecode | Yes | Yes | Yes |
| Frozen MD6/MD7-compatible inference wrapper | `src\self_improvement\md6_inference.py` | 6,037 B | Yes | Yes | Yes |
| MRB1 inference wrapper | `src\self_improvement\mrb1_inference.py` | 7,638 B | Yes | Yes | Yes |
| Context, policy, reward, runtime, lineage and persistence modules | `src\self_improvement\turn_lints_*.py`; `postgres_repository.py` | 118,868 B | Yes | Yes | Yes |
| Move-selection-owned integration helpers | `src\integration` | 12,755 B | Yes | Yes | Yes |
| Tutor coordinator/bridge | `apps\tutor\app\integrations\adaptive_component_coordinator.py` (read-only, outside scope) | External | Yes | No | No |

## 14. Tests

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| Automated move-selector/Tutor bridge tests | `tests\test_student_model_v3_bridge.py` | 34,961 B | No | Yes | Yes |
| Manual bridge smoke test | `tests\manual_smoke_student_model_v3_bridge.py` | 9,769 B | No | Yes | Yes |

## 15. Thesis/Figure Source Artefacts

| Purpose | Path | Size | RuntimeRequired | ResearchRequired | ThesisVivaRequired |
|---|---|---:|---|---|---|
| All final/raw experiment results, plots, tables, decision documents and metrics | `results` | 1,415,625,715 B | Some lineage artefacts | Yes | Yes |
| Final evaluation JSON/CSV/figure source | `results\adaptive_policy_final_evaluation_v1` | 2,272,636 B | No | Yes | Yes |
| Final architecture decision evidence and source tables | `results\turn_lints_final_architecture_selection_v1` | 2,243,042 B | No | Yes | Yes |
| Reward comparison figures and source tables | `results\self_improvement\reward_function_comparison_v2`; `reward_function_comparison_v3_signed` | 29,237,822 B | No | Yes | Yes |
| Full context-ablation raw metrics and figures | `results\self_improvement\context_ablation_v1` | 406,205,075 B | No | Yes | Yes |
