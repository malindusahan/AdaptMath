# Final AdaptMath project structure

1. Final frozen selector: `pedagogical-move-selection/models/frozen/md7_r2_tell_c1_epoch2/`
2. Turn-LinTS source: `pedagogical-move-selection/src/self_improvement/turn_lints_*.py`
3. S+K+L architecture: `pedagogical-move-selection/results/turn_lints_final_architecture_skl_freeze_v1/`
4. LIVE runtime: `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/`
5. Synthetic initialization: `pedagogical-move-selection/results/turn_lints_live_skl_synthetic_warmstart_v1/synthetic_warmstart_observations.jsonl`
6. Real interaction event log: `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/turn_events.jsonl` (created only by genuine interactions; may be absent before the first interaction)
7. HEADROOM reward: `pedagogical-move-selection/src/self_improvement/turn_lints_reward.py`
8. MRB1 model: `pedagogical-move-selection/models/frozen/mrb1/`
9. Student Modeling: `student-modeling/`
10. Notebooks: indexed in `NOTEBOOK_INDEX.csv` (36 retained)
11. Datasets: indexed in `DATASET_INDEX.csv` (1124 retained assets)
12. Models: indexed in `MODEL_INDEX.csv` (46 retained lineages)
13. Final experiment evidence: architecture selection, S+K+L freeze, synthetic warm-start LIVE audit, and final Tutor realization folders
14. Supporting/rejected evidence: indexed in `EXPERIMENT_EVIDENCE_INDEX.csv`, including anchor rejection and warm-start/cold-start diagnostics
15. Tutor source/frontend: `adaptive-math-tutor/backend/` and `adaptive-math-tutor/frontend/`
16. Launch scripts: `start-adaptmath-local.ps1`, `status-adaptmath-local.ps1`, and `stop-adaptmath-local.ps1`

The inventory is conservative: ambiguous assets, source, dependencies, historical results, runtime lineages, and logs are retained.
