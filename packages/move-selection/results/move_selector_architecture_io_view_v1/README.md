# Move Selector Architecture Input/Output View v1

This directory contains a read-only, human-readable Excel view of the current
LIVE AdaptMath Turn-LinTS runtime.

## Outputs

- `move_selector_architecture_io_view.xlsx` - the generated workbook.
- `summary.json` - source paths/hashes, exported counts, accounting, and checks.
- `build_move_selector_io_view.py` - reproducible generator.

## Authoritative sources

The generator reads:

- `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/turn_events.jsonl`
- `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/policy_state.json`
- `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/lineage_manifest.json`
- the frozen architecture manifest referenced and hash-pinned by the lineage;
- `pedagogical-move-selection/results/md_self_improvement_turn_data_v1/turn_outcomes.jsonl`
  only to enrich authoritative LIVE action identities with problem, Tutor,
  learner, evaluator, resolver, and MRB1 fields.

SHADOW or RANDOMIZED_WARMSTART rows are never mixed into the primary LIVE
sheets. If such rows occur in the source event file, the script places them in
`Historical_NonLIVE`.

## Regenerate

From the workspace root in PowerShell:

```powershell
& .\student-modeling\.venv\Scripts\python.exe `
  .\pedagogical-move-selection\results\move_selector_architecture_io_view_v1\build_move_selector_io_view.py
```

`openpyxl` and NumPy are required. The local `student-modeling\.venv` contains
both.

The script verifies source hashes before and after generation. If a real Tutor
interaction changes the event log or posterior while the workbook is being
built, generation fails instead of publishing a mixed-time snapshot. The
script never imports the runtime policy, never selects an action, never applies
an update, and never modifies a runtime source.

## Sheets

1. `Summary` - architecture, sources, counts, and state/log accounting.
2. `Decision_Only` - compact turn-by-turn pre-action inputs and downstream output.
3. `LIVE_Turn_IO` - wide forensic action/evidence/reward view.
4. `Field_Guide` - plain-language field definitions and timing.
5. `Architecture_Flow` - pre-action versus post-action pipeline.
6. `Posterior_State` - state metadata, A/b shapes, condition numbers, and means.
7. `LIVE_Update_Audit` - action eligibility and per-arm state/log reconciliation.

The workbook is observational. Values labelled `derived_for_view_only` are
recomputed solely for verification and are never written back to the runtime.
