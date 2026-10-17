# Move-selection old-research cleanup — final verification

Date: 2026-10-16

## Storage and deletion record

- Original size: 5,407,950,477 bytes
- Final size: 1,629,696,004 bytes
- Actual recovered space: 3,778,254,473 bytes
- Audit-approved deletion paths removed: 39 of 39
- Skipped approved paths: 0
- The deletion log contains 39 rows; all are marked `Deleted=Yes`.

The approved deletion targets totalled 3,778,420,572 bytes.  The small difference between that value and measured recovered space is attributable to retained/finalized reporting files and Python bytecode generated during non-destructive verification.

## Current runtime verification

- Normal selector mode resolves to `ordinary-md7-r2-tell-c1-v1`.
- Current live supervised model: **MD7-R2-TELL-C1 Epoch 2** at `models/frozen/md7_r2_tell_c1_epoch2`.
- MD7 loaded with the active Tutor integration Python environment and produced four finite probabilities summing to approximately one for `generic`, `probing`, `focus`, and `telling`.
- Normal execution uses MD7, not MD6.  `FrozenMD6Inference` is a historical loader name; normal mode passes the MD7 directory explicitly.
- MD6 remains only for explicit rollback/historical compatibility.  MD7-R1 remains as the supported rollback candidate.
- The final Turn-LinTS context built successfully with `S+K+L`: S=4, K=3, L=4, total dimension=11.
- Direct Turn-LinTS imported and initialized, selected a valid move, accepted a selected-arm update, and round-tripped its state in memory without touching the production snapshot.
- The `headroom_normalized` reward path is present and returned a finite result.
- MRB1 loaded and scored a representative response into four finite diagnostic scores.  Its role is diagnostic response scoring; it does not directly control move selection, LinTS reward, or LinTS update in the final S+K+L runtime.  The successful isolated Windows smoke disabled Transformers parallel model loading after the default isolated load encountered a native PyTorch/Transformers access violation during parallel weight materialization.
- Tutor selector integration imports successfully and resolves the protected MD7, MD6, and MRB1 paths.

## Protected models retained

| Model | Retained | SHA-256 |
|---|---|---|
| Current MD7-R2-TELL-C1 Epoch 2 | Yes | `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5` |
| MD6 rollback | Yes | `674f1abbaf8c6c23d4b736a80f40b3532e4681c125695ebbe3863e404acb56e7` |
| MD7-R1 rollback candidate | Yes | `d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13` |
| MRB1 diagnostic scorer | Yes | `9ad757e884beae6a242cf6fe17d5b3524e7599e413e9b9cc5f3b999a37a6f1d7` |

## Research preserved

- The MD0–MD13 map and all mapped retained training/evaluation paths are present.  MD3 remains the documented historical training-source gap; its retained evaluation script and validation evidence are present.
- Final LinTS research is preserved, including the 713-turn dataset with 31 learners, 92 attempts, and 22 skills.
- The final context-ablation result is preserved: five-fold `GroupKFold by attempt`, including the selected 11-dimensional S+K+L configuration.
- Reward selection, headroom-normalization analysis, grouped-CV evidence, and preflight/leakage audit are preserved.
- The bootstrap file contains exactly 5,000 learner-level replicate IDs.

## Automated tests and local health

- Move-selection suite: **17 passed, 0 failed** (`pytest packages/move-selection/tests -q -p no:cacheprovider`).
- No new automated-test failures were caused by cleanup.
- At verification time the local frontend was listening on `127.0.0.1:5173`; no listener was present on the usual backend port 8000, so backend readiness was not queried.

## Superseded source cleanup (2026-10-17)

- **Superseded legacy modules:** 7
- **Deleted:** 7 files (85,333 bytes)
- **Stale tests:** the test module dedicated to the retired learner-agency escalation implementation was removed, and its one remaining case/import was removed from the manual controlled-live suite.
- **Stale documentation:** the research artefact map now records the three retired validator modules as removed and points to the retained final preflight audit instead.
- **FinalSKLLive impact:** none. The active MD7-R2-TELL-C1 + direct Turn-LinTS runtime has no import or module-path dependency on the deleted files.
- **Verification:** all 563 Python files parsed, the retired-module AST import count is zero, the move-selection suite passed 17/17, and the unaffected FinalSKLLive regression files passed 42/42. The wider Tutor selection run passed 52 tests; three unrelated snapshot assertions still expect the pre-live counts (98 contexts and 500 updates) while the retained live evidence now contains 174 contexts and 548 updates, including 48 live updates.
- **Preserved evidence:** final evaluation, preflight/leakage audit, grouped-CV, reward-selection, model, and thesis evidence were not deleted.

The verification results for this source cleanup are recorded after the deletion entries in `FINAL_OLD_RESEARCH_DELETION_LOG.csv`. No protected model or retained final research-evidence file was deleted, moved, or renamed.
