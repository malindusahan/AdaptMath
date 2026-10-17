# Pre-unification restore point

Created before schema or runtime-code mutation on 2026-08-30.

- `filesystem/`: raw runtime and research persistence copies
- `sqlite_consistent/tutor_checkpoints.sqlite3`: consistent active Tutor snapshot
- `sqlite_consistent/student_model_active.sqlite3`: consistent active BKT snapshot
- `sqlite_consistent/student_model_root.sqlite3`: consistent fallback BKT snapshot
- `postgres/adaptmath_memory_pre_unification.dump`: custom-format PostgreSQL dump
- `configuration/`: ignored runtime configuration and migration metadata; treat as sensitive

Recorded SHA-256 values:

```text
tutor_checkpoints.sqlite3      F63CFC376F3057DB32FB2E76FE896AA28B454E6B6AF76EB05F9B0575A51701E8
student_model_active.sqlite3   8DE0DB9639A5195482BF9CC7D7FA561A76B0FB783C90C68F8480D4DC73E5CB2E
student_model_root.sqlite3     56F42CF8BFB2AD6BDB2653554CCC3727F0DEFFD261BB872649BB641E0A24DC89
PostgreSQL custom dump         24ED4A31ACA7BA16DCF4140E13DF8B6FC90546CC9B7B98A7935B8B7035D24DE5
```

Recompute and compare the SHA-256 digest before restoration. See
`docs/architecture/postgres-persistence-unification.md` for the inventory and
restore procedure.
Stop every writer before restoring. Never print or commit credentials from the
configuration copy.
