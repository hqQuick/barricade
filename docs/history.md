# Research History: v1 to v3.14

> Archived lineage. Use [README.md](../README.md), [docs/architecture.md](architecture.md), [docs/tests.md](tests.md), [docs/api_reference.md](api_reference.md), and [docs/phases.md](phases.md) for current behavior.

| Version | Theme | What Changed |
|---|---|---|
| v1 | Baseline | First multi-agent experiments. |
| v2 | Governance | Added explicit rules and thresholds. |
| v3.4 | Consolidation | Locked a stable regime. |
| v3.11 | Calibration | Introduced feed-sensitive residency and prior seeding. |
| v3.14 | Feed prior | Added feed-derived DNA prior and patch skeleton. |

The useful lesson from this lineage is simple: structured task shape, verification, and persisted state beat ad hoc prompting for this codebase.
The direction is working: let the input shape the bounded ecology.

## Comparison Notes

| Scenario | Prior | Result fragment |
|---|---|---|
| Raft node | `OBSERVE -> LM1 -> WRITE_PATCH -> PLAN -> REPAIR -> COMMIT -> WRITE_PLAN -> VERIFY -> SUMMARIZE` | `solve 1, threshold 0.8, stability 0.451` |
| Complex scenario | `OBSERVE -> LM1 -> WRITE_PATCH -> PLAN -> COMMIT -> WRITE_PLAN -> ROLLBACK -> REPAIR -> VERIFY -> SUMMARIZE` | `solve 0.714, threshold 0.5, stability 0.488` |
| Summary memo | `OBSERVE -> LM1 -> PLAN -> RETRIEVE -> WRITE_SUMMARY -> VERIFY -> SUMMARIZE -> WRITE_PLAN` | `solve 1, threshold 0.615, stability 0.405` |
