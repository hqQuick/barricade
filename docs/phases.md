# Development Phases

A record of what was built and when. Each phase added a capability and left tests behind.

## Phase Summary

| Phase   | Added                      | Lasting value                                                                   | Tests                                                                                                                                                                                     |
| ------- | -------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 | Persistence                | Learned macros, motifs, run logs, and lineage archives are opt-in and reusable. | `tests/test_phase1_persistence.py`                                                                                                                                                        |
| Phase 2 | Dispatch                   | File updates are staged in a clean copy and verified before commit.             | `tests/test_phase2_dispatch.py`                                                                                                                                                           |
| Phase 3 | Unified workflow           | A single `solve_problem` entry point with readiness gating.                     | `tests/test_phase3_workflow.py`, `tests/test_mcp_server.py`, `tests/test_mcp_tool_descriptions.py`, `tests/test_extreme_validation.py`, `tests/test_mcp_target_scenarios.py`              |
| Phase 4 | Scaling diagnostics        | Read-only benchmark diagnostics and comparison tooling.                         | `tests/test_phase4_scaling.py`, `tests/test_rotation_market_signal.py`                                                                                                                    |
| Phase 5 | Code quality and structure | Shared helpers, timeout-safe verification, and structured parsing.              | `tests/test_new_capabilities.py`, `tests/test_e2e_execution.py`, `tests/test_mcp_server.py`, `tests/test_performance.py`, `tests/test_action_layer.py`, `tests/test_barricade_runtime.py` |

## Phase 1: Persistence

**Goal:** Let the system remember what it learned between runs.

**What changed:**

- Added `state_dir` parameter to the benchmark pipeline
- Saves learned macros to `discoverables/macro_library.json`
- Saves motif caches to `discoverables/motif_cache.json`
- Saves run summaries to `runs.jsonl`
- Saves lineage archives to `lineages.jsonl`

**Key point:** Default behavior is unchanged when `state_dir` is omitted. The persistence layer is opt-in.

**Tests:** `tests/test_phase1_persistence.py`

---

## Phase 2: Dispatch

**Goal:** Let the system actually edit files — safely.

**What changed:**

- Added `barricade/dispatch.py` with `barricade_dispatch`
- Stages changes in a temporary copy, verifies there, only writes back on success
- Creates backups before overwriting
- Added `dispatch_plan` MCP tool

**Key point:** Dry-run by default. Nothing touches your files unless you set `commit=True` AND verification passes.

**Tests:** `tests/test_phase2_dispatch.py`

---

## Phase 3: Unified Workflow

**Goal:** One entry point for natural-language tasks.

**What changed:**

- Added `barricade/workflow.py` with task classification
- Classifies task text into goal, constraints, deliverables, risks, domain tags
- Added `solve_problem` MCP tool — the main entry point
- Synthesis layer reuses the benchmark engine

**Key point:** Turns "add a login endpoint" into structured intake → readiness policy → benchmark-backed DNA → execution-ready session.

**Tests:** `tests/test_phase3_workflow.py`, `tests/test_mcp_server.py`, `tests/test_mcp_tool_descriptions.py`, `tests/test_extreme_validation.py`, `tests/test_mcp_target_scenarios.py`

---

## Phase 4: Scaling Diagnostics

**Goal:** Understand what's happening inside the benchmark.

**What changed:**

- Added `barricade/scaling.py` with read-only diagnostics
- Reports dual-objective pressure, phase transitions, reward candidates, diversity floors
- Added `analyze_scaling_profile` MCP tool

**Key point:** Telemetry only. Doesn't change the benchmark engine — just helps you understand what it's doing.

**Tests:** `tests/test_phase4_scaling.py`, `tests/test_rotation_market_signal.py`

---

## Phase 5: Code Quality & Structure

**Goal:** Clean up the codebase as it grew.

**What changed:**

- Extracted shared helpers into `_shared.py` (eliminated duplication between dispatch, executor, and mcp_server)
- Centralized timeout-safe verification execution in `_shared.py`
- Added compact benchmark summaries and an ablation-study helper so the benchmark surface stays readable by default
- Split monolithic `executor.py` (1,338 lines) into `executor/` package:
  - `_protocol.py` — session delta protocol
  - `_parsing.py` — file block and patch parsing
  - `_scoring.py` — artifact scoring and pricing
  - `registry.py` — ExecutionRegistry class
- Added input validation at MCP boundaries (`_validation.py`)
- Added structured verification output (`_verification_parser.py`)
- Added persistent memory inspection (`_state_inspector.py`)

**Tests:** `tests/test_new_capabilities.py`, `tests/test_e2e_execution.py`, `tests/test_mcp_server.py`, `tests/test_performance.py`, `tests/test_action_layer.py`, `tests/test_barricade_runtime.py`

---

## Phase 6: Extrapolation and Proof Probes

**Goal:** Verify that evolved DNA scaffolds generalize beyond the training distribution.

**What changed:**

- Added `tests/test_extrapolation.py` — structural transfer, generalization gradient, macro essence, failure mode, novel composition, and intervention sensitivity tests
- Added `tests/test_extrapolation_probe_evidence.py` — evidence assertions for the extrapolation probe
- Added `tests/test_proof_probe_evidence.py` — novel capability synthesis, long-term learning, scaffold necessity, and solution quality evidence
- Added exploratory probe scripts: `tests/extrapolation_probe.py`, `tests/emergence_probe.py`, `tests/deep_emergence_probe.py`, `tests/proof_probe.py`

**Key findings:**

- All current prompt families route to a stable scaffold template: `OBSERVE → LM1 → WRITE_PATCH → PLAN → REPAIR → COMMIT → WRITE_PLAN → VERIFY → SUMMARIZE`
- The scaffold transfers to unseen programming tasks (Ackermann, Sieve, Sudoku, Collatz) at the same success rate as training tasks
- Learned macros (LM1) emerge from evolution, not hand-coding
- Long-term learning shows increasing support scores and cache reuse across sessions

**Tests:** `tests/test_extrapolation.py`, `tests/test_extrapolation_probe_evidence.py`, `tests/test_proof_probe_evidence.py`

---

## Current State

129 tests and 9 subtests are passing. The engine works end-to-end from task description to verified file changes, and the scaffold generalizes to unseen programming tasks.
