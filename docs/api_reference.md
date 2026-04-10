# Barricade API Reference

## Overview

Barricade is a governed execution engine for AI-assisted code changes. It exposes two surfaces:
- **MCP Server** — tools callable by an LLM client via the Model Context Protocol
- **Programmatic API** — module-level functions for direct programmatic use

Every public dict-shaped response includes an `api_version` field so clients can pin to the contract they are consuming.

---

## Surface Map

| Surface | Entry Points | Use |
|---|---|---|
| MCP | `solve_problem`, `begin_execution`, `manage_execution`, `dispatch_plan`, `run_benchmark_task`, `analyze_scaling_profile`, `describe_tools`, `inspect_state` | Interactive client workflows and governed execution. |
| Python | `barricade.workflow`, `barricade.runtime`, `barricade.dispatch`, `barricade.executor` | Direct library use and test coverage. |

## Common Response Fields

| Field | Meaning |
|---|---|
| `api_version` | Contract version for the response payload. |
| `status` | Lifecycle state such as synthesized, session_started, verification_failed, or completed. |
| `decision_policy` | Readiness gate with `mode`, `support_score`, `reasons`, and optional `clarifying_questions`. |
| `state_dir` | Resolved persistence root for learned state and run logs. |
| `execution_seed` | Deterministic seed used when the caller needs to reproduce a run. |

## MCP Tools

### `solve_problem`

Classify a natural-language task, synthesize a workflow, and return readiness-gated execution DNA.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `problem_text` | `str` | Yes | The task description in natural language |
| `context_json` | `str` | No | JSON object with additional context |
| `feed_json` | `str` | No | JSON feed for evolutionary benchmark |
| `dispatch_plan_json` | `str` | No | Pre-existing dispatch plan to apply |
| `workspace_root` | `str` | No | Path to the workspace directory |
| `state_dir` | `str` | No | Path for persisting state (macros, motifs); defaults to the repo-local `.barricade_state` folder when omitted |
| `commit` | `bool` | No | Whether to commit changes (default: `false`) |
| `prior_strength` | `float \| None` | No | Prior strength ∈ [0,1]; auto-set by task type |
| `trials` | `int` | No | Benchmark trial count (default: 16) |
| `population` | `int` | No | Evolutionary population size (default: 96) |
| `base_episodes` | `int` | No | Base episode count (default: 18) |
| `generations` | `int` | No | Evolutionary generations (default: 40) |
| `seed0` | `int` | No | Random seed (default: 100901) |

**Returns:** Structured intake, benchmark-backed synthesis, decision policy, `feed_prior_dna`, `dna_summary`, `patch_token_outline`, `execution_seed`, and optional execution report.

**Decision policy:** The `decision_policy` field contains `mode` (`act`, `ask`, or `stop`), `support_score`, `reasons`, and optional `clarifying_questions`. If the mode is `ask` or `stop`, the caller should not begin execution until the ambiguity is resolved.

---

### `begin_execution`

Expand DNA into a literal execution session and return the first step instructions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `synthesis_result_json` | `str` | Yes | JSON output from `solve_problem` |
| `state_dir` | `str` | No | Path for persisting state; defaults to the repo-local `.barricade_state` folder when omitted |

**Returns:** `session_id`, `status`, `current_step`, `instruction`, `tool_hint`, flattened DNA.

---

### `manage_execution`

Advance or inspect an execution session with a single action-based tool.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | `str` | Yes | Session identifier from `begin_execution` |
| `action` | `str` | Yes | One of: `submit`, `verify`, `read`, `report`, `complete` |
| `content` | `str` | Conditional | Required for `submit` action |
| `command` | `str` | Conditional | Required for `verify` action |
| `artifact_id` | `str` | Conditional | Required for `read` action |
| `limit` | `int` | No | Market snapshot limit (default: 8) |

**Actions:**
- `submit` — Submit artifact content for the current step
- `verify` — Run a verification command
- `read` — Read a specific artifact by ID
- `report` — View the current market snapshot
- `complete` — Finalize the session and produce a dispatch plan

---

### `dispatch_plan`

Stage, verify, and optionally commit a governed dispatch plan.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan_json` | `str` | Yes | JSON with `updates` mapping and optional `verification_command` |
| `workspace_root` | `str` | No | Target workspace directory |
| `commit` | `bool` | No | Whether to commit after verification (default: `false`) |

**Returns:** Dry-run preview, verification result, or committed file updates.

The dispatch tool is intentionally narrow: it stages changes in a clean copy, verifies them, and only writes back when the check passes.

---

### `run_benchmark_task`

Run the Barricade benchmark and return a compact summary by default.

Use `compact=false` when you need the raw benchmark payload. The compact path keeps the top artifacts and diagnostics while dropping the large archive and outcome blobs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trials` | `int` | 16 | Number of benchmark trials |
| `population` | `int` | 96 | Evolutionary population size |
| `base_episodes` | `int` | 18 | Base episode count |
| `generations` | `int` | 40 | Evolutionary generations |
| `seed0` | `int` | 100901 | Random seed |
| `feed_json` | `str` | `""` | Optional feed JSON |
| `state_dir` | `str` | `""` | Optional state directory |
| `config` | `str` | `""` | Optional JSON config for the benchmark run |
| `compact` | `bool` | `true` | Return the compact summary instead of the raw payload |
| `detail_limit` | `int` | `3` | Number of archive entries to keep in the compact summary |

**Returns:** A compact benchmark summary with `summary`, diagnostics, top artifacts, and outcome summary fields, or the full raw benchmark payload when `compact=false`.

---

### `run_ablation_study`

Run the benchmark against feature-flag ablations and compare each variant to the baseline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trials` | `int` | 16 | Number of benchmark trials |
| `population` | `int` | 96 | Evolutionary population size |
| `base_episodes` | `int` | 18 | Base episode count |
| `generations` | `int` | 40 | Evolutionary generations |
| `seed0` | `int` | 100901 | Random seed |
| `feed` | `dict` | `None` | Optional feed payload |
| `state_dir` | `str` | `None` | Optional state directory |
| `config` | `dict \| EvolutionConfig` | `None` | Baseline config to ablate from |
| `ablation_modes` | `tuple[str, ...]` | `None` | Which ablation modes to run |
| `detail_limit` | `int` | `3` | Number of archive entries to keep in summaries |

**Returns:** A compact baseline summary plus one comparison row per ablation mode.

---

### `analyze_scaling_profile`

Diagnose scaling and phase-pressure signals from a benchmark result payload.

Use this when you want to compare a candidate result against a baseline result and see which one won, why, and by how much.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `result_json` | `str` | Yes | Benchmark result JSON |
| `baseline_json` | `str` | No | Baseline result JSON for comparison |

---

### `describe_tools`

Return all MCP tool descriptions in a structured format.

**Returns:** List of `ToolDescription` objects with `name`, `purpose`, `when_to_use`, `inputs`, `output`, and `api_version`.

### `inspect_state`

Inspect the persisted product memory for learned macros, priors, motif caches, and recent run summaries.

**Returns:** A structured snapshot of the current state directory.

---

## Programmatic API

### `barricade.executor`

| Function | Signature | Description |
|----------|-----------|-------------|
| `begin_execution` | `(synthesis_result: dict \| str, state_dir: str \| Path \| None = None) -> dict` | Start an execution session |
| `submit_step` | `(session_id: str, content: str) -> dict` | Submit artifact content |
| `read_artifact` | `(session_id: str, artifact_id: str) -> dict` | Read a specific artifact |
| `view_market` | `(session_id: str, limit: int = 8) -> dict` | View market snapshot |
| `verify_step` | `(session_id: str, command: str) -> dict` | Run verification |
| `complete_execution` | `(session_id: str) -> dict` | Finalize and produce dispatch plan |

### `barricade.runtime`

| Function | Signature | Description |
|----------|-----------|-------------|
| `run_benchmark` | `Callable[..., dict]` | The benchmark runner used by the evolution pipeline. |
| `run_benchmark_comparison` | `(trials: int = 16, population: int = 96, base_episodes: int = 18, generations: int = 40, seed0: int = 100901, feed: dict[str, Any] | None = None, state_dir: str | None = None, baseline_config: dict[str, Any] | None = None, candidate_config: dict[str, Any] | None = None, baseline_label: str = "baseline", candidate_label: str = "candidate", compact: bool = True, detail_limit: int = 3) -> dict` | Compare a baseline and candidate run on the same feed and seed. The default return is compact; pass `compact=False` for the raw benchmark payloads. |
| `run_ablation_study` | `(trials: int = 16, population: int = 96, base_episodes: int = 18, generations: int = 40, seed0: int = 100901, feed: dict[str, Any] | None = None, state_dir: str | None = None, config: dict[str, Any] | EvolutionConfig | None = None, ablation_modes: tuple[str, ...] | None = None, detail_limit: int = 3) -> dict` | Compare the baseline against feature-flag ablations and return compact comparison rows. |

### `barricade.workflow`

| Function | Signature | Description |
|----------|-----------|-------------|
| `run_unified_workflow` | `(problem_text: str, *, context: dict[str, Any] | None = None, feed: dict[str, Any] | None = None, state_dir: str | Path | None = None, trials: int = 16, population: int = 96, base_episodes: int = 18, generations: int = 40, seed0: int = 100901, prior_strength: float | None = None, config: dict[str, Any] | None = None, dispatch_plan: dict[str, Any] | str | Path | None = None, workspace_root: str | Path | None = None, commit: bool = False) -> dict` | Build the full readiness-gated synthesis result. |

### `barricade.executor.Artifact`

Dataclass representing a single artifact on the market.

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | Unique identifier (e.g. `OBS_01`, `PATC_02`) |
| `token` | `str` | Token type (e.g. `OBSERVE`, `WRITE_PATCH`) |
| `kind` | `str` | Artifact kind (e.g. `observation`, `patch`) |
| `content` | `str` | Artifact body text |
| `creator` | `str` | Creator identifier |
| `epoch` | `int` | Step number when created |
| `price` | `float` | Market price |
| `score` | `float` | Quality score |
| `status` | `str` | `submitted`, `passed`, or `failed` |
| `metadata` | `dict` | Additional scoring and trace metadata |

### `barricade.executor.ExecutionSession`

Dataclass tracking the state of an execution run.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Unique session identifier |
| `dna` | `list[str]` | Original DNA sequence |
| `flattened` | `list[str]` | Expanded token sequence |
| `current_step` | `int` | Current position in the trace |
| `market` | `dict[str, Artifact]` | All artifacts produced |
| `patch_updates` | `dict[str, str]` | Accumulated file updates |
| `completed` | `bool` | Whether the session is finished |

---

## Dataclasses

### `barricade.workflow.WorkflowIntake`

Parsed task classification from the workflow intake layer.

| Field | Type | Description |
|-------|------|-------------|
| `raw_task` | `str` | Original task text |
| `goal` | `str` | Extracted goal |
| `constraints` | `list[str]` | Identified constraints |
| `deliverables` | `list[str]` | Expected deliverables |
| `risks` | `list[str]` | Identified risks |
| `expected_artifact_type` | `str` | `plan`, `patch`, `summary`, or `test` |
| `confidence` | `float` | Classification confidence [0, 1] |
| `domain_tags` | `list[str]` | Detected domain tags |

---

## DNA Tokens

The execution engine recognizes 20 tokens in two groups. The primary workflow traces use a subset of these tokens, but the full vocabulary is kept in code and surfaced here for completeness.

### Primitive Tokens

| Token | Description |
|-------|-------------|
| `OBSERVE` | Register task context and constraints |
| `RETRIEVE` | Fetch relevant artifacts or context |
| `VERIFY` | Run verification or a command |
| `REPAIR` | Describe correction after failure |
| `SUMMARIZE` | Summarize results and residual risk |
| `CACHE` | Cache state for future reuse |
| `COMMIT` | Prepare governed commit handoff |
| `ROLLBACK` | Describe or enact rollback |
| `SWITCH_CONTEXT` | Switch to a different execution context |
| `ESCALATE` | Escalate the task or risk signal |
| `QUERY_TOOL` | Ask an external tool for information |
| `PLAN` | Produce a decomposition of work |

### Artifact Ops

| Token | Description |
|-------|-------------|
| `WRITE_NOTE` | Write a note artifact |
| `WRITE_PLAN` | Write an architectural plan artifact |
| `WRITE_PATCH` | Write a patch artifact |
| `WRITE_SUMMARY` | Write a summary artifact |
| `READ_ARTIFACT` | Read a specific artifact |
| `REVISE_ARTIFACT` | Revise an existing artifact |
| `LINK_ARTIFACT` | Link an artifact to earlier evidence |
| `DROP_ARTIFACT` | Drop an artifact from consideration |

## Base Macros

Pre-defined macro sequences available in every session:

| Macro | Sequence |
|-------|----------|
| `DISCOVER` | `OBSERVE → RETRIEVE → VERIFY` |
| `SAFE_REPORT` | `VERIFY → REPAIR → SUMMARIZE` |
| `RECOVER` | `VERIFY → ROLLBACK → REPAIR` |
| `MEMOIZE` | `CACHE → RETRIEVE → COMMIT` |
| `PATCH_LOOP` | `PLAN → REPAIR → VERIFY` |
| `DIAGNOSE` | `SWITCH_CONTEXT → VERIFY → REPAIR` |
| `ARTIFACT_PLAN` | `WRITE_PLAN → READ_ARTIFACT → PLAN` |
| `ARTIFACT_PATCH` | `WRITE_PATCH → READ_ARTIFACT → REPAIR` |
| `ARTIFACT_SUMMARY` | `WRITE_SUMMARY → READ_ARTIFACT → SUMMARIZE` |
