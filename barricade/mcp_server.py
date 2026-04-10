from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

from mcp.server.fastmcp import FastMCP

from ._benchmarking import compact_benchmark_result
from .dispatch import barricade_dispatch
from .executor import begin_execution as begin_execution_session
from .executor import complete_execution as complete_execution_session
from .executor import read_artifact as read_execution_artifact
from .executor import submit_step as submit_execution_step
from .executor import verify_step as verify_execution_step
from .executor import view_market as view_execution_market
from .executor._protocol import _payload_metrics
from . import runtime as runtime_module
from .scaling import analyze_scaling_profile as analyze_scaling_diagnostics
from .workflow import run_unified_workflow
from ._shared import (
    as_mapping,
    as_string_list,
    as_macro_library,
    dna_summary as _format_dna_summary,
    is_unresolved_macro_token,
)
from ._validation import (
    validate_solve_problem,
    validate_begin_execution,
    validate_manage_execution,
    validate_dispatch_plan,
    validate_analyze_scaling_profile,
    validate_workspace_root,
)
from ._state_inspector import inspect_state as _inspect_state
from ._version import API_VERSION, with_api_version


class ToolDescription(TypedDict):
    name: str
    purpose: str
    when_to_use: str
    inputs: list[str]
    output: str
    api_version: str


DEFAULT_BARRICADE_STATE_DIR = ".barricade_state"
BARRICADE_REPO_ROOT = Path(__file__).resolve().parents[1]


def benchmark_contract() -> dict[str, Any]:
    return with_api_version(runtime_module.benchmark_contract())


def build_optimizer_frame(flat: list[str]) -> dict[str, Any]:
    return with_api_version(runtime_module.build_optimizer_frame(flat))


def build_patch_skeleton(
    task_pool: list[dict[str, Any]], feed_prior_dna: list[str]
) -> dict[str, Any]:
    return with_api_version(
        runtime_module.build_patch_skeleton(task_pool, feed_prior_dna)
    )


def derive_feed_dna_prior(task_pool: list[dict[str, Any]]) -> list[str]:
    return cast(list[str], runtime_module.derive_feed_dna_prior(task_pool))


def derive_task_ecology(feed: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], runtime_module.derive_task_ecology(feed))


def mine_macros_from_elites(
    elite_flattened: list[list[str]], max_macros: int = 8
) -> dict[str, Any]:
    return with_api_version(
        runtime_module.mine_macros_from_elites(elite_flattened, max_macros, (3, 4))
    )


def run_benchmark(
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    feed: dict[str, Any] | None = None,
    state_dir: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return with_api_version(
        runtime_module.run_benchmark(
            trials=trials,
            population=population,
            base_episodes=base_episodes,
            generations=generations,
            seed0=seed0,
            feed=feed,
            state_dir=state_dir,
            config=config,
        )
    )


def _default_barricade_state_dir(workspace_root: str | Path | None = None) -> Path:
    if workspace_root:
        base_root = Path(workspace_root).expanduser().resolve()
    else:
        base_root = BARRICADE_REPO_ROOT
    return base_root / DEFAULT_BARRICADE_STATE_DIR


TOOL_DESCRIPTIONS: list[ToolDescription] = [
    {
        "name": "run_benchmark_task",
        "purpose": "Run the Barricade benchmark and return a compact result by default.",
        "when_to_use": "Use for baseline checks, regression tracking, or stateful benchmark runs. Set compact=False when you need the raw benchmark payload.",
        "inputs": [
            "trials",
            "population",
            "base_episodes",
            "generations",
            "seed0",
            "feed_json",
            "state_dir",
            "config",
            "compact",
            "detail_limit",
        ],
        "output": "Compact benchmark summary with top artifacts and diagnostics, or the full raw payload when compact is false.",
        "api_version": API_VERSION,
    },
    {
        "name": "dispatch_plan",
        "purpose": "Stage, verify, and optionally commit a governed dispatch plan.",
        "when_to_use": "Use only when you have an explicit update set and a verification command.",
        "inputs": ["plan_json", "workspace_root", "commit"],
        "output": "Dry-run preview, verification result, or committed file updates.",
        "api_version": API_VERSION,
    },
    {
        "name": "solve_problem",
        "purpose": "Classify a natural-language task, synthesize a workflow, and return execution-ready DNA plus compact protocol metadata.",
        "when_to_use": "Use as the main MCP entry point for day-to-day problem intake, workflow synthesis, and to seed the execution protocol. When omitted, the state directory defaults to the repo-local .barricade_state folder.",
        "inputs": [
            "problem_text",
            "context_json",
            "feed_json",
            "dispatch_plan_json",
            "workspace_root",
            "state_dir",
            "commit",
            "prior_strength",
            "config",
        ],
        "output": "Structured intake, benchmark-backed synthesis, prior_profile, surfaced feed_prior_dna, dna_summary, patch_token_outline, workspace_root, state_dir, and optional execution report.",
        "api_version": API_VERSION,
    },
    {
        "name": "analyze_scaling_profile",
        "purpose": "Diagnose scaling and phase-pressure signals from a benchmark result payload.",
        "when_to_use": "Use after benchmark runs to inspect objective pressure, phase drift, reward candidates, and diversity floors.",
        "inputs": ["result_json", "baseline_json"],
        "output": "Telemetry report covering pressure, phase detection, reward weights, and diversity risk.",
        "api_version": API_VERSION,
    },
    {
        "name": "begin_execution",
        "purpose": "Expand the DNA into a literal execution session and return the first step instructions.",
        "when_to_use": "Use after solve_problem when you want Barricade to turn DNA into a stepwise host-model protocol. When omitted, the execution session uses the repo-local .barricade_state folder.",
        "inputs": ["synthesis_result_json", "state_dir"],
        "output": "Execution session identifier, session-delta cursor, flattened DNA, first step instructions, and compact session metadata.",
        "api_version": API_VERSION,
    },
    {
        "name": "manage_execution",
        "purpose": "Advance or inspect an execution session with submit, verify, read, report, or complete actions.",
        "when_to_use": "Use after begin_execution to submit artifacts, run verification, inspect the market, read artifacts, report status, or complete the session.",
        "inputs": [
            "session_id",
            "action",
            "content",
            "command",
            "artifact_id",
            "limit",
        ],
        "output": "Step result, execution report, or completion payload depending on action.",
        "api_version": API_VERSION,
    },
    {
        "name": "describe_tools",
        "purpose": "Return Barricade MCP tool descriptions in a structured, client-friendly format.",
        "when_to_use": "Use when a client or operator needs to discover what the Barricade server can do.",
        "inputs": [],
        "output": "List of tool descriptions, inputs, usage notes, and outputs.",
        "api_version": API_VERSION,
    },
    {
        "name": "inspect_state",
        "purpose": "Inspect persistent state — learned macros, motif patterns, and historical run data.",
        "when_to_use": "Use before starting work to understand what the system has learned from past sessions on this codebase. When omitted, it inspects the repo-local .barricade_state folder.",
        "inputs": ["state_dir", "max_macros", "max_motifs", "max_runs"],
        "output": "Learned macros, motif frequencies, recent run summaries, and actionable context hints.",
        "api_version": API_VERSION,
    },
]


mcp = FastMCP("Barricade")


def _load_json_object(payload: str) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}


def _build_execution_seed(result: dict[str, Any]) -> dict[str, Any]:
    synthesis = as_mapping(result.get("synthesis"))
    if not synthesis:
        return {}

    policy = as_mapping(result.get("decision_policy"))
    if not policy:
        policy = as_mapping(synthesis.get("decision_policy"))
    policy_mode = str(policy.get("mode", "act") or "act")
    if policy_mode != "act":
        seed: dict[str, Any] = {
            "dna": [],
            "dna_summary": "",
            "problem_text": str(result.get("problem_text", "")),
            "tool_hint": "clarify_problem"
            if policy_mode == "ask"
            else "stop_execution",
            "next_instruction": str(
                policy.get(
                    "next_instruction",
                    "Clarify the task before beginning execution."
                    if policy_mode == "ask"
                    else "Stop execution for this task.",
                )
            ),
            "decision_policy": policy,
        }
        clarifying_questions = policy.get("clarifying_questions", [])
        if (
            policy_mode == "ask"
            and isinstance(clarifying_questions, list)
            and clarifying_questions
        ):
            seed["clarifying_questions"] = [
                str(question) for question in clarifying_questions if str(question)
            ]
        return seed

    dna_container = as_mapping(result.get("dna"))
    raw_dna = as_string_list(dna_container.get("feed_prior_dna"))
    if not raw_dna:
        raw_dna = as_string_list(synthesis.get("feed_prior_dna"))

    learned_macros = as_macro_library(synthesis.get("learned_macros"))

    execution_dna: list[str] = []
    omitted_macros: list[str] = []
    for token in raw_dna:
        if is_unresolved_macro_token(token, learned_macros):
            omitted_macros.append(token)
            continue
        execution_dna.append(token)

    execution_seed: dict[str, Any] = {
        "dna": execution_dna,
        "dna_summary": _format_dna_summary(execution_dna),
        "problem_text": str(result.get("problem_text", "")),
        "tool_hint": "begin_execution",
        "next_instruction": "Call begin_execution(synthesis_result_json) to expand the DNA into a literal execution session.",
        "decision_policy": policy,
    }
    if omitted_macros:
        execution_seed["omitted_macros"] = omitted_macros
        execution_seed["execution_note"] = (
            "Unresolved LM tokens were omitted from the host-facing execution seed."
        )
    return execution_seed


def _complete_execution_payload(session_id: str) -> dict[str, Any]:
    payload = with_api_version(complete_execution_session(session_id))

    market = payload.get("market")
    market_count = int(payload.get("market_count", 0) or 0)

    if isinstance(market, list) and market:
        if market_count != len(market):
            payload["market_count"] = len(market)
            completion_summary = payload.get("completion_summary")
            if isinstance(completion_summary, dict):
                completion_summary["market_count"] = len(market)
                payload["completion_summary"] = completion_summary
            payload["payload_metrics"] = _payload_metrics(payload)
        return payload

    live_market = view_execution_market(session_id, limit=10000)
    recovered_market = live_market.get("market")
    if isinstance(recovered_market, list) and recovered_market:
        recovered_count = int(
            live_market.get("market_count", len(recovered_market))
            or len(recovered_market)
        )
        payload["market"] = recovered_market
        payload["market_count"] = recovered_count
        completion_summary = payload.get("completion_summary")
        if isinstance(completion_summary, dict):
            if not completion_summary.get("artifact_count"):
                completion_summary["artifact_count"] = recovered_count
            completion_summary["market_count"] = recovered_count
            payload["completion_summary"] = completion_summary
        payload["payload_metrics"] = _payload_metrics(payload)
    return payload


def _surface_workflow_dna(result: dict[str, Any]) -> dict[str, Any]:
    synthesis = as_mapping(result.get("synthesis"))
    if not synthesis:
        return result

    feed_prior_dna = as_string_list(synthesis.get("feed_prior_dna"))
    patch_skeleton = as_mapping(synthesis.get("patch_skeleton"))
    patch_token_outline = as_string_list(patch_skeleton.get("token_outline"))
    dna_summary = _format_dna_summary(feed_prior_dna)

    result["feed_prior_dna"] = feed_prior_dna
    result["patch_skeleton"] = patch_skeleton
    result["patch_token_outline"] = patch_token_outline
    result["dna_summary"] = dna_summary
    result["dna"] = {
        "feed_prior_dna": feed_prior_dna,
        "dna_summary": dna_summary,
        "patch_token_outline": patch_token_outline,
    }
    return result


@mcp.tool()
def run_benchmark_task(
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    feed_json: str = "",
    state_dir: str = "",
    config: str = "",
    compact: bool = True,
    detail_limit: int = 3,
) -> dict[str, Any]:
    """Run the Barricade benchmark and return a compact summary by default.

    Use this for regression checks, baseline comparisons, and stateful benchmark runs.
    Set compact=False when you need the raw benchmark payload. The call itself is
    read-only unless you explicitly provide a state directory.
    """
    feed = _load_json_object(feed_json) if feed_json else None
    config_payload = _load_json_object(config) if config else None
    benchmark_result = run_benchmark(
        trials=trials,
        population=population,
        base_episodes=base_episodes,
        generations=generations,
        seed0=seed0,
        feed=feed,
        state_dir=state_dir or None,
        config=config_payload,
    )
    if compact:
        return with_api_version(
            compact_benchmark_result(benchmark_result, detail_limit)
        )
    return with_api_version(benchmark_result)


@mcp.tool()
def dispatch_plan(
    plan_json: str, workspace_root: str = "", commit: bool = False
) -> dict[str, Any]:
    """Stage, verify, and optionally commit a governed dispatch plan.

    Use this only when you have an explicit file update set and a verification command.
    The tool will dry-run by default and only commit when verification succeeds.
    """
    validate_dispatch_plan(plan_json)
    plan = _load_json_object(plan_json)
    return with_api_version(
        barricade_dispatch(plan, workspace_root=workspace_root or None, commit=commit)
    )


@mcp.tool()
def solve_problem(
    problem_text: str,
    context_json: str = "",
    feed_json: str = "",
    dispatch_plan_json: str = "",
    workspace_root: str = "",
    state_dir: str = "",
    commit: bool = False,
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    prior_strength: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a natural-language task, synthesize a workflow, and optionally execute a dispatch plan.

    This is the main Barricade MCP entry point for regular work: it turns problem text
    into structured intake, benchmark-backed synthesis, surfaced DNA order, and an execution seed for the host model.
    """
    validate_solve_problem(
        problem_text,
        prior_strength=prior_strength,
        trials=trials,
        population=population,
        base_episodes=base_episodes,
        generations=generations,
    )
    context = _load_json_object(context_json) if context_json else None
    feed = _load_json_object(feed_json) if feed_json else None
    dispatch_plan = (
        _load_json_object(dispatch_plan_json) if dispatch_plan_json else None
    )
    workflow_result = run_unified_workflow(
        problem_text,
        context=context,
        feed=feed,
        state_dir=state_dir
        or _default_barricade_state_dir(validate_workspace_root(workspace_root)),
        trials=trials,
        population=population,
        base_episodes=base_episodes,
        generations=generations,
        seed0=seed0,
        prior_strength=prior_strength,
        config=config,
        dispatch_plan=dispatch_plan,
        workspace_root=workspace_root or None,
        commit=commit,
    )
    result = _surface_workflow_dna(workflow_result)
    validated_workspace_root = validate_workspace_root(workspace_root)
    resolved_state_dir = state_dir or _default_barricade_state_dir(
        validated_workspace_root
    )
    result["state_dir"] = str(resolved_state_dir)
    if validated_workspace_root is not None:
        result["workspace_root"] = str(validated_workspace_root)
    result["execution_seed"] = _build_execution_seed(result)
    return with_api_version(result)


@mcp.tool()
def begin_execution(synthesis_result_json: str, state_dir: str = "") -> dict[str, Any]:
    """Expand the DNA into a literal execution session and return the first step instructions.

    Use after solve_problem when you want Barricade to turn DNA into a stepwise host-model protocol.
    """
    validate_begin_execution(synthesis_result_json)
    result = _load_json_object(synthesis_result_json)
    validated_workspace_root = validate_workspace_root(result.get("workspace_root"))
    if validated_workspace_root is not None:
        result["workspace_root"] = str(validated_workspace_root)
    resolved_state_dir = (
        state_dir
        or result.get("state_dir")
        or _default_barricade_state_dir(validated_workspace_root)
    )
    return with_api_version(
        begin_execution_session(result, state_dir=resolved_state_dir)
    )


@mcp.tool()
def manage_execution(
    session_id: str,
    action: str,
    content: str = "",
    command: str = "",
    artifact_id: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    """Advance or inspect an execution session with a single action-based tool.

    Use `submit`, `verify`, `read`, `report`, or `complete` to keep the public MCP
    surface compact while preserving the execution flow and the debug report.
    """
    validate_manage_execution(
        session_id, action, content=content, command=command, artifact_id=artifact_id
    )
    normalized_action = action.strip().lower()
    if normalized_action in {"submit", "step"}:
        if not content:
            raise ValueError("submit action requires content")
        return with_api_version(submit_execution_step(session_id, content))
    if normalized_action == "verify":
        if not command:
            raise ValueError("verify action requires command")
        return with_api_version(verify_execution_step(session_id, command))
    if normalized_action in {"read", "artifact"}:
        if not artifact_id:
            raise ValueError("read action requires artifact_id")
        return with_api_version(read_execution_artifact(session_id, artifact_id))
    if normalized_action in {"report", "market", "status"}:
        return with_api_version(view_execution_market(session_id, limit=limit))
    if normalized_action == "complete":
        return _complete_execution_payload(session_id)
    raise ValueError(f"unknown execution action: {action}")


@mcp.tool()
def analyze_scaling_profile(
    result_json: str, baseline_json: str = ""
) -> dict[str, Any]:
    """Diagnose scaling and phase-pressure signals from a benchmark result payload.

    Use this after benchmark runs to inspect objective pressure, regime drift,
    reward-weight candidates, and diversity-floor risk without changing the engine.
    """
    validate_analyze_scaling_profile(result_json)
    result = _load_json_object(result_json)
    baseline = _load_json_object(baseline_json) if baseline_json else None
    return with_api_version(analyze_scaling_diagnostics(result, baseline))


def derive_task_ecology_from_feed(feed_json: str) -> list[dict[str, Any]]:
    """Derive the bounded task ecology from a feed JSON payload.

    Use this when you need the task pool that will drive the synthesis loop.
    """
    feed = _load_json_object(feed_json)
    return derive_task_ecology(feed)


def derive_feed_prior(feed_json: str) -> list[str]:
    """Convert a feed JSON payload into a soft DNA prior.

    Use this when you want the feed shape translated into a starting DNA sequence.
    """
    feed = _load_json_object(feed_json)
    return derive_feed_dna_prior(derive_task_ecology(feed))


def build_patch_skeleton_from_feed(feed_json: str) -> dict[str, Any]:
    """Build the patch-oriented transfer skeleton for a feed JSON payload.

    Use this when you need a structured scaffold for a patch-oriented task.
    """
    feed = _load_json_object(feed_json)
    tasks = derive_task_ecology(feed)
    prior = derive_feed_dna_prior(tasks)
    return build_patch_skeleton(tasks, prior)


def summarize_optimizer_frame(tokens: list[str]) -> dict[str, Any]:
    """Summarize a token trace into optimizer-frame metrics.

    Use this when you want a compact readout of token structure, combinators, and geometry.
    """
    return build_optimizer_frame(tokens)


def get_benchmark_contract() -> dict[str, Any]:
    """Return the benchmark contract exposed by Barricade.

    Use this when a client needs the required inputs, outputs, and helper surface.
    """
    return benchmark_contract()


def mine_macros(tokens: list[list[str]], max_macros: int = 8) -> dict[str, list[str]]:
    """Mine reusable macro candidates from elite token traces.

    Use this when you want to compress repeated elite token patterns into learned macros.
    """
    return mine_macros_from_elites(tokens, max_macros=max_macros)


@mcp.tool()
def describe_tools() -> list[ToolDescription]:
    """Return Barricade MCP tool descriptions in a structured, client-friendly format.

    Use this when a client or operator needs to discover what the Barricade server can do.
    """
    return TOOL_DESCRIPTIONS


@mcp.tool()
def inspect_state(
    state_dir: str = "",
    max_macros: int = 20,
    max_motifs: int = 20,
    max_runs: int = 10,
) -> dict[str, Any]:
    """Inspect persistent state — learned macros, motif patterns, and historical run data.

    Use this before starting work to understand what the system has learned from past
    sessions on this codebase. Reveals macro vocabulary, common motifs, and failure patterns.
    """
    resolved_state_dir = state_dir or _default_barricade_state_dir()
    return with_api_version(
        _inspect_state(
            resolved_state_dir,
            max_macros=max_macros,
            max_motifs=max_motifs,
            max_runs=max_runs,
        )
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
