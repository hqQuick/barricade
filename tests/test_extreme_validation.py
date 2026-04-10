from __future__ import annotations

import json
from pathlib import Path

from barricade import mcp_server


TARGET_PROBLEM = "Patch the repository to add a small parser helper and verify it before commit."
HOLDOUT_PROBLEM = "Prove that the centroid of a triangle lies on the medians and explain the geometry."
TRAINING_PROBLEMS = [
    (
        "Refactor a JSON loader and add validation tests.",
        106,
        "warm-train-1",
    ),
    (
        "Write a short architecture note for the MCP execution flow.",
        107,
        "warm-train-2",
    ),
    (
        "Triage a security issue around leaked secrets and explain the remediation path.",
        108,
        "warm-train-3",
    ),
]


def _solve_problem(problem_text: str, workspace_root: Path, state_dir: Path, seed0: int, ticket: str) -> dict:
    return mcp_server.solve_problem(
        problem_text,
        context_json=json.dumps({"ticket": ticket}),
        feed_json=json.dumps({"source": ticket}),
        workspace_root=str(workspace_root),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=seed0,
    )


def _drive_execution(result: dict, state_dir: Path) -> dict:
    if result["decision_policy"]["mode"] != "act":
        return {"acted": False, "steps": 0, "final_status": None}

    session = mcp_server.begin_execution(json.dumps(result), state_dir=str(state_dir))
    session_id = session["session_id"]
    token = session.get("current_token")
    last_artifact_id = None
    steps = 0

    while token is not None and steps < 25:
        if token == "VERIFY":
            payload = mcp_server.manage_execution(
                session_id,
                action="verify",
                command='python3 -c "pass"',
            )
        elif token == "READ_ARTIFACT":
            if not last_artifact_id:
                market = mcp_server.manage_execution(session_id, action="report", limit=3)
                market_entries = market.get("market", [])
                if market_entries:
                    last_artifact_id = market_entries[0].get("artifact_id")
            if not last_artifact_id:
                break
            payload = mcp_server.manage_execution(
                session_id,
                action="read",
                artifact_id=last_artifact_id,
            )
        else:
            payload = mcp_server.manage_execution(
                session_id,
                action="submit",
                content=f"Extreme validation step for {token}.",
            )

        artifact = payload.get("artifact") or payload.get("market_entry") or {}
        last_artifact_id = artifact.get("artifact_id") or payload.get("artifact_id") or last_artifact_id
        token = payload.get("next_token")
        steps += 1

    final_status = None
    if token is None:
        final = mcp_server.manage_execution(session_id, action="complete")
        final_status = final.get("completion_summary", {}).get("status")

    return {"acted": True, "steps": steps, "final_status": final_status}


def _run_problem(problem_text: str, workspace_root: Path, state_dir: Path, seed0: int, ticket: str) -> tuple[dict, dict]:
    result = _solve_problem(problem_text, workspace_root, state_dir, seed0, ticket)
    session = _drive_execution(result, state_dir)
    return result, session


def test_extreme_validation_shows_stateful_learning(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()

    cold_target_state = tmp_path / "cold-target-state"
    cold_heldout_state = tmp_path / "cold-heldout-state"
    warm_state = tmp_path / "warm-state"

    cold_target, _ = _run_problem(
        TARGET_PROBLEM,
        workspace_root,
        cold_target_state,
        seed0=11,
        ticket="cold-target",
    )
    cold_heldout, _ = _run_problem(
        HOLDOUT_PROBLEM,
        workspace_root,
        cold_heldout_state,
        seed0=12,
        ticket="cold-heldout",
    )

    _run_problem(TARGET_PROBLEM, workspace_root, warm_state, seed0=21, ticket="warm-target-1")

    for training_problem, seed0, ticket in TRAINING_PROBLEMS:
        _run_problem(training_problem, workspace_root, warm_state, seed0=seed0, ticket=ticket)

    warm_repeat, _ = _run_problem(
        TARGET_PROBLEM,
        workspace_root,
        warm_state,
        seed0=106,
        ticket="warm-target-2",
    )
    warm_heldout, _ = _run_problem(
        HOLDOUT_PROBLEM,
        workspace_root,
        warm_state,
        seed0=109,
        ticket="warm-heldout",
    )

    cold_summary = mcp_server.inspect_state(state_dir=str(cold_target_state), max_runs=10)
    warm_summary = mcp_server.inspect_state(state_dir=str(warm_state), max_runs=10)

    assert warm_repeat["synthesis"]["task_shape_signature"] == cold_target["synthesis"]["task_shape_signature"]
    assert warm_heldout["synthesis"]["task_shape_signature"] == cold_heldout["synthesis"]["task_shape_signature"]

    assert warm_repeat["decision_policy"]["support_score"] >= cold_target["decision_policy"]["support_score"] - 0.01
    assert warm_repeat["synthesis"]["outcome_memory"]["match_count"] >= cold_target["synthesis"]["outcome_memory"]["match_count"]
    assert warm_repeat["synthesis"]["execution_feedback"]["match_count"] >= cold_target["synthesis"]["execution_feedback"]["match_count"]
    assert len(warm_repeat["synthesis"]["learned_macros"]) >= len(cold_target["synthesis"]["learned_macros"])

    assert warm_heldout["decision_policy"]["support_score"] >= cold_heldout["decision_policy"]["support_score"] - 0.05
    assert warm_summary["summary"]["total_runs"] >= 2 + len(TRAINING_PROBLEMS)
    assert warm_summary["summary"]["feedback_entries"] >= 1
    assert warm_summary["summary"]["macro_count"] >= cold_summary["summary"]["macro_count"]