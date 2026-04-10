from __future__ import annotations

import json
import time

from barricade import mcp_server


BENCHMARK_BEGIN_EXECUTION_S = 0.5
BENCHMARK_COMPLETE_EXECUTION_S = 1.0
BENCHMARK_SUBMIT_STEP_S = 0.2


def _make_synthesis_result(workspace_root, state_dir) -> dict:
    return {
        "problem_text": "Performance smoke test.",
        "workspace_root": str(workspace_root),
        "state_dir": str(state_dir),
        "intake": {
            "raw_task": "Performance smoke test.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "WRITE_PATCH", "VERIFY", "SUMMARIZE"],
            "patch_skeleton": {
                "token_outline": ["WRITE_PATCH", "VERIFY", "SUMMARIZE"],
            },
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }


def test_begin_execution_performance(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)

    start = time.perf_counter()
    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    elapsed = time.perf_counter() - start

    assert elapsed < BENCHMARK_BEGIN_EXECUTION_S, (
        f"begin_execution took {elapsed:.3f}s (threshold: {BENCHMARK_BEGIN_EXECUTION_S}s)"
    )
    assert session["status"] == "started"


def test_submit_step_performance(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)
    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    start = time.perf_counter()
    result = mcp_server.manage_execution(
        session_id, action="submit", content="Observation content."
    )
    elapsed = time.perf_counter() - start

    assert elapsed < BENCHMARK_SUBMIT_STEP_S, (
        f"submit_step took {elapsed:.3f}s (threshold: {BENCHMARK_SUBMIT_STEP_S}s)"
    )
    assert result["status"] == "step_recorded"


def test_complete_execution_performance(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = {
        "problem_text": "Performance smoke test.",
        "workspace_root": str(workspace_root),
        "state_dir": str(state_dir),
        "intake": {
            "raw_task": "Performance smoke test.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "WRITE_PATCH", "SUMMARIZE"],
            "patch_skeleton": {
                "token_outline": ["WRITE_PATCH", "SUMMARIZE"],
            },
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }
    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    mcp_server.manage_execution(session_id, action="submit", content="Observation.")
    mcp_server.manage_execution(session_id, action="submit", content="Patch content.")
    mcp_server.manage_execution(session_id, action="submit", content="Summary content.")

    start = time.perf_counter()
    result = mcp_server.manage_execution(session_id, action="complete")
    elapsed = time.perf_counter() - start

    assert elapsed < BENCHMARK_COMPLETE_EXECUTION_S, (
        f"complete_execution took {elapsed:.3f}s (threshold: {BENCHMARK_COMPLETE_EXECUTION_S}s)"
    )
    assert result["status"] == "completed"


def test_execution_session_scales_linearly(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = {
        "problem_text": "Performance smoke test.",
        "workspace_root": str(workspace_root),
        "state_dir": str(state_dir),
        "intake": {
            "raw_task": "Performance smoke test.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": [
                "OBSERVE",
                "WRITE_PATCH",
                "SUMMARIZE",
                "OBSERVE",
                "WRITE_PATCH",
                "SUMMARIZE",
                "OBSERVE",
                "WRITE_PATCH",
                "SUMMARIZE",
                "OBSERVE",
                "WRITE_PATCH",
                "SUMMARIZE",
            ],
            "patch_skeleton": {
                "token_outline": ["WRITE_PATCH", "SUMMARIZE"],
            },
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }
    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    steps = 12
    start = time.perf_counter()
    for i in range(steps):
        mcp_server.manage_execution(
            session_id, action="submit", content=f"Step {i} content."
        )
    elapsed = time.perf_counter() - start

    per_step = elapsed / steps
    assert per_step < BENCHMARK_SUBMIT_STEP_S * 2, (
        f"Average step took {per_step:.3f}s (threshold: {BENCHMARK_SUBMIT_STEP_S * 2}s)"
    )
