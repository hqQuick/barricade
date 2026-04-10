from __future__ import annotations

import json

import pytest

from barricade import mcp_server


def _make_synthesis_result(workspace_root, state_dir) -> dict:
    return {
        "problem_text": "Add a hello world function.",
        "workspace_root": str(workspace_root),
        "state_dir": str(state_dir),
        "intake": {
            "raw_task": "Add a hello world function.",
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


def test_full_execution_pipeline_without_mocks(tmp_path, monkeypatch) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]
    assert session_id.startswith("exec_")
    assert session["status"] == "started"
    assert session["current_step"] == 0

    result = mcp_server.manage_execution(
        session_id,
        action="submit",
        content="Observed the repo layout.",
    )
    assert result["status"] == "step_recorded"
    assert result["next_token"] is not None

    result = mcp_server.manage_execution(
        session_id,
        action="submit",
        content='{"updates": {"hello.py": "def hello(): return \'world\'"}}',
    )
    assert result["status"] == "step_recorded"

    (workspace_root / "hello.py").write_text("def hello(): return 'world'\n")

    result = mcp_server.manage_execution(
        session_id,
        action="verify",
        command="python3 -c \"import ast; ast.parse('def hello(): pass')\"",
    )
    assert result["status"] in ("verification_passed", "verification_failed")

    result = mcp_server.manage_execution(
        session_id,
        action="submit",
        content="Execution completed successfully.",
    )

    final = mcp_server.manage_execution(session_id, action="complete")
    assert final["status"] == "completed"
    assert final["completed"] is True
    assert "dispatch_plan" in final
    assert "learned_macros" in final


def test_execution_session_tracks_artifacts(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    mcp_server.manage_execution(
        session_id, action="submit", content="First observation."
    )
    mcp_server.manage_execution(
        session_id, action="submit", content="Implementation plan."
    )

    market = mcp_server.manage_execution(session_id, action="report")
    assert market["market_count"] >= 2


def test_execution_session_read_artifact_flow(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    mcp_server.manage_execution(
        session_id, action="submit", content="Initial observation."
    )

    market = mcp_server.manage_execution(session_id, action="report")
    assert len(market["market"]) >= 1

    artifact_id = market["market"][0]["artifact_id"]
    read_result = mcp_server.manage_execution(
        session_id, action="read", artifact_id=artifact_id
    )
    assert read_result["artifact"]["artifact_id"] == artifact_id


def test_complete_execution_rejects_incomplete_and_stale_sessions(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    synthesis_result = _make_synthesis_result(workspace_root, state_dir)

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )
    session_id = session["session_id"]

    mcp_server.manage_execution(
        session_id, action="submit", content="Initial observation."
    )

    with pytest.raises(ValueError, match="remaining steps"):
        mcp_server.manage_execution(session_id, action="complete")

    mcp_server.manage_execution(
        session_id, action="submit", content="Implementation plan."
    )
    mcp_server.manage_execution(
        session_id,
        action="verify",
        command="python3 -c \"pass\"",
    )
    mcp_server.manage_execution(
        session_id, action="submit", content="Execution completed successfully."
    )

    final = mcp_server.manage_execution(session_id, action="complete")
    assert final["status"] == "completed"
    assert final["completed"] is True

    with pytest.raises(ValueError, match="already complete"):
        mcp_server.manage_execution(session_id, action="complete")
