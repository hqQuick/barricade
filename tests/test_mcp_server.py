from __future__ import annotations

import json
from pathlib import Path

import pytest

from barricade import mcp_server


def _solve_result(tmp_path):
    state_dir = tmp_path / "state"
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    return mcp_server.solve_problem(
        problem_text="Add a simple function",
        state_dir=str(state_dir),
        workspace_root=str(workspace_root),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=42,
    )


class TestMcpServerEndpoints:
    def test_describe_tools_returns_all_tools(self) -> None:
        tools = mcp_server.describe_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 6
        tool_names = {t["name"] for t in tools}
        assert "solve_problem" in tool_names
        assert "begin_execution" in tool_names
        assert "manage_execution" in tool_names
        assert "dispatch_plan" in tool_names
        assert "run_benchmark_task" in tool_names
        assert "analyze_scaling_profile" in tool_names
        assert all(tool["api_version"] == mcp_server.API_VERSION for tool in tools)

    def test_describe_tools_each_has_required_fields(self) -> None:
        tools = mcp_server.describe_tools()
        for tool in tools:
            assert "name" in tool
            assert "purpose" in tool
            assert "when_to_use" in tool
            assert "inputs" in tool
            assert "output" in tool
            assert "api_version" in tool
            assert isinstance(tool["inputs"], list)

    def test_dispatch_plan_dry_run_with_empty_plan(self, tmp_path) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        plan = {
            "workspace_root": str(workspace_root),
            "updates": {"test.txt": "hello"},
            "verification_command": ["python3", "-c", "pass"],
        }
        result = mcp_server.dispatch_plan(
            json.dumps(plan), workspace_root=str(workspace_root)
        )
        assert result["status"] == "dry_run"
        assert result["verified"] is True or result["verified"] is False
        assert result["api_version"] == mcp_server.API_VERSION

    def test_analyze_scaling_profile_with_minimal_payload(self) -> None:
        result = {
            "summary": {"score": 0.9},
            "controller_summary": {"mode": "balanced"},
            "phase_summary": {"transition_count": 1, "phases": ["planning"]},
            "archive_stable_top3": [],
            "archive_raw_top3": [],
            "transition_events_head": [],
            "lineage_mutation_scale": {"root": 1.0},
            "feed_profile": {"feed_enabled": True},
        }
        diagnostics = mcp_server.analyze_scaling_profile(json.dumps(result))
        assert isinstance(diagnostics, dict)
        assert diagnostics["api_version"] == mcp_server.API_VERSION

    def test_solve_problem_minimal(self, tmp_path) -> None:
        result = _solve_result(tmp_path)
        assert "synthesis" in result
        assert "feed_prior_dna" in result or "dna" in result
        assert "execution_seed" in result
        assert result["api_version"] == mcp_server.API_VERSION

    def test_solve_problem_rejects_workspace_escape(self) -> None:
        with pytest.raises(ValueError, match="workspace_root"):
            mcp_server.solve_problem(
                "Add a simple function",
                workspace_root="/etc",
                trials=1,
                population=2,
                base_episodes=1,
                generations=1,
                seed0=42,
            )

    def test_solve_problem_defaults_state_dir_to_repo_folder(
        self, tmp_path, monkeypatch
    ) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        captured: dict[str, object] = {}

        def fake_run_unified_workflow(problem_text: str, **kwargs):
            captured["state_dir"] = kwargs["state_dir"]
            captured["workspace_root"] = kwargs["workspace_root"]
            return {
                "problem_text": problem_text,
                "intake": {"raw_task": problem_text, "route_hint": "planning"},
                "synthesis": {
                    "feed_prior_dna": ["OBSERVE", "VERIFY"],
                    "patch_skeleton": {"token_outline": ["VERIFY"]},
                    "task_pool": [],
                    "summary": {"route_hint": "planning"},
                },
                "execution": None,
                "prior_profile": {"effective_strength": 0.5, "cache_min_similarity": 0.0},
                "cache_hit": False,
            }

        monkeypatch.setattr(mcp_server, "run_unified_workflow", fake_run_unified_workflow)

        result = mcp_server.solve_problem(
            "Add a simple function",
            workspace_root=str(workspace_root),
            trials=1,
            population=2,
            base_episodes=1,
            generations=1,
            seed0=42,
        )

        expected_state_dir = workspace_root / ".barricade_state"
        assert Path(str(captured["state_dir"])) == expected_state_dir
        assert Path(result["state_dir"]) == expected_state_dir
        assert Path(result["workspace_root"]) == workspace_root

    def test_solve_problem_defaults_state_dir_to_repo_root_without_workspace_root(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_unified_workflow(problem_text: str, **kwargs):
            captured["state_dir"] = kwargs["state_dir"]
            captured["workspace_root"] = kwargs["workspace_root"]
            return {
                "problem_text": problem_text,
                "intake": {"raw_task": problem_text, "route_hint": "planning"},
                "synthesis": {
                    "feed_prior_dna": ["OBSERVE", "VERIFY"],
                    "patch_skeleton": {"token_outline": ["VERIFY"]},
                    "task_pool": [],
                    "summary": {"route_hint": "planning"},
                },
                "execution": None,
                "prior_profile": {"effective_strength": 0.5, "cache_min_similarity": 0.0},
                "cache_hit": False,
            }

        monkeypatch.setattr(mcp_server, "run_unified_workflow", fake_run_unified_workflow)

        result = mcp_server.solve_problem(
            "Add a simple function",
            trials=1,
            population=2,
            base_episodes=1,
            generations=1,
            seed0=42,
        )

        expected_state_dir = Path(mcp_server.__file__).resolve().parents[1] / ".barricade_state"
        assert Path(str(captured["state_dir"])) == expected_state_dir
        assert captured["workspace_root"] is None
        assert Path(result["state_dir"]) == expected_state_dir

    def test_begin_execution_returns_session_id(self, tmp_path) -> None:
        solve_result = _solve_result(tmp_path)

        session = mcp_server.begin_execution(
            json.dumps(solve_result), state_dir=str(tmp_path / "state")
        )
        assert "session_id" in session
        assert session["session_id"].startswith("exec_")
        assert session["current_step"] == 0
        assert "instruction" in session
        assert session["api_version"] == mcp_server.API_VERSION

    def test_begin_execution_rejects_workspace_escape(self) -> None:
        synthesis_result = {
            "problem_text": "Add a simple function",
            "workspace_root": "/etc",
            "synthesis": {
                "feed_prior_dna": ["OBSERVE", "VERIFY"],
                "patch_skeleton": {"token_outline": ["VERIFY"]},
                "task_pool": [],
                "summary": {"route_hint": "planning"},
            },
        }

        with pytest.raises(ValueError, match="workspace_root"):
            mcp_server.begin_execution(json.dumps(synthesis_result))

    def test_begin_execution_defaults_state_dir_to_repo_folder(
        self, tmp_path, monkeypatch
    ) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        captured: dict[str, object] = {}
        synthesis_result = {
            "problem_text": "Add a simple function",
            "workspace_root": str(workspace_root),
            "synthesis": {
                "feed_prior_dna": ["OBSERVE", "VERIFY"],
                "patch_skeleton": {"token_outline": ["VERIFY"]},
                "task_pool": [],
                "summary": {"route_hint": "planning"},
            },
        }

        def fake_begin_execution_session(result, state_dir=None):
            captured["state_dir"] = state_dir
            return {
                "session_id": "exec_test",
                "status": "started",
                "current_step": 0,
                "current_token": "OBSERVE",
                "cursor": {"current_token": "OBSERVE"},
                "payload_metrics": {"approx_bytes": 1, "field_count": 5},
            }

        monkeypatch.setattr(
            mcp_server, "begin_execution_session", fake_begin_execution_session
        )

        session = mcp_server.begin_execution(json.dumps(synthesis_result))

        expected_state_dir = workspace_root / ".barricade_state"
        assert Path(str(captured["state_dir"])) == expected_state_dir
        assert session["session_id"] == "exec_test"
        assert session["api_version"] == mcp_server.API_VERSION

    def test_begin_execution_defaults_state_dir_to_repo_root_without_workspace_root(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}
        synthesis_result = {
            "problem_text": "Add a simple function",
            "synthesis": {
                "feed_prior_dna": ["OBSERVE", "VERIFY"],
                "patch_skeleton": {"token_outline": ["VERIFY"]},
                "task_pool": [],
                "summary": {"route_hint": "planning"},
            },
        }

        def fake_begin_execution_session(result, state_dir=None):
            captured["state_dir"] = state_dir
            return {
                "session_id": "exec_test",
                "status": "started",
                "current_step": 0,
                "current_token": "OBSERVE",
                "cursor": {"current_token": "OBSERVE"},
                "payload_metrics": {"approx_bytes": 1, "field_count": 5},
            }

        monkeypatch.setattr(
            mcp_server, "begin_execution_session", fake_begin_execution_session
        )

        session = mcp_server.begin_execution(json.dumps(synthesis_result))

        expected_state_dir = Path(mcp_server.__file__).resolve().parents[1] / ".barricade_state"
        assert Path(str(captured["state_dir"])) == expected_state_dir
        assert session["session_id"] == "exec_test"
        assert session["api_version"] == mcp_server.API_VERSION

    def test_manage_execution_submit_and_view_market(
        self, tmp_path
    ) -> None:
        solve_result = _solve_result(tmp_path)

        session = mcp_server.begin_execution(
            json.dumps(solve_result), state_dir=str(tmp_path / "state")
        )
        session_id = session["session_id"]

        step_result = mcp_server.manage_execution(
            session_id, action="submit", content="Initial observation."
        )
        assert step_result["status"] == "step_recorded"
        assert "artifact" in step_result
        assert step_result["api_version"] == mcp_server.API_VERSION

        market = mcp_server.manage_execution(session_id, action="report")
        assert market["market_count"] >= 1
        assert len(market["market"]) >= 1
        assert market["api_version"] == mcp_server.API_VERSION

    def test_manage_execution_unknown_action_raises(
        self, tmp_path
    ) -> None:
        solve_result = _solve_result(tmp_path)

        session = mcp_server.begin_execution(
            json.dumps(solve_result), state_dir=str(tmp_path / "state")
        )
        session_id = session["session_id"]

        with pytest.raises(ValueError, match="unknown execution action"):
            mcp_server.manage_execution(session_id, action="bogus_action")
