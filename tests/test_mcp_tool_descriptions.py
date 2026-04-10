from __future__ import annotations

import json
import sys

import pytest

from barricade import mcp_server
import barricade.executor as executor
from barricade.feed_derived_dna.tasks import (
    build_patch_skeleton,
    derive_feed_dna_prior,
    derive_task_ecology,
)
from barricade.mcp_server import describe_tools, dispatch_plan
from barricade.workflow import build_workflow_intake


HARD_TOPIC_CASES = [
    {
        "key": "multi_region_chat",
        "text": "Architect a zero-downtime multi-region chat backend with WebSockets, Redis Pub/Sub, presence sync, auth, and reconnect handling. Provide a detailed plan and implementation path.",
        "expected_route": "planning",
        "expected_confidence_floor": 0.70,
        "expected_prior": [
            "OBSERVE",
            "LM1",
            "PLAN",
            "WRITE_PATCH",
            "REPAIR",
            "COMMIT",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
        "expected_outline": [
            "PLAN",
            "WRITE_PATCH",
            "REPAIR",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
    },
    {
        "key": "raft_node",
        "text": "Design and implement a Raft node with leader election, membership changes, log compaction, and crash recovery. Include a strict plan and a patch path.",
        "expected_route": "patching",
        "expected_confidence_floor": 0.70,
        "expected_prior": [
            "OBSERVE",
            "LM1",
            "PLAN",
            "WRITE_PATCH",
            "SWITCH_CONTEXT",
            "ROLLBACK",
            "REPAIR",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
        "expected_outline": [
            "PLAN",
            "WRITE_PATCH",
            "ROLLBACK",
            "REPAIR",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
    },
    {
        "key": "token_rotation",
        "text": "Design a security-sensitive token rotation and secrets revocation system across microservices with audit trails, rollback, and zero-downtime rollout.",
        "expected_route": "planning",
        "expected_confidence_floor": 0.70,
        "expected_prior": [
            "OBSERVE",
            "LM1",
            "PLAN",
            "WRITE_PATCH",
            "SWITCH_CONTEXT",
            "WRITE_PLAN",
            "ROLLBACK",
            "REPAIR",
            "VERIFY",
            "SUMMARIZE",
        ],
        "expected_outline": [
            "PLAN",
            "WRITE_PATCH",
            "WRITE_PLAN",
            "ROLLBACK",
            "REPAIR",
            "VERIFY",
            "SUMMARIZE",
        ],
    },
    {
        "key": "ledger_migration",
        "text": "Plan a payment-ledger migration with dual-write, backfill, reconciliation, rollback, and verification under high load.",
        "expected_route": "planning",
        "expected_confidence_floor": 0.70,
        "expected_prior": [
            "OBSERVE",
            "LM1",
            "PLAN",
            "WRITE_PATCH",
            "SWITCH_CONTEXT",
            "WRITE_PLAN",
            "ROLLBACK",
            "REPAIR",
            "VERIFY",
            "SUMMARIZE",
        ],
        "expected_outline": [
            "PLAN",
            "WRITE_PATCH",
            "WRITE_PLAN",
            "ROLLBACK",
            "REPAIR",
            "VERIFY",
            "SUMMARIZE",
        ],
    },
    {
        "key": "streaming_etl",
        "text": "Implement a streaming ETL pipeline with out-of-order event handling, idempotency, backpressure, dead-letter routing, and observability.",
        "expected_route": "patching",
        "expected_confidence_floor": 0.70,
        "expected_prior": [
            "OBSERVE",
            "LM1",
            "WRITE_PATCH",
            "PLAN",
            "REPAIR",
            "COMMIT",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
        "expected_outline": [
            "WRITE_PATCH",
            "PLAN",
            "REPAIR",
            "WRITE_PLAN",
            "VERIFY",
            "SUMMARIZE",
        ],
    },
]


def test_describe_tools_covers_barricade_surface() -> None:
    tools = describe_tools()
    names = {tool["name"] for tool in tools}

    assert len(tools) == 8
    assert {
        "run_benchmark_task",
        "dispatch_plan",
        "solve_problem",
        "analyze_scaling_profile",
        "begin_execution",
        "manage_execution",
        "describe_tools",
        "inspect_state",
    } <= names
    for tool in tools:
        assert tool["purpose"].strip()
        assert tool["when_to_use"].strip()
        assert isinstance(tool["inputs"], list)
        assert tool["output"].strip()

    manage_execution_tool = next(
        tool for tool in tools if tool["name"] == "manage_execution"
    )
    assert "action" in manage_execution_tool["inputs"]
    assert "report" in manage_execution_tool["purpose"].lower()

    solve_problem_tool = next(tool for tool in tools if tool["name"] == "solve_problem")
    assert "prior_strength" in solve_problem_tool["inputs"]


def test_hard_topics_stay_well_classified() -> None:
    for case in HARD_TOPIC_CASES:
        intake = build_workflow_intake(case["text"])
        top_phase = max(
            (
                (phase, score)
                for phase, score in intake["phase_scores"].items()
                if phase != "mixed"
            ),
            key=lambda item: item[1],
        )[0]

        assert intake["route_hint"] == top_phase
        assert intake["route_hint"] == case["expected_route"]
        assert intake["confidence"] >= case["expected_confidence_floor"]
        assert intake["deliverables"]
        assert intake["risks"]


def test_hard_topics_show_explicit_dna_order() -> None:
    for case in HARD_TOPIC_CASES:
        task_pool = derive_task_ecology({"problem_text": case["text"]})
        prior = derive_feed_dna_prior(task_pool)
        skeleton = build_patch_skeleton(task_pool, prior)

        assert prior == case["expected_prior"]
        assert skeleton["token_outline"] == case["expected_outline"]
        if case["key"] == "raft_node":
            assert "WRITE_PATCH" in prior
            assert prior.index("WRITE_PATCH") > prior.index("PLAN")


def test_execution_session_tools_bridge_solve_problem_to_first_step(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Design a resilient cache layer.",
        "workspace_root": str(tmp_path / "repo"),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Design a resilient cache layer.",
            "route_hint": "planning",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "PLAN", "VERIFY"],
            "patch_skeleton": {"token_outline": ["PLAN", "VERIFY"]},
            "task_pool": [],
            "summary": {"route_hint": "planning"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    assert session["status"] == "started"
    assert session["protocol"] == {"name": "session-delta", "version": 1}
    assert session["current_step"] == 0
    assert session["current_token"] == "OBSERVE"
    assert session["cursor"]["current_token"] == "OBSERVE"
    assert session["cursor"]["remaining_steps"] == 3
    assert session["tool_hint"] == "submit_step"
    assert session["instruction"].startswith("Step 1/3: OBSERVE.")
    assert session["delta"]["route_hint"] == "planning"
    assert session["delta"]["dna_summary"] == "OBSERVE -> PLAN -> VERIFY"
    assert session["delta"]["omitted_macros"] == []
    assert "context_hint" not in session
    assert "macro_library" not in session
    assert "learned_macros" not in session
    assert session["payload_metrics"]["approx_bytes"] < 1200

    submitted_content = "Observed the repo layout and constraints. " * 12
    step_result = mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content=submitted_content,
    )

    assert step_result["status"] == "step_recorded"
    assert step_result["artifact"]["token"] == "OBSERVE"
    assert len(step_result["artifact"]["content"]) < len(submitted_content)
    assert step_result["next_token"] == "PLAN"
    assert step_result["tool_hint"] == "submit_step"
    assert step_result["cursor"]["current_token"] == "PLAN"

    report = mcp_server.manage_execution(session["session_id"], action="report")

    assert report["status"] == "market_view"
    assert report["session_id"] == session["session_id"]
    assert report["market_count"] == 1
    assert report["cursor"]["current_token"] == "PLAN"


def test_begin_execution_filters_unresolved_macro_fallback(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Design a resilient cache layer.",
        "workspace_root": str(tmp_path / "repo"),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Design a resilient cache layer.",
            "route_hint": "planning",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "LM1", "PLAN", "VERIFY"],
            "patch_skeleton": {"token_outline": ["PLAN", "VERIFY"]},
            "task_pool": [],
            "summary": {"route_hint": "planning"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    assert session["dna"] == ["OBSERVE", "PLAN", "VERIFY"]
    assert session["protocol"]["name"] == "session-delta"
    assert session["current_token"] == "OBSERVE"
    assert session["cursor"]["current_token"] == "OBSERVE"
    assert session["omitted_macros"] == ["LM1"]
    assert session["delta"]["omitted_macros"] == ["LM1"]
    assert "LM1" not in session["instruction"]


def test_submit_step_rejects_read_artifact_turn(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Inspect an artifact before continuing.",
        "workspace_root": str(tmp_path / "repo"),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Inspect an artifact before continuing.",
            "route_hint": "mixed",
        },
        "synthesis": {
            "feed_prior_dna": ["READ_ARTIFACT", "PLAN", "VERIFY"],
            "patch_skeleton": {"token_outline": ["PLAN", "VERIFY"]},
            "task_pool": [],
            "summary": {"route_hint": "mixed"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    with pytest.raises(ValueError, match="READ_ARTIFACT"):
        mcp_server.manage_execution(
            session["session_id"],
            action="submit",
            content="This should not be accepted.",
        )


def test_verify_step_rejects_traceback_output_even_with_zero_exit(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Verify a change and reject traceback noise.",
        "workspace_root": str(tmp_path / "repo"),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Verify a change and reject traceback noise.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "VERIFY", "SUMMARIZE"],
            "patch_skeleton": {"token_outline": ["VERIFY", "SUMMARIZE"]},
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Observed the repo layout and current MCP surface.",
    )

    result = mcp_server.manage_execution(
        session["session_id"],
        action="verify",
        command=json.dumps(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.stderr.write('Traceback (most recent call last):\\n'); "
                    "sys.stderr.write('ImportError: boom\\n'); "
                    "sys.exit(0)"
                ),
            ]
        ),
    )

    assert result["status"] == "verification_failed"
    assert result["verification"]["passed"] is False


def test_verify_step_accepts_raw_command_string_without_shell(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Reject raw shell commands in verification.",
        "workspace_root": str(tmp_path / "repo"),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Reject raw shell commands in verification.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "VERIFY", "SUMMARIZE"],
            "patch_skeleton": {"token_outline": ["VERIFY", "SUMMARIZE"]},
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Observed the repo layout and current MCP surface.",
    )

    result = mcp_server.manage_execution(
        session["session_id"],
        action="verify",
        command=f'{sys.executable} -c "pass"',
    )

    assert result["verification"]["passed"] is True
    assert result["verification"]["command"][0] == sys.executable


def test_complete_execution_derives_targeted_dispatch_verification(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()

    synthesis_result = {
        "problem_text": "Write a small Python smoke note.",
        "workspace_root": str(workspace_root),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Write a small Python smoke note.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "WRITE_PATCH", "VERIFY", "SUMMARIZE"],
            "patch_skeleton": {"token_outline": ["WRITE_PATCH", "VERIFY", "SUMMARIZE"]},
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Observed the repo layout and current MCP surface.",
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content=json.dumps(
            {"updates": {"pywatcher.py": "print('hello from barricade')\n"}}
        ),
    )
    (workspace_root / "pywatcher.py").write_text("print('hello from barricade')\n")
    mcp_server.manage_execution(
        session["session_id"],
        action="verify",
        command=json.dumps(
            {
                "command": [sys.executable, "-c", "pass"],
                "verification_spec": {
                    "kind": "syntax",
                    "target_paths": ["pywatcher.py"],
                },
            }
        ),
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Summarized the smoke-note patch and its validation path.",
    )

    final = mcp_server.manage_execution(session["session_id"], action="complete")

    verification_command = final["dispatch_plan"]["verification_command"]
    assert isinstance(verification_command, list)
    assert verification_command[0] == sys.executable
    assert verification_command[1] == "-c"
    assert "pywatcher.py" in verification_command[-1]

    verification_spec = final["dispatch_plan"]["verification_spec"]
    assert verification_spec["kind"] == "syntax"
    assert verification_spec["target_paths"] == ["pywatcher.py"]
    assert verification_spec["artifact_paths"]
    assert final["market"]
    assert final["completion_summary"]["status"] == "completed"
    assert final["completion_summary"]["artifact_count"] == len(final["market"])
    assert final["completion_summary"]["top_artifact"]

    committed = dispatch_plan(
        json.dumps(final["dispatch_plan"]),
        workspace_root=str(workspace_root),
        commit=True,
    )

    assert committed["status"] == "committed"
    assert (
        workspace_root / "pywatcher.py"
    ).read_text() == "print('hello from barricade')\n"


def test_complete_execution_mines_only_successful_suffix(tmp_path, monkeypatch) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()

    synthesis_result = {
        "problem_text": "Write a repairable Python smoke note.",
        "workspace_root": str(workspace_root),
        "state_dir": str(tmp_path / "state"),
        "intake": {
            "raw_task": "Write a repairable Python smoke note.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": [
                "OBSERVE",
                "WRITE_PATCH",
                "VERIFY",
                "REPAIR",
                "VERIFY",
                "SUMMARIZE",
            ],
            "patch_skeleton": {
                "token_outline": [
                    "WRITE_PATCH",
                    "VERIFY",
                    "REPAIR",
                    "VERIFY",
                    "SUMMARIZE",
                ]
            },
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }

    captured = {}

    def fake_mine_macros(elite_flattened, max_macros=8):
        captured["elite_flattened"] = [list(trace) for trace in elite_flattened]
        captured["max_macros"] = max_macros
        return {"LMX": ["REPAIR", "VERIFY", "SUMMARIZE"]}

    monkeypatch.setattr(executor.registry, "mine_macros_from_elites", fake_mine_macros)

    session = mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(tmp_path / "state")
    )

    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Observed the repo layout and current MCP surface.",
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content=json.dumps(
            {"updates": {"pywatcher.py": "print('hello from barricade')\n"}}
        ),
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="verify",
        command=json.dumps(
            {
                "command": [sys.executable, "-c", "raise SystemExit(3)"],
                "verification_spec": {
                    "kind": "returncode",
                    "returncode": 3,
                    "require_empty_stderr": True,
                },
            }
        ),
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Repaired the failed verification path.",
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="verify",
        command=json.dumps(
            {
                "command": [sys.executable, "-c", "pass"],
                "verification_spec": {
                    "kind": "returncode",
                    "returncode": 0,
                    "require_empty_stderr": True,
                },
            }
        ),
    )
    mcp_server.manage_execution(
        session["session_id"],
        action="submit",
        content="Summarized the repaired smoke-note patch.",
    )

    final = mcp_server.manage_execution(session["session_id"], action="complete")

    assert captured["max_macros"] == 8
    assert captured["elite_flattened"] == [
        ["REPAIR", "VERIFY", "SUMMARIZE"],
        ["REPAIR", "VERIFY", "SUMMARIZE"],
    ]
    assert final["execution_learning"]["successful_trace_count"] == 1
    assert final["learned_macros"] == {"LMX": ["REPAIR", "VERIFY", "SUMMARIZE"]}
    assert final["market"]
    assert final["completion_summary"]["artifact_count"] == len(final["market"])
    assert final["completion_summary"]["verification_pass_rate"] >= 0.0


def test_complete_execution_restores_missing_market_ledger(monkeypatch) -> None:
    empty_completion = {
        "status": "completed",
        "completed": True,
        "market": [],
        "market_count": 0,
        "completion_summary": {
            "status": "completed",
            "artifact_count": 0,
            "market_count": 0,
        },
        "payload_metrics": {"approx_bytes": 1, "field_count": 5},
    }
    live_market = {
        "market_count": 2,
        "market": [
            {
                "artifact_id": "OBS_01",
                "token": "OBSERVE",
                "kind": "note",
                "creator": "host",
                "epoch": 1,
                "price": 1.0,
                "score": 1.0,
                "status": "submitted",
                "content": "observed",
            },
            {
                "artifact_id": "PLAN_02",
                "token": "PLAN",
                "kind": "plan",
                "creator": "host",
                "epoch": 2,
                "price": 2.0,
                "score": 2.0,
                "status": "submitted",
                "content": "planned",
            },
        ],
    }

    monkeypatch.setattr(mcp_server, "complete_execution_session", lambda session_id: dict(empty_completion))
    monkeypatch.setattr(mcp_server, "view_execution_market", lambda session_id, limit=8: dict(live_market))

    result = mcp_server.manage_execution("exec_test", action="complete")

    assert result["market_count"] == 2
    assert result["market"] == live_market["market"]
    assert result["completion_summary"]["market_count"] == 2
    assert result["completion_summary"]["artifact_count"] == 2
    assert result["payload_metrics"]["field_count"] >= 7
