from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from barricade import mcp_server
from barricade.dispatch import _verification_semantic_failures
from barricade.executor._parsing import _verification_spec_for_dispatch_updates


def _make_synthesis_result_with_verify(workspace_root, state_dir) -> dict:
    return {
        "problem_text": "Add hello world.",
        "workspace_root": str(workspace_root),
        "state_dir": str(state_dir),
        "intake": {
            "raw_task": "Add hello world.",
            "route_hint": "patching",
        },
        "synthesis": {
            "feed_prior_dna": ["OBSERVE", "WRITE_PATCH", "VERIFY", "SUMMARIZE"],
            "patch_skeleton": {"token_outline": ["WRITE_PATCH", "VERIFY", "SUMMARIZE"]},
            "task_pool": [],
            "summary": {"route_hint": "patching"},
        },
    }


class TestStructuredVerification:
    def test_verify_step_returns_structured_report_on_pass(self, tmp_path) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        state_dir = tmp_path / "state"

        synthesis_result = _make_synthesis_result_with_verify(workspace_root, state_dir)
        session = mcp_server.begin_execution(
            json.dumps(synthesis_result), state_dir=str(state_dir)
        )
        session_id = session["session_id"]

        mcp_server.manage_execution(session_id, action="submit", content="Observation.")
        mcp_server.manage_execution(
            session_id, action="submit", content="Patch content."
        )

        result = mcp_server.manage_execution(
            session_id,
            action="verify",
            command='python3 -c "pass"',
        )
        assert result["status"] == "verification_passed"
        assert "verification" in result
        assert "structured_report" in result["verification"]
        report = result["verification"]["structured_report"]
        assert report["passed"] is True
        assert report["summary"] == "Verification passed."

    def test_verify_step_returns_structured_report_on_failure(self, tmp_path) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        state_dir = tmp_path / "state"

        synthesis_result = _make_synthesis_result_with_verify(workspace_root, state_dir)
        session = mcp_server.begin_execution(
            json.dumps(synthesis_result), state_dir=str(state_dir)
        )
        session_id = session["session_id"]

        mcp_server.manage_execution(session_id, action="submit", content="Observation.")
        mcp_server.manage_execution(
            session_id, action="submit", content="Patch content."
        )

        result = mcp_server.manage_execution(
            session_id,
            action="verify",
            command='python3 -c "import nonexistent_module_xyz"',
        )
        assert result["status"] == "verification_failed"
        assert "structured_report" in result["verification"]
        report = result["verification"]["structured_report"]
        assert report["passed"] is False
        assert report["returncode"] != 0
        assert "actionable_hints" in report
        assert len(report["actionable_hints"]) > 0

    def test_verify_step_uses_semantic_failures_in_structured_report(self, tmp_path) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        state_dir = tmp_path / "state"

        synthesis_result = _make_synthesis_result_with_verify(workspace_root, state_dir)
        session = mcp_server.begin_execution(
            json.dumps(synthesis_result), state_dir=str(state_dir)
        )
        session_id = session["session_id"]

        mcp_server.manage_execution(session_id, action="submit", content="Observation.")
        mcp_server.manage_execution(
            session_id,
            action="submit",
            content=json.dumps({"updates": {"bad.py": "def broken("}}),
        )

        result = mcp_server.manage_execution(
            session_id,
            action="verify",
            command=json.dumps(["python3", "-c", "pass"]),
        )

        report = result["verification"]["structured_report"]
        assert result["status"] == "verification_failed"
        assert result["verification"]["passed"] is False
        assert report["passed"] is False
        assert report["semantic_failures"]
        assert any("Semantic failures" in hint for hint in report["actionable_hints"])

    def test_verify_step_times_out_without_advancing(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        state_dir = tmp_path / "state"

        synthesis_result = {
            "problem_text": "Add a verification step that times out.",
            "workspace_root": str(workspace_root),
            "state_dir": str(state_dir),
            "intake": {
                "raw_task": "Add a verification step that times out.",
                "route_hint": "patching",
            },
            "synthesis": {
                "feed_prior_dna": ["OBSERVE", "VERIFY", "SUMMARIZE"],
                "patch_skeleton": {"token_outline": ["OBSERVE", "VERIFY", "SUMMARIZE"]},
                "task_pool": [],
                "summary": {"route_hint": "patching"},
            },
        }

        session = mcp_server.begin_execution(
            json.dumps(synthesis_result), state_dir=str(state_dir)
        )
        session_id = session["session_id"]

        mcp_server.manage_execution(
            session_id,
            action="submit",
            content="Observed the repository layout.",
        )

        def fake_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=kwargs.get("command", args[0] if args else []),
                returncode=124,
                stdout="",
                stderr="Verification timed out after 300 seconds.",
            )

        monkeypatch.setattr("barricade.executor.registry.run_command_with_timeout", fake_timeout)

        result = mcp_server.manage_execution(
            session_id,
            action="verify",
            command=json.dumps(["python3", "-c", "import time; time.sleep(10)"]),
        )

        assert result["status"] == "verification_failed"
        assert result["verification"]["passed"] is False
        assert "timed out" in result["verification"]["stderr"].lower()
        assert result["next_token"] == "VERIFY"
        assert "does not contain REPAIR or ROLLBACK" in result["next_instruction"]["instruction"]

    def test_structured_report_parses_syntax_errors(self) -> None:
        from barricade._verification_parser import parse_verification_output

        completed = subprocess.run(
            ["python3", "-c", "def foo("],
            capture_output=True,
            text=True,
        )
        report = parse_verification_output(completed)
        assert report.passed is False
        assert len(report.syntax_errors) > 0 or len(report.runtime_errors) > 0
        assert report.summary.startswith("Verification failed")

    def test_structured_report_handles_truncated_traceback_line(self) -> None:
        from barricade._verification_parser import parse_verification_output

        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr='  File "example.py", line 3',
        )

        report = parse_verification_output(completed)

        assert report.passed is False
        assert report.syntax_errors
        assert report.syntax_errors[0].path == "example.py"

    def test_structured_report_parses_pytest_summary_failures(self) -> None:
        from barricade._verification_parser import parse_verification_output

        completed = subprocess.CompletedProcess(
            args=["pytest"],
            returncode=1,
            stdout="",
            stderr=(
                "=========================== short test summary info ============================\n"
                "FAILED tests/test_example.py::test_thing - assert 1 == 2\n"
            ),
        )

        report = parse_verification_output(completed)

        assert report.passed is False
        assert report.test_failures
        assert report.test_failures[0].test_name == "tests/test_example.py::test_thing"
        assert report.test_failures[0].message == "assert 1 == 2"
        assert report.summary == "Verification failed. 1 test failure(s)."

    def test_structured_report_parses_compiler_style_errors(self) -> None:
        from barricade._verification_parser import parse_verification_output

        completed = subprocess.CompletedProcess(
            args=["rustc"],
            returncode=1,
            stdout="",
            stderr=(
                "src/main.rs:12:5: error[E0425]: cannot find value `x` in this scope\n"
            ),
        )

        report = parse_verification_output(completed)

        assert report.passed is False
        assert report.syntax_errors
        assert report.syntax_errors[0].path == "src/main.rs"
        assert report.syntax_errors[0].line == 12
        assert report.syntax_errors[0].column == 5

    def test_dispatch_spec_infers_source_language(self) -> None:
        spec = _verification_spec_for_dispatch_updates(
            {"src/main.rs": "fn main() {}"},
            {"artifact_01": ["src/main.rs"]},
            artifact_id="artifact_01",
        )

        assert spec["kind"] == "file_exists"
        assert spec["language"] == "rust"
        assert spec["target_paths"] == ["src/main.rs"]

    def test_semantic_source_verification_uses_language_hint(self, tmp_path, monkeypatch) -> None:
        workspace_root = tmp_path / "repo"
        source = workspace_root / "src" / "main.rs"
        source.parent.mkdir(parents=True)
        source.write_text("fn main() {")

        spec = {
            "kind": "syntax",
            "target_paths": ["src/main.rs"],
            "language": "rust",
        }

        fake_completed = subprocess.CompletedProcess(
            args=["rustc"],
            returncode=1,
            stdout="",
            stderr=(
                "src/main.rs:1:12: error: mismatched closing delimiter\n"
            ),
        )

        monkeypatch.setattr(
            "barricade.dispatch.shutil.which",
            lambda name: "/usr/bin/rustc" if name == "rustc" else None,
        )
        monkeypatch.setattr(
            "barricade.dispatch.run_command_with_timeout",
            lambda *args, **kwargs: fake_completed,
        )
        failures = _verification_semantic_failures(fake_completed, spec, workspace_root)

        assert failures
        assert "src/main.rs" in failures[0]
        assert "Verification failed" in failures[0]


class TestInspectState:
    def test_inspect_state_no_state_dir(self) -> None:
        result = mcp_server.inspect_state("")
        if result["available"]:
            assert result["state_root"].endswith(".barricade_state")
        else:
            assert "state directory" in result["message"].lower()

    def test_inspect_state_empty_directory(self, tmp_path) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        result = mcp_server.inspect_state(str(state_dir))
        assert result["available"] is True
        assert result["summary"]["macro_count"] == 0
        assert result["summary"]["motif_count"] == 0
        assert result["summary"]["total_runs"] == 0

    def test_inspect_state_after_run(self, tmp_path, monkeypatch) -> None:
        def fake_run_benchmark(**kwargs):
            return {
                "summary": {"score": 0.91},
                "best_governed": {"id": "governed"},
                "best_raw": {"id": "raw"},
                "archive_stable_top3": [],
                "archive_raw_top3": [],
                "controller_summary": {"mode": "balanced"},
                "phase_summary": [{"phase": "planning"}],
                "transition_events_head": [],
                "lineage_mutation_scale": {"root": 1.0},
                "feed_profile": {"feed_enabled": True},
                "learned_macros": {"LM1": ["OBSERVE", "WRITE_PLAN"]},
            }

        monkeypatch.setattr(
            mcp_server.runtime_module, "run_benchmark", fake_run_benchmark
        )

        workspace_root = tmp_path / "repo"
        workspace_root.mkdir()
        state_dir = tmp_path / "state"

        mcp_server.solve_problem(
            problem_text="Test task",
            state_dir=str(state_dir),
            workspace_root=str(workspace_root),
            trials=1,
            population=2,
            base_episodes=1,
            generations=1,
            seed0=42,
        )

        result = mcp_server.inspect_state(str(state_dir))
        assert result["available"] is True
        assert result["summary"]["total_runs"] >= 1

    def test_inspect_state_reads_patch_update_count(self, tmp_path) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "runs.jsonl").write_text(
            json.dumps(
                {
                    "summary": {
                        "status": "completed",
                        "artifact_count": 3,
                        "patch_update_count": 2,
                        "verification_passed": True,
                    },
                    "feed_prior_dna": ["A"],
                    "learned_macros": {},
                }
            )
            + "\n"
        )

        result = mcp_server.inspect_state(str(state_dir))

        assert result["recent_runs"][0]["patches"] == 2

    def test_feed_derived_dna_package_exports_are_available(self) -> None:
        import barricade.feed_derived_dna as feed_derived_dna

        missing = [
            name
            for name in feed_derived_dna.__all__
            if not hasattr(feed_derived_dna, name)
        ]

        assert missing == []
        assert feed_derived_dna.random_token is not None
        assert feed_derived_dna.PRIMITIVES
        assert feed_derived_dna.MOMENTUM_WINDOW > 0


