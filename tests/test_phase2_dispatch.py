from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from barricade.dispatch import barricade_dispatch
from barricade.mcp_server import dispatch_plan


def _verification_command(expected_text: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from pathlib import Path; assert Path('src/app.txt').read_text() == " + repr(expected_text),
    ]


def test_dispatch_dry_run_and_governed_commit(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.txt": "after"},
        "verification_command": _verification_command("after"),
    }

    dry_run = barricade_dispatch(plan, commit=False)
    assert dry_run["status"] == "dry_run"
    assert target.read_text() == "before"

    committed = barricade_dispatch(plan, commit=True)
    assert committed["status"] == "committed"
    assert committed["verified"] is True
    assert target.read_text() == "after"
    assert (workspace_root / ".barricade_backups").exists()


def test_dispatch_refuses_failed_verification(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.txt": "after"},
        "verification_command": [
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        ],
    }

    result = barricade_dispatch(plan, commit=True)

    assert result["status"] == "verification_failed"
    assert result["committed"] is False
    assert target.read_text() == "before"


def test_dispatch_refuses_verification_timeout(tmp_path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.txt": "after"},
        "verification_command": [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
    }

    def fake_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=kwargs.get("command", args[0] if args else []),
            returncode=124,
            stdout="",
            stderr="Verification timed out after 300 seconds.",
        )

    monkeypatch.setattr("barricade.dispatch.run_command_with_timeout", fake_timeout)

    result = barricade_dispatch(plan, commit=True)

    assert result["status"] == "verification_failed"
    assert result["committed"] is False
    assert "timed out" in result["verification_stderr"].lower()
    assert target.read_text() == "before"


def test_dispatch_refuses_traceback_output_even_with_zero_exit(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.txt": "after"},
        "verification_command": [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stderr.write('Traceback (most recent call last):\\n'); "
                "sys.stderr.write('ImportError: boom\\n'); "
                "sys.exit(0)"
            ),
        ],
    }

    result = barricade_dispatch(plan, commit=True)

    assert result["status"] == "verification_failed"
    assert result["committed"] is False
    assert result["verification_returncode"] == 0
    assert result["verification_failure_signatures"]
    assert target.read_text() == "before"


def test_dispatch_refuses_semantic_verification_failure(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.py": "def broken(:\n    pass\n"},
        "verification_command": [
            sys.executable,
            "-c",
            "pass",
        ],
        "verification_spec": {
            "kind": "syntax",
            "target_paths": ["src/app.py"],
            "artifact_id": "WRITE_PATCH_01",
        },
    }

    result = barricade_dispatch(plan, commit=True)

    assert result["status"] == "verification_failed"
    assert result["committed"] is False
    assert result["verification_returncode"] == 0
    assert result["verification_failure_signatures"]
    assert target.read_text() == "before"


def test_dispatch_rejects_workspace_escape(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    plan = {
        "workspace_root": "/etc",
        "updates": {"src/app.txt": "after"},
        "verification_command": _verification_command("after"),
    }

    try:
        barricade_dispatch(plan, commit=False)
    except ValueError as exc:
        assert "workspace_root" in str(exc)
    else:
        raise AssertionError("expected workspace escape to be rejected")


def test_dispatch_mcp_wrapper_uses_same_core(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "src" / "app.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    plan = {
        "workspace_root": str(workspace_root),
        "updates": {"src/app.txt": "after"},
        "verification_command": _verification_command("after"),
    }

    result = dispatch_plan(json.dumps(plan), workspace_root=str(workspace_root), commit=True)

    assert result["status"] == "committed"
    assert result["committed"] is True
    assert target.read_text() == "after"


def test_dispatch_ignores_corrupted_plan_file(tmp_path) -> None:
    plan_file = tmp_path / "broken_plan.json"
    plan_file.write_text("{broken")

    try:
        barricade_dispatch(plan_file, commit=False)
    except ValueError as exc:
        assert "dispatch plan must define" in str(exc)
    else:
        raise AssertionError("expected malformed dispatch plan to be rejected")
