from __future__ import annotations

import argparse
import ast
import difflib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ._shared import run_command_with_timeout
from ._verification_parser import parse_verification_output
from ._validation import validate_workspace_root
from ._version import with_api_version


FAILURE_SIGNATURES = (
    "Traceback (most recent call last):",
    "ImportError:",
    "ModuleNotFoundError:",
    "SyntaxError:",
    "IndentationError:",
    "AttributeError:",
    "NameError:",
    "TypeError:",
    "AssertionError:",
    "ValueError:",
)


_LANGUAGE_SUFFIXES = {
    ".py": "python",
    ".rs": "rust",
    ".java": "java",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}


def _load_plan(plan_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(plan_or_path, (str, Path)):
        candidate = Path(str(plan_or_path).strip())
        try:
            if candidate.exists():
                payload = json.loads(candidate.read_text())
            else:
                text = str(plan_or_path).strip()
                payload = json.loads(text) if text else {}
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return {}
        return payload if isinstance(payload, dict) else {}
    try:
        return dict(plan_or_path)
    except (TypeError, ValueError):
        return {}


def _normalize_workspace_root(workspace_root: str | Path | None) -> Path:
    root = validate_workspace_root(workspace_root) or Path.cwd().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_updates(plan: dict[str, Any]) -> dict[str, str]:
    updates = plan.get("updates")
    if updates is None and "target_path" in plan and "replacement_text" in plan:
        updates = {plan["target_path"]: plan["replacement_text"]}
    if not isinstance(updates, dict) or not updates:
        raise ValueError(
            "dispatch plan must define updates or target_path/replacement_text"
        )

    normalized = {}
    for relative_path, content in updates.items():
        normalized[str(relative_path)] = str(content)
    return normalized


def _resolve_relative_path(workspace_root: Path, relative_path: str) -> Path:
    workspace_root = workspace_root.resolve()
    candidate = (workspace_root / relative_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace root: {relative_path}") from exc
    if candidate == workspace_root:
        return candidate
    return candidate


def _diff_preview(before: str, after: str, relative_path: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )


def _copy_tree(source_root: Path, destination_root: Path) -> None:
    shutil.copytree(source_root, destination_root, dirs_exist_ok=True)


def _verification_failure_signatures(stdout: str, stderr: str) -> list[str]:
    combined = f"{stdout}\n{stderr}"
    return [signature for signature in FAILURE_SIGNATURES if signature in combined]


def _language_for_path(path: Path, fallback: str = "") -> str:
    return _LANGUAGE_SUFFIXES.get(path.suffix.lower(), fallback)


def _normalize_expected_sequence(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _language_syntax_command(language: str, target: Path) -> list[str] | None:
    path = str(target)
    if language == "rust":
        tool = shutil.which("rustc")
        if tool:
            return [tool, "--crate-type=lib", "--emit=metadata", "--edition=2021", path]
        return None
    if language == "java":
        tool = shutil.which("javac")
        if tool:
            return [tool, "-proc:none", path]
        return None
    if language in {"javascript", "js"}:
        tool = shutil.which("node")
        if tool:
            return [tool, "--check", path]
        return None
    if language == "typescript":
        tool = shutil.which("tsc")
        if tool:
            return [tool, "--noEmit", path]
        return None
    if language in {"c", "cpp"}:
        tool = (
            shutil.which("cc")
            or shutil.which("clang")
            or shutil.which("gcc")
            or shutil.which("c++")
        )
        if tool:
            return [tool, "-fsyntax-only", path]
        return None
    return None


def _syntax_failure_for_target(
    relative_path: str, target: Path, language: str, cwd: Path
) -> str | None:
    if language in {"python", "py"}:
        try:
            ast.parse(target.read_text())
            return None
        except (SyntaxError, IndentationError) as exc:
            return f"{relative_path}: {exc.__class__.__name__}: {exc.msg}"

    command = _language_syntax_command(language, target)
    if command is None:
        label = language or target.suffix.lstrip(".") or "unknown"
        return (
            f"{relative_path}: unsupported syntax verification for language '{label}'; "
            "provide verification_command"
        )

    completed = run_command_with_timeout(command, cwd)
    if completed.returncode == 0 and not _verification_failure_signatures(
        completed.stdout, completed.stderr
    ):
        return None

    report = parse_verification_output(completed)
    message = report.summary or "Verification failed."
    hints = report._actionable_hints()
    if hints:
        message = f"{message} {'; '.join(hints[:2])}"
    return f"{relative_path}: {message}"


def _verification_semantic_failures(
    verification: subprocess.CompletedProcess[str],
    spec: dict[str, Any] | None,
    cwd: Path,
) -> list[str]:
    if not isinstance(spec, dict) or not spec:
        return []

    failures: list[str] = []
    kind = str(spec.get("kind", "")).strip().lower()
    target_paths = (
        [str(path) for path in spec.get("target_paths", [])]
        if isinstance(spec.get("target_paths"), list)
        else []
    )

    if kind in {
        "syntax",
        "syntax_check",
        "python_syntax",
        "source_check",
        "source_syntax",
    }:
        if not target_paths:
            failures.append("missing target_paths for source verification")
        for relative_path in target_paths:
            target = _resolve_relative_path(cwd, relative_path)
            if not target.exists():
                failures.append(f"missing target file: {relative_path}")
                continue
            language = str(spec.get("language", "")).strip().lower()
            language = language or _language_for_path(target)
            syntax_failure = _syntax_failure_for_target(
                relative_path, target, language, cwd
            )
            if syntax_failure:
                failures.append(syntax_failure)

    expected_stdout = _normalize_expected_sequence(spec.get("stdout_contains"))
    expected_stderr = _normalize_expected_sequence(spec.get("stderr_contains"))
    forbidden_stdout = _normalize_expected_sequence(spec.get("stdout_not_contains"))
    forbidden_stderr = _normalize_expected_sequence(spec.get("stderr_not_contains"))

    for needle in expected_stdout:
        if needle not in verification.stdout:
            failures.append(f"stdout missing expected text: {needle}")
    for needle in expected_stderr:
        if needle not in verification.stderr:
            failures.append(f"stderr missing expected text: {needle}")
    for needle in forbidden_stdout:
        if needle in verification.stdout:
            failures.append(f"stdout contained forbidden text: {needle}")
    for needle in forbidden_stderr:
        if needle in verification.stderr:
            failures.append(f"stderr contained forbidden text: {needle}")

    expected_returncode = spec.get("returncode")
    if expected_returncode is not None and verification.returncode != int(
        expected_returncode
    ):
        failures.append(
            f"expected returncode {int(expected_returncode)} but got {verification.returncode}"
        )

    if spec.get("require_empty_stderr") and verification.stderr.strip():
        failures.append("stderr was not empty")

    if spec.get("require_empty_stdout") and verification.stdout.strip():
        failures.append("stdout was not empty")

    return failures


def verification_passed(
    verification: subprocess.CompletedProcess[str],
    spec: dict[str, Any] | None = None,
    *,
    cwd: Path | None = None,
) -> tuple[bool, list[str]]:
    failure_signatures = _verification_failure_signatures(
        verification.stdout, verification.stderr
    )
    semantic_failures = _verification_semantic_failures(
        verification, spec, cwd or Path.cwd()
    )
    return (
        verification.returncode == 0
        and not failure_signatures
        and not semantic_failures,
        failure_signatures + semantic_failures,
    )


def barricade_dispatch(
    plan_or_path: dict[str, Any] | str | Path,
    *,
    workspace_root: str | Path | None = None,
    commit: bool = False,
) -> dict[str, Any]:
    plan = _load_plan(plan_or_path)
    root = _normalize_workspace_root(workspace_root or plan.get("workspace_root"))
    updates = _normalize_updates(plan)
    verification_command = plan.get("verification_command")

    preview = []
    for relative_path, content in updates.items():
        target_path = _resolve_relative_path(root, relative_path)
        before = target_path.read_text() if target_path.exists() else ""
        preview.append(
            {
                "path": relative_path,
                "before_size": len(before),
                "after_size": len(content),
                "diff": _diff_preview(before, content, relative_path),
            }
        )

    if not commit:
        return with_api_version(
            {
                "status": "dry_run",
                "committed": False,
                "verified": False,
                "workspace_root": str(root),
                "updates": preview,
            }
        )

    if verification_command is None:
        return with_api_version(
            {
                "status": "rejected",
                "committed": False,
                "verified": False,
                "workspace_root": str(root),
                "reason": "verification_command is required for governed commit",
                "updates": preview,
            }
        )

    if isinstance(verification_command, tuple):
        verification_command = list(verification_command)
    if not isinstance(verification_command, list):
        raise ValueError("verification_command must be a list of arguments")

    staging_parent = Path(tempfile.mkdtemp(prefix="barricade_dispatch_"))
    staged_root = staging_parent / root.name
    _copy_tree(root, staged_root)

    for relative_path, content in updates.items():
        staged_target = _resolve_relative_path(staged_root, relative_path)
        staged_target.parent.mkdir(parents=True, exist_ok=True)
        staged_target.write_text(content)

    verification_spec = plan.get("verification_spec")
    verification = run_command_with_timeout(verification_command, staged_root)
    passed, failure_signatures = verification_passed(
        verification, verification_spec, cwd=staged_root
    )
    if not passed:
        return with_api_version(
            {
                "status": "verification_failed",
                "committed": False,
                "verified": False,
                "workspace_root": str(root),
                "verification_returncode": verification.returncode,
                "verification_stdout": verification.stdout,
                "verification_stderr": verification.stderr,
                "verification_failure_signatures": failure_signatures,
                "verification_spec": verification_spec or {},
                "updates": preview,
            }
        )

    backup_root = root / ".barricade_backups" / staging_parent.name
    backup_root.mkdir(parents=True, exist_ok=True)

    for relative_path, content in updates.items():
        target_path = _resolve_relative_path(root, relative_path)
        if target_path.exists():
            backup_path = backup_root / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, backup_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)

    return with_api_version(
        {
            "status": "committed",
            "committed": True,
            "verified": True,
            "workspace_root": str(root),
            "verification_returncode": verification.returncode,
            "verification_stdout": verification.stdout,
            "verification_stderr": verification.stderr,
            "verification_failure_signatures": failure_signatures,
            "verification_spec": verification_spec or {},
            "backup_root": str(backup_root),
            "updates": preview,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Barricade governed dispatch")
    parser.add_argument(
        "--plan-file", required=True, help="Path to a JSON dispatch plan"
    )
    parser.add_argument(
        "--workspace-root", default="", help="Workspace root that the plan targets"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Apply the staged change after verification passes",
    )
    args = parser.parse_args()

    result = barricade_dispatch(
        args.plan_file, workspace_root=args.workspace_root or None, commit=args.commit
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
