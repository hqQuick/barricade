from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


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


def _language_for_path(path: str) -> str | None:
    return _LANGUAGE_SUFFIXES.get(Path(path).suffix.lower())


def _language_for_paths(paths: list[str]) -> str | None:
    languages = {
        language
        for language in (_language_for_path(path) for path in paths)
        if language is not None
    }
    if len(languages) == 1:
        return next(iter(languages))
    return None


def _parse_file_blocks(content: str) -> dict[str, str]:
    updates: dict[str, str] = {}
    current_path: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_path, current_lines
        if current_path is None:
            return
        body = "\n".join(current_lines).strip()
        if body.startswith("```"):
            lines = body.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            body = "\n".join(lines).strip()
        if body:
            updates[current_path] = body
        current_path = None
        current_lines = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("File:"):
            flush()
            current_path = stripped.removeprefix("File:").strip()
            continue
        if stripped.startswith("*** Update File:"):
            flush()
            current_path = stripped.removeprefix("*** Update File:").strip()
            continue
        if current_path is not None:
            current_lines.append(line)

    flush()
    return updates


def _parse_patch_updates(content: str) -> dict[str, str]:
    stripped = content.strip()
    if not stripped:
        return {}

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        updates = payload.get("updates")
        if isinstance(updates, dict) and updates:
            return {str(path): str(text) for path, text in updates.items()}
        if "target_path" in payload and "replacement_text" in payload:
            return {str(payload["target_path"]): str(payload["replacement_text"])}

    return _parse_file_blocks(stripped)


def _artifacts_to_dispatch_updates(
    artifacts: list,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    updates: dict[str, str] = {}
    provenance: dict[str, list[str]] = {}
    for artifact in artifacts:
        parsed_updates = artifact.metadata.get("parsed_updates")
        if isinstance(parsed_updates, dict) and parsed_updates:
            normalized_updates = {
                str(path): str(text) for path, text in parsed_updates.items()
            }
            updates.update(normalized_updates)
            provenance[artifact.artifact_id] = sorted(normalized_updates)
            continue

        target_path = artifact.metadata.get("target_path")
        if isinstance(target_path, str) and target_path.strip():
            normalized_path = target_path.strip()
            updates[normalized_path] = artifact.content
            provenance[artifact.artifact_id] = [normalized_path]
    return updates, provenance


def _verification_command_for_dispatch_updates(
    updates: dict[str, str],
) -> list[str] | None:
    import sys

    python_paths = sorted(path for path in updates if Path(path).suffix == ".py")
    if not python_paths:
        return None

    script = (
        "from pathlib import Path; import ast, sys; "
        "[ast.parse(Path(path).read_text()) for path in sys.argv[1:]]"
    )
    return [sys.executable, "-c", script, *python_paths]


def _verification_spec_for_dispatch_updates(
    updates: dict[str, str],
    provenance: dict[str, list[str]],
    artifact_id: str | None = None,
) -> dict[str, Any]:
    python_paths = sorted(path for path in updates if Path(path).suffix == ".py")
    if not python_paths and not updates:
        return {}

    target_paths = python_paths or sorted(updates)
    language = _language_for_paths(target_paths)
    spec: dict[str, Any] = {
        "artifact_id": artifact_id or "",
        "target_paths": target_paths,
    }
    if language:
        spec["language"] = language
    if python_paths:
        spec["kind"] = "syntax"
        spec["require_empty_stderr"] = True
    else:
        spec["kind"] = "file_exists"
    if provenance:
        spec["artifact_paths"] = provenance
    return spec
def _verification_spec_from_payload(
    payload: Any, fallback_spec: dict[str, Any] | None = None
) -> tuple[list[str] | str, dict[str, Any]]:
    parsed_command: list[str] | str
    spec = dict(fallback_spec or {})

    if isinstance(payload, dict):
        command_value = (
            payload.get("command")
            or payload.get("verification_command")
            or payload.get("cmd")
        )
        if isinstance(command_value, list):
            parsed_command = [str(item) for item in command_value]
        elif isinstance(command_value, str):
            parsed_command = shlex.split(command_value)
        else:
            parsed_command = (
                json.dumps(command_value) if command_value is not None else ""
            )

        spec_payload = payload.get("verification_spec") or payload.get("expected") or {}
        if isinstance(spec_payload, dict) and spec_payload:
            spec.update({str(key): value for key, value in spec_payload.items()})
        artifact_id = payload.get("artifact_id")
        if artifact_id and not spec.get("artifact_id"):
            spec["artifact_id"] = str(artifact_id)
        target_paths = payload.get("target_paths")
        if (
            isinstance(target_paths, list)
            and target_paths
            and not spec.get("target_paths")
        ):
            spec["target_paths"] = [str(path) for path in target_paths]
        return parsed_command, spec

    if isinstance(payload, list):
        parsed_command = [str(item) for item in payload]
    else:
        parsed_command = str(payload)
    return parsed_command, spec
