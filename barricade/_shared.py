from __future__ import annotations

from collections.abc import Iterable, Mapping
import subprocess
from pathlib import Path
from typing import Any, cast


VERIFICATION_TIMEOUT_SECONDS = 300


def run_command_with_timeout(
    command: list[str] | tuple[str, ...],
    cwd: Path,
    timeout: int = VERIFICATION_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        raise TypeError("command must be a sequence of arguments, not a string")
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = str(getattr(exc, "output", "") or "")
        stderr = f"Verification timed out after {timeout} seconds."
        extra_stderr = str(getattr(exc, "stderr", "") or "")
        if extra_stderr:
            stderr = f"{stderr}\n{extra_stderr}"
        return subprocess.CompletedProcess(
            args=getattr(exc, "cmd", command),
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )


def as_mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def as_macro_library(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    macro_library: dict[str, list[str]] = {}
    mapping = cast(Mapping[Any, Any], value)
    for name, sequence in mapping.items():
        if isinstance(sequence, (list, tuple)):
            macro_library[str(name)] = [str(token) for token in sequence]
    return macro_library


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def short_preview(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def dna_summary(tokens: list[str], max_tokens: int = 5) -> str:
    if not tokens:
        return ""
    preview = tokens[:max_tokens]
    if len(tokens) > max_tokens:
        preview.append("...")
    return " -> ".join(preview)


def is_unresolved_macro_token(token: str, learned_macros: Mapping[str, object]) -> bool:
    return (
        token.startswith("LM") and token[2:].isdigit() and token not in learned_macros
    )
