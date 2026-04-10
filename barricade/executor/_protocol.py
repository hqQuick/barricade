from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._shared import dna_summary as _dna_summary, short_preview as _short_preview
from .._version import API_VERSION

__all__ = [
    "_dna_summary",
    "_short_preview",
    "_jsonable",
    "_payload_metrics",
    "_session_cursor",
    "_session_payload",
    "_market_entry_summary",
]

if TYPE_CHECKING:
    from .registry import ExecutionSession


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    if isinstance(value, Path):
        return str(value)
    return value


SESSION_PROTOCOL = {"name": "session-delta", "version": 1}


def _payload_metrics(payload: dict[str, Any]) -> dict[str, int]:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"approx_bytes": len(encoded), "field_count": len(payload)}


def _session_cursor(session: ExecutionSession) -> dict[str, Any]:
    return {
        "current_step": session.current_step,
        "total_steps": len(session.flattened),
        "current_token": session.current_token(),
        "remaining_steps": max(0, len(session.flattened) - session.current_step),
        "completed": session.completed,
    }


def _session_payload(
    session: ExecutionSession,
    status: str,
    delta: dict[str, Any] | None = None,
    *,
    instruction: str | None = None,
    tool_hint: str | None = None,
    include_dna: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "api_version": API_VERSION,
        "protocol": SESSION_PROTOCOL,
        "session_id": session.session_id,
        "status": status,
        "revision": session.revision,
        "current_step": session.current_step,
        "total_steps": len(session.flattened),
        "current_token": session.current_token(),
        "cursor": _session_cursor(session),
    }
    if include_dna:
        payload["dna"] = session.dna
    if instruction is not None:
        payload["instruction"] = instruction
    if tool_hint is not None:
        payload["tool_hint"] = tool_hint
    if delta:
        payload["delta"] = _jsonable(delta)
    payload["payload_metrics"] = _payload_metrics(payload)
    return _jsonable(payload)


def _market_entry_summary(artifact: Any) -> dict[str, Any]:
    payload = {
        "artifact_id": artifact.artifact_id,
        "token": artifact.token,
        "kind": artifact.kind,
        "creator": artifact.creator,
        "epoch": artifact.epoch,
        "price": artifact.price,
        "score": artifact.score,
        "status": artifact.status,
        "content": _short_preview(artifact.content, 140),
    }
    metadata = artifact.metadata
    if isinstance(metadata, dict) and metadata:
        compact_metadata: dict[str, Any] = {}
        for key in ("specialization", "motif_sig", "family_signature"):
            if key in metadata:
                compact_metadata[key] = _jsonable(metadata[key])
        signals = metadata.get("signals")
        if isinstance(signals, dict):
            compact_metadata["signals"] = {
                "token_count": signals.get("token_count"),
                "line_count": signals.get("line_count"),
            }
        if compact_metadata:
            payload["metadata"] = compact_metadata
    return payload
