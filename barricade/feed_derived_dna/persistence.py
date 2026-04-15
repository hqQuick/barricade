from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .._validation import validate_workspace_root
from .models import Individual


__all__ = [
    "benchmark_run_path",
    "benchmark_run_signature",
    "benchmark_state_fingerprint",
    "load_forbidden_subsequence_memory",
    "load_execution_feedback",
    "load_benchmark_run",
    "load_outcome_ledger",
    "load_macro_library",
    "load_motif_cache",
    "load_task_shape_priors",
    "resolve_state_root",
    "save_benchmark_run",
    "save_execution_feedback",
    "save_forbidden_subsequence_memory",
    "save_outcome_ledger",
    "save_discoverables",
    "save_lineage_archive",
    "save_run_summary",
    "save_task_shape_prior",
    "task_shape_signature",
    "serialize_individual",
]

DISCOVERABLES_DIR = "discoverables"
MACRO_LIBRARY_FILE = "macro_library.json"
MOTIF_CACHE_FILE = "motif_cache.json"
FORBIDDEN_SUBSEQUENCE_MEMORY_FILE = "forbidden_subsequence_memory.json"
LINEAGE_LOG_FILE = "lineages.jsonl"
RUN_LOG_FILE = "runs.jsonl"
BENCHMARK_RUNS_DIR = "benchmark_runs"
BENCHMARK_RUN_LOG_FILE = "benchmark_runs.jsonl"
TASK_SHAPE_PRIOR_LOG_FILE = "task_shape_priors.jsonl"
OUTCOME_LEDGER_FILE = "outcome_ledger.jsonl"
EXECUTION_FEEDBACK_FILE = "execution_feedback.jsonl"
MAX_RETENTION_ENTRIES = 250
MAX_RETENTION_BENCHMARKS = 100


def resolve_state_root(state_dir: str | Path | None):
    validated_state_dir = validate_workspace_root(state_dir)
    if validated_state_dir is None:
        return None

    root = validated_state_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / DISCOVERABLES_DIR).mkdir(parents=True, exist_ok=True)
    return root


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")


def _rewrite_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(_jsonable(entry), sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def serialize_individual(individual: Individual) -> dict[str, Any]:
    return _jsonable(asdict(individual))


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_state_fingerprint(
    learned_macros: dict[str, Any],
    motif_archive: dict[str, Any],
    forbidden_subsequence_memory: dict[str, Any] | None = None,
) -> str:
    forbidden_patterns = []
    if isinstance(forbidden_subsequence_memory, dict):
        pattern_records = forbidden_subsequence_memory.get("patterns", [])
        if isinstance(pattern_records, list):
            seen: set[tuple[str, ...]] = set()
            for record in pattern_records:
                tokens: list[str] = []
                if isinstance(record, dict):
                    raw_tokens = record.get("tokens", [])
                    if isinstance(raw_tokens, list):
                        tokens = [str(token) for token in raw_tokens if str(token)]
                elif isinstance(record, (list, tuple)):
                    tokens = [str(token) for token in record if str(token)]
                if len(tokens) < 2:
                    continue
                key = tuple(tokens)
                if key in seen:
                    continue
                seen.add(key)
                forbidden_patterns.append(tokens)
    forbidden_patterns.sort()
    return _stable_digest(
        {
            "macros": learned_macros,
            "motifs": motif_archive,
            "forbidden_subsequences": forbidden_patterns,
        }
    )


def benchmark_run_signature(
    benchmark_inputs: dict[str, Any], state_fingerprint: str
) -> str:
    return _stable_digest(
        {"inputs": benchmark_inputs, "state_fingerprint": state_fingerprint}
    )


def task_shape_signature(shape_profile: dict[str, Any]) -> str:
    return _stable_digest({"shape_profile": shape_profile})


def benchmark_run_path(state_root: Path, signature: str) -> Path:
    return state_root / BENCHMARK_RUNS_DIR / f"{signature}.json"


def load_benchmark_run(state_root, signature: str):
    if state_root is None:
        return {}

    path = benchmark_run_path(state_root, signature)
    payload = _read_json(path)
    if not payload:
        return {}
    if payload.get("signature") != signature:
        return {}

    result = payload.get("result", payload)
    return result if isinstance(result, dict) else {}


def load_outcome_ledger(state_root):
    if state_root is None:
        return []

    entries = _read_jsonl(state_root / OUTCOME_LEDGER_FILE)
    return [entry for entry in entries if isinstance(entry, dict)]


def load_execution_feedback(state_root):
    if state_root is None:
        return []

    entries = _read_jsonl(state_root / EXECUTION_FEEDBACK_FILE)
    return [entry for entry in entries if isinstance(entry, dict)]


def _normalize_forbidden_subsequence_record(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, dict):
        raw_tokens = entry.get("tokens", [])
        tokens = (
            [str(token) for token in raw_tokens if str(token)]
            if isinstance(raw_tokens, list)
            else []
        )
        if len(tokens) < 2:
            return None
        try:
            count = int(entry.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        try:
            total_score = float(
                entry.get("total_score", entry.get("score", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            total_score = 0.0
        metadata = entry.get("metadata", {})
        return {
            "tokens": tokens,
            "count": max(0, count),
            "total_score": max(0.0, total_score),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    if isinstance(entry, (list, tuple)):
        tokens = [str(token) for token in entry if str(token)]
        if len(tokens) < 2:
            return None
        return {
            "tokens": tokens,
            "count": 1,
            "total_score": 0.0,
            "metadata": {},
        }
    return None


def load_forbidden_subsequence_memory(state_root, limit: int = 12):
    if state_root is None:
        return {
            "available": False,
            "pattern_count": 0,
            "patterns": [],
            "pattern_records": [],
            "top_patterns": [],
            "top_pattern_records": [],
            "metadata": {},
        }

    payload = _read_json(
        state_root / DISCOVERABLES_DIR / FORBIDDEN_SUBSEQUENCE_MEMORY_FILE
    )
    records = payload.get("patterns", [])
    normalized: list[dict[str, Any]] = []
    if isinstance(records, list):
        for entry in records:
            normalized_record = _normalize_forbidden_subsequence_record(entry)
            if normalized_record is not None:
                normalized.append(normalized_record)

    normalized.sort(
        key=lambda record: (
            -int(record.get("count", 0) or 0),
            -float(record.get("total_score", 0.0) or 0.0),
            tuple(record.get("tokens", [])),
        )
    )
    for record in normalized:
        count = max(1, int(record.get("count", 0) or 0))
        total_score = float(record.get("total_score", 0.0) or 0.0)
        record["score"] = round(total_score / count, 3)

    metadata = (
        payload.get("metadata", {})
        if isinstance(payload.get("metadata", {}), dict)
        else {}
    )
    return {
        "available": bool(normalized),
        "pattern_count": len(normalized),
        "patterns": [record["tokens"] for record in normalized],
        "pattern_records": normalized,
        "top_patterns": [record["tokens"] for record in normalized[:limit]],
        "top_pattern_records": normalized[:limit],
        "metadata": metadata,
    }


def save_forbidden_subsequence_memory(
    state_root,
    patterns: list[list[str]] | list[tuple[str, ...]],
    metadata: dict[str, Any] | None = None,
    score: float = 0.0,
):
    if state_root is None:
        return

    if not patterns:
        return

    path = state_root / DISCOVERABLES_DIR / FORBIDDEN_SUBSEQUENCE_MEMORY_FILE
    payload = _read_json(path)
    existing_records = payload.get("patterns", [])
    records_by_key: dict[tuple[str, ...], dict[str, Any]] = {}

    if isinstance(existing_records, list):
        for entry in existing_records:
            normalized_record = _normalize_forbidden_subsequence_record(entry)
            if normalized_record is None:
                continue
            key = tuple(normalized_record["tokens"])
            records_by_key[key] = normalized_record

    increment = max(0.0, float(score or 0.0))
    for pattern in patterns:
        tokens = [str(token) for token in pattern if str(token)]
        if len(tokens) < 2:
            continue
        key = tuple(tokens)
        record = records_by_key.setdefault(
            key,
            {
                "tokens": tokens,
                "count": 0,
                "total_score": 0.0,
                "metadata": {},
            },
        )
        record["count"] = int(record.get("count", 0) or 0) + 1
        record["total_score"] = round(
            float(record.get("total_score", 0.0) or 0.0) + increment, 3
        )
        if metadata:
            existing_metadata = record.get("metadata", {})
            record["metadata"] = (
                {**existing_metadata, **metadata}
                if isinstance(existing_metadata, dict)
                else dict(metadata)
            )

    ordered_records = sorted(
        records_by_key.values(),
        key=lambda record: (
            -int(record.get("count", 0) or 0),
            -float(record.get("total_score", 0.0) or 0.0),
            tuple(record.get("tokens", [])),
        ),
    )

    _write_json(
        path,
        {
            "kind": "forbidden_subsequence_memory",
            "patterns": ordered_records,
            "metadata": metadata or payload.get("metadata", {}),
        },
    )


def load_task_shape_priors(state_root):
    if state_root is None:
        return []
    return _read_jsonl(state_root / TASK_SHAPE_PRIOR_LOG_FILE)


def save_task_shape_prior(
    state_root, record: dict[str, Any], metadata: dict[str, Any] | None = None
):
    if state_root is None:
        return

    payload = {
        "kind": "task_shape_prior",
        "signature": record.get("signature", ""),
        "score": record.get("score", 0.0),
        "shape_profile": record.get("shape_profile", {}),
        "snapshot": record.get("snapshot", {}),
        "source": record.get("source", "benchmark"),
        "metadata": metadata or record.get("metadata", {}),
    }
    _append_jsonl(state_root / TASK_SHAPE_PRIOR_LOG_FILE, payload)


def load_macro_library(state_root):
    if state_root is None:
        return {}

    payload = _read_json(state_root / DISCOVERABLES_DIR / MACRO_LIBRARY_FILE)
    macros = payload.get("macros", payload)
    if not isinstance(macros, dict):
        return {}

    loaded = {}
    for name, sequence in macros.items():
        if isinstance(sequence, dict):
            tokens = [str(token) for token in sequence.get("tokens", [])]
            trust = float(sequence.get("trust", 0.0) or 0.0)
            reuse_count = int(sequence.get("reuse_count", 0) or 0)
            decay = int(sequence.get("decay", 0) or 0)
            if trust < 0.45 or decay >= 3:
                continue
            if reuse_count < 2:
                continue
        elif isinstance(sequence, list):
            tokens = [str(token) for token in sequence]
            trust = 0.0
            reuse_count = 0
            decay = 0
        else:
            continue
        if len(tokens) < 2 or len(tokens) > 6:
            continue
        if all(token.startswith("LM") for token in tokens):
            if len(set(tokens)) <= 1:
                continue
            if len(tokens) > 4:
                continue
        if (
            tokens[0].startswith("LM")
            and tokens[-1].startswith("LM")
            and len(set(tokens)) <= 2
        ):
            continue
        loaded[str(name)] = tokens
    return loaded


def load_motif_cache(state_root):
    if state_root is None:
        return {}

    payload = _read_json(state_root / DISCOVERABLES_DIR / MOTIF_CACHE_FILE)
    motifs = payload.get("motifs", payload)
    if not isinstance(motifs, dict):
        return {}

    loaded = {}
    for motif, meta in motifs.items():
        if not isinstance(meta, dict):
            continue
        loaded[str(motif)] = {
            "count": int(meta.get("count", 0)),
            "mean_governed": float(meta.get("mean_governed", 0.0)),
            "mean_profit": float(meta.get("mean_profit", 0.0)),
            "mean_stability": float(meta.get("mean_stability", 0.0)),
            "best_governed_fitness": float(meta.get("best_governed_fitness", 0.0)),
            "specializations": dict(meta.get("specializations", {})),
        }
    return loaded


def save_discoverables(state_root, learned_macros, motif_archive, metadata=None):
    if state_root is None:
        return

    discoverables_dir = state_root / DISCOVERABLES_DIR
    filtered_macros = {}
    for name, sequence in learned_macros.items():
        tokens = sequence.get("tokens", []) if isinstance(sequence, dict) else sequence
        if not isinstance(tokens, list):
            continue
        normalized_tokens = [str(token) for token in tokens]
        if not (2 <= len(normalized_tokens) <= 6):
            continue
        if (
            normalized_tokens[0].startswith("LM")
            and normalized_tokens[-1].startswith("LM")
            and len(set(normalized_tokens)) <= 2
        ):
            continue
        filtered_macros[str(name)] = normalized_tokens
    macro_metadata = {
        "trust": (metadata or {}).get("macro_trust", {}),
        "reuse_count": (metadata or {}).get("macro_reuse_count", {}),
        "decay": (metadata or {}).get("macro_decay", {}),
        "evidence": (metadata or {}).get("macro_evidence", {}),
    }
    cached_motifs = dict(motif_archive)
    if not cached_motifs:
        for name, sequence in filtered_macros.items():
            tokens = sequence
            motif_name = "|".join(tokens[:3]) if len(tokens) >= 3 else "|".join(tokens)
            if not motif_name:
                continue
            cached_motifs.setdefault(
                motif_name,
                {
                    "count": 1,
                    "mean_governed": 0.0,
                    "mean_profit": 0.0,
                    "mean_stability": 0.0,
                    "best_governed_fitness": 0.0,
                    "specializations": {},
                },
            )
    macro_payload = {
        "macros": filtered_macros,
        "metadata": macro_metadata,
    }
    motif_payload = {
        "motifs": cached_motifs,
        "metadata": metadata or {},
    }
    _write_json(discoverables_dir / MACRO_LIBRARY_FILE, macro_payload)
    _write_json(discoverables_dir / MOTIF_CACHE_FILE, motif_payload)


def _retention_slice(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(entries) <= limit:
        return entries
    head = entries[-limit:]
    return head


def save_outcome_ledger(state_root, record, metadata=None):
    if state_root is None:
        return

    payload = dict(record)
    if metadata is not None:
        payload.setdefault("metadata", metadata)
    path = state_root / OUTCOME_LEDGER_FILE
    entries = _read_jsonl(path)
    entries.append(payload)
    _rewrite_jsonl(path, _retention_slice(entries, MAX_RETENTION_ENTRIES))


def save_execution_feedback(state_root, record, metadata=None):
    if state_root is None:
        return

    payload = dict(record)
    if metadata is not None:
        payload.setdefault("metadata", metadata)
    path = state_root / EXECUTION_FEEDBACK_FILE
    entries = _read_jsonl(path)
    entries.append(payload)
    _rewrite_jsonl(path, _retention_slice(entries, MAX_RETENTION_ENTRIES))


def save_run_summary(state_root, result, metadata=None):
    if state_root is None:
        return

    payload = dict(result)
    if metadata is not None:
        payload.setdefault("metadata", metadata)
    path = state_root / RUN_LOG_FILE
    entries = _read_jsonl(path)
    entries.append(payload)
    _rewrite_jsonl(path, _retention_slice(entries, MAX_RETENTION_ENTRIES))


def save_lineage_archive(state_root, label, individuals, metadata=None):
    if state_root is None:
        return

    payload = {
        "label": label,
        "count": len(individuals),
        "metadata": metadata or {},
        "individuals": [serialize_individual(individual) for individual in individuals],
    }
    path = state_root / LINEAGE_LOG_FILE
    entries = _read_jsonl(path)
    entries.append(payload)
    _rewrite_jsonl(path, _retention_slice(entries, MAX_RETENTION_ENTRIES))


def save_benchmark_run(
    state_root, signature, result, metadata=None, reused: bool = False
):
    if state_root is None:
        return

    payload = {"signature": signature, "result": result}
    if metadata is not None:
        payload["metadata"] = metadata
    path = benchmark_run_path(state_root, signature)
    _write_json(path, payload)
    log_path = state_root / BENCHMARK_RUN_LOG_FILE
    entries = _read_jsonl(log_path)
    log_payload = dict(payload)
    log_payload["kind"] = "benchmark"
    log_payload["result_path"] = str(path)
    log_payload["reused"] = reused
    entries.append(log_payload)
    _rewrite_jsonl(log_path, _retention_slice(entries, MAX_RETENTION_BENCHMARKS))
