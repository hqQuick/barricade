from __future__ import annotations

from pathlib import Path
from typing import Any

from ._version import with_api_version

from .feed_derived_dna.persistence import (
    DISCOVERABLES_DIR,
    OUTCOME_LEDGER_FILE,
    MACRO_LIBRARY_FILE,
    MOTIF_CACHE_FILE,
    RUN_LOG_FILE,
    LINEAGE_LOG_FILE,
    TASK_SHAPE_PRIOR_LOG_FILE,
    BENCHMARK_RUNS_DIR,
    _read_json,
    _read_jsonl,
    load_execution_feedback,
)


def _summarize_macro(name: str, sequence: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "sequence": sequence,
        "length": len(sequence),
        "pattern": " -> ".join(sequence),
    }


def _summarize_macro_from_payload(
    name: str, sequence: Any, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    if isinstance(sequence, dict):
        tokens = sequence.get("tokens", [])
    else:
        tokens = sequence
    if not isinstance(tokens, list):
        return None

    summary = _summarize_macro(name, [str(token) for token in tokens])
    if metadata:
        summary["trust"] = round(float(metadata.get("trust", 0.0) or 0.0), 3)
        summary["reuse_count"] = int(metadata.get("reuse_count", 0) or 0)
        summary["decay"] = int(metadata.get("decay", 0) or 0)
    return summary


def _summarize_motif(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "motif": name,
        "occurrences": int(meta.get("count", 0)),
        "avg_quality": round(float(meta.get("mean_governed", 0.0)), 3),
        "avg_stability": round(float(meta.get("mean_stability", 0.0)), 3),
        "best_fitness": round(float(meta.get("best_governed_fitness", 0.0)), 3),
        "specializations": dict(meta.get("specializations", {})),
    }


def _summarize_run(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary", {})
    learned_macros = entry.get("learned_macros", {})
    if isinstance(learned_macros, dict):
        macro_names = list(learned_macros.keys())
    elif isinstance(learned_macros, list):
        macro_names = [str(item) for item in learned_macros]
    else:
        macro_names = []
    return {
        "status": summary.get("status", "unknown"),
        "artifacts": summary.get("artifact_count", 0),
        "patches": summary.get("patch_update_count", summary.get("patch_updates", 0)),
        "verified": summary.get("verification_passed", False),
        "dna": entry.get("feed_prior_dna", [])[:5],
        "macros_learned": macro_names[:8],
        "macro_count": len(macro_names),
    }


def _summarize_outcome(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary", {})
    outcome_vector = entry.get("outcome_vector", {})
    return {
        "benchmark_signature": entry.get("benchmark_signature", ""),
        "task_shape_signature": entry.get("task_shape_signature", ""),
        "outcome_class": entry.get("outcome_class", "failure"),
        "success": bool(entry.get("success", False)),
        "score": round(float(entry.get("outcome_score", 0.0) or 0.0), 3),
        "governed_mean": round(float(summary.get("governed_mean", 0.0) or 0.0), 3),
        "solve_rate_test_mean": round(
            float(summary.get("solve_rate_test_mean", 0.0) or 0.0), 3
        ),
        "task_threshold_pass_mean": round(
            float(summary.get("task_threshold_pass_mean", 0.0) or 0.0), 3
        ),
        "rotation_market_signal": round(
            float(summary.get("rotation_market_signal", 0.0) or 0.0), 3
        ),
        "controller_transition_count": int(
            outcome_vector.get("controller_transition_count", 0) or 0
        ),
    }


def _summarize_feedback(entry: dict[str, Any]) -> dict[str, Any]:
    completion_summary = entry.get("completion_summary", {})
    decision_policy = entry.get("decision_policy", {})
    execution_learning = entry.get("execution_learning", {})
    return {
        "session_id": entry.get("session_id", ""),
        "task_shape_signature": entry.get("task_shape_signature", ""),
        "route_hint": entry.get("route_hint", ""),
        "decision_mode": str(decision_policy.get("mode", "act") or "act"),
        "verification_passed": bool(
            completion_summary.get("verification_passed", False)
        ),
        "verification_pass_rate": round(
            float(completion_summary.get("verification_pass_rate", 0.0) or 0.0), 3
        ),
        "successful_trace_count": int(
            completion_summary.get("successful_trace_count", 0) or 0
        ),
        "learned_macro_count": int(
            completion_summary.get("learned_macro_count", 0) or 0
        ),
        "artifact_count": int(
            completion_summary.get("artifact_count", entry.get("artifact_count", 0))
            or 0
        ),
        "market_count": int(entry.get("market_count", 0) or 0),
        "dispatch_update_count": int(entry.get("dispatch_update_count", 0) or 0),
        "feedback_learning": {
            "successful_trace_count": int(
                execution_learning.get("successful_trace_count", 0) or 0
            ),
            "learned_macro_count": int(
                execution_learning.get("learned_macro_count", 0) or 0
            ),
        },
    }


def inspect_state(
    state_dir: str | Path | None,
    max_macros: int = 20,
    max_motifs: int = 20,
    max_runs: int = 10,
) -> dict[str, Any]:
    if not state_dir:
        return with_api_version(
            {
                "available": False,
                "message": "No state directory configured. Use state_dir parameter to enable persistent memory.",
            }
        )

    root = Path(state_dir).expanduser()
    if not root.exists():
        return with_api_version(
            {
                "available": False,
                "message": f"State directory does not exist: {root}",
            }
        )

    discoverables = root / DISCOVERABLES_DIR

    macro_data = _read_json(discoverables / MACRO_LIBRARY_FILE)
    macros_raw = (
        macro_data.get("macros", macro_data) if isinstance(macro_data, dict) else {}
    )
    macro_metadata = (
        macro_data.get("metadata", {}) if isinstance(macro_data, dict) else {}
    )
    macro_trust = (
        macro_metadata.get("trust", {}) if isinstance(macro_metadata, dict) else {}
    )
    macro_reuse_count = (
        macro_metadata.get("reuse_count", {})
        if isinstance(macro_metadata, dict)
        else {}
    )
    macro_decay = (
        macro_metadata.get("decay", {}) if isinstance(macro_metadata, dict) else {}
    )
    macros = []
    for name, seq in list(macros_raw.items())[:max_macros]:
        summary = _summarize_macro_from_payload(
            name,
            seq,
            {
                "trust": macro_trust.get(name, 0.0),
                "reuse_count": macro_reuse_count.get(name, 0),
                "decay": macro_decay.get(name, 0),
            },
        )
        if summary is not None:
            macros.append(summary)

    motif_data = _read_json(discoverables / MOTIF_CACHE_FILE)
    motifs_raw = (
        motif_data.get("motifs", motif_data) if isinstance(motif_data, dict) else {}
    )
    motifs = [
        _summarize_motif(name, meta)
        for name, meta in list(motifs_raw.items())[:max_motifs]
        if isinstance(meta, dict)
    ]

    runs = _read_jsonl(root / RUN_LOG_FILE)
    run_summaries = [_summarize_run(r) for r in runs[-max_runs:]]

    lineages = _read_jsonl(root / LINEAGE_LOG_FILE)
    lineage_count = len(lineages)

    priors = _read_jsonl(root / TASK_SHAPE_PRIOR_LOG_FILE)
    prior_count = len(priors)

    outcomes = _read_jsonl(root / OUTCOME_LEDGER_FILE)
    outcome_count = len(outcomes)

    feedback_entries = load_execution_feedback(root)
    feedback_count = len(feedback_entries)

    benchmark_run_dir = root / BENCHMARK_RUNS_DIR
    benchmark_count = (
        len(list(benchmark_run_dir.glob("*.json"))) if benchmark_run_dir.exists() else 0
    )

    recent_outcomes = [_summarize_outcome(entry) for entry in outcomes[-max_runs:]]
    successful_outcomes = sum(1 for entry in recent_outcomes if entry["success"])
    recent_feedback = [
        _summarize_feedback(entry) for entry in feedback_entries[-max_runs:]
    ]
    successful_feedback = sum(
        1 for entry in recent_feedback if entry["verification_passed"]
    )

    top_macros = sorted(macros, key=lambda m: m["length"], reverse=True)[:5]
    frequent_motifs = sorted(motifs, key=lambda m: m["occurrences"], reverse=True)[:5]
    recent_failures = [r for r in run_summaries if not r["verified"]][-3:]

    macros = macros[:max_macros]
    motifs = motifs[:max_motifs]
    run_summaries = run_summaries[-max_runs:]

    actionable_context: list[str] = []
    if macros:
        actionable_context.append(
            f"{len(macros)} learned macros available. Top patterns: {', '.join(m['pattern'] for m in top_macros)}"
        )
    if frequent_motifs:
        motif_strings = [f"{m['motif']}({m['occurrences']}x)" for m in frequent_motifs]
        actionable_context.append(f"Common motifs: {', '.join(motif_strings)}")
    if recent_failures:
        actionable_context.append(
            f"{len(recent_failures)} recent run(s) failed verification"
        )
    if recent_outcomes:
        actionable_context.append(
            f"{successful_outcomes}/{len(recent_outcomes)} recent outcome(s) were successful"
        )
    if recent_feedback:
        actionable_context.append(
            f"{successful_feedback}/{len(recent_feedback)} recent execution feedback item(s) passed verification"
        )

    return with_api_version(
        {
            "available": True,
            "state_root": str(root),
            "summary": {
                "macro_count": len(macros_raw),
                "motif_count": len(motifs),
                "total_runs": len(run_summaries),
                "lineage_entries": lineage_count,
                "prior_entries": prior_count,
                "outcome_entries": outcome_count,
                "feedback_entries": feedback_count,
                "benchmark_runs": benchmark_count,
            },
            "macros": macros,
            "motifs": motifs,
            "recent_runs": run_summaries,
            "recent_outcomes": recent_outcomes,
            "recent_feedback": recent_feedback,
            "actionable_context": actionable_context,
            "recent_failures": recent_failures,
        }
    )
