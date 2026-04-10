from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


COMPACT_DETAIL_LIMIT = 3

_ARTIFACT_SUMMARY_KEYS = (
    "artifact_id",
    "token",
    "kind",
    "creator",
    "epoch",
    "price",
    "score",
    "fitness",
    "governed_fitness",
    "selection_score",
    "stability_score",
    "economic_stability",
    "task_threshold_pass",
    "solve_rate_test",
    "market_score",
    "profit",
    "wallet_end",
    "specialization",
    "family_sig",
    "motif_sig",
    "lineage_id",
    "status",
)


def _compact_mapping(
    mapping: Mapping[str, Any] | None, keys: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return {key: mapping[key] for key in keys if key in mapping}


def _summarize_artifact(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {}

    summary = _compact_mapping(artifact, _ARTIFACT_SUMMARY_KEYS)

    dna = artifact.get("dna")
    if isinstance(dna, Sequence) and not isinstance(dna, (str, bytes, bytearray)):
        summary["dna_length"] = len(dna)

    parent_ids = artifact.get("parent_ids")
    if isinstance(parent_ids, Sequence) and not isinstance(
        parent_ids, (str, bytes, bytearray)
    ):
        summary["parent_count"] = len(parent_ids)

    return summary


def _summarize_top_artifacts(
    result: Mapping[str, Any], detail_limit: int
) -> dict[str, Any]:
    top_artifacts: dict[str, Any] = {}

    for key in ("best_governed", "best_raw"):
        artifact = result.get(key)
        if isinstance(artifact, Mapping):
            top_artifacts[key] = _summarize_artifact(artifact)

    for key in ("archive_stable_top3", "archive_raw_top3"):
        archive = result.get(key)
        if isinstance(archive, Sequence) and not isinstance(
            archive, (str, bytes, bytearray)
        ):
            top_artifacts[key] = [
                _summarize_artifact(item)
                for item in list(archive)[:detail_limit]
                if isinstance(item, Mapping)
            ]

    return top_artifacts


def _summarize_outcome_record(result: Mapping[str, Any]) -> dict[str, Any]:
    outcome_record = result.get("outcome_record")
    if not isinstance(outcome_record, Mapping):
        return {}

    summary = _compact_mapping(
        outcome_record,
        (
            "benchmark_signature",
            "task_shape_signature",
            "outcome_score",
            "success",
            "outcome_class",
            "source",
        ),
    )

    outcome_summary = outcome_record.get("summary")
    if isinstance(outcome_summary, Mapping):
        summary["summary"] = _compact_mapping(
            outcome_summary,
            (
                "governed_mean",
                "solve_rate_test_mean",
                "stability_score_mean",
                "task_threshold_pass_mean",
                "specialization_entropy",
                "rotation_market_signal",
                "market_score_mean",
                "artifact_yield_mean",
                "gen_efficiency_mean",
            ),
        )

    metadata = outcome_record.get("metadata")
    if isinstance(metadata, Mapping):
        summary["metadata"] = _compact_mapping(
            metadata,
            (
                "trials",
                "population",
                "state_dir",
                "state_fingerprint",
                "benchmark_reused",
            ),
        )

    return summary


def _compact_comparison_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status", ""),
        "baseline_label": report.get("baseline_label", ""),
        "candidate_label": report.get("candidate_label", ""),
        "delta": report.get("delta", {}),
        "selection_delta": report.get("selection_delta", {}),
        "rotation_delta": report.get("rotation_delta", {}),
        "parallax_delta": report.get("parallax_delta", {}),
        "score": report.get("score", {}),
        "winner": report.get("winner", ""),
        "recommendation": report.get("recommendation", ""),
        "risk_flags": list(report.get("risk_flags", []))
        if isinstance(report.get("risk_flags", []), list)
        else [],
    }


def compact_benchmark_result(
    result: Mapping[str, Any] | None, detail_limit: int = COMPACT_DETAIL_LIMIT
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}

    compact: dict[str, Any] = {}
    for key in (
        "summary",
        "selection_report",
        "rotation_analysis",
        "parallax_telemetry",
        "controller_summary",
        "phase_summary",
        "task_shape_signature",
    ):
        value = result.get(key)
        if value is not None:
            compact[key] = value

    top_artifacts = _summarize_top_artifacts(result, detail_limit)
    if top_artifacts:
        compact["top_artifacts"] = top_artifacts

    outcome_record_summary = _summarize_outcome_record(result)
    if outcome_record_summary:
        compact["outcome_record_summary"] = outcome_record_summary

    return compact


__all__ = ["compact_benchmark_result"]
