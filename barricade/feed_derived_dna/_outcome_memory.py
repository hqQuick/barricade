from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .persistence import load_outcome_ledger, task_shape_signature


OUTCOME_SUCCESS_MIN_SOLVE_RATE = 0.78
OUTCOME_SUCCESS_MIN_TASK_PASS = 0.72
OUTCOME_SUCCESS_MIN_STABILITY = 0.58


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _coerce_trace(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(token) for token in value if str(token)]


def _trace_from_payload(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    return _coerce_trace(payload.get("dna", []))


def _success_trace_bank_from_record(
    record: Mapping[str, Any], similarity: float
) -> list[dict[str, Any]]:
    if not bool(record.get("success", False)):
        return []

    outcome_score = float(record.get("outcome_score", 0.0) or 0.0)
    base_weight = max(
        0.45, min(4.0, (0.75 + outcome_score / 140.0) * max(0.5, similarity))
    )
    source_weights = (
        ("best_governed", 1.0),
        ("archive_stable_top3", 0.75),
        ("best_raw", 0.6),
        ("archive_raw_top3", 0.45),
    )

    trace_bank: dict[tuple[str, ...], dict[str, Any]] = {}
    for source, source_weight in source_weights:
        payload = record.get(source)
        if source.startswith("archive"):
            for index, item in enumerate(_as_list(payload)[:3]):
                trace = _trace_from_payload(item)
                if not trace:
                    continue
                key = tuple(trace)
                weight = round(base_weight * source_weight * (1.0 - 0.05 * index), 3)
                existing = trace_bank.get(key)
                if existing is None or weight > float(
                    existing.get("weight", 0.0) or 0.0
                ):
                    trace_bank[key] = {
                        "trace": trace,
                        "weight": weight,
                        "source": f"{source}[{index}]",
                        "benchmark_signature": str(
                            record.get("benchmark_signature", "") or ""
                        ),
                        "outcome_score": round(outcome_score, 3),
                        "similarity": round(similarity, 3),
                    }
        else:
            trace = _trace_from_payload(payload)
            if not trace:
                continue
            key = tuple(trace)
            weight = round(base_weight * source_weight, 3)
            existing = trace_bank.get(key)
            if existing is None or weight > float(existing.get("weight", 0.0) or 0.0):
                trace_bank[key] = {
                    "trace": trace,
                    "weight": weight,
                    "source": source,
                    "benchmark_signature": str(
                        record.get("benchmark_signature", "") or ""
                    ),
                    "outcome_score": round(outcome_score, 3),
                    "similarity": round(similarity, 3),
                }

    return sorted(
        trace_bank.values(),
        key=lambda item: (
            -float(item.get("weight", 0.0) or 0.0),
            str(item.get("source", "")),
            tuple(item.get("trace", [])),
        ),
    )


def _shape_profile_from_benchmark_result(
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    task_pool = _as_list(benchmark_result.get("task_pool", []))
    focus_counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    task_names: list[str] = []
    for task in task_pool:
        if not isinstance(task, Mapping):
            continue
        focus_counts.update([str(task.get("focus", "planning") or "planning")])
        need_counts.update(
            {
                str(key): int(value)
                for key, value in _as_dict(task.get("needs", {})).items()
            }
        )
        name = str(task.get("name", "") or "")
        if name:
            task_names.append(name)

    curriculum_profile = _as_dict(benchmark_result.get("curriculum_profile", {}))
    feed_prior_dna = [
        str(token)
        for token in _as_list(benchmark_result.get("feed_prior_dna", []))[:12]
    ]
    return {
        "task_count": len(task_pool),
        "focus_counts": dict(focus_counts),
        "need_counts": dict(need_counts),
        "task_names": task_names[:6],
        "feed_prior_dna": feed_prior_dna,
        "curriculum_stage_counts": _as_dict(curriculum_profile.get("stage_counts", {})),
        "curriculum_stages": list(curriculum_profile.get("stages", []))
        if isinstance(curriculum_profile.get("stages", []), list)
        else [],
        "mode": str(benchmark_result.get("mode", "") or ""),
    }


def normalize_task_shape_profile(
    shape_profile: Mapping[str, Any] | None,
    benchmark_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = _as_dict(shape_profile)
    if not profile and benchmark_result is not None:
        profile = _shape_profile_from_benchmark_result(benchmark_result)
    if not profile:
        return {}

    normalized = dict(profile)
    normalized.pop("task_shape_signature", None)
    normalized["task_shape_signature"] = task_shape_signature(normalized)
    return normalized


def _outcome_score(
    summary: Mapping[str, Any], outcome_vector: Mapping[str, Any]
) -> float:
    governed_mean = float(summary.get("governed_mean", 0.0) or 0.0)
    solve_rate = float(summary.get("solve_rate_test_mean", 0.0) or 0.0)
    stability = float(summary.get("stability_score_mean", 0.0) or 0.0)
    task_pass = float(summary.get("task_threshold_pass_mean", 0.0) or 0.0)
    rotation = float(summary.get("rotation_market_signal", 0.0) or 0.0)
    market_score = float(summary.get("market_score_mean", 0.0) or 0.0)
    specialization_entropy = float(summary.get("specialization_entropy", 0.0) or 0.0)

    return round(
        governed_mean * 0.45
        + solve_rate * 120.0
        + stability * 55.0
        + task_pass * 35.0
        + rotation * 10.0
        + market_score * 3.0
        + float(outcome_vector.get("controller_transition_count", 0) or 0.0) * 0.5
        - specialization_entropy * 10.0,
        3,
    )


def _task_shape_similarity(
    current_profile: Mapping[str, Any], candidate_profile: Mapping[str, Any]
) -> float:
    from ..workflow_intake import task_shape_similarity

    return task_shape_similarity(dict(current_profile), dict(candidate_profile))


def _is_success(summary: Mapping[str, Any], outcome_score: float) -> bool:
    solve_rate = float(summary.get("solve_rate_test_mean", 0.0) or 0.0)
    task_pass = float(summary.get("task_threshold_pass_mean", 0.0) or 0.0)
    stability = float(summary.get("stability_score_mean", 0.0) or 0.0)
    if (
        solve_rate >= OUTCOME_SUCCESS_MIN_SOLVE_RATE
        and task_pass >= OUTCOME_SUCCESS_MIN_TASK_PASS
    ):
        return True
    if stability >= OUTCOME_SUCCESS_MIN_STABILITY and outcome_score >= 120.0:
        return True
    return outcome_score >= 150.0


def build_outcome_record(
    benchmark_result: dict[str, Any],
    benchmark_signature: str,
    shape_profile: Mapping[str, Any] | None,
    metadata: dict[str, Any] | None = None,
    source: str = "benchmark",
) -> dict[str, Any]:
    summary = _as_dict(benchmark_result.get("summary", {}))
    normalized_profile = normalize_task_shape_profile(shape_profile, benchmark_result)
    outcome_vector = {
        "governed_mean": float(summary.get("governed_mean", 0.0) or 0.0),
        "solve_rate_test_mean": float(summary.get("solve_rate_test_mean", 0.0) or 0.0),
        "stability_score_mean": float(summary.get("stability_score_mean", 0.0) or 0.0),
        "task_threshold_pass_mean": float(
            summary.get("task_threshold_pass_mean", 0.0) or 0.0
        ),
        "specialization_entropy": float(
            summary.get("specialization_entropy", 0.0) or 0.0
        ),
        "rotation_market_signal": float(
            summary.get("rotation_market_signal", 0.0) or 0.0
        ),
        "market_score_mean": float(summary.get("market_score_mean", 0.0) or 0.0),
        "artifact_yield_mean": float(summary.get("artifact_yield_mean", 0.0) or 0.0),
        "gen_efficiency_mean": float(summary.get("gen_efficiency_mean", 0.0) or 0.0),
        "controller_transition_count": int(
            _as_dict(benchmark_result.get("controller_summary", {})).get(
                "transition_count", 0
            )
        ),
        "phase_transition_count": int(
            _as_dict(benchmark_result.get("phase_summary", {})).get(
                "transition_count", 0
            )
        ),
        "parallax_replication_bursts": int(
            benchmark_result.get("parallax_replication_bursts", 0) or 0
        ),
        "parallax_replication_offspring": int(
            benchmark_result.get("parallax_replication_offspring", 0) or 0
        ),
    }
    outcome_score = _outcome_score(summary, outcome_vector)
    success = _is_success(summary, outcome_score)

    return {
        "kind": "outcome",
        "source": source,
        "benchmark_signature": benchmark_signature,
        "task_shape_signature": normalized_profile.get("task_shape_signature", ""),
        "shape_profile": normalized_profile,
        "outcome_score": outcome_score,
        "success": success,
        "outcome_class": "success" if success else "failure",
        "outcome_vector": outcome_vector,
        "summary": summary,
        "best_governed": _as_dict(benchmark_result.get("best_governed", {})),
        "best_raw": _as_dict(benchmark_result.get("best_raw", {})),
        "winner_specialization_counts": _as_dict(
            benchmark_result.get("winner_specialization_counts", {})
        ),
        "controller_summary": _as_dict(benchmark_result.get("controller_summary", {})),
        "phase_summary": _as_dict(benchmark_result.get("phase_summary", {})),
        "metadata": metadata or {},
    }


def _decorate_record(record: Mapping[str, Any], similarity: float) -> dict[str, Any]:
    summary = _as_dict(record.get("summary", {}))
    outcome_vector = _as_dict(record.get("outcome_vector", {}))
    return {
        "benchmark_signature": str(record.get("benchmark_signature", "") or ""),
        "task_shape_signature": str(record.get("task_shape_signature", "") or ""),
        "source": str(record.get("source", "") or ""),
        "similarity": round(similarity, 3),
        "outcome_score": round(float(record.get("outcome_score", 0.0) or 0.0), 3),
        "success": bool(record.get("success", False)),
        "outcome_class": str(record.get("outcome_class", "failure") or "failure"),
        "governed_mean": round(float(summary.get("governed_mean", 0.0) or 0.0), 3),
        "solve_rate_test_mean": round(
            float(summary.get("solve_rate_test_mean", 0.0) or 0.0), 3
        ),
        "stability_score_mean": round(
            float(summary.get("stability_score_mean", 0.0) or 0.0), 3
        ),
        "task_threshold_pass_mean": round(
            float(summary.get("task_threshold_pass_mean", 0.0) or 0.0), 3
        ),
        "rotation_market_signal": round(
            float(summary.get("rotation_market_signal", 0.0) or 0.0), 3
        ),
        "specialization_entropy": round(
            float(summary.get("specialization_entropy", 0.0) or 0.0), 3
        ),
        "controller_transition_count": int(
            outcome_vector.get("controller_transition_count", 0) or 0
        ),
        "phase_transition_count": int(
            outcome_vector.get("phase_transition_count", 0) or 0
        ),
        "parallax_replication_bursts": int(
            outcome_vector.get("parallax_replication_bursts", 0) or 0
        ),
        "parallax_replication_offspring": int(
            outcome_vector.get("parallax_replication_offspring", 0) or 0
        ),
        "success_trace_bank": _success_trace_bank_from_record(record, similarity),
    }


def load_outcome_memory(
    state_root,
    shape_profile: Mapping[str, Any] | None,
    benchmark_signature: str | None = None,
    limit: int = 5,
    min_similarity: float = 0.55,
) -> dict[str, Any]:
    entries = load_outcome_ledger(state_root)
    normalized_profile = normalize_task_shape_profile(shape_profile)
    if not entries:
        return {
            "available": False,
            "task_shape_signature": normalized_profile.get("task_shape_signature", ""),
            "benchmark_signature": benchmark_signature or "",
            "match_count": 0,
            "nearest_success": [],
            "nearest_failure": [],
            "best_success": None,
            "best_failure": None,
            "exact_match": None,
            "success_trace_bank": [],
        }

    ranked: list[dict[str, Any]] = []
    exact_match: dict[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        candidate_profile = _as_dict(entry.get("shape_profile", {}))
        if not candidate_profile or not normalized_profile:
            similarity = 0.0
        else:
            try:
                similarity = _task_shape_similarity(
                    normalized_profile, candidate_profile
                )
            except Exception:
                similarity = 0.0
        if similarity < min_similarity:
            if (
                benchmark_signature
                and entry.get("benchmark_signature") == benchmark_signature
            ):
                exact_match = _decorate_record(entry, similarity)
            continue

        decorated = _decorate_record(entry, similarity)
        ranked.append(decorated)
        if (
            benchmark_signature
            and entry.get("benchmark_signature") == benchmark_signature
        ):
            exact_match = decorated

    nearest_success = sorted(
        [record for record in ranked if record["success"]],
        key=lambda record: (-record["similarity"], -record["outcome_score"]),
    )
    nearest_failure = sorted(
        [record for record in ranked if not record["success"]],
        key=lambda record: (-record["similarity"], record["outcome_score"]),
    )
    nearest = sorted(
        ranked,
        key=lambda record: (-record["similarity"], -record["outcome_score"]),
    )[:limit]

    success_trace_bank: dict[tuple[str, ...], dict[str, Any]] = {}

    def _merge_trace_bank(records: list[dict[str, Any]]) -> None:
        for record in records:
            for trace_item in record.get("success_trace_bank", []):
                trace = trace_item.get("trace", [])
                if not isinstance(trace, list):
                    continue
                key = tuple(trace)
                existing = success_trace_bank.get(key)
                if existing is None:
                    success_trace_bank[key] = dict(trace_item)
                    continue

                existing["weight"] = round(
                    float(existing.get("weight", 0.0) or 0.0)
                    + float(trace_item.get("weight", 0.0) or 0.0),
                    3,
                )
                existing["similarity"] = round(
                    max(
                        float(existing.get("similarity", 0.0) or 0.0),
                        float(trace_item.get("similarity", 0.0) or 0.0),
                    ),
                    3,
                )
                existing["outcome_score"] = round(
                    max(
                        float(existing.get("outcome_score", 0.0) or 0.0),
                        float(trace_item.get("outcome_score", 0.0) or 0.0),
                    ),
                    3,
                )

    def _is_novel_trace(trace_item: Mapping[str, Any]) -> bool:
        trace = trace_item.get("trace", [])
        if not isinstance(trace, list) or not trace:
            return False
        key = tuple(str(token) for token in trace if str(token))
        if not key:
            return False
        return key not in success_trace_bank

    _merge_trace_bank(nearest_success)
    if exact_match and exact_match.get("success"):
        _merge_trace_bank([exact_match])

    success_trace_bank = {
        key: value
        for key, value in success_trace_bank.items()
        if _is_novel_trace(value) or float(value.get("weight", 0.0) or 0.0) >= 1.25
    }

    return {
        "available": True,
        "task_shape_signature": normalized_profile.get("task_shape_signature", ""),
        "benchmark_signature": benchmark_signature or "",
        "match_count": len(ranked),
        "nearest": nearest,
        "nearest_success": nearest_success[:limit],
        "nearest_failure": nearest_failure[:limit],
        "best_success": nearest_success[0] if nearest_success else None,
        "best_failure": nearest_failure[0] if nearest_failure else None,
        "exact_match": exact_match,
        "success_trace_bank": sorted(
            success_trace_bank.values(),
            key=lambda item: (
                -float(item.get("weight", 0.0) or 0.0),
                str(item.get("source", "")),
                tuple(item.get("trace", [])),
            ),
        )[: max(0, limit * 4)],
    }
