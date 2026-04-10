from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._version import API_VERSION, with_api_version

__all__ = ["API_VERSION", "benchmark_comparison_report", "analyze_scaling_profile", "main"]


def _load_payload(
    payload_or_path: dict[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if payload_or_path is None:
        return {}
    if isinstance(payload_or_path, dict):
        return dict(payload_or_path)
    if isinstance(payload_or_path, Path):
        try:
            payload = json.loads(payload_or_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
    text = str(payload_or_path).strip()
    if not text:
        return {}
    candidate = Path(text)
    try:
        if candidate.exists():
            payload = json.loads(candidate.read_text())
        else:
            payload = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    normalized = {key: _round(value / total) for key, value in weights.items()}
    residual = _round(1.0 - sum(normalized.values()))
    if residual:
        dominant_key = max(normalized.items(), key=lambda item: item[1])[0]
        normalized[dominant_key] = _round(normalized[dominant_key] + residual)
    return normalized


def _regime_from_temperature(temperature: float) -> str:
    if temperature < 0.52:
        return "exploit"
    if temperature < 0.78:
        return "balanced"
    return "explore"


def _transition_density(controller_summary: dict[str, Any]) -> float:
    regime_counts = controller_summary.get("regime_counts", {}) or {}
    total = sum(int(value) for value in regime_counts.values()) or 1
    return _clamp(_value(controller_summary, "transition_count") / total)


def _phase_pressure(snapshot: dict[str, float], transition_density: float) -> float:
    from .feed_derived_dna.controller import phase_signal

    base = phase_signal(snapshot)
    return _clamp(base + transition_density * 0.08)


def _build_snapshot(result: dict[str, Any]) -> dict[str, float]:
    summary = result.get("summary", {}) or {}
    controller_summary = result.get("controller_summary", {}) or {}
    phase_summary = result.get("phase_summary", {}) or {}
    best_governed = result.get("best_governed", {}) or {}
    best_raw = result.get("best_raw", {}) or {}

    snapshot = {
        "mean": _round(_value(summary, "mean")),
        "governed_mean": _round(_value(summary, "governed_mean")),
        "selection_score_mean": _round(_value(summary, "selection_score_mean")),
        "task_threshold_pass_mean": _round(_value(summary, "task_threshold_pass_mean")),
        "solve_rate_test_mean": _round(_value(summary, "solve_rate_test_mean")),
        "stability_score_mean": _round(_value(summary, "stability_score_mean")),
        "cognitive_stability_mean": _round(_value(summary, "cognitive_stability_mean")),
        "economic_stability_mean": _round(_value(summary, "economic_stability_mean")),
        "profit_mean": _round(_value(summary, "profit_mean")),
        "market_score_mean": _round(_value(summary, "market_score_mean")),
        "governance_variance": _round(_value(summary, "governance_variance")),
        "motif_entropy": _round(_value(summary, "motif_entropy")),
        "specialization_entropy": _round(_value(summary, "specialization_entropy")),
        "lineage_entropy": _round(_value(summary, "lineage_entropy")),
        "optimizer_plan_axis_mean": _round(_value(summary, "optimizer_plan_axis_mean")),
        "optimizer_patch_axis_mean": _round(
            _value(summary, "optimizer_patch_axis_mean")
        ),
        "optimizer_summary_axis_mean": _round(
            _value(summary, "optimizer_summary_axis_mean")
        ),
        "optimizer_verify_axis_mean": _round(
            _value(summary, "optimizer_verify_axis_mean")
        ),
        "optimizer_balance_mean": _round(_value(summary, "optimizer_balance_mean")),
        "optimizer_mixed_pressure_mean": _round(
            _value(summary, "optimizer_mixed_pressure_mean")
        ),
        "optimizer_ast_depth_mean": _round(_value(summary, "optimizer_ast_depth_mean")),
        "rotation_market_signal": _round(_value(summary, "rotation_market_signal")),
        "parallax_replication_bursts": int(
            _value(result, "parallax_replication_bursts")
        ),
        "parallax_replication_offspring": int(
            _value(result, "parallax_replication_offspring")
        ),
        "transition_count": int(_value(controller_summary, "transition_count")),
        "loopback_count": int(_value(controller_summary, "loopback_count")),
        "exploit_entries": int(_value(controller_summary, "exploit_entries")),
        "exploit_bounce_count": int(_value(controller_summary, "exploit_bounce_count")),
        "episodes_min": int(_value(controller_summary, "episodes_min")),
        "episodes_max": int(_value(controller_summary, "episodes_max")),
        "temperature_min": _round(_value(controller_summary, "temperature_min")),
        "temperature_max": _round(_value(controller_summary, "temperature_max")),
        "phase_transition_count": int(_value(phase_summary, "transition_count")),
        "best_governed_fitness": _round(_value(best_governed, "governed_fitness")),
        "best_governed_selection_score": _round(
            _value(best_governed, "selection_score")
        ),
        "best_governed_raw_fitness": _round(_value(best_governed, "fitness")),
        "best_governed_stability": _round(_value(best_governed, "stability_score")),
        "best_governed_cognitive_stability": _round(
            _value(best_governed, "cognitive_stability")
        ),
        "best_governed_economic_stability": _round(
            _value(best_governed, "economic_stability")
        ),
        "best_governed_profit": _round(_value(best_governed, "profit")),
        "best_governed_task_threshold_pass": _round(
            _value(best_governed, "task_threshold_pass")
        ),
        "best_governed_solve_rate_test": _round(
            _value(best_governed, "solve_rate_test")
        ),
        "best_raw_fitness": _round(_value(best_raw, "governed_fitness")),
        "best_raw_selection_score": _round(_value(best_raw, "selection_score")),
        "best_raw_raw_fitness": _round(_value(best_raw, "fitness")),
        "best_raw_stability": _round(_value(best_raw, "stability_score")),
        "best_raw_cognitive_stability": _round(_value(best_raw, "cognitive_stability")),
        "best_raw_economic_stability": _round(_value(best_raw, "economic_stability")),
        "best_raw_profit": _round(_value(best_raw, "profit")),
        "best_raw_task_threshold_pass": _round(_value(best_raw, "task_threshold_pass")),
        "best_raw_solve_rate_test": _round(_value(best_raw, "solve_rate_test")),
    }
    return snapshot


def _dual_objective_pressure(snapshot: dict[str, float]) -> dict[str, Any]:
    fitness_reference = max(
        1.0,
        abs(snapshot["best_governed_raw_fitness"]),
        abs(snapshot["best_raw_raw_fitness"]),
    )
    governed_reference = max(
        1.0, abs(snapshot["best_governed_fitness"]), abs(snapshot["best_raw_fitness"])
    )
    selection_reference = max(
        1.0,
        abs(snapshot["best_governed_selection_score"]),
        abs(snapshot["best_raw_selection_score"]),
    )
    pressure_index = _clamp(
        0.34
        * abs(snapshot["best_governed_raw_fitness"] - snapshot["best_raw_raw_fitness"])
        / fitness_reference
        + 0.28
        * abs(snapshot["best_governed_fitness"] - snapshot["best_raw_fitness"])
        / governed_reference
        + 0.14
        * abs(snapshot["best_governed_stability"] - snapshot["best_raw_stability"])
        + 0.12
        * abs(
            snapshot["best_governed_economic_stability"]
            - snapshot["best_raw_economic_stability"]
        )
        + 0.06
        * abs(
            snapshot["best_governed_task_threshold_pass"]
            - snapshot["best_raw_task_threshold_pass"]
        )
        + 0.06
        * abs(
            snapshot["best_governed_solve_rate_test"]
            - snapshot["best_raw_solve_rate_test"]
        )
        + 0.08
        * abs(
            snapshot["best_governed_selection_score"]
            - snapshot["best_raw_selection_score"]
        )
        / selection_reference
    )

    return {
        "fitness_gap": _round(
            snapshot["best_governed_raw_fitness"] - snapshot["best_raw_raw_fitness"]
        ),
        "governed_gap": _round(
            snapshot["best_governed_fitness"] - snapshot["best_raw_fitness"]
        ),
        "stability_gap": _round(
            snapshot["best_governed_stability"] - snapshot["best_raw_stability"]
        ),
        "cognitive_gap": _round(
            snapshot["best_governed_cognitive_stability"]
            - snapshot["best_raw_cognitive_stability"]
        ),
        "economic_gap": _round(
            snapshot["best_governed_economic_stability"]
            - snapshot["best_raw_economic_stability"]
        ),
        "threshold_gap": _round(
            snapshot["best_governed_task_threshold_pass"]
            - snapshot["best_raw_task_threshold_pass"]
        ),
        "solve_rate_gap": _round(
            snapshot["best_governed_solve_rate_test"]
            - snapshot["best_raw_solve_rate_test"]
        ),
        "selection_gap": _round(
            snapshot["best_governed_selection_score"]
            - snapshot["best_raw_selection_score"]
        ),
        "pressure_index": _round(pressure_index),
        "governed_is_ahead": snapshot["best_governed_fitness"]
        >= snapshot["best_raw_fitness"],
        "selection_is_ahead": snapshot["best_governed_selection_score"]
        >= snapshot["best_raw_selection_score"],
        "recommendation": (
            "keep dual-objective weighting narrow"
            if pressure_index < 0.25
            else "tune governed/raw weighting and verify reward stability"
        ),
    }


def _phase_detection(
    snapshot: dict[str, float], result: dict[str, Any]
) -> dict[str, Any]:
    controller_summary = result.get("controller_summary", {}) or {}
    transition_density = _transition_density(controller_summary)
    verify_density = snapshot["optimizer_verify_axis_mean"]
    plan_chain_stability = _clamp(
        0.52 * snapshot["optimizer_plan_axis_mean"]
        + 0.28 * snapshot["optimizer_balance_mean"]
        + 0.20 * (1.0 - snapshot["optimizer_mixed_pressure_mean"])
    )
    pressure = _phase_pressure(snapshot, transition_density)
    recommended_regime = _regime_from_temperature(pressure)

    regime_counts = controller_summary.get("regime_counts", {}) or {}
    dominant_regime = None
    if regime_counts:
        dominant_regime = max(
            regime_counts.items(), key=lambda item: (item[1], item[0])
        )[0]

    return {
        "dominant_regime": dominant_regime,
        "transition_density": _round(transition_density),
        "loopback_rate": _round(
            snapshot["loopback_count"] / max(1, snapshot["transition_count"])
        ),
        "exploit_share": _round(
            snapshot["exploit_entries"] / max(1, snapshot["transition_count"])
        ),
        "verify_density": _round(verify_density),
        "plan_chain_stability": _round(plan_chain_stability),
        "phase_pressure": _round(pressure),
        "recommended_temperature": _round(pressure),
        "recommended_regime": recommended_regime,
        "transition_risk": _round(
            _clamp(
                (1.0 - verify_density) * 0.35
                + (1.0 - plan_chain_stability) * 0.35
                + transition_density * 0.30
            )
        ),
        "recommendation": (
            f"bias the controller toward {recommended_regime}"
            if recommended_regime != dominant_regime
            else "current regime choice is consistent with the observed pressure"
        ),
    }


def _reward_model_candidates(
    snapshot: dict[str, float],
    dual_pressure: dict[str, Any],
    phase_detection: dict[str, Any],
) -> dict[str, Any]:
    raw_weights = {
        "quality": 0.42
        + 0.20 * (1.0 - snapshot["task_threshold_pass_mean"])
        + 0.10 * (1.0 - snapshot["solve_rate_test_mean"])
        + 0.08 * dual_pressure["pressure_index"],
        "economy": 0.30
        + 0.20 * (1.0 - snapshot["economic_stability_mean"])
        + 0.10 * (1.0 if snapshot["profit_mean"] < 0 else 0.0)
        + 0.05 * dual_pressure["pressure_index"],
        "verification": 0.18
        + 0.18 * (1.0 - snapshot["optimizer_verify_axis_mean"])
        + 0.10 * phase_detection["transition_density"],
        "diversity": 0.10
        + 0.20
        * (
            1.0
            - min(
                snapshot["specialization_entropy"],
                snapshot["motif_entropy"],
                snapshot["lineage_entropy"],
            )
        ),
    }
    candidate_weights = _normalize_weights(raw_weights)
    ranked = sorted(
        candidate_weights.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    top_label, top_weight = ranked[0]
    return {
        "candidate_weights": candidate_weights,
        "top_candidate": top_label,
        "top_weight": top_weight,
        "recommendation": (
            "treat the weights as telemetry-only until a rebaseline validates them"
            if dual_pressure["pressure_index"] >= 0.25
            else "current reward balance looks stable enough for a candidate fit"
        ),
    }


def _diversity_enforcement(
    result: dict[str, Any], snapshot: dict[str, float]
) -> dict[str, Any]:
    winner_specializations = result.get("winner_specialization_counts", {}) or {}
    winner_families = result.get("winner_family_counts", {}) or {}
    winner_motifs = result.get("winner_motif_counts", {}) or {}

    floors = {
        "specialization_entropy_floor": 0.55,
        "motif_entropy_floor": 0.60,
        "lineage_entropy_floor": 0.55,
    }
    gaps = {
        "specialization_entropy_gap": _round(
            snapshot["specialization_entropy"] - floors["specialization_entropy_floor"]
        ),
        "motif_entropy_gap": _round(
            snapshot["motif_entropy"] - floors["motif_entropy_floor"]
        ),
        "lineage_entropy_gap": _round(
            snapshot["lineage_entropy"] - floors["lineage_entropy_floor"]
        ),
    }
    collapse_flags = [label for label, gap in gaps.items() if gap < 0]
    dominant_winner_share = 0.0
    if winner_specializations:
        total = sum(winner_specializations.values()) or 1
        dominant_winner_share = max(winner_specializations.values()) / total

    collapse_pressure = _clamp(
        (1.0 - snapshot["specialization_entropy"]) * 0.40
        + (1.0 - snapshot["motif_entropy"]) * 0.35
        + (1.0 - snapshot["lineage_entropy"]) * 0.25
        + max(0.0, dominant_winner_share - 0.55) * 0.40
    )
    if collapse_flags:
        status = "enforce"
    elif collapse_pressure >= 0.45:
        status = "watch"
    else:
        status = "healthy"

    recommendations = []
    if "specialization_entropy_gap" in collapse_flags:
        recommendations.append(
            "widen niche pressure and increase cross-specialization pairing"
        )
    if "motif_entropy_gap" in collapse_flags:
        recommendations.append(
            "raise motif diversity floors before reinforcing learned macros"
        )
    if "lineage_entropy_gap" in collapse_flags:
        recommendations.append(
            "increase lineage mutation spread or reduce elite carryover"
        )
    if not recommendations:
        recommendations.append(
            "diversity floors are currently within the safe envelope"
        )

    return {
        "status": status,
        "floors": floors,
        "gaps": gaps,
        "dominant_winner_share": _round(dominant_winner_share),
        "collapse_pressure": _round(collapse_pressure),
        "winner_specialization_counts": winner_specializations,
        "winner_family_counts": winner_families,
        "winner_motif_counts": winner_motifs,
        "recommendations": recommendations,
    }


def _baseline_comparison(
    snapshot: dict[str, float], baseline: dict[str, Any] | None
) -> dict[str, Any]:
    if not baseline:
        return {"present": False, "matched": None, "deltas": {}}
    baseline_snapshot = _build_snapshot(baseline)
    keys = [
        "governed_mean",
        "selection_score_mean",
        "task_threshold_pass_mean",
        "solve_rate_test_mean",
        "stability_score_mean",
        "economic_stability_mean",
        "profit_mean",
        "motif_entropy",
        "specialization_entropy",
        "lineage_entropy",
        "optimizer_plan_axis_mean",
        "optimizer_verify_axis_mean",
        "transition_count",
    ]
    deltas = {key: _round(snapshot[key] - baseline_snapshot[key]) for key in keys}
    matched = all(abs(value) < 1e-9 for value in deltas.values())
    return {"present": True, "matched": matched, "deltas": deltas}


def _comparison_score(snapshot: dict[str, float]) -> float:
    return round(
        0.30 * snapshot["governed_mean"]
        + 0.18 * snapshot["selection_score_mean"]
        + 48.0 * snapshot["solve_rate_test_mean"]
        + 32.0 * snapshot["stability_score_mean"]
        + 24.0 * snapshot["task_threshold_pass_mean"]
        + 10.0 * snapshot["economic_stability_mean"]
        + 8.0 * snapshot["market_score_mean"]
        + 6.0 * snapshot["rotation_market_signal"]
        - 0.01 * snapshot["governance_variance"],
        3,
    )


def _nested_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def benchmark_comparison_report(
    baseline_or_path: dict[str, Any] | str | Path,
    candidate_or_path: dict[str, Any] | str | Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    baseline = _load_payload(baseline_or_path)
    candidate = _load_payload(candidate_or_path)

    baseline_diagnostics = analyze_scaling_profile(baseline)
    candidate_diagnostics = analyze_scaling_profile(candidate, baseline)

    baseline_snapshot = baseline_diagnostics.get("snapshot", {})
    candidate_snapshot = candidate_diagnostics.get("snapshot", {})
    delta_keys = [
        "governed_mean",
        "selection_score_mean",
        "solve_rate_test_mean",
        "stability_score_mean",
        "task_threshold_pass_mean",
        "economic_stability_mean",
        "profit_mean",
        "market_score_mean",
        "rotation_market_signal",
        "governance_variance",
        "motif_entropy",
        "specialization_entropy",
        "lineage_entropy",
        "transition_count",
        "loopback_count",
        "exploit_entries",
        "exploit_bounce_count",
    ]
    delta = {
        key: _round(candidate_snapshot.get(key, 0.0) - baseline_snapshot.get(key, 0.0))
        for key in delta_keys
    }

    baseline_score = _comparison_score(baseline_snapshot)
    candidate_score = _comparison_score(candidate_snapshot)
    score_delta = _round(candidate_score - baseline_score)

    baseline_selection = _nested_mapping(baseline, "selection_report")
    candidate_selection = _nested_mapping(candidate, "selection_report")
    selection_delta = {
        "selection_score_mean": _round(
            float(candidate_snapshot.get("selection_score_mean", 0.0) or 0.0)
            - float(baseline_snapshot.get("selection_score_mean", 0.0) or 0.0)
        ),
        "pareto_front_count": int(candidate_selection.get("pareto_front_count", 0) or 0)
        - int(baseline_selection.get("pareto_front_count", 0) or 0),
        "rotation_signal": _round(
            float(candidate_selection.get("rotation_signal", 0.0) or 0.0)
            - float(baseline_selection.get("rotation_signal", 0.0) or 0.0)
        ),
        "rotation_strength": _round(
            float(candidate_selection.get("rotation_strength", 0.0) or 0.0)
            - float(baseline_selection.get("rotation_strength", 0.0) or 0.0)
        ),
        "probe_count": int(candidate_selection.get("probe_count", 0) or 0)
        - int(baseline_selection.get("probe_count", 0) or 0),
        "forbidden_subsequence_count": int(
            candidate_selection.get("forbidden_subsequence_count", 0) or 0
        )
        - int(baseline_selection.get("forbidden_subsequence_count", 0) or 0),
    }

    baseline_rotation = _nested_mapping(baseline, "rotation_analysis")
    candidate_rotation = _nested_mapping(candidate, "rotation_analysis")
    rotation_delta = {
        "condition_number": _round(
            float(candidate_rotation.get("condition_number", 0.0) or 0.0)
            - float(baseline_rotation.get("condition_number", 0.0) or 0.0)
        ),
        "principal_eigenvalue": _round(
            float(candidate_rotation.get("principal_eigenvalue", 0.0) or 0.0)
            - float(baseline_rotation.get("principal_eigenvalue", 0.0) or 0.0)
        ),
        "secondary_eigenvalue": _round(
            float(candidate_rotation.get("secondary_eigenvalue", 0.0) or 0.0)
            - float(baseline_rotation.get("secondary_eigenvalue", 0.0) or 0.0)
        ),
    }

    baseline_parallax = _nested_mapping(baseline, "parallax_telemetry")
    candidate_parallax = _nested_mapping(candidate, "parallax_telemetry")
    parallax_delta = {
        "pressure": _round(
            float(candidate_parallax.get("pressure", 0.0) or 0.0)
            - float(baseline_parallax.get("pressure", 0.0) or 0.0)
        ),
        "valley_depth": _round(
            float(candidate_parallax.get("valley_depth", 0.0) or 0.0)
            - float(baseline_parallax.get("valley_depth", 0.0) or 0.0)
        ),
        "gradient_steepness": _round(
            float(candidate_parallax.get("gradient_steepness", 0.0) or 0.0)
            - float(baseline_parallax.get("gradient_steepness", 0.0) or 0.0)
        ),
        "failure_probe_count": int(
            candidate_parallax.get("failure_probe_count", 0) or 0
        )
        - int(baseline_parallax.get("failure_probe_count", 0) or 0),
        "forbidden_subsequence_count": int(
            candidate_parallax.get("forbidden_subsequence_count", 0) or 0
        )
        - int(baseline_parallax.get("forbidden_subsequence_count", 0) or 0),
    }

    risk_flags: list[str] = []
    if candidate_diagnostics["dual_objective_pressure"]["pressure_index"] >= 0.35:
        risk_flags.append("candidate dual-objective pressure is elevated")
    if candidate_diagnostics["phase_detection"]["transition_risk"] >= 0.45:
        risk_flags.append("candidate phase transitions are unstable")
    if candidate_diagnostics["diversity_enforcement"]["status"] == "enforce":
        risk_flags.append("candidate diversity floors are breached")
    if (
        candidate_snapshot["governance_variance"]
        > baseline_snapshot["governance_variance"] * 1.2
    ):
        risk_flags.append("candidate governance variance is higher")
    if (
        candidate_snapshot["solve_rate_test_mean"]
        < baseline_snapshot["solve_rate_test_mean"]
    ):
        risk_flags.append("candidate solve rate regresses")

    if score_delta > 0.0 and not risk_flags:
        recommendation = f"prefer {candidate_label}; the candidate improves the benchmark without adding obvious risk"
    elif score_delta > 0.0:
        recommendation = (
            f"prefer {candidate_label} only after reviewing the flagged regressions"
        )
    elif score_delta < 0.0:
        recommendation = f"keep {baseline_label}; the baseline still scores better on the comparison set"
    else:
        recommendation = "scores are effectively tied; use the smaller risk surface or the cleaner state"

    winner = candidate_label if score_delta >= 0.0 else baseline_label
    return with_api_version(
        {
            "status": "compared",
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "baseline": baseline_diagnostics,
            "candidate": candidate_diagnostics,
            "delta": delta,
            "selection_delta": selection_delta,
            "rotation_delta": rotation_delta,
            "parallax_delta": parallax_delta,
            "score": {
                "baseline": baseline_score,
                "candidate": candidate_score,
                "delta": score_delta,
                "winner": winner,
            },
            "winner": winner,
            "recommendation": recommendation,
            "risk_flags": risk_flags,
        }
    )


def analyze_scaling_profile(
    result_or_path: dict[str, Any] | str | Path,
    baseline_or_path: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    result = _load_payload(result_or_path)
    baseline = _load_payload(baseline_or_path) if baseline_or_path is not None else None
    snapshot = _build_snapshot(result)
    dual_pressure = _dual_objective_pressure(snapshot)
    phase_detection = _phase_detection(snapshot, result)
    reward_candidates = _reward_model_candidates(
        snapshot, dual_pressure, phase_detection
    )
    diversity_enforcement = _diversity_enforcement(result, snapshot)
    baseline_comparison = _baseline_comparison(snapshot, baseline)

    risk_flags = []
    if dual_pressure["pressure_index"] >= 0.35:
        risk_flags.append("dual-objective pressure is elevated")
    if phase_detection["transition_risk"] >= 0.45:
        risk_flags.append("phase transitions are unstable")
    if diversity_enforcement["status"] == "enforce":
        risk_flags.append("diversity floors are breached")
    if snapshot["governance_variance"] > 1200:
        risk_flags.append("governance variance is high")
    if snapshot["optimizer_mixed_pressure_mean"] > 0.30:
        risk_flags.append("mixed-pressure traces are dominant")

    recommendations = []
    recommendations.append(dual_pressure["recommendation"])
    recommendations.append(phase_detection["recommendation"])
    recommendations.extend(diversity_enforcement["recommendations"])
    recommendations.append(reward_candidates["recommendation"])
    if baseline_comparison["present"] and baseline_comparison["matched"]:
        recommendations.append("baseline comparison matches exactly")
    elif baseline_comparison["present"]:
        recommendations.append("baseline comparison diverges and should be reviewed")

    return with_api_version(
        {
            "status": "diagnosed",
            "snapshot": snapshot,
            "dual_objective_pressure": dual_pressure,
            "phase_detection": phase_detection,
            "reward_model_candidates": reward_candidates,
            "diversity_enforcement": diversity_enforcement,
            "baseline_comparison": baseline_comparison,
            "risk_flags": risk_flags,
            "recommendations": recommendations,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Barricade scaling diagnostics")
    parser.add_argument(
        "--result-file", default="", help="Path to a benchmark result JSON file"
    )
    parser.add_argument(
        "--result-json", default="", help="Inline benchmark result JSON"
    )
    parser.add_argument(
        "--baseline-file",
        default="",
        help="Optional path to a baseline result JSON file",
    )
    parser.add_argument(
        "--baseline-json", default="", help="Optional inline baseline JSON"
    )
    parser.add_argument(
        "--out", default="", help="Optional output file for the diagnostics JSON"
    )
    args = parser.parse_args()

    if args.result_json:
        result = json.loads(args.result_json)
    elif args.result_file:
        result = json.loads(Path(args.result_file).read_text())
    else:
        parser.error("--result-file or --result-json is required")

    baseline = None
    if args.baseline_json:
        baseline = json.loads(args.baseline_json)
    elif args.baseline_file:
        baseline = json.loads(Path(args.baseline_file).read_text())

    diagnostics = analyze_scaling_profile(result, baseline)
    payload = json.dumps(diagnostics, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
