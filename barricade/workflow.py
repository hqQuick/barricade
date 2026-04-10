from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._version import API_VERSION
from .dispatch import barricade_dispatch
from .feed_derived_dna.persistence import (
    load_macro_library,
    load_execution_feedback,
    resolve_state_root,
    save_task_shape_prior,
    task_shape_signature,
)
from .feed_derived_dna._outcome_memory import load_outcome_memory
from .runtime import (
    benchmark_contract,
    build_patch_skeleton,
    derive_feed_dna_prior,
    derive_task_ecology,
    run_benchmark,
)
from .workflow_intake import (
    _build_feed,
    build_workflow_intake,
    load_best_task_shape_prior,
    materialize_task_shape_hit,
    prior_strength_profile,
    task_shape_cache_score,
    task_shape_profile,
)


def derive_act_ask_stop_policy(
    intake: dict[str, Any],
    prior_profile: dict[str, Any],
    outcome_memory: dict[str, Any] | None,
    execution_feedback: dict[str, Any] | None = None,
    benchmark_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase_scores = intake.get("phase_scores", {}) if isinstance(intake, dict) else {}
    ranked_phases = sorted(
        (
            (phase, float(score or 0.0))
            for phase, score in phase_scores.items()
            if phase != "mixed"
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    route_hint = str(intake.get("route_hint", "mixed") or "mixed")
    confidence = float(intake.get("confidence", 0.0) or 0.0)
    risk_count = (
        len(intake.get("risks", [])) if isinstance(intake.get("risks", []), list) else 0
    )
    constraint_count = (
        len(intake.get("constraints", []))
        if isinstance(intake.get("constraints", []), list)
        else 0
    )
    deliverable_count = (
        len(intake.get("deliverables", []))
        if isinstance(intake.get("deliverables", []), list)
        else 0
    )
    top_phase_score = ranked_phases[0][1] if ranked_phases else 0.0
    runner_up_score = ranked_phases[1][1] if len(ranked_phases) > 1 else 0.0
    route_margin = max(0.0, top_phase_score - runner_up_score)
    route_certainty = min(
        1.0, route_margin * 4.0 + (0.08 if route_hint != "mixed" else 0.0)
    )
    prior_strength = float(prior_profile.get("effective_strength", 0.0) or 0.0)
    outcome_memory = outcome_memory or {}
    execution_feedback = execution_feedback or {}
    outcome_match_count = int(outcome_memory.get("match_count", 0) or 0)
    best_success = (
        outcome_memory.get("best_success") if isinstance(outcome_memory, dict) else None
    )
    best_failure = (
        outcome_memory.get("best_failure") if isinstance(outcome_memory, dict) else None
    )
    feedback_match_count = int(execution_feedback.get("match_count", 0) or 0)
    best_feedback = execution_feedback.get("best_success") or execution_feedback.get(
        "latest"
    )
    best_success_signal = 0.0
    if isinstance(best_success, dict):
        best_success_signal = min(
            1.0,
            0.65 * float(best_success.get("similarity", 0.0) or 0.0)
            + 0.35
            * min(1.0, float(best_success.get("outcome_score", 0.0) or 0.0) / 200.0),
        )
    best_failure_signal = 0.0
    if isinstance(best_failure, dict):
        best_failure_signal = min(
            1.0,
            0.55 * float(best_failure.get("similarity", 0.0) or 0.0)
            + 0.45
            * min(1.0, float(best_failure.get("outcome_score", 0.0) or 0.0) / 200.0),
        )
    feedback_signal = 0.0
    if isinstance(best_feedback, dict):
        feedback_signal = min(
            1.0,
            0.50 * float(best_feedback.get("verification_pass_rate", 0.0) or 0.0)
            + 0.25
            * min(
                1.0, float(best_feedback.get("learned_macro_count", 0.0) or 0.0) / 4.0
            )
            + 0.25
            * min(
                1.0,
                float(best_feedback.get("successful_trace_count", 0.0) or 0.0) / 3.0,
            ),
        )
    governed_mean = 0.0
    if isinstance(benchmark_result, dict):
        governed_mean = float(
            (
                benchmark_result.get("summary", {})
                if isinstance(benchmark_result.get("summary", {}), dict)
                else {}
            ).get("governed_mean", 0.0)
            or 0.0
        )
    benchmark_signal = min(1.0, governed_mean / 180.0)
    support_score = round(
        0.35 * confidence
        + 0.22 * prior_strength
        + 0.18 * best_success_signal
        + 0.07 * benchmark_signal
        + 0.06 * min(1.0, outcome_match_count / 3.0)
        + 0.08 * feedback_signal
        + 0.03 * min(1.0, feedback_match_count / 3.0)
        + 0.05 * route_certainty
        + 0.03 * min(1.0, deliverable_count / 4.0)
        + 0.02 * min(1.0, constraint_count / 4.0)
        - 0.20
        * (
            0.08 * risk_count
            + (0.18 if route_hint == "mixed" else 0.0)
            + (0.04 if confidence < 0.5 else 0.0)
        )
        - 0.05 * best_failure_signal,
        3,
    )

    if (
        confidence >= 0.52
        and support_score >= 0.50
        and route_hint != "mixed"
        and risk_count <= 2
    ):
        mode = "act"
    elif support_score <= 0.40 or confidence <= 0.42 or risk_count >= 4:
        mode = "stop"
    else:
        mode = "ask"

    reasons = [
        f"confidence={confidence:.3f}",
        f"prior_strength={prior_strength:.3f}",
        f"route_hint={route_hint}",
        f"route_margin={route_margin:.3f}",
        f"outcome_match_count={outcome_match_count}",
        f"best_success_signal={best_success_signal:.3f}",
        f"best_failure_signal={best_failure_signal:.3f}",
        f"feedback_match_count={feedback_match_count}",
        f"feedback_signal={feedback_signal:.3f}",
        f"risk_count={risk_count}",
        f"support_score={support_score:.3f}",
    ]
    clarifying_questions: list[str] = []
    if mode == "ask":
        if route_hint == "mixed":
            clarifying_questions.append(
                "Which lane should I optimize for: planning, patching, summarizing, or recovery?"
            )
        if not outcome_memory.get("available", False):
            clarifying_questions.append(
                "Should I proceed from fresh state or reuse an existing state_dir?"
            )
        if constraint_count > 2:
            clarifying_questions.append(
                "Which constraint should take priority if they conflict?"
            )
        if not clarifying_questions:
            clarifying_questions.append(
                "What outcome matters most so I can narrow the execution plan?"
            )

    next_instruction = {
        "act": "Call begin_execution(synthesis_result_json) to expand the DNA into a literal execution session.",
        "ask": "Clarify the open ambiguity before beginning execution.",
        "stop": "Stop: the current signal is too weak or risky to proceed safely.",
    }[mode]

    policy: dict[str, Any] = {
        "mode": mode,
        "actionable": mode == "act",
        "confidence": round(confidence, 3),
        "prior_strength": round(prior_strength, 3),
        "route_hint": route_hint,
        "route_margin": round(route_margin, 3),
        "outcome_match_count": outcome_match_count,
        "feedback_match_count": feedback_match_count,
        "outcome_signal": round(best_success_signal, 3),
        "feedback_signal": round(feedback_signal, 3),
        "support_score": support_score,
        "risk_count": risk_count,
        "reasons": reasons,
        "next_instruction": next_instruction,
    }
    if mode == "ask":
        policy["clarifying_questions"] = clarifying_questions
    if mode == "stop":
        policy["stop_reason"] = next_instruction
    return policy


def _resolve_task_shape_signature(shape_profile: dict[str, Any]) -> str:
    signature = str(shape_profile.get("task_shape_signature", "") or "")
    if signature:
        return signature
    return task_shape_signature(shape_profile)


def _summarize_execution_feedback(entry: dict[str, Any]) -> dict[str, Any]:
    completion_summary = (
        entry.get("completion_summary", {})
        if isinstance(entry.get("completion_summary"), dict)
        else {}
    )
    decision_policy = (
        entry.get("decision_policy", {})
        if isinstance(entry.get("decision_policy"), dict)
        else {}
    )
    return {
        "session_id": str(entry.get("session_id", "") or ""),
        "task_text": str(entry.get("task_text", "") or ""),
        "task_shape_signature": str(entry.get("task_shape_signature", "") or ""),
        "route_hint": str(entry.get("route_hint", "") or ""),
        "decision_mode": str(decision_policy.get("mode", "act") or "act"),
        "verification_passed": bool(
            completion_summary.get("verification_passed", False)
        ),
        "verification_pass_rate": round(
            float(completion_summary.get("verification_pass_rate", 0.0) or 0.0),
            3,
        ),
        "successful_trace_count": int(
            completion_summary.get("successful_trace_count", 0) or 0
        ),
        "learned_macro_count": int(
            completion_summary.get("learned_macro_count", 0) or 0
        ),
        "artifact_count": int(completion_summary.get("artifact_count", 0) or 0),
        "market_total_score": round(
            float(completion_summary.get("market_total_score", 0.0) or 0.0),
            3,
        ),
        "market_total_price": round(
            float(completion_summary.get("market_total_price", 0.0) or 0.0),
            3,
        ),
        "support_score": round(
            float(decision_policy.get("support_score", 0.0) or 0.0),
            3,
        ),
    }


def load_execution_feedback_for_shape(
    state_root,
    shape_profile: dict[str, Any],
    benchmark_signature: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    entries = load_execution_feedback(state_root)
    target_signature = _resolve_task_shape_signature(shape_profile)
    if not entries:
        return {
            "available": False,
            "task_shape_signature": target_signature,
            "benchmark_signature": benchmark_signature or "",
            "match_count": 0,
            "recent": [],
            "recent_success": [],
            "recent_failure": [],
            "best_success": None,
            "best_failure": None,
            "latest": None,
            "exact_match": None,
        }

    ranked: list[dict[str, Any]] = []
    exact_match: dict[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_signature = str(entry.get("task_shape_signature", "") or "")
        if target_signature and entry_signature != target_signature:
            continue
        summary = _summarize_execution_feedback(entry)
        ranked.append(summary)
        if (
            benchmark_signature
            and str(entry.get("benchmark_signature", "") or "") == benchmark_signature
        ):
            exact_match = summary

    if not ranked:
        return {
            "available": False,
            "task_shape_signature": target_signature,
            "benchmark_signature": benchmark_signature or "",
            "match_count": 0,
            "recent": [],
            "recent_success": [],
            "recent_failure": [],
            "best_success": None,
            "best_failure": None,
            "latest": None,
            "exact_match": None,
        }

    ranked.sort(
        key=lambda record: (
            record["verification_passed"],
            record["verification_pass_rate"],
            record["learned_macro_count"],
            record["successful_trace_count"],
            record["artifact_count"],
        ),
        reverse=True,
    )
    recent = ranked[-limit:]
    recent_success = [record for record in ranked if record["verification_passed"]][
        :limit
    ]
    recent_failure = [record for record in ranked if not record["verification_passed"]][
        :limit
    ]

    return {
        "available": True,
        "task_shape_signature": target_signature,
        "benchmark_signature": benchmark_signature or "",
        "match_count": len(ranked),
        "recent": recent,
        "recent_success": recent_success,
        "recent_failure": recent_failure,
        "best_success": recent_success[0] if recent_success else None,
        "best_failure": recent_failure[0] if recent_failure else None,
        "latest": recent[-1] if recent else None,
        "exact_match": exact_match,
    }


def _build_prototype_summary(
    intake: dict[str, Any],
    prior_profile: dict[str, Any],
    cache_hit: dict[str, Any] | None,
    outcome_memory: dict[str, Any] | None,
    execution_feedback: dict[str, Any] | None,
    decision_policy: dict[str, Any] | None,
    selection_report: dict[str, Any] | None,
    rotation_analysis: dict[str, Any] | None,
    parallax_telemetry: dict[str, Any] | None,
    forbidden_subsequence_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    centroid = intake.get("shape_centroid", {}) if isinstance(intake, dict) else {}
    route_explanation = (
        list(intake.get("route_explanation", [])) if isinstance(intake, dict) else []
    )
    outcome_memory = outcome_memory or {}
    execution_feedback = execution_feedback or {}
    decision_policy = decision_policy or {}
    selection_report = selection_report or {}
    rotation_analysis = rotation_analysis or {}
    parallax_telemetry = parallax_telemetry or {}
    forbidden_subsequence_memory = forbidden_subsequence_memory or {}

    reuse_sources: list[str] = []
    reuse_notes: list[str] = []
    if cache_hit:
        reuse_sources.append("task_shape_prior")
        reuse_notes.append(
            f"task-shape prior similarity {float(cache_hit.get('similarity', 0.0) or 0.0):.3f}"
        )
        reuse_notes.append(
            f"task-shape prior score {float(cache_hit.get('score', 0.0) or 0.0):.3f}"
        )
    if execution_feedback.get("available"):
        reuse_sources.append("execution_feedback")
        reuse_notes.append(
            f"execution feedback matches {int(execution_feedback.get('match_count', 0) or 0)}"
        )
    if outcome_memory.get("available"):
        reuse_sources.append("outcome_memory")
        reuse_notes.append(
            f"outcome memory matches {int(outcome_memory.get('match_count', 0) or 0)}"
        )
    if not reuse_sources:
        reuse_sources.append("fresh_benchmark")
        reuse_notes.append("no reusable prior passed the minimum similarity threshold")

    if cache_hit:
        reuse_reason = " ; ".join(reuse_notes[:3])
    else:
        reuse_reason = reuse_notes[0]

    return {
        "prototype_stage": str(
            intake.get("prototype_stage", "candidate") or "candidate"
        ),
        "holdout_bucket": int(intake.get("holdout_bucket", 0) or 0),
        "route_hint": str(intake.get("route_hint", "") or ""),
        "decision_mode": str(decision_policy.get("mode", "act") or "act"),
        "support_score": round(
            float(decision_policy.get("support_score", 0.0) or 0.0), 3
        ),
        "dominant_lane": str(centroid.get("dominant_lane", "planning") or "planning"),
        "runner_up_lane": str(centroid.get("runner_up_lane", "planning") or "planning"),
        "centroid_margin": round(float(centroid.get("lane_margin", 0.0) or 0.0), 3),
        "centroid_axis": round(float(centroid.get("centroid_axis", 0.0) or 0.0), 3),
        "centroid_entropy": round(float(centroid.get("entropy", 0.0) or 0.0), 3),
        "prior_strength": round(
            float(prior_profile.get("effective_strength", 0.0) or 0.0), 3
        ),
        "cache_hit": bool(cache_hit),
        "cache_similarity": round(float(cache_hit.get("similarity", 0.0) or 0.0), 3)
        if cache_hit
        else 0.0,
        "cache_score": round(float(cache_hit.get("score", 0.0) or 0.0), 3)
        if cache_hit
        else 0.0,
        "reuse_sources": reuse_sources,
        "reuse_reason": reuse_reason,
        "execution_feedback_matches": int(
            execution_feedback.get("match_count", 0) or 0
        ),
        "outcome_memory_matches": int(outcome_memory.get("match_count", 0) or 0),
        "selection_mode": str(selection_report.get("selection_mode", "") or ""),
        "pareto_front_count": int(selection_report.get("pareto_front_count", 0) or 0),
        "rotation_condition_number": round(
            float(rotation_analysis.get("condition_number", 0.0) or 0.0), 3
        ),
        "parallax_pressure": round(
            float(parallax_telemetry.get("pressure", 0.0) or 0.0), 3
        ),
        "forbidden_subsequence_count": int(
            forbidden_subsequence_memory.get("pattern_count", 0) or 0
        ),
        "route_explanation": route_explanation[:4],
    }


def build_unified_workflow(
    problem_text: str,
    *,
    context: dict[str, Any] | None = None,
    feed: dict[str, Any] | None = None,
    state_dir: str | Path | None = None,
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    prior_strength: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intake = build_workflow_intake(problem_text, context=context, feed=feed)
    workflow_feed = _build_feed(problem_text, context=context, feed=feed)
    task_pool = derive_task_ecology(workflow_feed)
    feed_prior_dna = derive_feed_dna_prior(task_pool)
    patch_skeleton = build_patch_skeleton(task_pool, feed_prior_dna)
    state_root = resolve_state_root(state_dir)
    current_profile = task_shape_profile(intake, task_pool, feed_prior_dna)
    persisted_macros = load_macro_library(state_root) if state_root is not None else {}
    prior_profile = prior_strength_profile(
        intake, problem_text, task_pool, prior_strength
    )
    cache_hit = load_best_task_shape_prior(
        state_root,
        current_profile,
        min_similarity=prior_profile["cache_min_similarity"],
    )
    if cache_hit is not None:
        benchmark_result = materialize_task_shape_hit(
            cache_hit,
            task_pool,
            feed_prior_dna,
            patch_skeleton,
            persisted_macros,
            prior_profile,
        )
    else:
        benchmark_result = run_benchmark(
            trials=trials,
            population=population,
            base_episodes=base_episodes,
            generations=generations,
            seed0=seed0,
            feed=workflow_feed,
            state_dir=state_dir or None,
            config=config,
            task_shape_profile=current_profile,
        )
        if state_root is not None:
            save_task_shape_prior(
                state_root,
                {
                    "signature": "|".join(feed_prior_dna) or problem_text[:128],
                    "shape_profile": current_profile,
                    "score": task_shape_cache_score(benchmark_result),
                    "prior_profile": prior_profile,
                    "snapshot": {
                        "mode": benchmark_result.get("mode"),
                        "summary": benchmark_result.get("summary", {}),
                        "best_governed": benchmark_result.get("best_governed", {}),
                        "best_raw": benchmark_result.get("best_raw", {}),
                        "feed_prior_dna": benchmark_result.get(
                            "feed_prior_dna", feed_prior_dna
                        ),
                        "patch_skeleton": benchmark_result.get(
                            "patch_skeleton", patch_skeleton
                        ),
                        "archive_stable_top3": benchmark_result.get(
                            "archive_stable_top3", []
                        ),
                        "archive_raw_top3": benchmark_result.get(
                            "archive_raw_top3", []
                        ),
                        "controller_summary": benchmark_result.get(
                            "controller_summary", {}
                        ),
                        "phase_summary": benchmark_result.get("phase_summary", {}),
                        "transition_events_head": benchmark_result.get(
                            "transition_events_head", []
                        ),
                        "lineage_mutation_scale": benchmark_result.get(
                            "lineage_mutation_scale", {}
                        ),
                        "feed_profile": benchmark_result.get("feed_profile", {}),
                        "learned_macros": benchmark_result.get("learned_macros", {}),
                    },
                    "source": "benchmark",
                },
                metadata={
                    "problem_text": problem_text,
                    "route_hint": intake.get("route_hint", ""),
                    "cache_hit": False,
                },
            )

    if state_root is not None:
        benchmark_result.setdefault(
            "task_shape_signature", task_shape_signature(current_profile)
        )
        benchmark_result.setdefault(
            "outcome_memory",
            load_outcome_memory(
                state_root,
                current_profile,
                benchmark_result.get("benchmark_signature"),
            ),
        )
    execution_feedback = load_execution_feedback_for_shape(
        state_root,
        current_profile,
        benchmark_result.get("benchmark_signature"),
    )
    decision_policy = derive_act_ask_stop_policy(
        intake,
        prior_profile,
        benchmark_result.get("outcome_memory", {}),
        execution_feedback,
        benchmark_result,
    )
    prototype_summary = _build_prototype_summary(
        intake,
        prior_profile,
        cache_hit,
        benchmark_result.get("outcome_memory", {}),
        execution_feedback,
        decision_policy,
        benchmark_result.get("selection_report", {}),
        benchmark_result.get("rotation_analysis", {}),
        benchmark_result.get("parallax_telemetry", {}),
        benchmark_result.get("forbidden_subsequence_memory", {}),
    )
    synthesis = {
        "benchmark_contract": benchmark_contract(),
        "task_pool": task_pool,
        "feed_prior_dna": feed_prior_dna,
        "patch_skeleton": patch_skeleton,
        "prototype_summary": prototype_summary,
        "summary": benchmark_result["summary"],
        "best_governed": benchmark_result["best_governed"],
        "best_raw": benchmark_result["best_raw"],
        "archive_stable_top3": benchmark_result["archive_stable_top3"],
        "archive_raw_top3": benchmark_result["archive_raw_top3"],
        "semantic_promotions": benchmark_result.get("semantic_promotions", []),
        "semantic_counterexamples": benchmark_result.get(
            "semantic_counterexamples", []
        ),
        "held_out_semantic_credit": benchmark_result.get(
            "held_out_semantic_credit", {}
        ),
        "prototype_lifecycle": benchmark_result.get("prototype_lifecycle", []),
        "primitive_contracts": benchmark_result.get("primitive_contracts", []),
        "curriculum_profile": benchmark_result.get("curriculum_profile", {}),
        "devolution_candidates": benchmark_result.get("devolution_candidates", {}),
        "landscape_profile": benchmark_result.get("landscape_profile", {}),
        "selection_report": benchmark_result.get("selection_report", {}),
        "rotation_analysis": benchmark_result.get("rotation_analysis", {}),
        "parallax_telemetry": benchmark_result.get("parallax_telemetry", {}),
        "forbidden_subsequence_memory": benchmark_result.get(
            "forbidden_subsequence_memory", {}
        ),
        "controller_summary": benchmark_result.get("controller_summary"),
        "phase_summary": benchmark_result.get("phase_summary"),
        "transition_events_head": benchmark_result.get("transition_events_head"),
        "lineage_mutation_scale": benchmark_result.get("lineage_mutation_scale"),
        "feed_profile": benchmark_result.get("feed_profile"),
        "learned_macros": benchmark_result.get("learned_macros"),
        "task_shape_signature": benchmark_result.get("task_shape_signature", ""),
        "outcome_memory": benchmark_result.get("outcome_memory", {}),
        "execution_feedback": execution_feedback,
        "decision_policy": decision_policy,
        "prior_strength": prior_profile["effective_strength"],
        "prior_profile": prior_profile,
        "cache_hit": bool(benchmark_result.get("cache_hit", False)),
        "cache_similarity": benchmark_result.get("cache_similarity", 0.0),
    }
    return {
        "status": "synthesized",
        "problem_text": problem_text,
        "feed": workflow_feed,
        "intake": intake,
        "synthesis": synthesis,
        "execution": None,
        "prior_profile": prior_profile,
        "execution_feedback": execution_feedback,
        "decision_policy": decision_policy,
        "prototype_summary": prototype_summary,
        "selection_report": benchmark_result.get("selection_report", {}),
        "rotation_analysis": benchmark_result.get("rotation_analysis", {}),
        "parallax_telemetry": benchmark_result.get("parallax_telemetry", {}),
        "forbidden_subsequence_memory": benchmark_result.get(
            "forbidden_subsequence_memory", {}
        ),
        "cache_hit": bool(benchmark_result.get("cache_hit", False)),
        "api_version": API_VERSION,
    }


def run_unified_workflow(
    problem_text: str,
    *,
    context: dict[str, Any] | None = None,
    feed: dict[str, Any] | None = None,
    state_dir: str | Path | None = None,
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    prior_strength: float | None = None,
    config: dict[str, Any] | None = None,
    dispatch_plan: dict[str, Any] | str | Path | None = None,
    workspace_root: str | Path | None = None,
    commit: bool = False,
) -> dict[str, Any]:
    result = build_unified_workflow(
        problem_text,
        context=context,
        feed=feed,
        state_dir=state_dir,
        trials=trials,
        population=population,
        base_episodes=base_episodes,
        generations=generations,
        seed0=seed0,
        prior_strength=prior_strength,
        config=config,
    )
    execution = None
    if dispatch_plan is not None:
        execution = barricade_dispatch(
            dispatch_plan, workspace_root=workspace_root, commit=commit
        )
    elif commit:
        raise ValueError("dispatch_plan is required when commit is requested")

    if execution is not None:
        result["execution"] = execution
        result["status"] = execution.get("status", result["status"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Barricade unified workflow")
    parser.add_argument(
        "--problem-text", default="", help="Natural language description of the task"
    )
    parser.add_argument(
        "--problem-file", default="", help="Read the task description from a file"
    )
    parser.add_argument(
        "--context-json", default="", help="Optional JSON context payload"
    )
    parser.add_argument(
        "--feed-json", default="", help="Optional upstream feed JSON payload"
    )
    parser.add_argument(
        "--dispatch-plan-json", default="", help="JSON dispatch plan for execution"
    )
    parser.add_argument(
        "--dispatch-plan-file", default="", help="Path to a JSON dispatch plan"
    )
    parser.add_argument(
        "--workspace-root", default="", help="Workspace root for dispatch"
    )
    parser.add_argument(
        "--state-dir",
        default="",
        help="Optional state directory for synthesis persistence",
    )
    parser.add_argument(
        "--commit", action="store_true", help="Apply dispatch after verification passes"
    )
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--base-episodes", type=int, default=18)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seed0", type=int, default=100901)
    parser.add_argument("--prior-strength", type=float, default=None)
    args = parser.parse_args()

    problem_text = args.problem_text.strip()
    if args.problem_file:
        problem_text = Path(args.problem_file).read_text().strip()
    if not problem_text:
        parser.error("--problem-text or --problem-file is required")

    context = json.loads(args.context_json) if args.context_json else None
    feed = json.loads(args.feed_json) if args.feed_json else None
    dispatch_plan: dict[str, Any] | str | Path | None = None
    if args.dispatch_plan_json:
        dispatch_plan = json.loads(args.dispatch_plan_json)
    elif args.dispatch_plan_file:
        dispatch_plan = Path(args.dispatch_plan_file)

    result = run_unified_workflow(
        problem_text,
        context=context,
        feed=feed,
        state_dir=args.state_dir or None,
        trials=args.trials,
        population=args.population,
        base_episodes=args.base_episodes,
        generations=args.generations,
        seed0=args.seed0,
        prior_strength=args.prior_strength,
        dispatch_plan=dispatch_plan,
        workspace_root=args.workspace_root or None,
        commit=args.commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
