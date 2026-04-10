from __future__ import annotations

import math
import heapq
import random
import statistics
from collections import Counter
from typing import Any, Sequence, cast

from .analysis import (
    artifact_profile,
    build_curriculum_profile,
    build_forbidden_subsequences,
    build_optimizer_frame,
    build_primitive_contract_bank,
    build_semantic_promotion_bank,
    count_non_overlapping_subseq,
    estimate_artifact_quality,
    family_signature,
    flatten_trace,
    governance_variance,
    infer_specialization,
    library_metrics,
    lineage_entropy,
    motif_entropy,
    motif_key,
    compute_parallax_gradient,
    population_landscape_profile,
    semantic_family_credit,
    semantic_task_profile,
    semantic_trace_alignment,
    specialization_entropy,
)
from .constants import (
    BASE_MACROS,
    TASKS,
)
from .controller import (
    boost_for_missing_niche,
    decide_controller,
    exploit_dwell_stats,
    exploit_entry_barrier,
    exploit_score,
    motif_reinforcement_candidates,
    niche_target_counts,
    niche_overrepresentation_penalty,
    rolling_momentum,
    update_motif_market,
)
from .models import ControllerState, EvolutionConfig, Individual
from ._operators import (
    random_individual,
    specialization_pair_bonus,
    build_protected_tokens,
    mutate,
    crossover,
)


__all__ = [
    "choose_parent",
    "compute_governed_fitness",
    "ecology_round",
    "pick_diverse_elites",
    "summarize",
    "task_score",
    "best_payload",
    "ControllerState",
    "EvolutionConfig",
    "Individual",
    "boost_for_missing_niche",
    "decide_controller",
    "exploit_dwell_stats",
    "exploit_entry_barrier",
    "exploit_score",
    "motif_reinforcement_candidates",
    "niche_target_counts",
    "rolling_momentum",
    "update_motif_market",
    "random_individual",
    "specialization_pair_bonus",
    "build_protected_tokens",
    "mutate",
    "crossover",
]


OFFER_KINDS = ("plan", "patch", "summary")
FITNESS_AXES = (
    "train_score",
    "test_score",
    "stability_score",
    "artifact_yield",
    "gen_efficiency",
    "task_threshold_pass",
)


def _rotation_signal_from_metrics(metrics: dict[str, float]) -> float:
    motif_entropy = max(0.0, min(1.0, float(metrics.get("motif_entropy", 0.0) or 0.0)))
    specialization_entropy = max(
        0.0, min(1.0, float(metrics.get("specialization_entropy", 0.0) or 0.0))
    )
    lineage_entropy = max(
        0.0, min(1.0, float(metrics.get("lineage_entropy", 0.0) or 0.0))
    )
    dominant_macro_ratio = max(
        0.0, min(1.0, float(metrics.get("dominant_macro_ratio_mean", 0.0) or 0.0))
    )
    task_pressure = max(
        0.0, 0.82 - float(metrics.get("task_threshold_pass_mean", 0.0) or 0.0)
    )
    solve_pressure = max(
        0.0, 0.80 - float(metrics.get("solve_rate_test_mean", 0.0) or 0.0)
    )
    stability_pressure = max(
        0.0, 0.74 - float(metrics.get("stability_score_mean", 0.0) or 0.0)
    )
    balance_pressure = max(
        0.0, 1.0 - float(metrics.get("optimizer_balance_mean", 0.0) or 0.0)
    )
    variance_pressure = max(
        0.0,
        1.0 - min(1.0, float(metrics.get("governance_variance", 0.0) or 0.0) / 350.0),
    )
    parallax_pressure = max(
        0.0, min(1.0, float(metrics.get("parallax_pressure", 0.0) or 0.0))
    )

    signal = (
        0.24 * (1.0 - motif_entropy)
        + 0.20 * (1.0 - specialization_entropy)
        + 0.12 * (1.0 - lineage_entropy)
        + 0.16 * dominant_macro_ratio
        + 0.12 * task_pressure
        + 0.10 * solve_pressure
        + 0.04 * stability_pressure
        + 0.03 * balance_pressure
        + 0.03 * variance_pressure
        + 0.02 * parallax_pressure
    )
    signal += (
        max(0.0, float(metrics.get("rotation_condition", 0.0) or 0.0) - 1.5) * 0.01
    )
    return max(0.0, min(1.0, signal))


def _rotation_selection_scale(rotation_signal: float) -> float:
    return 0.08 + 0.70 * max(0.0, min(1.0, rotation_signal))


def _coerce_evolution_config(config) -> EvolutionConfig:
    resolved = EvolutionConfig()
    if config is None:
        return resolved
    if isinstance(config, EvolutionConfig):
        resolved = config
    elif isinstance(config, dict):
        for key in resolved.__dict__.keys():
            if key in config:
                setattr(resolved, key, config[key])
    else:
        for key in resolved.__dict__.keys():
            if hasattr(config, key):
                setattr(resolved, key, getattr(config, key))
    if resolved.enable_unified:
        resolved.enable_parallax = True
        resolved.enable_orthogonality = True
        resolved.enable_rotation = True
        resolved.enable_curriculum = True
        resolved.enable_primitive_contracts = True
    return resolved


def _normalize_axes(
    population: list[Individual], axes: tuple[str, ...]
) -> list[list[float]]:
    if not population:
        return []

    axis_values = {
        axis: [
            float(getattr(individual, axis, 0.0) or 0.0) for individual in population
        ]
        for axis in axes
    }
    minimums = {
        axis: min(values) if values else 0.0 for axis, values in axis_values.items()
    }
    maximums = {
        axis: max(values) if values else 0.0 for axis, values in axis_values.items()
    }

    normalized_vectors: list[list[float]] = []
    for index, individual in enumerate(population):
        vector: list[float] = []
        for axis in axes:
            span = maximums[axis] - minimums[axis]
            value = (
                0.0
                if span <= 1e-12
                else (axis_values[axis][index] - minimums[axis]) / span
            )
            value = max(0.0, min(1.0, value))
            vector.append(value)
        individual.fitness_vector = vector
        individual.selection_axes = {axis: vector[i] for i, axis in enumerate(axes)}
        normalized_vectors.append(vector)
    return normalized_vectors


def _dominates(left: list[float], right: list[float], dimensions: int) -> bool:
    at_least_as_good = all(left[index] >= right[index] for index in range(dimensions))
    strictly_better = any(left[index] > right[index] for index in range(dimensions))
    return at_least_as_good and strictly_better


def _compute_crowding_distance(
    population: list[Individual], front: list[int], dimensions: int
) -> None:
    if len(front) <= 2:
        for index in front:
            population[index].crowding_distance = float("inf")
        return

    for index in front:
        population[index].crowding_distance = 0.0

    for dimension in range(dimensions):
        front.sort(key=lambda index: population[index].fitness_vector[dimension])
        population[front[0]].crowding_distance = float("inf")
        population[front[-1]].crowding_distance = float("inf")
        minimum = population[front[0]].fitness_vector[dimension]
        maximum = population[front[-1]].fitness_vector[dimension]
        span = max(maximum - minimum, 1e-12)

        for index in range(1, len(front) - 1):
            population[front[index]].crowding_distance += (
                population[front[index + 1]].fitness_vector[dimension]
                - population[front[index - 1]].fitness_vector[dimension]
            ) / span


def _compute_pareto_fronts(
    population: list[Individual], dimensions: int
) -> list[list[int]]:
    if not population:
        return []

    domination_count = [0 for _ in population]
    dominates_by: list[list[int]] = [[] for _ in population]
    fronts: list[list[int]] = [[]]

    for left_index, left in enumerate(population):
        left_vector = left.fitness_vector[:dimensions]
        for right_index in range(left_index + 1, len(population)):
            right_vector = population[right_index].fitness_vector[:dimensions]
            if _dominates(left_vector, right_vector, dimensions):
                dominates_by[left_index].append(right_index)
                domination_count[right_index] += 1
            elif _dominates(right_vector, left_vector, dimensions):
                dominates_by[right_index].append(left_index)
                domination_count[left_index] += 1

    for index, count in enumerate(domination_count):
        if count == 0:
            population[index].pareto_rank = 0
            fronts[0].append(index)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for index in fronts[current]:
            for dominated_index in dominates_by[index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    population[dominated_index].pareto_rank = current + 1
                    next_front.append(dominated_index)
        current += 1
        fronts.append(next_front)

    return fronts[:-1]


def _assign_rotation_scores(
    population: list[Individual],
    landscape: dict[str, Any],
    rotation_strength: float,
) -> None:
    axes = landscape.get("axis_labels", [])
    rotation = landscape.get("rotation", {}) if isinstance(landscape, dict) else {}
    principal = {
        str(entry.get("label", "")): float(entry.get("loading", 0.0) or 0.0)
        for entry in rotation.get("principal_axis", [])
        if isinstance(entry, dict)
    }
    secondary = {
        str(entry.get("label", "")): float(entry.get("loading", 0.0) or 0.0)
        for entry in rotation.get("secondary_axis", [])
        if isinstance(entry, dict)
    }

    specialization_counts = Counter(
        individual.specialization for individual in population
    )
    max_specialization_count = (
        max(specialization_counts.values()) if specialization_counts else 1
    )

    primary_raw = []
    secondary_raw = []
    balance_scores = []
    market_scores = [
        float(getattr(individual, "market_score", 0.0) or 0.0)
        for individual in population
    ]
    market_min = min(market_scores) if market_scores else 0.0
    market_span = (max(market_scores) - market_min) if market_scores else 0.0

    for individual in population:
        primary_raw.append(
            sum(
                individual.selection_axes.get(axis, 0.0) * principal.get(axis, 0.0)
                for axis in axes
            )
        )
        secondary_raw.append(
            sum(
                individual.selection_axes.get(axis, 0.0) * secondary.get(axis, 0.0)
                for axis in axes
            )
        )
        balance_scores.append(
            sum(individual.selection_axes.get(axis, 0.0) for axis in axes)
            / max(1, len(axes))
        )

    primary_min = min(primary_raw) if primary_raw else 0.0
    primary_span = (max(primary_raw) - primary_min) if primary_raw else 0.0
    secondary_min = min(secondary_raw) if secondary_raw else 0.0
    secondary_span = (max(secondary_raw) - secondary_min) if secondary_raw else 0.0
    blend = max(0.0, min(1.0, rotation_strength))

    for individual, primary_value, secondary_value, balance in zip(
        population,
        primary_raw,
        secondary_raw,
        balance_scores,
    ):
        principal_norm = (
            0.5
            if primary_span <= 1e-12
            else (primary_value - primary_min) / primary_span
        )
        secondary_norm = (
            0.5
            if secondary_span <= 1e-12
            else (secondary_value - secondary_min) / secondary_span
        )
        market_norm = (
            0.5
            if market_span <= 1e-12
            else (float(getattr(individual, "market_score", 0.0) or 0.0) - market_min)
            / market_span
        )
        rarity = 1.0 - (
            specialization_counts.get(individual.specialization, 0)
            / max_specialization_count
        )
        market_alignment = (
            0.08 * market_norm
            + 0.18 * individual.task_threshold_pass
            + 0.18 * individual.solve_rate_test
            + 0.16 * individual.economic_stability
            + 0.20 * balance
            + 0.20 * rarity
        )
        diversity_alignment = 0.42 * balance + 0.58 * rarity
        individual.rotated_fitness = [
            round(blend * principal_norm + (1.0 - blend) * market_alignment, 3),
            round(blend * secondary_norm + (1.0 - blend) * diversity_alignment, 3),
        ]


def _assign_unified_selection_scores(population: list[Individual]) -> None:
    if not population:
        return

    max_pareto_rank = max(
        (individual.pareto_rank for individual in population), default=0
    )
    finite_crowding = [
        individual.crowding_distance
        for individual in population
        if not math.isinf(individual.crowding_distance)
    ]
    crowding_min = min(finite_crowding) if finite_crowding else 0.0
    crowding_span = (
        max(finite_crowding) - crowding_min if len(finite_crowding) > 1 else 0.0
    )

    rotation_values: list[float] = []
    vector_values: list[float] = []
    for individual in population:
        if individual.rotated_fitness:
            rotation_values.append(statistics.mean(individual.rotated_fitness))
        elif individual.selection_axes:
            rotation_values.append(statistics.mean(individual.selection_axes.values()))
        else:
            rotation_values.append(0.0)
        vector_values.append(
            statistics.mean(individual.fitness_vector)
            if individual.fitness_vector
            else 0.0
        )

    rotation_min = min(rotation_values) if rotation_values else 0.0
    rotation_span = (
        max(rotation_values) - rotation_min if len(rotation_values) > 1 else 0.0
    )
    vector_min = min(vector_values) if vector_values else 0.0
    vector_span = max(vector_values) - vector_min if len(vector_values) > 1 else 0.0

    specialization_counts = Counter(
        individual.specialization for individual in population
    )
    max_specialization_count = (
        max(specialization_counts.values()) if specialization_counts else 1
    )

    for index, individual in enumerate(population):
        pareto_term = (
            1.0
            if max_pareto_rank <= 0
            else 1.0 - (individual.pareto_rank / max_pareto_rank)
        )
        if math.isinf(individual.crowding_distance):
            crowding_term = 1.0
        else:
            crowding_term = (
                0.5
                if crowding_span <= 1e-12
                else (individual.crowding_distance - crowding_min) / crowding_span
            )
        rotation_value = rotation_values[index]
        rotation_term = (
            0.5
            if rotation_span <= 1e-12
            else (rotation_value - rotation_min) / rotation_span
        )
        vector_value = vector_values[index]
        vector_term = (
            0.5 if vector_span <= 1e-12 else (vector_value - vector_min) / vector_span
        )
        behavior_term = (
            0.26 * max(0.0, min(1.0, individual.stability_score))
            + 0.16 * max(0.0, min(1.0, individual.cognitive_stability))
            + 0.14 * max(0.0, min(1.0, individual.economic_stability))
            + 0.12 * max(0.0, min(1.0, individual.task_threshold_pass))
            + 0.10 * max(0.0, min(1.0, individual.solve_rate_test))
            + 0.08 * max(0.0, min(1.0, individual.gen_efficiency / 2.0))
            + 0.07 * max(0.0, min(1.0, max(0.0, individual.artifact_yield) / 6.0))
            + 0.07 * max(0.0, min(1.0, (individual.market_score + 12.0) / 24.0))
        )
        rarity_term = 1.0 - (
            specialization_counts.get(individual.specialization, 0)
            / max_specialization_count
        )
        selection_score = (
            0.32 * behavior_term
            + 0.20 * vector_term
            + 0.18 * rotation_term
            + 0.14 * pareto_term
            + 0.10 * crowding_term
            + 0.10 * rarity_term
        )
        individual.selection_score = round(max(0.0, min(1.0, selection_score)), 3)


def _rotation_market_signal(
    population: list[Individual], landscape: dict[str, Any]
) -> float:
    if not population:
        return 0.0

    metrics = {
        "task_threshold_pass_mean": statistics.mean(
            individual.task_threshold_pass for individual in population
        ),
        "economic_stability_mean": statistics.mean(
            individual.economic_stability for individual in population
        ),
        "market_score_mean": statistics.mean(
            individual.market_score for individual in population
        ),
        "profit_mean": statistics.mean(individual.profit for individual in population),
        "specialization_entropy": specialization_entropy(population),
        "parallax_pressure": float(
            landscape.get("parallax", {}).get("pressure", 0.0) or 0.0
        ),
        "rotation_condition": float(
            landscape.get("rotation", {}).get("condition_number", 0.0) or 0.0
        ),
    }
    return _rotation_signal_from_metrics(metrics)


def _unique_population(individuals: list[Individual]) -> list[Individual]:
    unique: list[Individual] = []
    seen: set[str] = set()
    for individual in individuals:
        lineage = individual.lineage_id
        if lineage in seen:
            continue
        seen.add(lineage)
        unique.append(individual)
    return unique


def _selection_profiles(
    population: list[Individual],
    macro_lib: dict[str, list[str]],
    tasks: list[dict[str, Any]],
    config: EvolutionConfig,
    forbidden_subsequence_memory: list[Sequence[str]] | None = None,
) -> dict[str, Any]:
    if not population:
        memory_forbidden = [
            tuple(str(token) for token in seq if str(token))
            for seq in forbidden_subsequence_memory or ()
            if len(seq) >= 2
        ]
        return {
            "ranked": [],
            "governors": [],
            "explorers": [],
            "probes": [],
            "pareto_front_sizes": [],
            "elite_traces": [],
            "probe_traces": [],
            "gradient": {},
            "forbidden_subsequences": memory_forbidden,
            "curriculum_profile": build_curriculum_profile(tasks),
            "primitive_contracts": [],
            "landscape_profile": {},
            "top_axis": {},
            "bottom_axis": {},
            "rotation_top": {},
            "rotation_bottom": {},
            "rotation_market_signal": 0.0,
            "rotation_strength": 0.0,
            "selection_mode": "unified"
            if config.enable_unified
            else (
                "multi_objective"
                if config.enable_parallax
                or config.enable_orthogonality
                or config.enable_rotation
                else "scalar"
            ),
            "contract_guard_tokens": set(),
        }

    landscape = population_landscape_profile(population)
    axes = tuple(landscape.get("axis_labels", FITNESS_AXES))[
        : config.selection_axis_count
    ]
    _normalize_axes(population, axes)
    rotation_signal = _rotation_market_signal(population, landscape)
    rotation_strength = min(0.62, 0.08 + 0.42 * rotation_signal)
    if config.enable_orthogonality:
        fronts = _compute_pareto_fronts(
            population, min(config.pareto_dimensions, len(axes))
        )
    else:
        for individual in population:
            individual.pareto_rank = 0
            individual.crowding_distance = 0.0
        fronts = [[index for index in range(len(population))]]

    if config.enable_rotation:
        _assign_rotation_scores(population, landscape, rotation_strength)
    else:
        for individual in population:
            individual.rotated_fitness = []

    _assign_unified_selection_scores(population)

    ranked = sorted(
        population,
        key=lambda individual: (
            individual.pareto_rank,
            -individual.selection_score,
            -individual.crowding_distance,
            -individual.stability_score,
        ),
    )
    probe_count = max(2, int(round(len(population) * config.parallax_probe_fraction)))
    probe_pool = ranked[-probe_count:] if config.enable_parallax else []
    elite_cutoff = max(4, len(population) // 6)
    elite_pool = ranked[:elite_cutoff]

    def _fitness_at(individual: Individual, index: int) -> float:
        return float(individual.fitness_vector[index])

    top_axis = {}
    for idx, axis in enumerate(axes):
        def _top_key(ind: Individual, i: int = idx) -> float:
            return _fitness_at(ind, i)
        top_axis[axis] = max(population, key=_top_key)

    bottom_axis = {}
    for idx, axis in enumerate(axes):
        def _bottom_key(ind: Individual, i: int = idx) -> float:
            return _fitness_at(ind, i)
        bottom_axis[axis] = min(population, key=_bottom_key)

    if config.enable_rotation and any(
        individual.rotated_fitness for individual in population
    ):

        def principal_key(individual):
            return individual.rotated_fitness[0] if individual.rotated_fitness else 0.0

        def secondary_key(individual):
            return individual.rotated_fitness[1] if individual.rotated_fitness else 0.0

        rotation_top = {
            "principal": max(population, key=principal_key),
            "secondary": max(population, key=secondary_key),
        }
        rotation_bottom = {
            "principal": min(population, key=principal_key),
            "secondary": min(population, key=secondary_key),
        }
    else:
        rotation_top = {}
        rotation_bottom = {}

    elite_traces = [
        list(
            getattr(
                individual, "flattened_trace", flatten_trace(individual.dna, macro_lib)
            )
        )
        for individual in elite_pool
    ]
    probe_traces = [
        list(
            getattr(
                individual, "flattened_trace", flatten_trace(individual.dna, macro_lib)
            )
        )
        for individual in probe_pool
    ]
    gradient = (
        compute_parallax_gradient(elite_traces, probe_traces)
        if config.enable_parallax
        else {}
    )
    forbidden_subsequences = (
        build_forbidden_subsequences(probe_traces, elite_traces)
        if config.enable_parallax
        else []
    )
    memory_forbidden = [
        tuple(str(token) for token in seq if str(token))
        for seq in forbidden_subsequence_memory or ()
        if len(seq) >= 2
    ]
    if memory_forbidden:
        combined_forbidden: list[tuple[str, ...]] = []
        seen_forbidden: set[tuple[str, ...]] = set()
        for subsequence in list(forbidden_subsequences) + memory_forbidden:
            key = tuple(subsequence)
            if len(key) < 2 or key in seen_forbidden:
                continue
            seen_forbidden.add(key)
            combined_forbidden.append(key)
        forbidden_subsequences = combined_forbidden[:24]

    pareto_front_sizes = [len(front) for front in fronts]

    curriculum_profile = (
        build_curriculum_profile(tasks)
        if config.enable_curriculum
        else {
            "task_count": len(tasks),
            "stage_order": {"candidate": 0, "emerging": 1, "stable": 2, "mature": 3},
            "stage_counts": dict(
                Counter(str(task.get("focus", "general")) for task in tasks)
            ),
            "schedule": [
                {
                    "task_name": str(task.get("name", f"task_{index}")),
                    "focus": str(task.get("focus", "general")),
                    "prototype_stage": str(
                        task.get("prototype_stage", task.get("focus", "candidate"))
                    ),
                    "difficulty": 0.0,
                    "uncertainty": 0.0,
                    "rule_weight": 1.0,
                    "holdout_bucket": int(task.get("holdout_bucket", 0) or 0),
                }
                for index, task in enumerate(tasks)
            ],
            "ordered_tasks": tasks,
        }
    )
    primitive_contracts = (
        build_primitive_contract_bank(
            tasks, build_semantic_promotion_bank(tasks, elite_traces), None, None
        )
        if config.enable_primitive_contracts
        else []
    )

    contract_guard_tokens = {
        str(token)
        for contract in primitive_contracts
        if contract.get("contract_ready")
        for token in contract.get("guard_tokens", [])
    }

    governors = _unique_population(
        list(top_axis.values()) + list(rotation_top.values()) + elite_pool
    )
    explorers = _unique_population(
        governors
        + list(bottom_axis.values())
        + list(rotation_bottom.values())
        + probe_pool
        + ranked
    )

    return {
        "ranked": ranked,
        "governors": governors,
        "explorers": explorers,
        "probes": probe_pool,
        "pareto_front_sizes": pareto_front_sizes,
        "elite_traces": elite_traces,
        "probe_traces": probe_traces,
        "gradient": gradient,
        "forbidden_subsequences": forbidden_subsequences,
        "curriculum_profile": curriculum_profile,
        "primitive_contracts": primitive_contracts,
        "landscape_profile": landscape,
        "top_axis": top_axis,
        "bottom_axis": bottom_axis,
        "rotation_top": rotation_top,
        "rotation_bottom": rotation_bottom,
        "rotation_market_signal": round(rotation_signal, 3),
        "rotation_strength": round(
            rotation_strength if config.enable_rotation else 0.0, 3
        ),
        "selection_mode": "unified"
        if config.enable_unified
        else (
            "multi_objective"
            if config.enable_parallax
            or config.enable_orthogonality
            or config.enable_rotation
            else "scalar"
        ),
        "contract_guard_tokens": contract_guard_tokens,
    }


def task_score(flat, task, inventory, rng):
    req = task["req"]
    score = 0.0
    pos = -1
    hits = 0
    for op in req:
        try:
            pos = flat.index(op, pos + 1)
            hits += 1
        except ValueError:
            break
    score += hits * 9.0 - (len(req) - hits) * 6.5
    fam_score, _ = semantic_family_credit(flat)
    score += fam_score
    semantic_profile = semantic_task_profile(task)
    score += 3.5 * semantic_trace_alignment(flat, semantic_profile)
    optimizer_frame = build_optimizer_frame(flat)
    geometry = optimizer_frame["geometry"]
    combinators = optimizer_frame["combinators"]
    token_hist = optimizer_frame["token_histogram"]
    score += 1.4 * geometry["plan_axis"] * task["needs"].get("plan", 0)
    score += 1.4 * geometry["patch_axis"] * task["needs"].get("patch", 0)
    score += 1.4 * geometry["summary_axis"] * task["needs"].get("summary", 0)
    score += 0.8 * geometry["verify_axis"]
    score += 0.9 * combinators["discover_chain"]
    score += 0.35 * geometry["balance"]
    score += 0.15 * token_hist.get("macro", 0)
    tool_calls = flat.count("QUERY_TOOL")
    if tool_calls:
        good = sum(1 for _ in range(tool_calls) if rng.random() > task["tool_noise"])
        score += good * 2.0 - (tool_calls - good) * 1.8
    if task["hazard_if_no_verify"] and "VERIFY" not in flat:
        score -= 15.0
    for k, needed in task["needs"].items():
        if inventory.get(k, 0) >= needed:
            score += 12.0 * needed
        elif inventory.get(k, 0) > 0:
            score += 6.0 * inventory.get(k, 0)
    if task["name"] in ("stale_artifact_audit", "cross_team_handoff"):
        score += 8.0 if ("REVISE_ARTIFACT" in flat or "LINK_ARTIFACT" in flat) else -6.0
    score -= len(flat) * 1.0
    solved = hits >= max(3, len(req) - 1) and score > 0
    return score, solved


def ecology_round(pop, macro_lib, episodes, rng_seed, task_pool=None):
    rng = random.Random(rng_seed)
    tasks = task_pool or TASKS
    offers = []
    offer_profiles = []
    offer_desirability = []
    seller_versions = []
    seller_heaps = {kind: [] for kind in OFFER_KINDS}
    for idx, ind in enumerate(pop):
        flat = flatten_trace(ind.dna, macro_lib)
        setattr(ind, "flattened_trace", flat)
        inv = artifact_profile(flat)
        quality = estimate_artifact_quality(flat)
        specialization = infer_specialization(flat)
        sell = {
            "plan": inv.get("plan", 0),
            "patch": inv.get("patch", 0),
            "summary": inv.get("summary", 0),
        }
        prices = {
            k: max(0.8, 3.0 + 1.4 * quality[k] - 0.45 * sell[k] + 0.6 * rng.random())
            for k in OFFER_KINDS
        }
        base_desirability = {
            k: prices[k] / max(0.4, quality[k] + 0.4) for k in OFFER_KINDS
        }
        offers.append((idx, sell, prices, flat, quality, specialization))
        offer_profiles.append(inv)
        offer_desirability.append(base_desirability)
        seller_versions.append(0)
        for kind in OFFER_KINDS:
            if sell.get(kind, 0) > 0:
                heapq.heappush(
                    seller_heaps[kind],
                    (base_desirability[kind], idx, 0, kind),
                )

    wallets = [30.0 for _ in pop]
    inventories = [Counter() for _ in pop]
    profits = [0.0 for _ in pop]
    reputation = [0.0 for _ in pop]

    def refresh_seller_entries(seller_idx):
        seller_version = seller_versions[seller_idx]
        current_reputation = reputation[seller_idx]
        seller_stock = offers[seller_idx][1]
        for kind in OFFER_KINDS:
            if seller_stock.get(kind, 0) > 0:
                heapq.heappush(
                    seller_heaps[kind],
                    (
                        offer_desirability[seller_idx][kind] - 0.1 * current_reputation,
                        seller_idx,
                        seller_version,
                        kind,
                    ),
                )

    for round_i in range(max(5, episodes // 2)):
        demand_task = tasks[(round_i * 3 + 1) % len(tasks)]
        for buyer_idx, _ in enumerate(pop):
            local_inv = offer_profiles[buyer_idx]
            for k, needed in demand_task["needs"].items():
                local_have = local_inv.get(k, 0) + inventories[buyer_idx].get(k, 0)
                shortage = max(0, needed - local_have)
                while shortage > 0 and wallets[buyer_idx] > 0:
                    heap = seller_heaps[k]
                    skipped_self = []
                    chosen_sid = None
                    chosen_entry = None
                    while heap:
                        desirability, sid, version, heap_kind = heapq.heappop(heap)
                        if version != seller_versions[sid]:
                            continue
                        if offers[sid][1].get(k, 0) <= 0:
                            continue
                        if sid == buyer_idx:
                            skipped_self.append((desirability, sid, version, heap_kind))
                            continue
                        chosen_sid = sid
                        chosen_entry = (desirability, sid, version, heap_kind)
                        break
                    for entry in skipped_self:
                        heapq.heappush(heap, entry)
                    if chosen_sid is None:
                        break
                    price = offers[chosen_sid][2][k]
                    quality = offers[chosen_sid][4][k]
                    if wallets[buyer_idx] < price:
                        if chosen_entry is not None:
                            heapq.heappush(heap, chosen_entry)
                        break
                    wallets[buyer_idx] -= price
                    wallets[chosen_sid] += price
                    profits[chosen_sid] += price
                    profits[buyer_idx] -= price
                    inventories[buyer_idx][k] += 1
                    offers[chosen_sid][1][k] -= 1
                    reputation[chosen_sid] += min(0.5, quality * 0.08)
                    seller_versions[chosen_sid] += 1
                    refresh_seller_entries(chosen_sid)
                    shortage -= 1

    evaluated = []
    for idx, ind in enumerate(pop):
        flat = offers[idx][3]
        inv = artifact_profile(flat)
        optimizer_frame = build_optimizer_frame(flat)
        ind.flattened_len = len(flat)
        all_macros = {**BASE_MACROS, **macro_lib}
        ind.macro_hits = {}
        for name, pat in all_macros.items():
            hits = count_non_overlapping_subseq(flat, pat)
            if hits:
                ind.macro_hits[name] = hits
        _, fam_counts = semantic_family_credit(flat)
        ind.reuse_score = sum(1 for v in fam_counts.values() if v > 0)

        scores = []
        solves = []
        threshold_hits = []
        art_reuse_scores = []
        art_yields = []
        for ep in range(episodes):
            task = tasks[(ep * 5 + 2) % len(tasks)]
            s, solved = task_score(flat, task, inventories[idx], rng)
            scores.append(s)
            solves.append(1 if solved else 0)
            threshold_hits.append(1 if (solved and s >= 24.0) else 0)

            useful = 0
            for k in ("plan", "patch", "summary"):
                if task["needs"].get(k, 0) and (
                    inv.get(k, 0) > 0 or inventories[idx].get(k, 0) > 0
                ):
                    useful += 1
            writes = (
                inv.get("plan", 0)
                + inv.get("patch", 0)
                + inv.get("summary", 0)
                + inv.get("note", 0)
            )
            reads = inv.get("reads", 0)
            art_reuse_scores.append(useful)
            art_yields.append(
                (
                    useful * 5.0
                    - max(0, writes - useful) * 1.0
                    - max(0, reads - useful) * 0.35
                )
                / max(1.0, writes + reads)
            )

        ind.test_score = statistics.mean(scores)
        ind.solve_rate_test = statistics.mean(solves)
        ind.train_score = ind.test_score - (8.0 + 6.0 * rng.random())
        ind.solve_rate_train = ind.solve_rate_test * (0.82 + 0.14 * rng.random())
        ind.generalization_gap = ind.test_score - ind.train_score
        ind.generalization_gap_norm = ind.generalization_gap / max(
            1e-6, abs(ind.train_score)
        )
        ind.artifact_reuse_score = statistics.mean(art_reuse_scores)
        ind.artifact_yield = statistics.mean(art_yields)
        ind.profit = profits[idx]
        ind.wallet_end = wallets[idx]
        ind.gen_efficiency = ind.test_score / max(1.0, ind.flattened_len)
        ind.library_concentration, ind.dominant_macro_ratio = library_metrics(
            ind.dna, macro_lib
        )
        ind.artifact_inventory = dict(inventories[idx])
        ind.specialization = offers[idx][5]
        ind.task_threshold_pass = statistics.mean(threshold_hits)
        ind.family_sig = family_signature(flat)
        ind.motif_sig = motif_key(ind.dna, 3)
        ind.optimizer_frame = optimizer_frame

        gated_market = (
            profits[idx] + 4.0 * sum(inventories[idx].values()) + 3.0 * reputation[idx]
        ) * (0.25 + 0.75 * ind.task_threshold_pass)
        ind.market_score = gated_market

        specialization_bonus = 0.0
        if ind.specialization != "generalist":
            specialization_bonus += 2.5 + min(
                2.0, sum(inventories[idx].values()) * 0.25
            )

        dominant_penalty = 0.0
        if ind.dominant_macro_ratio > 0.60:
            dominant_penalty += (ind.dominant_macro_ratio - 0.60) * 20.0
        if ind.library_concentration > 0.80:
            dominant_penalty += (ind.library_concentration - 0.80) * 10.0

        ind.fitness = (
            0.42 * ind.train_score
            + 0.58 * ind.test_score
            + ind.solve_rate_test * 18.0
            + ind.reuse_score * 3.0
            + ind.artifact_reuse_score * 4.0
            + ind.artifact_yield * 8.0
            + ind.market_score * 1.15
            + ind.gen_efficiency * 1.5
            + ind.task_threshold_pass * 22.0
            + optimizer_frame["geometry"]["plan_axis"] * 4.0
            + optimizer_frame["geometry"]["balance"] * 2.5
            + optimizer_frame["combinators"]["discover_chain"] * 1.1
            + specialization_bonus
            - max(0, ind.flattened_len - 30) * 0.34
            - dominant_penalty
        )
        evaluated.append(ind)
    return evaluated


def compute_governed_fitness(pop):
    for x in pop:
        geometry = (
            x.optimizer_frame.get("geometry", {})
            if isinstance(x.optimizer_frame, dict)
            else {}
        )
        combinators = (
            x.optimizer_frame.get("combinators", {})
            if isinstance(x.optimizer_frame, dict)
            else {}
        )
        cognitive = (
            0.48 * x.task_threshold_pass
            + 0.28 * x.solve_rate_test
            + 0.14 * min(1.0, x.gen_efficiency / 2.0)
            + 0.10 * min(1.0, max(0.0, x.artifact_yield) / 6.0)
            + 0.05
            * min(
                1.0, geometry.get("plan_axis", 0.0) + geometry.get("summary_axis", 0.0)
            )
        )
        economic = (
            0.48 * min(1.0, max(0.0, x.market_score + 12.0) / 24.0)
            + 0.30 * min(1.0, max(0.0, x.profit + 10.0) / 22.0)
            + 0.22 * (1.0 if x.specialization != "generalist" else 0.4)
            + 0.03 * min(1.0, combinators.get("artifact_loops", 0) / 4.0)
        )
        mono_penalty = max(0.0, x.dominant_macro_ratio - 0.65) * 0.9
        econ_penalty = 0.0 if x.profit >= -2 else min(0.38, abs(x.profit + 2) / 40.0)
        x.cognitive_stability = max(0.0, min(1.0, cognitive))
        x.economic_stability = max(0.0, min(1.0, economic - econ_penalty))
        x.stability_score = max(
            0.0,
            0.62 * x.cognitive_stability
            + 0.38 * x.economic_stability
            - mono_penalty
            + 0.03 * geometry.get("balance", 0.0),
        )
        x.governed_fitness = x.fitness * (0.60 + 0.62 * x.stability_score)


def summarize(pop):
    landscape_profile = population_landscape_profile(pop)
    axis_variances = list(landscape_profile.get("axis_variance", {}).values())
    variance_spread = (
        round(max(axis_variances) - min(axis_variances), 3) if axis_variances else 0.0
    )
    geometry = [
        x.optimizer_frame.get("geometry", {})
        if isinstance(x.optimizer_frame, dict)
        else {}
        for x in pop
    ]
    combinators = [
        x.optimizer_frame.get("combinators", {})
        if isinstance(x.optimizer_frame, dict)
        else {}
        for x in pop
    ]
    ast_profiles = [
        x.optimizer_frame.get("ast_profile", {})
        if isinstance(x.optimizer_frame, dict)
        else {}
        for x in pop
    ]
    summary = {
        "mean": round(statistics.mean(x.fitness for x in pop), 3),
        "governed_mean": round(statistics.mean(x.governed_fitness for x in pop), 3),
        "selection_score_mean": round(
            statistics.mean(x.selection_score for x in pop), 3
        ),
        "train_mean": round(statistics.mean(x.train_score for x in pop), 3),
        "test_mean": round(statistics.mean(x.test_score for x in pop), 3),
        "solve_rate_test_mean": round(
            statistics.mean(x.solve_rate_test for x in pop), 3
        ),
        "task_threshold_pass_mean": round(
            statistics.mean(x.task_threshold_pass for x in pop), 3
        ),
        "artifact_yield_mean": round(statistics.mean(x.artifact_yield for x in pop), 3),
        "profit_mean": round(statistics.mean(x.profit for x in pop), 3),
        "market_score_mean": round(statistics.mean(x.market_score for x in pop), 3),
        "stability_score_mean": round(
            statistics.mean(x.stability_score for x in pop), 3
        ),
        "cognitive_stability_mean": round(
            statistics.mean(x.cognitive_stability for x in pop), 3
        ),
        "economic_stability_mean": round(
            statistics.mean(x.economic_stability for x in pop), 3
        ),
        "dominant_macro_ratio_mean": round(
            statistics.mean(x.dominant_macro_ratio for x in pop), 3
        ),
        "governance_variance": round(governance_variance(pop), 3),
        "motif_entropy": round(motif_entropy(pop), 3),
        "specialization_entropy": round(specialization_entropy(pop), 3),
        "lineage_entropy": round(lineage_entropy(pop), 3),
        "optimizer_plan_axis_mean": round(
            statistics.mean(g.get("plan_axis", 0.0) for g in geometry), 3
        ),
        "optimizer_patch_axis_mean": round(
            statistics.mean(g.get("patch_axis", 0.0) for g in geometry), 3
        ),
        "optimizer_summary_axis_mean": round(
            statistics.mean(g.get("summary_axis", 0.0) for g in geometry), 3
        ),
        "optimizer_verify_axis_mean": round(
            statistics.mean(g.get("verify_axis", 0.0) for g in geometry), 3
        ),
        "optimizer_balance_mean": round(
            statistics.mean(g.get("balance", 0.0) for g in geometry), 3
        ),
        "optimizer_mixed_pressure_mean": round(
            statistics.mean(g.get("mixed_pressure", 0.0) for g in geometry), 3
        ),
        "optimizer_combinator_mass_mean": round(
            statistics.mean(g.get("combinator_mass", 0.0) for g in geometry), 3
        ),
        "optimizer_ast_depth_mean": round(
            statistics.mean(a.get("max_depth", 0) for a in ast_profiles), 3
        ),
        "optimizer_discover_chain_mean": round(
            statistics.mean(c.get("discover_chain", 0.0) for c in combinators), 3
        ),
        "landscape_profile": landscape_profile,
        "parallax_pressure": round(
            landscape_profile.get("parallax", {}).get("pressure", 0.0), 3
        ),
        "rotation_condition": round(
            landscape_profile.get("rotation", {}).get("condition_number", 0.0), 3
        ),
        "devolution_pressure": round(
            abs(landscape_profile.get("parallax", {}).get("pressure", 0.0)), 3
        ),
        "orthogonality_spread": variance_spread,
        "dominant_axis": landscape_profile.get("dominant_axis", ""),
    }
    summary["rotation_market_signal"] = round(_rotation_signal_from_metrics(summary), 3)
    return summary


def best_payload(x):
    return {
        "dna": x.dna,
        "fitness": round(x.fitness, 3),
        "governed_fitness": round(x.governed_fitness, 3),
        "selection_score": round(x.selection_score, 3),
        "stability_score": round(x.stability_score, 3),
        "cognitive_stability": round(x.cognitive_stability, 3),
        "economic_stability": round(x.economic_stability, 3),
        "train_score": round(x.train_score, 3),
        "test_score": round(x.test_score, 3),
        "solve_rate_test": round(x.solve_rate_test, 3),
        "task_threshold_pass": round(x.task_threshold_pass, 3),
        "artifact_yield": round(x.artifact_yield, 3),
        "profit": round(x.profit, 3),
        "market_score": round(x.market_score, 3),
        "specialization": x.specialization,
        "family_sig": x.family_sig,
        "motif_sig": x.motif_sig,
        "lineage_id": x.lineage_id,
        "parent_ids": x.parent_ids,
        "macro_hits": x.macro_hits,
        "optimizer_frame": x.optimizer_frame,
    }


def _selection_value(individual: Any) -> float:
    if isinstance(individual, dict):
        if "selection_score" in individual:
            selection_score = individual.get("selection_score")
            if selection_score is not None:
                try:
                    value = float(selection_score)
                except (TypeError, ValueError):
                    value = None
                else:
                    if (
                        value != 0.0
                        or individual.get("selection_axes")
                        or individual.get("fitness_vector")
                    ):
                        return cast(float, value)
        governed_fitness = individual.get("governed_fitness", 0.0)
    else:
        selection_score = getattr(individual, "selection_score", None)
        if selection_score is not None:
            try:
                value = float(selection_score)
            except (TypeError, ValueError):
                value = None
            else:
                if (
                    value != 0.0
                    or getattr(individual, "selection_axes", None)
                    or getattr(individual, "fitness_vector", None)
                ):
                    return value
        governed_fitness = getattr(individual, "governed_fitness", 0.0)
    try:
        return float(governed_fitness or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pick_diverse_elites(pop, limit):
    chosen = []
    seen_specs, seen_fams, seen_motifs = set(), set(), set()
    for e in sorted(pop, key=_selection_value, reverse=True):
        if (
            e.specialization not in seen_specs
            or e.family_sig not in seen_fams
            or e.motif_sig not in seen_motifs
        ):
            chosen.append(e)
            seen_specs.add(e.specialization)
            seen_fams.add(e.family_sig)
            seen_motifs.add(e.motif_sig)
            if len(chosen) >= limit:
                return chosen
    for e in sorted(pop, key=_selection_value, reverse=True):
        if e not in chosen:
            chosen.append(e)
            if len(chosen) >= limit:
                break
    return chosen


def choose_parent(
    explorers,
    rng,
    spec_counts,
    targets=None,
    rotation_enabled: bool = False,
    rotation_signal: float = 1.0,
):
    pool = rng.sample(explorers, min(8, len(explorers)))
    target_map = targets or {}
    rotation_scale = (
        _rotation_selection_scale(rotation_signal)
        if rotation_enabled and rotation_signal >= 0.25
        else 0.0
    )
    return max(
        pool,
        key=lambda x: (
            _selection_value(x)
            - niche_overrepresentation_penalty(
                x.specialization, spec_counts, target_map
            )
            + 2.5 * (1.0 / (1 + spec_counts.get(x.specialization, 0)))
            + rotation_scale
            * 5.0
            * (x.rotated_fitness[0] if x.rotated_fitness else 0.0)
            + rotation_scale
            * 2.0
            * (x.rotated_fitness[1] if x.rotated_fitness else 0.0)
            + (
                rotation_scale
                * 4.0
                * boost_for_missing_niche(x.specialization, spec_counts, target_map)
                if rotation_enabled
                else 0.0
            )
        ),
    )
