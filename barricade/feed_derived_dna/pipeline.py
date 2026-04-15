from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .analysis import (
    build_curriculum_profile,
    build_semantic_counterexample_bank,
    build_semantic_promotion_bank,
    build_primitive_contract_bank,
    held_out_semantic_credit,
    flatten_trace,
    infer_specialization,
    mine_macros_from_elites,
    population_landscape_profile,
    semantic_prototype_lifecycle,
)
from .constants import TASKS
from ._operators import adversarial_push, replicate_scout, scout_teleport
from ._outcome_memory import (
    build_outcome_record,
    load_outcome_memory,
    normalize_task_shape_profile,
)
from .evolution import (
    ControllerState,
    boost_for_missing_niche,
    build_protected_tokens,
    _coerce_evolution_config,
    _rotation_selection_scale,
    _selection_profiles,
    choose_parent,
    compute_governed_fitness,
    crossover,
    decide_controller,
    ecology_round,
    exploit_dwell_stats,
    exploit_entry_barrier,
    exploit_score,
    best_payload,
    motif_reinforcement_candidates,
    mutate,
    niche_target_counts,
    Individual,
    random_individual,
    rolling_momentum,
    summarize,
    update_motif_market,
    specialization_pair_bonus,
)
from .controller import anti_reinforcement_candidates
from .persistence import (
    benchmark_run_signature,
    benchmark_state_fingerprint,
    load_macro_library,
    load_forbidden_subsequence_memory,
    load_motif_cache,
    load_benchmark_run,
    resolve_state_root,
    save_forbidden_subsequence_memory,
    save_discoverables,
    save_lineage_archive,
    save_benchmark_run,
    save_outcome_ledger,
    save_run_summary,
    task_shape_signature,
)
from .tasks import (
    _collect_text_fragments,
    build_patch_skeleton,
    derive_feed_dna_prior,
    derive_task_ecology,
    seed_population_from_prior,
)


SEMANTIC_ENGINE_VERSION = "semantic-ir-v2"


def _successful_benchmark_traces(
    population: list[Individual], macro_lib: dict[str, list[str]], limit: int = 14
) -> list[list[str]]:
    successful_traces = [
        (
            cached_flattened
            if cached_flattened is not None
            else flatten_trace(individual.dna, macro_lib)
        )
        for individual in sorted(population, key=_selection_score_value, reverse=True)
        if individual.task_threshold_pass >= 0.50 or individual.stability_score >= 0.45
        for cached_flattened in (getattr(individual, "flattened_trace", None),)
    ][:limit]
    if len(successful_traces) == 1:
        successful_traces = successful_traces * 2
    return successful_traces


def _elite_traces_from_result(benchmark_result: dict[str, Any]) -> list[list[str]]:
    traces: list[list[str]] = []
    for key in ("archive_stable_top3", "archive_raw_top3"):
        entries = benchmark_result.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            dna = entry.get("dna")
            if isinstance(dna, list) and dna:
                traces.append([str(token) for token in dna])

    for metric_key in ("best_governed", "best_raw"):
        entry = benchmark_result.get(metric_key)
        if not isinstance(entry, dict):
            continue
        dna = entry.get("dna")
        if isinstance(dna, list) and dna:
            traces.append([str(token) for token in dna])

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for trace in traces:
        trace_key = tuple(trace)
        if trace_key in seen:
            continue
        seen.add(trace_key)
        unique.append(trace)
    return unique


def _landscape_population_from_result(
    benchmark_result: dict[str, Any],
) -> list[dict[str, Any]]:
    population: list[dict[str, Any]] = []
    for key in ("archive_stable_top3", "archive_raw_top3"):
        entries = benchmark_result.get(key, [])
        if isinstance(entries, list):
            population.extend(entry for entry in entries if isinstance(entry, dict))

    for metric_key in ("best_governed", "best_raw"):
        entry = benchmark_result.get(metric_key)
        if isinstance(entry, dict):
            population.append(entry)

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, object]] = set()
    for item in population:
        identity = (str(item.get("lineage_id", "")), item.get("motif_sig", ""))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _selection_score_value(individual: Any) -> float:
    if isinstance(individual, dict):
        selection_score = individual.get("selection_score")
        governed_fitness = individual.get("governed_fitness", 0.0)
        has_selection_context = bool(
            individual.get("selection_axes") or individual.get("fitness_vector")
        )
    else:
        selection_score = getattr(individual, "selection_score", None)
        governed_fitness = getattr(individual, "governed_fitness", 0.0)
        has_selection_context = bool(
            getattr(individual, "selection_axes", None)
            or getattr(individual, "fitness_vector", None)
        )

    try:
        selection_value = (
            float(selection_score) if selection_score is not None else None
        )
    except (TypeError, ValueError):
        selection_value = None

    if selection_value is not None and (
        selection_value != 0.0 or has_selection_context
    ):
        return selection_value

    try:
        return float(governed_fitness or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _selection_mode_from_config(config: Any) -> str:
    if getattr(config, "enable_unified", False):
        return "unified"
    if any(
        getattr(config, flag, False)
        for flag in ("enable_parallax", "enable_orthogonality", "enable_rotation")
    ):
        return "multi_objective"
    return "scalar"


def _build_selection_report(
    selection_profile: dict[str, Any] | None,
    benchmark_result: dict[str, Any],
    config: Any,
    forbidden_memory: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(selection_profile, dict):
        selection_profile = {}
    pareto_front_sizes = selection_profile.get("pareto_front_sizes", [])
    if not isinstance(pareto_front_sizes, list):
        pareto_front_sizes = []
    probes = selection_profile.get("probes", [])
    elite_traces = selection_profile.get("elite_traces", [])
    forbidden = selection_profile.get("forbidden_subsequences", [])
    contract_guard_tokens = selection_profile.get("contract_guard_tokens", [])
    if not isinstance(probes, list):
        probes = []
    if not isinstance(elite_traces, list):
        elite_traces = []
    if not isinstance(forbidden, list):
        forbidden = []
    if not isinstance(contract_guard_tokens, set):
        contract_guard_tokens = (
            set(contract_guard_tokens)
            if isinstance(contract_guard_tokens, (list, tuple))
            else set()
        )

    selection_mode = str(
        selection_profile.get("selection_mode", _selection_mode_from_config(config))
        if selection_profile
        else _selection_mode_from_config(config)
    )
    return {
        "selection_mode": selection_mode,
        "feature_flags": {
            "parallax": bool(getattr(config, "enable_parallax", False)),
            "orthogonality": bool(getattr(config, "enable_orthogonality", False)),
            "rotation": bool(getattr(config, "enable_rotation", False)),
            "unified": bool(getattr(config, "enable_unified", False)),
            "curriculum": bool(getattr(config, "enable_curriculum", False)),
            "primitive_contracts": bool(
                getattr(config, "enable_primitive_contracts", False)
            ),
        },
        "pareto_front_count": len(pareto_front_sizes),
        "pareto_front_sizes": pareto_front_sizes,
        "probe_count": len(probes),
        "elite_count": len(elite_traces),
        "rotation_signal": round(
            float(selection_profile.get("rotation_market_signal", 0.0) or 0.0), 3
        ),
        "rotation_strength": round(
            float(selection_profile.get("rotation_strength", 0.0) or 0.0), 3
        ),
        "forbidden_subsequence_count": len(forbidden),
        "forbidden_memory_count": int(forbidden_memory.get("pattern_count", 0) or 0),
        "contract_guard_token_count": len(contract_guard_tokens),
        "task_count": len(benchmark_result.get("task_pool", []))
        if isinstance(benchmark_result.get("task_pool", []), list)
        else 0,
    }


def _build_rotation_analysis(landscape_profile: dict[str, Any]) -> dict[str, Any]:
    rotation = (
        landscape_profile.get("rotation", {})
        if isinstance(landscape_profile, dict)
        else {}
    )
    principal_axis = (
        rotation.get("principal_axis", []) if isinstance(rotation, dict) else []
    )
    secondary_axis = (
        rotation.get("secondary_axis", []) if isinstance(rotation, dict) else []
    )
    return {
        "sample_size": int(landscape_profile.get("sample_size", 0) or 0)
        if isinstance(landscape_profile, dict)
        else 0,
        "axis_labels": list(landscape_profile.get("axis_labels", []))
        if isinstance(landscape_profile.get("axis_labels", []), list)
        else [],
        "dominant_axis": str(landscape_profile.get("dominant_axis", "") or ""),
        "condition_number": round(
            float(rotation.get("condition_number", 0.0) or 0.0), 3
        ),
        "principal_eigenvalue": round(
            float(rotation.get("principal_eigenvalue", 0.0) or 0.0), 3
        ),
        "secondary_eigenvalue": round(
            float(rotation.get("secondary_eigenvalue", 0.0) or 0.0), 3
        ),
        "principal_axis": principal_axis if isinstance(principal_axis, list) else [],
        "secondary_axis": secondary_axis if isinstance(secondary_axis, list) else [],
        "axis_variance": landscape_profile.get("axis_variance", {})
        if isinstance(landscape_profile.get("axis_variance", {}), dict)
        else {},
        "axis_range": landscape_profile.get("axis_range", {})
        if isinstance(landscape_profile.get("axis_range", {}), dict)
        else {},
    }


def _build_parallax_telemetry(
    selection_profile: dict[str, Any] | None,
    benchmark_result: dict[str, Any],
    forbidden_memory: dict[str, Any],
) -> dict[str, Any]:
    selection_profile = selection_profile or {}
    landscape_profile = benchmark_result.get("landscape_profile", {})
    parallax = (
        landscape_profile.get("parallax", {})
        if isinstance(landscape_profile, dict)
        else {}
    )
    gradient = (
        selection_profile.get("gradient", {})
        if isinstance(selection_profile, dict)
        else {}
    )
    probes = (
        selection_profile.get("probes", [])
        if isinstance(selection_profile, dict)
        else []
    )
    forbidden = (
        selection_profile.get("forbidden_subsequences", [])
        if isinstance(selection_profile, dict)
        else []
    )
    if not isinstance(probes, list):
        probes = []
    if not isinstance(forbidden, list):
        forbidden = []
    sample_size = (
        int(landscape_profile.get("sample_size", 0) or 0)
        if isinstance(landscape_profile, dict)
        else 0
    )
    top_governed = float(parallax.get("top_governed_mean", 0.0) or 0.0)
    bottom_governed = float(parallax.get("bottom_governed_mean", 0.0) or 0.0)
    gradient_steepness = float(gradient.get("contrast", 0.0) or 0.0)
    valley_depth = top_governed - bottom_governed
    return {
        "pressure": round(float(parallax.get("pressure", 0.0) or 0.0), 3),
        "top_governed_mean": round(top_governed, 3),
        "bottom_governed_mean": round(bottom_governed, 3),
        "valley_depth": round(valley_depth, 3),
        "gradient_steepness": round(gradient_steepness, 3),
        "gradient_positive_tokens": list(gradient.get("positive_tokens", []))
        if isinstance(gradient.get("positive_tokens", []), list)
        else [],
        "gradient_negative_tokens": list(gradient.get("negative_tokens", []))
        if isinstance(gradient.get("negative_tokens", []), list)
        else [],
        "failure_probe_count": len(probes),
        "failure_probe_ratio": round(len(probes) / max(1, sample_size), 3),
        "forbidden_subsequence_count": len(forbidden),
        "forbidden_subsequence_samples": [list(pattern) for pattern in forbidden[:6]],
        "memory_forbidden_count": int(forbidden_memory.get("pattern_count", 0) or 0),
        "memory_forbidden_samples": [
            list(pattern) for pattern in forbidden_memory.get("top_patterns", [])[:6]
        ]
        if isinstance(forbidden_memory.get("top_patterns", []), list)
        else [],
    }


def run_v311(
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
    seed0: int = 100901,
    feed: dict[str, Any] | None = None,
    state_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    task_shape_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed0)
    task_pool = derive_task_ecology(feed) if feed is not None else TASKS
    feed_prior_dna = derive_feed_dna_prior(task_pool) if feed is not None else []
    evolution_config = _coerce_evolution_config(config)
    curriculum_profile = build_curriculum_profile(task_pool)
    curriculum_task_pool = (
        curriculum_profile["ordered_tasks"]
        if evolution_config.enable_curriculum
        else task_pool
    )
    state_root = resolve_state_root(state_dir)
    persisted_macros = load_macro_library(state_root)
    persisted_motifs = load_motif_cache(state_root)
    forbidden_subsequence_memory = load_forbidden_subsequence_memory(state_root)
    learned_macros: dict[str, list[str]] = dict(persisted_macros)
    state_fingerprint = benchmark_state_fingerprint(
        persisted_macros,
        persisted_motifs,
        forbidden_subsequence_memory,
    )
    benchmark_inputs = {
        "trials": trials,
        "population": population,
        "base_episodes": base_episodes,
        "generations": generations,
        "seed0": seed0,
        "feed": feed,
        "semantic_engine_version": SEMANTIC_ENGINE_VERSION,
        "evolution_config": {
            key: getattr(evolution_config, key)
            for key in evolution_config.__dict__.keys()
        },
    }
    benchmark_signature = benchmark_run_signature(benchmark_inputs, state_fingerprint)
    benchmark_result = load_benchmark_run(state_root, benchmark_signature) or None
    benchmark_reused = benchmark_result is not None
    selection_profile: dict[str, Any] | None = None
    resolved_task_shape_profile = normalize_task_shape_profile(
        task_shape_profile,
        {
            "task_pool": task_pool,
            "feed_prior_dna": feed_prior_dna,
            "curriculum_profile": curriculum_profile,
            "mode": "benchmark",
        },
    )
    outcome_memory_for_mining = load_outcome_memory(
        state_root,
        resolved_task_shape_profile,
        benchmark_signature,
    )
    bootstrap_macro_lib = dict(persisted_macros)
    bootstrap = [
        random_individual(rng, bootstrap_macro_lib, f"boot{i}")
        for i in range(population)
    ]
    if feed_prior_dna:
        bootstrap = (
            seed_population_from_prior(
                rng, population, bootstrap_macro_lib, "boot_prior", feed_prior_dna
            )
            + bootstrap
        )
        bootstrap = bootstrap[:population]
    semantic_elite_traces: list[list[str]] = []
    landscape_source: list[dict[str, Any]] = []
    semantic_counterexamples: list[dict[str, Any]] = []
    held_out_credit: dict[str, Any] = {}
    prototype_lifecycle: list[dict[str, Any]] = []
    primitive_contracts: list[dict[str, Any]] = []
    devolution_candidates: dict[str, list[tuple[float, str]]] = {}
    if not benchmark_reused:
        boot_eval = ecology_round(
            bootstrap,
            bootstrap_macro_lib,
            base_episodes,
            seed0 + 1,
            task_pool=curriculum_task_pool,
        )
        boot_successful_traces = _successful_benchmark_traces(
            boot_eval, bootstrap_macro_lib
        )
        learned_macros = mine_macros_from_elites(
            boot_successful_traces,
            8,
            semantic_context=task_pool,
            macro_lib=bootstrap_macro_lib,
            outcome_memory=outcome_memory_for_mining,
        )
        semantic_promotions = build_semantic_promotion_bank(
            task_pool,
            boot_successful_traces,
            learned_macros,
        )
        if persisted_macros:
            learned_macros = {**persisted_macros, **learned_macros}

        semantic_counterexamples = build_semantic_counterexample_bank(task_pool)
        held_out_credit = held_out_semantic_credit(task_pool, learned_macros)
        primitive_contracts = build_primitive_contract_bank(
            task_pool,
            semantic_promotions,
            None,
            held_out_credit,
        )

        winners = []
        archive_raw = []
        archive_stable = []
        motif_archive = dict(persisted_motifs)
        motif_market: dict[str, Any] = {}
        lineage_mut_scale: dict[str, float] = defaultdict(lambda: 1.0)
        replication_burst_count = 0
        replication_offspring_count = 0
        controller_trace = []
        phase_trace = []
        transition_events = []

        for t in range(trials):
            rng = random.Random(seed0 + 100 + t)
            pop = [
                random_individual(rng, learned_macros, f"t{t}_g0_i{i}")
                for i in range(population)
            ]
            if feed_prior_dna:
                seeded_pop = seed_population_from_prior(
                    rng, population, learned_macros, f"t{t}_g0_prior", feed_prior_dna
                )
                for index, seeded in enumerate(seeded_pop):
                    if index < len(pop):
                        pop[index] = seeded
            controller = ControllerState(temperature=0.62, regime="balanced")
            episodes = base_episodes
            pop = ecology_round(
                pop,
                learned_macros,
                episodes,
                seed0 + 200 + t,
                task_pool=curriculum_task_pool,
            )
            compute_governed_fitness(pop)

            for g in range(generations):
                pop_metrics = summarize(pop)
                rotation_signal = pop_metrics["rotation_market_signal"]
                rotation_scale = _rotation_selection_scale(rotation_signal)
                prev_regime = controller.regime
                controller, episodes, loopback_triggered, reasons = decide_controller(
                    controller, pop_metrics, 12, 30, g
                )
                current_exploit_score = exploit_score(pop_metrics, controller)
                current_exploit_barrier = exploit_entry_barrier(pop_metrics, controller)
                current_momentum = rolling_momentum(controller)
                pressure = episodes * population

                if controller.regime != prev_regime or loopback_triggered:
                    transition_events.append(
                        {
                            "trial": t,
                            "generation": g,
                            "from_regime": prev_regime,
                            "to_regime": controller.regime,
                            "temperature": round(controller.temperature, 3),
                            "episodes": episodes,
                            "phase_hold": controller.phase_hold,
                            "loopback_triggered": loopback_triggered,
                            "exploit_score": round(current_exploit_score, 3),
                            "exploit_barrier": round(current_exploit_barrier, 3),
                            "exploit_momentum": round(current_momentum, 3),
                            "reasons": reasons[:],
                            "governed_mean": pop_metrics["governed_mean"],
                            "task_threshold_pass_mean": pop_metrics[
                                "task_threshold_pass_mean"
                            ],
                            "economic_stability_mean": pop_metrics[
                                "economic_stability_mean"
                            ],
                            "specialization_entropy": pop_metrics[
                                "specialization_entropy"
                            ],
                        }
                    )

                controller_trace.append(
                    {
                        "trial": t,
                        "generation": g,
                        "episodes": episodes,
                        "evaluation_pressure": round(pressure, 3),
                        "temperature": round(controller.temperature, 3),
                        "regime": controller.regime,
                        "phase_hold": controller.phase_hold,
                        "replication_hold": controller.replication_hold,
                        "loopback_triggered": loopback_triggered,
                        "exploit_score": round(current_exploit_score, 3),
                        "exploit_barrier": round(current_exploit_barrier, 3),
                        "exploit_momentum": round(current_momentum, 3),
                        "reasons": reasons,
                        "task_threshold_pass_mean": pop_metrics[
                            "task_threshold_pass_mean"
                        ],
                        "profit_mean": pop_metrics["profit_mean"],
                        "stability_score_mean": pop_metrics["stability_score_mean"],
                        "economic_stability_mean": pop_metrics[
                            "economic_stability_mean"
                        ],
                        "governance_variance": pop_metrics["governance_variance"],
                        "motif_entropy": pop_metrics["motif_entropy"],
                        "specialization_entropy": pop_metrics["specialization_entropy"],
                        "lineage_entropy": pop_metrics["lineage_entropy"],
                    }
                )

                phase_trace.append(
                    {
                        "trial": t,
                        "generation": g,
                        "episodes": episodes,
                        "evaluation_pressure": round(pressure, 3),
                        "temperature": round(controller.temperature, 3),
                        "regime": controller.regime,
                        "phase_hold": controller.phase_hold,
                        "replication_hold": controller.replication_hold,
                        "exploit_score": round(current_exploit_score, 3),
                        "exploit_barrier": round(current_exploit_barrier, 3),
                        "exploit_momentum": round(current_momentum, 3),
                        "governed_mean": pop_metrics["governed_mean"],
                        "task_threshold_pass_mean": pop_metrics[
                            "task_threshold_pass_mean"
                        ],
                        "profit_mean": pop_metrics["profit_mean"],
                        "stability_score_mean": pop_metrics["stability_score_mean"],
                        "economic_stability_mean": pop_metrics[
                            "economic_stability_mean"
                        ],
                        "motif_entropy": pop_metrics["motif_entropy"],
                        "specialization_entropy": pop_metrics["specialization_entropy"],
                        "lineage_entropy": pop_metrics["lineage_entropy"],
                    }
                )

                selection_profile = _selection_profiles(
                    pop,
                    learned_macros,
                    curriculum_task_pool,
                    evolution_config,
                    forbidden_subsequence_memory.get("top_patterns", [])
                    if isinstance(forbidden_subsequence_memory, dict)
                    else [],
                )
                governors = selection_profile["governors"]
                explorers = selection_profile["explorers"]
                probe_pool = selection_profile["probes"]
                parallax_gradient = selection_profile["gradient"]
                forbidden_subsequences = selection_profile["forbidden_subsequences"]
                contract_guard_tokens = selection_profile["contract_guard_tokens"]

                update_motif_market(motif_market, governors)

                for elite in governors[: max(8, population // 10)]:
                    if (
                        elite.task_threshold_pass >= 0.70
                        and elite.stability_score >= 0.60
                    ):
                        key = elite.motif_sig
                        slot = motif_archive.setdefault(
                            key,
                            {
                                "count": 0,
                                "mean_stability": 0.0,
                                "best_governed_fitness": -1e9,
                                "specializations": Counter(),
                            },
                        )
                        slot["count"] += 1
                        slot["mean_stability"] += (
                            elite.stability_score - slot["mean_stability"]
                        ) / slot["count"]
                        slot["best_governed_fitness"] = max(
                            slot["best_governed_fitness"], elite.governed_fitness
                        )
                        slot["specializations"][elite.specialization] = (
                            slot["specializations"].get(elite.specialization, 0) + 1
                        )

                archive_raw.extend(
                    sorted(pop, key=lambda x: x.fitness, reverse=True)[:2]
                )
                archive_stable.extend(
                    sorted(pop, key=_selection_score_value, reverse=True)[:2]
                )

                elite_keep = max(4, population // 16)
                carry = [
                    e
                    for e in governors
                    if e.task_threshold_pass >= 0.68 and e.stability_score >= 0.60
                ][:elite_keep]
                if len(carry) < elite_keep:
                    carry = governors[:elite_keep]

                next_pop = [
                    Individual(
                        e.dna[:], e.chart, f"{e.lineage_id}:carry", [e.lineage_id]
                    )
                    for e in carry
                ]
                for e in governors[: max(2, elite_keep // 2)]:
                    if len(next_pop) < population:
                        next_pop.append(
                            Individual(
                                e.dna[:],
                                e.chart,
                                f"{e.lineage_id}:elite",
                                [e.lineage_id],
                            )
                        )

                if evolution_config.enable_parallax and probe_pool:
                    probe_seed_count = min(len(probe_pool), max(2, population // 8))
                    for probe_index, probe in enumerate(probe_pool[:probe_seed_count]):
                        if len(next_pop) >= population:
                            break
                        probe_lineage = f"t{t}_g{g + 1}_probe{len(next_pop)}"
                        if probe_index % 2 == 0:
                            probe_child = adversarial_push(
                                probe,
                                rng,
                                learned_macros,
                                probe_lineage,
                                parallax_gradient.get("gradient", {}),
                            )
                        else:
                            probe_child = scout_teleport(
                                probe,
                                rng,
                                learned_macros,
                                probe_lineage,
                                evolution_config.scout_replace_fraction_min,
                                evolution_config.scout_replace_fraction_max,
                            )
                        next_pop.append(probe_child)

                spec_counts = Counter(x.specialization for x in governors)
                targets = niche_target_counts(population)

                replication_candidates = []
                if evolution_config.enable_parallax:
                    replication_candidates = [
                        scout
                        for scout in sorted(
                            pop, key=_selection_score_value, reverse=True
                        )
                        if getattr(scout, "parallax_role", "member") == "scout"
                        and boost_for_missing_niche(
                            scout.specialization, spec_counts, targets
                        )
                        > 1.0
                        and scout.selection_score >= pop_metrics["selection_score_mean"]
                        and scout.task_threshold_pass
                        >= max(0.68, pop_metrics["task_threshold_pass_mean"])
                        and scout.solve_rate_test
                        >= max(0.68, pop_metrics["solve_rate_test_mean"])
                        and scout.stability_score
                        >= max(0.58, pop_metrics["stability_score_mean"])
                    ]

                if (
                    replication_candidates
                    and len(next_pop) < population
                    and controller.replication_hold == 0
                ):
                    scout = replication_candidates[0]
                    scout.replication_gene = True
                    scout.replication_origin = getattr(
                        scout, "scout_origin", scout.lineage_id
                    )
                    scout.replication_depth = getattr(scout, "replication_depth", 0) + 1
                    replicas = replicate_scout(
                        scout,
                        rng,
                        learned_macros,
                        f"t{t}_g{g + 1}_rep{len(next_pop)}",
                        burst_size=2,
                    )
                    replication_burst_count += 1
                    replication_offspring_count += len(replicas)
                    controller.replication_hold = 2
                    for replica in replicas:
                        if len(next_pop) >= population:
                            break
                        next_pop.append(replica)

                while len(next_pop) < population:
                    regime = controller.regime
                    if regime == "explore":
                        cx_prob, reinforce_prob = 0.46, 0.08
                    elif regime == "balanced":
                        cx_prob, reinforce_prob = 0.62, 0.12
                    else:
                        if controller.exploit_dwell_total >= 3:
                            cx_prob, reinforce_prob = 0.86, 0.24
                        else:
                            cx_prob, reinforce_prob = 0.83, 0.20

                    if rng.random() < cx_prob:
                        p1 = choose_parent(
                            explorers,
                            rng,
                            spec_counts,
                            targets,
                            evolution_config.enable_rotation,
                            rotation_signal,
                        )
                        cand = rng.sample(explorers, min(10, len(explorers)))
                        p2 = max(
                            cand,
                            key=lambda x: (
                                _selection_score_value(x)
                                + 5.0
                                * specialization_pair_bonus(
                                    p1.specialization, x.specialization
                                )
                                + (
                                    rotation_scale
                                    * 5.0
                                    * boost_for_missing_niche(
                                        x.specialization, spec_counts, targets
                                    )
                                    if evolution_config.enable_rotation
                                    else 0.0
                                )
                                + rotation_scale
                                * 6.0
                                * (x.rotated_fitness[0] if x.rotated_fitness else 0.0)
                            ),
                        )
                        child = crossover(
                            p1, p2, rng, f"t{t}_g{g + 1}_cx{len(next_pop)}"
                        )
                    else:
                        p = choose_parent(
                            explorers,
                            rng,
                            spec_counts,
                            targets,
                            evolution_config.enable_rotation,
                            rotation_signal,
                        )
                        intensity = lineage_mut_scale[p.lineage_id.split(":")[0]]
                        if regime == "explore":
                            intensity *= 1.08
                        elif regime == "exploit":
                            intensity *= (
                                0.76 if controller.exploit_dwell_total >= 3 else 0.82
                            )
                        protected_tokens = build_protected_tokens(
                            motif_archive, p.specialization
                        ) | set(contract_guard_tokens)
                        child = mutate(
                            p,
                            rng,
                            learned_macros,
                            f"t{t}_g{g + 1}_mu{len(next_pop)}",
                            intensity,
                            protected_tokens,
                            forbidden_subsequences=forbidden_subsequences,
                        )

                    child_specialization = infer_specialization(
                        flatten_trace(child.dna, learned_macros)
                    )
                    candidates = motif_reinforcement_candidates(
                        motif_market, child_specialization
                    )
                    if candidates and rng.random() < reinforce_prob:
                        _, chosen = candidates[0]
                        block = chosen.split("|")
                        if len(child.dna) >= len(block):
                            pos = rng.randrange(
                                0, max(1, len(child.dna) - len(block) + 1)
                            )
                            child.dna[pos : pos + len(block)] = block[:]

                    if rng.random() < 0.18:
                        scored_targets = {
                            k: boost_for_missing_niche(k, spec_counts, targets)
                            for k in ("summary", "patch", "plan", "generalist")
                        }
                        force_role = max(
                            scored_targets, key=lambda k: scored_targets.get(k, 0.0)
                        )
                        if (
                            force_role != "generalist"
                            and scored_targets[force_role] > 1.10
                        ):
                            if force_role == "summary":
                                child.dna += [
                                    "VERIFY",
                                    "REPAIR",
                                    "SUMMARIZE",
                                    "WRITE_SUMMARY",
                                ]
                            elif force_role == "patch":
                                child.dna += [
                                    "VERIFY",
                                    "ROLLBACK",
                                    "REPAIR",
                                    "WRITE_PATCH",
                                ]
                            elif force_role == "plan":
                                child.dna += ["PLAN", "REPAIR", "VERIFY", "WRITE_PLAN"]
                            child.dna = child.dna[:36]

                    next_pop.append(child)

                pop = ecology_round(
                    next_pop,
                    learned_macros,
                    episodes,
                    seed0 + 300 + t * 100 + g,
                    task_pool=task_pool,
                )
                compute_governed_fitness(pop)

                roots = defaultdict(list)
                for x in pop:
                    roots[x.lineage_id.split(":")[0]].append(x)
                for root, vals in roots.items():
                    mean_stab = statistics.mean(v.stability_score for v in vals)
                    mean_thr = statistics.mean(v.task_threshold_pass for v in vals)
                    mean_profit = statistics.mean(v.profit for v in vals)
                    mean_econ = statistics.mean(v.economic_stability for v in vals)
                    if mean_stab > 0.78 and mean_thr > 0.84 and mean_econ > 0.56:
                        lineage_mut_scale[root] = 0.78
                    elif mean_thr < 0.54 or mean_profit < -10 or mean_econ < 0.28:
                        lineage_mut_scale[root] = 1.22
                    else:
                        lineage_mut_scale[root] = 1.0

            pop.sort(key=_selection_score_value, reverse=True)
            winners.append(pop[0])

            refreshed_successful_traces = _successful_benchmark_traces(
                pop, learned_macros
            )
            refreshed_macros = mine_macros_from_elites(
                refreshed_successful_traces,
                8,
                semantic_context=task_pool,
                macro_lib=learned_macros,
                outcome_memory=outcome_memory_for_mining,
            )
            if refreshed_macros:
                learned_macros = {**learned_macros, **refreshed_macros}
            semantic_promotions = build_semantic_promotion_bank(
                task_pool,
                refreshed_successful_traces,
                learned_macros,
            )

        semantic_elite_traces = refreshed_successful_traces or boot_successful_traces
        landscape_source = winners[:]
        devolution_candidates = {
            specialization: anti_reinforcement_candidates(motif_market, specialization)
            for specialization in sorted(
                set(Counter(x.specialization for x in winners)) | {"generalist"}
            )
        }
        prototype_lifecycle = semantic_prototype_lifecycle(
            task_pool,
            semantic_elite_traces,
            learned_macros,
            semantic_counterexamples,
        )
        primitive_contracts = build_primitive_contract_bank(
            task_pool,
            semantic_promotions,
            prototype_lifecycle,
            held_out_credit,
        )

        compute_governed_fitness(winners)
        archive_raw = sorted(archive_raw, key=lambda x: x.fitness, reverse=True)[:12]
        archive_stable = sorted(
            archive_stable, key=_selection_score_value, reverse=True
        )[:12]

        if state_root is not None:
            save_discoverables(
                state_root,
                learned_macros,
                motif_archive,
                metadata={
                    "task_count": len(task_pool),
                    "feed_enabled": feed is not None,
                    "semantic_rule_count": len(semantic_promotions),
                    "semantic_counterexample_count": len(semantic_counterexamples),
                    "primitive_contract_count": len(primitive_contracts),
                    "curriculum_stage_counts": curriculum_profile.get(
                        "stage_counts", {}
                    ),
                },
            )
            save_lineage_archive(
                state_root, "winners", winners, metadata={"trial_count": trials}
            )
            save_lineage_archive(
                state_root, "archive_raw", archive_raw, metadata={"trial_count": trials}
            )
            save_lineage_archive(
                state_root,
                "archive_stable",
                archive_stable,
                metadata={"trial_count": trials},
            )

        benchmark_result = {
            "mode": "miniops_crisis_ecology_v3_11_exploit_residency",
            "release_features": [
                "thermodynamic_cycle_closure",
                "momentum_gated_exploit_basin",
                "exploit_hysteresis",
                "exploit_retention_bonus",
                "directed_loopback_escape",
                "transition_event_logging",
                "exploit_residency_stabilization",
                "exploit_residency_bias",
                "patch_skeleton_emission",
                "semantic_probe_checks",
                "counterexample_bank",
                "prototype_lifecycle",
                "held_out_credit_assignment",
                "landscape_profile",
                "feature_flags",
                "pareto_selection",
                "multi_axis_selection",
                "curriculum_schedule",
                "primitive_contract_bank",
                "parallax_probe_selection",
                "rotation_axis_selection",
                "forbidden_subsequence_mutation",
            ],
            "reference_regime": {
                "population": population,
                "base_episodes": base_episodes,
            },
            "task_pool": task_pool,
            "curriculum_profile": curriculum_profile,
            "feed_prior_dna": feed_prior_dna,
            "patch_skeleton": build_patch_skeleton(task_pool, feed_prior_dna)
            if feed is not None
            else None,
            "feed_profile": {
                "feed_enabled": feed is not None,
                "fragment_count": len(_collect_text_fragments(feed))
                if feed is not None
                else 0,
                "task_count": len(task_pool),
                "focus_counts": dict(
                    Counter(task.get("focus", "planning") for task in task_pool)
                ),
            },
            "semantic_promotions": semantic_promotions,
            "semantic_counterexamples": semantic_counterexamples,
            "held_out_semantic_credit": held_out_semantic_credit(
                task_pool, learned_macros
            ),
            "prototype_lifecycle": semantic_prototype_lifecycle(
                task_pool,
                semantic_elite_traces,
                learned_macros,
                semantic_counterexamples,
            ),
            "primitive_contracts": primitive_contracts,
            "devolution_candidates": devolution_candidates,
            "landscape_profile": population_landscape_profile(landscape_source),
            "learned_macros": learned_macros,
            "summary": summarize(winners),
            "best_governed": best_payload(max(winners, key=_selection_score_value)),
            "best_raw": best_payload(max(winners, key=lambda x: x.fitness)),
            "archive_stable_top3": [best_payload(x) for x in archive_stable[:3]],
            "archive_raw_top3": [best_payload(x) for x in archive_raw[:3]],
            "winner_specialization_counts": dict(
                Counter(x.specialization for x in winners)
            ),
            "winner_family_counts": dict(Counter(x.family_sig for x in winners)),
            "winner_motif_counts": dict(Counter(x.motif_sig for x in winners)),
            "parallax_replication_bursts": replication_burst_count,
            "parallax_replication_offspring": replication_offspring_count,
            "motif_archive": {
                k: {
                    "count": v["count"],
                    "mean_stability": round(v["mean_stability"], 3),
                    "best_governed_fitness": round(v["best_governed_fitness"], 3),
                    "specializations": dict(v["specializations"]),
                }
                for k, v in motif_archive.items()
            },
            "controller_trace_head": controller_trace[:120],
            "controller_summary": {
                "episodes_min": min(x["episodes"] for x in controller_trace),
                "episodes_max": max(x["episodes"] for x in controller_trace),
                "temperature_min": round(
                    min(x["temperature"] for x in controller_trace), 3
                ),
                "temperature_max": round(
                    max(x["temperature"] for x in controller_trace), 3
                ),
                "unique_episode_levels": sorted(
                    set(x["episodes"] for x in controller_trace)
                ),
                "regime_counts": dict(Counter(x["regime"] for x in controller_trace)),
                "transition_count": len(transition_events),
                "loopback_count": sum(
                    1 for x in transition_events if x["loopback_triggered"]
                ),
                "exploit_entries": sum(
                    1 for x in transition_events if x["to_regime"] == "exploit"
                ),
                "exploit_dwell": exploit_dwell_stats(controller_trace),
                "exploit_bounce_count": sum(
                    1
                    for x in transition_events
                    if x["from_regime"] == "exploit" and x["to_regime"] == "balanced"
                ),
                "replication_hold_final": controller.replication_hold,
            },
            "phase_trace_head": phase_trace[:120],
            "phase_summary": {
                "best_governed": max(phase_trace, key=lambda x: x["governed_mean"])
                if phase_trace
                else None,
                "best_threshold": max(
                    phase_trace, key=lambda x: x["task_threshold_pass_mean"]
                )
                if phase_trace
                else None,
                "best_profit": max(phase_trace, key=lambda x: x["profit_mean"])
                if phase_trace
                else None,
                "regime_counts": dict(Counter(x["regime"] for x in phase_trace)),
                "transition_count": len(transition_events),
            },
            "transition_events_head": transition_events[:60],
            "lineage_mutation_scale": {
                k: round(v, 3) for k, v in dict(lineage_mut_scale).items()
            },
        }

    assert benchmark_result is not None
    semantic_promotions = benchmark_result.get(
        "semantic_promotions"
    ) or build_semantic_promotion_bank(
        task_pool,
        semantic_elite_traces,
        benchmark_result.get("learned_macros", learned_macros),
    )

    if not semantic_counterexamples:
        semantic_counterexamples = build_semantic_counterexample_bank(task_pool)
    if not held_out_credit:
        held_out_credit = held_out_semantic_credit(
            task_pool, benchmark_result.get("learned_macros", learned_macros)
        )
    if not semantic_elite_traces:
        semantic_elite_traces = _elite_traces_from_result(benchmark_result)
    if not landscape_source:
        landscape_source = _landscape_population_from_result(benchmark_result)
    if not prototype_lifecycle:
        prototype_lifecycle = semantic_prototype_lifecycle(
            task_pool,
            semantic_elite_traces,
            benchmark_result.get("learned_macros", learned_macros),
            semantic_counterexamples,
        )
    if not primitive_contracts:
        primitive_contracts = build_primitive_contract_bank(
            task_pool,
            semantic_promotions,
            prototype_lifecycle,
            held_out_credit,
        )
    benchmark_result.update(
        {
            "semantic_promotions": semantic_promotions,
            "semantic_counterexamples": semantic_counterexamples,
            "held_out_semantic_credit": held_out_credit,
            "prototype_lifecycle": prototype_lifecycle,
            "primitive_contracts": primitive_contracts,
            "curriculum_profile": curriculum_profile,
            "devolution_candidates": devolution_candidates
            or {
                specialization: anti_reinforcement_candidates(
                    benchmark_result.get("motif_archive", {}), specialization
                )
                for specialization in sorted(
                    set(benchmark_result.get("winner_specialization_counts", {}).keys())
                    | {"generalist", "patch", "summary", "plan"}
                )
            },
            "landscape_profile": population_landscape_profile(landscape_source),
            "summary": benchmark_result.get("summary")
            if benchmark_reused
            else summarize(landscape_source),
        }
    )

    current_forbidden_subsequences: list[tuple[str, ...]] = []
    if isinstance(selection_profile, dict):
        current_forbidden_subsequences = [
            tuple(str(token) for token in pattern if str(token))
            for pattern in selection_profile.get("forbidden_subsequences", [])
            if isinstance(pattern, (list, tuple)) and len(pattern) >= 2
        ]

    forbidden_memory_report = forbidden_subsequence_memory
    if (
        state_root is not None
        and current_forbidden_subsequences
        and not benchmark_reused
    ):
        selection_mode = str(
            selection_profile.get("selection_mode", "unified")
            if isinstance(selection_profile, dict)
            else "unified"
        )
        gradient = (
            selection_profile.get("gradient", {})
            if isinstance(selection_profile, dict)
            else {}
        )
        if not isinstance(gradient, dict):
            gradient = {}
        save_forbidden_subsequence_memory(
            state_root,
            current_forbidden_subsequences,
            metadata={
                "benchmark_signature": benchmark_signature,
                "task_shape_signature": task_shape_signature(
                    resolved_task_shape_profile
                ),
                "selection_mode": selection_mode,
                "parallax_pressure": round(
                    float(
                        benchmark_result.get("summary", {}).get(
                            "parallax_pressure", 0.0
                        )
                        or 0.0
                    ),
                    3,
                ),
            },
            score=round(float(gradient.get("contrast", 0.0) or 0.0), 3),
        )
        forbidden_memory_report = load_forbidden_subsequence_memory(state_root)

    selection_report = _build_selection_report(
        selection_profile,
        benchmark_result,
        evolution_config,
        forbidden_memory_report,
    )
    rotation_analysis = _build_rotation_analysis(
        benchmark_result.get("landscape_profile", {})
    )
    parallax_telemetry = _build_parallax_telemetry(
        selection_profile,
        benchmark_result,
        forbidden_memory_report,
    )
    benchmark_result.update(
        {
            "selection_report": selection_report,
            "rotation_analysis": rotation_analysis,
            "parallax_telemetry": parallax_telemetry,
            "forbidden_subsequence_memory": forbidden_memory_report,
        }
    )

    outcome_record = build_outcome_record(
        benchmark_result,
        benchmark_signature,
        resolved_task_shape_profile,
        metadata={
            "trials": trials,
            "population": population,
            "state_dir": str(state_root) if state_root is not None else "",
            "state_fingerprint": state_fingerprint,
            "benchmark_reused": benchmark_reused,
        },
        source="benchmark_reused" if benchmark_reused else "benchmark",
    )
    benchmark_result["task_shape_signature"] = outcome_record["task_shape_signature"]
    benchmark_result["outcome_record"] = outcome_record

    if state_root is not None:
        save_outcome_ledger(state_root, outcome_record)
        save_run_summary(
            state_root,
            benchmark_result,
            metadata={
                "trials": trials,
                "population": population,
                "state_dir": str(state_root),
                "benchmark_reused": benchmark_reused,
            },
        )
        save_benchmark_run(
            state_root,
            benchmark_signature,
            benchmark_result,
            metadata={
                "trials": trials,
                "population": population,
                "state_dir": str(state_root) if state_root is not None else "",
                "state_fingerprint": state_fingerprint,
            },
            reused=benchmark_reused,
        )

    benchmark_result["outcome_memory"] = load_outcome_memory(
        state_root,
        resolved_task_shape_profile,
        benchmark_signature,
    )

    result = benchmark_result
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--population", type=int, default=96)
    ap.add_argument("--base-episodes", type=int, default=18)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100901)
    ap.add_argument("--feed-file", type=str, default="")
    ap.add_argument("--feed-json", type=str, default="")
    ap.add_argument("--state-dir", type=str, default="")
    ap.add_argument(
        "--out", type=str, default="miniops_crisis_ecology_v3_11_results.json"
    )
    args = ap.parse_args()

    feed = None
    if args.feed_json:
        feed = json.loads(args.feed_json)
    elif args.feed_file:
        with open(args.feed_file, "r", encoding="utf-8") as f:
            feed = json.load(f)

    result = run_v311(
        trials=args.trials,
        population=args.population,
        base_episodes=args.base_episodes,
        generations=args.generations,
        seed0=args.seed,
        feed=feed,
        state_dir=args.state_dir or None,
    )
    payload = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


__all__ = ["main", "run_v311"]
