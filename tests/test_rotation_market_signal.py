from __future__ import annotations

from random import Random

from barricade.feed_derived_dna import analysis
from barricade.feed_derived_dna._operators import specialization_pair_bonus
from barricade.feed_derived_dna.controller import phase_signal, rotation_market_signal
from barricade.feed_derived_dna.evolution import (
    _rotation_signal_from_metrics,
    _selection_profiles,
    choose_parent,
    summarize,
)
from barricade.feed_derived_dna.models import EvolutionConfig, Individual


def _build_population(rows: list[dict[str, object]]) -> list[Individual]:
    population: list[Individual] = []
    for index, row in enumerate(rows):
        specialization = str(row["specialization"])
        individual = Individual(
            dna=["PLAN", "VERIFY", "SUMMARIZE"],
            chart="analysis",
            lineage_id=f"ind_{index}",
        )
        individual.specialization = specialization
        for key, value in row.items():
            if key == "specialization":
                continue
            setattr(individual, key, value)
        population.append(individual)
    return population


def test_rotation_market_signal_accessor_reads_scalar() -> None:
    metrics = {"rotation_market_signal": 0.73}

    assert rotation_market_signal(metrics) == 0.73


def test_phase_signal_consumes_rotation_market_signal() -> None:
    base_metrics = {
        "task_threshold_pass_mean": 0.64,
        "specialization_entropy": 0.50,
        "economic_stability_mean": 0.61,
        "governance_variance": 120.0,
        "motif_entropy": 0.55,
        "lineage_entropy": 0.52,
        "stability_score_mean": 0.59,
        "optimizer_plan_axis_mean": 0.32,
        "optimizer_mixed_pressure_mean": 0.24,
        "optimizer_ast_depth_mean": 2.0,
        "parallax_pressure": 0.20,
        "rotation_condition": 1.9,
        "orthogonality_spread": 0.12,
        "devolution_pressure": 0.14,
    }

    low = dict(base_metrics, rotation_market_signal=0.10)
    high = dict(base_metrics, rotation_market_signal=0.90)

    assert phase_signal(low) > phase_signal(high)


def test_rotation_signal_prefers_diversity_collapse() -> None:
    diverse_metrics = {
        "motif_entropy": 0.94,
        "specialization_entropy": 0.91,
        "lineage_entropy": 0.88,
        "dominant_macro_ratio_mean": 0.18,
        "task_threshold_pass_mean": 0.84,
        "solve_rate_test_mean": 0.82,
        "stability_score_mean": 0.79,
        "optimizer_balance_mean": 0.74,
        "governance_variance": 420.0,
        "parallax_pressure": 0.10,
        "rotation_condition": 1.20,
    }
    collapsed_metrics = {
        "motif_entropy": 0.18,
        "specialization_entropy": 0.12,
        "lineage_entropy": 0.20,
        "dominant_macro_ratio_mean": 0.88,
        "task_threshold_pass_mean": 0.58,
        "solve_rate_test_mean": 0.52,
        "stability_score_mean": 0.46,
        "optimizer_balance_mean": 0.31,
        "governance_variance": 120.0,
        "parallax_pressure": 0.22,
        "rotation_condition": 2.10,
    }

    assert _rotation_signal_from_metrics(collapsed_metrics) > _rotation_signal_from_metrics(diverse_metrics)


def test_rotation_strength_tracks_diversity_collapse() -> None:
    strong_population = _build_population(
        [
            {
                "specialization": "plan",
                "train_score": 88.0,
                "test_score": 86.0,
                "stability_score": 0.82,
                "artifact_yield": 3.6,
                "gen_efficiency": 1.8,
                "task_threshold_pass": 0.88,
                "market_score": 8.8,
                "profit": 4.8,
                "economic_stability": 0.80,
                "fitness": 140.0,
                "governed_fitness": 128.0,
                "cognitive_stability": 0.84,
                "solve_rate_test": 0.90,
            },
            {
                "specialization": "patch",
                "train_score": 84.0,
                "test_score": 82.0,
                "stability_score": 0.78,
                "artifact_yield": 3.3,
                "gen_efficiency": 1.7,
                "task_threshold_pass": 0.84,
                "market_score": 7.9,
                "profit": 4.1,
                "economic_stability": 0.77,
                "fitness": 132.0,
                "governed_fitness": 120.0,
                "cognitive_stability": 0.80,
                "solve_rate_test": 0.86,
            },
            {
                "specialization": "summary",
                "train_score": 90.0,
                "test_score": 88.0,
                "stability_score": 0.85,
                "artifact_yield": 3.8,
                "gen_efficiency": 1.9,
                "task_threshold_pass": 0.90,
                "market_score": 9.2,
                "profit": 5.0,
                "economic_stability": 0.82,
                "fitness": 145.0,
                "governed_fitness": 134.0,
                "cognitive_stability": 0.86,
                "solve_rate_test": 0.92,
            },
            {
                "specialization": "generalist",
                "train_score": 86.0,
                "test_score": 83.0,
                "stability_score": 0.79,
                "artifact_yield": 3.1,
                "gen_efficiency": 1.6,
                "task_threshold_pass": 0.82,
                "market_score": 7.2,
                "profit": 3.9,
                "economic_stability": 0.76,
                "fitness": 128.0,
                "governed_fitness": 117.0,
                "cognitive_stability": 0.81,
                "solve_rate_test": 0.84,
            },
        ]
    )
    weak_population = _build_population(
        [
            {
                "specialization": "generalist",
                "train_score": 24.0,
                "test_score": 26.0,
                "stability_score": 0.22,
                "artifact_yield": 0.2,
                "gen_efficiency": 0.35,
                "task_threshold_pass": 0.24,
                "market_score": -7.2,
                "profit": -6.8,
                "economic_stability": 0.18,
                "fitness": 32.0,
                "governed_fitness": 18.0,
                "cognitive_stability": 0.20,
                "solve_rate_test": 0.12,
            },
            {
                "specialization": "generalist",
                "train_score": 22.0,
                "test_score": 23.0,
                "stability_score": 0.18,
                "artifact_yield": 0.1,
                "gen_efficiency": 0.30,
                "task_threshold_pass": 0.21,
                "market_score": -6.5,
                "profit": -7.1,
                "economic_stability": 0.16,
                "fitness": 28.0,
                "governed_fitness": 16.0,
                "cognitive_stability": 0.18,
                "solve_rate_test": 0.10,
            },
            {
                "specialization": "generalist",
                "train_score": 26.0,
                "test_score": 24.0,
                "stability_score": 0.20,
                "artifact_yield": 0.15,
                "gen_efficiency": 0.32,
                "task_threshold_pass": 0.22,
                "market_score": -6.9,
                "profit": -6.5,
                "economic_stability": 0.17,
                "fitness": 30.0,
                "governed_fitness": 17.0,
                "cognitive_stability": 0.19,
                "solve_rate_test": 0.11,
            },
            {
                "specialization": "generalist",
                "train_score": 25.0,
                "test_score": 25.0,
                "stability_score": 0.19,
                "artifact_yield": 0.12,
                "gen_efficiency": 0.31,
                "task_threshold_pass": 0.23,
                "market_score": -6.7,
                "profit": -6.9,
                "economic_stability": 0.15,
                "fitness": 29.0,
                "governed_fitness": 15.0,
                "cognitive_stability": 0.18,
                "solve_rate_test": 0.11,
            },
        ]
    )

    strong_summary = summarize(strong_population)
    weak_summary = summarize(weak_population)

    assert weak_summary["rotation_market_signal"] > strong_summary["rotation_market_signal"]

    config = EvolutionConfig(
        enable_parallax=False,
        enable_orthogonality=False,
        enable_rotation=True,
        enable_unified=False,
        enable_curriculum=False,
        enable_primitive_contracts=False,
    )

    strong_profile = _selection_profiles(strong_population, {}, [], config)
    weak_profile = _selection_profiles(weak_population, {}, [], config)

    assert weak_profile["rotation_market_signal"] > strong_profile["rotation_market_signal"]
    assert weak_profile["rotation_strength"] > strong_profile["rotation_strength"]


def test_expanded_rotation_basis_includes_new_axes() -> None:
    profile = analysis.population_landscape_profile(
        _build_population(
            [
                {
                    "specialization": "plan",
                    "train_score": 78.0,
                    "test_score": 83.0,
                    "solve_rate_test": 0.87,
                    "stability_score": 0.81,
                    "cognitive_stability": 0.84,
                    "economic_stability": 0.79,
                    "artifact_yield": 2.8,
                    "gen_efficiency": 1.4,
                    "task_threshold_pass": 0.85,
                    "market_score": 8.2,
                    "profit": 4.2,
                    "governed_fitness": 116.0,
                    "fitness": 132.0,
                },
                {
                    "specialization": "patch",
                    "train_score": 74.0,
                    "test_score": 77.0,
                    "solve_rate_test": 0.78,
                    "stability_score": 0.76,
                    "cognitive_stability": 0.79,
                    "economic_stability": 0.74,
                    "artifact_yield": 2.5,
                    "gen_efficiency": 1.3,
                    "task_threshold_pass": 0.79,
                    "market_score": 7.4,
                    "profit": 3.8,
                    "governed_fitness": 108.0,
                    "fitness": 124.0,
                },
                {
                    "specialization": "summary",
                    "train_score": 81.0,
                    "test_score": 85.0,
                    "solve_rate_test": 0.91,
                    "stability_score": 0.83,
                    "cognitive_stability": 0.86,
                    "economic_stability": 0.82,
                    "artifact_yield": 3.0,
                    "gen_efficiency": 1.5,
                    "task_threshold_pass": 0.89,
                    "market_score": 8.8,
                    "profit": 4.9,
                    "governed_fitness": 122.0,
                    "fitness": 138.0,
                },
            ]
        )
    )

    assert len(profile["axis_labels"]) == 9
    assert {"solve_rate_test", "cognitive_stability", "market_score"} <= set(
        profile["axis_labels"]
    )
    assert profile["rotation"]["principal_axis"]
    assert profile["rotation"]["secondary_axis"]


def test_default_evolution_config_uses_expanded_rotation_budget() -> None:
    config = EvolutionConfig()

    assert config.pareto_dimensions == 9
    assert config.selection_axis_count == 9


def test_choose_parent_prefers_market_conditioned_rotation() -> None:
    dominant = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="dominant",
    )
    dominant.governed_fitness = 104.0
    dominant.rotated_fitness = [0.10, 0.10]

    market_aligned = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="market_aligned",
    )
    market_aligned.governed_fitness = 100.0
    market_aligned.rotated_fitness = [0.90, 0.80]

    chosen = choose_parent(
        [dominant, market_aligned],
        Random(7),
        {"generalist": 1, "summary": 1},
        rotation_enabled=True,
        rotation_signal=1.0,
    )

    assert chosen is market_aligned


def test_choose_parent_falls_back_when_rotation_signal_is_low() -> None:
    dominant = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="dominant",
    )
    dominant.governed_fitness = 104.0
    dominant.rotated_fitness = [0.10, 0.10]

    market_aligned = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="market_aligned",
    )
    market_aligned.governed_fitness = 100.0
    market_aligned.rotated_fitness = [0.90, 0.80]

    chosen = choose_parent(
        [dominant, market_aligned],
        Random(7),
        {"generalist": 1, "summary": 1},
        rotation_enabled=True,
        rotation_signal=0.0,
    )

    assert chosen is dominant


def test_choose_parent_penalizes_overrepresented_summary() -> None:
    summary = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="summary",
    )
    summary.governed_fitness = 110.0
    summary.specialization = "summary"

    patch = Individual(
        dna=["PLAN", "VERIFY", "SUMMARIZE"],
        chart="analysis",
        lineage_id="patch",
    )
    patch.governed_fitness = 106.0
    patch.specialization = "patch"

    chosen = choose_parent(
        [summary, patch],
        Random(9),
        {"summary": 18, "patch": 4},
        {"summary": 12, "patch": 12},
        rotation_enabled=False,
        rotation_signal=0.0,
    )

    assert chosen is patch


def test_infer_specialization_promotes_mixed_summary_traces_to_bridge(monkeypatch) -> None:
    monkeypatch.setattr(
        analysis,
        "estimate_artifact_quality",
        lambda flat: {"plan": 1.28, "patch": 0.32, "summary": 1.62},
    )

    assert (
        analysis.infer_specialization(["WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE"])
        == "summary_bridge"
    )


def test_specialization_pair_bonus_treats_summary_bridge_as_compatible() -> None:
    assert specialization_pair_bonus("summary_bridge", "plan") == 1.0
    assert specialization_pair_bonus("summary_bridge", "generalist") == 1.0
