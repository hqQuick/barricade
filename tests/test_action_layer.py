from __future__ import annotations

import random
from pathlib import Path

from barricade import runtime
from barricade.feed_derived_dna.controller import decide_controller
from barricade.feed_derived_dna._operators import (
    adversarial_push,
    mutate,
    replicate_scout,
    scout_teleport,
)
from barricade.feed_derived_dna.models import ControllerState, Individual


def _has_forbidden(dna: list[str], subsequence: tuple[str, ...]) -> bool:
    if len(subsequence) > len(dna):
        return False
    return any(
        tuple(dna[index : index + len(subsequence)]) == subsequence
        for index in range(len(dna) - len(subsequence) + 1)
    )


def test_adversarial_push_scout_teleport_and_forbidden_subsequences() -> None:
    rng = random.Random(7)
    base = Individual(
        dna=[
            "OBSERVE",
            "PLAN",
            "WRITE_PATCH",
            "VERIFY",
            "SUMMARIZE",
            "PLAN",
            "WRITE_PATCH",
        ],
        chart="analysis",
        lineage_id="seed",
    )
    gradient = {"PLAN": 0.9, "WRITE_PATCH": 0.7, "VERIFY": -0.5, "SUMMARIZE": -0.4}

    pushed = adversarial_push(base, rng, {}, "seed:adv", gradient)
    scout = scout_teleport(base, rng, {}, "seed:scout")
    mutated = mutate(
        base,
        rng,
        {},
        "seed:mut",
        forbidden_subsequences=[("PLAN", "WRITE_PATCH", "VERIFY")],
    )

    assert pushed.parallax_role == "canary"
    assert pushed.valley_membership in {"adversarial", "adversarial_fallback"}
    assert pushed.gradient_signal["PLAN"] == 0.9
    assert scout.parallax_role == "scout"
    assert scout.scout_origin == base.lineage_id
    assert not _has_forbidden(mutated.dna, ("PLAN", "WRITE_PATCH", "VERIFY"))


def test_replicate_scout_burst_marks_descendants() -> None:
    rng = random.Random(11)
    scout = Individual(
        dna=[
            "OBSERVE",
            "PLAN",
            "WRITE_PATCH",
            "VERIFY",
            "SUMMARIZE",
            "PLAN",
            "WRITE_PATCH",
            "VERIFY",
        ],
        chart="analysis",
        lineage_id="seed:scout",
    )
    scout.parallax_role = "scout"
    scout.scout_origin = scout.lineage_id

    descendants = replicate_scout(scout, rng, {}, "seed:rep", burst_size=3)

    assert len(descendants) == 3
    assert descendants[0].dna == scout.dna
    assert any(child.dna != scout.dna for child in descendants[1:])
    assert all(child.parallax_role == "replica" for child in descendants)
    assert all(child.valley_membership == "replication" for child in descendants)
    assert all(child.replication_gene for child in descendants)
    assert all(child.scout_origin == scout.lineage_id for child in descendants)


def test_decide_controller_decrements_replication_hold() -> None:
    state = ControllerState(replication_hold=2)
    metrics = {
        "governed_mean": 100.0,
        "task_threshold_pass_mean": 0.7,
        "solve_rate_test_mean": 0.65,
        "stability_score_mean": 0.6,
        "specialization_entropy": 0.5,
        "economic_stability_mean": 0.55,
        "governance_variance": 120.0,
        "motif_entropy": 0.6,
        "lineage_entropy": 0.6,
        "profit_mean": 0.0,
        "optimizer_plan_axis_mean": 0.3,
        "optimizer_summary_axis_mean": 0.3,
        "optimizer_balance_mean": 0.4,
        "optimizer_mixed_pressure_mean": 0.2,
        "optimizer_ast_depth_mean": 1.0,
        "parallax_pressure": 0.1,
        "rotation_condition": 1.0,
        "orthogonality_spread": 0.2,
        "devolution_pressure": 0.1,
    }

    updated, _, _, reasons = decide_controller(state, metrics, 12, 30, 1)

    assert updated.replication_hold == 1
    assert "replication_hold" in reasons


def test_run_benchmark_accepts_feature_flags(tmp_path: Path) -> None:
    result = runtime.run_benchmark(
        trials=1,
        population=8,
        base_episodes=2,
        generations=1,
        seed0=5,
        feed={"ticket": "Add feature flags, curriculum staging, and primitive contracts."},
        state_dir=str(tmp_path / "state"),
        config={
            "enable_unified": False,
            "enable_parallax": False,
            "enable_orthogonality": False,
            "enable_rotation": False,
            "enable_curriculum": False,
            "enable_primitive_contracts": False,
        },
    )

    assert result["release_features"]
    assert "feature_flags" in result["release_features"]
    assert result["curriculum_profile"]["task_count"] >= 1
    assert isinstance(result["primitive_contracts"], list)
