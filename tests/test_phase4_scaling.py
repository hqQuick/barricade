from __future__ import annotations

import json

from barricade import workflow
import barricade.scaling as scaling
from barricade.mcp_server import analyze_scaling_profile as analyze_scaling_profile_tool


def _sample_result() -> dict:
    return {
        "summary": {
            "mean": 0.82,
            "governed_mean": 0.74,
            "task_threshold_pass_mean": 0.61,
            "solve_rate_test_mean": 0.58,
            "stability_score_mean": 0.56,
            "cognitive_stability_mean": 0.60,
            "economic_stability_mean": 0.48,
            "profit_mean": -3.2,
            "market_score_mean": -1.1,
            "governance_variance": 1320.0,
            "motif_entropy": 0.41,
            "specialization_entropy": 0.36,
            "lineage_entropy": 0.44,
            "optimizer_plan_axis_mean": 0.38,
            "optimizer_patch_axis_mean": 0.29,
            "optimizer_summary_axis_mean": 0.18,
            "optimizer_verify_axis_mean": 0.22,
            "optimizer_balance_mean": 0.37,
            "optimizer_mixed_pressure_mean": 0.31,
            "optimizer_ast_depth_mean": 2.4,
        },
        "best_governed": {
            "fitness": 88.0,
            "governed_fitness": 79.0,
            "stability_score": 0.74,
            "cognitive_stability": 0.78,
            "economic_stability": 0.68,
            "profit": 1.2,
            "task_threshold_pass": 0.82,
            "solve_rate_test": 0.80,
        },
        "best_raw": {
            "fitness": 46.0,
            "governed_fitness": 31.0,
            "stability_score": 0.39,
            "cognitive_stability": 0.41,
            "economic_stability": 0.29,
            "profit": -8.2,
            "task_threshold_pass": 0.41,
            "solve_rate_test": 0.33,
        },
        "controller_summary": {
            "regime_counts": {"balanced": 80, "exploit": 20, "explore": 12},
            "transition_count": 18,
            "loopback_count": 4,
            "exploit_entries": 6,
            "exploit_bounce_count": 3,
            "episodes_min": 12,
            "episodes_max": 30,
            "temperature_min": 0.41,
            "temperature_max": 0.87,
        },
        "phase_summary": {
            "transition_count": 18,
            "regime_counts": {"balanced": 76, "exploit": 20, "explore": 16},
        },
        "transition_events_head": [
            {"generation": 0, "from_regime": "balanced", "to_regime": "exploit"},
            {"generation": 1, "from_regime": "exploit", "to_regime": "balanced"},
        ],
        "lineage_mutation_scale": {"root": 1.1},
        "winner_specialization_counts": {"generalist": 8, "patch": 2},
        "winner_family_counts": {"fam1": 6, "fam2": 4},
        "winner_motif_counts": {"motif1": 7, "motif2": 3},
    }


def _candidate_result() -> dict:
    result = _sample_result()
    result["summary"] = dict(result["summary"])
    result["summary"].update(
        {
            "governed_mean": 0.92,
            "solve_rate_test_mean": 0.72,
            "stability_score_mean": 0.67,
            "task_threshold_pass_mean": 0.78,
            "economic_stability_mean": 0.62,
            "profit_mean": 0.8,
            "market_score_mean": 0.3,
            "rotation_market_signal": 0.82,
            "governance_variance": 840.0,
            "motif_entropy": 0.53,
            "specialization_entropy": 0.51,
            "lineage_entropy": 0.55,
        }
    )
    result["best_governed"] = {
        "fitness": 96.0,
        "governed_fitness": 88.0,
        "stability_score": 0.78,
        "cognitive_stability": 0.80,
        "economic_stability": 0.72,
        "profit": 4.1,
        "task_threshold_pass": 0.84,
        "solve_rate_test": 0.83,
    }
    result["best_raw"] = {
        "fitness": 63.0,
        "governed_fitness": 52.0,
        "stability_score": 0.44,
        "cognitive_stability": 0.46,
        "economic_stability": 0.39,
        "profit": -0.4,
        "task_threshold_pass": 0.56,
        "solve_rate_test": 0.59,
    }
    result["selection_report"] = {
        "selection_mode": "unified",
        "feature_flags": {"parallax": True, "orthogonality": True, "rotation": True, "unified": True},
        "pareto_front_count": 2,
        "pareto_front_sizes": [5, 3],
        "probe_count": 6,
        "elite_count": 4,
        "rotation_signal": 0.81,
        "rotation_strength": 0.52,
        "forbidden_subsequence_count": 3,
        "forbidden_memory_count": 5,
        "contract_guard_token_count": 4,
    }
    result["rotation_analysis"] = {
        "sample_size": 12,
        "axis_labels": ["train_score", "test_score"],
        "dominant_axis": "test_score",
        "condition_number": 3.2,
        "principal_eigenvalue": 1.9,
        "secondary_eigenvalue": 0.6,
        "principal_axis": [],
        "secondary_axis": [],
        "axis_variance": {"train_score": 0.12, "test_score": 0.24},
        "axis_range": {"train_score": 0.4, "test_score": 0.7},
    }
    result["parallax_telemetry"] = {
        "pressure": 0.31,
        "top_governed_mean": 0.83,
        "bottom_governed_mean": 0.41,
        "valley_depth": 0.42,
        "gradient_steepness": 0.29,
        "gradient_positive_tokens": ["PLAN"],
        "gradient_negative_tokens": ["ROLLBACK"],
        "failure_probe_count": 6,
        "failure_probe_ratio": 0.5,
        "forbidden_subsequence_count": 3,
        "forbidden_subsequence_samples": [["PLAN", "REPAIR", "VERIFY"]],
        "memory_forbidden_count": 5,
        "memory_forbidden_samples": [["WRITE_PATCH", "VERIFY"]],
    }
    return result


def test_analyze_scaling_profile_reads_result_payload() -> None:
    diagnostics = scaling.analyze_scaling_profile(_sample_result())

    assert diagnostics["status"] == "diagnosed"
    assert diagnostics["api_version"] == scaling.API_VERSION
    assert diagnostics["snapshot"]["governed_mean"] == 0.74
    assert diagnostics["dual_objective_pressure"]["pressure_index"] > 0.25
    assert diagnostics["phase_detection"]["recommended_regime"] in {"balanced", "explore", "exploit"}
    assert diagnostics["diversity_enforcement"]["status"] == "enforce"
    assert diagnostics["risk_flags"]

    weights = diagnostics["reward_model_candidates"]["candidate_weights"]
    assert round(sum(weights.values()), 3) == 1.0


def test_analyze_scaling_profile_matches_baseline() -> None:
    diagnostics = scaling.analyze_scaling_profile(_sample_result(), baseline_or_path=_sample_result())

    assert diagnostics["baseline_comparison"]["present"] is True
    assert diagnostics["baseline_comparison"]["matched"] is True
    assert all(delta == 0.0 for delta in diagnostics["baseline_comparison"]["deltas"].values())
    assert diagnostics["api_version"] == scaling.API_VERSION


def test_benchmark_comparison_report_prefers_candidate() -> None:
    report = scaling.benchmark_comparison_report(_sample_result(), _candidate_result())

    assert report["status"] == "compared"
    assert report["winner"] == "candidate"
    assert report["score"]["delta"] > 0.0
    assert report["delta"]["solve_rate_test_mean"] > 0.0
    assert report["selection_delta"]["pareto_front_count"] >= 0
    assert report["rotation_delta"]["condition_number"] >= 0.0


def test_analyze_scaling_profile_ignores_corrupted_result_file(tmp_path) -> None:
    result_file = tmp_path / "broken_result.json"
    result_file.write_text("{broken")

    diagnostics = scaling.analyze_scaling_profile(result_file)

    assert diagnostics["status"] == "diagnosed"
    assert diagnostics["api_version"] == scaling.API_VERSION
    assert diagnostics["snapshot"]
    assert diagnostics["baseline_comparison"]["present"] is False


def test_mcp_wrapper_and_cli_entrypoint(tmp_path, monkeypatch) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "notes.txt").write_text("before\n")

    result = workflow.run_unified_workflow(
        "Patch the repository to add a unified workflow and verify it before commit.",
        context={"ticket": "P4"},
        state_dir=str(tmp_path / "state"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=7,
        dispatch_plan={
            "workspace_root": str(workspace_root),
            "updates": {"notes.txt": "after\n"},
            "verification_command": ["python3", "-c", "pass"],
        },
        workspace_root=str(workspace_root),
        commit=True,
    )
    assert result["status"] == "committed"
    assert result["execution"]["status"] == "committed"

    result_file = tmp_path / "result.json"
    out_file = tmp_path / "diagnostics.json"
    result_file.write_text(json.dumps(result))

    wrapped = analyze_scaling_profile_tool(result_file.read_text(), baseline_json=result_file.read_text())
    assert wrapped["baseline_comparison"]["matched"] is True
    assert wrapped["api_version"] == scaling.API_VERSION

    monkeypatch.setattr(
        "sys.argv",
        [
            "barricade-diagnostics",
            "--result-file",
            str(result_file),
            "--baseline-file",
            str(result_file),
            "--out",
            str(out_file),
        ],
    )
    scaling.main()

    payload = json.loads(out_file.read_text())
    assert payload["status"] == "diagnosed"
    assert payload["baseline_comparison"]["matched"] is True
    assert payload["api_version"] == scaling.API_VERSION
