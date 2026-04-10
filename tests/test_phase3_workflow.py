from __future__ import annotations

import json

import pytest

from barricade import mcp_server
import barricade.workflow as workflow
from barricade.feed_derived_dna.tasks import derive_feed_dna_prior, derive_task_ecology
from barricade.feed_derived_dna.persistence import save_execution_feedback, task_shape_signature
from barricade.mcp_server import solve_problem
from barricade.workflow_intake import task_shape_similarity


def test_build_workflow_intake_classifies_patch_request() -> None:
    problem_text = "Patch the repository to add a unified MCP workflow, test it, and verify the changes before commit."

    intake = workflow.build_workflow_intake(problem_text)

    assert intake["raw_task"] == problem_text
    assert intake["goal"] == problem_text
    assert intake["expected_artifact_type"] == "patch"
    assert "patch artifact" in intake["deliverables"]
    assert "test/verification artifact" in intake["deliverables"]
    assert "mcp" in intake["domain_tags"]
    assert intake["phase_scores"]["patching"] >= intake["phase_scores"]["planning"]
    assert intake["api_version"] == workflow.API_VERSION


def test_build_workflow_intake_routes_math_proofs_to_planning() -> None:
    problem_text = (
        "Prove that the centroid of a triangle lies on the medians and explain the geometric intuition."
    )

    intake = workflow.build_workflow_intake(problem_text)

    assert intake["task_class"] == "math"
    assert intake["route_hint"] == "planning"
    assert "math" in intake["domain_tags"]
    assert "reasoning" in intake["domain_tags"]
    assert intake["shape_centroid"]["dominant_lane"] == "reasoning"
    assert intake["phase_scores"]["planning"] > intake["phase_scores"]["summarizing"]
    assert intake["problem_ir"]["kind"] == "proof"
    assert intake["problem_ir_signature"]
    assert any("problem_ir.kind=proof" in entry for entry in intake["route_explanation"])


def test_build_workflow_intake_canonicalizes_math_paraphrases() -> None:
    first = (
        "Prove that the centroid of a triangle lies on the medians and explain the geometric intuition."
    )
    second = (
        "Show that the centroid of a triangle lies on its medians and explain the geometry behind it."
    )

    intake_a = workflow.build_workflow_intake(first)
    intake_b = workflow.build_workflow_intake(second)

    assert intake_a["problem_ir_signature"] == intake_b["problem_ir_signature"]
    assert intake_a["problem_ir"]["kind"] == "proof"
    assert intake_b["problem_ir"]["kind"] == "proof"
    assert intake_a["problem_ir"]["nodes"]
    assert intake_a["problem_ir"]["edges"]


def test_build_workflow_intake_exposes_semantic_probes() -> None:
    intake = workflow.build_workflow_intake(
        "Prove that the centroid of a triangle lies on the medians and explain the geometric intuition."
    )

    assert intake["semantic_probes"]["probe_checks"]
    assert intake["semantic_probes"]["counterexample_hints"]
    assert intake["prototype_stage"] in {"candidate", "emerging", "stable", "mature"}
    assert intake["holdout_bucket"] in {0, 1, 2, 3}


def test_task_shape_similarity_reuses_reasoning_lane() -> None:
    math_a = (
        "Prove that the centroid of a triangle lies on the medians and explain the geometric intuition."
    )
    math_b = (
        "Show that the incenter of a triangle is the intersection of the angle bisectors and derive its geometric meaning."
    )
    patch_text = (
        "Patch the repository to add a unified MCP workflow, test it, and verify the changes before commit."
    )

    def profile(text: str) -> dict[str, object]:
        intake = workflow.build_workflow_intake(text)
        task_pool = derive_task_ecology({"problem_text": text})
        prior = derive_feed_dna_prior(task_pool)
        return workflow.task_shape_profile(intake, task_pool, prior)

    math_profile_a = profile(math_a)
    math_profile_b = profile(math_b)
    patch_profile = profile(patch_text)

    math_similarity = task_shape_similarity(math_profile_a, math_profile_b)
    patch_similarity = task_shape_similarity(math_profile_a, patch_profile)

    assert math_similarity > patch_similarity
    assert math_similarity >= 0.55


def test_unified_workflow_orchestrates_synthesis_and_dispatch(tmp_path) -> None:
    problem_text = "Patch the repository to add a unified workflow and verify it before commit."
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "notes.txt").write_text("before\n")
    dispatch_plan = {
        "workspace_root": str(workspace_root),
        "updates": {"notes.txt": "after\n"},
        "verification_command": ["python3", "-c", "pass"],
    }

    result = workflow.run_unified_workflow(
        problem_text,
        context={"ticket": "P3"},
        state_dir=str(tmp_path / "state"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=7,
        dispatch_plan=dispatch_plan,
        workspace_root=str(workspace_root),
        commit=True,
    )

    assert result["status"] == "committed"
    assert result["intake"]["raw_task"] == problem_text
    assert result["execution"]["status"] == "committed"
    assert result["execution"]["verified"] is True
    assert result["execution"]["workspace_root"] == str(workspace_root)
    assert result["execution"]["updates"]
    assert result["execution"]["updates"][0]["path"] == "notes.txt"
    assert (workspace_root / "notes.txt").read_text() == "after\n"


def test_run_unified_workflow_end_to_end(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "notes.txt").write_text("before\n")

    dispatch_plan = {
        "workspace_root": str(workspace_root),
        "updates": {"notes.txt": "after\n"},
        "verification_command": ["python3", "-c", "pass"],
    }

    result = workflow.run_unified_workflow(
        "Patch the repository to add a unified workflow and verify it before commit.",
        context={"ticket": "P3"},
        state_dir=str(tmp_path / "state"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=7,
        dispatch_plan=dispatch_plan,
        workspace_root=str(workspace_root),
        commit=True,
    )

    assert result["status"] == "committed"
    assert result["api_version"] == workflow.API_VERSION
    assert result["execution"]["api_version"] == workflow.API_VERSION
    assert (workspace_root / "notes.txt").read_text() == "after\n"


def test_mcp_solve_problem_uses_same_core(tmp_path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    problem_text = "Patch the repository to add a unified workflow and verify the changes before commit."
    result = solve_problem(
        problem_text,
        context_json=json.dumps({"ticket": "P3"}),
        feed_json=json.dumps({"source": "ticket"}),
        dispatch_plan_json=json.dumps({
            "workspace_root": str(workspace_root),
            "updates": {"src/app.py": "after"},
            "verification_command": ["python3", "-c", "pass"],
        }),
        workspace_root=str(workspace_root),
        state_dir=str(tmp_path / "state"),
        commit=True,
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=11,
    )

    assert result["status"] == "committed"
    assert result["intake"]["raw_task"] == problem_text
    assert result["execution"]["status"] == "committed"
    assert result["execution"]["verified"] is True
    assert result["execution"]["updates"]
    assert result["execution"]["updates"][0]["path"] == "src/app.py"
    assert result["feed_prior_dna"] == result["synthesis"]["feed_prior_dna"]
    assert result["patch_skeleton"] == result["synthesis"]["patch_skeleton"]
    assert result["patch_token_outline"] == result["patch_skeleton"]["token_outline"]
    expected_dna_summary = " -> ".join(result["feed_prior_dna"][:5])
    if len(result["feed_prior_dna"]) > 5:
        expected_dna_summary += " -> ..."
    assert result["dna_summary"] == expected_dna_summary
    assert result["dna"] == {
        "feed_prior_dna": result["feed_prior_dna"],
        "dna_summary": result["dna_summary"],
        "patch_token_outline": result["patch_token_outline"],
    }
    assert result["synthesis"]["semantic_promotions"]
    assert result["synthesis"]["semantic_counterexamples"]
    assert result["synthesis"]["held_out_semantic_credit"]["kind_credit"]
    assert result["synthesis"]["prototype_lifecycle"]
    assert result["synthesis"]["primitive_contracts"]
    assert result["synthesis"]["curriculum_profile"]["ordered_tasks"]
    assert result["synthesis"]["landscape_profile"]["rotation"]["condition_number"] >= 0.0
    expected_execution_dna = result["feed_prior_dna"]
    expected_execution_summary = " -> ".join(expected_execution_dna[:5])
    if len(expected_execution_dna) > 5:
        expected_execution_summary += " -> ..."
    assert result["execution_seed"]["dna"] == expected_execution_dna
    assert result["execution_seed"]["dna_summary"] == expected_execution_summary
    assert result["execution_seed"].get("omitted_macros", []) == []
    assert result["execution_seed"]["tool_hint"] == "begin_execution"
    assert result["execution_seed"]["problem_text"] == problem_text
    assert result["execution_seed"]["next_instruction"].startswith("Call begin_execution(")
    assert result["decision_policy"]["mode"] == "act"
    assert result["synthesis"]["decision_policy"]["mode"] == "act"
    assert result["prototype_summary"]["centroid_margin"] >= 0.0
    assert result["prototype_summary"]["route_explanation"]
    assert result["prototype_summary"]["reuse_sources"]
    assert result["selection_report"]["selection_mode"] == "unified"
    assert result["selection_report"]["pareto_front_count"] >= 1
    assert result["rotation_analysis"]["condition_number"] >= 0.0
    assert result["parallax_telemetry"]["failure_probe_count"] >= 0
    assert result["execution_seed"]["decision_policy"]["mode"] == "act"
    assert result["prior_profile"] == result["synthesis"]["prior_profile"]

    session = mcp_server.begin_execution(json.dumps(result), state_dir=str(tmp_path / "state"))
    assert session["dna"] == expected_execution_dna
    assert session["current_token"] == expected_execution_dna[0]
    assert (workspace_root / "src" / "app.py").read_text() == "after"


def test_solve_problem_uses_task_shape_prior_cache(tmp_path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    first_problem = "Architect a FastAPI websocket chat backend with Redis pub/sub and room routing."
    second_problem = "Architect a FastAPI websocket chat backend with Redis pub/sub, room routing, and reconnect handling."

    first_result = solve_problem(
        first_problem,
        context_json=json.dumps({"ticket": "cache-seed"}),
        feed_json=json.dumps({"source": "cache"}),
        workspace_root=str(repo_root),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=13,
    )

    assert first_result["cache_hit"] is False
    assert first_result["synthesis"]["cache_hit"] is False
    assert (state_dir / "task_shape_priors.jsonl").exists()
    assert (state_dir / "outcome_ledger.jsonl").exists()
    assert first_result["synthesis"]["task_shape_signature"]
    assert first_result["synthesis"]["outcome_memory"]["available"] is True
    assert first_result["synthesis"]["outcome_memory"]["match_count"] >= 1

    second_result = solve_problem(
        second_problem,
        context_json=json.dumps({"ticket": "cache-hit"}),
        feed_json=json.dumps({"source": "cache"}),
        workspace_root=str(repo_root),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=13,
    )

    assert second_result["cache_hit"] is True
    assert second_result["synthesis"]["cache_hit"] is True
    assert second_result["synthesis"]["cache_similarity"] > 0.0
    assert second_result["feed_prior_dna"] == second_result["synthesis"]["feed_prior_dna"]
    assert second_result["prior_profile"] == second_result["synthesis"]["prior_profile"]
    assert second_result["synthesis"]["outcome_memory"]["available"] is True
    assert "task_shape_prior" in second_result["prototype_summary"]["reuse_sources"]


def test_solve_problem_surfaces_execution_feedback(tmp_path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    problem_text = "Patch the repository to add a unified workflow and verify it before commit."
    context = {"ticket": "feedback"}
    feed = {"source": "feedback"}
    intake = workflow.build_workflow_intake(problem_text, context=context, feed=feed)
    task_pool = derive_task_ecology({"problem_text": problem_text, "context": context, "upstream_feed": feed})
    feed_prior_dna = derive_feed_dna_prior(task_pool)
    shape_profile = workflow.task_shape_profile(intake, task_pool, feed_prior_dna)
    shape_signature = task_shape_signature(shape_profile)

    save_execution_feedback(
        state_dir,
        {
            "kind": "execution_feedback",
            "session_id": "exec_feedback_seed",
            "task_text": problem_text,
            "task_shape_signature": shape_signature,
            "route_hint": intake["route_hint"],
            "decision_policy": {"mode": "act", "support_score": 0.83},
            "completion_summary": {
                "verification_passed": True,
                "verification_pass_rate": 1.0,
                "successful_trace_count": 2,
                "learned_macro_count": 1,
                "artifact_count": 4,
                "market_total_score": 14.0,
                "market_total_price": 9.5,
            },
            "verification_result": {"passed": True},
            "execution_learning": {"successful_trace_count": 2, "learned_macro_count": 1},
            "artifact_count": 4,
            "market_count": 4,
            "dispatch_update_count": 1,
        },
    )

    result = solve_problem(
        problem_text,
        context_json=json.dumps(context),
        feed_json=json.dumps(feed),
        workspace_root=str(repo_root),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=31,
    )

    assert result["synthesis"]["execution_feedback"]["available"] is True
    assert result["synthesis"]["execution_feedback"]["match_count"] >= 1
    assert result["decision_policy"]["feedback_match_count"] >= 1


def test_act_ask_stop_policy_responds_to_signal_strength() -> None:
    act_policy = workflow.derive_act_ask_stop_policy(
        {
            "confidence": 0.84,
            "route_hint": "patching",
            "phase_scores": {
                "planning": 0.18,
                "patching": 0.62,
                "summarizing": 0.08,
                "recovery": 0.07,
                "reasoning": 0.05,
            },
            "risks": ["reliability risk: parse/verification failures"],
            "constraints": ["must preserve the public contract"],
            "deliverables": ["patch artifact", "test/verification artifact"],
        },
        {"effective_strength": 0.72},
        {
            "available": True,
            "match_count": 2,
            "best_success": {"similarity": 0.88, "outcome_score": 182.0},
            "best_failure": {"similarity": 0.22, "outcome_score": 64.0},
        },
        {"summary": {"governed_mean": 148.0}},
    )
    ask_policy = workflow.derive_act_ask_stop_policy(
        {
            "confidence": 0.62,
            "route_hint": "mixed",
            "phase_scores": {
                "planning": 0.31,
                "patching": 0.29,
                "summarizing": 0.15,
                "recovery": 0.14,
                "reasoning": 0.11,
            },
            "risks": ["low-confidence classification may require fallback heuristics"],
            "constraints": ["must keep the API stable"],
            "deliverables": ["patch artifact", "test/verification artifact"],
        },
        {"effective_strength": 0.56},
        {
            "available": True,
            "match_count": 1,
            "best_success": {"similarity": 0.55, "outcome_score": 120.0},
        },
        {"summary": {"governed_mean": 88.0}},
    )
    stop_policy = workflow.derive_act_ask_stop_policy(
        {
            "confidence": 0.31,
            "route_hint": "mixed",
            "phase_scores": {
                "planning": 0.25,
                "patching": 0.24,
                "summarizing": 0.22,
                "recovery": 0.15,
                "reasoning": 0.14,
            },
            "risks": [
                "low-confidence classification may require fallback heuristics",
                "operational risk: integration drift or unclear contract",
                "reliability risk: parse/verification failures",
                "scaling risk: stateful coupling across runners",
            ],
            "constraints": [],
            "deliverables": ["actionable plan artifact"],
        },
        {"effective_strength": 0.12},
        {"available": False, "match_count": 0},
        {"summary": {"governed_mean": 12.0}},
    )

    assert act_policy["mode"] == "act"
    assert act_policy["actionable"] is True
    assert act_policy["next_instruction"].startswith("Call begin_execution(")
    assert ask_policy["mode"] == "ask"
    assert ask_policy["clarifying_questions"]
    assert stop_policy["mode"] == "stop"
    assert "Stop:" in stop_policy["stop_reason"]


def test_begin_execution_rejects_non_act_policy(tmp_path) -> None:
    synthesis_result = {
        "problem_text": "Need more context before execution.",
        "state_dir": str(tmp_path / "state"),
        "synthesis": {
            "feed_prior_dna": ["PLAN", "VERIFY"],
            "learned_macros": {},
            "decision_policy": {
                "mode": "stop",
                "next_instruction": "Stop: the current signal is too weak or risky to proceed safely.",
            },
        },
        "decision_policy": {
            "mode": "stop",
            "next_instruction": "Stop: the current signal is too weak or risky to proceed safely.",
        },
    }

    with pytest.raises(ValueError, match="Stop: the current signal is too weak or risky to proceed safely."):
        mcp_server.begin_execution(json.dumps(synthesis_result), state_dir=str(tmp_path / "state"))


def test_prior_strength_defaults_follow_task_class(tmp_path) -> None:
    summary_result = workflow.build_unified_workflow(
        "Summarize the incident report and explain the findings.",
        state_dir=str(tmp_path / "summary"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=17,
    )
    raft_result = workflow.build_unified_workflow(
        "Design a Raft node with leader election, log replication, and quorum handling.",
        state_dir=str(tmp_path / "raft"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=19,
    )
    complex_result = workflow.build_unified_workflow(
        "Coordinate a brittle recovery plan across multiple failing services with rollback, retries, and manual intervention.",
        state_dir=str(tmp_path / "complex"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=23,
    )

    summary_profile = summary_result["prior_profile"]
    raft_profile = raft_result["prior_profile"]
    complex_profile = complex_result["prior_profile"]

    assert summary_profile["task_class"] == "summary"
    assert summary_profile["effective_strength"] == 0.9
    assert raft_profile["task_class"] == "raft"
    assert raft_profile["effective_strength"] == 0.7
    assert complex_profile["task_class"] == "complex"
    assert complex_profile["effective_strength"] == 0.4
    assert summary_profile["cache_min_similarity"] < raft_profile["cache_min_similarity"] < complex_profile["cache_min_similarity"]


def test_prior_strength_override_is_guarded(tmp_path) -> None:
    result = workflow.build_unified_workflow(
        "Coordinate a brittle recovery plan across multiple failing services with rollback, retries, and manual intervention.",
        state_dir=str(tmp_path / "override"),
        trials=1,
        population=2,
        base_episodes=1,
        generations=1,
        seed0=29,
        prior_strength=0.95,
    )

    profile = result["prior_profile"]

    assert profile["task_class"] == "complex"
    assert profile["requested_strength"] == 0.95
    assert profile["effective_strength"] == 0.55
    assert profile["guarded"] is True
    assert result["synthesis"]["prior_profile"] == profile
    assert result["synthesis"]["prior_strength"] == profile["effective_strength"]



