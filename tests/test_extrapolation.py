"""
Prompt-family scaffold tests.

These checks verify the current routing behavior of the scaffold:
- programming prompts cluster on one DNA template,
- proof-style prompts cluster on a second template,
- one structural prompt uses a reordered template,
- scrambling DNA order does not change completion in the current workflow.

They do not prove open-world extrapolation.
"""

from __future__ import annotations

import json
import random as rnd
from pathlib import Path

from barricade import mcp_server

FUNCTION_SCAFFOLD = (
    "OBSERVE",
    "LM1",
    "WRITE_PATCH",
    "PLAN",
    "REPAIR",
    "COMMIT",
    "WRITE_PLAN",
    "VERIFY",
    "SUMMARIZE",
)

PROOF_SCAFFOLD = (
    "OBSERVE",
    "LM1",
    "WRITE_PATCH",
    "PLAN",
    "REPAIR",
    "RETRIEVE",
    "WRITE_PLAN",
    "VERIFY",
    "SUMMARIZE",
)

STRUCTURAL_SCAFFOLD = (
    "OBSERVE",
    "LM1",
    "PLAN",
    "WRITE_PATCH",
    "REPAIR",
    "COMMIT",
    "WRITE_PLAN",
    "VERIFY",
    "SUMMARIZE",
)


def _dna_tuple(result: dict) -> tuple[str, ...]:
    return tuple(result.get("synthesis", {}).get("feed_prior_dna", []))


def _execute_minimal(
    problem_text: str, workspace_root: Path, state_dir: Path, seed0: int
) -> dict:
    """Execute a problem with minimal evo (1 trial, 1 gen) to test scaffold transfer."""
    result = mcp_server.solve_problem(
        problem_text,
        workspace_root=str(workspace_root),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=4,
        base_episodes=1,
        generations=1,
        seed0=seed0,
    )
    return result


def _drive_execution(result: dict, state_dir: Path) -> dict:
    """Execution driver for extrapolation tests - follows the extreme validation pattern."""
    if result.get("decision_policy", {}).get("mode") != "act":
        return {"acted": False, "status": "not_acted", "steps": 0}

    session = mcp_server.begin_execution(json.dumps(result), state_dir=str(state_dir))
    session_id = session["session_id"]
    token = session.get("current_token")
    last_artifact_id = None
    steps = 0

    while token is not None and steps < 30:
        if token == "VERIFY":
            payload = mcp_server.manage_execution(
                session_id,
                action="verify",
                command='python3 -c "pass"',
            )
        elif token == "READ_ARTIFACT":
            if not last_artifact_id:
                market = mcp_server.manage_execution(
                    session_id, action="report", limit=3
                )
                market_entries = market.get("market", [])
                if market_entries:
                    last_artifact_id = market_entries[0].get("artifact_id")
            if not last_artifact_id:
                break
            payload = mcp_server.manage_execution(
                session_id,
                action="read",
                artifact_id=last_artifact_id,
            )
        else:
            payload = mcp_server.manage_execution(
                session_id,
                action="submit",
                content=f"Extrapolation test step for {token}.",
            )

        artifact = payload.get("artifact") or payload.get("market_entry") or {}
        last_artifact_id = (
            artifact.get("artifact_id")
            or payload.get("artifact_id")
            or last_artifact_id
        )
        token = payload.get("next_token")
        steps += 1

    final_status = None
    if token is None:
        final = mcp_server.manage_execution(session_id, action="complete")
        final_status = final.get("completion_summary", {}).get("status")

    return {"acted": True, "status": final_status, "steps": steps}


# =============================================================================
# TEST 1: Prompt-Family Routing Across Domains
# =============================================================================

DOMAIN_A_PROBLEMS = [
    "Add a factorial function to utils.py",
    "Add a function to compute the GCD of two numbers",
]

DOMAIN_B_PROBLEMS = [
    "Prove that the centroid divides medians in 2:1 ratio",
    "Show that the sum of angles in a triangle is 180 degrees",
]

STRUCTURAL_TRANSFER_PROBLEM = "Analyze the structure of any recursive algorithm."


def test_structural_transfer_different_domains(tmp_path: Path) -> None:
    """
    Verify the current prompt families map to stable scaffold templates.

    Programming prompts keep the function scaffold, proof-like prompts keep
    the proof scaffold, and the structural prompt uses the reordered scaffold.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    domain_a_state = tmp_path / "domain_a"
    domain_b_state = tmp_path / "domain_b"
    transfer_state = tmp_path / "transfer"

    # Train on Domain A (math functions)
    domain_a_results = []
    for i, problem in enumerate(DOMAIN_A_PROBLEMS):
        result = _execute_minimal(problem, workspace, domain_a_state, seed0=100 + i)
        session = _drive_execution(result, domain_a_state)
        domain_a_results.append(
            {
                "problem": problem,
                "solved": session["status"] == "completed",
                "dna": _dna_tuple(result),
            }
        )

    # Now test transfer to Domain B (geometric proofs) - never trained on this
    domain_b_results = []
    for i, problem in enumerate(DOMAIN_B_PROBLEMS):
        result = _execute_minimal(problem, workspace, domain_b_state, seed0=200 + i)
        session = _drive_execution(result, domain_b_state)
        domain_b_results.append(
            {
                "problem": problem,
                "solved": session["status"] == "completed",
                "dna": _dna_tuple(result),
            }
        )

    transfer_result = _execute_minimal(
        STRUCTURAL_TRANSFER_PROBLEM, workspace, transfer_state, seed0=300
    )
    transfer_session = _drive_execution(transfer_result, transfer_state)
    transfer_dna = _dna_tuple(transfer_result)

    assert all(result["solved"] for result in domain_a_results)
    assert all(result["dna"] == FUNCTION_SCAFFOLD for result in domain_a_results)

    assert all(result["solved"] for result in domain_b_results)
    assert all(result["dna"] == PROOF_SCAFFOLD for result in domain_b_results)

    assert transfer_session["acted"], "Transfer task should be attempted"
    assert transfer_session["status"] == "completed"
    assert transfer_dna == STRUCTURAL_SCAFFOLD


# =============================================================================
# TEST 2: Prompt-Family Boundary Under Perturbation
# =============================================================================

TRAINING_PROBLEMS = [
    "Add a helper function to calculate factorial",
    "Add a function to compute fibonacci numbers",
    "Add a function to check if a number is prime",
]

PERTURBATION_PROBLEMS = [
    # Small perturbations - similar to training
    ("Add a function to compute factorial iteratively", 0.2),
    # Medium perturbations - requires generalization
    ("Prove that factorial(n) = n! using induction", 0.5),
    # Large perturbations - requires extrapolation
    ("Derive the relationship between factorial and gamma function", 0.8),
]


def test_generalization_gradient(tmp_path: Path) -> None:
    """
    Verify that small perturbations stay on the programming scaffold while
    proof-like prompts select the proof scaffold.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    training_state = tmp_path / "training"

    # Run training problems
    training_results = []
    for i, problem in enumerate(TRAINING_PROBLEMS):
        result = _execute_minimal(problem, workspace, training_state, seed0=400 + i)
        session = _drive_execution(result, training_state)
        training_results.append(
            {
                "problem": problem,
                "solved": session["status"] == "completed",
                "dna": _dna_tuple(result),
            }
        )

    # Run perturbations at different distances
    perturbation_results = []
    for i, (problem, distance) in enumerate(PERTURBATION_PROBLEMS):
        result = _execute_minimal(
            problem, workspace, training_state, seed0=500 + int(distance * 100)
        )
        session = _drive_execution(result, training_state)
        perturbation_results.append(
            {
                "problem": problem,
                "distance": distance,
                "solved": session["status"] == "completed",
                "dna": _dna_tuple(result),
            }
        )

    training_success_rate = sum(1 for result in training_results if result["solved"]) / len(training_results)

    assert training_success_rate == 1.0
    assert all(result["dna"] == FUNCTION_SCAFFOLD for result in training_results)

    for result in perturbation_results:
        expected_scaffold = FUNCTION_SCAFFOLD if result["distance"] <= 0.3 else PROOF_SCAFFOLD
        assert result["solved"]
        assert result["dna"] == expected_scaffold


# =============================================================================
# TEST 3: Compact Scaffold Consistency
# =============================================================================

COMPRESSIBLE_MACRO_PROBLEMS = [
    "Add a function that validates email addresses",
    "Add a function that parses simple arithmetic expressions",
    "Add a function that implements binary search",
]


def test_macro_essence_extraction(tmp_path: Path) -> None:
    """
    Verify that the programming scaffold stays compact across the prompts
    that currently route to it.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    state = tmp_path / "state"

    # Generate solutions for multiple problems
    results = []
    for i, problem in enumerate(COMPRESSIBLE_MACRO_PROBLEMS):
        result = _execute_minimal(problem, workspace, state, seed0=600 + i)
        session = _drive_execution(result, state)

        results.append(
            {
                "problem": problem,
                "dna": _dna_tuple(result),
                "dna_length": len(_dna_tuple(result)),
                "solved": session["status"] == "completed",
            }
        )

    # Check for macro patterns
    dna_lengths = [r["dna_length"] for r in results]
    avg_length = sum(dna_lengths) / len(dna_lengths)

    assert avg_length == len(FUNCTION_SCAFFOLD)
    assert all(result["solved"] for result in results)
    assert all(result["dna"] == FUNCTION_SCAFFOLD for result in results)


# =============================================================================
# TEST 4: Proof-Style Prompt Routing
# =============================================================================

FAILURE_TEST_PROBLEMS = [
    "Prove that there are infinitely many primes",
    "Show that the derivative of x^n is nx^(n-1) using first principles",
    "Derive the quadratic formula from standard form",
]


def test_failure_mode_analysis(tmp_path: Path) -> None:
    """
    Verify that proof-style prompts route to the proof scaffold and complete.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    state = tmp_path / "state"

    failure_modes = []
    for i, problem in enumerate(FAILURE_TEST_PROBLEMS):
        result = _execute_minimal(problem, workspace, state, seed0=700 + i)
        session = _drive_execution(result, state)

        failure_modes.append(
            {
                "problem": problem,
                "solved": session["status"] == "completed",
                "steps": session.get("steps", 0),
                "decision_mode": result.get("decision_policy", {}).get("mode"),
                "dna": _dna_tuple(result),
            }
        )

    assert len(failure_modes) == len(FAILURE_TEST_PROBLEMS)
    assert all(mode["decision_mode"] == "act" for mode in failure_modes)
    assert all(mode["solved"] for mode in failure_modes)
    assert all(mode["dna"] == PROOF_SCAFFOLD for mode in failure_modes)


# =============================================================================
# TEST 5: Common and Novel Composition Reuse the Same Scaffold
# =============================================================================

NOVEL_COMPOSITION_PROBLEMS = [
    # Requires combining: recursion + memoization (common pattern)
    ("Add a function that computes fibonacci with memoization", "common"),
    # Requires combining: recursion + type system + math (less common)
    (
        "Add a type-safe factorial function that works for positive integers only",
        "novel",
    ),
    # Requires combining: multiple concepts in non-standard way
    (
        "Add a function that computes factorial using only addition and subtraction",
        "novel",
    ),
]


def test_novel_composition_ability(tmp_path: Path) -> None:
    """
    Verify that the common and novel composition prompts currently reuse the
    same programming scaffold.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    state = tmp_path / "state"

    composition_results = []
    for i, (problem, novelty) in enumerate(NOVEL_COMPOSITION_PROBLEMS):
        result = _execute_minimal(problem, workspace, state, seed0=800 + i)
        session = _drive_execution(result, state)
        composition_results.append(
            {
                "problem": problem,
                "novelty": novelty,
                "solved": session["status"] == "completed",
                "dna": _dna_tuple(result),
                "dna_length": len(_dna_tuple(result)),
            }
        )

    common_problems = [r for r in composition_results if r["novelty"] == "common"]
    novel_problems = [r for r in composition_results if r["novelty"] == "novel"]

    assert common_problems and novel_problems
    assert all(result["solved"] for result in composition_results)
    assert all(result["dna"] == FUNCTION_SCAFFOLD for result in composition_results)


# =============================================================================
# TEST 6: DNA Order Does Not Change Completion
# =============================================================================

INTERVENTION_PROBLEMS = [
    "Add a helper function to swap two variables",
]


def _scramble_dna(dna: list[str], seed: int) -> list[str]:
    """Scramble DNA while preserving tokens - tests if ORDER matters."""
    rng = rnd.Random(seed)
    scrambled = dna[:]
    rng.shuffle(scrambled)
    return scrambled


def _create_scrambled_synthesis(
    original_result: dict, scrambled_dna: list[str]
) -> dict:
    """Create a new synthesis result with scrambled DNA."""
    synthesis = dict(original_result.get("synthesis", {}))
    synthesis["feed_prior_dna"] = scrambled_dna
    return {
        **original_result,
        "synthesis": synthesis,
        "execution_seed": {"dna": scrambled_dna},
    }


def _run_with_dna(synthesis_result: dict, state_dir: Path) -> dict:
    """Run execution with a specific synthesis result."""
    return mcp_server.begin_execution(
        json.dumps(synthesis_result), state_dir=str(state_dir)
    )


def _execute_session_steps(session: dict, state_dir: Path) -> dict:
    """Execute all steps in a session."""
    session_id = session["session_id"]
    token = session.get("current_token")
    last_artifact_id = None
    steps = 0

    while token is not None and steps < 30:
        if token == "VERIFY":
            payload = mcp_server.manage_execution(
                session_id, action="verify", command='python3 -c "pass"'
            )
        elif token == "READ_ARTIFACT":
            if not last_artifact_id:
                market = mcp_server.manage_execution(
                    session_id, action="report", limit=3
                )
                market_entries = market.get("market", [])
                if market_entries:
                    last_artifact_id = market_entries[0].get("artifact_id")
            if not last_artifact_id:
                break
            payload = mcp_server.manage_execution(
                session_id, action="read", artifact_id=last_artifact_id
            )
        else:
            payload = mcp_server.manage_execution(
                session_id,
                action="submit",
                content=f"Intervention test step for {token}.",
            )

        artifact = payload.get("artifact") or payload.get("market_entry") or {}
        last_artifact_id = (
            artifact.get("artifact_id")
            or payload.get("artifact_id")
            or last_artifact_id
        )
        token = payload.get("next_token")
        steps += 1

    final_status = None
    if token is None:
        final = mcp_server.manage_execution(session_id, action="complete")
        final_status = final.get("completion_summary", {}).get("status")

    return {"status": final_status, "steps": steps, "completed": token is None}


def test_intervention_sensitivity(tmp_path: Path) -> None:
    """
    Verify that scrambling DNA order does not change completion in the current
    workflow.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    baseline_state = tmp_path / "baseline"
    scrambled_state = tmp_path / "scrambled"

    problem = INTERVENTION_PROBLEMS[0]

    # Run multiple trials to get statistically meaningful results
    baseline_results = []
    scrambled_results = []

    for trial in range(3):
        baseline_trial_state = baseline_state / f"trial_{trial}"
        scrambled_trial_state = scrambled_state / f"trial_{trial}"
        baseline_trial_state.mkdir(parents=True, exist_ok=True)
        scrambled_trial_state.mkdir(parents=True, exist_ok=True)

        baseline_result = _execute_minimal(
            problem, workspace, baseline_trial_state, seed0=900 + trial * 10
        )
        baseline_dna = baseline_result.get("synthesis", {}).get("feed_prior_dna", [])

        if not baseline_dna:
            continue

        baseline_session = _run_with_dna(baseline_result, baseline_trial_state)
        baseline_exec = _execute_session_steps(baseline_session, baseline_trial_state)
        baseline_results.append(
            {
                "dna_length": len(baseline_dna),
                "dna": baseline_dna,
                "solved": baseline_exec["status"] == "completed",
            }
        )

        scrambled_dna = _scramble_dna(baseline_dna, seed=42 + trial)
        scrambled_synthesis = _create_scrambled_synthesis(
            baseline_result, scrambled_dna
        )

        scrambled_session = _run_with_dna(scrambled_synthesis, scrambled_trial_state)
        scrambled_exec = _execute_session_steps(scrambled_session, scrambled_trial_state)
        scrambled_results.append(
            {
                "original_dna": baseline_dna,
                "scrambled_dna": scrambled_dna,
                "solved": scrambled_exec["status"] == "completed",
            }
        )

    assert len(baseline_results) > 0, "Should have baseline results"
    assert len(scrambled_results) > 0, "Should have scrambled results"

    baseline_success = sum(1 for r in baseline_results if r["solved"])
    scrambled_success = sum(1 for r in scrambled_results if r["solved"])

    baseline_rate = baseline_success / len(baseline_results)
    scrambled_rate = scrambled_success / len(scrambled_results)

    assert baseline_rate == 1.0
    assert scrambled_rate == 1.0
    assert all(result["solved"] for result in baseline_results)
    assert all(result["solved"] for result in scrambled_results)
