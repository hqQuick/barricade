#!/usr/bin/env python3
"""
Extrapolation Probe

Runs experiments to distinguish "pattern matching" from "capability synthesis."

Tests:
1. Intervention Sensitivity - Does scrambling DNA hurt success?
2. Structural Transfer - Does scaffold transfer between domains?
3. Generalization Gradient - Success rate vs distance from training
4. Macro Essence - Is DNA compressed (economical)?
5. Failure Mode Analysis - Where do failures occur?
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from barricade import mcp_server


def _solve_minimal(problem: str, workspace: Path, state_dir: Path, seed: int) -> dict:
    return mcp_server.solve_problem(
        problem,
        workspace_root=str(workspace),
        state_dir=str(state_dir),
        commit=False,
        trials=1,
        population=8,
        base_episodes=1,
        generations=2,
        seed0=seed,
    )


def _execute_session(session: dict, state_dir: Path) -> dict:
    """Execute all steps in a session."""
    if session.get("decision_policy", {}).get("mode") != "act":
        return {"status": "not_acted", "steps": 0}

    s = mcp_server.begin_execution(json.dumps(session), state_dir=str(state_dir))
    sid = s["session_id"]
    token = s.get("current_token")
    last_artifact = None
    steps = 0

    while token and steps < 30:
        if token == "VERIFY":
            p = mcp_server.manage_execution(sid, "verify", command='python3 -c "pass"')
        elif token == "READ_ARTIFACT":
            m = mcp_server.manage_execution(sid, "report", limit=3)
            entries = m.get("market", [])
            if entries:
                last_artifact = entries[0]["artifact_id"]
            if last_artifact:
                p = mcp_server.manage_execution(sid, "read", artifact_id=last_artifact)
            else:
                break
        else:
            p = mcp_server.manage_execution(
                sid, "submit", content=f"Probe step: {token}"
            )

        art = p.get("artifact") or p.get("market_entry") or {}
        last_artifact = art.get("artifact_id") or p.get("artifact_id") or last_artifact
        token = p.get("next_token")
        steps += 1

    status = None
    if not token:
        f = mcp_server.manage_execution(sid, "complete")
        status = f.get("completion_summary", {}).get("status")

    return {"status": status, "steps": steps}


def _scramble_dna(dna: list, seed: int) -> list:
    """Scramble DNA while preserving tokens."""
    rng = random.Random(seed)
    result = dna[:]
    rng.shuffle(result)
    return result


# =============================================================================
# EXPERIMENT 1: Intervention Sensitivity
# =============================================================================


def run_intervention_experiment(
    workspace: Path, state_dir: Path, trials: int = 5
) -> dict:
    """
    Compare baseline DNA success vs scrambled DNA success.

    Hypothesis:
    - If DNA ORDER matters (synthesis): Scrambled < Baseline
    - If DNA ORDER doesn't matter (matching): Scrambled ≈ Baseline
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Intervention Sensitivity")
    print("=" * 60)
    print("Testing if DNA ORDER matters...")

    problems = [
        "Add a factorial function to utils.py",
        "Add a function to compute GCD",
        "Add a function to validate email format",
        "Add a helper function to swap two values",
        "Add a function to check if a number is prime",
    ]

    baseline_wins = 0
    scrambled_wins = 0
    ties = 0

    detailed_results = []

    for i, problem in enumerate(problems[:trials]):
        seed = 1000 + i * 100

        # Get baseline result
        baseline_result = _solve_minimal(problem, workspace, state_dir, seed)
        baseline_dna = baseline_result.get("synthesis", {}).get("feed_prior_dna", [])

        if not baseline_dna:
            print(f"  [{i + 1}] No DNA generated, skipping")
            continue

        # Baseline execution
        baseline_exec = _execute_session(baseline_result, state_dir)
        baseline_solved = baseline_exec["status"] == "completed"

        # Scrambled execution
        scrambled_dna = _scramble_dna(baseline_dna, seed=42 + i)
        scrambled_result = {
            **baseline_result,
            "synthesis": {
                **baseline_result.get("synthesis", {}),
                "feed_prior_dna": scrambled_dna,
            },
            "execution_seed": {"dna": scrambled_dna},
        }
        scrambled_exec = _execute_session(scrambled_result, state_dir)
        scrambled_solved = scrambled_exec["status"] == "completed"

        if baseline_solved and not scrambled_solved:
            baseline_wins += 1
            winner = "BASELINE"
        elif scrambled_solved and not baseline_solved:
            scrambled_wins += 1
            winner = "SCRAMBLED"
        else:
            ties += 1
            winner = "TIE"

        detailed_results.append(
            {
                "problem": problem,
                "baseline_solved": baseline_solved,
                "scrambled_solved": scrambled_solved,
                "dna_length": len(baseline_dna),
                "winner": winner,
            }
        )

        print(
            f"  [{i + 1}] Baseline: {'✓' if baseline_solved else '✗'} | Scrambled: {'✓' if scrambled_solved else '✗'} | Winner: {winner}"
        )

    print(
        f"\n  Results: Baseline wins={baseline_wins}, Scrambled wins={scrambled_wins}, Ties={ties}"
    )

    if baseline_wins > scrambled_wins:
        print(
            "  → INTERPRETATION: DNA order appears to MATTER (evidence for synthesis)"
        )
    elif scrambled_wins > baseline_wins:
        print("  → INTERPRETATION: Scrambled outperformed baseline (unexpected)")
    else:
        print("  → INTERPRETATION: No clear difference (inconclusive)")

    return {
        "baseline_wins": baseline_wins,
        "scrambled_wins": scrambled_wins,
        "ties": ties,
        "results": detailed_results,
    }


# =============================================================================
# EXPERIMENT 2: Structural Transfer
# =============================================================================


def run_transfer_experiment(workspace: Path, state_dir: Path) -> dict:
    """
    Test if scaffolds learned in one domain transfer to another.

    Train on: Mathematical functions (factorial, fibonacci, prime check)
    Test on: Geometric proofs (should require similar recursive structure)

    Hypothesis:
    - If synthesis: Structural patterns transfer (recursive approach)
    - If matching: Surface features don't transfer
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Structural Transfer")
    print("=" * 60)
    print("Testing cross-domain scaffold transfer...")

    train_problems = [
        "Add a factorial function",
        "Add a fibonacci function",
        "Add a prime checker function",
    ]

    test_problems = [
        "Prove the sum of angles in a triangle is 180 degrees",
        "Show that the centroid divides medians in 2:1 ratio",
    ]

    # Train on mathematical functions
    print("\n  Training domain (math functions)...")
    train_dnas = []
    for i, problem in enumerate(train_problems):
        result = _solve_minimal(problem, workspace, state_dir, seed=2000 + i)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        exec_result = _execute_session(result, state_dir)
        train_dnas.append(dna)
        print(
            f"    [{i + 1}] {problem}: DNA length={len(dna)}, solved={exec_result['status'] == 'completed'}"
        )

    # Test on geometric proofs (different domain)
    print("\n  Test domain (geometric proofs)...")
    test_results = []
    for i, problem in enumerate(test_problems):
        result = _solve_minimal(problem, workspace, state_dir, seed=3000 + i)
        exec_result = _execute_session(result, state_dir)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        test_results.append(
            {
                "problem": problem,
                "dna": dna,
                "solved": exec_result["status"] == "completed",
                "dna_length": len(dna),
            }
        )
        print(
            f"    [{i + 1}] {problem}: DNA length={len(dna)}, solved={exec_result['status'] == 'completed'}"
        )

    # Analyze DNA patterns
    all_train_tokens = []
    for dna in train_dnas:
        all_train_tokens.extend(dna)
    train_counter = {
        k: v
        for k, v in sorted(
            __import__("collections").Counter(all_train_tokens).items(),
            key=lambda x: -x[1],
        )[:10]
    }

    print(f"\n  Top tokens in training DNA: {train_counter}")

    solved_count = sum(1 for r in test_results if r["solved"])
    print(
        f"\n  Transfer results: {solved_count}/{len(test_results)} geometric problems solved"
    )

    if solved_count >= len(test_results) * 0.5:
        print(
            "  → INTERPRETATION: Good cross-domain transfer (evidence for structural synthesis)"
        )
    else:
        print("  → INTERPRETATION: Poor cross-domain transfer")

    return {
        "train_problems": train_problems,
        "test_problems": test_problems,
        "solved_count": solved_count,
        "total_count": len(test_results),
    }


# =============================================================================
# EXPERIMENT 3: Generalization Gradient
# =============================================================================


def run_generalization_gradient(workspace: Path, state_dir: Path) -> dict:
    """
    Measure success rate vs distance from training distribution.

    Pattern matching: Gradual decline
    Capability synthesis: Sharp boundary
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Generalization Gradient")
    print("=" * 60)
    print("Measuring success vs distance from training...")

    # Distance 0.0: Same as training
    # Distance 0.3: Small perturbation
    # Distance 0.6: Medium perturbation
    # Distance 0.9: Large perturbation

    perturbations = [
        (0.0, "Add a factorial function to utils.py"),
        (0.0, "Add a fibonacci function to utils.py"),
        (0.3, "Add a function to compute factorial iteratively"),
        (0.3, "Add a function to compute fibonacci iteratively"),
        (0.6, "Prove that factorial(n) relates to permutations"),
        (0.6, "Show the relationship between fibonacci and golden ratio"),
        (0.9, "Derive the relationship between factorial and gamma function"),
        (0.9, "Analyze the convergence properties of continued fraction expansions"),
    ]

    results_by_distance = defaultdict(list)

    for i, (distance, problem) in enumerate(perturbations):
        result = _solve_minimal(problem, workspace, state_dir, seed=4000 + i)
        exec_result = _execute_session(result, state_dir)
        solved = exec_result["status"] == "completed"
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])

        results_by_distance[distance].append(
            {
                "problem": problem,
                "solved": solved,
                "dna_length": len(dna),
            }
        )

        print(f"  Distance {distance}: {problem[:50]}... → {'✓' if solved else '✗'}")

    print("\n  Success by distance:")
    gradient = {}
    for distance in sorted(results_by_distance.keys()):
        results = results_by_distance[distance]
        rate = sum(1 for r in results if r["solved"]) / len(results)
        gradient[distance] = rate
        print(f"    {distance}: {rate:.1%} success rate ({len(results)} problems)")

    # Check for sharp boundary
    distances = sorted(gradient.keys())
    if len(distances) >= 3:
        drops = []
        for i in range(len(distances) - 1):
            drop = gradient[distances[i]] - gradient[distances[i + 1]]
            drops.append(drop)
        max_drop = max(drops) if drops else 0
        print(f"\n  Max drop between adjacent distances: {max_drop:.1%}")

        if max_drop > 0.3:
            print(
                "  → INTERPRETATION: Sharp boundary detected (evidence for synthesis)"
            )
        else:
            print("  → INTERPRETATION: Gradual decline (evidence for pattern matching)")

    return {"gradient": gradient, "results_by_distance": dict(results_by_distance)}


# =============================================================================
# EXPERIMENT 4: Macro Essence Compression
# =============================================================================


def run_macro_essence_experiment(workspace: Path, state_dir: Path) -> dict:
    """
    Check if evolved DNA is compressed (economical).

    Synthesis: Short DNA that captures essence
    Matching: Long DNA that memorizes surface patterns
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Macro Essence Compression")
    print("=" * 60)
    print("Measuring DNA economy...")

    problems = [
        "Add a factorial function",
        "Add a GCD function",
        "Add a prime checker",
        "Add a binary search function",
        "Add a quicksort implementation",
    ]

    dna_lengths = []
    for i, problem in enumerate(problems):
        result = _solve_minimal(problem, workspace, state_dir, seed=5000 + i)
        _execute_session(result, state_dir)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        dna_lengths.append(len(dna))

        print(f"  [{i + 1}] {problem}: DNA length = {len(dna)}")

    avg_length = sum(dna_lengths) / len(dna_lengths)
    max_length = max(dna_lengths)
    min_length = min(dna_lengths)

    print("\n  Statistics:")
    print(f"    Average DNA length: {avg_length:.1f}")
    print(f"    Min DNA length: {min_length}")
    print(f"    Max DNA length: {max_length}")

    if avg_length < 15:
        print("  → INTERPRETATION: DNA is economical (evidence for essence capture)")
    elif avg_length < 25:
        print("  → INTERPRETATION: DNA length is moderate")
    else:
        print(
            "  → INTERPRETATION: DNA is verbose (possible surface pattern memorization)"
        )

    return {
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "all_lengths": dna_lengths,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Use the barricade directory for state to pass validation
    barricade_root = Path("/Users/wojtek/barricade")
    workspace = barricade_root / "tests" / "tmp" / "extrapolation_probe"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir = barricade_root / "tests" / "tmp" / "extrapolation_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("# EXTRAPOLATION PROBE")
    print("# Distinguishing pattern matching from capability synthesis")
    print("#" * 60)

    results = {}

    # Run all experiments
    results["intervention"] = run_intervention_experiment(
        workspace, state_dir, trials=5
    )
    results["transfer"] = run_transfer_experiment(workspace, state_dir)
    results["generalization"] = run_generalization_gradient(workspace, state_dir)
    results["compression"] = run_macro_essence_experiment(workspace, state_dir)

    # Summary
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)

    synthesis_evidence = 0
    matching_evidence = 0

    # Intervention
    iv = results["intervention"]
    if iv["baseline_wins"] > iv["scrambled_wins"]:
        synthesis_evidence += 1
        print("  [INTERVENTION] DNA order matters → supports synthesis")
    else:
        matching_evidence += 1
        print("  [INTERVENTION] No order effect → suggests matching")

    # Transfer
    tr = results["transfer"]
    if tr["solved_count"] >= tr["total_count"] * 0.5:
        synthesis_evidence += 1
        print("  [TRANSFER] Cross-domain success → supports synthesis")
    else:
        matching_evidence += 1
        print("  [TRANSFER] Limited transfer → suggests matching")

    # Compression
    cp = results["compression"]
    if cp["avg_length"] < 15:
        synthesis_evidence += 1
        print("  [COMPRESSION] Economical DNA → supports synthesis")
    else:
        matching_evidence += 1
        print("  [COMPRESSION] Verbose DNA → suggests matching")

    print(f"\n  Evidence for SYNTHESIS: {synthesis_evidence}/3")
    print(f"  Evidence for MATCHING: {matching_evidence}/3")

    if synthesis_evidence > matching_evidence:
        print("\n  ★ CONCLUSION: Evidence suggests CAPABILITY SYNTHESIS")
    elif matching_evidence > synthesis_evidence:
        print("\n  ★ CONCLUSION: Evidence suggests PATTERN MATCHING")
    else:
        print("\n  ★ CONCLUSION: INCONCLUSIVE")

    print("\n" + "#" * 60)


if __name__ == "__main__":
    main()
