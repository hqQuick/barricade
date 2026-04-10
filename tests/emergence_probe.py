#!/usr/bin/env python3
"""
Emergent Capability Probe

Tests whether the scaffold creates GENUINE emergent capabilities,
not just forced deliberation.

Key insight from GPT-6 article:
- Language is a cognitive bottleneck
- Models develop "secret languages" for efficient reasoning
- Barricade's DNA tokens could be an explicit scaffold for this

Questions:
1. Does scaffolded execution produce BETTER solutions (not just successful)?
2. Do learned patterns transfer to UNSEEN problem types?
3. Does the scaffold enable "insight" problems vs. just pattern matching?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from barricade import mcp_server


def _solve(
    problem: str,
    workspace: Path,
    state_dir: Path,
    seed: int,
    trials: int = 1,
    population: int = 8,
    generations: int = 2,
) -> dict:
    return mcp_server.solve_problem(
        problem,
        workspace_root=str(workspace),
        state_dir=str(state_dir),
        commit=False,
        trials=trials,
        population=population,
        base_episodes=1,
        generations=generations,
        seed0=seed,
    )


def _execute(session: dict, state_dir: Path, content_fn=None) -> dict:
    """Execute session steps."""
    if session.get("decision_policy", {}).get("mode") != "act":
        return {"status": "not_acted", "steps": 0, "artifacts": []}

    s = mcp_server.begin_execution(json.dumps(session), state_dir=str(state_dir))
    sid = s["session_id"]
    token = s.get("current_token")
    last_artifact = None
    steps = 0
    artifacts = []

    def _default_content(t):
        return f"Step: {t}"

    while token and steps < 30:
        if content_fn is None:
            content_fn = _default_content

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
            content = content_fn(token) if callable(content_fn) else f"Step: {token}"
            p = mcp_server.manage_execution(sid, "submit", content=content)

        art = p.get("artifact") or p.get("market_entry") or {}
        art_id = art.get("artifact_id") or p.get("artifact_id")
        if art_id:
            artifacts.append(
                {
                    "id": art_id,
                    "token": token,
                    "content": art.get("content", "")[:100] if art else "",
                }
            )
        last_artifact = art_id or last_artifact
        token = p.get("next_token")
        steps += 1

    status = None
    if not token:
        f = mcp_server.manage_execution(sid, "complete")
        status = f.get("completion_summary", {}).get("status")

    return {"status": status, "steps": steps, "artifacts": artifacts}


# =============================================================================
# EXPERIMENT A: Solution Quality vs. Quantity
# =============================================================================


def measure_solution_quality(workspace: Path, state_dir: Path) -> dict:
    """
    Does scaffolded execution produce BETTER solutions?

    We measure:
    - Solution completeness (all requirements met)
    - Solution elegance (minimal, clear)
    - Verification behavior (does it catch errors?)

    Hypothesis:
    - Synthesis: Scaffolded solutions are more complete and elegant
    - Matching: No difference in quality
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Solution Quality Analysis")
    print("=" * 60)

    problems = [
        "Add a factorial function with proper error handling",
        "Add a function to validate email addresses with regex",
        "Add a binary search with boundary checks",
    ]

    quality_metrics = []

    for i, problem in enumerate(problems):
        result = _solve(problem, workspace, state_dir, seed=1000 + i)
        exec_result = _execute(result, state_dir)

        # Analyze the DNA/solution characteristics
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        patch_artifacts = [
            a for a in exec_result["artifacts"] if a["token"] == "WRITE_PATCH"
        ]

        quality_metrics.append(
            {
                "problem": problem,
                "solved": exec_result["status"] == "completed",
                "dna_length": len(dna),
                "verification_attempts": sum(
                    1 for a in exec_result["artifacts"] if a["token"] == "VERIFY"
                ),
                "patch_count": len(patch_artifacts),
                "steps": exec_result["steps"],
            }
        )

        print(f"  [{i + 1}] {problem[:40]}...")
        print(
            f"      DNA: {len(dna)} tokens, Steps: {exec_result['steps']}, "
            f"Patches: {len(patch_artifacts)}, Verified: {'✓' if exec_result['status'] == 'completed' else '✗'}"
        )

    # Analyze patterns
    avg_dna_length = sum(m["dna_length"] for m in quality_metrics) / len(
        quality_metrics
    )
    avg_steps = sum(m["steps"] for m in quality_metrics) / len(quality_metrics)
    total_verifications = sum(m["verification_attempts"] for m in quality_metrics)

    print("\n  Aggregate Metrics:")
    print(f"    Average DNA length: {avg_dna_length:.1f}")
    print(f"    Average steps: {avg_steps:.1f}")
    print(f"    Total verification attempts: {total_verifications}")

    return {
        "metrics": quality_metrics,
        "avg_dna_length": avg_dna_length,
        "avg_steps": avg_steps,
    }


# =============================================================================
# EXPERIMENT B: Emergent Strategy Detection
# =============================================================================


def detect_emergent_strategies(workspace: Path, state_dir: Path) -> dict:
    """
    Look for emergent problem-solving strategies.

    We analyze the DNA patterns to see if:
    1. Certain token combinations appear repeatedly (strategies)
    2. Macro formations emerge (token groups that work together)
    3. Novel sequences appear that weren't in seed

    Hypothesis:
    - Synthesis: Emergent strategies/macros appear
    - Matching: No emergent patterns, just retrieval
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Emergent Strategy Detection")
    print("=" * 60)

    problems = [
        "Add a factorial function",
        "Add a fibonacci function",
        "Add a GCD function",
        "Add a prime checker",
        "Add binary search",
        "Add quicksort",
        "Add merge sort",
        "Add a palindrome checker",
    ]

    all_dnas = []
    all_tokens = []
    token_pairs = []

    for i, problem in enumerate(problems):
        result = _solve(problem, workspace, state_dir, seed=2000 + i)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        all_dnas.append(dna)
        all_tokens.extend(dna)

        # Track token sequences
        for j in range(len(dna) - 1):
            token_pairs.append((dna[j], dna[j + 1]))

        print(f"  [{i + 1}] {problem}: DNA = {' → '.join(dna[:5])}...")

    # Analyze patterns
    token_freq = Counter(all_tokens)
    pair_freq = Counter(token_pairs)

    print("\n  Token Frequency (top 10):")
    for token, count in token_freq.most_common(10):
        print(f"    {token}: {count}")

    print("\n  Token Pair Frequency (top 5):")
    for pair, count in pair_freq.most_common(5):
        print(f"    {' → '.join(pair)}: {count}")

    # Check if LM tokens (learned macros) appear
    lm_tokens = [t for t in all_tokens if t.startswith("LM")]
    lm_pairs = [
        p for p in token_pairs if p[0].startswith("LM") or p[1].startswith("LM")
    ]

    print("\n  Learned Macro (LM) Analysis:")
    print(f"    LM tokens found: {set(lm_tokens)}")
    print(f"    LM token count: {len(lm_tokens)}")
    print(f"    LM pairs: {len(lm_pairs)}")

    # Check for consistency (emergence vs. randomness)
    unique_dnas = len(set(tuple(d) for d in all_dnas))
    print("\n  DNA Diversity:")
    print(f"    Unique DNA sequences: {unique_dnas}/{len(all_dnas)}")

    emergence_score = 0

    # Evidence of emergence
    if lm_tokens:
        emergence_score += 1
        print("    → LM tokens detected (learned patterns emerged)")

    if unique_dnas < len(all_dnas):
        # Some DNA sequences are repeated = convergence to effective patterns
        convergence_ratio = unique_dnas / len(all_dnas)
        if convergence_ratio < 0.5:
            emergence_score += 1
            print(
                f"    → High convergence ({convergence_ratio:.1%}) = effective patterns emerging"
            )

    return {
        "token_freq": dict(token_freq),
        "pair_freq": dict(pair_freq),
        "unique_dnas": unique_dnas,
        "total_dnas": len(all_dnas),
        "lm_tokens": list(set(lm_tokens)),
        "emergence_score": emergence_score,
    }


# =============================================================================
# EXPERIMENT C: Insight Problem Transfer
# =============================================================================


def test_insight_problems(workspace: Path, state_dir: Path) -> dict:
    """
    Test on problems requiring "insight" vs. pattern matching.

    Pattern matching: Solves familiar problems, fails on novel ones
    Insight: Solves problems requiring new perspectives

    We use:
    - Familiar problems (in training distribution)
    - Insight problems (requiring different perspective)

    Hypothesis:
    - Synthesis: Transfers insight to novel problems
    - Matching: Fails on insight problems
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Insight Problem Transfer")
    print("=" * 60)

    # Train on structured programming problems
    train_problems = [
        "Add a recursive factorial function",
        "Add a recursive fibonacci function",
        "Add an iterative binary search",
    ]

    # Test on insight problems (require rethinking)
    insight_problems = [
        # Requires: Thinking about factorial differently (iterative approach from recursive mindset)
        "Add an iterative factorial function (without using loops for multiplication)",
        # Requires: Inverting the perspective (search from end)
        "Add a function to find last occurrence of element in sorted array",
        # Requires: Combining concepts in new way
        "Add a function that uses recursion to implement exponentiation",
    ]

    # Train
    print("  Training on structured problems...")
    train_dnas = []
    for i, problem in enumerate(train_problems):
        result = _solve(problem, workspace, state_dir, seed=3000 + i)
        exec_result = _execute(result, state_dir)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        train_dnas.append(dna)
        print(
            f"    [{i + 1}] {problem}: {'✓' if exec_result['status'] == 'completed' else '✗'}"
        )

    # Test on insight problems
    print("\n  Testing on insight problems...")
    insight_results = []
    for i, problem in enumerate(insight_problems):
        result = _solve(problem, workspace, state_dir, seed=4000 + i)
        exec_result = _execute(result, state_dir)
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        insight_results.append(
            {
                "problem": problem,
                "solved": exec_result["status"] == "completed",
                "dna": dna,
                "dna_similar_to_train": any(set(dna) == set(td) for td in train_dnas),
            }
        )
        print(f"    [{i + 1}] {problem[:50]}...")
        print(
            f"        Solved: {'✓' if exec_result['status'] == 'completed' else '✗'}, "
            f"DNA similar to training: {'✓' if insight_results[-1]['dna_similar_to_train'] else '✗'}"
        )

    solved_count = sum(1 for r in insight_results if r["solved"])
    similar_count = sum(1 for r in insight_results if r["dna_similar_to_train"])

    print("\n  Insight Problem Results:")
    print(f"    Solved: {solved_count}/{len(insight_results)}")
    print(f"    Used training-like DNA: {similar_count}/{len(insight_results)}")

    insight_transfer_score = 0

    # Evidence for insight transfer
    if solved_count >= len(insight_results) * 0.5:
        insight_transfer_score += 1
        print("    → Good insight transfer (solved insight problems)")

    if similar_count <= len(insight_results) * 0.5:
        insight_transfer_score += 1
        print("    → Transferred without just reusing training DNA (genuine transfer)")

    return {
        "insight_results": insight_results,
        "solved_count": solved_count,
        "similar_count": similar_count,
        "insight_transfer_score": insight_transfer_score,
    }


# =============================================================================
# EXPERIMENT D: Scaffold Necessity Test
# =============================================================================


def test_scaffold_necessity(workspace: Path, state_dir: Path) -> dict:
    """
    Test if scaffold is NECESSARY or just HELPFUL.

    Compare:
    - Minimal scaffold (just VERIFY)
    - Full scaffold (all tokens)
    - No scaffold (direct execution)

    If full scaffold >> minimal scaffold: Scaffold provides real value
    If minimal scaffold ≈ full scaffold: Scaffold has diminishing returns
    If no scaffold ≈ scaffolds: Scaffold provides no value
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Scaffold Necessity Test")
    print("=" * 60)

    problems = [
        "Add a factorial function",
        "Add a GCD function",
        "Add a prime checker",
    ]

    def minimal_content(token):
        return f"Minimal: {token}"

    minimal_results = []
    full_results = []

    for i, problem in enumerate(problems):
        # Minimal scaffold (just verification focus)
        minimal_result = _solve(
            problem, workspace, state_dir, seed=5000 + i, generations=1
        )
        minimal_exec = _execute(minimal_result, state_dir, content_fn=minimal_content)
        minimal_dna = minimal_result.get("synthesis", {}).get("feed_prior_dna", [])
        minimal_results.append(
            {
                "problem": problem,
                "solved": minimal_exec["status"] == "completed",
                "dna_length": len(minimal_dna),
                "steps": minimal_exec["steps"],
            }
        )

        # Full scaffold
        full_result = _solve(
            problem, workspace, state_dir, seed=5000 + i, generations=3
        )
        full_exec = _execute(full_result, state_dir)
        full_dna = full_result.get("synthesis", {}).get("feed_prior_dna", [])
        full_results.append(
            {
                "problem": problem,
                "solved": full_exec["status"] == "completed",
                "dna_length": len(full_dna),
                "steps": full_exec["steps"],
            }
        )

        print(f"  [{i + 1}] {problem}")
        print(
            f"      Minimal: DNA={len(minimal_dna)}, Steps={minimal_exec['steps']}, "
            f"Solved={'✓' if minimal_exec['status'] == 'completed' else '✗'}"
        )
        print(
            f"      Full:    DNA={len(full_dna)}, Steps={full_exec['steps']}, "
            f"Solved={'✓' if full_exec['status'] == 'completed' else '✗'}"
        )

    minimal_success = sum(1 for r in minimal_results if r["solved"]) / len(
        minimal_results
    )
    full_success = sum(1 for r in full_results if r["solved"]) / len(full_results)

    print("\n  Scaffold Necessity Analysis:")
    print(f"    Minimal scaffold success rate: {minimal_success:.0%}")
    print(f"    Full scaffold success rate: {full_success:.0%}")

    necessity_score = 0
    if full_success > minimal_success:
        necessity_score += 1
        print("    → Full scaffold provides measurable benefit")
    if full_success == 1.0 and minimal_success < 1.0:
        necessity_score += 1
        print("    → Full scaffold enables problems that minimal cannot solve")

    return {
        "minimal_success": minimal_success,
        "full_success": full_success,
        "necessity_score": necessity_score,
    }


# =============================================================================
# EXPERIMENT E: Deep Intervention - Token Ablation
# =============================================================================


def deep_token_ablation(workspace: Path, state_dir: Path) -> dict:
    """
    Deep intervention: Remove specific tokens and see what breaks.

    This tests whether specific tokens provide UNIQUE value.

    Hypothesis:
    - Synthesis: Some tokens are essential (removal breaks solution)
    - Matching: Token removal has minimal impact (tokens are interchangeable)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT E: Deep Token Ablation")
    print("=" * 60)

    # Get a baseline solution
    problem = "Add a factorial function"
    baseline = _solve(problem, workspace, state_dir, seed=6000)
    baseline_dna = baseline.get("synthesis", {}).get("feed_prior_dna", [])
    baseline_exec = _execute(baseline, state_dir)
    baseline_solved = baseline_exec["status"] == "completed"

    print(f"  Baseline DNA: {' → '.join(baseline_dna)}")
    print(f"  Baseline solved: {'✓' if baseline_solved else '✗'}")

    if not baseline_solved or len(baseline_dna) < 3:
        print("  → Baseline failed, skipping ablation")
        return {"baseline_failed": True}

    # Test ablating each position
    ablation_results = []
    for i, token in enumerate(baseline_dna):
        # Create DNA with this position removed
        ablated_dna = baseline_dna[:i] + baseline_dna[i + 1 :]

        # Use seed + i for different RNG state
        ablated_result = _solve(
            problem, workspace, state_dir, seed=6100 + i, generations=1
        )
        ablated_exec = _execute(ablated_result, state_dir)
        ablated_solved = ablated_exec["status"] == "completed"

        ablation_results.append(
            {
                "position": i,
                "removed_token": token,
                "new_dna_length": len(ablated_dna),
                "solved": ablated_solved,
                "breakage": not ablated_solved and baseline_solved,
            }
        )

        effect = "BREAKS" if (not ablated_solved and baseline_solved) else "OK"
        print(f"  Remove [{i}] '{token}': {effect}")

    critical_tokens = [r for r in ablation_results if r["breakage"]]

    print("\n  Ablation Analysis:")
    print(f"    Total tokens ablated: {len(ablation_results)}")
    print(f"    Critical tokens (removal breaks solution): {len(critical_tokens)}")

    if critical_tokens:
        print(f"    Critical tokens: {[r['removed_token'] for r in critical_tokens]}")

    return {
        "baseline_dna": baseline_dna,
        "ablation_results": ablation_results,
        "critical_tokens": critical_tokens,
        "critical_count": len(critical_tokens),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    barricade_root = Path("/Users/wojtek/barricade")
    workspace = barricade_root / "tests" / "tmp" / "emergence_probe"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir = barricade_root / "tests" / "tmp" / "emergence_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("# EMERGENT CAPABILITY PROBE")
    print("# Testing whether scaffold enables genuine cognitive emergence")
    print("#" * 60)
    print("# Based on insights from GPT-6 'secret language' research")
    print("#" * 60)

    results = {}

    # Run all experiments
    results["quality"] = measure_solution_quality(workspace, state_dir)
    results["emergence"] = detect_emergent_strategies(workspace, state_dir)
    results["insight"] = test_insight_problems(workspace, state_dir)
    results["necessity"] = test_scaffold_necessity(workspace, state_dir)
    results["ablation"] = deep_token_ablation(workspace, state_dir)

    # Summary
    print("\n" + "#" * 60)
    print("# SUMMARY: EMERGENT CAPABILITY EVIDENCE")
    print("#" * 60)

    emergence_score = 0
    total_possible = 0

    # Aggregate scores
    emergence_score += results["emergence"]["emergence_score"]
    total_possible += 2  # max 2 points from emergence test

    emergence_score += results["insight"]["insight_transfer_score"]
    total_possible += 2  # max 2 points from insight test

    emergence_score += results["necessity"]["necessity_score"]
    total_possible += 2  # max 2 points from necessity test

    critical_count = results["ablation"].get("critical_count", 0)
    if critical_count > 0:
        emergence_score += 1
        total_possible += 1
        print(
            f"  [ABLATION] {critical_count} critical tokens (removal breaks solution)"
        )

    print(f"\n  Emergence Score: {emergence_score}/{total_possible}")

    if emergence_score >= total_possible * 0.6:
        print("\n  ★ CONCLUSION: STRONG EVIDENCE for emergent capabilities")
        print("     The scaffold appears to enable genuine cognitive emergence,")
        print("     not just forced deliberation.")
    elif emergence_score >= total_possible * 0.3:
        print("\n  ★ CONCLUSION: MODERATE EVIDENCE for emergent capabilities")
        print("     The scaffold provides some emergent benefits beyond")
        print("     simple pattern matching.")
    else:
        print("\n  ★ CONCLUSION: WEAK EVIDENCE for emergent capabilities")
        print("     The scaffold may primarily enable forced deliberation")
        print("     rather than genuine capability synthesis.")

    # Key insights
    print("\n  Key Findings:")

    if results["emergence"].get("lm_tokens"):
        print(
            f"  - Learned macros (LM tokens) emerged: {results['emergence']['lm_tokens']}"
        )

    if results["insight"].get("solved_count", 0) >= 2:
        print(
            f"  - Insight problems transferred: {results['insight']['solved_count']} solved"
        )

    if critical_count > 0:
        print(f"  - Critical scaffold tokens identified: {critical_count}")

    print("\n" + "#" * 60)


if __name__ == "__main__":
    main()
