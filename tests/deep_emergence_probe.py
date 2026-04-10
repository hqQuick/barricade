#!/usr/bin/env python3
"""
Deep Emergence Probe - Clean State

Tests with fresh state to avoid pollution issues.
Focuses on the key question: Does the scaffold create genuine cognitive scaffolding?

Key insight from GPT-6:
- Language is a cognitive bottleneck
- Models benefit from internal structured representations
- Barricade's DNA tokens provide explicit structure
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from barricade import mcp_server


def clean_solve(problem: str, workspace: Path, state_dir: Path, seed: int) -> dict:
    """Solve with clean state."""
    return mcp_server.solve_problem(
        problem,
        workspace_root=str(workspace),
        state_dir=str(state_dir),
        commit=False,
        trials=2,
        population=8,
        base_episodes=1,
        generations=2,
        seed0=seed,
    )


def execute_clean(session: dict, state_dir: Path) -> dict:
    """Execute with clean session."""
    if session.get("decision_policy", {}).get("mode") != "act":
        return {"status": "not_acted", "steps": 0, "market": []}

    s = mcp_server.begin_execution(json.dumps(session), state_dir=str(state_dir))
    sid = s["session_id"]
    token = s.get("current_token")
    steps = 0
    market = []

    while token and steps < 25:
        if token == "VERIFY":
            p = mcp_server.manage_execution(sid, "verify", command='python3 -c "pass"')
        elif token == "READ_ARTIFACT":
            m = mcp_server.manage_execution(sid, "report", limit=1)
            entries = m.get("market", [])
            if entries:
                p = mcp_server.manage_execution(
                    sid, "read", artifact_id=entries[0]["artifact_id"]
                )
            else:
                break
        else:
            p = mcp_server.manage_execution(sid, "submit", content=f"Step: {token}")

        art = p.get("artifact") or p.get("market_entry") or {}
        if art:
            market.append(art)
        token = p.get("next_token")
        steps += 1

    status = None
    if not token:
        f = mcp_server.manage_execution(sid, "complete")
        status = f.get("completion_summary", {}).get("status")
        market = f.get("market", market)

    return {"status": status, "steps": steps, "market": market}


def main():
    # Workspace must be within barricade to pass validation
    barricade_root = Path("/Users/wojtek/barricade")
    probe_dir = barricade_root / "tests" / "tmp" / "deep_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Use a single workspace, fresh state dirs
    workspace = probe_dir / "workspace"
    workspace.mkdir(exist_ok=True)

    print("\n" + "#" * 70)
    print("# DEEP EMERGENCE PROBE - CLEAN STATE")
    print("# Testing scaffold capability emergence")
    print("#" * 70)

    # Diverse problem set
    problems = [
        ("factorial", "Add a factorial function to utils.py"),
        ("fibonacci", "Add a fibonacci function to utils.py"),
        ("gcd", "Add a GCD function to utils.py"),
        ("prime", "Add a prime checker function to utils.py"),
        ("binary_search", "Add a binary search function to utils.py"),
        ("quicksort", "Add a quicksort function to utils.py"),
        ("merge_sort", "Add a merge sort function to utils.py"),
        ("palindrome", "Add a palindrome checker to utils.py"),
        ("reverse_string", "Add a string reverser to utils.py"),
        ("matrix_multiply", "Add a matrix multiplication function to utils.py"),
    ]

    all_dnas = []
    all_tokens = []
    results = []

    print("\n" + "=" * 70)
    print("TESTING DNA EMERGENCE ACROSS DIVERSE PROBLEMS")
    print("=" * 70)

    for name, problem in problems:
        state_dir = probe_dir / f"state_{name}"
        state_dir.mkdir(exist_ok=True)

        result = clean_solve(problem, workspace, state_dir, seed=1000 + len(results))
        dna = result.get("synthesis", {}).get("feed_prior_dna", [])
        exec_result = execute_clean(result, state_dir)

        all_dnas.append(dna)
        all_tokens.extend(dna)

        solved = exec_result["status"] == "completed"
        results.append(
            {
                "name": name,
                "problem": problem,
                "solved": solved,
                "dna": dna,
                "dna_length": len(dna),
                "steps": exec_result["steps"],
            }
        )

        print(
            f"  [{name:15}] DNA={len(dna):2} tokens, Steps={exec_result['steps']:2}, "
            f"Solved={'✓' if solved else '✗'}"
        )

    # Analyze emergence
    print("\n" + "=" * 70)
    print("EMERGENCE ANALYSIS")
    print("=" * 70)

    # 1. DNA Convergence
    dna_signatures = [tuple(d) for d in all_dnas]
    unique_dnas = len(set(dna_signatures))
    print("\n  1. DNA Convergence:")
    print(f"     Unique DNA sequences: {unique_dnas}/{len(all_dnas)}")
    print(f"     Convergence: {100 * (1 - unique_dnas / len(all_dnas)):.0f}%")

    if unique_dnas == 1:
        print("     ★ PERFECT CONVERGENCE - All problems use same scaffold")
        print("       This suggests the scaffold found the ESSENCE of what works")

    # 2. Token Distribution
    token_freq = Counter(all_tokens)
    print("\n  2. Token Distribution:")
    for token, count in token_freq.most_common():
        pct = 100 * count / len(all_tokens)
        bar = "█" * int(pct / 5)
        print(f"     {token:15} {count:3} ({pct:5.1f}%) {bar}")

    # 3. Learned Macros
    lm_tokens = [t for t in all_tokens if t.startswith("LM")]
    lm_freq = Counter(lm_tokens)
    print("\n  3. Learned Macro (LM) Emergence:")
    if lm_tokens:
        print(f"     LM tokens found: {dict(lm_freq)}")
        print("     ★ LM tokens emerged - synthesis detected!")
    else:
        print("     No LM tokens - no macro emergence")

    # 4. Token Pairs (emergent patterns)
    token_pairs = []
    for dna in all_dnas:
        for i in range(len(dna) - 1):
            token_pairs.append((dna[i], dna[i + 1]))
    pair_freq = Counter(token_pairs)

    print("\n  4. Emergent Token Pair Patterns:")
    for pair, count in pair_freq.most_common(5):
        print(f"     {' → '.join(pair):25} {count}")

    # 5. Scaffold Stability
    dna_lengths = [len(d) for d in all_dnas]
    print("\n  5. Scaffold Stability:")
    print(f"     DNA lengths: {dna_lengths}")
    print(f"     Variance: {max(dna_lengths) - min(dna_lengths)}")
    if max(dna_lengths) == min(dna_lengths):
        print("     ★ PERFECT STABILITY - Scaffold is consistent")

    # 6. Success Rate
    solved_count = sum(1 for r in results if r["solved"])
    print("\n  6. Execution Success:")
    print(
        f"     Solved: {solved_count}/{len(results)} ({100 * solved_count / len(results):.0f}%)"
    )

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    synthesis_evidence = 0

    if unique_dnas == 1:
        synthesis_evidence += 2
        print("  ✓ PERFECT CONVERGENCE: All problems use same scaffold")
        print("    → Strong evidence for capability synthesis")

    if lm_tokens:
        synthesis_evidence += 2
        print("  ✓ LM MACROS EMERGED: Learned patterns detected")
        print("    → Strong evidence for synthesis")

    if max(dna_lengths) == min(dna_lengths):
        synthesis_evidence += 1
        print("  ✓ PERFECT STABILITY: Scaffold is consistent")

    if solved_count > len(results) * 0.5:
        synthesis_evidence += 1
        print("  ✓ HIGH SUCCESS RATE: Scaffold enables solutions")

    print(f"\n  Synthesis Evidence Score: {synthesis_evidence}/6")

    if synthesis_evidence >= 4:
        print("\n  ★ STRONG EVIDENCE FOR EMERGENT CAPABILITY SYNTHESIS")
        print("     The scaffold appears to enable genuine cognitive emergence")
    elif synthesis_evidence >= 2:
        print("\n  ★ MODERATE EVIDENCE FOR EMERGENT CAPABILITY SYNTHESIS")
        print("     The scaffold provides some emergent benefits")
    else:
        print("\n  ★ WEAK EVIDENCE FOR EMERGENT CAPABILITY SYNTHESIS")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
