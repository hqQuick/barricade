from __future__ import annotations

import json
import unittest
from pathlib import Path

from barricade import mcp_server, runtime
from barricade.feed_derived_dna.models import EvolutionConfig
from barricade.feed_derived_dna.analysis import flatten_trace as recursive_flatten_trace


MINIOPS_CRISIS_ECOLOGY_V3_12_SUMMARY_MEMO_SCENARIO = {
    "scenario_name": "tul_v3_12_summary_memo_benchmark",
    "objective": "Produce a concise technical memo from repository evidence without drifting into implementation work.",
    "route_hint": "summarizing",
    "benchmark_task": "Summarize the current TUL / MiniOps repository state into a decision memo for a senior engineer. Focus on architecture, coordination flow, current risks, and the operational steps that remain. The output should be compact, explicit, and suitable for review without code changes.",
    "context": {
        "repo": "TUL / MiniOps",
        "source_material": ["docs/purpose.md", "docs/analysis_results.md", "miniops_crisis_ecology_v3_12_contract_bridge.py"],
        "constraints": [
            "must stay focused on summarization",
            "must avoid proposing code edits as the primary output",
            "must be deterministic for identical seeds",
            "must surface clear review risks",
            "must keep the response compact"
        ],
        "outputs": ["normalized frame", "phase scores", "dna path", "summary memo", "verification"],
        "failure_modes": [
            "overlong output",
            "mixing analysis with patch planning",
            "loss of review focus"
        ]
    },
    "subtasks": [
        {"id": "normalize", "goal": "Convert the repository context into a canonical memo frame."},
        {"id": "score", "goal": "Prioritize summarizing cues and avoid patch bias."},
        {"id": "route", "goal": "Select a summary-oriented DNA path."},
        {"id": "verify", "goal": "Check that the memo is compact and reviewable."}
    ],
    "conflicts": [
        "the source evidence mixes architecture, benchmarks, and live-ops notes",
        "the output should summarize without turning into a plan or patch",
        "the memo must stay short even though the evidence is broad"
    ],
    "reference_solution": {
        "expected_phase": "summarizing",
        "expected_artifact_type": "summary",
        "expected_dna_prefix": ["OBSERVE", "LM1", "WRITE_SUMMARY"],
        "must_include_steps": [
            "Normalize the request",
            "Score phases",
            "Select DNA",
            "Verify contract"
        ],
        "notes": "Review memo baseline for an evidence synthesis task."
    }
}

MINIOPS_CRISIS_ECOLOGY_V3_12_RAFT_NODE_SCENARIO = {
    "ticket_description": "Design and implement the core state machine and async networking stub for a Raft Consensus Algorithm node in Python. The node must handle Follower, Candidate, and Leader states, execute randomized election timeouts, and process skeletal RequestVote and AppendEntries RPCs. Provide a strict architectural plan and the foundational node class code.",
    "use_elite_dna": True
}

MINIOPS_CRISIS_ECOLOGY_V3_12_COMPLEX_SCENARIO = {
    "scenario_name": "tul_v3_12_sota_contract_benchmark",
    "objective": "Stress test the v3.12 contract bridge on a nested, conflicting, real-world task frame.",
    "benchmark_task": "Normalize this nested JSON task into a deterministic contract. The bridge must score planning, patching, summarizing, and recovery; select DNA; and emit a parseable plan. It should support multiple seeds, survive ambiguous cues, and keep verification explicit. Treat the input as a realistic multi-agent orchestration scenario with several competing goals.",
    "context": {
        "repo": "TUL / MiniOps",
        "mode": "benchmark",
        "inputs": ["plain text", "nested JSON"],
        "constraints": [
            "must be deterministic for the same seed",
            "must expose seed-sensitive behavior for ambiguous tasks",
            "must produce a structured plan artifact",
            "must not depend on external APIs",
            "must compare against a hand-authored reference"
        ],
        "outputs": ["normalized frame", "phase scores", "dna path", "plan", "verification"]
    },
    "subtasks": [
        {"id": "normalize", "goal": "Convert nested JSON into a canonical task frame."},
        {"id": "score", "goal": "Score phases and identify the dominant route."},
        {"id": "route", "goal": "Select DNA and a plan shape that fits the request."},
        {"id": "verify", "goal": "Check contract completeness and parseability."}
    ],
    "conflicts": [
        "planning and patching cues are intentionally mixed",
        "summary and verification cues are also present",
        "the benchmark wants deterministic structure but seed-sensitive tie breaks"
    ],
    "reference_solution": {
        "expected_phase": "mixed",
        "expected_artifact_type": "plan",
        "expected_dna_prefix": ["OBSERVE", "LM1"],
        "must_include_steps": [
            "Normalize the request",
            "Score phases",
            "Select DNA",
            "Verify contract"
        ],
        "notes": "Hand-authored baseline representing the SOTA target for this benchmark."
    }
}

MINIOPS_CRISIS_ECOLOGY_V3_12_ROUTER_REPAIR_SCENARIO = {
    "scenario_name": "tul_v3_12_router_repair_benchmark",
    "objective": "Stabilize the live LLM router when the provider fails and make its outputs machine-checkable.",
    "route_hint": "patching",
    "benchmark_task": "Refactor prototype/tul_live/llm_router.py so role execution no longer depends on a brittle provider call path. Add a safe fallback when Gemini errors, preserve artifact typing per role, and emit explicit artifact IDs and a compact latent macro so downstream agents can inspect and replay the decision. Keep the change deterministic and self-contained.",
    "context": {
        "repo": "TUL / MiniOps",
        "module": "prototype/tul_live/llm_router.py",
        "inputs": ["raw task context", "purchased artifacts", "market ledger"],
        "constraints": [
            "must remain deterministic for identical seeds",
            "must not depend on external APIs for validation",
            "must preserve role-specific artifact types",
            "must surface explicit artifact IDs",
            "must keep fallback behavior observable"
        ],
        "outputs": ["normalized frame", "phase scores", "dna path", "patch plan", "verification"],
        "failure_modes": [
            "provider quota failures",
            "malformed artifact payloads",
            "ambiguous role routing"
        ]
    },
    "subtasks": [
        {"id": "normalize", "goal": "Convert the router task into a canonical repair frame."},
        {"id": "score", "goal": "Prioritize patching and recovery cues without losing structure."},
        {"id": "route", "goal": "Select a code-oriented DNA path for the repair."},
        {"id": "verify", "goal": "Check the contract and fallback integrity."}
    ],
    "conflicts": [
        "the router currently calls an external model in the hot path",
        "role outputs must remain type-correct even when fallback is used",
        "the system needs clear artifact IDs for downstream market inspection"
    ],
    "reference_solution": {
        "expected_phase": "patching",
        "expected_artifact_type": "patch",
        "expected_dna_prefix": ["OBSERVE", "LM1", "WRITE_PLAN", "WRITE_PATCH"],
        "must_include_steps": [
            "Normalize the request",
            "Score phases",
            "Select DNA",
            "Verify contract"
        ],
        "notes": "Code-repair baseline for a live router resilience task."
    }
}


def normalize_patch_skeleton_paths(value):
    normalized = json.loads(json.dumps(value))
    patch_skeleton = normalized.get("patch_skeleton")
    if isinstance(patch_skeleton, dict):
        patch_skeleton["source_module"] = "<normalized>"
        patch_skeleton["transfer_target"] = "<normalized>"
        code_skeleton = patch_skeleton.get("code_skeleton")
        if isinstance(code_skeleton, list) and code_skeleton:
            code_skeleton[-1] = 'return {"transfer_target": "<normalized>", ...}'
    return normalized


def load_research_module():
    class MockResearch:
        def derive_task_ecology(self, feed): return runtime.derive_task_ecology(feed)
        def derive_feed_dna_prior(self, tp): return runtime.derive_feed_dna_prior(tp)
        def build_patch_skeleton(self, tp, prior): return runtime.build_patch_skeleton(tp, prior)
        def run_v311(self, trials, population, base_episodes, generations, seed0, feed):
            return runtime.run_benchmark(trials=trials, population=population, base_episodes=base_episodes, generations=generations, seed0=seed0, feed=feed)
        def build_optimizer_frame(self, dna): return runtime.build_optimizer_frame(dna)
        def mine_macros_from_elites(self, sequences, max_macros=4): return runtime.mine_macros_from_elites(sequences, max_macros=max_macros)
    return MockResearch()


class BarricadeRuntimeTests(unittest.TestCase):
    @staticmethod
    def assert_json_like_equal(actual, expected, path="root"):
        if isinstance(expected, float):
            unittest.TestCase().assertAlmostEqual(
                float(actual), expected, places=9, msg=path
            )
            return

        if isinstance(expected, dict):
            testcase = unittest.TestCase()
            actual_keys = set(actual.keys()) - {"api_version"}
            expected_keys = set(expected.keys()) - {"api_version"}
            testcase.assertEqual(actual_keys, expected_keys, path)
            for key in expected:
                if key == "api_version":
                    continue
                BarricadeRuntimeTests.assert_json_like_equal(
                    actual[key], expected[key], f"{path}.{key}"
                )
            return

        if isinstance(expected, list):
            testcase = unittest.TestCase()
            testcase.assertEqual(len(actual), len(expected), path)
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                BarricadeRuntimeTests.assert_json_like_equal(
                    actual_item, expected_item, f"{path}[{index}]"
                )
            return

        unittest.TestCase().assertEqual(actual, expected, path)

    def test_contract_and_public_aliases(self):
        contract = runtime.benchmark_contract()

        self.assertEqual(
            contract["mode"], "miniops_crisis_ecology_v3_11_exploit_residency"
        )
        self.assertEqual(
            contract["required_inputs"],
            ["trials", "population", "base_episodes", "generations", "seed0", "feed"],
        )
        self.assertEqual(
            contract["helpers"],
            [
                "derive_task_ecology",
                "derive_feed_dna_prior",
                "build_patch_skeleton",
                "mine_macros_from_elites",
                "build_optimizer_frame",
                "run_benchmark_comparison",
                "run_ablation_study",
            ],
        )
        self.assertIs(runtime.run_benchmark, runtime.run_v311)

    def test_run_benchmark_comparison_prefers_better_candidate(self):
        baseline_result = {
            "summary": {
                "governed_mean": 78.0,
                "solve_rate_test_mean": 0.41,
                "stability_score_mean": 0.29,
                "task_threshold_pass_mean": 0.24,
                "economic_stability_mean": 0.21,
                "profit_mean": -6.0,
                "market_score_mean": -1.2,
                "rotation_market_signal": 0.12,
                "governance_variance": 1320.0,
                "motif_entropy": 0.41,
                "specialization_entropy": 0.36,
                "lineage_entropy": 0.44,
            },
            "best_governed": {"governed_fitness": 61.0, "fitness": 42.0, "stability_score": 0.28, "cognitive_stability": 0.31, "economic_stability": 0.21, "profit": -6.0, "task_threshold_pass": 0.24, "solve_rate_test": 0.41},
            "best_raw": {"governed_fitness": 38.0, "fitness": 26.0, "stability_score": 0.18, "cognitive_stability": 0.18, "economic_stability": 0.15, "profit": -8.0, "task_threshold_pass": 0.18, "solve_rate_test": 0.22},
            "controller_summary": {"regime_counts": {"balanced": 12}, "transition_count": 4, "loopback_count": 1, "exploit_entries": 1, "exploit_bounce_count": 0, "episodes_min": 12, "episodes_max": 18, "temperature_min": 0.42, "temperature_max": 0.61},
            "phase_summary": {"transition_count": 4, "regime_counts": {"balanced": 12}},
            "transition_events_head": [],
            "lineage_mutation_scale": {},
            "feed_profile": {},
        }
        candidate_result = {
            "summary": {
                "governed_mean": 101.0,
                "solve_rate_test_mean": 0.68,
                "stability_score_mean": 0.53,
                "task_threshold_pass_mean": 0.49,
                "economic_stability_mean": 0.44,
                "profit_mean": -1.4,
                "market_score_mean": 0.6,
                "rotation_market_signal": 0.29,
                "governance_variance": 980.0,
                "motif_entropy": 0.52,
                "specialization_entropy": 0.49,
                "lineage_entropy": 0.53,
            },
            "best_governed": {"governed_fitness": 94.0, "fitness": 67.0, "stability_score": 0.51, "cognitive_stability": 0.56, "economic_stability": 0.42, "profit": -1.0, "task_threshold_pass": 0.49, "solve_rate_test": 0.66},
            "best_raw": {"governed_fitness": 74.0, "fitness": 51.0, "stability_score": 0.41, "cognitive_stability": 0.45, "economic_stability": 0.33, "profit": -2.0, "task_threshold_pass": 0.42, "solve_rate_test": 0.51},
            "controller_summary": {"regime_counts": {"balanced": 10, "explore": 2}, "transition_count": 2, "loopback_count": 0, "exploit_entries": 0, "exploit_bounce_count": 0, "episodes_min": 12, "episodes_max": 18, "temperature_min": 0.38, "temperature_max": 0.58},
            "phase_summary": {"transition_count": 2, "regime_counts": {"balanced": 10, "explore": 2}},
            "transition_events_head": [],
            "lineage_mutation_scale": {},
            "feed_profile": {},
        }

        calls = iter([baseline_result, candidate_result, baseline_result, candidate_result])

        def fake_run_benchmark(**kwargs):
            return next(calls)

        original_run_benchmark = runtime.run_benchmark
        runtime.run_benchmark = fake_run_benchmark
        try:
            report = runtime.run_benchmark_comparison(
                feed={"source": "demo"},
                state_dir=None,
                compact=False,
            )
            compact_report = runtime.run_benchmark_comparison(
                feed={"source": "demo"},
                state_dir=None,
            )
        finally:
            runtime.run_benchmark = original_run_benchmark

        self.assertEqual(report["comparison"]["winner"], "candidate")
        self.assertGreater(report["comparison"]["score"]["delta"], 0.0)
        self.assertIs(report["baseline_result"], baseline_result)
        self.assertIs(report["candidate_result"], candidate_result)
        self.assertIn("baseline_summary", compact_report)
        self.assertIn("candidate_summary", compact_report)
        self.assertNotIn("baseline_result", compact_report)
        self.assertEqual(compact_report["comparison"]["winner"], "candidate")

    def test_run_benchmark_comparison_beats_unguided_baseline(self):
        baseline_config = EvolutionConfig(
            enable_parallax=False,
            enable_orthogonality=False,
            enable_rotation=False,
            enable_unified=False,
            enable_curriculum=False,
            enable_primitive_contracts=False,
        )

        report = runtime.run_benchmark_comparison(
            trials=2,
            population=12,
            base_episodes=6,
            generations=6,
            seed0=100901,
            feed={"source": "demo", "tasks": [{"text": "Add a login endpoint"}]},
            baseline_config=baseline_config.__dict__,
            candidate_config=None,
            baseline_label="unguided_baseline",
            candidate_label="engine",
            compact=False,
        )

        self.assertEqual(report["comparison"]["winner"], "engine")
        self.assertGreater(report["comparison"]["score"]["delta"], 0.0)
        self.assertGreater(
            report["candidate_result"]["summary"]["governed_mean"],
            report["baseline_result"]["summary"]["governed_mean"],
        )

    def test_run_ablation_study_compares_feature_flags(self):
        baseline_config = EvolutionConfig(
            enable_parallax=True,
            enable_orthogonality=True,
            enable_rotation=True,
            enable_unified=True,
            enable_curriculum=True,
            enable_primitive_contracts=True,
        )

        def feature_flags_for(config: object) -> dict[str, bool]:
            if isinstance(config, dict):
                getter = config.get
            else:
                def getter(key, default=False):
                    return getattr(config, key, default)
            return {
                "parallax": bool(getter("enable_parallax", False)),
                "orthogonality": bool(getter("enable_orthogonality", False)),
                "rotation": bool(getter("enable_rotation", False)),
                "unified": bool(getter("enable_unified", False)),
                "curriculum": bool(getter("enable_curriculum", False)),
                "primitive_contracts": bool(getter("enable_primitive_contracts", False)),
            }

        def build_result(config: object) -> dict[str, object]:
            feature_flags = feature_flags_for(config)
            score = 120.0
            if not feature_flags["parallax"]:
                score -= 10.0
            if not feature_flags["orthogonality"]:
                score -= 8.0
            if not feature_flags["rotation"]:
                score -= 6.0
            if not feature_flags["curriculum"]:
                score -= 4.0
            if not feature_flags["primitive_contracts"]:
                score -= 2.0
            return {
                "summary": {
                    "governed_mean": score,
                    "solve_rate_test_mean": score / 200.0,
                    "stability_score_mean": score / 240.0,
                    "task_threshold_pass_mean": score / 300.0,
                    "economic_stability_mean": score / 360.0,
                    "profit_mean": score - 125.0,
                    "market_score_mean": score - 118.0,
                    "rotation_market_signal": 0.4 if feature_flags["rotation"] else 0.2,
                    "governance_variance": 900.0,
                    "motif_entropy": 0.42,
                    "specialization_entropy": 0.37,
                    "lineage_entropy": 0.44,
                },
                "selection_report": {
                    "selection_mode": "unified" if feature_flags["unified"] else "exploit",
                    "feature_flags": feature_flags,
                    "pareto_front_count": 1,
                    "pareto_front_sizes": [2],
                    "probe_count": 1,
                    "elite_count": 1,
                    "rotation_signal": 0.2,
                    "rotation_strength": 0.3,
                    "forbidden_subsequence_count": 0,
                    "forbidden_memory_count": 0,
                    "contract_guard_token_count": 0,
                    "task_count": 2,
                },
                "rotation_analysis": {
                    "sample_size": 2,
                    "axis_labels": ["x", "y"],
                    "dominant_axis": "x",
                    "condition_number": 1.2,
                    "principal_eigenvalue": 1.0,
                    "secondary_eigenvalue": 0.8,
                    "principal_axis": [1.0, 0.0],
                    "secondary_axis": [0.0, 1.0],
                    "axis_variance": {},
                    "axis_range": {},
                },
                "parallax_telemetry": {
                    "pressure": 0.2,
                    "top_governed_mean": 2.0,
                    "bottom_governed_mean": 1.0,
                    "valley_depth": 1.0,
                    "gradient_steepness": 0.5,
                    "gradient_positive_tokens": ["PLAN"],
                    "gradient_negative_tokens": ["REPAIR"],
                    "failure_probe_count": 1,
                    "failure_probe_ratio": 0.5,
                    "forbidden_subsequence_count": 0,
                    "forbidden_subsequence_samples": [],
                    "memory_forbidden_count": 0,
                    "memory_forbidden_samples": [],
                },
                "best_governed": {
                    "artifact_id": "best-governed",
                    "token": "PLAN",
                    "kind": "plan",
                    "creator": "winner",
                    "epoch": 1,
                    "price": 1.0,
                    "score": score,
                    "fitness": score - 1.0,
                    "governed_fitness": score + 1.0,
                    "selection_score": 0.7,
                    "stability_score": 0.6,
                    "economic_stability": 0.55,
                    "task_threshold_pass": 0.64,
                    "solve_rate_test": 0.59,
                    "market_score": 0.5,
                    "profit": 0.25,
                    "wallet_end": 1.5,
                    "specialization": "generalist",
                    "family_sig": "F1",
                    "motif_sig": "M1",
                    "lineage_id": "lineage-best",
                    "status": "passed",
                    "dna": ["PLAN", "VERIFY"],
                    "parent_ids": ["root"],
                },
                "best_raw": {
                    "artifact_id": "best-raw",
                    "token": "WRITE_PATCH",
                    "kind": "patch",
                    "creator": "runner-up",
                    "epoch": 2,
                    "price": 1.0,
                    "score": score - 3.0,
                    "fitness": score - 5.0,
                    "governed_fitness": score - 4.0,
                    "selection_score": 0.6,
                    "stability_score": 0.5,
                    "economic_stability": 0.45,
                    "task_threshold_pass": 0.55,
                    "solve_rate_test": 0.52,
                    "market_score": 0.4,
                    "profit": 0.15,
                    "wallet_end": 1.2,
                    "specialization": "generalist",
                    "family_sig": "F2",
                    "motif_sig": "M2",
                    "lineage_id": "lineage-raw",
                    "status": "passed",
                    "dna": ["WRITE_PATCH", "VERIFY"],
                    "parent_ids": ["root"],
                },
                "controller_summary": {
                    "regime_counts": {"balanced": 1},
                    "transition_count": 1,
                    "loopback_count": 0,
                    "exploit_entries": 0,
                    "exploit_bounce_count": 0,
                    "episodes_min": 12,
                    "episodes_max": 18,
                    "temperature_min": 0.42,
                    "temperature_max": 0.61,
                },
                "phase_summary": {
                    "transition_count": 1,
                    "regime_counts": {"balanced": 1},
                },
                "outcome_record": {
                    "benchmark_signature": "signature",
                    "task_shape_signature": "shape",
                    "outcome_score": score,
                    "success": True,
                    "outcome_class": "success",
                    "summary": {
                        "governed_mean": score,
                        "solve_rate_test_mean": score / 200.0,
                        "stability_score_mean": score / 240.0,
                        "task_threshold_pass_mean": score / 300.0,
                        "specialization_entropy": 0.37,
                        "rotation_market_signal": 0.4 if feature_flags["rotation"] else 0.2,
                        "market_score_mean": score - 118.0,
                        "artifact_yield_mean": 0.75,
                        "gen_efficiency_mean": 0.68,
                    },
                    "metadata": {
                        "trials": 2,
                        "population": 12,
                        "state_dir": "",
                        "state_fingerprint": "fingerprint",
                        "benchmark_reused": False,
                    },
                },
            }

        def fake_run_benchmark(**kwargs):
            return build_result(kwargs.get("config"))

        original_run_benchmark = runtime.run_benchmark
        runtime.run_benchmark = fake_run_benchmark
        try:
            report = runtime.run_ablation_study(
                trials=2,
                population=12,
                base_episodes=6,
                generations=6,
                seed0=100901,
                feed={"source": "demo", "tasks": [{"text": "Add a login endpoint"}]},
                config=baseline_config,
                ablation_modes=("no_parallax", "minimal"),
            )
        finally:
            runtime.run_benchmark = original_run_benchmark

        self.assertEqual(report["baseline_summary"]["selection_report"]["feature_flags"]["parallax"], True)
        self.assertEqual(report["baseline_summary"]["summary"]["governed_mean"], 120.0)
        self.assertEqual(len(report["ablation_runs"]), 2)
        self.assertEqual(report["ablation_runs"][0]["mode"], "no_parallax")
        self.assertEqual(report["ablation_runs"][0]["feature_flags"]["parallax"], False)
        self.assertEqual(report["ablation_runs"][1]["mode"], "minimal")
        self.assertEqual(report["ablation_runs"][1]["feature_flags"]["rotation"], False)
        self.assertEqual(report["ablation_runs"][0]["comparison"]["winner"], "baseline")
        self.assertEqual(report["ablation_runs"][1]["comparison"]["winner"], "baseline")

    def test_core_helpers_match_research_source(self):
        research = load_research_module()

        sample_flat = [
            "OBSERVE",
            "LM1",
            "PLAN",
            "WRITE_PATCH",
            "VERIFY",
            "SUMMARIZE",
            "LM2",
        ]
        self.assert_json_like_equal(
            runtime.build_optimizer_frame(sample_flat),
            research.build_optimizer_frame(sample_flat),
        )

        elite_sequences = [
            ["PLAN", "REPAIR", "VERIFY", "SUMMARIZE"],
            ["OBSERVE", "PLAN", "WRITE_PLAN", "VERIFY"],
            ["WRITE_SUMMARY", "VERIFY", "SUMMARIZE"],
        ]
        self.assert_json_like_equal(
            runtime.mine_macros_from_elites(elite_sequences, max_macros=4),
            research.mine_macros_from_elites(elite_sequences, max_macros=4),
        )

    def test_recursive_macros_expand_and_reseed(self):
        seed_macros = {"LM1": ["PLAN", "VERIFY"]}
        recursive_macros = {"LM2": ["LM1", "SUMMARIZE", "WRITE_PLAN"]}

        self.assertEqual(
            recursive_flatten_trace(["OBSERVE", "LM2", "WRITE_PATCH"], {**seed_macros, **recursive_macros}),
            ["OBSERVE", "PLAN", "VERIFY", "SUMMARIZE", "WRITE_PLAN", "WRITE_PATCH"],
        )

        elite_sequences = [
            ["PLAN", "VERIFY", "SUMMARIZE", "WRITE_PLAN", "REPAIR"],
            ["PLAN", "VERIFY", "SUMMARIZE", "WRITE_PLAN", "COMMIT"],
            ["PLAN", "VERIFY", "SUMMARIZE", "WRITE_PLAN", "VERIFY"],
        ]
        learned = runtime.mine_macros_from_elites(
            elite_sequences,
            max_macros=1,
            macro_lib=seed_macros,
        )

        self.assertEqual(learned, recursive_macros)
        self.assertEqual(
            recursive_flatten_trace(["LM2"], {**seed_macros, **learned}),
            ["PLAN", "VERIFY", "SUMMARIZE", "WRITE_PLAN"],
        )

    def test_selection_value_keeps_zero_score(self):
        from barricade.feed_derived_dna.evolution import _selection_value
        from barricade.feed_derived_dna.models import Individual

        individual = Individual(
            dna=[],
            chart="ANALYSIS",
            lineage_id="root",
            selection_score=0.0,
            governed_fitness=99.0,
            fitness_vector=[0.0],
            selection_axes={"train_score": 0.0},
        )

        self.assertEqual(_selection_value(individual), 0.0)
        self.assertEqual(
            _selection_value(
                {
                    "selection_score": 0.0,
                    "governed_fitness": 99.0,
                    "fitness_vector": [0.0],
                    "selection_axes": {"train_score": 0.0},
                }
            ),
            0.0,
        )

    def test_outcome_memory_biases_macro_promotion(self):
        elite_sequences = [
            ["PLAN", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
            ["PLAN", "WRITE_PLAN", "VERIFY", "COMMIT"],
            ["PLAN", "WRITE_PLAN", "VERIFY", "WRITE_SUMMARY"],
            ["OBSERVE", "PLAN", "WRITE_PLAN", "VERIFY"],
        ]
        outcome_memory = {
            "available": True,
            "success_trace_bank": [
                {
                    "trace": ["OBSERVE", "PLAN", "WRITE_PLAN"],
                    "weight": 4.0,
                    "source": "best_governed",
                    "benchmark_signature": "sig-success",
                    "outcome_score": 184.0,
                    "similarity": 1.0,
                }
            ],
        }

        baseline = runtime.mine_macros_from_elites(elite_sequences, max_macros=1, lengths=(3,))
        boosted = runtime.mine_macros_from_elites(
            elite_sequences,
            max_macros=1,
            lengths=(3, 4),
            outcome_memory=outcome_memory,
        )

        self.assertEqual(baseline, {"LM1": ["PLAN", "WRITE_PLAN", "VERIFY"]})
        self.assertEqual(boosted, {"LM1": ["OBSERVE", "PLAN", "WRITE_PLAN"]})

    def test_feed_derived_helpers_match_archived_v314_outputs(self):
        research = load_research_module()
        cases = [
            (MINIOPS_CRISIS_ECOLOGY_V3_12_RAFT_NODE_SCENARIO, "raft_node_scenario"),
            (MINIOPS_CRISIS_ECOLOGY_V3_12_SUMMARY_MEMO_SCENARIO, "summary_memo_scenario"),
            (MINIOPS_CRISIS_ECOLOGY_V3_12_COMPLEX_SCENARIO, "complex_scenario"),
        ]

        for scenario, scenario_name in cases:
            with self.subTest(scenario=scenario_name):
                expected_task_pool = research.derive_task_ecology(scenario)

                derived_task_pool = runtime.derive_task_ecology(scenario)
                self.assert_json_like_equal(derived_task_pool, expected_task_pool)

                derived_prior = runtime.derive_feed_dna_prior(derived_task_pool)
                self.assert_json_like_equal(
                    derived_prior, research.derive_feed_dna_prior(expected_task_pool)
                )

        router_scenario = MINIOPS_CRISIS_ECOLOGY_V3_12_ROUTER_REPAIR_SCENARIO
        router_task_pool = research.derive_task_ecology(router_scenario)
        router_prior = research.derive_feed_dna_prior(router_task_pool)
        router_expected_skeleton = research.build_patch_skeleton(
            router_task_pool, router_prior
        )

        router_task_pool = runtime.derive_task_ecology(router_scenario)
        router_prior = runtime.derive_feed_dna_prior(router_task_pool)
        router_skeleton = runtime.build_patch_skeleton(router_task_pool, router_prior)

        self.assert_json_like_equal(
            normalize_patch_skeleton_paths(router_skeleton),
            normalize_patch_skeleton_paths(router_expected_skeleton),
        )

    def test_run_benchmark_matches_research_source_for_smoke_case(self):
        research = load_research_module()
        feed = MINIOPS_CRISIS_ECOLOGY_V3_12_SUMMARY_MEMO_SCENARIO
        expected_result = research.run_v311(
            trials=1, population=8, base_episodes=2, generations=2, seed0=7, feed=feed
        )

        first_result = runtime.run_benchmark(
            trials=1,
            population=8,
            base_episodes=2,
            generations=2,
            seed0=7,
            feed=feed,
        )
        second_result = runtime.run_benchmark(
            trials=1,
            population=8,
            base_episodes=2,
            generations=2,
            seed0=7,
            feed=feed,
        )

        self.assert_json_like_equal(
            normalize_patch_skeleton_paths(first_result),
            normalize_patch_skeleton_paths(second_result),
        )

        derived_task_pool = runtime.derive_task_ecology(feed)
        derived_prior = runtime.derive_feed_dna_prior(derived_task_pool)
        runtime.build_patch_skeleton(
            derived_task_pool, derived_prior
        )

        self.assertEqual(
            first_result["mode"], "miniops_crisis_ecology_v3_11_exploit_residency"
        )
        self.assert_json_like_equal(first_result["task_pool"], derived_task_pool)
        self.assert_json_like_equal(first_result["feed_prior_dna"], derived_prior)
        self.assert_json_like_equal(
            normalize_patch_skeleton_paths(first_result),
            normalize_patch_skeleton_paths(expected_result),
        )

    def test_run_benchmark_matches_archived_feed_outputs(self):
        research = load_research_module()
        cases = [
            (MINIOPS_CRISIS_ECOLOGY_V3_12_RAFT_NODE_SCENARIO, "raft_node_scenario"),
            (MINIOPS_CRISIS_ECOLOGY_V3_12_COMPLEX_SCENARIO, "complex_scenario"),
            (MINIOPS_CRISIS_ECOLOGY_V3_12_SUMMARY_MEMO_SCENARIO, "summary_memo_scenario"),
        ]

        for scenario, scenario_name in cases:
            with self.subTest(scenario=scenario_name):
                expected_result = research.run_v311(
                    trials=1,
                    population=8,
                    base_episodes=2,
                    generations=2,
                    seed0=7,
                    feed=scenario,
                )

                first_result = runtime.run_benchmark(
                    trials=1,
                    population=8,
                    base_episodes=2,
                    generations=2,
                    seed0=7,
                    feed=scenario,
                )
                second_result = runtime.run_benchmark(
                    trials=1,
                    population=8,
                    base_episodes=2,
                    generations=2,
                    seed0=7,
                    feed=scenario,
                )

                self.assert_json_like_equal(
                    normalize_patch_skeleton_paths(first_result),
                    normalize_patch_skeleton_paths(second_result),
                )

                derived_task_pool = runtime.derive_task_ecology(scenario)
                derived_prior = runtime.derive_feed_dna_prior(derived_task_pool)
                runtime.build_patch_skeleton(
                    derived_task_pool, derived_prior
                )

                self.assertEqual(
                    first_result["mode"],
                    "miniops_crisis_ecology_v3_11_exploit_residency",
                )
                self.assert_json_like_equal(
                    first_result["task_pool"], derived_task_pool
                )
                self.assert_json_like_equal(
                    first_result["feed_prior_dna"], derived_prior
                )
                self.assert_json_like_equal(
                    normalize_patch_skeleton_paths(first_result),
                    normalize_patch_skeleton_paths(expected_result),
                )

    def test_mcp_tools_match_runtime_outputs(self):
        feed = MINIOPS_CRISIS_ECOLOGY_V3_12_ROUTER_REPAIR_SCENARIO
        feed_json = json.dumps(feed)

        task_pool = runtime.derive_task_ecology(feed)
        prior = runtime.derive_feed_dna_prior(task_pool)
        skeleton = runtime.build_patch_skeleton(task_pool, prior)
        tokens = ["PLAN", "REPAIR", "VERIFY", "SUMMARIZE", "WRITE_SUMMARY"]
        elite_sequences = [
            ["PLAN", "REPAIR", "VERIFY", "SUMMARIZE"],
            ["OBSERVE", "PLAN", "WRITE_PLAN", "VERIFY"],
            ["WRITE_SUMMARY", "VERIFY", "SUMMARIZE"],
        ]

        self.assert_json_like_equal(
            mcp_server.get_benchmark_contract(), runtime.benchmark_contract()
        )
        self.assert_json_like_equal(
            mcp_server.derive_task_ecology_from_feed(feed_json), task_pool
        )
        self.assert_json_like_equal(mcp_server.derive_feed_prior(feed_json), prior)
        self.assert_json_like_equal(
            mcp_server.build_patch_skeleton_from_feed(feed_json), skeleton
        )
        self.assert_json_like_equal(
            mcp_server.summarize_optimizer_frame(tokens),
            runtime.build_optimizer_frame(tokens),
        )
        self.assert_json_like_equal(
            mcp_server.mine_macros(elite_sequences, max_macros=4),
            runtime.mine_macros_from_elites(elite_sequences, max_macros=4),
        )

        benchmark_result = mcp_server.run_benchmark_task(
            trials=1,
            population=8,
            base_episodes=2,
            generations=2,
            seed0=7,
            feed_json=feed_json,
            compact=False,
        )
        compact_benchmark_result = mcp_server.run_benchmark_task(
            trials=1,
            population=8,
            base_episodes=2,
            generations=2,
            seed0=7,
            feed_json=feed_json,
            compact=True,
        )
        runtime_result = runtime.run_benchmark(
            trials=1,
            population=8,
            base_episodes=2,
            generations=2,
            seed0=7,
            feed=feed,
        )
        self.assert_json_like_equal(
            normalize_patch_skeleton_paths(benchmark_result),
            normalize_patch_skeleton_paths(runtime_result),
        )
        self.assertLess(len(json.dumps(compact_benchmark_result)), len(json.dumps(benchmark_result)))
        self.assertIn("top_artifacts", compact_benchmark_result)
        self.assertIn("outcome_record_summary", compact_benchmark_result)
        self.assertNotIn("best_governed", compact_benchmark_result)

    def test_v314_comparison_notes_match_library_outputs(self):
        comparison_text = (Path(__file__).resolve().parents[1] / "docs" / "history.md").read_text()

        cases = [
            (
                MINIOPS_CRISIS_ECOLOGY_V3_12_RAFT_NODE_SCENARIO,
                "OBSERVE -> LM1 -> WRITE_PATCH -> PLAN -> REPAIR -> COMMIT -> WRITE_PLAN -> VERIFY -> SUMMARIZE",
                "solve 1, threshold 0.8, stability 0.451",
            ),
            (
                MINIOPS_CRISIS_ECOLOGY_V3_12_COMPLEX_SCENARIO,
                "OBSERVE -> LM1 -> WRITE_PATCH -> PLAN -> COMMIT -> WRITE_PLAN -> ROLLBACK -> REPAIR -> VERIFY -> SUMMARIZE",
                "solve 0.714, threshold 0.5, stability 0.488",
            ),
            (
                MINIOPS_CRISIS_ECOLOGY_V3_12_SUMMARY_MEMO_SCENARIO,
                "OBSERVE -> LM1 -> PLAN -> RETRIEVE -> WRITE_SUMMARY -> VERIFY -> SUMMARIZE -> WRITE_PLAN",
                "solve 1, threshold 0.615, stability 0.405",
            ),
        ]

        for scenario, expected_prior, expected_result_fragment in cases:
            task_pool = runtime.derive_task_ecology(scenario)
            prior = runtime.derive_feed_dna_prior(task_pool)
            self.assertEqual(" -> ".join(prior), expected_prior)
            self.assertIn(expected_prior, comparison_text)
            self.assertIn(expected_result_fragment, comparison_text)

        self.assertIn(
            "The direction is working: let the input shape the bounded ecology",
            comparison_text,
        )

    def test_odd_feed_shapes_remain_coherent(self):
        research = load_research_module()
        cases = [
            (
                "empty_dict",
                {},
                1,
                ["OBSERVE", "LM1", "WRITE_PATCH", "PLAN", "REPAIR", "COMMIT", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
                "patching",
                ["WRITE_PATCH", "PLAN", "REPAIR", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
            ),
            (
                "nested_weird",
                {
                    "alpha": [
                        "",
                        None,
                        {
                            "beta": [
                                "plan",
                                True,
                                3.14,
                                {
                                    "gamma": "PATCH",
                                    "delta": ["recover", {"z": "summary"}],
                                },
                            ]
                        },
                    ],
                    "meta": {"notes": "verify rpc and rollback"},
                },
                6,
                [
                    "OBSERVE",
                    "LM1",
                    "WRITE_PATCH",
                    "PLAN",
                    "COMMIT",
                    "WRITE_PLAN",
                    "ROLLBACK",
                    "REPAIR",
                    "VERIFY",
                    "SUMMARIZE",
                ],
                "recovery",
                [
                    "WRITE_PATCH",
                    "PLAN",
                    "WRITE_PLAN",
                    "ROLLBACK",
                    "REPAIR",
                    "VERIFY",
                    "SUMMARIZE",
                ],
            ),
            (
                "odd_list",
                [
                    None,
                    "",
                    {
                        "x": "design structure",
                        "y": [42, False, {"z": "summarize review memo"}],
                    },
                ],
                5,
                [
                    "OBSERVE",
                    "LM1",
                    "PLAN",
                    "RETRIEVE",
                    "WRITE_SUMMARY",
                    "VERIFY",
                    "SUMMARIZE",
                    "WRITE_PLAN",
                ],
                "summarizing",
                [
                    "WRITE_PATCH",
                    "PLAN",
                    "WRITE_SUMMARY",
                    "VERIFY",
                    "SUMMARIZE",
                    "WRITE_PLAN",
                ],
            ),
        ]

        for (
            name,
            feed,
            expected_task_count,
            expected_prior,
            expected_primary_focus,
            expected_token_prefix,
        ) in cases:
            with self.subTest(name=name):
                task_pool = runtime.derive_task_ecology(feed)
                prior = runtime.derive_feed_dna_prior(task_pool)
                skeleton = runtime.build_patch_skeleton(task_pool, prior)
                expected_skeleton = research.build_patch_skeleton(task_pool, prior)

                self.assertEqual(len(task_pool), expected_task_count)
                self.assertEqual(prior, expected_prior)
                self.assertEqual(skeleton["primary_focus"], expected_primary_focus)
                self.assertEqual(skeleton["token_outline"], expected_token_prefix)
                self.assert_json_like_equal(
                    normalize_patch_skeleton_paths(skeleton),
                    normalize_patch_skeleton_paths(expected_skeleton),
                )
                self.assertTrue(
                    all(
                        isinstance(task, dict) and task.get("name")
                        for task in task_pool
                    )
                )

    def test_mcp_handles_odd_feed_json(self):
        feed = {
            "alpha": [
                "",
                None,
                0,
                {
                    "beta": [
                        "plan",
                        True,
                        3.14,
                        {"gamma": "PATCH", "delta": ["recover", {"z": "summary"}]},
                    ]
                },
            ],
            "meta": {"notes": "verify rpc and rollback"},
        }
        feed_json = json.dumps(feed)

        task_pool = runtime.derive_task_ecology(feed)
        prior = runtime.derive_feed_dna_prior(task_pool)
        skeleton = runtime.build_patch_skeleton(task_pool, prior)

        self.assert_json_like_equal(
            mcp_server.derive_task_ecology_from_feed(feed_json), task_pool
        )
        self.assert_json_like_equal(mcp_server.derive_feed_prior(feed_json), prior)
        self.assert_json_like_equal(
            mcp_server.build_patch_skeleton_from_feed(feed_json), skeleton
        )

    def test_odd_feed_benchmark_is_deterministic(self):
        research = load_research_module()
        feed = [
            None,
            "",
            {"x": "design structure", "y": [42, False, {"z": "summarize review memo"}]},
        ]
        expected_result = research.run_v311(
            trials=1, population=8, base_episodes=2, generations=2, seed0=11, feed=feed
        )

        first_result = runtime.run_benchmark(
            trials=1, population=8, base_episodes=2, generations=2, seed0=11, feed=feed
        )
        second_result = runtime.run_benchmark(
            trials=1, population=8, base_episodes=2, generations=2, seed0=11, feed=feed
        )

        self.assertEqual(first_result, second_result)
        self.assertEqual(
            first_result["mode"], "miniops_crisis_ecology_v3_11_exploit_residency"
        )
        self.assert_json_like_equal(
            normalize_patch_skeleton_paths(first_result),
            normalize_patch_skeleton_paths(expected_result),
        )


if __name__ == "__main__":
    unittest.main()
