from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from barricade import runtime
from barricade.executor.registry import Artifact, ExecutionSession, REGISTRY
from barricade.feed_derived_dna._outcome_memory import build_outcome_record, load_outcome_memory
from barricade.feed_derived_dna.persistence import (
    benchmark_run_signature,
    benchmark_state_fingerprint,
    RUN_LOG_FILE,
    load_forbidden_subsequence_memory,
    load_execution_feedback,
    load_benchmark_run,
    load_macro_library,
    load_outcome_ledger,
    load_motif_cache,
    save_execution_feedback,
    save_forbidden_subsequence_memory,
    save_outcome_ledger,
    task_shape_signature,
)


ROOT = Path(__file__).resolve().parents[1]

MINIOPS_CRISIS_SCENARIO = {
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


class Phase1PersistenceTests(unittest.TestCase):
    def test_state_dir_persists_macros_lineages_and_runs(self):
        feed = MINIOPS_CRISIS_SCENARIO

        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)

            first_result = runtime.run_benchmark(
                trials=1,
                population=8,
                base_episodes=2,
                generations=2,
                seed0=7,
                feed=feed,
                state_dir=str(state_dir),
            )
            second_result = runtime.run_benchmark(
                trials=1,
                population=8,
                base_episodes=2,
                generations=2,
                seed0=7,
                feed=feed,
                state_dir=str(state_dir),
            )

            macro_path = state_dir / "discoverables" / "macro_library.json"
            motif_path = state_dir / "discoverables" / "motif_cache.json"
            lineage_path = state_dir / "lineages.jsonl"
            run_path = state_dir / "runs.jsonl"
            benchmark_run_path = state_dir / "benchmark_runs.jsonl"
            outcome_path = state_dir / "outcome_ledger.jsonl"

            self.assertTrue(macro_path.exists())
            self.assertTrue(motif_path.exists())
            self.assertTrue(lineage_path.exists())
            self.assertTrue(run_path.exists())
            self.assertTrue(benchmark_run_path.exists())
            self.assertTrue(outcome_path.exists())

            macro_payload = json.loads(macro_path.read_text())
            motif_payload = json.loads(motif_path.read_text())
            lineage_entries = [json.loads(line) for line in lineage_path.read_text().splitlines() if line.strip()]
            run_entries = [json.loads(line) for line in run_path.read_text().splitlines() if line.strip()]
            benchmark_entries = [json.loads(line) for line in benchmark_run_path.read_text().splitlines() if line.strip()]
            outcome_entries = [json.loads(line) for line in outcome_path.read_text().splitlines() if line.strip()]

            self.assertIn("macros", macro_payload)
            self.assertIn("motifs", motif_payload)
            self.assertTrue(macro_payload["macros"])
            self.assertTrue(motif_payload["motifs"])
            self.assertEqual(macro_payload["macros"], second_result["learned_macros"])
            self.assertIn("selection_report", first_result)
            self.assertIn("rotation_analysis", first_result)
            self.assertIn("parallax_telemetry", first_result)
            self.assertIn("forbidden_subsequence_memory", first_result)
            self.assertEqual(first_result["selection_report"]["selection_mode"], "unified")
            self.assertGreaterEqual(first_result["selection_report"]["pareto_front_count"], 1)
            self.assertGreaterEqual(first_result["rotation_analysis"]["condition_number"], 0.0)
            self.assertGreaterEqual(first_result["parallax_telemetry"]["failure_probe_count"], 0)
            self.assertGreaterEqual(first_result["forbidden_subsequence_memory"]["pattern_count"], 0)
            self.assertEqual(run_entries[-1]["metadata"]["state_dir"], str(state_dir))
            self.assertEqual(run_entries[-1]["summary"], second_result["summary"])
            self.assertEqual(len(benchmark_entries), 2)
            self.assertEqual(len(outcome_entries), 2)
            self.assertTrue(all(entry["kind"] == "benchmark" for entry in benchmark_entries))
            self.assertTrue(all(entry["kind"] == "outcome" for entry in outcome_entries))
            self.assertTrue(all(entry["task_shape_signature"] for entry in outcome_entries))
            self.assertTrue(all("outcome_score" in entry for entry in outcome_entries))

            self.assertEqual({entry["label"] for entry in lineage_entries}, {"winners", "archive_raw", "archive_stable"})
            self.assertTrue(all(entry["count"] > 0 for entry in lineage_entries))
            self.assertEqual(len(run_entries), 2)
            self.assertTrue(set(first_result["learned_macros"]).issubset(second_result["learned_macros"]))

    def test_outcome_memory_ranks_best_success_and_failure(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)
            shape_profile = {
                "task_count": 3,
                "focus_counts": {"planning": 2, "patching": 1},
                "need_counts": {"plan": 2, "patch": 1},
                "task_names": ["alpha", "beta"],
                "feed_prior_dna": ["PLAN", "WRITE_PATCH", "VERIFY"],
                "curriculum_stage_counts": {"planning": 3},
                "curriculum_stages": ["planning"],
                "mode": "benchmark",
            }

            success_record = build_outcome_record(
                {
                    "summary": {
                        "governed_mean": 182.0,
                        "solve_rate_test_mean": 0.95,
                        "stability_score_mean": 0.86,
                        "task_threshold_pass_mean": 0.91,
                        "specialization_entropy": 0.33,
                        "rotation_market_signal": 0.71,
                        "market_score_mean": 0.48,
                        "artifact_yield_mean": 0.61,
                        "gen_efficiency_mean": 0.55,
                    },
                    "controller_summary": {"transition_count": 5},
                    "phase_summary": {"transition_count": 6},
                    "task_pool": [],
                    "feed_prior_dna": [],
                    "best_governed": {"id": "success"},
                    "best_raw": {"id": "success_raw"},
                },
                "sig-success",
                shape_profile,
                metadata={"case": "success"},
                source="benchmark",
            )
            failure_record = build_outcome_record(
                {
                    "summary": {
                        "governed_mean": 78.0,
                        "solve_rate_test_mean": 0.41,
                        "stability_score_mean": 0.29,
                        "task_threshold_pass_mean": 0.24,
                        "specialization_entropy": 0.79,
                        "rotation_market_signal": 0.14,
                        "market_score_mean": 0.10,
                        "artifact_yield_mean": 0.12,
                        "gen_efficiency_mean": 0.18,
                    },
                    "controller_summary": {"transition_count": 2},
                    "phase_summary": {"transition_count": 3},
                    "task_pool": [],
                    "feed_prior_dna": [],
                    "best_governed": {"id": "failure"},
                    "best_raw": {"id": "failure_raw"},
                },
                "sig-failure",
                shape_profile,
                metadata={"case": "failure"},
                source="benchmark",
            )

            save_outcome_ledger(state_dir, failure_record)
            save_outcome_ledger(state_dir, success_record)

            ledger = load_outcome_ledger(state_dir)
            memory = load_outcome_memory(state_dir, shape_profile, benchmark_signature="sig-success")

            assert len(ledger) == 2
            assert ledger[-1]["task_shape_signature"] == task_shape_signature(shape_profile)
            assert memory["available"] is True
            assert memory["match_count"] == 2
            assert memory["exact_match"]["benchmark_signature"] == "sig-success"
            assert memory["best_success"]["benchmark_signature"] == "sig-success"
            assert memory["best_failure"]["benchmark_signature"] == "sig-failure"

    def test_forbidden_subsequence_memory_round_trip(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)
            save_forbidden_subsequence_memory(
                state_dir,
                [["PLAN", "REPAIR", "VERIFY"], ["WRITE_PATCH", "VERIFY"]],
                metadata={"source": "test"},
                score=0.75,
            )

            memory = load_forbidden_subsequence_memory(state_dir)

            assert memory["available"] is True
            assert memory["pattern_count"] == 2
            assert ["PLAN", "REPAIR", "VERIFY"] in memory["patterns"]
            assert memory["top_pattern_records"]
            assert memory["top_pattern_records"][0]["score"] >= 0.0

            fingerprint_a = benchmark_state_fingerprint({}, {}, memory)
            fingerprint_b = benchmark_state_fingerprint(
                {},
                {},
                {"patterns": [["OTHER", "PATTERN"]]},
            )
            assert fingerprint_a != fingerprint_b

    def test_execution_feedback_round_trip(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)
            feedback_record = {
                "kind": "execution_feedback",
                "session_id": "exec_1",
                "task_shape_signature": "shape-1",
                "route_hint": "patching",
                "decision_policy": {"mode": "act", "support_score": 0.84},
                "completion_summary": {
                    "verification_passed": True,
                    "verification_pass_rate": 1.0,
                    "successful_trace_count": 2,
                    "learned_macro_count": 1,
                    "artifact_count": 4,
                    "market_total_score": 12.5,
                    "market_total_price": 8.25,
                },
                "verification_result": {"passed": True},
                "execution_learning": {"successful_trace_count": 2, "learned_macro_count": 1},
                "artifact_count": 4,
                "market_count": 4,
                "dispatch_update_count": 1,
            }

            save_execution_feedback(state_dir, feedback_record)
            loaded = load_execution_feedback(state_dir)

            assert len(loaded) == 1
            assert loaded[0]["kind"] == "execution_feedback"
            assert loaded[0]["task_shape_signature"] == "shape-1"
            assert loaded[0]["completion_summary"]["verification_passed"] is True

    def test_complete_execution_persists_feedback(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)
            session_id = "exec_feedback"
            session = ExecutionSession(
                session_id=session_id,
                task_text="Patch the repository to add a unified workflow and verify it before commit.",
                dna=[],
                flattened=[],
                current_step=0,
                revision=1,
                context={
                    "task_shape_signature": "shape-feedback",
                    "route_hint": "patching",
                    "decision_policy": {"mode": "act", "support_score": 0.82},
                    "synthesis": {
                        "task_pool": [],
                        "summary": {},
                        "task_shape_signature": "shape-feedback",
                        "decision_policy": {"mode": "act", "support_score": 0.82},
                    },
                },
                state_root=state_dir,
            )

            REGISTRY._sessions[session_id] = session
            try:
                payload = REGISTRY.complete_execution(session_id)
            finally:
                REGISTRY._sessions.pop(session_id, None)

            loaded = load_execution_feedback(state_dir)

            assert payload["completion_summary"]["status"] == "completed"
            assert loaded[-1]["kind"] == "execution_feedback"
            assert loaded[-1]["task_shape_signature"] == "shape-feedback"
            assert loaded[-1]["decision_policy"]["mode"] == "act"

    def test_complete_execution_persists_best_artifact_snapshot(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            state_dir = Path(tmp_dir)
            session_id = "exec_best_artifact"
            session = ExecutionSession(
                session_id=session_id,
                task_text="Patch the repository to add a unified workflow and verify it before commit.",
                dna=[],
                flattened=[],
                current_step=0,
                revision=1,
                context={
                    "synthesis": {
                        "task_pool": [],
                        "summary": {},
                    },
                },
                state_root=state_dir,
            )

            top_artifact = Artifact(
                artifact_id="ART_01",
                token="WRITE_PATCH",
                kind="patch",
                content="print('top')\n",
                creator="host",
                epoch=1,
                price=9.5,
                score=8.5,
                status="submitted",
                metadata={"specialization": "patch"},
            )
            later_artifact = Artifact(
                artifact_id="ART_02",
                token="SUMMARIZE",
                kind="summary",
                content="later artifact\n",
                creator="host",
                epoch=2,
                price=1.5,
                score=1.0,
                status="submitted",
                metadata={"specialization": "summary"},
            )
            session.market[top_artifact.artifact_id] = top_artifact
            session.market[later_artifact.artifact_id] = later_artifact

            REGISTRY._sessions[session_id] = session
            try:
                payload = REGISTRY.complete_execution(session_id)
            finally:
                REGISTRY._sessions.pop(session_id, None)

            run_log = [
                json.loads(line)
                for line in (state_dir / RUN_LOG_FILE).read_text().splitlines()
                if line.strip()
            ]

            assert payload["completion_summary"]["top_artifact"]["artifact_id"] == "ART_01"
            assert run_log[-1]["best_governed"]["artifact_id"] == "ART_01"
            assert run_log[-1]["best_raw"]["artifact_id"] == "ART_01"


def test_run_benchmark_reuses_archived_result(monkeypatch, tmp_path) -> None:
    from barricade.feed_derived_dna import pipeline

    archived_result = {
        "mode": "miniops_crisis_ecology_v3_11_exploit_residency",
        "summary": {"score": 0.91},
        "best_governed": {"id": "governed"},
        "best_raw": {"id": "raw"},
        "archive_stable_top3": [],
        "archive_raw_top3": [],
        "controller_summary": {"mode": "balanced"},
        "phase_summary": [],
        "transition_events_head": [],
        "lineage_mutation_scale": {},
        "feed_profile": {},
        "learned_macros": {},
        "task_pool": [],
        "feed_prior_dna": [],
    }

    monkeypatch.setattr(pipeline, "load_benchmark_run", lambda state_root, signature: archived_result)

    def fail_ecology_round(*args, **kwargs):
        raise AssertionError("archived benchmark should have been reused")

    monkeypatch.setattr(pipeline, "ecology_round", fail_ecology_round)
    monkeypatch.setattr(pipeline, "save_discoverables", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "save_lineage_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "save_run_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "save_benchmark_run", lambda *args, **kwargs: None)

    feed = MINIOPS_CRISIS_SCENARIO

    result = runtime.run_benchmark(
        trials=1,
        population=8,
        base_episodes=2,
        generations=2,
        seed0=7,
        feed=feed,
        state_dir=str(tmp_path),
    )

    assert result is archived_result


def test_successful_benchmark_traces_skip_failed_elites() -> None:
    from barricade.feed_derived_dna import pipeline

    failed = pipeline.Individual(dna=["FAIL_A", "FAIL_B", "FAIL_C"], chart="chart", lineage_id="failed")
    failed.governed_fitness = 99.0
    failed.task_threshold_pass = 0.10
    failed.stability_score = 0.20

    successful = pipeline.Individual(dna=["OBSERVE", "PLAN", "VERIFY"], chart="chart", lineage_id="successful")
    successful.governed_fitness = 10.0
    successful.task_threshold_pass = 0.95
    successful.stability_score = 0.80

    traces = pipeline._successful_benchmark_traces([failed, successful], {})

    assert traces == [["OBSERVE", "PLAN", "VERIFY"], ["OBSERVE", "PLAN", "VERIFY"]]


def test_corrupted_cached_state_is_ignored_and_rebuilt(tmp_path) -> None:
    state_dir = tmp_path / "state"
    discoverables_dir = state_dir / "discoverables"
    discoverables_dir.mkdir(parents=True)

    (discoverables_dir / "macro_library.json").write_text("{broken")
    (discoverables_dir / "motif_cache.json").write_text("{broken")

    signature = benchmark_run_signature(
        {"feed": "demo"}, benchmark_state_fingerprint({}, {})
    )
    benchmark_path = state_dir / "benchmark_runs" / f"{signature}.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("{broken")

    assert load_macro_library(state_dir) == {}
    assert load_motif_cache(state_dir) == {}
    assert load_benchmark_run(state_dir, signature) == {}

    feed = MINIOPS_CRISIS_SCENARIO
    result = runtime.run_benchmark(
        trials=1,
        population=8,
        base_episodes=2,
        generations=2,
        seed0=7,
        feed=feed,
        state_dir=str(state_dir),
    )

    assert result["summary"]
    assert load_macro_library(state_dir) == result["learned_macros"]
