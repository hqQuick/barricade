from __future__ import annotations

import shutil
from pathlib import Path

from tests import proof_probe


def _make_probe_dirs(base: Path) -> tuple[Path, Path, Path]:
    shutil.rmtree(base, ignore_errors=True)
    workspace = base / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir = base / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = base / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    return workspace, state_dir, probe_dir


def test_novel_capability_synthesis_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path("/Users/wojtek/barricade"))
    base = Path("/Users/wojtek/barricade") / "tests" / "tmp" / "proof_probe_evidence" / tmp_path.name
    workspace, state_dir, probe_dir = _make_probe_dirs(base / "proof_evidence")

    quality = proof_probe.test_solution_quality(workspace, state_dir)
    transfer = proof_probe.test_novel_domain_transfer(probe_dir, workspace)
    learning = proof_probe.test_long_term_learning(workspace, state_dir)
    necessity = proof_probe.test_scaffold_necessity(workspace, state_dir)

    assert quality["quality_score"] == 2
    assert quality["avg_market"] >= 11.0
    assert all(metric["solved"] for metric in quality["metrics"])
    assert all(metric["has_verification"] for metric in quality["metrics"])
    assert all(metric["has_repair"] for metric in quality["metrics"])
    assert all(metric["has_observation"] for metric in quality["metrics"])

    assert transfer["train_success"] >= 2
    assert transfer["novel_success"] >= 3
    assert transfer["dna_overlap"] == 1
    assert transfer["transfer_score"] == 4

    assert learning["learning_delta"] > 0
    assert learning["cache_hits"] == 5
    assert learning["learning_score"] == 3

    assert necessity["minimal_solved"] is True
    assert necessity["full_solved"] is True
    assert necessity["minimal_dna_length"] == 9
    assert necessity["full_dna_length"] == 9
    assert necessity["has_lm_in_full"] is True
    assert necessity["necessity_score"] == 2