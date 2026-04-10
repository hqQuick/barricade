from __future__ import annotations

import shutil
from pathlib import Path

from tests.extrapolation_probe import (
    run_generalization_gradient,
    run_intervention_experiment,
    run_macro_essence_experiment,
    run_transfer_experiment,
)


def _make_probe_dirs(base: Path) -> tuple[Path, Path]:
    shutil.rmtree(base, ignore_errors=True)
    workspace = base / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir = base / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return workspace, state_dir


def test_intervention_sensitivity_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path("/Users/wojtek/barricade"))
    base = Path("/Users/wojtek/barricade") / "tests" / "tmp" / "extrapolation_probe_evidence" / tmp_path.name
    workspace, state_dir = _make_probe_dirs(base / "extrapolation_intervention")

    result = run_intervention_experiment(workspace, state_dir, trials=5)

    assert result["baseline_wins"] == 0
    assert result["scrambled_wins"] == 0
    assert result["ties"] == 5
    assert all(entry["baseline_solved"] for entry in result["results"])
    assert all(entry["scrambled_solved"] for entry in result["results"])


def test_structural_transfer_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path("/Users/wojtek/barricade"))
    base = Path("/Users/wojtek/barricade") / "tests" / "tmp" / "extrapolation_probe_evidence" / tmp_path.name
    workspace, state_dir = _make_probe_dirs(base / "extrapolation_transfer")

    result = run_transfer_experiment(workspace, state_dir)

    assert result["solved_count"] == 2
    assert result["total_count"] == 2


def test_generalization_gradient_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path("/Users/wojtek/barricade"))
    base = Path("/Users/wojtek/barricade") / "tests" / "tmp" / "extrapolation_probe_evidence" / tmp_path.name
    workspace, state_dir = _make_probe_dirs(base / "extrapolation_gradient")

    result = run_generalization_gradient(workspace, state_dir)

    assert result["gradient"] == {0.0: 1.0, 0.3: 1.0, 0.6: 1.0, 0.9: 1.0}


def test_macro_essence_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path("/Users/wojtek/barricade"))
    base = Path("/Users/wojtek/barricade") / "tests" / "tmp" / "extrapolation_probe_evidence" / tmp_path.name
    workspace, state_dir = _make_probe_dirs(base / "extrapolation_compression")

    result = run_macro_essence_experiment(workspace, state_dir)

    assert result["avg_length"] == 9.0
    assert result["min_length"] == 9
    assert result["max_length"] == 9