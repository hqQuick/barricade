"""Barricade package."""

from ._version import API_VERSION, __version__

from .runtime import (
    benchmark_contract,
    build_optimizer_frame,
    build_patch_skeleton,
    derive_feed_dna_prior,
    derive_task_ecology,
    main,
    mine_macros_from_elites,
    run_benchmark,
)
from .scaling import analyze_scaling_profile
from .workflow import build_workflow_intake, run_unified_workflow

__all__ = [
    "API_VERSION",
    "benchmark_contract",
    "build_optimizer_frame",
    "build_patch_skeleton",
    "build_workflow_intake",
    "analyze_scaling_profile",
    "derive_feed_dna_prior",
    "derive_task_ecology",
    "main",
    "mine_macros_from_elites",
    "run_benchmark",
    "run_unified_workflow",
    "__version__",
]
