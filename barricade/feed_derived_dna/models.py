from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .constants import MOMENTUM_WINDOW


__all__ = ["EvolutionConfig", "Individual", "ControllerState"]


@dataclass
class EvolutionConfig:
    enable_parallax: bool = True
    enable_orthogonality: bool = True
    enable_rotation: bool = True
    enable_unified: bool = True
    enable_curriculum: bool = True
    enable_primitive_contracts: bool = True
    parallax_probe_fraction: float = 0.25
    parallax_adversarial_fraction: float = 0.5
    scout_replace_fraction_min: float = 0.6
    scout_replace_fraction_max: float = 0.8
    pareto_dimensions: int = 9
    selection_axis_count: int = 9
    contract_min_support: int = 3
    contract_min_credit: float = 0.55


@dataclass
class Individual:
    dna: List[str]
    chart: str
    lineage_id: str
    parent_ids: List[str] = field(default_factory=list)
    fitness: float = 0.0
    governed_fitness: float = 0.0
    selection_score: float = 0.0
    stability_score: float = 0.0
    cognitive_stability: float = 0.0
    economic_stability: float = 0.0
    train_score: float = 0.0
    test_score: float = 0.0
    solve_rate_train: float = 0.0
    solve_rate_test: float = 0.0
    generalization_gap: float = 0.0
    generalization_gap_norm: float = 0.0
    flattened_len: float = 0.0
    reuse_score: float = 0.0
    gen_efficiency: float = 0.0
    artifact_reuse_score: float = 0.0
    artifact_yield: float = 0.0
    market_score: float = 0.0
    profit: float = 0.0
    wallet_end: float = 0.0
    task_threshold_pass: float = 0.0
    artifact_inventory: Dict[str, int] = field(default_factory=dict)
    library_concentration: float = 0.0
    dominant_macro_ratio: float = 0.0
    macro_hits: Dict[str, int] = field(default_factory=dict)
    specialization: str = "generalist"
    family_sig: str = "NONE"
    motif_sig: str = "NONE"
    fitness_vector: list[float] = field(default_factory=list)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    rotated_fitness: list[float] = field(default_factory=list)
    selection_axes: Dict[str, float] = field(default_factory=dict)
    parallax_role: str = "member"
    gradient_signal: Dict[str, float] = field(default_factory=dict)
    valley_membership: str = ""
    scout_origin: str = ""
    replication_gene: bool = False
    replication_depth: int = 0
    replication_origin: str = ""
    optimizer_frame: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerState:
    temperature: float = 0.62
    regime: str = "balanced"
    stuck_counter: int = 0
    last_governed: float = 0.0
    rotation_market_ema: float = 0.0
    rotation_market_initialized: bool = False
    replication_hold: int = 0
    balanced_success_counter: int = 0
    exploit_stay_counter: int = 0
    exploit_exit_counter: int = 0
    exploit_dwell_total: int = 0
    exploit_bounce_count: int = 0
    phase_hold: int = 0
    recent_governed: deque = field(
        default_factory=lambda: deque(maxlen=MOMENTUM_WINDOW)
    )
    recent_threshold: deque = field(
        default_factory=lambda: deque(maxlen=MOMENTUM_WINDOW)
    )
    recent_stability: deque = field(
        default_factory=lambda: deque(maxlen=MOMENTUM_WINDOW)
    )
    recent_regimes: deque = field(default_factory=lambda: deque(maxlen=12))
