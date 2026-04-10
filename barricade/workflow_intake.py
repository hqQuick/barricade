from __future__ import annotations

import math
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runtime import build_optimizer_frame
from .problem_ir import build_problem_ir
from ._version import with_api_version
from ._task_shape_prior import (
    _classify_task_shape,
    _load_best_task_shape_prior,
    _materialize_task_shape_hit,
    _prior_strength_profile,
    _task_shape_cache_score,
    _task_shape_profile,
    _task_shape_similarity,
)


ARTIFACT_KEYWORDS = {
    "plan": (
        "plan",
        "design",
        "architect",
        "architecture",
        "strategy",
        "contract",
        "roadmap",
    ),
    "patch": (
        "patch",
        "implement",
        "implementation",
        "code",
        "change",
        "modify",
        "refactor",
        "fix",
        "handler",
        "script",
    ),
    "summary": (
        "summary",
        "summarize",
        "report",
        "review",
        "analyze",
        "analysis",
        "document",
        "explain",
        "compare",
    ),
    "test": (
        "test",
        "verify",
        "validation",
        "benchmark",
        "smoke",
        "suite",
        "e2e",
        "end to end",
    ),
}

PHASE_KEYWORDS = {
    "planning": (
        "plan",
        "design",
        "architect",
        "architecture",
        "schema",
        "contract",
        "spec",
        "proposal",
        "structure",
    ),
    "patching": (
        "patch",
        "implement",
        "code",
        "change",
        "modify",
        "add",
        "fix",
        "refactor",
        "wire",
        "handler",
    ),
    "summarizing": (
        "summary",
        "summarize",
        "report",
        "review",
        "explain",
        "document",
        "interpret",
        "compare",
    ),
    "recovery": (
        "error",
        "fail",
        "failure",
        "bug",
        "crash",
        "broken",
        "rollback",
        "recover",
        "retry",
        "quota",
        "parseable",
    ),
}

DOMAIN_TAGS = {
    "mcp": ("mcp", "tool", "tools", "server", "dispatch", "artifact"),
    "backend": (
        "backend",
        "api",
        "fastapi",
        "websocket",
        "websockets",
        "redis",
        "pub/sub",
        "pubsub",
        "scaling",
        "asgi",
    ),
    "benchmark": (
        "benchmark",
        "validation",
        "calibration",
        "matrix",
        "sweep",
        "phase",
        "v3.11",
        "v3.12",
    ),
    "security": (
        "security",
        "secret",
        "apikey",
        "api key",
        "quota",
        "env var",
        "environment variable",
    ),
    "analysis": (
        "analyze",
        "analysis",
        "compare",
        "measure",
        "efficiency",
        "capability",
        "understand",
    ),
    "math": (
        "math",
        "geometry",
        "geometric",
        "triangle",
        "centroid",
        "median",
        "proof",
        "prove",
        "derive",
        "equation",
        "theorem",
        "lemma",
        "combinatorics",
        "probability",
        "algebra",
        "solve",
    ),
    "reasoning": (
        "reason",
        "reasoning",
        "intuition",
        "derive",
        "proof",
        "prove",
        "show that",
        "invariant",
        "symbolic",
    ),
}

SHAPE_LANE_ORDER = ("planning", "patching", "summarizing", "recovery", "reasoning")

REASONING_KEYWORDS = (
    "prove",
    "proof",
    "derive",
    "derivation",
    "theorem",
    "lemma",
    "corollary",
    "geometry",
    "geometric",
    "triangle",
    "centroid",
    "median",
    "equation",
    "solve",
    "symbolic",
    "reasoning",
    "intuition",
    "combinatorics",
    "probability",
)


@dataclass
class WorkflowIntake:
    raw_task: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    expected_artifact_type: str = "plan"
    confidence: float = 0.0
    domain_tags: list[str] = field(default_factory=list)
    signals: dict[str, int] = field(default_factory=dict)
    route_hint: str = ""
    task_class: str = "general"
    phase_scores: dict[str, float] = field(default_factory=dict)
    problem_ir: dict[str, Any] = field(default_factory=dict)
    problem_ir_signature: str = ""
    semantic_probes: dict[str, Any] = field(default_factory=dict)
    counterexample_hints: list[str] = field(default_factory=list)
    prototype_stage: str = "candidate"
    holdout_bucket: int = 0
    route_explanation: list[str] = field(default_factory=list)
    shape_lanes: dict[str, float] = field(default_factory=dict)
    shape_centroid: dict[str, Any] = field(default_factory=dict)
    optimizer_frame: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-+/.]+", text.lower())


def sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def score_keywords(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def choose_goal(text: str) -> str:
    parts = sentences(text)
    if not parts:
        return text.strip()[:180]
    first = parts[0]
    if len(first) > 220 and len(parts) > 1:
        return parts[0][:180]
    return first[:220]


def extract_constraints(text: str) -> list[str]:
    lowered = text.lower()
    constraints: list[str] = []
    patterns = [
        r"\bmust\b[^.\n!?]*",
        r"\bshould\b[^.\n!?]*",
        r"\bwithout\b[^.\n!?]*",
        r"\bavoid\b[^.\n!?]*",
        r"\bkeep\b[^.\n!?]*",
        r"\bpreserve\b[^.\n!?]*",
        r"\bonly\b[^.\n!?]*",
        r"\bno\b[^.\n!?]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            phrase = re.sub(r"\s+", " ", match.group(0)).strip()
            constraints.append(phrase[:180])
    seen: list[str] = []
    for item in constraints:
        if item not in seen:
            seen.append(item)
    return seen[:6]


def extract_deliverables(text: str) -> list[str]:
    lowered = text.lower()
    deliverables: list[str] = []
    mapping = [
        ("plan", "plan artifact"),
        ("patch", "patch artifact"),
        ("summary", "summary artifact"),
        ("test", "test/verification artifact"),
        ("schema", "normalized schema"),
        ("contract", "contract/spec artifact"),
        ("code", "code artifact"),
        ("mcp", "mcp integration output"),
    ]
    for keyword, label in mapping:
        if keyword in lowered:
            deliverables.append(label)
    if not deliverables:
        deliverables.append("actionable plan artifact")
    seen: list[str] = []
    for item in deliverables:
        if item not in seen:
            seen.append(item)
    return seen[:6]


def extract_risks(text: str) -> list[str]:
    lowered = text.lower()
    risks: list[str] = []
    rules = [
        (
            "security risk: secrets or credentials exposure"
            if any(
                term in lowered
                for term in ("secret", "apikey", "api key", "credential")
            )
            else None
        ),
        (
            "operational risk: integration drift or unclear contract"
            if any(
                term in lowered
                for term in ("contract", "integration", "bridge", "handoff")
            )
            else None
        ),
        (
            "reliability risk: parse/verification failures"
            if any(
                term in lowered
                for term in ("verify", "parse", "parseable", "validate", "verification")
            )
            else None
        ),
        (
            "scaling risk: stateful coupling across runners"
            if any(
                term in lowered
                for term in ("scale", "scaling", "asgi", "redis", "websocket")
            )
            else None
        ),
        (
            "model quota risk: upstream provider exhaustion"
            if any(
                term in lowered for term in ("quota", "rate limit", "429", "provider")
            )
            else None
        ),
    ]
    for risk in rules:
        if risk and risk not in risks:
            risks.append(risk)
    if not risks:
        risks.append("low-confidence classification may require fallback heuristics")
    return risks[:6]


def extract_domain_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = [
        name
        for name, keywords in DOMAIN_TAGS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    if not tags:
        tags.append("general")
    return tags


def build_signals(text: str) -> dict[str, int]:
    lowered = text.lower()
    tokens = tokenize(text)
    return {
        "length_tokens": len(tokens),
        "sentence_count": len(sentences(text)),
        "plan_cues": score_keywords(lowered, ARTIFACT_KEYWORDS["plan"]),
        "patch_cues": score_keywords(lowered, ARTIFACT_KEYWORDS["patch"]),
        "summary_cues": score_keywords(lowered, ARTIFACT_KEYWORDS["summary"]),
        "test_cues": score_keywords(lowered, ARTIFACT_KEYWORDS["test"]),
        "recovery_cues": score_keywords(lowered, PHASE_KEYWORDS["recovery"]),
        "planning_cues": score_keywords(lowered, PHASE_KEYWORDS["planning"]),
        "patching_cues": score_keywords(lowered, PHASE_KEYWORDS["patching"]),
        "summarizing_cues": score_keywords(lowered, PHASE_KEYWORDS["summarizing"]),
        "reasoning_cues": score_keywords(lowered, REASONING_KEYWORDS),
        "structure_cues": score_keywords(
            lowered,
            (
                "json",
                "nested",
                "subtask",
                "reference",
                "scenario",
                "contract",
                "normalize",
                "deterministic",
            ),
        ),
    }


def _shape_lane_scores(
    frame: WorkflowIntake, optimizer_frame: dict[str, Any]
) -> dict[str, float]:
    signals = frame.signals
    ast_profile = (
        optimizer_frame.get("ast_profile", {})
        if isinstance(optimizer_frame, dict)
        else {}
    )
    domain_tags = {str(tag) for tag in frame.domain_tags if str(tag)}
    problem_ir = frame.problem_ir if isinstance(frame.problem_ir, dict) else {}
    relation_kinds = {
        str(relation.get("kind", ""))
        for relation in problem_ir.get("relations", [])
        if isinstance(relation, dict) and str(relation.get("kind", ""))
    }
    reasoning_domain = 1.0 if {"math", "reasoning"} & domain_tags else 0.0
    analysis_domain = 1.0 if "analysis" in domain_tags else 0.0
    math_style = (
        1.0 if any(tag in domain_tags for tag in ("math", "reasoning")) else 0.0
    )

    raw = {
        "planning": 1.0
        + 0.24 * signals["planning_cues"]
        + 0.16 * signals["plan_cues"]
        + 0.10 * signals["structure_cues"]
        + 0.02 * float(ast_profile.get("branch_count", 0) or 0.0)
        + 0.02 * float(ast_profile.get("max_depth", 0) or 0.0),
        "patching": 1.0
        + 0.24 * signals["patching_cues"]
        + 0.18 * signals["patch_cues"]
        + 0.05 * signals["test_cues"]
        + 0.02 * float(ast_profile.get("leaf_count", 0) or 0.0),
        "summarizing": 1.0
        + 0.28 * signals["summarizing_cues"]
        + 0.16 * signals["summary_cues"]
        + 0.03 * analysis_domain,
        "recovery": 1.0 + 0.30 * signals["recovery_cues"] + 0.12 * len(frame.risks),
        "reasoning": 1.0
        + 0.30 * signals["reasoning_cues"]
        + 0.18 * signals["structure_cues"]
        + 0.30 * reasoning_domain
        + 0.10 * math_style
        + 0.03 * float(ast_profile.get("branch_count", 0) or 0.0)
        + 0.02 * float(ast_profile.get("max_depth", 0) or 0.0),
    }

    if frame.expected_artifact_type == "plan":
        raw["planning"] += 0.12
    elif frame.expected_artifact_type == "patch":
        raw["patching"] += 0.12
    elif frame.expected_artifact_type == "summary":
        raw["summarizing"] += 0.12
    elif frame.expected_artifact_type == "test":
        raw["recovery"] += 0.08

    if reasoning_domain:
        raw["planning"] += 0.06 * math_style
        raw["summarizing"] -= 0.02 * math_style

    if problem_ir.get("kind") == "proof":
        raw["reasoning"] += 0.18
        raw["planning"] += 0.05
    if "proof" in relation_kinds:
        raw["reasoning"] += 0.14
    if "constraint" in relation_kinds:
        raw["planning"] += 0.05
    if "dependency" in relation_kinds:
        raw["planning"] += 0.03
        raw["recovery"] += 0.03
    if problem_ir.get("uncertainty", 0.0) >= 0.45:
        raw["recovery"] += 0.04

    return {lane: round(max(0.0, value), 3) for lane, value in raw.items()}


def _shape_centroid(
    frame: WorkflowIntake, optimizer_frame: dict[str, Any]
) -> dict[str, Any]:
    lane_scores = _shape_lane_scores(frame, optimizer_frame)
    total = sum(lane_scores.values()) or 1.0
    lane_weights = {
        lane: round(score / total, 3) for lane, score in lane_scores.items()
    }
    ranked = sorted(
        lane_weights.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    centroid_axis = sum(
        index * lane_weights.get(lane, 0.0)
        for index, lane in enumerate(SHAPE_LANE_ORDER)
    ) / max(1, len(SHAPE_LANE_ORDER) - 1)
    lane_entropy = -sum(
        weight * math.log(weight, 2) for weight in lane_weights.values() if weight > 0.0
    )
    ast_profile = (
        optimizer_frame.get("ast_profile", {})
        if isinstance(optimizer_frame, dict)
        else {}
    )
    return {
        "lane_weights": lane_weights,
        "dominant_lane": ranked[0][0] if ranked else "planning",
        "runner_up_lane": ranked[1][0]
        if len(ranked) > 1
        else (ranked[0][0] if ranked else "planning"),
        "lane_margin": round(
            (ranked[0][1] - ranked[1][1])
            if len(ranked) > 1
            else (ranked[0][1] if ranked else 0.0),
            3,
        ),
        "centroid_axis": round(centroid_axis, 3),
        "entropy": round(lane_entropy, 3),
        "lexical_projection": {
            "token_count": frame.signals["length_tokens"],
            "sentence_count": frame.signals["sentence_count"],
            "non_general_domain_tags": len(
                [tag for tag in frame.domain_tags if tag != "general"]
            ),
            "risk_count": len(frame.risks),
        },
        "ast_projection": {
            "node_count": int(ast_profile.get("node_count", 0) or 0),
            "leaf_count": int(ast_profile.get("leaf_count", 0) or 0),
            "branch_count": int(ast_profile.get("branch_count", 0) or 0),
            "max_depth": int(ast_profile.get("max_depth", 0) or 0),
        },
    }


def _route_explanation(frame: WorkflowIntake) -> list[str]:
    centroid = frame.shape_centroid if isinstance(frame.shape_centroid, dict) else {}
    problem_ir = frame.problem_ir if isinstance(frame.problem_ir, dict) else {}
    explanation: list[str] = []
    if problem_ir.get("kind"):
        explanation.append(f"problem_ir.kind={problem_ir.get('kind')}")
    explanation.append("route_source=canonical_problem_ir")
    if problem_ir.get("prototype_stage"):
        explanation.append(f"prototype_stage={problem_ir.get('prototype_stage')}")
    if problem_ir.get("holdout_bucket") is not None:
        explanation.append(
            f"holdout_bucket={int(problem_ir.get('holdout_bucket', 0) or 0)}"
        )
    if problem_ir.get("signature"):
        explanation.append(f"problem_ir.signature={problem_ir.get('signature')}")
    domain_tags = [
        str(tag) for tag in frame.domain_tags if str(tag) and str(tag) != "general"
    ]
    if domain_tags:
        explanation.append(f"domain_tags={', '.join(domain_tags[:4])}")
    dominant_lane = str(centroid.get("dominant_lane", frame.route_hint or "mixed"))
    runner_up_lane = str(centroid.get("runner_up_lane", "mixed"))
    lane_margin = float(centroid.get("lane_margin", 0.0) or 0.0)
    explanation.append(
        f"centroid={dominant_lane} (runner_up={runner_up_lane}, margin={lane_margin:.3f})"
    )
    if frame.phase_scores:
        ranked = sorted(
            (
                (phase, score)
                for phase, score in frame.phase_scores.items()
                if phase != "mixed"
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if ranked:
            explanation.append(f"phase_winner={ranked[0][0]}:{ranked[0][1]:.3f}")
    return explanation[:6]


def _select_route_hint(frame: WorkflowIntake, scores: dict[str, float]) -> str:
    problem_ir = frame.problem_ir if isinstance(frame.problem_ir, dict) else {}
    centroid = frame.shape_centroid if isinstance(frame.shape_centroid, dict) else {}
    lane_weights = (
        centroid.get("lane_weights", {}) if isinstance(centroid, dict) else {}
    )
    if not isinstance(lane_weights, dict):
        lane_weights = {}

    phase_ranked = sorted(
        ((phase, score) for phase, score in scores.items() if phase != "mixed"),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    phase_top = phase_ranked[0] if phase_ranked else ("mixed", 0.0)

    route_scores = {
        "planning": float(scores.get("planning", 0.0) or 0.0),
        "patching": float(scores.get("patching", 0.0) or 0.0),
        "summarizing": float(scores.get("summarizing", 0.0) or 0.0),
        "recovery": float(scores.get("recovery", 0.0) or 0.0),
    }

    kind = str(problem_ir.get("kind", "") or "")
    stage = str(
        problem_ir.get("prototype_stage", frame.prototype_stage)
        or frame.prototype_stage
        or "candidate"
    )
    uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)
    holdout_bucket = int(problem_ir.get("holdout_bucket", frame.holdout_bucket) or 0)
    dominant_lane = str(centroid.get("dominant_lane", "mixed") or "mixed")
    str(centroid.get("runner_up_lane", "mixed") or "mixed")
    lane_margin = float(centroid.get("lane_margin", 0.0) or 0.0)

    route_scores["planning"] += 0.18 * float(lane_weights.get("planning", 0.0) or 0.0)
    route_scores["patching"] += 0.18 * float(lane_weights.get("patching", 0.0) or 0.0)
    route_scores["summarizing"] += 0.18 * float(
        lane_weights.get("summarizing", 0.0) or 0.0
    )
    route_scores["recovery"] += 0.18 * float(lane_weights.get("recovery", 0.0) or 0.0)

    if kind == "proof" or dominant_lane == "reasoning":
        route_scores["planning"] += 0.24
    if kind == "patching" or frame.expected_artifact_type == "patch":
        route_scores["patching"] += 0.30
    if kind == "summarizing" or frame.expected_artifact_type == "summary":
        route_scores["summarizing"] += 0.30
    if kind == "recovery":
        route_scores["recovery"] += 0.30

    if stage in {"candidate", "emerging"}:
        route_scores["planning"] += 0.05
        route_scores["recovery"] += 0.03
    elif stage == "mature":
        route_scores["summarizing"] += 0.04

    if holdout_bucket == 0:
        route_scores["planning"] += 0.03
    if holdout_bucket == 3:
        route_scores["recovery"] += 0.03

    if uncertainty >= 0.58:
        route_scores["mixed"] = 0.30
        route_scores["recovery"] += 0.08
    else:
        route_scores["mixed"] = 0.06 + 0.10 * max(0.0, 0.08 - lane_margin)

    if dominant_lane == "planning":
        route_scores["planning"] += 0.12
    elif dominant_lane == "patching":
        route_scores["patching"] += 0.12
    elif dominant_lane == "summarizing":
        route_scores["summarizing"] += 0.12
    elif dominant_lane == "recovery":
        route_scores["recovery"] += 0.12

    ranked = sorted(
        route_scores.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else (None, 0.0)
    if top[0] == "mixed" or top[1] - second[1] < 0.03 or top[1] < 0.18:
        return phase_top[0]
    if top[0] != phase_top[0] and top[1] - phase_top[1] < 0.12:
        return phase_top[0]
    return top[0]


def infer_expected_artifact_type(
    text: str, deliverables: list[str], route_hint: str = ""
) -> str:
    lowered = text.lower()
    scores = {
        "plan": score_keywords(lowered, ARTIFACT_KEYWORDS["plan"]),
        "patch": score_keywords(lowered, ARTIFACT_KEYWORDS["patch"]),
        "summary": score_keywords(lowered, ARTIFACT_KEYWORDS["summary"]),
        "test": score_keywords(lowered, ARTIFACT_KEYWORDS["test"]),
    }
    if route_hint == "planning":
        scores["plan"] += 3
    elif route_hint == "patching":
        scores["patch"] += 3
    elif route_hint == "summarizing":
        scores["summary"] += 3
    elif route_hint == "recovery":
        scores["test"] += 3
    if "code artifact" in deliverables or "patch artifact" in deliverables:
        scores["patch"] += 2
    if (
        "actionable plan artifact" in deliverables
        or "normalized schema" in deliverables
    ):
        scores["plan"] += 2
    if "summary artifact" in deliverables:
        scores["summary"] += 2
    if route_hint != "summarizing" and any(
        keyword in lowered
        for keyword in (
            "nested",
            "json",
            "subtask",
            "scenario",
            "reference",
            "contract",
            "normalize",
            "deterministic",
        )
    ):
        scores["plan"] += 3
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _build_phase_scores(frame: WorkflowIntake) -> dict[str, float]:
    s = frame.signals
    plan = 0.10 + 0.22 * s["planning_cues"] + 0.16 * s["plan_cues"]
    patch = 0.10 + 0.22 * s["patching_cues"] + 0.18 * s["patch_cues"]
    summary = 0.10 + 0.24 * s["summarizing_cues"] + 0.18 * s["summary_cues"]
    reasoning = 0.10 + 0.28 * s["reasoning_cues"] + 0.12 * s["structure_cues"]
    recovery = (
        0.10
        + 0.25 * s["recovery_cues"]
        + 0.18
        * len(
            [
                risk
                for risk in frame.risks
                if "risk" in risk or "failure" in risk or "parse" in risk
            ]
        )
    )
    geometry = frame.optimizer_frame.get("geometry", {})
    combinators = frame.optimizer_frame.get("combinators", {})

    if s["structure_cues"]:
        plan += 0.20 + 0.06 * s["structure_cues"]
        summary += 0.04 * min(s["structure_cues"], 2)
        recovery += 0.03 * min(s["structure_cues"], 2)

    if frame.shape_lanes:
        plan += 0.12 * frame.shape_lanes.get("planning", 0.0)
        patch += 0.12 * frame.shape_lanes.get("patching", 0.0)
        summary += 0.12 * frame.shape_lanes.get("summarizing", 0.0)
        recovery += 0.12 * frame.shape_lanes.get("recovery", 0.0)
        reasoning += 0.16 * frame.shape_lanes.get("reasoning", 0.0)

    if "math" in frame.domain_tags or "reasoning" in frame.domain_tags:
        plan += 0.18 * frame.shape_lanes.get("reasoning", 0.0)
        summary -= 0.04 * frame.shape_lanes.get("reasoning", 0.0)

    if frame.expected_artifact_type == "plan":
        plan += 0.10
    elif frame.expected_artifact_type == "patch":
        patch += 0.10
    elif frame.expected_artifact_type == "summary":
        summary += 0.10
    elif frame.expected_artifact_type == "test":
        recovery += 0.04

    plan += 0.12 * geometry.get("plan_axis", 0.0) + 0.04 * combinators.get(
        "plan_x_structure", 0.0
    )
    patch += 0.12 * geometry.get("patch_axis", 0.0) + 0.04 * combinators.get(
        "patch_x_recovery", 0.0
    )
    summary += 0.12 * geometry.get("summary_axis", 0.0) + 0.04 * combinators.get(
        "summary_x_verify", 0.0
    )
    recovery += 0.12 * geometry.get("recovery_axis", 0.0)
    reasoning += 0.12 * frame.shape_lanes.get("reasoning", 0.0)

    scores = {
        "planning": plan,
        "patching": patch,
        "summarizing": summary,
        "recovery": recovery,
        "reasoning": reasoning,
    }
    total = sum(scores.values()) or 1.0
    normalized = {key: round(value / total, 3) for key, value in scores.items()}
    ranked = sorted(
        normalized.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    dominant = ranked[0][1]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    ambiguity = runner_up / dominant if dominant > 0 else 1.0
    mixed = round(
        min(
            0.36,
            max(
                0.03,
                0.03
                + 0.14 * ambiguity
                + 0.015 * s["structure_cues"]
                + 0.03 * max(0, len(frame.risks) - 1),
            ),
        ),
        3,
    )
    normalized["mixed"] = mixed
    phase_total = sum(normalized.values()) or 1.0
    return {key: round(value / phase_total, 3) for key, value in normalized.items()}


def _build_feed(
    problem_text: str,
    context: dict[str, Any] | None = None,
    feed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "problem_text": problem_text,
    }
    if context:
        payload["context"] = context
    if feed:
        payload["upstream_feed"] = feed
    return payload


def build_workflow_intake(
    problem_text: str,
    *,
    context: dict[str, Any] | None = None,
    feed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_feed = _build_feed(problem_text, context=context, feed=feed)
    optimizer_frame = build_optimizer_frame(
        tokenize(json.dumps(merged_feed, sort_keys=True))
    )
    goal = choose_goal(problem_text)
    constraints = extract_constraints(problem_text)
    deliverables = extract_deliverables(problem_text)
    risks = extract_risks(problem_text)
    domain_tags = extract_domain_tags(problem_text)
    signals = build_signals(problem_text)
    expected_artifact_type = infer_expected_artifact_type(
        problem_text, deliverables, optimizer_frame.get("route_hint", "")
    )
    problem_ir = build_problem_ir(
        problem_text,
        goal=goal,
        constraints=constraints,
        deliverables=deliverables,
        risks=risks,
        domain_tags=domain_tags,
        signals=signals,
        route_hint=optimizer_frame.get("route_hint", ""),
        expected_artifact_type=expected_artifact_type,
    )
    frame = WorkflowIntake(
        raw_task=problem_text.strip(),
        goal=goal,
        constraints=constraints,
        deliverables=deliverables,
        risks=risks,
        expected_artifact_type=expected_artifact_type,
        confidence=0.34,
        domain_tags=domain_tags,
        signals=signals,
        route_hint=optimizer_frame.get("route_hint", ""),
        problem_ir=problem_ir,
        problem_ir_signature=str(problem_ir.get("signature", "")),
        semantic_probes=dict(problem_ir.get("semantic_probes", {})),
        counterexample_hints=list(problem_ir.get("counterexample_hints", [])),
        prototype_stage=str(problem_ir.get("prototype_stage", "candidate")),
        holdout_bucket=int(problem_ir.get("holdout_bucket", 0) or 0),
        optimizer_frame=optimizer_frame,
    )
    frame.shape_lanes = _shape_lane_scores(frame, optimizer_frame)
    frame.shape_centroid = _shape_centroid(frame, optimizer_frame)
    frame.phase_scores = _build_phase_scores(frame)
    frame.route_hint = _select_route_hint(frame, frame.phase_scores)
    if frame.route_hint == "mixed" and frame.phase_scores.get("planning", 0.0) >= 0.30:
        frame.route_hint = "planning"
    if frame.route_hint == "mixed" and frame.phase_scores.get("patching", 0.0) >= 0.30:
        frame.route_hint = "patching"
    if (
        frame.route_hint == "mixed"
        and frame.phase_scores.get("summarizing", 0.0) >= 0.30
    ):
        frame.route_hint = "summarizing"
    if frame.route_hint == "mixed" and frame.phase_scores.get("recovery", 0.0) >= 0.30:
        frame.route_hint = "recovery"
    if frame.shape_centroid.get(
        "dominant_lane"
    ) == "reasoning" and frame.route_hint in {"mixed", "summarizing"}:
        frame.route_hint = "planning"
    phase_ranked = sorted(
        (
            (phase, score)
            for phase, score in frame.phase_scores.items()
            if phase != "mixed"
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if phase_ranked and phase_ranked[0][0] != "mixed":
        frame.route_hint = phase_ranked[0][0]
    if frame.shape_centroid.get("dominant_lane") == "reasoning":
        frame.route_hint = "planning"
    if isinstance(frame.problem_ir, dict):
        frame.problem_ir["route_hint"] = frame.route_hint
    frame.route_explanation = _route_explanation(frame)

    confidence = 0.34
    confidence += min(0.24, 0.03 * frame.signals["sentence_count"])
    confidence += min(0.24, 0.02 * frame.signals["length_tokens"])
    confidence += 0.04 * len([tag for tag in frame.domain_tags if tag != "general"])
    confidence += 0.03 * len(frame.constraints)
    confidence += 0.03 * len(frame.deliverables)
    if frame.route_hint != "mixed":
        confidence += 0.08
    frame.confidence = max(0.2, min(0.95, round(confidence, 3)))
    frame.task_class = _classify_task_shape(asdict(frame), problem_text, [])
    return with_api_version(asdict(frame))


def task_shape_profile(
    intake: dict[str, Any], task_pool: list[dict[str, Any]], feed_prior_dna: list[str]
) -> dict[str, Any]:
    profile = _task_shape_profile(intake, task_pool, feed_prior_dna)
    problem_ir = intake.get("problem_ir", {}) if isinstance(intake, dict) else {}
    profile["problem_ir_signature"] = str(intake.get("problem_ir_signature", ""))
    profile["problem_ir"] = problem_ir if isinstance(problem_ir, dict) else {}
    profile["semantic_probes"] = dict(intake.get("semantic_probes", {}))
    profile["counterexample_hints"] = list(intake.get("counterexample_hints", []))
    profile["prototype_stage"] = str(intake.get("prototype_stage", "candidate"))
    profile["holdout_bucket"] = int(intake.get("holdout_bucket", 0) or 0)
    profile["route_explanation"] = list(intake.get("route_explanation", []))
    return profile


def task_shape_similarity(
    current_profile: dict[str, Any], candidate_profile: dict[str, Any]
) -> float:
    return _task_shape_similarity(current_profile, candidate_profile)


def task_shape_cache_score(benchmark_result: dict[str, Any]) -> float:
    return _task_shape_cache_score(benchmark_result)


def classify_task_shape(
    intake: dict[str, Any], problem_text: str, task_pool: list[dict[str, Any]]
) -> str:
    return _classify_task_shape(intake, problem_text, task_pool)


def prior_strength_profile(
    intake: dict[str, Any],
    problem_text: str,
    task_pool: list[dict[str, Any]],
    requested_prior_strength: float | None = None,
) -> dict[str, Any]:
    return _prior_strength_profile(
        intake, problem_text, task_pool, requested_prior_strength
    )


def load_best_task_shape_prior(
    state_root: Path | None,
    current_profile: dict[str, Any],
    min_similarity: float = 0.60,
) -> dict[str, Any] | None:
    return _load_best_task_shape_prior(state_root, current_profile, min_similarity)


def materialize_task_shape_hit(
    cache_hit: dict[str, Any],
    task_pool: list[dict[str, Any]],
    feed_prior_dna: list[str],
    patch_skeleton: dict[str, Any],
    persisted_macros: dict[str, list[str]],
    prior_profile: dict[str, Any],
) -> dict[str, Any]:
    return _materialize_task_shape_hit(
        cache_hit,
        task_pool,
        feed_prior_dna,
        patch_skeleton,
        persisted_macros,
        prior_profile,
    )
