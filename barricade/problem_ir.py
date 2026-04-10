from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, cast
import re

from ._shared import dedupe_preserve_order as _dedupe_preserve_order


STOPWORDS: set[str] = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "before",
    "between",
    "but",
    "by",
    "can",
    "could",
    "define",
    "describe",
    "design",
    "do",
    "during",
    "each",
    "end",
    "for",
    "from",
    "give",
    "go",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "keep",
    "make",
    "may",
    "need",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "over",
    "prove",
    "show",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "use",
    "using",
    "via",
    "we",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
    "would",
    "you",
    "your",
    "workflow",
    "write",
    "implement",
    "patch",
    "change",
    "fix",
    "add",
    "plan",
    "test",
    "verify",
    "summarize",
    "summary",
    "analysis",
    "analyze",
    "compare",
    "explain",
    "derive",
    "demonstrate",
    "build",
    "create",
    "commit",
    "repository",
    "task",
    "problem",
    "behind",
    "demonstrate",
    "derive",
    "explain",
    "intuition",
    "meaning",
}

GOAL_KIND_HINTS: dict[str, tuple[str, ...]] = {
    "proof": (
        "prove",
        "proof",
        "theorem",
        "lemma",
        "derive",
        "derivation",
        "show that",
        "why",
    ),
    "planning": (
        "plan",
        "design",
        "architect",
        "architecture",
        "roadmap",
        "contract",
        "spec",
        "structure",
    ),
    "patching": (
        "patch",
        "implement",
        "code",
        "change",
        "modify",
        "fix",
        "refactor",
        "handler",
        "script",
    ),
    "summarizing": (
        "summary",
        "summarize",
        "review",
        "report",
        "compare",
        "analyze",
        "analysis",
        "explain",
    ),
    "recovery": (
        "recover",
        "recovery",
        "rollback",
        "retry",
        "failure",
        "crash",
        "bug",
        "incident",
    ),
}

RELATION_MARKERS: dict[str, tuple[str, ...]] = {
    "proof": (
        "prove",
        "proof",
        "show that",
        "derive",
        "demonstrate",
        "geometric",
        "intuition",
    ),
    "constraint": (
        "must",
        "should",
        "without",
        "avoid",
        "keep",
        "preserve",
        "only",
        "no",
    ),
    "dependency": (
        "depends on",
        "requires",
        "needs",
        "because",
        "therefore",
        "so that",
    ),
    "composition": (
        "with",
        "including",
        "via",
        "through",
        "using",
        "and",
    ),
    "comparison": (
        "compare",
        "versus",
        "vs",
        "contrast",
        "different",
    ),
    "recovery": (
        "rollback",
        "retry",
        "recover",
        "failure",
        "resilience",
    ),
}

ENTITY_HINTS: set[str] = {
    "analysis",
    "benchmark",
    "branch",
    "centroid",
    "commit",
    "constraint",
    "context",
    "contract",
    "coverage",
    "curve",
    "data",
    "domain",
    "edge",
    "entity",
    "equation",
    "error",
    "evidence",
    "function",
    "geometry",
    "goal",
    "graph",
    "heuristic",
    "incenter",
    "input",
    "landscape",
    "lemma",
    "lineage",
    "logic",
    "medians",
    "mcp",
    "method",
    "model",
    "node",
    "object",
    "operator",
    "path",
    "pipeline",
    "proof",
    "prototype",
    "problem",
    "relation",
    "request",
    "routing",
    "schema",
    "signature",
    "solution",
    "symmetry",
    "task",
    "theorem",
    "triangle",
    "uncertainty",
    "verification",
    "workflow",
}

ENTITY_NORMALIZATION: dict[str, str] = {
    "analyses": "analysis",
    "centroids": "centroid",
    "constraints": "constraint",
    "dependencies": "dependency",
    "deliverables": "deliverable",
    "entities": "entity",
    "equations": "equation",
    "geometrical": "geometry",
    "geometric": "geometry",
    "geometry": "geometry",
    "graphs": "graph",
    "meanings": "meaning",
    "medians": "median",
    "nodes": "node",
    "patches": "patch",
    "proofs": "proof",
    "prototypes": "prototype",
    "relations": "relation",
    "risks": "risk",
    "summaries": "summary",
    "triangles": "triangle",
    "workflows": "workflow",
}


def _empty_str_list() -> list[str]:
    return []


def _empty_relation_list() -> list[dict[str, Any]]:
    return []


def _empty_nodes_list() -> list[dict[str, Any]]:
    return []


def _empty_edges_list() -> list[dict[str, Any]]:
    return []


@dataclass
class ProblemIR:
    raw_text: str
    normalized_text: str
    goal: str
    kind: str
    domain_tags: list[str] = field(default_factory=_empty_str_list)
    entities: list[str] = field(default_factory=_empty_str_list)
    constraints: list[str] = field(default_factory=_empty_str_list)
    deliverables: list[str] = field(default_factory=_empty_str_list)
    risks: list[str] = field(default_factory=_empty_str_list)
    relations: list[dict[str, Any]] = field(default_factory=_empty_relation_list)
    canonical_tokens: list[str] = field(default_factory=_empty_str_list)
    signature: str = ""
    uncertainty: float = 0.0
    nodes: list[dict[str, Any]] = field(default_factory=_empty_nodes_list)
    edges: list[dict[str, Any]] = field(default_factory=_empty_edges_list)
    explanation: list[str] = field(default_factory=_empty_str_list)


def tokenize_problem(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-+/.]+", text.lower())


def normalize_problem_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9_\-+/. ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _clean_token(token: str) -> str:
    return token.strip("._-+/,;:!?()[]{}\"' ").lower()


def _keyword_tokens(value: str) -> list[str]:
    cleaned = [_clean_token(token) for token in tokenize_problem(value)]
    return [token for token in cleaned if token and token not in STOPWORDS]


def _normalize_entity_token(token: str) -> str:
    if not token:
        return token
    token = _clean_token(token)
    return ENTITY_NORMALIZATION.get(token, token)


def infer_goal_kind(
    goal: str,
    *,
    domain_tags: Iterable[str] = (),
    route_hint: str = "",
    expected_artifact_type: str = "",
    relation_kinds: Iterable[str] = (),
) -> str:
    lowered = goal.lower()
    relation_set = {relation for relation in relation_kinds if relation}
    domain_set = {tag.lower() for tag in domain_tags if tag}

    if "math" in domain_set or "reasoning" in domain_set or "proof" in relation_set:
        return "proof"
    if route_hint == "summarizing" or expected_artifact_type == "summary":
        return "summarizing"
    if route_hint == "recovery" or "recovery" in relation_set:
        return "recovery"
    if any(marker in lowered for marker in GOAL_KIND_HINTS["patching"]):
        return "patching"
    if any(marker in lowered for marker in GOAL_KIND_HINTS["planning"]):
        return "planning"
    if any(marker in lowered for marker in GOAL_KIND_HINTS["summarizing"]):
        return "summarizing"
    if any(marker in lowered for marker in GOAL_KIND_HINTS["recovery"]):
        return "recovery"
    return "general"


def extract_entities(
    text: str,
    *,
    domain_tags: Iterable[str] = (),
    goal_kind: str = "general",
) -> list[str]:
    domain_set = {tag.lower() for tag in domain_tags if tag}
    tokens = tokenize_problem(text)
    salient: list[str] = []
    for token in tokens:
        token = _normalize_entity_token(token)
        if not token or token in STOPWORDS:
            continue
        if len(token) < 4 and token not in {"api", "mcp", "ast", "ir"}:
            continue
        if token in ENTITY_HINTS or token in domain_set or token.isdigit():
            salient.append(token)
            continue
        if goal_kind == "proof" and token in {
            "triangle",
            "centroid",
            "median",
            "medians",
            "incenter",
            "symmetry",
            "equation",
        }:
            salient.append(token)
            continue
        if goal_kind == "patching" and token in {
            "workflow",
            "patch",
            "verify",
            "plan",
            "dispatch",
            "mcp",
            "repository",
        }:
            salient.append(token)
            continue
        if goal_kind == "planning" and token in {
            "plan",
            "design",
            "contract",
            "schema",
            "structure",
        }:
            salient.append(token)
            continue
        if len(token) > 5 and token not in {"showing", "explaining", "summarizing"}:
            salient.append(token)
    return _dedupe_preserve_order(salient)[:12]


def extract_relation_kinds(text: str) -> list[str]:
    lowered = text.lower()
    relation_kinds: list[str] = []
    for relation_kind, markers in RELATION_MARKERS.items():
        if any(marker in lowered for marker in markers):
            relation_kinds.append(relation_kind)
    if not relation_kinds:
        relation_kinds.append("association")
    return _dedupe_preserve_order(relation_kinds)


def _relation_kind_list(problem_ir: Mapping[str, Any]) -> list[str]:
    relations = problem_ir.get("relations", [])
    relation_kinds: list[str] = []
    if isinstance(relations, list):
        typed_relations = cast(list[dict[str, Any]], relations)
        for relation_dict in typed_relations:
            kind = _clean_token(str(relation_dict.get("kind", "")))
            if kind:
                relation_kinds.append(kind)
    return _dedupe_preserve_order(relation_kinds)


def _canonical_risk_tokens(risks: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    for risk in risks:
        lowered = risk.lower()
        if any(term in lowered for term in ("security", "credential", "secret")):
            canonical.append("risk:security")
        elif any(
            term in lowered for term in ("verify", "parse", "validation", "reliability")
        ):
            canonical.append("risk:reliability")
        elif any(term in lowered for term in ("scale", "scaling", "coupling")):
            canonical.append("risk:scaling")
        elif any(term in lowered for term in ("quota", "provider", "rate limit")):
            canonical.append("risk:quota")
        else:
            canonical.append("risk:general")
    return _dedupe_preserve_order(canonical)


def _canonical_deliverable_tokens(deliverables: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    for deliverable in deliverables:
        lowered = deliverable.lower()
        if "patch" in lowered or "code" in lowered:
            canonical.append("deliverable:patch")
        elif "summary" in lowered:
            canonical.append("deliverable:summary")
        elif "test" in lowered or "verification" in lowered:
            canonical.append("deliverable:verify")
        elif "schema" in lowered or "contract" in lowered:
            canonical.append("deliverable:contract")
        else:
            canonical.append("deliverable:plan")
    return _dedupe_preserve_order(canonical)


def _canonical_constraint_tokens(constraints: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    for constraint in constraints:
        lowered = constraint.lower()
        if any(term in lowered for term in ("must", "should")):
            canonical.append("constraint:requirement")
        if any(term in lowered for term in ("without", "avoid", "no")):
            canonical.append("constraint:avoidance")
        if any(term in lowered for term in ("keep", "preserve")):
            canonical.append("constraint:preserve")
        if any(term in lowered for term in ("only", "exact", "deterministic")):
            canonical.append("constraint:precision")
        if any(term in lowered for term in ("verify", "validation", "parse")):
            canonical.append("constraint:verification")
    return _dedupe_preserve_order(canonical)


def _canonical_goal_tokens(goal_kind: str, goal: str) -> list[str]:
    tokens = [f"goal:{goal_kind}"]
    goal_terms = _keyword_tokens(goal)
    if goal_kind == "proof":
        if any(
            token in goal_terms
            for token in (
                "geometry",
                "geometric",
                "triangle",
                "centroid",
                "median",
                "medians",
            )
        ):
            tokens.append("goal:math")
        if any(token in goal_terms for token in ("intuition", "derive", "derivation")):
            tokens.append("goal:reasoning")
    elif goal_kind == "patching":
        tokens.append("goal:implementation")
    elif goal_kind == "planning":
        tokens.append("goal:structure")
    elif goal_kind == "summarizing":
        tokens.append("goal:summary")
    elif goal_kind == "recovery":
        tokens.append("goal:recovery")
    return _dedupe_preserve_order(tokens)


def _problem_ir_uncertainty(
    *,
    goal_kind: str,
    entities: list[str],
    relation_kinds: list[str],
    constraints: list[str],
    risks: list[str],
    domain_tags: Iterable[str],
    signals: dict[str, int] | None,
) -> float:
    signal_count = signals.get("length_tokens", 0) if signals else 0
    sentence_count = signals.get("sentence_count", 0) if signals else 0
    uncertainty = 0.18
    uncertainty += 0.04 * max(0, 3 - len(entities))
    uncertainty += 0.03 * max(0, 2 - len(relation_kinds))
    uncertainty += 0.02 * max(0, 2 - len(constraints))
    uncertainty += 0.03 * len(risks)
    if goal_kind == "general":
        uncertainty += 0.05
    if signal_count < 12:
        uncertainty += 0.03
    if sentence_count > 2:
        uncertainty += 0.03
    domain_count = len([tag for tag in domain_tags if tag and tag != "general"])
    uncertainty -= 0.02 * min(domain_count, 3)
    return max(0.0, min(1.0, round(uncertainty, 3)))


def _build_nodes_and_edges(
    *,
    goal: str,
    entities: list[str],
    constraints: list[str],
    deliverables: list[str],
    risks: list[str],
    domain_tags: list[str],
    relation_kinds: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = [
        {"id": "goal", "kind": "goal", "label": goal, "weight": 1.0},
    ]
    edges: list[dict[str, Any]] = []

    for entity in entities:
        node_id = f"entity:{entity}"
        nodes.append({"id": node_id, "kind": "entity", "label": entity, "weight": 1.0})
        edges.append({"source": "goal", "target": node_id, "kind": "mentions"})

    for index, constraint in enumerate(constraints):
        node_id = f"constraint:{index}"
        nodes.append(
            {"id": node_id, "kind": "constraint", "label": constraint, "weight": 0.9}
        )
        edges.append({"source": "goal", "target": node_id, "kind": "bounded_by"})

    for index, deliverable in enumerate(deliverables):
        node_id = f"deliverable:{index}"
        nodes.append(
            {"id": node_id, "kind": "deliverable", "label": deliverable, "weight": 0.8}
        )
        edges.append({"source": "goal", "target": node_id, "kind": "targets"})

    for index, risk in enumerate(risks):
        node_id = f"risk:{index}"
        nodes.append({"id": node_id, "kind": "risk", "label": risk, "weight": 0.7})
        edges.append({"source": "goal", "target": node_id, "kind": "warns_about"})

    for tag in domain_tags:
        if not tag:
            continue
        node_id = f"domain:{tag}"
        nodes.append({"id": node_id, "kind": "domain", "label": tag, "weight": 0.7})
        edges.append({"source": "goal", "target": node_id, "kind": "scoped_to"})

    for relation_kind in relation_kinds:
        node_id = f"relation:{relation_kind}"
        nodes.append(
            {"id": node_id, "kind": "relation", "label": relation_kind, "weight": 0.8}
        )
        edges.append({"source": "goal", "target": node_id, "kind": "shaped_by"})

    return nodes, edges


def _problem_ir_signature(canonical_tokens: Iterable[str]) -> str:
    tokens = _dedupe_preserve_order(
        sorted(token for token in canonical_tokens if token)
    )
    return "|".join(tokens)


def build_problem_ir(
    raw_text: str,
    *,
    goal: str,
    constraints: list[str],
    deliverables: list[str],
    risks: list[str],
    domain_tags: list[str],
    signals: dict[str, int] | None = None,
    route_hint: str = "",
    expected_artifact_type: str = "",
) -> dict[str, Any]:
    normalized_text = normalize_problem_text(raw_text)
    relation_kinds = extract_relation_kinds(raw_text)
    goal_kind = infer_goal_kind(
        goal,
        domain_tags=domain_tags,
        route_hint=route_hint,
        expected_artifact_type=expected_artifact_type,
        relation_kinds=relation_kinds,
    )
    entities = extract_entities(raw_text, domain_tags=domain_tags, goal_kind=goal_kind)
    canonical_tokens = _dedupe_preserve_order(
        [
            *sorted(f"domain:{tag}" for tag in domain_tags if tag),
            *sorted(f"entity:{entity}" for entity in entities),
            *sorted(f"relation:{relation}" for relation in relation_kinds),
            *sorted(_canonical_constraint_tokens(constraints)),
            *sorted(_canonical_deliverable_tokens(deliverables)),
            *sorted(_canonical_risk_tokens(risks)),
            *sorted(_canonical_goal_tokens(goal_kind, goal)),
        ]
    )
    signature = _problem_ir_signature(canonical_tokens)
    nodes, edges = _build_nodes_and_edges(
        goal=goal,
        entities=entities,
        constraints=constraints,
        deliverables=deliverables,
        risks=risks,
        domain_tags=domain_tags,
        relation_kinds=relation_kinds,
    )
    uncertainty = _problem_ir_uncertainty(
        goal_kind=goal_kind,
        entities=entities,
        relation_kinds=relation_kinds,
        constraints=constraints,
        risks=risks,
        domain_tags=domain_tags,
        signals=signals,
    )
    explanation = [
        f"goal_kind={goal_kind}",
        f"entities={', '.join(entities[:5]) if entities else 'none'}",
        f"relations={', '.join(relation_kinds)}",
        f"domains={', '.join(tag for tag in domain_tags if tag)}",
        f"signature={signature}",
        f"uncertainty={uncertainty:.3f}",
    ]
    problem_ir = asdict(
        ProblemIR(
            raw_text=raw_text.strip(),
            normalized_text=normalized_text,
            goal=goal,
            kind=goal_kind,
            domain_tags=[tag for tag in domain_tags if tag],
            entities=entities,
            constraints=constraints,
            deliverables=deliverables,
            risks=risks,
            relations=[
                {"kind": relation_kind, "weight": 1.0}
                for relation_kind in relation_kinds
            ],
            canonical_tokens=canonical_tokens,
            signature=signature,
            uncertainty=uncertainty,
            nodes=nodes,
            edges=edges,
            explanation=explanation,
        )
    )
    semantic_probes = semantic_probe_profile(problem_ir)
    problem_ir["semantic_probes"] = semantic_probes
    problem_ir["prototype_stage"] = semantic_probes["stage"]
    problem_ir["counterexample_hints"] = semantic_probes["counterexample_hints"]
    problem_ir["holdout_bucket"] = semantic_probes["holdout_bucket"]
    return problem_ir


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(item) for item in left if str(item)}
    right_set = {str(item) for item in right if str(item)}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def problem_ir_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    signature_bonus = (
        0.2
        if left.get("signature") and left.get("signature") == right.get("signature")
        else 0.0
    )
    kind_bonus = (
        0.08 if left.get("kind") and left.get("kind") == right.get("kind") else 0.0
    )
    entity_score = _set_similarity(left.get("entities", []), right.get("entities", []))
    domain_score = _set_similarity(
        left.get("domain_tags", []), right.get("domain_tags", [])
    )
    constraint_score = _set_similarity(
        left.get("constraints", []), right.get("constraints", [])
    )
    deliverable_score = _set_similarity(
        left.get("deliverables", []), right.get("deliverables", [])
    )
    risk_score = _set_similarity(left.get("risks", []), right.get("risks", []))
    left_relations = _relation_kind_list(left)
    right_relations = _relation_kind_list(right)
    relation_score = _set_similarity(left_relations, right_relations)
    uncertainty_gap = abs(
        float(left.get("uncertainty", 0.0) or 0.0)
        - float(right.get("uncertainty", 0.0) or 0.0)
    )
    uncertainty_score = max(0.0, 1.0 - uncertainty_gap)

    score = (
        signature_bonus
        + kind_bonus
        + 0.30 * entity_score
        + 0.16 * domain_score
        + 0.12 * constraint_score
        + 0.10 * deliverable_score
        + 0.08 * risk_score
        + 0.14 * relation_score
        + 0.08 * uncertainty_score
    )
    return round(min(1.0, score), 3)


def _stable_bucket(value: str, buckets: int = 4) -> int:
    if buckets <= 0:
        return 0
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return digest[0] % buckets


def _problem_ir_support(problem_ir: Mapping[str, Any]) -> int:
    entities = problem_ir.get("entities", [])
    constraints = problem_ir.get("constraints", [])
    deliverables = problem_ir.get("deliverables", [])
    risks = problem_ir.get("risks", [])
    relation_kinds = _relation_kind_list(problem_ir)
    return (
        len(entities)
        + len(constraints)
        + len(deliverables)
        + len(risks)
        + len(relation_kinds)
    )


def problem_ir_prototype_stage(problem_ir: Mapping[str, Any]) -> str:
    kind = str(problem_ir.get("kind", "general") or "general")
    uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)
    support = _problem_ir_support(problem_ir)
    domain_tags = {str(tag).lower() for tag in problem_ir.get("domain_tags", []) if tag}

    if uncertainty >= 0.68 or support <= 2:
        return "candidate"
    if uncertainty >= 0.48 or support <= 4:
        return "emerging"
    if kind == "proof" and {"math", "reasoning"} & domain_tags and support >= 5:
        return "mature"
    if support >= 7 and uncertainty <= 0.28:
        return "mature"
    return "stable"


def semantic_probe_profile(problem_ir: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(problem_ir.get("kind", "general") or "general")
    domain_tags = [str(tag).lower() for tag in problem_ir.get("domain_tags", []) if tag]
    entities = [str(entity) for entity in problem_ir.get("entities", []) if entity]
    constraints = [
        str(constraint)
        for constraint in problem_ir.get("constraints", [])
        if constraint
    ]
    deliverables = [
        str(deliverable)
        for deliverable in problem_ir.get("deliverables", [])
        if deliverable
    ]
    risks = [str(risk) for risk in problem_ir.get("risks", []) if risk]
    relation_kinds = _relation_kind_list(problem_ir)
    signature = str(problem_ir.get("signature", ""))
    uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)

    probe_checks: list[str] = []
    if constraints:
        probe_checks.extend(
            f"check constraint: {constraint}" for constraint in constraints
        )
    if entities:
        probe_checks.append(f"check entity coverage: {', '.join(entities[:4])}")
    if deliverables:
        probe_checks.extend(
            f"verify deliverable: {deliverable}" for deliverable in deliverables
        )
    if risks:
        probe_checks.extend(f"guard against risk: {risk}" for risk in risks)

    if kind == "proof":
        probe_checks.extend(
            [
                "prove the claim under a degenerate case",
                "check boundary and symmetry cases",
                "confirm the relation remains invariant under paraphrase",
            ]
        )
    elif kind == "patching":
        probe_checks.extend(
            [
                "verify the patch preserves existing behavior",
                "check the failure path still rolls back cleanly",
                "confirm the change is idempotent on repeated apply",
            ]
        )
    elif kind == "summarizing":
        probe_checks.extend(
            [
                "check that no key facts are omitted",
                "compare against an adversarial paraphrase",
                "verify the summary preserves the original intent",
            ]
        )
    elif kind == "recovery":
        probe_checks.extend(
            [
                "simulate a partial failure and ensure recovery",
                "verify rollback completes without residual state",
                "confirm the retry path does not amplify the failure",
            ]
        )
    elif kind == "planning":
        probe_checks.extend(
            [
                "check the plan covers all required constraints",
                "verify the plan can absorb a scope change",
                "confirm dependencies are ordered correctly",
            ]
        )
    else:
        probe_checks.extend(
            [
                "check the canonical form is stable",
                "compare against a near-neighbor paraphrase",
                "verify the routing decision is explainable",
            ]
        )

    counterexample_hints: list[str] = []
    if kind == "proof" or {"math", "reasoning"} & set(domain_tags):
        counterexample_hints.extend(
            [
                "degenerate geometry that breaks the claim",
                "swap the named entities while preserving the wording",
                "a near-identical statement with one false relation",
            ]
        )
    elif kind == "patching":
        counterexample_hints.extend(
            [
                "a patch that compiles but violates the rollback path",
                "an implementation that mutates the API contract",
                "a near-miss that only works on the happy path",
            ]
        )
    elif kind == "summarizing":
        counterexample_hints.extend(
            [
                "a paraphrase that omits the key assertion",
                "a summary that swaps the cause and effect",
                "a response that sounds right but drops the central fact",
            ]
        )
    elif kind == "recovery":
        counterexample_hints.extend(
            [
                "a retry loop that retries the wrong step",
                "a rollback that leaves shared state dirty",
                "a partial outage that never resumes the main path",
            ]
        )
    else:
        counterexample_hints.extend(
            [
                "a paraphrase with one flipped constraint",
                "a task with the same words but a different deliverable",
                "a neighboring problem that shares the route but not the proof obligation",
            ]
        )

    if "composition" in relation_kinds:
        probe_checks.append("check component interactions remain compositional")
    if "dependency" in relation_kinds:
        probe_checks.append("check dependencies still resolve in order")
    if "comparison" in relation_kinds:
        probe_checks.append(
            "check the centroid remains distinct from the nearest neighbor"
        )

    support = _problem_ir_support(problem_ir)
    stage = problem_ir_prototype_stage(problem_ir)
    holdout_bucket = _stable_bucket(
        signature or problem_ir.get("normalized_text", "") or kind, 4
    )
    abstain = uncertainty >= 0.60 or support <= 2

    return {
        "kind": kind,
        "stage": stage,
        "support": support,
        "uncertainty": round(uncertainty, 3),
        "probe_checks": _dedupe_preserve_order(probe_checks)[:8],
        "counterexample_hints": _dedupe_preserve_order(counterexample_hints)[:8],
        "holdout_bucket": holdout_bucket,
        "abstain": abstain,
        "probe_count": len(_dedupe_preserve_order(probe_checks)),
    }
