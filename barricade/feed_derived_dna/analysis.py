from __future__ import annotations

import hashlib
import math
import statistics
from functools import lru_cache
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .._shared import dedupe_preserve_order as _dedupe_preserve_order
from ..problem_ir import build_problem_ir, problem_ir_similarity, tokenize_problem
from .constants import (
    ARTIFACT_OPS,
    ARTIFACT_TYPES,
    BASE_MACROS,
    ALL_TOKENS,
    CONTROL_TOKENS,
    PRIMITIVES,
    SEMANTIC_FAMILIES,
)


__all__ = [
    "artifact_profile",
    "build_semantic_promotion_bank",
    "build_semantic_counterexample_bank",
    "build_curriculum_profile",
    "build_forbidden_subsequences",
    "build_primitive_contract_bank",
    "build_optimizer_frame",
    "count_non_overlapping_subseq",
    "compute_parallax_gradient",
    "estimate_artifact_quality",
    "family_signature",
    "flatten_trace",
    "governance_variance",
    "infer_specialization",
    "library_metrics",
    "lineage_entropy",
    "mine_macros_from_elites",
    "motif_entropy",
    "motif_key",
    "normalized_entropy",
    "optimizer_combinators",
    "optimizer_geometry",
    "optimizer_token_histogram",
    "population_landscape_profile",
    "semantic_prototype_lifecycle",
    "held_out_semantic_credit",
    "semantic_family_credit",
    "semantic_priority_tokens_from_task_pool",
    "semantic_task_profile",
    "semantic_trace_alignment",
    "specialization_entropy",
]


MACRO_EXPANSION_MAX_DEPTH = 8


def flatten_trace(
    dna: Sequence[str],
    macro_lib: Mapping[str, Sequence[str]],
    max_depth: int = MACRO_EXPANSION_MAX_DEPTH,
) -> list[str]:
    return list(
        _flatten_trace_cached(
            _trace_key(dna), _macro_lib_key(macro_lib), int(max_depth)
        )
    )


def _trace_key(trace: Sequence[str]) -> tuple[str, ...]:
    return tuple(trace)


def _macro_lib_key(
    macro_lib: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted((token, tuple(expansion)) for token, expansion in macro_lib.items())
    )


def _next_macro_index(macro_lib: Mapping[str, Sequence[str]] | None) -> int:
    highest = 0
    if not macro_lib:
        return 1
    for name in macro_lib:
        if isinstance(name, str) and name.startswith("LM") and name[2:].isdigit():
            highest = max(highest, int(name[2:]))
    return highest + 1


def _expand_outcome_trace_bank(
    outcome_memory: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], ...]:
    if not outcome_memory:
        return ()

    trace_bank = (
        outcome_memory.get("success_trace_bank")
        or outcome_memory.get("trace_bank")
        or []
    )
    if not isinstance(trace_bank, list):
        return ()

    expanded: list[tuple[str, ...]] = []
    for item in trace_bank:
        if not isinstance(item, Mapping):
            continue
        trace = item.get("trace", [])
        if not isinstance(trace, (list, tuple)):
            continue
        trace_tuple = tuple(str(token) for token in trace if str(token))
        if not trace_tuple:
            continue

        weight = float(item.get("weight", 1.0) or 1.0)
        copies = max(1, min(4, int(round(weight))))
        expanded.extend([trace_tuple] * copies)

    return tuple(expanded)


@lru_cache(maxsize=4096)
def _flatten_trace_cached(
    dna_key: tuple[str, ...],
    macro_lib_key: tuple[tuple[str, tuple[str, ...]], ...],
    max_depth: int,
) -> tuple[str, ...]:
    macro_lookup = dict(macro_lib_key)
    out: list[str] = []
    for tok in dna_key:
        out.extend(_expand_macro_token(tok, macro_lookup, max_depth, ()))
    return tuple(out)


def _expand_macro_token(
    token: str,
    macro_lookup: Mapping[str, tuple[str, ...]],
    remaining_depth: int,
    ancestry: tuple[str, ...],
) -> tuple[str, ...]:
    expansion = macro_lookup.get(token)
    if expansion is None or not expansion:
        return (token,)
    if remaining_depth <= 0 or token in ancestry:
        return (token,)

    next_ancestry = ancestry + (token,)
    expanded: list[str] = []
    for child in expansion:
        expanded.extend(
            _expand_macro_token(child, macro_lookup, remaining_depth - 1, next_ancestry)
        )
    return tuple(expanded)


@lru_cache(maxsize=1024)
def _macro_signature_sequences(
    macro_lib_key: tuple[tuple[str, tuple[str, ...]], ...],
    max_depth: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    macro_lookup = dict(macro_lib_key)
    signatures: list[tuple[str, tuple[str, ...]]] = []
    for name in sorted(macro_lookup):
        signature = _expand_macro_token(name, macro_lookup, max_depth, ())
        if signature:
            signatures.append((name, signature))
    signatures.sort(key=lambda item: (-len(item[1]), item[0]))
    return tuple(signatures)


@lru_cache(maxsize=4096)
def _compress_trace_for_mining_cached(
    trace_key: tuple[str, ...],
    macro_lib_key: tuple[tuple[str, tuple[str, ...]], ...],
    max_depth: int,
) -> tuple[str, ...]:
    signatures = _macro_signature_sequences(macro_lib_key, max_depth)
    if not signatures:
        return trace_key

    compressed: list[str] = []
    index = 0
    while index < len(trace_key):
        best_name: str | None = None
        best_length = 0
        for name, signature in signatures:
            length = len(signature)
            if length <= best_length or length == 0:
                continue
            if index + length > len(trace_key):
                continue
            if trace_key[index : index + length] == signature:
                best_name = name
                best_length = length
        if best_name is None:
            compressed.append(trace_key[index])
            index += 1
        else:
            compressed.append(best_name)
            index += best_length
    return tuple(compressed)


def count_non_overlapping_subseq(seq: Sequence[str], pat: Sequence[str]) -> int:
    return _count_non_overlapping_subseq_cached(_trace_key(seq), _trace_key(pat))


@lru_cache(maxsize=4096)
def _count_non_overlapping_subseq_cached(
    seq_key: tuple[str, ...], pat_key: tuple[str, ...]
) -> int:
    if not pat_key or len(pat_key) > len(seq_key):
        return 0
    i = 0
    c = 0
    m = len(pat_key)
    while i <= len(seq_key) - m:
        if seq_key[i : i + m] == pat_key:
            c += 1
            i += m
        else:
            i += 1
    return c


SEMANTIC_KIND_RULES: dict[str, dict[str, Any]] = {
    "proof": {
        "priority_tokens": (
            "PLAN",
            "VERIFY",
            "REPAIR",
            "OBSERVE",
            "RETRIEVE",
        ),
        "macro_templates": (
            ("PLAN", "REPAIR", "VERIFY"),
            ("OBSERVE", "RETRIEVE", "VERIFY"),
        ),
        "rule_label": "proof",
        "rule_weight": 1.25,
    },
    "planning": {
        "priority_tokens": (
            "WRITE_PLAN",
            "PLAN",
            "VERIFY",
            "READ_ARTIFACT",
        ),
        "macro_templates": (
            ("WRITE_PLAN", "READ_ARTIFACT", "PLAN"),
            ("PLAN", "REPAIR", "VERIFY"),
        ),
        "rule_label": "planning",
        "rule_weight": 1.10,
    },
    "patching": {
        "priority_tokens": (
            "PLAN",
            "WRITE_PATCH",
            "REPAIR",
            "VERIFY",
            "COMMIT",
        ),
        "macro_templates": (
            ("WRITE_PATCH", "READ_ARTIFACT", "REPAIR"),
            ("PLAN", "REPAIR", "VERIFY"),
        ),
        "rule_label": "patching",
        "rule_weight": 1.18,
    },
    "summarizing": {
        "priority_tokens": (
            "OBSERVE",
            "RETRIEVE",
            "VERIFY",
            "SUMMARIZE",
            "WRITE_SUMMARY",
        ),
        "macro_templates": (
            ("WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE"),
            ("VERIFY", "REPAIR", "SUMMARIZE"),
        ),
        "rule_label": "summarizing",
        "rule_weight": 1.08,
    },
    "recovery": {
        "priority_tokens": (
            "SWITCH_CONTEXT",
            "VERIFY",
            "ROLLBACK",
            "REPAIR",
            "SUMMARIZE",
        ),
        "macro_templates": (
            ("VERIFY", "ROLLBACK", "REPAIR"),
            ("SWITCH_CONTEXT", "VERIFY", "REPAIR"),
        ),
        "rule_label": "recovery",
        "rule_weight": 1.15,
    },
    "general": {
        "priority_tokens": (
            "OBSERVE",
            "PLAN",
            "VERIFY",
            "REPAIR",
        ),
        "macro_templates": (
            ("OBSERVE", "PLAN", "VERIFY"),
            ("PLAN", "REPAIR", "VERIFY"),
        ),
        "rule_label": "general",
        "rule_weight": 0.9,
    },
}


def _semantic_focus_from_task(
    task_name: str, focus: str, req: Sequence[str], source_text: str
) -> str:
    normalized_focus = str(focus or "").strip().lower()
    if normalized_focus in SEMANTIC_KIND_RULES:
        return normalized_focus

    req_set = {str(token).upper() for token in req if token}
    lowered_name = task_name.lower()
    lowered_text = source_text.lower()

    if "SUMMARIZE" in req_set or "summary" in lowered_name or "summary" in lowered_text:
        return "summarizing"
    if "ROLLBACK" in req_set or "recover" in lowered_name or "recover" in lowered_text:
        return "recovery"
    if "WRITE_PATCH" in req_set or "patch" in lowered_name or "patch" in lowered_text:
        return "patching"
    if "WRITE_PLAN" in req_set or "PLAN" in req_set or "plan" in lowered_name:
        return "planning"
    if any(
        marker in lowered_text
        for marker in (
            "triangle",
            "centroid",
            "median",
            "theorem",
            "lemma",
            "proof",
            "geometry",
        )
    ):
        return "proof"
    return "general"


def _semantic_domain_tags(kind: str, source_text: str) -> list[str]:
    lowered = source_text.lower()
    tags = [kind]
    if any(
        marker in lowered
        for marker in (
            "triangle",
            "centroid",
            "median",
            "theorem",
            "lemma",
            "proof",
            "geometry",
            "equation",
        )
    ):
        tags.extend(["math", "reasoning"])
    if any(
        marker in lowered
        for marker in (
            "patch",
            "code",
            "implement",
            "class",
            "function",
            "module",
            "runtime",
            "bug",
            "refactor",
        )
    ):
        tags.extend(["code", "implementation"])
    if any(
        marker in lowered
        for marker in ("summary", "summarize", "report", "review", "memo", "analysis")
    ):
        tags.extend(["docs", "summary"])
    if any(
        marker in lowered
        for marker in ("rollback", "recover", "failure", "incident", "timeout", "retry")
    ):
        tags.extend(["operations", "recovery"])
    return _dedupe_preserve_order(tags)


def _semantic_goal_text(kind: str, task_name: str) -> str:
    base_goals = {
        "proof": "prove the stated claim",
        "planning": "design a safe implementation plan",
        "patching": "implement the requested patch",
        "summarizing": "summarize the relevant material",
        "recovery": "recover from the failure and stabilize the workflow",
        "general": "solve the task",
    }
    base = base_goals.get(kind, base_goals["general"])
    if task_name:
        return f"{base} for {task_name}"
    return base


def _semantic_constraints(req: Sequence[str], hazard_if_no_verify: bool) -> list[str]:
    constraints = [f"must include {token.lower()}" for token in req if token]
    if hazard_if_no_verify:
        constraints.append("must verify before concluding")
    return _dedupe_preserve_order(constraints)


def _semantic_deliverables(kind: str, req: Sequence[str]) -> list[str]:
    req_set = {str(token).upper() for token in req if token}
    if kind == "patching" or "WRITE_PATCH" in req_set or "COMMIT" in req_set:
        return ["patch"]
    if kind == "summarizing" or "WRITE_SUMMARY" in req_set or "SUMMARIZE" in req_set:
        return ["summary"]
    if kind == "recovery" or "ROLLBACK" in req_set:
        return ["verify", "patch"]
    return ["plan"]


def _semantic_risks(kind: str, hazard_if_no_verify: bool) -> list[str]:
    risks = []
    if hazard_if_no_verify or kind == "recovery":
        risks.append("verification risk")
    if kind == "patching":
        risks.append("integration drift")
    elif kind == "summarizing":
        risks.append("loss of detail")
    elif kind == "proof":
        risks.append("reasoning ambiguity")
    elif kind == "planning":
        risks.append("scope drift")
    if not risks:
        risks.append("general risk")
    return _dedupe_preserve_order(risks)


def _semantic_priority_tokens(
    problem_ir: Mapping[str, Any], req: Sequence[str]
) -> list[str]:
    kind = str(problem_ir.get("kind", "general") or "general")
    profile = SEMANTIC_KIND_RULES.get(kind, SEMANTIC_KIND_RULES["general"])
    tokens = list(profile["priority_tokens"])

    domain_tags = {str(tag).lower() for tag in problem_ir.get("domain_tags", []) if tag}
    if kind == "proof" and {"math", "reasoning"} & domain_tags:
        tokens.extend(["PLAN", "VERIFY", "REPAIR", "OBSERVE", "RETRIEVE"])
    if kind == "patching" or "code" in domain_tags:
        tokens.extend(["WRITE_PATCH", "PLAN", "REPAIR", "VERIFY", "COMMIT"])
    if kind == "summarizing" or "docs" in domain_tags:
        tokens.extend(["OBSERVE", "RETRIEVE", "VERIFY", "SUMMARIZE", "WRITE_SUMMARY"])
    if kind == "recovery" or "recovery" in domain_tags:
        tokens.extend(["SWITCH_CONTEXT", "ROLLBACK", "REPAIR", "VERIFY"])
    if kind == "planning" or "implementation" in domain_tags:
        tokens.extend(["WRITE_PLAN", "PLAN", "VERIFY", "READ_ARTIFACT"])

    tokens.extend(
        str(token).upper()
        for token in req
        if token in ALL_TOKENS or token in CONTROL_TOKENS
    )
    return _dedupe_preserve_order(tokens)


def _semantic_macro_templates(
    problem_ir: Mapping[str, Any], req: Sequence[str]
) -> list[tuple[str, ...]]:
    kind = str(problem_ir.get("kind", "general") or "general")
    profile = SEMANTIC_KIND_RULES.get(kind, SEMANTIC_KIND_RULES["general"])
    templates = [tuple(template) for template in profile["macro_templates"]]
    req_tokens = [str(token).upper() for token in req if token]
    if len(req_tokens) >= 3:
        templates.append(tuple(req_tokens[:3]))
    if len(req_tokens) >= 4:
        templates.append(tuple(req_tokens[:4]))
    return [
        tuple(template.split("|"))
        for template in _dedupe_preserve_order(
            ["|".join(template) for template in templates]
        )
    ]


@lru_cache(maxsize=512)
def _semantic_task_profile_cached(
    task_name: str,
    focus: str,
    source_text: str,
    req_key: tuple[str, ...],
    needs_key: tuple[tuple[str, int], ...],
    hazard_if_no_verify: bool,
) -> dict[str, Any]:
    kind = _semantic_focus_from_task(task_name, focus, req_key, source_text)
    domain_tags = _semantic_domain_tags(kind, source_text)
    problem_ir = build_problem_ir(
        raw_text=source_text or task_name or kind,
        goal=_semantic_goal_text(kind, task_name),
        constraints=_semantic_constraints(req_key, hazard_if_no_verify),
        deliverables=_semantic_deliverables(kind, req_key),
        risks=_semantic_risks(kind, hazard_if_no_verify),
        domain_tags=domain_tags,
        signals={
            "length_tokens": len(tokenize_problem(source_text or task_name or kind)),
            "sentence_count": max(
                1,
                (source_text or task_name or kind).count(".")
                + (source_text or task_name or kind).count(";"),
            ),
        },
        route_hint=kind,
        expected_artifact_type=(
            "summary"
            if kind == "summarizing"
            else "patch"
            if kind == "patching"
            else "verify"
            if kind == "recovery"
            else "plan"
        ),
    )
    priority_tokens = _semantic_priority_tokens(problem_ir, req_key)
    macro_templates = _semantic_macro_templates(problem_ir, req_key)
    rule_weight = max(
        0.6,
        round(
            1.18
            - 0.24 * float(problem_ir.get("uncertainty", 0.0) or 0.0)
            + 0.04 * len(domain_tags),
            3,
        ),
    )
    return {
        "task_name": task_name,
        "focus": kind,
        "problem_ir": problem_ir,
        "priority_tokens": priority_tokens,
        "macro_templates": macro_templates,
        "rule_examples": list(problem_ir.get("explanation", [])),
        "rule_weight": rule_weight,
        "source_text": source_text[:240],
        "req_tokens": list(req_key),
        "needs": dict(needs_key),
        "hazard_if_no_verify": hazard_if_no_verify,
    }


def semantic_task_profile(task: Mapping[str, Any]) -> dict[str, Any]:
    req = tuple(str(token).upper() for token in task.get("req", []) if token)
    needs = tuple(
        sorted(
            (str(key), int(value)) for key, value in dict(task.get("needs", {})).items()
        )
    )
    task_name = str(task.get("name", "task"))
    focus = str(task.get("focus", ""))
    source_text = str(task.get("source_text", task_name))
    hazard_if_no_verify = bool(task.get("hazard_if_no_verify", False))
    return _semantic_task_profile_cached(
        task_name,
        focus,
        source_text,
        req,
        needs,
        hazard_if_no_verify,
    )


def semantic_family_credit(flat: Sequence[str]) -> tuple[float, dict[str, int]]:
    score, counts = _semantic_family_credit_cached(_trace_key(flat))
    return score, counts


@lru_cache(maxsize=4096)
def _semantic_family_credit_cached(
    flat_key: tuple[str, ...],
) -> tuple[float, dict[str, int]]:
    weights: dict[str, float] = {
        "DISCOVERY": 5.0,
        "SAFE_REPORT": 6.0,
        "RECOVERY": 5.5,
        "MEMORY": 5.0,
        "PATCHING": 4.5,
        "DIAGNOSIS": 4.5,
        "ARTIFACT_PLAN": 4.0,
        "ARTIFACT_PATCH": 4.0,
        "ARTIFACT_SUMMARY": 4.0,
    }
    score = 0.0
    counts: dict[str, int] = {}
    for fam, pats in SEMANTIC_FAMILIES.items():
        best = 0
        for pat in pats:
            best = max(best, _count_non_overlapping_subseq_cached(flat_key, tuple(pat)))
        counts[fam] = best
        score += best * weights[fam]
    return score, counts


def artifact_profile(flat: Sequence[str]) -> dict[str, int]:
    return _artifact_profile_cached(_trace_key(flat))


@lru_cache(maxsize=4096)
def _artifact_profile_cached(flat_key: tuple[str, ...]) -> dict[str, int]:
    writes: Counter[str] = Counter()
    reads = 0
    revisions = 0
    links = 0
    for tok in flat_key:
        if tok in ARTIFACT_TYPES:
            writes[ARTIFACT_TYPES[tok]] += 1
        if tok == "READ_ARTIFACT":
            reads += 1
        elif tok == "REVISE_ARTIFACT":
            revisions += 1
        elif tok == "LINK_ARTIFACT":
            links += 1
    return {
        **dict(writes),
        "reads": reads,
        "revisions": revisions,
        "links": links,
    }


def estimate_artifact_quality(flat: Sequence[str]) -> dict[str, float]:
    return _estimate_artifact_quality_cached(_trace_key(flat))


@lru_cache(maxsize=4096)
def _estimate_artifact_quality_cached(flat_key: tuple[str, ...]) -> dict[str, float]:
    inv = _artifact_profile_cached(flat_key)
    return {
        "plan": _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_PLAN", "READ_ARTIFACT", "PLAN")
        )
        * 1.8
        + _count_non_overlapping_subseq_cached(flat_key, ("PLAN", "REPAIR", "VERIFY"))
        * 1.2
        + inv.get("revisions", 0) * 0.15,
        "patch": _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_PATCH", "READ_ARTIFACT", "REPAIR")
        )
        * 1.8
        + _count_non_overlapping_subseq_cached(
            flat_key, ("VERIFY", "ROLLBACK", "REPAIR")
        )
        * 1.0
        + inv.get("revisions", 0) * 0.20,
        "summary": _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE")
        )
        * 1.8
        + _count_non_overlapping_subseq_cached(
            flat_key, ("VERIFY", "REPAIR", "SUMMARIZE")
        )
        * 1.2
        + inv.get("links", 0) * 0.10,
    }


def infer_specialization(flat: Sequence[str]) -> str:
    q = estimate_artifact_quality(flat)
    ranked = sorted(q.items(), key=lambda kv: kv[1], reverse=True)
    art, val = ranked[0]
    if val < 1.5:
        return "generalist"

    runner_up, runner_up_val = ranked[1] if len(ranked) > 1 else ("", 0.0)
    if art == "summary" and runner_up in {"plan", "patch"} and runner_up_val >= 0.9:
        if val - runner_up_val <= 0.75:
            return "summary_bridge"
    if art in {"plan", "patch"} and q["summary"] >= 0.9:
        if val - q["summary"] <= 0.75:
            return "summary_bridge"
    return art


def motif_key(dna: Sequence[str], topn: int = 3) -> str:
    c = Counter(dna)
    return "|".join(k for k, _ in c.most_common(topn))


def family_signature(flat: Sequence[str]) -> str:
    _, fam = semantic_family_credit(flat)
    active = [k for k, v in fam.items() if v > 0]
    if not active:
        return "NONE"
    active.sort()
    return "|".join(active[:4])


def library_metrics(
    dna: Sequence[str], learned_macros: Mapping[str, Sequence[str]]
) -> tuple[float, float]:
    calls = [tok for tok in dna if tok in learned_macros]
    if not calls:
        return 0.0, 0.0
    cnt = Counter(calls)
    total = sum(cnt.values())
    probs = [v / total for v in cnt.values()]
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    max_h = math.log(len(learned_macros) + 1e-12)
    concentration = 1.0 - (h / max_h if max_h > 0 else 0.0)
    dominant = max(cnt.values()) / total
    return concentration, dominant


def normalized_entropy(items: Sequence[Any]) -> float:
    if not items:
        return 0.0
    c = Counter(items)
    n = sum(c.values())
    if len(c) == 1:
        return 0.0
    h = -sum((v / n) * math.log((v / n) + 1e-12) for v in c.values())
    return h / math.log(len(c) + 1e-12)


def motif_entropy(pop: Sequence[Any]) -> float:
    return normalized_entropy([x.motif_sig for x in pop if x.motif_sig])


def specialization_entropy(pop: Sequence[Any]) -> float:
    return normalized_entropy([x.specialization for x in pop if x.specialization])


def lineage_entropy(pop: Sequence[Any]) -> float:
    return normalized_entropy([x.lineage_id.split(":")[0] for x in pop])


def governance_variance(pop: Sequence[Any]) -> float:
    vals = [x.governed_fitness for x in pop]
    return statistics.pvariance(vals) if len(vals) > 1 else 0.0


def mine_macros_from_elites(
    elite_flattened: Sequence[Sequence[str]],
    max_macros: int = 8,
    lengths: Sequence[int] = (3, 4),
    semantic_context: Sequence[Mapping[str, Any]] | None = None,
    macro_lib: Mapping[str, Sequence[str]] | None = None,
    outcome_memory: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    normalized_elites = tuple(tuple(trace) for trace in elite_flattened)
    if outcome_memory:
        normalized_elites = normalized_elites + _expand_outcome_trace_bank(
            outcome_memory
        )
    if macro_lib:
        normalized_elites = tuple(
            _compress_trace_for_mining_cached(
                trace,
                _macro_lib_key(macro_lib),
                MACRO_EXPANSION_MAX_DEPTH,
            )
            for trace in normalized_elites
        )
    next_macro_index = _next_macro_index(macro_lib)

    if semantic_context:
        mined = _mine_semantic_macros_from_elites(
            normalized_elites,
            semantic_context,
            max_macros=max_macros,
            lengths=lengths,
        )
    else:
        cached_macros = _mine_macros_from_elites_cached(
            normalized_elites, max_macros, tuple(lengths)
        )
        mined = {name: list(seq) for name, seq in cached_macros}

    if next_macro_index > 1:
        return {
            f"LM{next_macro_index + index}": seq
            for index, seq in enumerate(mined.values())
        }
    return mined


@lru_cache(maxsize=1024)
def _mine_macros_from_elites_cached(
    elite_flattened: tuple[tuple[str, ...], ...],
    max_macros: int,
    lengths: tuple[int, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    counter: Counter[tuple[str, ...]] = Counter()
    blacklist: set[tuple[str, ...]] = {tuple(v) for v in BASE_MACROS.values()}
    learned: list[tuple[str, tuple[str, ...]]] = []
    for trace in elite_flattened:
        seen: set[tuple[str, ...]] = set()
        for length in lengths:
            if length <= 0 or length > len(trace):
                continue
            for i in range(len(trace) - length + 1):
                seq = trace[i : i + length]
                if len(set(seq)) >= 2:
                    seen.add(seq)
        for seq in seen:
            counter[seq] += 1
    idx = 1
    for seq, c in counter.most_common():
        if c < 2 or seq in blacklist:
            continue
        learned.append((f"LM{idx}", seq))
        idx += 1
        if len(learned) >= max_macros:
            break
    return tuple(learned)


def _mine_semantic_macros_from_elites(
    elite_flattened: Sequence[Sequence[str]],
    semantic_context: Sequence[Mapping[str, Any]],
    *,
    max_macros: int,
    lengths: Sequence[int],
) -> dict[str, list[str]]:
    semantic_profiles = [semantic_task_profile(task) for task in semantic_context]
    if not semantic_profiles:
        return {
            name: list(seq)
            for name, seq in _mine_macros_from_elites_cached(
                tuple(tuple(trace) for trace in elite_flattened),
                max_macros,
                tuple(lengths),
            )
        }

    token_bias: defaultdict[str, float] = defaultdict(float)
    template_bias: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for profile in semantic_profiles:
        weight = float(profile.get("rule_weight", 1.0) or 1.0)
        for token in profile.get("priority_tokens", []):
            token_bias[str(token)] += weight
        for template in profile.get("macro_templates", []):
            template_bias[tuple(template)] += weight

    counter: defaultdict[tuple[str, ...], float] = defaultdict(float)
    blacklist: set[tuple[str, ...]] = {tuple(v) for v in BASE_MACROS.values()}
    for trace in elite_flattened:
        seen: set[tuple[str, ...]] = set()
        for length in lengths:
            if length <= 0 or length > len(trace):
                continue
            for i in range(len(trace) - length + 1):
                seq = tuple(trace[i : i + length])
                if len(set(seq)) >= 2:
                    seen.add(seq)
        for seq in seen:
            if seq in blacklist:
                continue
            token_overlap = sum(token_bias.get(tok, 0.0) for tok in seq) / max(
                1, len(seq)
            )
            template_support = template_bias.get(seq, 0.0)
            bonus = 0.22 * token_overlap + 0.25 * template_support
            counter[seq] += 1.0 + bonus

    learned: list[tuple[str, tuple[str, ...]]] = []
    idx = 1
    for seq, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        if seq in blacklist:
            continue
        if count < 1.35:
            continue
        learned.append((f"LM{idx}", seq))
        idx += 1
        if len(learned) >= max_macros:
            break
    return {name: list(seq) for name, seq in learned}


def semantic_priority_tokens_from_task_pool(
    task_pool: Sequence[Mapping[str, Any]], limit: int = 5
) -> list[str]:
    token_weights: defaultdict[str, float] = defaultdict(float)
    for task in task_pool:
        profile = semantic_task_profile(task)
        for token in profile["priority_tokens"]:
            token_weights[str(token)] += float(profile["rule_weight"])
    return [
        token
        for token, _ in sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]
    ]


def semantic_trace_alignment(
    flat: Sequence[str], semantic_profile: Mapping[str, Any]
) -> float:
    if not semantic_profile:
        return 0.0

    priority_tokens = [
        str(token) for token in semantic_profile.get("priority_tokens", []) if token
    ]
    templates = [
        tuple(template)
        for template in semantic_profile.get("macro_templates", [])
        if template
    ]
    if not priority_tokens and not templates:
        return 0.0

    flat_tokens = [str(token) for token in flat if token]
    flat_set = set(flat_tokens)
    priority_set = set(priority_tokens)

    token_coverage = len(flat_set & priority_set) / max(1, len(priority_set))
    token_density = sum(1 for token in flat_tokens if token in priority_set) / max(
        1, len(flat_tokens)
    )
    template_score = 0.0
    if templates:
        template_score = min(
            1.0,
            max(
                _count_non_overlapping_subseq_cached(_trace_key(flat_tokens), template)
                for template in templates
            ),
        )

    problem_ir = semantic_profile.get("problem_ir", {})
    uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)
    rule_weight = float(semantic_profile.get("rule_weight", 1.0) or 1.0)

    score = (
        0.50 * token_coverage + 0.30 * token_density + 0.20 * template_score
    ) * rule_weight
    score *= max(0.45, 1.0 - 0.45 * uncertainty)
    return round(min(1.0, score), 3)


def build_semantic_promotion_bank(
    task_pool: Sequence[Mapping[str, Any]],
    elite_flattened: Sequence[Sequence[str]],
    learned_macros: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    profiles_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in task_pool:
        profile = semantic_task_profile(task)
        profiles_by_kind[str(profile.get("focus", "general") or "general")].append(
            profile
        )

    trace_pool = [list(trace) for trace in elite_flattened if trace]
    semantic_rules: list[dict[str, Any]] = []

    for kind, profiles in sorted(
        profiles_by_kind.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if not profiles:
            continue

        aggregate_priority: defaultdict[str, float] = defaultdict(float)
        aggregate_templates: defaultdict[tuple[str, ...], float] = defaultdict(float)
        task_examples: list[str] = []
        for profile in profiles:
            weight = float(profile.get("rule_weight", 1.0) or 1.0)
            for token in profile.get("priority_tokens", []):
                aggregate_priority[str(token)] += weight
            for template in profile.get("macro_templates", []):
                aggregate_templates[tuple(template)] += weight
            task_examples.extend(profile.get("rule_examples", []))

        representative = profiles[0]
        trace_examples: list[dict[str, Any]] = []
        if trace_pool:
            trace_scores = []
            for trace in trace_pool:
                trace_scores.append(
                    (
                        semantic_trace_alignment(trace, representative),
                        list(trace[:12]),
                    )
                )
            trace_scores.sort(key=lambda item: item[0], reverse=True)
            for score, trace in trace_scores[:2]:
                trace_examples.append({"trace": trace, "alignment": round(score, 3)})

        promoted_macros: list[str] = []
        if learned_macros:
            priority_set = set(aggregate_priority)
            template_set = {tuple(template) for template in aggregate_templates}
            for name, expansion in learned_macros.items():
                expansion_tuple = tuple(expansion)
                if len(expansion_tuple) < 3 or len(expansion_tuple) > 6:
                    continue
                if all(token.startswith("LM") for token in expansion_tuple):
                    continue
                if expansion_tuple in template_set:
                    promoted_macros.append(name)
                    continue
                if len(priority_set & set(expansion_tuple)) >= max(
                    4, len(expansion_tuple)
                ):
                    promoted_macros.append(name)

        semantic_rules.append(
            {
                "kind": kind,
                "task_count": len(profiles),
                "signature": str(
                    representative.get("problem_ir", {}).get("signature", "")
                ),
                "domain_tags": list(
                    representative.get("problem_ir", {}).get("domain_tags", [])
                ),
                "priority_tokens": [
                    token
                    for token, _ in sorted(
                        aggregate_priority.items(), key=lambda x: x[1], reverse=True
                    )[:6]
                ],
                "macro_templates": [
                    list(template)
                    for template, _ in sorted(
                        aggregate_templates.items(), key=lambda x: x[1], reverse=True
                    )[:4]
                ],
                "task_examples": _dedupe_preserve_order(task_examples)[:3],
                "trace_examples": trace_examples,
                "promoted_macros": _dedupe_preserve_order(promoted_macros)[:6],
                "macro_budget": {
                    "max_promotions": 6,
                    "min_alignment": 0.68,
                    "min_length": 3,
                    "max_length": 6,
                },
                "rule_weight": round(
                    sum(
                        float(profile.get("rule_weight", 1.0) or 1.0)
                        for profile in profiles
                    )
                    / len(profiles),
                    3,
                ),
                "uncertainty": round(
                    sum(
                        float(
                            profile.get("problem_ir", {}).get("uncertainty", 0.0) or 0.0
                        )
                        for profile in profiles
                    )
                    / len(profiles),
                    3,
                ),
            }
        )

    semantic_rules.sort(key=lambda entry: (-entry["task_count"], entry["kind"]))
    return semantic_rules


def build_semantic_counterexample_bank(
    task_pool: Sequence[Mapping[str, Any]], max_counterexamples: int = 8
) -> list[dict[str, Any]]:
    profiles = [semantic_task_profile(task) for task in task_pool]
    counterexamples: list[dict[str, Any]] = []

    for index, left in enumerate(profiles):
        left_ir = left.get("problem_ir", {})
        for right in profiles[index + 1 :]:
            right_ir = right.get("problem_ir", {})
            similarity = problem_ir_similarity(left_ir, right_ir)
            if similarity < 0.35:
                continue

            left_kind = str(left.get("focus", "general") or "general")
            right_kind = str(right.get("focus", "general") or "general")
            left_stage = str(left_ir.get("prototype_stage", ""))
            right_stage = str(right_ir.get("prototype_stage", ""))
            if (
                left_kind == right_kind
                and left_stage == right_stage
                and similarity < 0.55
            ):
                continue

            if left_kind == right_kind:
                contrast = "same_kind_near_neighbor"
            elif left_stage != right_stage:
                contrast = "stage_shift"
            else:
                contrast = "kind_shift"

            counterexamples.append(
                {
                    "left_kind": left_kind,
                    "right_kind": right_kind,
                    "left_signature": str(left_ir.get("signature", "")),
                    "right_signature": str(right_ir.get("signature", "")),
                    "similarity": round(similarity, 3),
                    "contrast": contrast,
                    "left_probe": list(left_ir.get("counterexample_hints", []))[:2],
                    "right_probe": list(right_ir.get("counterexample_hints", []))[:2],
                    "left_holdout_bucket": int(left_ir.get("holdout_bucket", 0) or 0),
                    "right_holdout_bucket": int(right_ir.get("holdout_bucket", 0) or 0),
                }
            )

    counterexamples.sort(
        key=lambda item: (
            -item["similarity"],
            item["contrast"],
            item["left_kind"],
            item["right_kind"],
        )
    )
    return counterexamples[:max_counterexamples]


def _task_holdout_bucket(profile: Mapping[str, Any], holdout_modulus: int = 5) -> int:
    problem_ir = profile.get("problem_ir", {})
    signature = str(problem_ir.get("signature", ""))
    if not signature:
        signature = str(profile.get("task_name", "")) or str(
            profile.get("source_text", "")
        )
    if not signature:
        return 0
    digest = hashlib.sha1(signature.encode("utf-8")).digest()
    return digest[0] % max(1, holdout_modulus)


def held_out_semantic_credit(
    task_pool: Sequence[Mapping[str, Any]],
    learned_macros: Mapping[str, Sequence[str]] | None = None,
    holdout_modulus: int = 5,
) -> dict[str, Any]:
    semantic_profiles = [semantic_task_profile(task) for task in task_pool]
    if not semantic_profiles:
        return {
            "train_count": 0,
            "held_out_count": 0,
            "kind_credit": {},
            "macro_credit": {},
            "split_strategy": f"mod_{holdout_modulus}",
        }

    train_profiles = [
        profile
        for profile in semantic_profiles
        if _task_holdout_bucket(profile, holdout_modulus) != 0
    ]
    held_out_profiles = [
        profile
        for profile in semantic_profiles
        if _task_holdout_bucket(profile, holdout_modulus) == 0
    ]
    if not train_profiles:
        train_profiles = semantic_profiles[:]
    if not held_out_profiles:
        held_out_profiles = semantic_profiles[-max(1, len(semantic_profiles) // 3) :]

    macro_credit: dict[str, dict[str, float]] = {}
    if learned_macros:
        for name, expansion in learned_macros.items():
            train_scores = [
                semantic_trace_alignment(expansion, profile)
                for profile in train_profiles
            ]
            held_out_scores = [
                semantic_trace_alignment(expansion, profile)
                for profile in held_out_profiles
            ]
            train_alignment = sum(train_scores) / max(1, len(train_scores))
            held_out_alignment = sum(held_out_scores) / max(1, len(held_out_scores))
            gap = max(0.0, train_alignment - held_out_alignment)
            macro_credit[name] = {
                "train_alignment": round(train_alignment, 3),
                "held_out_alignment": round(held_out_alignment, 3),
                "gap": round(gap, 3),
                "credit": round(
                    max(
                        0.0,
                        0.62 * held_out_alignment + 0.38 * train_alignment - 0.22 * gap,
                    ),
                    3,
                ),
            }

    kind_credit: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in semantic_profiles:
        by_kind[str(profile.get("focus", "general") or "general")].append(profile)

    for kind, profiles in sorted(
        by_kind.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        train_scores = []
        held_out_scores = []
        for profile in profiles:
            if learned_macros:
                macro_alignment = max(
                    (
                        semantic_trace_alignment(expansion, profile)
                        for expansion in learned_macros.values()
                    ),
                    default=float(profile.get("rule_weight", 0.0) or 0.0),
                )
            else:
                macro_alignment = float(profile.get("rule_weight", 0.0) or 0.0)
            if _task_holdout_bucket(profile, holdout_modulus) == 0:
                held_out_scores.append(float(macro_alignment))
            else:
                train_scores.append(float(macro_alignment))

        train_alignment = sum(train_scores) / max(1, len(train_scores))
        held_out_alignment = sum(held_out_scores) / max(1, len(held_out_scores))
        gap = max(0.0, train_alignment - held_out_alignment)
        credit = max(
            0.0, 0.64 * held_out_alignment + 0.36 * train_alignment - 0.18 * gap
        )
        kind_credit[kind] = {
            "train_alignment": round(train_alignment, 3),
            "held_out_alignment": round(held_out_alignment, 3),
            "gap": round(gap, 3),
            "credit": round(credit, 3),
            "support": len(profiles),
        }

    return {
        "train_count": len(train_profiles),
        "held_out_count": len(held_out_profiles),
        "kind_credit": kind_credit,
        "macro_credit": macro_credit,
        "split_strategy": f"mod_{holdout_modulus}",
    }


def semantic_prototype_lifecycle(
    task_pool: Sequence[Mapping[str, Any]],
    elite_flattened: Sequence[Sequence[str]],
    learned_macros: Mapping[str, Sequence[str]] | None = None,
    counterexamples: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    semantic_profiles = [semantic_task_profile(task) for task in task_pool]
    if not semantic_profiles:
        return []

    if counterexamples is None:
        counterexamples = build_semantic_counterexample_bank(task_pool)
    held_out_credit = held_out_semantic_credit(task_pool, learned_macros)
    counterexample_counts: Counter[str] = Counter()
    for entry in counterexamples:
        counterexample_counts[str(entry.get("left_kind", "general") or "general")] += 1
        counterexample_counts[str(entry.get("right_kind", "general") or "general")] += 1

    trace_pool = [list(trace) for trace in elite_flattened if trace]
    lifecycles: list[dict[str, Any]] = []
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in semantic_profiles:
        by_kind[str(profile.get("focus", "general") or "general")].append(profile)

    for kind, profiles in sorted(
        by_kind.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        representative = profiles[0]
        problem_ir = representative.get("problem_ir", {})
        support = len(profiles)
        uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)
        credit_info = held_out_credit["kind_credit"].get(kind, {})
        held_out_alignment = float(credit_info.get("held_out_alignment", 0.0) or 0.0)
        train_alignment = float(credit_info.get("train_alignment", 0.0) or 0.0)
        gap = float(credit_info.get("gap", 0.0) or 0.0)
        credit = float(credit_info.get("credit", 0.0) or 0.0)
        counterexample_count = int(counterexample_counts.get(kind, 0))
        trace_alignment = 0.0
        trace_examples: list[dict[str, Any]] = []
        if trace_pool:
            trace_scores = [
                (
                    semantic_trace_alignment(trace, representative),
                    list(trace[:12]),
                )
                for trace in trace_pool
            ]
            trace_scores.sort(key=lambda item: item[0], reverse=True)
            for score, trace in trace_scores[:2]:
                trace_examples.append({"trace": trace, "alignment": round(score, 3)})
            trace_alignment = trace_scores[0][0] if trace_scores else 0.0

        if credit >= 0.68 and counterexample_count <= 2 and support >= 3:
            stage = "mature"
            action = "promote"
        elif credit >= 0.52 and support >= 3:
            stage = "stable"
            action = "maintain"
        elif credit < 0.30 or uncertainty >= 0.60:
            stage = "candidate"
            action = "abstain"
        elif counterexample_count >= 4 and credit < 0.60:
            stage = "emerging"
            action = "split"
        elif support < 3:
            stage = "candidate"
            action = "merge"
        else:
            stage = "emerging"
            action = "refine"

        promoted_macros: list[str] = []
        if learned_macros:
            priority_tokens = set(representative.get("priority_tokens", []))
            for name, expansion in learned_macros.items():
                expansion_set = set(str(token) for token in expansion)
                if len(priority_tokens & expansion_set) >= max(
                    2, len(expansion_set) // 2
                ):
                    promoted_macros.append(name)

        lifecycles.append(
            {
                "kind": kind,
                "stage": stage,
                "action": action,
                "support": support,
                "signature": str(problem_ir.get("signature", "")),
                "prototype_stage": str(problem_ir.get("prototype_stage", stage)),
                "uncertainty": round(uncertainty, 3),
                "train_alignment": round(train_alignment, 3),
                "held_out_alignment": round(held_out_alignment, 3),
                "gap": round(gap, 3),
                "credit": round(credit, 3),
                "trace_alignment": round(trace_alignment, 3),
                "counterexample_count": counterexample_count,
                "counterexample_hint": list(problem_ir.get("counterexample_hints", []))[
                    :2
                ],
                "trace_examples": trace_examples,
                "promoted_macros": _dedupe_preserve_order(promoted_macros)[:6],
                "abstain": action == "abstain",
            }
        )

    lifecycles.sort(key=lambda entry: (-entry["support"], entry["kind"]))
    return lifecycles


def build_curriculum_profile(task_pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    semantic_profiles = [semantic_task_profile(task) for task in task_pool]
    stage_order = {"candidate": 0, "emerging": 1, "stable": 2, "mature": 3}
    schedule: list[dict[str, Any]] = []

    for task, profile in zip(task_pool, semantic_profiles):
        problem_ir = profile.get("problem_ir", {})
        stage = str(problem_ir.get("prototype_stage", "candidate") or "candidate")
        uncertainty = float(problem_ir.get("uncertainty", 0.0) or 0.0)
        support = len(problem_ir.get("entities", []))
        support += len(problem_ir.get("constraints", []))
        support += len(problem_ir.get("deliverables", []))
        support += len(problem_ir.get("risks", []))
        difficulty = round(
            0.48 * uncertainty
            + 0.05 * max(0, 6 - support)
            + 0.04 * len(problem_ir.get("risks", []))
            - 0.03 * float(profile.get("rule_weight", 1.0) or 1.0),
            3,
        )
        schedule.append(
            {
                "task": task,
                "task_name": str(task.get("name", f"task_{len(schedule)}")),
                "focus": str(profile.get("focus", "general") or "general"),
                "prototype_stage": stage,
                "difficulty": difficulty,
                "uncertainty": round(uncertainty, 3),
                "rule_weight": round(float(profile.get("rule_weight", 1.0) or 1.0), 3),
                "holdout_bucket": int(problem_ir.get("holdout_bucket", 0) or 0),
            }
        )

    schedule.sort(
        key=lambda entry: (
            stage_order.get(entry["prototype_stage"], 0),
            entry["difficulty"],
            entry["task_name"],
        )
    )
    stage_counts = Counter(entry["prototype_stage"] for entry in schedule)
    return {
        "task_count": len(schedule),
        "stage_order": stage_order,
        "stage_counts": dict(stage_counts),
        "schedule": [
            {key: value for key, value in entry.items() if key != "task"}
            for entry in schedule
        ],
        "ordered_tasks": [entry["task"] for entry in schedule],
    }


def build_primitive_contract_bank(
    task_pool: Sequence[Mapping[str, Any]],
    semantic_promotions: Sequence[Mapping[str, Any]],
    prototype_lifecycle: Sequence[Mapping[str, Any]] | None = None,
    held_out_credit: Mapping[str, Any] | None = None,
    max_contracts: int = 8,
) -> list[dict[str, Any]]:
    if not semantic_promotions and task_pool:
        semantic_promotions = build_semantic_promotion_bank(task_pool, [])

    lifecycle_by_kind = {
        str(entry.get("kind", "general") or "general"): dict(entry)
        for entry in (prototype_lifecycle or [])
    }
    held_out_by_kind = {
        str(kind): dict(value)
        for kind, value in dict(held_out_credit or {}).get("kind_credit", {}).items()
        if isinstance(value, Mapping)
    }

    contracts: list[dict[str, Any]] = []
    for index, rule in enumerate(semantic_promotions):
        kind = str(rule.get("kind", "general") or "general")
        lifecycle = lifecycle_by_kind.get(kind, {})
        credit_info = held_out_by_kind.get(kind, {})
        stage = str(
            lifecycle.get("stage", rule.get("prototype_stage", "candidate"))
            or "candidate"
        )
        support = int(rule.get("task_count", 0) or 0)
        credit = float(
            credit_info.get(
                "credit", lifecycle.get("credit", rule.get("rule_weight", 0.0))
            )
            or 0.0
        )
        if stage not in {"stable", "mature"} and support < 3 and credit < 0.55:
            continue

        primitive_tokens = _dedupe_preserve_order(
            [str(token) for token in rule.get("priority_tokens", [])[:5]]
            + [
                str(token)
                for template in rule.get("macro_templates", [])[:2]
                for token in template[:3]
            ]
        )
        if not primitive_tokens:
            continue

        guard_tokens = primitive_tokens[: min(3, len(primitive_tokens))]
        trace_examples = list(rule.get("trace_examples", []))[:2]
        task_examples = list(rule.get("task_examples", []))[:2]
        counterexample_count = int(lifecycle.get("counterexample_count", 0) or 0)
        promotion_score = round(
            0.46 * credit
            + 0.34 * float(rule.get("rule_weight", 0.0) or 0.0)
            + 0.20 * min(1.0, support / 5.0),
            3,
        )

        contracts.append(
            {
                "name": f"PRIM_{kind.upper()}_{index + 1}",
                "kind": kind,
                "stage": stage,
                "promotion_target": "primitive",
                "primitive_tokens": primitive_tokens[:6],
                "guard_tokens": guard_tokens,
                "preconditions": [
                    f"prototype_stage={stage}",
                    f"support>={support}",
                ]
                + (
                    [f"counterexamples<={counterexample_count}"]
                    if counterexample_count
                    else []
                ),
                "postconditions": [
                    f"held_out_credit>={credit:.3f}",
                    f"trace_alignment>={float(lifecycle.get('trace_alignment', 0.0) or 0.0):.3f}",
                ],
                "evidence": {
                    "task_examples": task_examples,
                    "trace_examples": trace_examples,
                    "counterexample_count": counterexample_count,
                    "held_out_credit": round(credit, 3),
                },
                "promotion_score": promotion_score,
                "contract_ready": stage in {"stable", "mature"}
                and promotion_score >= 0.55,
            }
        )

    contracts.sort(
        key=lambda entry: (
            -entry["promotion_score"],
            -int(entry["contract_ready"]),
            entry["kind"],
        )
    )
    return contracts[:max_contracts]


def compute_parallax_gradient(
    elite_flattened: Sequence[Sequence[str]],
    probe_flattened: Sequence[Sequence[str]],
    limit: int = 12,
) -> dict[str, Any]:
    elite_counts: Counter[str] = Counter()
    probe_counts: Counter[str] = Counter()
    elite_tokens: set[str] = set()
    probe_tokens: set[str] = set()

    for trace in elite_flattened:
        elite_counts.update(trace)
        elite_tokens.update(str(token) for token in trace)
    for trace in probe_flattened:
        probe_counts.update(trace)
        probe_tokens.update(str(token) for token in trace)

    total_elite = max(1, sum(elite_counts.values()))
    total_probe = max(1, sum(probe_counts.values()))
    gradient: dict[str, float] = {}
    for token in sorted(elite_tokens | probe_tokens):
        gradient[token] = round(
            (elite_counts[token] / total_elite) - (probe_counts[token] / total_probe),
            3,
        )

    positive = sorted(
        ((token, value) for token, value in gradient.items() if value > 0.0),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[:limit]
    negative = sorted(
        ((token, value) for token, value in gradient.items() if value < 0.0),
        key=lambda item: (item[1], item[0]),
    )[:limit]
    contrast = round(
        sum(abs(value) for value in gradient.values()) / max(1, len(gradient)),
        3,
    )
    return {
        "gradient": gradient,
        "positive_tokens": [token for token, _ in positive],
        "negative_tokens": [token for token, _ in negative],
        "contrast": contrast,
    }


def build_forbidden_subsequences(
    probe_flattened: Sequence[Sequence[str]],
    elite_flattened: Sequence[Sequence[str]] | None = None,
    lengths: Sequence[int] = (3, 4),
    max_patterns: int = 12,
) -> list[tuple[str, ...]]:
    probe_counts: Counter[tuple[str, ...]] = Counter()
    elite_counts: Counter[tuple[str, ...]] = Counter()

    for trace in probe_flattened:
        for length in lengths:
            if length <= 0 or length > len(trace):
                continue
            for index in range(len(trace) - length + 1):
                subsequence = tuple(trace[index : index + length])
                if len(set(subsequence)) >= 2:
                    probe_counts[subsequence] += 1

    for trace in elite_flattened or ():
        for length in lengths:
            if length <= 0 or length > len(trace):
                continue
            for index in range(len(trace) - length + 1):
                subsequence = tuple(trace[index : index + length])
                if len(set(subsequence)) >= 2:
                    elite_counts[subsequence] += 1

    scored: list[tuple[float, int, tuple[str, ...]]] = []
    for subsequence, count in probe_counts.items():
        if count < 2:
            continue
        score = count - elite_counts.get(subsequence, 0)
        if score > 0:
            scored.append((float(score), count, subsequence))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [subsequence for _, _, subsequence in scored[:max_patterns]]


def _matrix_vector_product(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        sum(row[index] * vector[index] for index in range(len(vector)))
        for row in matrix
    ]


def _vector_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        left[index] * right[index] for index in range(min(len(left), len(right)))
    )


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _outer_product(left: Sequence[float], right: Sequence[float]) -> list[list[float]]:
    return [[left_value * right_value for right_value in right] for left_value in left]


def _matrix_subtract(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [left_row[index] - right_row[index] for index in range(len(left_row))]
        for left_row, right_row in zip(left, right)
    ]


def _covariance_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    if not vectors:
        return []
    dimension = len(vectors[0])
    means = [0.0 for _ in range(dimension)]
    for vector in vectors:
        for index, value in enumerate(vector):
            means[index] += value
    means = [value / len(vectors) for value in means]

    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    denom = max(1, len(vectors) - 1)
    for vector in vectors:
        centered = [value - means[index] for index, value in enumerate(vector)]
        for row in range(dimension):
            for col in range(dimension):
                matrix[row][col] += centered[row] * centered[col]
    return [[value / denom for value in row] for row in matrix]


def _power_iteration(
    matrix: Sequence[Sequence[float]], max_iter: int = 40, tolerance: float = 1e-7
) -> tuple[float, list[float]]:
    if not matrix:
        return 0.0, []
    size = len(matrix)
    vector = [1.0 / size for _ in range(size)]
    for _ in range(max_iter):
        next_vector = _matrix_vector_product(matrix, vector)
        norm = _vector_norm(next_vector)
        if norm <= 1e-12:
            break
        next_vector = [value / norm for value in next_vector]
        if (
            max(abs(next_vector[index] - vector[index]) for index in range(size))
            < tolerance
        ):
            vector = next_vector
            break
        vector = next_vector
    eigenvalue = _vector_dot(vector, _matrix_vector_product(matrix, vector))
    return eigenvalue, vector


def _normalize_population_axes(
    population: Sequence[Any], labels: Sequence[str]
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    values_by_label: dict[str, list[float]] = {label: [] for label in labels}
    for individual in population:
        for label in labels:
            if isinstance(individual, Mapping):
                raw_value = individual.get(label, 0.0)
            else:
                raw_value = getattr(individual, label, 0.0)
            values_by_label[label].append(float(raw_value or 0.0))

    minimums = {
        label: min(values) if values else 0.0
        for label, values in values_by_label.items()
    }
    maximums = {
        label: max(values) if values else 0.0
        for label, values in values_by_label.items()
    }
    normalized_vectors: list[list[float]] = []
    for index in range(len(population)):
        normalized_vectors.append(
            [
                0.0
                if maximums[label] - minimums[label] <= 1e-12
                else (values_by_label[label][index] - minimums[label])
                / (maximums[label] - minimums[label])
                for label in labels
            ]
        )
    return normalized_vectors, minimums, maximums


def _population_value(item: Any, label: str, default: float = 0.0) -> float:
    if isinstance(item, Mapping):
        return float(item.get(label, default) or default)
    return float(getattr(item, label, default) or default)


def population_landscape_profile(population: Sequence[Any]) -> dict[str, Any]:
    if not population:
        return {
            "sample_size": 0,
            "axis_labels": [],
            "parallax": {},
            "rotation": {},
            "devolution": {},
        }

    axis_labels = [
        "train_score",
        "test_score",
        "solve_rate_test",
        "stability_score",
        "cognitive_stability",
        "market_score",
        "artifact_yield",
        "gen_efficiency",
        "task_threshold_pass",
    ]
    normalized_vectors, minimums, maximums = _normalize_population_axes(
        population, axis_labels
    )
    covariance = _covariance_matrix(normalized_vectors)
    principal_value, principal_vector = _power_iteration(covariance)
    deflated = covariance
    if principal_vector:
        deflated = _matrix_subtract(
            covariance,
            [
                [principal_value * value for value in row]
                for row in _outer_product(principal_vector, principal_vector)
            ],
        )
    secondary_value, secondary_vector = _power_iteration(deflated)

    axis_variance = {
        label: round(
            statistics.pvariance([vector[index] for vector in normalized_vectors]), 3
        )
        if len(normalized_vectors) > 1
        else 0.0
        for index, label in enumerate(axis_labels)
    }
    axis_spread = [axis_variance[label] for label in axis_labels]
    positive_spread = [value for value in axis_spread if value > 1e-9]
    condition_number = (
        max(positive_spread) / max(min(positive_spread), 1e-9)
        if positive_spread
        else 0.0
    )

    ordered = sorted(
        population, key=lambda item: _population_value(item, "governed_fitness")
    )
    lower_slice = ordered[: max(1, len(ordered) // 4)]
    upper_slice = ordered[-max(1, len(ordered) // 4) :]
    top_governed = statistics.mean(
        _population_value(item, "governed_fitness") for item in upper_slice
    )
    bottom_governed = statistics.mean(
        _population_value(item, "governed_fitness") for item in lower_slice
    )
    parallax_pressure = top_governed - bottom_governed
    governed_spread = (
        statistics.pstdev(
            _population_value(item, "governed_fitness") for item in ordered
        )
        if len(ordered) > 1
        else 0.0
    )
    normalized_pressure = parallax_pressure / max(
        abs(top_governed), abs(bottom_governed), governed_spread, 1.0
    )

    gradient = {
        label: round(
            statistics.mean(_population_value(item, label) for item in upper_slice)
            - statistics.mean(_population_value(item, label) for item in lower_slice),
            3,
        )
        for label in axis_labels
    }

    deformation_counts = Counter(
        (
            item.get("specialization", "generalist")
            if isinstance(item, Mapping)
            else getattr(item, "specialization", "generalist")
        )
        for item in lower_slice
    )
    dominant_axis = max(axis_variance.items(), key=lambda item: (item[1], item[0]))[0]
    rotation_basis = {
        "principal_axis": [
            {"label": label, "loading": round(value, 3)}
            for label, value in sorted(
                zip(axis_labels, principal_vector),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]
        if principal_vector
        else [],
        "secondary_axis": [
            {"label": label, "loading": round(value, 3)}
            for label, value in sorted(
                zip(axis_labels, secondary_vector),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]
        if secondary_vector
        else [],
        "principal_eigenvalue": round(principal_value, 3),
        "secondary_eigenvalue": round(secondary_value, 3),
        "condition_number": round(condition_number, 3),
    }

    return {
        "sample_size": len(population),
        "axis_labels": axis_labels,
        "axis_means": {
            label: round(
                statistics.mean(
                    getattr(item, label, 0.0) or 0.0 for item in population
                ),
                3,
            )
            for label in axis_labels
        },
        "axis_variance": axis_variance,
        "axis_range": {
            label: round(maximums[label] - minimums[label], 3) for label in axis_labels
        },
        "dominant_axis": dominant_axis,
        "rotation": rotation_basis,
        "parallax": {
            "top_governed_mean": round(top_governed, 3),
            "bottom_governed_mean": round(bottom_governed, 3),
            "pressure": round(normalized_pressure, 3),
            "valley_depth": round(bottom_governed, 3),
            "gradient": gradient,
        },
        "devolution": {
            "worst_count": len(lower_slice),
            "worst_mean_governed": round(bottom_governed, 3),
            "worst_specializations": dict(deformation_counts),
        },
        "orthogonality": {
            "condition_number": round(condition_number, 3),
            "principal_axis": dominant_axis,
            "principal_loading_count": len(
                cast(list[Any], rotation_basis["principal_axis"])
            ),
        },
    }


def optimizer_token_histogram(flat: Sequence[str]) -> dict[str, int]:
    return _optimizer_token_histogram_cached(_trace_key(flat))


@lru_cache(maxsize=4096)
def _optimizer_token_histogram_cached(flat_key: tuple[str, ...]) -> dict[str, int]:
    return {
        "primitive": sum(1 for tok in flat_key if tok in PRIMITIVES),
        "artifact_op": sum(1 for tok in flat_key if tok in ARTIFACT_OPS),
        "macro": sum(
            1 for tok in flat_key if tok not in PRIMITIVES and tok not in ARTIFACT_OPS
        ),
        "verify": flat_key.count("VERIFY"),
        "repair": flat_key.count("REPAIR"),
        "plan": flat_key.count("PLAN") + flat_key.count("WRITE_PLAN"),
        "patch": flat_key.count("WRITE_PATCH"),
        "summary": flat_key.count("WRITE_SUMMARY") + flat_key.count("SUMMARIZE"),
    }


def optimizer_combinators(
    flat: Sequence[str], hist: Mapping[str, int], inv: Mapping[str, int]
) -> dict[str, int]:
    return _optimizer_combinators_cached(_trace_key(flat))


@lru_cache(maxsize=4096)
def _optimizer_combinators_cached(flat_key: tuple[str, ...]) -> dict[str, int]:
    inv = _artifact_profile_cached(flat_key)
    return {
        "discover_chain": _count_non_overlapping_subseq_cached(
            flat_key, ("OBSERVE", "RETRIEVE", "VERIFY")
        ),
        "plan_chain": _count_non_overlapping_subseq_cached(
            flat_key, ("PLAN", "REPAIR", "VERIFY")
        )
        + _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_PLAN", "READ_ARTIFACT", "PLAN")
        ),
        "patch_chain": _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_PATCH", "READ_ARTIFACT", "REPAIR")
        ),
        "summary_chain": _count_non_overlapping_subseq_cached(
            flat_key, ("WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE")
        ),
        "artifact_loops": inv.get("reads", 0)
        + inv.get("revisions", 0)
        + inv.get("links", 0),
        "macro_diversity": max(
            0, len([k for k, v in Counter(flat_key).items() if v > 0]) - 1
        ),
    }


def optimizer_geometry(
    flat: Sequence[str], hist: Mapping[str, int], combinators: Mapping[str, int]
) -> dict[str, float]:
    vector = [
        float(hist["primitive"]),
        float(hist["artifact_op"]),
        float(hist["macro"]),
        float(hist["verify"]),
        float(hist["repair"]),
        float(hist["plan"]),
        float(hist["patch"]),
        float(hist["summary"]),
    ]
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    normalized = [value / magnitude for value in vector]
    phase_total = max(
        1.0, float(hist["plan"] + hist["patch"] + hist["summary"] + hist["verify"])
    )
    plan_axis = hist["plan"] / phase_total
    patch_axis = hist["patch"] / phase_total
    summary_axis = hist["summary"] / phase_total
    verify_axis = hist["verify"] / phase_total
    balance = 1.0 - max(plan_axis, patch_axis, summary_axis, verify_axis)
    centroid = sum(index * value for index, value in enumerate(normalized)) / max(
        1, len(normalized)
    )
    spread = sum(
        (value - (sum(normalized) / len(normalized))) ** 2 for value in normalized
    ) / len(normalized)
    mixed_pressure = max(
        0.0, 1.0 - max(plan_axis, patch_axis, summary_axis, verify_axis)
    )
    return {
        "magnitude": round(magnitude, 3),
        "centroid": round(centroid, 3),
        "spread": round(spread, 3),
        "balance": round(balance, 3),
        "plan_axis": round(plan_axis, 3),
        "patch_axis": round(patch_axis, 3),
        "summary_axis": round(summary_axis, 3),
        "verify_axis": round(verify_axis, 3),
        "mixed_pressure": round(mixed_pressure, 3),
        "combinator_mass": round(sum(combinators.values()), 3),
    }


def build_optimizer_frame(flat: Sequence[str]) -> dict[str, Any]:
    return _build_optimizer_frame_cached(_trace_key(flat))


@lru_cache(maxsize=4096)
def _build_optimizer_frame_cached(flat_key: tuple[str, ...]) -> dict[str, Any]:
    inv = _artifact_profile_cached(flat_key)
    hist = _optimizer_token_histogram_cached(flat_key)
    combinators = _optimizer_combinators_cached(flat_key)
    geometry = optimizer_geometry(flat_key, hist, combinators)
    ast_profile = {
        "node_count": len(flat_key),
        "leaf_count": sum(
            1 for tok in flat_key if tok in PRIMITIVES or tok in ARTIFACT_OPS
        ),
        "branch_count": len([k for k, v in Counter(flat_key).items() if v > 1]),
        "max_depth": 3
        if combinators["discover_chain"]
        or combinators["plan_chain"]
        or combinators["patch_chain"]
        or combinators["summary_chain"]
        else 1,
    }
    return {
        "ast_profile": ast_profile,
        "token_histogram": hist,
        "combinators": combinators,
        "geometry": geometry,
        "artifact_profile": inv,
    }
