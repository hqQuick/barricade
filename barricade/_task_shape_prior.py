from __future__ import annotations

from typing import Any

from .problem_ir import problem_ir_similarity


def _task_shape_tokens(task_pool: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for task in task_pool:
        if not isinstance(task, dict):
            continue
        name = str(task.get("name", "")).strip()
        if name:
            tokens.append(f"name:{name}")
        focus = str(task.get("focus", "")).strip()
        if focus:
            tokens.append(f"focus:{focus}")
        requirements = task.get("req", [])
        if isinstance(requirements, list):
            tokens.extend(
                f"req:{str(token)}" for token in requirements if str(token).strip()
            )
        needs = task.get("needs", {})
        if isinstance(needs, dict):
            for key, value in sorted(needs.items()):
                tokens.append(f"need:{key}:{value}")
    return tokens


def _sequence_similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(left), len(right))


def _centroid_similarity(
    current_profile: dict[str, Any], candidate_profile: dict[str, Any]
) -> float:
    current_centroid = current_profile.get("shape_centroid", {})
    candidate_centroid = candidate_profile.get("shape_centroid", {})
    if not isinstance(current_centroid, dict) or not isinstance(
        candidate_centroid, dict
    ):
        return 0.0

    current_weights = current_centroid.get("lane_weights", {})
    candidate_weights = candidate_centroid.get("lane_weights", {})
    if not isinstance(current_weights, dict) or not isinstance(candidate_weights, dict):
        return 0.0

    lanes = ("planning", "patching", "summarizing", "recovery", "reasoning")
    current_vec = [float(current_weights.get(lane, 0.0) or 0.0) for lane in lanes]
    candidate_vec = [float(candidate_weights.get(lane, 0.0) or 0.0) for lane in lanes]
    l1_distance = sum(
        abs(left - right) for left, right in zip(current_vec, candidate_vec)
    )
    centroid_score = max(0.0, 1.0 - 0.5 * l1_distance)
    if current_centroid.get("dominant_lane") == candidate_centroid.get("dominant_lane"):
        centroid_score += 0.05
    return round(min(1.0, centroid_score), 3)


def _task_shape_profile(
    intake: dict[str, Any], task_pool: list[dict[str, Any]], feed_prior_dna: list[str]
) -> dict[str, Any]:
    shape_lanes = intake.get("shape_lanes", {})
    shape_centroid = intake.get("shape_centroid", {})
    problem_ir = intake.get("problem_ir", {})
    return {
        "route_hint": str(intake.get("route_hint", "")),
        "expected_artifact_type": str(intake.get("expected_artifact_type", "")),
        "domain_tags": [str(tag) for tag in intake.get("domain_tags", []) if str(tag)],
        "shape_tokens": _task_shape_tokens(task_pool),
        "shape_lanes": shape_lanes if isinstance(shape_lanes, dict) else {},
        "shape_centroid": shape_centroid if isinstance(shape_centroid, dict) else {},
        "problem_ir_signature": str(intake.get("problem_ir_signature", "")),
        "problem_ir": problem_ir if isinstance(problem_ir, dict) else {},
        "route_explanation": list(intake.get("route_explanation", []))
        if isinstance(intake.get("route_explanation", []), list)
        else [],
        "feed_prior_dna": [str(token) for token in feed_prior_dna],
    }


def _task_shape_similarity(
    current_profile: dict[str, Any], candidate_profile: dict[str, Any]
) -> float:
    current_tokens = [str(token) for token in current_profile.get("shape_tokens", [])]
    candidate_tokens = [
        str(token) for token in candidate_profile.get("shape_tokens", [])
    ]
    current_dna = [str(token) for token in current_profile.get("feed_prior_dna", [])]
    candidate_dna = [
        str(token) for token in candidate_profile.get("feed_prior_dna", [])
    ]

    dna_score = _sequence_similarity(current_dna, candidate_dna)
    token_score = _sequence_similarity(current_tokens, candidate_tokens)
    route_bonus = (
        0.08
        if current_profile.get("route_hint")
        and current_profile.get("route_hint") == candidate_profile.get("route_hint")
        else 0.0
    )
    artifact_bonus = (
        0.04
        if current_profile.get("expected_artifact_type")
        and current_profile.get("expected_artifact_type")
        == candidate_profile.get("expected_artifact_type")
        else 0.0
    )

    current_domains = set(current_profile.get("domain_tags", []))
    candidate_domains = set(candidate_profile.get("domain_tags", []))
    domain_bonus = 0.0
    if current_domains and candidate_domains:
        domain_bonus = (
            0.05
            * len(current_domains & candidate_domains)
            / len(current_domains | candidate_domains)
        )

    current_ir = current_profile.get("problem_ir", {})
    candidate_ir = candidate_profile.get("problem_ir", {})
    ir_bonus = 0.18 * problem_ir_similarity(current_ir, candidate_ir)

    centroid_bonus = 0.10 * _centroid_similarity(current_profile, candidate_profile)

    return round(
        min(
            1.0,
            0.62 * dna_score
            + 0.18 * token_score
            + ir_bonus
            + centroid_bonus
            + route_bonus
            + artifact_bonus
            + domain_bonus,
        ),
        3,
    )


def _task_shape_cache_score(benchmark_result: dict[str, Any]) -> float:
    summary = (
        benchmark_result.get("summary", {})
        if isinstance(benchmark_result.get("summary", {}), dict)
        else {}
    )
    best_governed = (
        benchmark_result.get("best_governed", {})
        if isinstance(benchmark_result.get("best_governed", {}), dict)
        else {}
    )
    best_raw = (
        benchmark_result.get("best_raw", {})
        if isinstance(benchmark_result.get("best_raw", {}), dict)
        else {}
    )

    summary_score = float(
        summary.get("score", summary.get("governed_score", 0.0)) or 0.0
    )
    governed_score = float(best_governed.get("governed_fitness", 0.0) or 0.0)
    price_score = float(best_governed.get("price", best_raw.get("price", 0.0)) or 0.0)
    return round(summary_score + 0.02 * governed_score + 0.01 * price_score, 3)


PRIOR_STRENGTH_DEFAULTS = {
    "summary": 0.9,
    "raft": 0.7,
    "complex": 0.4,
    "math": 0.62,
    "general": 0.55,
}

PRIOR_STRENGTH_CAPS = {
    "summary": 0.98,
    "raft": 0.82,
    "complex": 0.55,
    "math": 0.78,
    "general": 0.85,
}


def _classify_task_shape(
    intake: dict[str, Any], problem_text: str, task_pool: list[dict[str, Any]]
) -> str:
    from collections import Counter

    lowered = problem_text.lower()
    route_hint = str(intake.get("route_hint", "")).lower()
    artifact_type = str(intake.get("expected_artifact_type", "")).lower()
    confidence = float(intake.get("confidence", 0.0) or 0.0)
    risk_count = len([risk for risk in intake.get("risks", []) if risk])
    domain_tags = {
        str(tag).lower() for tag in intake.get("domain_tags", []) if str(tag)
    }
    problem_ir = intake.get("problem_ir", {})
    problem_ir_kind = (
        str(problem_ir.get("kind", "")) if isinstance(problem_ir, dict) else ""
    )
    centroid = intake.get("shape_centroid", {})
    centroid_weights = (
        centroid.get("lane_weights", {}) if isinstance(centroid, dict) else {}
    )
    reasoning_weight = float(centroid_weights.get("reasoning", 0.0) or 0.0)
    planning_weight = float(centroid_weights.get("planning", 0.0) or 0.0)

    raft_markers = (
        "raft",
        "leader election",
        "requestvote",
        "appendentries",
        "log replication",
        "quorum",
    )
    summary_markers = (
        "summarize",
        "summary",
        "review",
        "report",
        "compare",
        "document",
        "explain the findings",
    )
    complex_markers = (
        "rollback",
        "retries",
        "retry",
        "recovery",
        "recover",
        "failing",
        "failure",
        "incident",
        "manual intervention",
        "brittle",
    )
    math_markers = (
        "prove",
        "proof",
        "theorem",
        "lemma",
        "centroid",
        "geometry",
        "geometric",
        "triangle",
        "median",
        "medians",
        "incenter",
        "circumcenter",
        "equation",
        "derive",
        "derivation",
        "combinatorics",
        "probability",
        "solve",
    )

    if any(marker in lowered for marker in raft_markers):
        return "raft"
    if problem_ir_kind == "proof" or any(marker in lowered for marker in math_markers):
        return "math"
    if (
        reasoning_weight >= 0.25
        and reasoning_weight >= planning_weight
        and {"math", "reasoning"} & domain_tags
    ):
        return "math"
    if (
        route_hint == "summarizing"
        or artifact_type == "summary"
        or any(marker in lowered for marker in summary_markers)
    ):
        return "summary"
    if route_hint == "recovery" or any(marker in lowered for marker in complex_markers):
        return "complex"
    if route_hint == "mixed" or confidence < 0.60 or risk_count >= 3:
        return "complex"

    focus_counts = Counter(
        str(task.get("focus", "")) for task in task_pool if isinstance(task, dict)
    )
    if focus_counts.get("summarizing", 0) >= 2:
        return "summary"
    if focus_counts.get("patching", 0) >= 3 and focus_counts.get("planning", 0) >= 2:
        return "complex"
    return "general"


def _prior_strength_profile(
    intake: dict[str, Any],
    problem_text: str,
    task_pool: list[dict[str, Any]],
    requested_prior_strength: float | None = None,
) -> dict[str, Any]:
    task_class = _classify_task_shape(intake, problem_text, task_pool)
    default_strength = PRIOR_STRENGTH_DEFAULTS.get(
        task_class, PRIOR_STRENGTH_DEFAULTS["general"]
    )
    requested_strength = (
        default_strength
        if requested_prior_strength is None
        else max(0.0, min(1.0, round(float(requested_prior_strength), 3)))
    )
    max_strength = PRIOR_STRENGTH_CAPS.get(task_class, PRIOR_STRENGTH_CAPS["general"])
    effective_strength = min(requested_strength, max_strength)
    cache_min_similarity = round(
        max(0.45, min(0.90, 0.56 + 0.30 * (1.0 - effective_strength))), 3
    )
    return {
        "task_class": task_class,
        "default_strength": default_strength,
        "requested_strength": requested_strength,
        "effective_strength": effective_strength,
        "max_strength": max_strength,
        "cache_min_similarity": cache_min_similarity,
        "blend_strength": round(effective_strength, 3),
        "guarded": effective_strength < requested_strength,
    }


def _load_best_task_shape_prior(
    state_root, current_profile: dict[str, Any], min_similarity: float = 0.60
) -> dict[str, Any] | None:
    from .feed_derived_dna.persistence import load_task_shape_priors

    if state_root is None:
        return None

    best_entry: dict[str, Any] | None = None
    best_similarity = 0.0
    best_score = -1.0
    for entry in load_task_shape_priors(state_root):
        if not isinstance(entry, dict):
            continue
        candidate_profile = entry.get("shape_profile", {})
        if not isinstance(candidate_profile, dict):
            continue
        similarity = _task_shape_similarity(current_profile, candidate_profile)
        score = float(entry.get("score", 0.0) or 0.0)
        if similarity < min_similarity:
            continue
        if similarity > best_similarity or (
            similarity == best_similarity and score > best_score
        ):
            best_entry = entry
            best_similarity = similarity
            best_score = score

    if best_entry is None:
        return None

    return {
        "entry": best_entry,
        "similarity": best_similarity,
        "score": best_score,
    }


def _blend_token_sequence(
    current: list[str], cached: list[str], strength: float
) -> list[str]:
    current_sequence = [str(token) for token in current if str(token)]
    cached_sequence = [str(token) for token in cached if str(token)]
    if not cached_sequence:
        return current_sequence
    if strength <= 0.0:
        return current_sequence
    if strength >= 1.0:
        return cached_sequence

    cached_take = max(
        1, min(len(cached_sequence), round(len(cached_sequence) * strength))
    )
    blended = list(cached_sequence[:cached_take])
    for token in current_sequence:
        if token not in blended:
            blended.append(token)
    return blended[: max(len(current_sequence), len(cached_sequence))]


def _materialize_task_shape_hit(
    cache_hit: dict[str, Any],
    task_pool: list[dict[str, Any]],
    feed_prior_dna: list[str],
    patch_skeleton: dict[str, Any],
    persisted_macros: dict[str, list[str]],
    prior_profile: dict[str, Any],
) -> dict[str, Any]:
    entry = cache_hit["entry"]
    snapshot = (
        entry.get("snapshot", {}) if isinstance(entry.get("snapshot", {}), dict) else {}
    )
    learned_macros = dict(persisted_macros)
    cached_macros = (
        snapshot.get("learned_macros", {})
        if isinstance(snapshot.get("learned_macros", {}), dict)
        else {}
    )
    learned_macros.update(
        {
            str(name): [str(token) for token in sequence]
            for name, sequence in cached_macros.items()
            if isinstance(sequence, list)
        }
    )

    effective_strength = float(
        prior_profile.get(
            "effective_strength", prior_profile.get("blend_strength", 0.0)
        )
        or 0.0
    )
    cached_feed_prior_dna = (
        [str(token) for token in snapshot.get("feed_prior_dna", [])]
        if isinstance(snapshot.get("feed_prior_dna", []), list)
        else []
    )
    shaped_feed_prior_dna = _blend_token_sequence(
        feed_prior_dna, cached_feed_prior_dna or feed_prior_dna, effective_strength
    )

    shaped_patch_skeleton = dict(patch_skeleton)
    cached_patch_skeleton = (
        snapshot.get("patch_skeleton", {})
        if isinstance(snapshot.get("patch_skeleton", {}), dict)
        else {}
    )
    current_outline = (
        [str(token) for token in patch_skeleton.get("token_outline", [])]
        if isinstance(patch_skeleton.get("token_outline", []), list)
        else []
    )
    cached_outline = (
        [str(token) for token in cached_patch_skeleton.get("token_outline", [])]
        if isinstance(cached_patch_skeleton.get("token_outline", []), list)
        else []
    )
    if current_outline or cached_outline:
        shaped_patch_skeleton["token_outline"] = _blend_token_sequence(
            current_outline or feed_prior_dna,
            cached_outline or current_outline,
            effective_strength,
        )
    if effective_strength >= 0.75:
        for field in ("transfer_target", "primary_focus", "focus", "mode"):
            if cached_patch_skeleton.get(field):
                shaped_patch_skeleton[field] = cached_patch_skeleton[field]

    result = dict(snapshot)
    result.update(
        {
            "task_pool": task_pool,
            "feed_prior_dna": shaped_feed_prior_dna,
            "patch_skeleton": shaped_patch_skeleton,
            "learned_macros": learned_macros,
            "cache_hit": True,
            "cache_similarity": cache_hit["similarity"],
            "cache_score": cache_hit["score"],
            "prior_profile": prior_profile,
        }
    )
    return result
