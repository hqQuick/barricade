from __future__ import annotations

import re
from typing import Any

from ..workflow_intake import tokenize
from ..feed_derived_dna.analysis import (
    artifact_profile,
    build_optimizer_frame,
    estimate_artifact_quality,
    family_signature,
    infer_specialization,
    motif_key,
    semantic_family_credit,
)


STEP_GUIDANCE = {
    "OBSERVE": {
        "tool_hint": "submit_step",
        "kind": "observation",
        "title": "OBSERVE",
        "instruction": "Register the task context. Capture constraints, assumptions, relevant files, and the sharpest risks.",
    },
    "PLAN": {
        "tool_hint": "submit_step",
        "kind": "plan",
        "title": "PLAN",
        "instruction": "Produce a concise decomposition of the work. Keep the plan actionable and ordered.",
    },
    "REPAIR": {
        "tool_hint": "submit_step",
        "kind": "repair",
        "title": "REPAIR",
        "instruction": "Describe the correction or remediation needed after the latest failure. Keep it specific.",
    },
    "WRITE_PLAN": {
        "tool_hint": "submit_step",
        "kind": "plan",
        "title": "WRITE_PLAN",
        "instruction": "Write the architectural plan the executor should follow. Include phases, validation, and risk gates.",
    },
    "WRITE_PATCH": {
        "tool_hint": "submit_step",
        "kind": "patch",
        "title": "WRITE_PATCH",
        "instruction": "Write patch content that can become governed file updates. Prefer JSON with an updates mapping, or file blocks using 'File:' headers.",
    },
    "VERIFY": {
        "tool_hint": "verify_step",
        "kind": "verification",
        "title": "VERIFY",
        "instruction": "Run the verification command that proves the current change is safe.",
    },
    "VERIFY_CODE": {
        "tool_hint": "verify_step",
        "kind": "verification",
        "title": "VERIFY_CODE",
        "instruction": "Verify code-level correctness, syntax, and execution behavior.",
    },
    "VERIFY_DATA": {
        "tool_hint": "verify_step",
        "kind": "verification",
        "title": "VERIFY_DATA",
        "instruction": "Verify data shape, schema, and content consistency.",
    },
    "VERIFY_CONSTRAINTS": {
        "tool_hint": "verify_step",
        "kind": "verification",
        "title": "VERIFY_CONSTRAINTS",
        "instruction": "Verify IR constraints and invariants against the task outcome.",
    },
    "VERIFY_ENV": {
        "tool_hint": "verify_step",
        "kind": "verification",
        "title": "VERIFY_ENV",
        "instruction": "Verify environment assumptions, dependencies, and runtime context.",
    },
    "REANCHOR": {
        "tool_hint": "submit_step",
        "kind": "reanchor",
        "title": "REANCHOR",
        "instruction": "Reset the trajectory to verified truth using IR and trusted artifacts.",
    },
    "SUMMARIZE": {
        "tool_hint": "submit_step",
        "kind": "summary",
        "title": "SUMMARIZE",
        "instruction": "Summarize what happened, what changed, and what residual risk remains.",
    },
    "ROLLBACK": {
        "tool_hint": "submit_step",
        "kind": "rollback",
        "title": "ROLLBACK",
        "instruction": "Describe or enact the rollback/remediation path.",
    },
    "READ_ARTIFACT": {
        "tool_hint": "read_artifact",
        "kind": "read",
        "title": "READ_ARTIFACT",
        "instruction": "Inspect the referenced artifact before continuing.",
    },
    "LINK_ARTIFACT": {
        "tool_hint": "submit_step",
        "kind": "link",
        "title": "LINK_ARTIFACT",
        "instruction": "Explain how this artifact links back to earlier evidence or steps.",
    },
    "COMMIT": {
        "tool_hint": "submit_step",
        "kind": "commit",
        "title": "COMMIT",
        "instruction": "Prepare the governed commit handoff.",
    },
}


ARTIFACT_PREFIXES = {
    "OBSERVE": "OBS",
    "PLAN": "PLAN",
    "REPAIR": "REPA",
    "WRITE_PLAN": "PLAN",
    "WRITE_PATCH": "PATC",
    "VERIFY": "VERI",
    "VERIFY_CODE": "VCOD",
    "VERIFY_DATA": "VDAT",
    "VERIFY_CONSTRAINTS": "VCON",
    "VERIFY_ENV": "VENV",
    "REANCHOR": "REAN",
    "SUMMARIZE": "SUMM",
    "ROLLBACK": "ROLL",
    "READ_ARTIFACT": "READ",
    "LINK_ARTIFACT": "LINK",
    "COMMIT": "COMM",
}


def _token_prefix(token: str) -> str:
    return ARTIFACT_PREFIXES.get(
        token, re.sub(r"[^A-Z0-9]", "", token.upper())[:4] or "TOK"
    )


def _step_trace(
    token: str, content: str, verification: dict[str, Any] | None = None
) -> list[str]:
    if token == "VERIFY":
        if verification and verification.get("passed"):
            return ["VERIFY", "SUMMARIZE"]
        return ["VERIFY", "REPAIR"]
    if token in {"VERIFY_CODE", "VERIFY_DATA", "VERIFY_CONSTRAINTS", "VERIFY_ENV"}:
        if verification and verification.get("passed"):
            return [token, "SUMMARIZE"]
        return [token, "REPAIR"]
    if token == "REANCHOR":
        return ["REANCHOR", "VERIFY_CONSTRAINTS", "VERIFY"]
    if token == "OBSERVE":
        return ["OBSERVE", "RETRIEVE", "VERIFY"]
    if token == "PLAN":
        return ["PLAN", "REPAIR", "VERIFY"]
    if token == "REPAIR":
        return ["REPAIR", "VERIFY"]
    if token == "WRITE_PLAN":
        return ["WRITE_PLAN", "READ_ARTIFACT", "PLAN"]
    if token == "WRITE_PATCH":
        return ["WRITE_PATCH", "READ_ARTIFACT", "REPAIR"]
    if token == "SUMMARIZE":
        return ["WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE"]
    if token == "ROLLBACK":
        return ["ROLLBACK", "REPAIR", "VERIFY"]
    if token == "READ_ARTIFACT":
        return ["READ_ARTIFACT", "OBSERVE"]
    if token == "LINK_ARTIFACT":
        return ["LINK_ARTIFACT", "SUMMARIZE"]
    if token == "COMMIT":
        return ["COMMIT", "VERIFY"]
    return [token]


def _content_signals(content: str) -> dict[str, Any]:
    lowered = content.lower()
    return {
        "token_count": len(tokenize(content)),
        "line_count": content.count("\n") + (1 if content else 0),
        "has_code_fence": "```" in content,
        "has_json": content.lstrip().startswith("{")
        or content.lstrip().startswith("["),
        "has_python": any(
            marker in lowered
            for marker in ("import ", "def ", "class ", "async def", "fastapi")
        ),
        "mentions_plan": any(
            marker in lowered
            for marker in ("plan", "architecture", "phase", "roadmap", "steps")
        ),
        "mentions_patch": any(
            marker in lowered
            for marker in ("patch", "update", "replace", "file:", "*** update file:")
        ),
        "mentions_verify": any(
            marker in lowered
            for marker in ("test", "verify", "syntax", "lint", "pytest", "compile")
        ),
        "mentions_summary": any(
            marker in lowered
            for marker in ("summary", "summarize", "review", "wrap up")
        ),
    }


def _artifact_score(
    token: str, content: str, verification: dict[str, Any] | None = None
) -> tuple[float, dict[str, Any]]:
    synthetic_trace = _step_trace(token, content, verification=verification)
    family_score, family_counts = semantic_family_credit(synthetic_trace)
    quality_scores = estimate_artifact_quality(synthetic_trace)
    specialisation = infer_specialization(synthetic_trace)
    optimizer_frame = build_optimizer_frame(synthetic_trace)
    signals = _content_signals(content)

    base_quality = max(quality_scores.values()) if quality_scores else 0.0
    score = (
        family_score
        + base_quality * 2.4
        + signals["token_count"] * 0.03
        + signals["line_count"] * 0.05
    )
    if token in {"WRITE_PATCH", "REPAIR"} and signals["has_code_fence"]:
        score += 1.5
    if token in {"WRITE_PATCH", "REPAIR"} and signals["has_python"]:
        score += 1.0
    if token in {"WRITE_PLAN", "PLAN"} and signals["mentions_plan"]:
        score += 0.8
    if token == "SUMMARIZE" and signals["mentions_summary"]:
        score += 0.6
    if token == "VERIFY" and verification:
        score += 2.0 if verification.get("passed") else 0.2

    metadata = {
        "trace": synthetic_trace,
        "family_score": family_score,
        "family_counts": family_counts,
        "quality_scores": quality_scores,
        "optimizer_frame": optimizer_frame,
        "specialization": specialisation,
        "family_signature": family_signature(synthetic_trace),
        "motif_sig": motif_key(synthetic_trace),
        "artifact_profile": artifact_profile(synthetic_trace),
        "signals": signals,
    }
    return score, metadata


def _artifact_price(score: float, content: str, token: str) -> float:
    complexity = len(tokenize(content)) / 24.0
    bonus = (
        1.0
        if token == "WRITE_PATCH"
        else 0.6
        if token == "WRITE_PLAN"
        else 0.3
        if token == "SUMMARIZE"
        else 0.2
    )
    return round(max(1.0, 1.25 + score * 0.8 + complexity + bonus), 2)


def _artifact_kind(token: str) -> str:
    guidance = STEP_GUIDANCE.get(token)
    if guidance:
        return str(guidance["kind"])
    return token.lower()
