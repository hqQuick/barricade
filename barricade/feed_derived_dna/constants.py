from __future__ import annotations

PRIMITIVES = [
    "OBSERVE",
    "RETRIEVE",
    "VERIFY",
    "REPAIR",
    "SUMMARIZE",
    "CACHE",
    "COMMIT",
    "ROLLBACK",
    "SWITCH_CONTEXT",
    "ESCALATE",
    "QUERY_TOOL",
    "PLAN",
]
CONTROL_TOKENS = [
    "VERIFY_CODE",
    "VERIFY_DATA",
    "VERIFY_CONSTRAINTS",
    "VERIFY_ENV",
    "REANCHOR",
]
ARTIFACT_OPS = [
    "WRITE_NOTE",
    "WRITE_PLAN",
    "WRITE_PATCH",
    "WRITE_SUMMARY",
    "READ_ARTIFACT",
    "REVISE_ARTIFACT",
    "LINK_ARTIFACT",
    "DROP_ARTIFACT",
]
ALL_TOKENS = PRIMITIVES + ARTIFACT_OPS

BASE_MACROS = {
    "DISCOVER": ["OBSERVE", "RETRIEVE", "VERIFY"],
    "SAFE_REPORT": ["VERIFY", "REPAIR", "SUMMARIZE"],
    "RECOVER": ["VERIFY", "ROLLBACK", "REPAIR"],
    "MEMOIZE": ["CACHE", "RETRIEVE", "COMMIT"],
    "PATCH_LOOP": ["PLAN", "REPAIR", "VERIFY"],
    "DIAGNOSE": ["SWITCH_CONTEXT", "VERIFY", "REPAIR"],
    "REANCHOR_LOOP": ["REANCHOR", "VERIFY_CONSTRAINTS", "VERIFY"],
    "ARTIFACT_PLAN": ["WRITE_PLAN", "READ_ARTIFACT", "PLAN"],
    "ARTIFACT_PATCH": ["WRITE_PATCH", "READ_ARTIFACT", "REPAIR"],
    "ARTIFACT_SUMMARY": ["WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE"],
}

TASKS = [
    {
        "name": "evidence_case",
        "req": ["OBSERVE", "RETRIEVE", "VERIFY", "SUMMARIZE"],
        "needs": {"summary": 2},
        "tool_noise": 0.18,
        "hazard_if_no_verify": True,
    },
    {
        "name": "memory_patch",
        "req": ["CACHE", "RETRIEVE", "COMMIT", "VERIFY", "REPAIR"],
        "needs": {"patch": 2},
        "tool_noise": 0.10,
        "hazard_if_no_verify": False,
    },
    {
        "name": "incident_recover",
        "req": ["SWITCH_CONTEXT", "VERIFY", "ROLLBACK", "REPAIR", "SUMMARIZE"],
        "needs": {"plan": 1, "patch": 2},
        "tool_noise": 0.12,
        "hazard_if_no_verify": True,
    },
    {
        "name": "plan_and_fix",
        "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
        "needs": {"plan": 2},
        "tool_noise": 0.08,
        "hazard_if_no_verify": False,
    },
    {
        "name": "stale_artifact_audit",
        "req": ["OBSERVE", "VERIFY", "READ_ARTIFACT", "REVISE_ARTIFACT", "SUMMARIZE"],
        "needs": {"summary": 1},
        "tool_noise": 0.06,
        "hazard_if_no_verify": True,
    },
    {
        "name": "cross_team_handoff",
        "req": ["WRITE_SUMMARY", "LINK_ARTIFACT", "READ_ARTIFACT", "PLAN", "VERIFY"],
        "needs": {"summary": 1, "plan": 1},
        "tool_noise": 0.09,
        "hazard_if_no_verify": False,
    },
]

ARTIFACT_TYPES = {
    "WRITE_NOTE": "note",
    "WRITE_PLAN": "plan",
    "WRITE_PATCH": "patch",
    "WRITE_SUMMARY": "summary",
}
CHARTS = ["ANALYSIS", "MAINTENANCE", "RECOVERY", "PLANNING"]

SEMANTIC_FAMILIES = {
    "DISCOVERY": [["OBSERVE", "RETRIEVE", "VERIFY"]],
    "SAFE_REPORT": [["VERIFY", "REPAIR", "SUMMARIZE"]],
    "RECOVERY": [["VERIFY", "ROLLBACK", "REPAIR"]],
    "MEMORY": [["CACHE", "RETRIEVE", "COMMIT"]],
    "PATCHING": [["PLAN", "REPAIR", "VERIFY"]],
    "DIAGNOSIS": [["SWITCH_CONTEXT", "VERIFY", "REPAIR"]],
    "ARTIFACT_PLAN": [["WRITE_PLAN", "READ_ARTIFACT", "PLAN"]],
    "ARTIFACT_PATCH": [["WRITE_PATCH", "READ_ARTIFACT", "REPAIR"]],
    "ARTIFACT_SUMMARY": [["WRITE_SUMMARY", "READ_ARTIFACT", "SUMMARIZE"]],
}

REGIME_CENTER = {"explore": 0.92, "balanced": 0.60, "exploit": 0.32}
COOLING_SUCCESS_THRESHOLD = 0.82
COOLING_STABILITY_THRESHOLD = 0.72
EXPLOIT_ENTRY_THRESHOLD = 0.35
EXPLOIT_EXIT_THRESHOLD = 0.34
EXPLOIT_HOLD = 5
MOMENTUM_WINDOW = 7
EXPLOIT_RETENTION_BONUS = 0.08

__all__ = [
    "ALL_TOKENS",
    "ARTIFACT_OPS",
    "ARTIFACT_TYPES",
    "BASE_MACROS",
    "CHARTS",
    "CONTROL_TOKENS",
    "COOLING_STABILITY_THRESHOLD",
    "COOLING_SUCCESS_THRESHOLD",
    "EXPLOIT_ENTRY_THRESHOLD",
    "EXPLOIT_EXIT_THRESHOLD",
    "EXPLOIT_HOLD",
    "EXPLOIT_RETENTION_BONUS",
    "MOMENTUM_WINDOW",
    "PRIMITIVES",
    "REGIME_CENTER",
    "SEMANTIC_FAMILIES",
    "TASKS",
]
