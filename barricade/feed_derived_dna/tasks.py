from __future__ import annotations

import json
import re
from collections import Counter

from .analysis import semantic_priority_tokens_from_task_pool
from .constants import CHARTS, TASKS


__all__ = [
    "build_patch_skeleton",
    "derive_feed_dna_prior",
    "derive_task_ecology",
    "seed_population_from_prior",
]


def _label_from_text(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return "_".join(tokens[:6]) or "feed_task"


def _collect_text_fragments(value, path="root", fragments=None, max_fragments=12):
    if fragments is None:
        fragments = []
    if len(fragments) >= max_fragments:
        return fragments
    if isinstance(value, str):
        text = value.strip()
        if text:
            fragments.append((path, text))
    elif isinstance(value, dict):
        for key, child in value.items():
            if len(fragments) >= max_fragments:
                break
            _collect_text_fragments(child, f"{path}.{key}", fragments, max_fragments)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if len(fragments) >= max_fragments:
                break
            _collect_text_fragments(child, f"{path}[{index}]", fragments, max_fragments)
    return fragments


def _cue_count(text, keywords):
    return sum(
        1
        for keyword in keywords
        if re.search(
            rf"(?<!\w){re.escape(keyword)}(?!\w)", str(text), flags=re.IGNORECASE
        )
    )


def _has_keyword(text, keywords):
    return any(
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", str(text), flags=re.IGNORECASE)
        for keyword in keywords
    )


def _build_task_from_text(name, text, focus=None):
    text.lower()
    plan_cues = _cue_count(
        text,
        (
            "plan",
            "planning",
            "design",
            "architect",
            "structure",
            "contract",
            "schema",
            "deterministic",
            "foundation",
        ),
    )
    patch_cues = _cue_count(
        text,
        (
            "patch",
            "patching",
            "implement",
            "code",
            "class",
            "node",
            "state machine",
            "async",
            "rpc",
            "leader",
            "candidate",
            "follower",
            "fix",
        ),
    )
    summary_cues = _cue_count(
        text,
        (
            "summary",
            "summarize",
            "summarizing",
            "review",
            "memo",
            "report",
            "analysis",
            "explain",
        ),
    )
    recovery_cues = _cue_count(
        text,
        (
            "recover",
            "recovery",
            "rollback",
            "failure",
            "error",
            "timeout",
            "retry",
            "network",
            "rpc",
        ),
    )

    if focus is None:
        if recovery_cues >= 2 and recovery_cues >= patch_cues:
            focus = "recovery"
        elif patch_cues >= plan_cues and patch_cues >= summary_cues:
            focus = "patching"
        elif summary_cues > plan_cues:
            focus = "summarizing"
        else:
            focus = "planning"

    if focus == "patching":
        req = ["PLAN", "REPAIR", "VERIFY_CODE", "COMMIT"]
        needs = {"patch": 2}
        tool_noise = (
            0.08
            if _has_keyword(
                text, ("rpc", "async", "node", "leader", "candidate", "follower")
            )
            else 0.10
        )
        hazard = False
    elif focus == "summarizing":
        req = ["OBSERVE", "RETRIEVE", "VERIFY_DATA", "SUMMARIZE"]
        needs = {"summary": 2}
        tool_noise = 0.14
        hazard = True
    elif focus == "recovery":
        req = [
            "SWITCH_CONTEXT",
            "VERIFY_CONSTRAINTS",
            "ROLLBACK",
            "REPAIR",
            "SUMMARIZE",
        ]
        needs = {"plan": 1, "patch": 1}
        tool_noise = 0.11
        hazard = True
    else:
        req = ["PLAN", "REPAIR", "VERIFY_CONSTRAINTS", "COMMIT"]
        needs = {"plan": 2}
        tool_noise = 0.09
        hazard = False

    if _has_keyword(text, ("requestvote", "appendentries", "raft")):
        needs["plan"] = max(needs.get("plan", 0), 1)
        needs["patch"] = max(needs.get("patch", 0), 1)
        hazard = True
    if (
        _has_keyword(text, ("leader", "candidate", "follower", "node"))
        and focus != "summarizing"
    ):
        needs["plan"] = max(needs.get("plan", 0), 1)
    if _has_keyword(text, ("verify", "validation", "timeout", "rpc")):
        hazard = True
    if _has_keyword(text, ("json", "schema", "output", "data")):
        req = ["VERIFY_DATA" if token == "VERIFY" else token for token in req]
    if focus == "recovery" and _has_keyword(
        text, ("invariant", "constraint", "must remain", "must preserve")
    ):
        req = ["VERIFY_CONSTRAINTS" if token == "VERIFY" else token for token in req]
    if _has_keyword(
        text, ("code", "patch", "implement", "refactor", "class", "function")
    ):
        req = ["VERIFY_CODE" if token == "VERIFY" else token for token in req]
    if focus == "recovery" and _has_keyword(
        text, ("env", "runtime", "dependency", "workspace")
    ):
        req = ["VERIFY_ENV" if token == "VERIFY" else token for token in req]

    return {
        "name": name,
        "req": req,
        "needs": needs,
        "tool_noise": tool_noise,
        "hazard_if_no_verify": hazard,
        "focus": focus,
        "source_text": text[:240],
    }


def derive_task_ecology(feed):
    fragments = _collect_text_fragments(feed)
    source_text = "\n".join(text for _, text in fragments).strip()
    if not source_text:
        source_text = json.dumps(feed, indent=2, sort_keys=True)

    tasks = []
    seen = set()

    def add_task(task):
        signature = (
            task["name"],
            tuple(task["req"]),
            tuple(sorted(task["needs"].items())),
            task["focus"],
        )
        if signature not in seen:
            seen.add(signature)
            tasks.append(task)

    add_task(_build_task_from_text(_label_from_text(source_text), source_text))

    focus_map = [
        (
            "planning",
            (
                "plan",
                "planning",
                "design",
                "architect",
                "structure",
                "contract",
                "deterministic",
            ),
        ),
        (
            "patching",
            (
                "patch",
                "patching",
                "implement",
                "code",
                "class",
                "node",
                "state machine",
                "async",
                "rpc",
                "leader",
                "candidate",
                "follower",
            ),
        ),
        (
            "summarizing",
            (
                "summary",
                "summarize",
                "summarizing",
                "review",
                "memo",
                "analysis",
                "report",
            ),
        ),
        (
            "recovery",
            (
                "recover",
                "recovery",
                "rollback",
                "failure",
                "error",
                "timeout",
                "retry",
                "network",
                "validate",
            ),
        ),
    ]

    for focus, keywords in focus_map:
        if _has_keyword(source_text, keywords):
            add_task(
                _build_task_from_text(
                    f"{_label_from_text(source_text)}_{focus}", source_text, focus=focus
                )
            )

    for path, text in fragments:
        focus = None
        if _has_keyword(
            text, ("summary", "summarize", "summarizing", "review", "memo", "analysis")
        ):
            focus = "summarizing"
        elif _has_keyword(
            text,
            (
                "patch",
                "patching",
                "implement",
                "code",
                "class",
                "node",
                "rpc",
                "leader",
                "candidate",
                "follower",
            ),
        ):
            focus = "patching"
        elif _has_keyword(
            text, ("rollback", "recover", "recovery", "failure", "timeout", "retry")
        ):
            focus = "recovery"
        elif _has_keyword(
            text, ("plan", "planning", "design", "architect", "contract", "structure")
        ):
            focus = "planning"
        if focus is not None:
            add_task(_build_task_from_text(_label_from_text(path), text, focus=focus))

    return tasks[:6] if tasks else TASKS


def derive_feed_dna_prior(task_pool):
    focus_counts = Counter(task.get("focus", "planning") for task in task_pool)
    need_counts = Counter()
    hazard_count = 0
    for task in task_pool:
        need_counts.update(task.get("needs", {}))
        if task.get("hazard_if_no_verify"):
            hazard_count += 1

    prior = ["OBSERVE", "LM1"]

    if focus_counts.get("patching", 0) >= max(
        focus_counts.get("planning", 0), focus_counts.get("summarizing", 0)
    ):
        prior.extend(["WRITE_PLAN", "WRITE_PATCH", "VERIFY", "SUMMARIZE"])
    elif focus_counts.get("summarizing", 0) > focus_counts.get("planning", 0):
        prior.extend(["WRITE_SUMMARY", "VERIFY", "SUMMARIZE", "WRITE_PLAN"])
    elif focus_counts.get("recovery", 0) > 0:
        prior.extend(["WRITE_PLAN", "ROLLBACK", "REPAIR", "VERIFY", "SUMMARIZE"])
    else:
        prior.extend(["WRITE_PLAN", "VERIFY", "SUMMARIZE"])

    semantic_tokens = semantic_priority_tokens_from_task_pool(task_pool, limit=5)
    insert_at = min(2, len(prior))
    for token in semantic_tokens:
        if token not in prior:
            prior.insert(insert_at, token)
            insert_at += 1

    if need_counts.get("patch", 0) > need_counts.get("summary", 0):
        prior.insert(2, "WRITE_PATCH")
    if (
        focus_counts.get("summarizing", 0) > focus_counts.get("planning", 0)
        and "WRITE_SUMMARY" not in prior
    ):
        prior.insert(-1, "WRITE_SUMMARY")
    if need_counts.get("plan", 0) > 1 and "PLAN" not in prior:
        prior.insert(2, "PLAN")
    if hazard_count > 0 and "VERIFY" not in prior:
        prior.append("VERIFY")

    deduped = []
    for token in prior:
        if token not in deduped:
            deduped.append(token)
    return deduped[:12]


def seed_population_from_prior(
    rng, population, learned_macros, seed_lineage, prior_dna
):
    seeded = []
    if not prior_dna:
        return seeded

    base = prior_dna[:]
    variants = [
        base,
        base + (["WRITE_PATCH"] if "WRITE_PATCH" not in base else ["VERIFY"]),
        base + (["WRITE_SUMMARY"] if "WRITE_SUMMARY" not in base else ["PLAN"]),
        base + (["REPAIR", "VERIFY"] if "REPAIR" not in base else ["SUMMARIZE"]),
    ]
    from .models import Individual

    for index in range(min(len(variants), max(4, population // 8))):
        dna = variants[index][:36]
        seeded.append(Individual(dna, rng.choice(CHARTS), f"{seed_lineage}_{index}"))
    return seeded


def build_patch_skeleton(task_pool, feed_prior_dna):
    focus_counts = Counter(task.get("focus", "planning") for task in task_pool)
    primary_focus = "planning"
    if focus_counts:
        primary_focus = max(focus_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    priority_tokens = []
    for token in feed_prior_dna:
        if (
            token
            in (
                "WRITE_PATCH",
                "PLAN",
                "REPAIR",
                "VERIFY",
                "ROLLBACK",
                "WRITE_PLAN",
                "WRITE_SUMMARY",
                "SUMMARIZE",
            )
            and token not in priority_tokens
        ):
            priority_tokens.append(token)

    if not priority_tokens:
        priority_tokens = ["WRITE_PATCH", "PLAN", "REPAIR", "VERIFY"]
    if "WRITE_PATCH" not in priority_tokens:
        priority_tokens.insert(0, "WRITE_PATCH")
    if "PLAN" not in priority_tokens:
        priority_tokens.append("PLAN")
    if "VERIFY" not in priority_tokens:
        priority_tokens.append("VERIFY")

    transfer_target = "miniops_crisis_ecology_v3_12_contract_bridge.py"
    if focus_counts.get("summarizing", 0) > focus_counts.get(
        "patching", 0
    ) and focus_counts.get("planning", 0) >= focus_counts.get("patching", 0):
        transfer_target = "explore/miniops_v3_14.py"

    return {
        "intent": "patch-oriented code skeleton",
        "primary_focus": primary_focus,
        "focus_counts": dict(focus_counts),
        "source_module": "explore/miniops_v3_14.py",
        "transfer_target": transfer_target,
        "task_labels": [task.get("name", "task") for task in task_pool[:6]],
        "touchpoints": [
            {
                "symbol": "derive_task_ecology",
                "purpose": "extract a bounded ecology from the feed",
            },
            {
                "symbol": "derive_feed_dna_prior",
                "purpose": "compress the ecology into a soft DNA prior",
            },
            {
                "symbol": "seed_population_from_prior",
                "purpose": "bootstrap a prior-aligned population",
            },
            {
                "symbol": "run_v311",
                "purpose": "thread the feed-derived prior through the evolutionary loop",
            },
        ],
        "token_outline": priority_tokens[:8],
        "patch_outline": [
            "Normalize the input feed into a bounded ecology.",
            "Convert the ecology into a patch-biased prior.",
            "Seed the first population from that prior.",
            "Return a machine-readable scaffold for downstream patch transfer.",
        ],
        "code_skeleton": [
            "def build_patch_skeleton(task_pool, feed_prior_dna):",
            '    focus_counts = Counter(task.get("focus", "planning") for task in task_pool)',
            "    priority_tokens = [token for token in feed_prior_dna if token in ALLOWED_PATCH_TOKENS]",
            '    return {"transfer_target": "miniops_crisis_ecology_v3_12_contract_bridge.py", ...}',
        ],
    }
