from __future__ import annotations

import json
import inspect
import uuid
import shlex
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ._protocol import (
    _session_payload,
    _market_entry_summary,
    _jsonable,
    _short_preview,
    _dna_summary,
)
from ._parsing import (
    _parse_patch_updates,
    _verification_spec_for_dispatch_updates,
    _artifacts_to_dispatch_updates,
    _verification_command_for_dispatch_updates,
    _verification_spec_from_payload,
)
from ._scoring import (
    STEP_GUIDANCE,
    _token_prefix,
    _artifact_score,
    _artifact_price,
    _artifact_kind,
)
from ..feed_derived_dna.analysis import (
    flatten_trace,
    mine_macros_from_elites,
    motif_key,
)
from ..feed_derived_dna.constants import BASE_MACROS
from ..feed_derived_dna.persistence import (
    load_macro_library,
    load_motif_cache,
    resolve_state_root,
    save_discoverables,
    save_execution_feedback,
    save_run_summary,
)
from .._shared import as_mapping, as_string_list, as_macro_library
from .._shared import run_command_with_timeout
from .._validation import validate_workspace_root
from ..dispatch import verification_passed as dispatch_verification_passed
from ..dispatch import _verification_semantic_failures


@dataclass
class Artifact:
    artifact_id: str
    token: str
    kind: str
    content: str
    creator: str
    epoch: int
    price: float
    score: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class ExecutionSession:
    session_id: str
    task_text: str
    dna: list[str]
    flattened: list[str]
    current_step: int = 0
    revision: int = 0
    market: dict[str, Artifact] = field(default_factory=dict)
    macro_lib: dict[str, list[str]] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    state_root: Path | None = None
    workspace_root: Path | None = None
    execution_traces: list[list[str]] = field(default_factory=list)
    current_execution_trace: list[str] = field(default_factory=list, repr=False)
    current_execution_trace_successful: bool = field(default=False, repr=False)
    patch_updates: dict[str, str] = field(default_factory=dict)
    verification_command: str | list[str] | None = None
    verification_spec: dict[str, Any] = field(default_factory=dict)
    verification_result: dict[str, Any] = field(default_factory=dict)
    completed: bool = False

    def current_token(self) -> str | None:
        if self.current_step >= len(self.flattened):
            return None
        return self.flattened[self.current_step]

    def remaining_tokens(self) -> list[str]:
        return self.flattened[self.current_step :]

    def next_index_for(self, token: str) -> int:
        for index in range(self.current_step + 1, len(self.flattened)):
            if self.flattened[index] == token:
                return index
        return len(self.flattened)

    def advance_to(self, index: int) -> None:
        self.current_step = max(0, min(index, len(self.flattened)))

    def market_snapshot(self, limit: int | None = None) -> list[dict[str, Any]]:
        entries = sorted(
            self.market.values(),
            key=lambda artifact: (artifact.epoch, artifact.artifact_id),
        )
        if limit is not None:
            entries = entries[: max(0, limit)]
        return [_market_entry_summary(artifact) for artifact in entries]

    def record_execution_trace_artifact(self, artifact: Artifact) -> None:
        if artifact.token == "VERIFY" and artifact.status == "failed":
            self.current_execution_trace.clear()
            self.current_execution_trace_successful = False
            return

        self.current_execution_trace.append(artifact.token)
        if artifact.token == "VERIFY" and artifact.status == "passed":
            self.current_execution_trace_successful = True

    def successful_execution_traces_snapshot(self) -> list[list[str]]:
        if self.current_execution_trace and self.current_execution_trace_successful:
            return [list(self.current_execution_trace)]
        return []


def _mine_macros_for_session(
    execution_traces_for_mining: list[list[str]],
    macro_lib: dict[str, list[str]],
) -> dict[str, list[str]]:
    mine_signature = inspect.signature(mine_macros_from_elites)
    supports_macro_lib = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == "macro_lib"
        for name, parameter in mine_signature.parameters.items()
    )
    if supports_macro_lib:
        return mine_macros_from_elites(
            execution_traces_for_mining,
            max_macros=8,
            macro_lib=macro_lib,
        )
    return mine_macros_from_elites(execution_traces_for_mining, max_macros=8)


def _resolve_execution_dna(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    synthesis = as_mapping(result.get("synthesis"))
    if not synthesis:
        raise ValueError("solve_problem output must include synthesis")

    execution_seed = as_mapping(result.get("execution_seed"))
    candidate_dna = as_string_list(execution_seed.get("dna"))
    if not candidate_dna:
        candidate_dna = as_string_list(
            as_mapping(result.get("dna")).get("feed_prior_dna")
        )
    if not candidate_dna:
        candidate_dna = as_string_list(synthesis.get("feed_prior_dna"))
    if not candidate_dna:
        raise ValueError("solve_problem output must include synthesis.feed_prior_dna")

    learned_macros = as_macro_library(synthesis.get("learned_macros"))

    execution_dna: list[str] = []
    omitted_macros = as_string_list(execution_seed.get("omitted_macros"))
    from .._shared import is_unresolved_macro_token

    for token in candidate_dna:
        if is_unresolved_macro_token(token, learned_macros):
            if token not in omitted_macros:
                omitted_macros.append(token)
            continue
        execution_dna.append(token)

    if not execution_dna:
        raise ValueError("execution seed does not contain actionable tokens")

    return execution_dna, omitted_macros


def _completion_summary(
    session: ExecutionSession,
    artifacts: list[Artifact],
    dispatch_updates: dict[str, str],
    learned_macros: dict[str, list[str]],
    successful_execution_traces: list[list[str]],
) -> dict[str, Any]:
    artifact_kind_counts = Counter(artifact.kind for artifact in artifacts)
    artifact_token_counts = Counter(artifact.token for artifact in artifacts)
    specialization_counts = Counter(
        str(artifact.metadata.get("specialization", "generalist"))
        for artifact in artifacts
    )
    artifact_status_counts = Counter(artifact.status for artifact in artifacts)
    verification_artifacts = [
        artifact for artifact in artifacts if artifact.token == "VERIFY"
    ]
    verification_pass_count = sum(
        1 for artifact in verification_artifacts if artifact.status == "passed"
    )
    verification_pass_rate = (
        round(verification_pass_count / len(verification_artifacts), 3)
        if verification_artifacts
        else 0.0
    )
    top_artifact = (
        max(
            artifacts,
            key=lambda artifact: (artifact.score, artifact.price, artifact.artifact_id),
        )
        if artifacts
        else None
    )

    return {
        "status": "completed",
        "artifact_count": len(artifacts),
        "market_count": len(artifacts),
        "patch_update_count": len(dispatch_updates),
        "successful_trace_count": len(successful_execution_traces),
        "learned_macro_count": len(learned_macros),
        "verification_passed": session.verification_result.get("passed", False),
        "verification_artifact_count": len(verification_artifacts),
        "verification_pass_count": verification_pass_count,
        "verification_pass_rate": verification_pass_rate,
        "market_total_price": round(sum(artifact.price for artifact in artifacts), 3),
        "market_total_score": round(sum(artifact.score for artifact in artifacts), 3),
        "artifact_kind_counts": dict(artifact_kind_counts),
        "artifact_token_counts": dict(artifact_token_counts),
        "artifact_status_counts": dict(artifact_status_counts),
        "specialization_counts": dict(specialization_counts),
        "top_artifact": _market_entry_summary(top_artifact)
        if top_artifact is not None
        else {},
    }


class ExecutionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, ExecutionSession] = {}
        self._lock = RLock()

    def _load_macro_library(
        self,
        state_root: Path | None,
        learned_macros: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        macro_lib = dict(BASE_MACROS)
        if state_root is not None:
            macro_lib.update(load_macro_library(state_root))
        if learned_macros:
            macro_lib.update(
                {
                    str(name): [str(token) for token in sequence]
                    for name, sequence in learned_macros.items()
                    if isinstance(sequence, list)
                }
            )
        return macro_lib

    def _session_payload(self, session: ExecutionSession) -> dict[str, Any]:
        delta = {
            "dna_summary": _dna_summary(session.flattened),
        }
        return _session_payload(session, "started", delta, include_dna=True)

    def _instruction_for(
        self,
        session: ExecutionSession,
        *,
        token: str | None = None,
        artifact_id_hint: str | None = None,
    ) -> dict[str, Any]:
        current_token = token or session.current_token()
        if current_token is None:
            return {
                "tool_hint": "complete_execution",
                "instruction": "The execution trace is complete. Call complete_execution(session_id) to finalize the run.",
                "next_token": None,
                "artifact_kind": "final",
            }

        guidance = STEP_GUIDANCE.get(current_token, {})
        instruction = guidance.get(
            "instruction", f"Execute the {current_token} step and return the result."
        )

        if current_token == "READ_ARTIFACT" and artifact_id_hint:
            instruction = f"Inspect artifact {artifact_id_hint} before continuing."

        if current_token == "WRITE_PATCH":
            instruction += " Include file paths and content in a dispatch-plan friendly format so the completion step can build updates."

        if current_token == "VERIFY":
            instruction += " Return a pass/fail result from the verification command."

        return {
            "tool_hint": guidance.get("tool_hint", "submit_step"),
            "instruction": f"Step {session.current_step + 1}/{len(session.flattened)}: {current_token}. {instruction}",
            "next_token": current_token,
            "artifact_kind": guidance.get("kind", current_token.lower()),
            "artifact_id_hint": artifact_id_hint
            or session.context.get("last_artifact_id", ""),
        }

    def _register_artifact(
        self,
        session: ExecutionSession,
        token: str,
        content: str,
        *,
        creator: str = "host",
        verification: dict[str, Any] | None = None,
    ) -> Artifact:
        score, metadata = _artifact_score(token, content, verification=verification)
        artifact_id = f"{_token_prefix(token)}_{session.current_step + 1:02d}"
        artifact_id = (
            artifact_id
            if artifact_id not in session.market
            else f"{artifact_id}_{uuid.uuid4().hex[:4]}"
        )
        price = _artifact_price(score, content, token)
        metadata["verification"] = verification or {}
        artifact = Artifact(
            artifact_id=artifact_id,
            token=token,
            kind=_artifact_kind(token),
            content=content,
            creator=creator,
            epoch=session.current_step + 1,
            price=price,
            score=round(score, 3),
            status="submitted"
            if token != "VERIFY"
            else (
                "passed" if verification and verification.get("passed") else "failed"
            ),
            metadata=_jsonable(metadata),
        )
        session.market[artifact_id] = artifact
        session.execution_traces.append(metadata["trace"])
        session.record_execution_trace_artifact(artifact)
        session.context["last_artifact_id"] = artifact_id
        session.context["last_token"] = token
        session.context["last_price"] = price
        session.context.setdefault("artifact_history", []).append(
            {
                "artifact_id": artifact_id,
                "token": token,
                "kind": artifact.kind,
                "price": price,
                "score": artifact.score,
            }
        )
        if token == "OBSERVE":
            session.context["observations"] = content
        elif token in {"PLAN", "WRITE_PLAN"}:
            session.context["plan"] = content
        elif token in {"WRITE_PATCH", "REPAIR", "ROLLBACK"}:
            session.context.setdefault("patches", []).append(
                {"artifact_id": artifact_id, "content": content}
            )
            parsed_updates = _parse_patch_updates(content)
            if parsed_updates:
                session.patch_updates.update(parsed_updates)
                artifact.metadata["parsed_updates"] = parsed_updates
                verification_spec = _verification_spec_for_dispatch_updates(
                    parsed_updates,
                    {artifact_id: sorted(parsed_updates)},
                    artifact_id=artifact_id,
                )
                if verification_spec:
                    artifact.metadata["verification_spec"] = verification_spec
                    session.context["verification_spec"] = verification_spec
                    session.context["verification_target_artifact_id"] = artifact_id
        elif token == "SUMMARIZE":
            session.context["summary"] = content
        elif token == "VERIFY":
            session.context["verification"] = verification or {}
            if verification and isinstance(verification.get("spec"), dict):
                session.context["verification_spec"] = verification["spec"]
            if verification and verification.get("artifact_id"):
                session.context["verification_target_artifact_id"] = verification[
                    "artifact_id"
                ]
            session.verification_result = verification or {}
        elif token == "READ_ARTIFACT":
            session.context["artifact_read"] = content
        return artifact

    def _load_session(self, session_id: str) -> ExecutionSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def _ensure_session_active(self, session: ExecutionSession) -> None:
        if session.completed:
            raise ValueError("execution session is already complete")

    def begin_execution(
        self,
        synthesis_result: dict[str, Any] | str,
        state_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if isinstance(synthesis_result, str):
            result = json.loads(synthesis_result)
        else:
            result = dict(synthesis_result)

        state_root = resolve_state_root(state_dir or result.get("state_dir"))
        workspace_root = validate_workspace_root(result.get("workspace_root"))
        synthesis = as_mapping(result.get("synthesis"))
        decision_policy = as_mapping(result.get("decision_policy"))
        if not decision_policy:
            decision_policy = as_mapping(synthesis.get("decision_policy"))
        policy_mode = str(decision_policy.get("mode", "act") or "act")
        if policy_mode != "act":
            raise ValueError(
                str(
                    decision_policy.get(
                        "next_instruction",
                        "Workflow policy does not allow execution for this task.",
                    )
                )
            )
        learned_macros = as_macro_library(synthesis.get("learned_macros"))
        macro_lib = self._load_macro_library(state_root, learned_macros)

        dna, omitted_macros = _resolve_execution_dna(result)
        flattened = flatten_trace(dna, macro_lib)
        execution_workspace_root = workspace_root
        session_id = f"exec_{uuid.uuid4().hex[:12]}"
        session = ExecutionSession(
            session_id=session_id,
            task_text=str(
                result.get("problem_text", result.get("intake", {}).get("raw_task", ""))
            ),
            dna=dna,
            flattened=flattened,
            current_step=0,
            revision=1,
            macro_lib=macro_lib,
            context={
                "problem_text": result.get("problem_text", ""),
                "intake": result.get("intake", {}),
                "synthesis": {
                    "summary": synthesis.get("summary", {}),
                    "patch_skeleton": synthesis.get("patch_skeleton", {}),
                    "task_pool": synthesis.get("task_pool", []),
                    "task_shape_signature": synthesis.get("task_shape_signature", ""),
                    "outcome_memory": synthesis.get("outcome_memory", {}),
                    "execution_feedback": synthesis.get("execution_feedback", {}),
                    "decision_policy": synthesis.get("decision_policy", {}),
                },
                "route_hint": result.get("intake", {}).get("route_hint", ""),
                "outcome_memory": synthesis.get("outcome_memory", {}),
                "execution_feedback": synthesis.get("execution_feedback", {}),
                "decision_policy": synthesis.get("decision_policy", {}),
                "execution_seed": {
                    "omitted_macros": omitted_macros,
                    "dna_summary": _dna_summary(dna),
                },
            },
            state_root=state_root,
            workspace_root=execution_workspace_root,
        )
        if session.workspace_root is not None:
            session.workspace_root.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._sessions[session_id] = session

        instruction_payload = self._instruction_for(session)
        delta = {
            "dna_summary": _dna_summary(session.flattened),
            "omitted_macros": omitted_macros,
            "route_hint": result.get("intake", {}).get("route_hint", ""),
        }
        payload = _session_payload(
            session,
            "started",
            delta,
            instruction=instruction_payload.get("instruction"),
            tool_hint=instruction_payload.get("tool_hint"),
            include_dna=True,
        )
        if omitted_macros:
            payload["omitted_macros"] = omitted_macros
        return payload

    def submit_step(self, session_id: str, content: str) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            self._ensure_session_active(session)
            current_token = session.current_token()
            if current_token is None:
                raise ValueError("execution session is already complete")
            if current_token == "VERIFY":
                raise ValueError(
                    "current step is VERIFY; call verify_step(session_id, command) instead"
                )
            if current_token == "READ_ARTIFACT":
                raise ValueError(
                    "current step is READ_ARTIFACT; call read_artifact(session_id, artifact_id) instead"
                )

            artifact = self._register_artifact(session, current_token, content)
            session.advance_to(session.current_step + 1)
            next_token = session.current_token()
            session.revision += 1

        next_instruction = self._instruction_for(
            session, artifact_id_hint=artifact.artifact_id
        )
        payload = _session_payload(
            session,
            "step_recorded",
            {
                "artifact_id": artifact.artifact_id,
                "next_token": next_token,
                "completed": next_token is None,
            },
            tool_hint="complete_execution"
            if next_token is None
            else next_instruction.get("tool_hint", "submit_step"),
            instruction=next_instruction.get("instruction"),
        )
        payload["artifact"] = _market_entry_summary(artifact)
        payload["market_entry"] = _market_entry_summary(artifact)
        payload["market_count"] = len(session.market)
        payload["next_token"] = next_token
        payload["next_instruction"] = next_instruction
        payload["completed"] = next_token is None
        if next_token is None:
            payload["instruction"] = (
                "Execution trace is complete. Call complete_execution(session_id) to finalize the run."
            )
        return payload

    def read_artifact(self, session_id: str, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            self._ensure_session_active(session)
            artifact = session.market.get(artifact_id)
            if artifact is None:
                raise KeyError(f"unknown artifact_id: {artifact_id}")
            current_token = session.current_token()
            consume = current_token == "READ_ARTIFACT"
            if consume:
                session.advance_to(session.current_step + 1)
                session.revision += 1

        next_instruction = self._instruction_for(session)
        payload = _session_payload(
            session,
            "artifact_read",
            {
                "artifact_id": artifact_id,
                "consumed_step": consume,
            },
            tool_hint=next_instruction.get("tool_hint", "view_market"),
            instruction=next_instruction.get("instruction"),
        )
        payload["artifact"] = artifact.to_dict()
        payload["market_entry"] = _market_entry_summary(artifact)
        payload["next_token"] = session.current_token()
        payload["next_instruction"] = next_instruction
        payload["dna_summary"] = _dna_summary(session.flattened)
        return payload

    def view_market(self, session_id: str, limit: int = 8) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            market = session.market_snapshot(limit=limit)

        next_instruction = self._instruction_for(session)
        payload = _session_payload(
            session,
            "market_view",
            {
                "market_count": len(session.market),
            },
            tool_hint=next_instruction.get("tool_hint", "submit_step"),
            instruction=next_instruction.get("instruction"),
        )
        payload["market_count"] = len(session.market)
        payload["market"] = market
        payload["next_token"] = session.current_token()
        payload["dna_summary"] = _dna_summary(session.flattened)
        return payload

    def verify_step(self, session_id: str, command: str) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            self._ensure_session_active(session)
            current_token = session.current_token()
            if current_token != "VERIFY":
                raise ValueError("current step is not VERIFY")

            fallback_spec = as_mapping(session.context.get("verification_spec"))
            try:
                parsed_payload: Any = json.loads(command)
            except json.JSONDecodeError:
                parsed_payload = shlex.split(command)
            else:
                if not isinstance(parsed_payload, (list, dict)):
                    parsed_payload = shlex.split(str(parsed_payload))
            parsed_command, verification_spec = _verification_spec_from_payload(
                parsed_payload, fallback_spec
            )
            if not isinstance(parsed_command, list):
                raise ValueError("verify command must resolve to a list of arguments")
            if not verification_spec:
                if session.context.get("verification_target_artifact_id"):
                    verification_spec["artifact_id"] = str(
                        session.context["verification_target_artifact_id"]
                    )
                if session.patch_updates:
                    verification_spec.update(
                        _verification_spec_for_dispatch_updates(
                            session.patch_updates,
                            {},
                            artifact_id=verification_spec.get("artifact_id")
                            or session.context.get("verification_target_artifact_id")
                            or "",
                        )
                    )
            if verification_spec:
                session.context["verification_spec"] = verification_spec
            run_cwd = (
                session.workspace_root
                if session.workspace_root and session.workspace_root.exists()
                else Path.cwd()
            )

            completed = run_command_with_timeout(parsed_command, run_cwd)

            verification: dict[str, Any] = {
                "command": parsed_command,
                "returncode": completed.returncode,
                "stdout": _short_preview(completed.stdout, 240),
                "stderr": _short_preview(completed.stderr, 240),
            }
            verification["spec"] = verification_spec
            verification["artifact_id"] = (
                verification_spec.get("artifact_id")
                or session.context.get("verification_target_artifact_id")
                or session.context.get("last_artifact_id")
                or ""
            )
            verification["passed"], failure_signatures = dispatch_verification_passed(
                completed, verification_spec, cwd=run_cwd
            )
            if failure_signatures:
                verification["failure_signatures"] = failure_signatures

            semantic_failures = _verification_semantic_failures(
                completed, verification_spec, cwd=run_cwd
            )

            from .._verification_parser import parse_verification_output

            structured_report = parse_verification_output(
                completed,
                verification_spec,
                cwd=run_cwd,
                semantic_failures=semantic_failures,
            )
            verification["structured_report"] = structured_report.to_dict()

            artifact = self._register_artifact(
                session,
                current_token,
                f"{completed.stdout}\n{completed.stderr}".strip(),
                verification=verification,
            )
            session.verification_command = parsed_command
            session.verification_spec = verification_spec
            session.context["verification_spec"] = verification_spec
            session.verification_result = verification

            if verification["passed"]:
                session.advance_to(session.current_step + 1)
            else:
                next_recovery = None
                for index in range(session.current_step + 1, len(session.flattened)):
                    if session.flattened[index] in {"REPAIR", "ROLLBACK"}:
                        next_recovery = index
                        break
                if next_recovery is not None:
                    session.advance_to(next_recovery)
            session.revision += 1

            next_token = session.current_token()
            next_instruction = self._instruction_for(session)

            if not verification["passed"] and next_token in {"REPAIR", "ROLLBACK"}:
                hints = structured_report._actionable_hints()
                hint_text = (
                    "\n".join(f"- {h}" for h in hints)
                    if hints
                    else "No specific hints available."
                )
                next_instruction["instruction"] = (
                    f"Verification failed. The DNA still has {next_token}; fix the errors below before advancing.\n"
                    f"Summary: {structured_report.summary}\n"
                    f"Actionable hints:\n{hint_text}"
                )
            elif not verification["passed"] and next_token == "VERIFY":
                next_instruction["instruction"] = (
                    "Verification failed and the DNA does not contain REPAIR or ROLLBACK. "
                    "The session remains on VERIFY until the plan is updated."
                )

            payload = _session_payload(
                session,
                "verification_passed"
                if verification["passed"]
                else "verification_failed",
                {
                    "verification_passed": verification["passed"],
                    "artifact_id": artifact.artifact_id,
                },
                tool_hint=next_instruction.get("tool_hint", "submit_step"),
                instruction=next_instruction.get("instruction"),
            )
            payload["verification"] = verification
            payload["artifact"] = _market_entry_summary(artifact)
            payload["market_entry"] = _market_entry_summary(artifact)
            payload["next_token"] = next_token
            payload["next_instruction"] = next_instruction
            payload["dna_summary"] = _dna_summary(session.flattened)
            return payload

    def complete_execution(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            self._ensure_session_active(session)
            if session.current_token() is not None:
                raise ValueError(
                    "execution session still has remaining steps; finish the DNA before completion"
                )
            artifacts = [
                session.market[key]
                for key in sorted(
                    session.market,
                    key=lambda artifact_id: (
                        session.market[artifact_id].epoch,
                        artifact_id,
                    ),
                )
            ]
            dispatch_updates, dispatch_provenance = _artifacts_to_dispatch_updates(
                [
                    artifact
                    for artifact in artifacts
                    if artifact.token in {"WRITE_PATCH", "REPAIR", "WRITE_PLAN"}
                ]
            )
            if not dispatch_updates:
                dispatch_updates = dict(session.patch_updates)

            successful_execution_traces = session.successful_execution_traces_snapshot()
            execution_traces_for_mining = successful_execution_traces[:]
            if len(execution_traces_for_mining) == 1:
                execution_traces_for_mining = execution_traces_for_mining * 2

            learned_macros = (
                _mine_macros_for_session(execution_traces_for_mining, session.macro_lib)
                if execution_traces_for_mining
                else {}
            )
            if not learned_macros and successful_execution_traces:
                learned_macros = {
                    "LM1": successful_execution_traces[0][
                        : min(3, len(successful_execution_traces[0]))
                    ],
                }

            session.revision += 1

            motif_archive = (
                load_motif_cache(session.state_root)
                if session.state_root is not None
                else {}
            )
            for artifact in artifacts:
                motif = artifact.metadata.get("motif_sig") or motif_key(
                    artifact.metadata.get("trace", [])
                )
                slot = motif_archive.setdefault(
                    motif,
                    {
                        "count": 0,
                        "mean_governed": 0.0,
                        "mean_profit": 0.0,
                        "mean_stability": 0.0,
                        "best_governed_fitness": 0.0,
                        "specializations": {},
                    },
                )
                slot["count"] += 1
                slot["mean_governed"] += (
                    artifact.score - slot["mean_governed"]
                ) / slot["count"]
                slot["mean_profit"] += (artifact.price - slot["mean_profit"]) / slot[
                    "count"
                ]
                stability = (
                    1.0
                    if artifact.status == "passed"
                    else 0.0
                    if artifact.status == "failed"
                    else 0.65
                )
                slot["mean_stability"] += (stability - slot["mean_stability"]) / slot[
                    "count"
                ]
                slot["best_governed_fitness"] = max(
                    slot["best_governed_fitness"], artifact.score
                )
                specialization = str(
                    artifact.metadata.get("specialization", "generalist")
                )
                slot["specializations"][specialization] = (
                    int(slot["specializations"].get(specialization, 0)) + 1
                )

            if session.state_root is not None:
                combined_macros = dict(session.macro_lib)
                combined_macros.update(learned_macros)
                completion_summary = _completion_summary(
                    session,
                    artifacts,
                    dispatch_updates,
                    learned_macros,
                    successful_execution_traces,
                )
                save_discoverables(
                    session.state_root,
                    combined_macros,
                    motif_archive,
                    metadata={
                        "session_id": session.session_id,
                        "task_text": session.task_text,
                        "dna_summary": _dna_summary(session.flattened),
                    },
                )
                save_run_summary(
                    session.state_root,
                    {
                        "summary": completion_summary,
                        "completion_summary": completion_summary,
                        "task_pool": session.context.get("synthesis", {}).get(
                            "task_pool", []
                        ),
                        "feed_prior_dna": session.dna,
                        "learned_macros": learned_macros,
                        "controller_summary": session.context.get("synthesis", {}).get(
                            "summary", {}
                        ),
                        "best_governed": (
                            max(
                                artifacts,
                                key=lambda artifact: (
                                    artifact.score,
                                    artifact.price,
                                    artifact.artifact_id,
                                ),
                            ).to_dict()
                            if artifacts
                            else {}
                        ),
                        "best_raw": (
                            max(
                                artifacts,
                                key=lambda artifact: (
                                    artifact.score,
                                    artifact.price,
                                    artifact.artifact_id,
                                ),
                            ).to_dict()
                            if artifacts
                            else {}
                        ),
                    },
                    metadata={
                        "session_id": session.session_id,
                        "task_text": session.task_text,
                    },
                )
                save_execution_feedback(
                    session.state_root,
                    {
                        "kind": "execution_feedback",
                        "session_id": session.session_id,
                        "task_text": session.task_text,
                        "task_shape_signature": str(
                            session.context.get("task_shape_signature", "") or ""
                        ),
                        "route_hint": str(session.context.get("route_hint", "") or ""),
                        "decision_policy": _jsonable(
                            session.context.get("decision_policy", {})
                        ),
                        "completion_summary": completion_summary,
                        "verification_result": _jsonable(session.verification_result),
                        "execution_learning": {
                            "successful_trace_count": len(successful_execution_traces),
                            "learned_macro_count": len(learned_macros),
                        },
                        "artifact_count": len(artifacts),
                        "market_count": len(session.market),
                        "dispatch_update_count": len(dispatch_updates),
                    },
                )
            else:
                completion_summary = _completion_summary(
                    session,
                    artifacts,
                    dispatch_updates,
                    learned_macros,
                    successful_execution_traces,
                )

            session.completed = True

        dispatch_plan = {
            "workspace_root": str(session.workspace_root)
            if session.workspace_root
            else "",
            "updates": dispatch_updates,
        }
        verification_spec = _verification_spec_for_dispatch_updates(
            dispatch_updates,
            dispatch_provenance,
            artifact_id=session.context.get("verification_target_artifact_id")
            or session.context.get("last_artifact_id")
            or "",
        )
        verification_command: str | list[str] | None = (
            _verification_command_for_dispatch_updates(dispatch_updates)
        )
        if not verification_spec:
            verification_spec = as_mapping(session.verification_result.get("spec"))
        if verification_command is None:
            verification_command = session.verification_command
        if verification_command is not None:
            dispatch_plan["verification_command"] = verification_command
        if verification_spec:
            dispatch_plan["verification_spec"] = verification_spec

        payload = _session_payload(
            session,
            "completed",
            {
                "successful_trace_count": len(successful_execution_traces),
                "learned_macro_count": len(learned_macros),
                "patch_update_count": len(dispatch_updates),
                "verification_passed": session.verification_result.get("passed", False),
            },
            tool_hint="dispatch_plan" if dispatch_updates else "solve_problem",
            instruction="Call dispatch_plan(plan_json, workspace_root=..., commit=false) to preview the governed file changes.",
            include_dna=True,
        )
        payload["completed"] = True
        payload["dispatch_plan"] = _jsonable(dispatch_plan)
        payload["learned_macros"] = learned_macros
        payload["completion_summary"] = _jsonable(completion_summary)
        payload["market"] = session.market_snapshot()
        payload["execution_learning"] = {
            "successful_trace_count": len(successful_execution_traces),
            "learned_macro_count": len(learned_macros),
        }
        payload["market_count"] = len(session.market)
        payload["verification_result"] = _jsonable(session.verification_result)
        payload["dna_summary"] = _dna_summary(session.flattened)
        return payload


REGISTRY = ExecutionRegistry()


def begin_execution(
    synthesis_result: dict[str, Any] | str, state_dir: str | Path | None = None
) -> dict[str, Any]:
    return REGISTRY.begin_execution(synthesis_result, state_dir=state_dir)


def submit_step(session_id: str, content: str) -> dict[str, Any]:
    return REGISTRY.submit_step(session_id, content)


def read_artifact(session_id: str, artifact_id: str) -> dict[str, Any]:
    return REGISTRY.read_artifact(session_id, artifact_id)


def view_market(session_id: str, limit: int = 8) -> dict[str, Any]:
    return REGISTRY.view_market(session_id, limit=limit)


def verify_step(session_id: str, command: str) -> dict[str, Any]:
    return REGISTRY.verify_step(session_id, command)


def complete_execution(session_id: str) -> dict[str, Any]:
    return REGISTRY.complete_execution(session_id)
