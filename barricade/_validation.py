from __future__ import annotations

from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    pass


def require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")
    return value


def require_nonempty_str(value: Any, name: str) -> str:
    s = require_str(value, name)
    if not s.strip():
        raise ValidationError(f"{name} must not be empty")
    return s


def require_int(
    value: Any, name: str, min_val: int | None = None, max_val: int | None = None
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value


def require_positive_int(value: Any, name: str) -> int:
    return require_int(value, name, min_val=1)


def require_float_or_none(
    value: Any, name: str, min_val: float | None = None, max_val: float | None = None
) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(
            f"{name} must be a number or null, got {type(value).__name__}"
        )
    f = float(value)
    if min_val is not None and f < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {f}")
    if max_val is not None and f > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {f}")
    return f


def validate_solve_problem(
    problem_text: str,
    prior_strength: float | None = None,
    trials: int = 16,
    population: int = 96,
    base_episodes: int = 18,
    generations: int = 40,
) -> None:
    require_nonempty_str(problem_text, "problem_text")
    require_positive_int(trials, "trials")
    require_positive_int(population, "population")
    require_positive_int(base_episodes, "base_episodes")
    require_positive_int(generations, "generations")
    require_float_or_none(prior_strength, "prior_strength", min_val=0.0, max_val=1.0)


def validate_begin_execution(synthesis_result_json: str) -> None:
    require_nonempty_str(synthesis_result_json, "synthesis_result_json")


def validate_manage_execution(
    session_id: str,
    action: str,
    content: str = "",
    command: str = "",
    artifact_id: str = "",
) -> None:
    require_nonempty_str(session_id, "session_id")
    require_nonempty_str(action, "action")
    normalized = action.strip().lower()
    if normalized in {"submit", "step"} and not content:
        raise ValidationError("submit action requires content")
    if normalized == "verify" and not command:
        raise ValidationError("verify action requires command")
    if normalized in {"read", "artifact"} and not artifact_id:
        raise ValidationError("read action requires artifact_id")


def validate_dispatch_plan(plan_json: str) -> None:
    require_nonempty_str(plan_json, "plan_json")


def validate_workspace_root(
    workspace_root: str | Path | None,
    *,
    base_root: str | Path | None = None,
) -> Path | None:
    if workspace_root is None:
        return None
    if isinstance(workspace_root, str) and not workspace_root.strip():
        return None

    resolved_root = Path(workspace_root).expanduser().resolve()
    allowed_root = Path(base_root or Path.cwd()).expanduser().resolve()
    try:
        resolved_root.relative_to(allowed_root)
    except ValueError as exc:
        raise ValidationError(
            f"workspace_root must stay within {allowed_root}"
        ) from exc
    return resolved_root


def validate_analyze_scaling_profile(result_json: str) -> None:
    require_nonempty_str(result_json, "result_json")
