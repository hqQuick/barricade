from __future__ import annotations

from typing import Any

API_VERSION = "0.1.0"
__version__ = API_VERSION


def with_api_version(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["api_version"] = API_VERSION
    return result
