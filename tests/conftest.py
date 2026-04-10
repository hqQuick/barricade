from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)