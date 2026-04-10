from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileError:
    path: str
    line: int | None
    column: int | None
    error_type: str
    message: str


@dataclass
class TestFailure:
    test_name: str
    file: str | None
    line: int | None
    assertion: str
    message: str


@dataclass
class StructuredReport:
    passed: bool
    returncode: int
    syntax_errors: list[FileError] = field(default_factory=list)
    import_errors: list[FileError] = field(default_factory=list)
    runtime_errors: list[FileError] = field(default_factory=list)
    test_failures: list[TestFailure] = field(default_factory=list)
    semantic_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "returncode": self.returncode,
            "syntax_errors": [
                {
                    "path": e.path,
                    "line": e.line,
                    "column": e.column,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.syntax_errors
            ],
            "import_errors": [
                {
                    "path": e.path,
                    "line": e.line,
                    "column": e.column,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.import_errors
            ],
            "runtime_errors": [
                {
                    "path": e.path,
                    "line": e.line,
                    "column": e.column,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.runtime_errors
            ],
            "test_failures": [
                {
                    "test_name": t.test_name,
                    "file": t.file,
                    "line": t.line,
                    "assertion": t.assertion,
                    "message": t.message,
                }
                for t in self.test_failures
            ],
            "semantic_failures": self.semantic_failures,
            "warnings": self.warnings,
            "summary": self.summary,
            "actionable_hints": self._actionable_hints(),
        }

    def _actionable_hints(self) -> list[str]:
        hints: list[str] = []
        if self.syntax_errors:
            paths = ", ".join(e.path for e in self.syntax_errors[:3])
            hints.append(f"Fix source errors in: {paths}")
        if self.import_errors:
            modules = ", ".join(
                e.message.split("'")[1] if "'" in e.message else e.message
                for e in self.import_errors[:3]
            )
            hints.append(f"Missing imports: {modules}")
        if self.test_failures:
            names = ", ".join(t.test_name for t in self.test_failures[:3])
            hints.append(f"Failing tests: {names}")
        if self.semantic_failures:
            hints.append(f"Semantic failures: {'; '.join(self.semantic_failures[:3])}")
        if self.runtime_errors:
            hints.append(f"{len(self.runtime_errors)} runtime error(s) detected")
        return hints


_TRACEBACK_FILE_RE = re.compile(r'^\s*File\s+"([^"]+)",\s+line\s+(\d+)')
_COMPILER_ERROR_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+)(?::(?P<column>\d+))?:\s+"
    r"(?:(?P<level>error|warning|note)(?:\[[^\]]+\])?:\s*)?"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)
_IMPORT_ERROR_RE = re.compile(
    r"(?:ImportError|ModuleNotFoundError):\s+(?:.*?)(?:'([^']+)')?"
)
_ASSERTION_RE = re.compile(r"AssertionError:\s*(.*)")
_PYTEST_SUMMARY_RE = re.compile(
    r"^\s*(?:FAILED|FAIL|ERROR)(?::)?\s+(.+?)(?:\s+-\s+(.*))?$"
)
_PYTEST_FILE_RE = re.compile(r"file\s+(\S+),\s+line\s+(\d+)")
_DEPRECATION_RE = re.compile(
    r"DeprecationWarning|PendingDeprecationWarning|UserWarning"
)


def _parse_syntax_errors(stderr: str, stdout: str) -> list[FileError]:
    errors: list[FileError] = []
    lines = (f"{stderr}\n{stdout}").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TRACEBACK_FILE_RE.match(line)
        if m:
            path = m.group(1)
            lineno = int(m.group(2))
            error_type = "SyntaxError"
            message = ""
            end_index = i
            for j in range(i + 1, min(i + 4, len(lines))):
                end_index = j
                if "SyntaxError" in lines[j]:
                    error_type = "SyntaxError"
                    message = lines[j].split("SyntaxError:", 1)[-1].strip()
                    break
                if "IndentationError" in lines[j]:
                    error_type = "IndentationError"
                    message = lines[j].split("IndentationError:", 1)[-1].strip()
                    break
            errors.append(
                FileError(
                    path=path,
                    line=lineno,
                    column=None,
                    error_type=error_type,
                    message=message,
                )
            )
            i = end_index + 1
            continue

        compiler_match = _COMPILER_ERROR_RE.match(line)
        if compiler_match:
            level = (compiler_match.group("level") or "error").lower()
            message = compiler_match.group("message").strip()
            error_type = "CompilerError" if level == "error" else level.capitalize()
            errors.append(
                FileError(
                    path=compiler_match.group("path").strip(),
                    line=int(compiler_match.group("line")),
                    column=int(compiler_match.group("column"))
                    if compiler_match.group("column")
                    else None,
                    error_type=error_type,
                    message=message,
                )
            )
            i += 1
            continue
        i += 1
    return errors


def _parse_import_errors(stderr: str, stdout: str) -> list[FileError]:
    errors: list[FileError] = []
    combined = f"{stderr}\n{stdout}"
    for m in _IMPORT_ERROR_RE.finditer(combined):
        module = m.group(1) or "unknown"
        path = ""
        lines_before = combined[: m.start()].splitlines()
        for lb in reversed(lines_before):
            em = _TRACEBACK_FILE_RE.match(lb)
            if em:
                path = em.group(1)
                break
        errors.append(
            FileError(
                path=path,
                line=None,
                column=None,
                error_type="ModuleNotFoundError",
                message=f"Cannot import '{module}'",
            )
        )
    return errors


def _parse_test_failures(stderr: str, stdout: str) -> list[TestFailure]:
    failures: list[TestFailure] = []
    combined = f"{stderr}\n{stdout}"
    lines = combined.splitlines()

    for i, line in enumerate(lines):
        tm = _PYTEST_SUMMARY_RE.match(line)
        if not tm:
            continue

        test_name = tm.group(1).strip()
        file_path = test_name.split("::", 1)[0] if "::" in test_name else None
        file_line = None
        assertion_msg = tm.group(2).strip() if tm.group(2) else ""

        for j in range(max(0, i - 5), min(len(lines), i + 6)):
            pm = _PYTEST_FILE_RE.search(lines[j])
            if pm:
                file_path = pm.group(1)
                file_line = int(pm.group(2))
            am = _ASSERTION_RE.search(lines[j])
            if am and not assertion_msg:
                assertion_msg = am.group(1).strip()

        failures.append(
            TestFailure(
                test_name=test_name,
                file=file_path,
                line=file_line,
                assertion=assertion_msg,
                message=assertion_msg or line.strip(),
            )
        )
    return failures


def _parse_warnings(stderr: str, stdout: str) -> list[str]:
    warnings: list[str] = []
    combined = f"{stderr}\n{stdout}"
    for m in _DEPRECATION_RE.finditer(combined):
        start = max(0, m.start() - 20)
        end = min(len(combined), m.end() + 100)
        snippet = combined[start:end].strip()
        if snippet and snippet not in warnings:
            warnings.append(snippet)
    return warnings[:10]


def _build_summary(report: StructuredReport) -> str:
    parts: list[str] = []
    if report.passed:
        parts.append("Verification passed.")
        return " ".join(parts)
    parts.append("Verification failed.")
    if report.syntax_errors:
        parts.append(f"{len(report.syntax_errors)} syntax error(s).")
    if report.import_errors:
        parts.append(f"{len(report.import_errors)} import error(s).")
    if report.test_failures:
        parts.append(f"{len(report.test_failures)} test failure(s).")
    if report.runtime_errors:
        parts.append(f"{len(report.runtime_errors)} runtime error(s).")
    if not parts[1:]:
        parts.append(f"Process exited with code {report.returncode}.")
    return " ".join(parts)


def parse_verification_output(
    completed: subprocess.CompletedProcess[str],
    spec: dict[str, Any] | None = None,
    cwd: Path | None = None,
    semantic_failures: list[str] | None = None,
) -> StructuredReport:
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    normalized_semantic_failures = [
        str(failure).strip()
        for failure in (semantic_failures or [])
        if str(failure).strip()
    ]

    report = StructuredReport(
        passed=completed.returncode == 0,
        returncode=completed.returncode,
    )

    report.syntax_errors = _parse_syntax_errors(stderr, stdout)
    report.import_errors = _parse_import_errors(stderr, stdout)
    report.test_failures = _parse_test_failures(stderr, stdout)
    report.warnings = _parse_warnings(stderr, stdout)
    report.semantic_failures = normalized_semantic_failures

    if (
        report.syntax_errors
        or report.import_errors
        or report.test_failures
        or report.semantic_failures
    ):
        report.passed = False

    if (
        not report.passed
        and not report.syntax_errors
        and not report.import_errors
        and not report.test_failures
    ):
        if report.semantic_failures:
            report.runtime_errors.append(
                FileError(
                    path="",
                    line=None,
                    column=None,
                    error_type="SemanticVerificationError",
                    message="; ".join(report.semantic_failures[:3]),
                )
            )
        else:
            combined = f"{stdout}\n{stderr}".strip()
            if combined:
                report.runtime_errors.append(
                    FileError(
                        path="",
                        line=None,
                        column=None,
                        error_type="RuntimeError",
                        message=combined[:500],
                    )
                )

    report.summary = _build_summary(report)
    return report
