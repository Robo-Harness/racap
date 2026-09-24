"""Contract checks, deliberately without patch-size or score thresholds."""

from __future__ import annotations

import compileall
import os
import re
import subprocess
import sys
import tokenize
from io import StringIO
from dataclasses import dataclass
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
_GUARD_DEPTH_ENV = "RACAP_CANDIDATE_GUARD_DEPTH"


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


_FORBIDDEN = {
    "native oracle": re.compile(r"\b(runtime\.)?oracle\b"),
    "native predicate": re.compile(r"\b(task_completed|predicate_status|predicate_diagnostics)\b"),
    "task-id specialization": re.compile(r"\b(task_id|seed)\s*(?:==|in\s*[\(\[])"),
}


def _executable_tokens(text: str) -> str:
    """Return Python source with prose strings and comments removed.

    Leakage checks must inspect executable references, not English
    explanations such as ``"this is not a task-completion oracle"``.  Token
    filtering keeps identifiers and operators (including imports and actual
    attribute access) while preventing docstrings, prompts, and comments from
    creating false positives.
    """
    try:
        tokens = []
        for token in tokenize.generate_tokens(StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT}:
                token = tokenize.TokenInfo(
                    token.type, "", token.start, token.end, token.line
                )
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except (IndentationError, tokenize.TokenError):
        # Compilation already reports malformed Python. Returning the original
        # text keeps the safety gate conservative for an invalid candidate.
        return text


def check_candidate(path: Path) -> GuardResult:
    path = path.resolve()
    source = path
    errors: list[str] = []
    checks: list[str] = []
    if not (source / "solution" / "controller.py").is_file():
        errors.append("missing solution/controller.py")
    else:
        checks.append("controller present")

    if source.is_dir() and compileall.compile_dir(source, quiet=1, force=True):
        checks.append("syntax compiled")
    else:
        errors.append("Python compilation failed")

    for file in sorted(source.rglob("*.py")) if source.is_dir() else ():
        text = _executable_tokens(file.read_text(encoding="utf-8"))
        for label, pattern in _FORBIDDEN.items():
            if pattern.search(text):
                errors.append(f"{file.relative_to(path)}: forbidden {label}")

    environment = os.environ.copy()
    # Candidate repositories include the harness tests themselves.  Without a
    # depth marker, the orchestrator test creates a candidate whose guard runs
    # that same test again, recursively spawning pytest forever.  The outer
    # guard still runs the complete candidate suite; only a guard constructed
    # from inside that suite suppresses its nested test launch.
    nested_guard = bool(environment.get(_GUARD_DEPTH_ENV))
    environment[_GUARD_DEPTH_ENV] = str(
        int(environment.get(_GUARD_DEPTH_ENV, "0") or 0) + 1
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(source),
            str(FRAMEWORK_ROOT),
            str(FRAMEWORK_ROOT / "third_party" / "rats"),
            environment.get("PYTHONPATH", ""),
        )
    )
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from solution.controller import run_episode; assert callable(run_episode)",
        ],
        cwd=path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if imported.returncode:
        errors.append(f"controller import failed: {imported.stdout[-1200:]}")
    else:
        checks.append("controller imported")

    tests = path / "tests"
    if tests.is_dir() and not nested_guard:
        tested = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(tests)],
            cwd=path,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if tested.returncode:
            errors.append(f"candidate tests failed: {tested.stdout[-2000:]}")
        else:
            checks.append("candidate tests passed")
    elif tests.is_dir():
        checks.append("candidate tests delegated to outer guard")
    return GuardResult(not errors, tuple(checks), tuple(errors))
