"""Git-backed candidate isolation and lossless lineage."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(RuntimeError):
    pass


_APPLY_PATCH_HEADER = re.compile(
    r"^\*\*\* (Update|Add|Delete) File: (.+?)\s*$"
)


def _safe_candidate_relative(raw_path: str) -> Path:
    relative = Path(raw_path.strip())
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in {"solution", "memory", "tests"}
    ):
        raise WorkspaceError(f"unsafe candidate path: {raw_path!r}")
    return relative


def _find_context(source: list[str], context: list[str], start: int) -> int:
    """Locate one apply-patch hunk without trusting generated line numbers."""

    if not context:
        return start
    limit = len(source) - len(context) + 1
    for index in range(max(0, start), max(0, limit)):
        if source[index : index + len(context)] == context:
            return index
    # Trailing whitespace is a common model-copy artifact and carries no
    # Python meaning. Keep all other context matching exact and deterministic.
    stripped = [line.rstrip() for line in context]
    for index in range(max(0, start), max(0, limit)):
        if [line.rstrip() for line in source[index : index + len(context)]] == stripped:
            return index
    raise WorkspaceError("apply-patch hunk context was not found in candidate source")


def _apply_context_hunks(source: str, body: list[str]) -> str:
    """Apply Codex-style ``@@`` context hunks to one complete text file."""

    lines = source.splitlines()
    trailing_newline = source.endswith("\n")
    cursor = 0
    index = 0
    saw_hunk = False
    while index < len(body):
        if not body[index].startswith("@@"):
            if body[index].strip():
                raise WorkspaceError(
                    f"unexpected apply-patch line before hunk: {body[index]!r}"
                )
            index += 1
            continue
        saw_hunk = True
        index += 1
        hunk: list[str] = []
        while index < len(body) and not body[index].startswith("@@"):
            hunk.append(body[index])
            index += 1
        old: list[str] = []
        new: list[str] = []
        for line in hunk:
            if not line:
                # The custom patch notation normally prefixes blank context
                # with one space, but accepting an empty serialized line is
                # unambiguous and avoids a formatting-only rejection.
                old.append("")
                new.append("")
            elif line[0] == " ":
                old.append(line[1:])
                new.append(line[1:])
            elif line[0] == "-":
                old.append(line[1:])
            elif line[0] == "+":
                new.append(line[1:])
            else:
                raise WorkspaceError(f"invalid apply-patch hunk line: {line!r}")
        location = _find_context(lines, old, cursor)
        lines[location : location + len(old)] = new
        cursor = location + len(new)
    if not saw_hunk:
        raise WorkspaceError("apply-patch update contains no @@ hunk")
    result = "\n".join(lines)
    if trailing_newline or result:
        result += "\n"
    return result


def _apply_codex_patch(root: Path, payload: str) -> None:
    """Apply the common ``*** Begin Patch`` format emitted inside JSON files.

    Some hosted coding models put a valid context patch inside a ``files``
    value even when asked for complete file contents. Treating those bytes as
    Python manufactured syntax errors. This compatibility path preserves the
    model's intended edit while retaining path safety and exact context checks.
    """

    lines = payload.strip().splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise WorkspaceError("invalid apply-patch opening marker")
    if lines[-1].strip() != "*** End Patch":
        raise WorkspaceError("invalid apply-patch closing marker")
    index = 1
    sections = 0
    while index < len(lines) - 1:
        match = _APPLY_PATCH_HEADER.match(lines[index])
        if not match:
            raise WorkspaceError(f"invalid apply-patch file header: {lines[index]!r}")
        operation, raw_path = match.groups()
        relative = _safe_candidate_relative(raw_path)
        destination = root / relative
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not _APPLY_PATCH_HEADER.match(lines[index]):
            body.append(lines[index])
            index += 1
        if operation == "Update":
            if not destination.is_file():
                raise WorkspaceError(f"apply-patch update target missing: {relative}")
            updated = _apply_context_hunks(
                destination.read_text(encoding="utf-8"), body
            )
            destination.write_text(updated, encoding="utf-8")
        elif operation == "Add":
            if destination.exists():
                raise WorkspaceError(f"apply-patch add target exists: {relative}")
            added: list[str] = []
            for line in body:
                if not line.startswith("+"):
                    raise WorkspaceError(f"invalid apply-patch add line: {line!r}")
                added.append(line[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n".join(added) + "\n", encoding="utf-8")
        else:
            if body and any(line.strip() for line in body):
                raise WorkspaceError("apply-patch delete section must not contain hunks")
            if destination.is_dir():
                raise WorkspaceError(f"refusing directory deletion: {relative}")
            destination.unlink(missing_ok=True)
        sections += 1
    if not sections:
        raise WorkspaceError("apply-patch payload contains no file sections")


def _git(repository: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode:
        raise WorkspaceError(f"git {' '.join(args)} failed:\n{result.stdout}")
    return result.stdout.strip()


@dataclass(frozen=True)
class CandidateWorkspace:
    name: str
    path: Path
    branch: str
    parent_commit: str


class WorkspaceManager:
    """Own a small solution repository; the mature RACaP tree is never edited."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.repository = self.root / "repository"
        self.worktrees = self.root / "worktrees"

    def initialize(self, seed: Path) -> str:
        if self.repository.exists():
            if not (self.repository / ".git").exists():
                raise WorkspaceError(
                    f"existing path is not an experiment git repo: {self.repository}"
                )
            return self.head(self.repository)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            seed,
            self.repository,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        _git(self.repository, "init", "-b", "archive")
        _git(self.repository, "config", "user.name", "RACaP Evolution")
        _git(self.repository, "config", "user.email", "evolution@local")
        _git(self.repository, "add", "-A")
        _git(self.repository, "commit", "-m", "seed: minimally capable controller")
        _git(self.repository, "branch", "champion", "HEAD")
        return self.head(self.repository)

    @staticmethod
    def head(path: Path) -> str:
        return _git(path, "rev-parse", "HEAD")

    @staticmethod
    def _proposal_source_status(path: Path) -> tuple[str, ...]:
        """Return real candidate changes, excluding patch/reject bookkeeping."""
        lines = _git(path, "status", "--porcelain", "--untracked-files=all").splitlines()
        return tuple(
            line
            for line in lines
            if not line.endswith(" .racap-proposal.patch")
            and not line.rstrip().endswith(".rej")
        )

    def create_candidate(self, name: str, parent_commit: str) -> CandidateWorkspace:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise WorkspaceError(f"unsafe candidate name: {name!r}")
        self.worktrees.mkdir(parents=True, exist_ok=True)
        path = self.worktrees / name
        if path.exists():
            raise WorkspaceError(f"candidate already exists: {path}")
        branch = f"candidate/{name}"
        _git(self.repository, "worktree", "add", "-b", branch, str(path), parent_commit)
        return CandidateWorkspace(name, path, branch, parent_commit)

    def apply_patch(self, candidate: CandidateWorkspace, patch: str) -> tuple[str, ...]:
        """Apply as much of a model patch as Git can place unambiguously.

        A generated multi-file patch may contain one stale, non-essential
        context hunk while its executable hunks are otherwise valid.  First
        require an atomic application.  If that fails, let Git place only
        hunks whose context still matches, remove its ``.rej`` bookkeeping,
        and return the rejected hunks for the experiment record.  Compilation,
        contract tests, and simulator rollouts remain the actual acceptance
        path; a partial application with no source change is still rejected.
        """
        patch_path = candidate.path / ".racap-proposal.patch"
        # A unified diff is a line-oriented format.  Hosted models frequently
        # omit the final newline from an otherwise valid JSON string; Git then
        # reports the last hunk as a corrupt patch even though every context
        # line matches.  Normalize only that transport-level terminator.  The
        # patch body, paths, and context remain unchanged and fully checked.
        normalized_patch = patch if patch.endswith("\n") else patch + "\n"
        patch_path.write_text(normalized_patch, encoding="utf-8")
        try:
            # Hosted coding models occasionally emit correct unified-diff
            # bodies with stale hunk line counts.  Let Git infer those counts
            # from the body so a mechanical metadata error does not prevent a
            # syntactically valid candidate from reaching runtime evaluation.
            # Context and path checks remain fully enforced by ``git apply``.
            try:
                _git(candidate.path, "apply", "--check", "--recount", str(patch_path))
                _git(candidate.path, "apply", "--recount", str(patch_path))
                if not self._proposal_source_status(candidate.path):
                    raise WorkspaceError(
                        "git apply returned success but produced no candidate source change"
                    )
                return ()
            except WorkspaceError as atomic_error:
                partial = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(candidate.path),
                        "apply",
                        "--reject",
                        "--recount",
                        str(patch_path),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                rejected: list[str] = []
                for reject_path in sorted(candidate.path.rglob("*.rej")):
                    relative = reject_path.relative_to(candidate.path)
                    rejected.append(
                        f"{relative}:\n{reject_path.read_text(encoding='utf-8', errors='replace')}"
                    )
                    reject_path.unlink()
                status = self._proposal_source_status(candidate.path)
                if not status:
                    # A malformed hunk near one file boundary makes Git abort
                    # scanning the entire multi-file diff.  Retry each file
                    # section independently so unrelated runnable edits can
                    # still reach compilation and runtime evaluation.
                    sections = [
                        section
                        for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
                        if section.strip()
                    ]
                    section_outputs: list[str] = []
                    for index, section in enumerate(sections):
                        section_path = candidate.path / f".racap-section-{index}.patch"
                        section_path.write_text(section, encoding="utf-8")
                        try:
                            section_result = subprocess.run(
                                [
                                    "git",
                                    "-C",
                                    str(candidate.path),
                                    "apply",
                                    "--reject",
                                    "--recount",
                                    str(section_path),
                                ],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )
                            section_outputs.append(section_result.stdout)
                        finally:
                            section_path.unlink(missing_ok=True)
                        for reject_path in sorted(candidate.path.rglob("*.rej")):
                            relative = reject_path.relative_to(candidate.path)
                            rejected.append(
                                f"{relative}:\n{reject_path.read_text(encoding='utf-8', errors='replace')}"
                            )
                            reject_path.unlink()
                    status = self._proposal_source_status(candidate.path)
                    partial_output = partial.stdout + "\n" + "\n".join(section_outputs)
                else:
                    partial_output = partial.stdout
                if not status:
                    raise WorkspaceError(
                        f"atomic and partial patch application failed:\n{atomic_error}\n"
                        f"partial output:\n{partial_output}"
                    ) from atomic_error
                return tuple(rejected)
        finally:
            patch_path.unlink(missing_ok=True)

    def apply_files(
        self,
        candidate: CandidateWorkspace,
        files: dict[str, str | None],
    ) -> None:
        """Apply complete candidate-owned files without fragile diff syntax."""
        if not files:
            raise WorkspaceError("complete-file proposal is empty")
        context_patches: list[str] = []
        for raw_path, content in files.items():
            relative = _safe_candidate_relative(raw_path)
            if isinstance(content, str) and content.lstrip().startswith(
                "*** Begin Patch"
            ):
                context_patches.append(content)
                continue
            destination = candidate.path / relative
            if content is None:
                if destination.is_dir():
                    raise WorkspaceError(f"refusing directory deletion: {raw_path!r}")
                destination.unlink(missing_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        for payload in context_patches:
            _apply_codex_patch(candidate.path, payload)
        if not self._proposal_source_status(candidate.path):
            raise WorkspaceError("complete-file proposal produced no source change")

    def apply_proposal(
        self,
        candidate: CandidateWorkspace,
        *,
        patch: str = "",
        files: dict[str, str | None] | None = None,
    ) -> tuple[str, ...]:
        if files:
            self.apply_files(candidate, files)
            if patch:
                return self.apply_patch(candidate, patch)
            return ()
        return self.apply_patch(candidate, patch)

    def discard(self, candidate: CandidateWorkspace) -> None:
        """Remove an invalid worktree while retaining proposal diagnostics."""
        if candidate.path.exists():
            _git(self.repository, "worktree", "remove", "--force", str(candidate.path))
        _git(self.repository, "branch", "-D", candidate.branch, check=False)

    def commit_candidate(self, candidate: CandidateWorkspace, message: str) -> str:
        _git(candidate.path, "add", "-A")
        status = _git(candidate.path, "status", "--porcelain")
        if not status:
            raise WorkspaceError("proposal produced no source change")
        _git(candidate.path, "commit", "-m", message)
        return self.head(candidate.path)

    def changed_files(
        self,
        candidate: CandidateWorkspace,
        *,
        base_commit: str | None = None,
    ) -> tuple[str, ...]:
        output = _git(
            candidate.path,
            "diff",
            "--name-only",
            base_commit or candidate.parent_commit,
            self.head(candidate.path),
        )
        return tuple(line for line in output.splitlines() if line)

    def substantive_program_changes(
        self,
        candidate: CandidateWorkspace,
        *,
        base_commit: str | None = None,
    ) -> tuple[str, ...]:
        """Return executable or runtime-memory edits with nonzero content.

        A malformed unified diff can make Git create an empty new file while
        silently skipping every hunk that contained the intended program.  A
        compile/import/test guard then passes if the generated test file is
        empty as well.  Such a tree is not a candidate policy and must never
        consume a simulator-tested evolution slot.  Test-only edits likewise
        cannot change runtime behavior.  Renames or deletions with real text
        lines remain substantive through Git's numstat representation.
        """

        output = _git(
            candidate.path,
            "diff",
            "--numstat",
            "--no-renames",
            base_commit or candidate.parent_commit,
            self.head(candidate.path),
            "--",
            "solution",
            "memory",
        )
        changed: list[str] = []
        for line in output.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            try:
                content_delta = int(added) + int(deleted)
            except ValueError:
                # Binary runtime assets are rare but can still materially
                # alter a candidate; numstat represents them as ``-``.
                content_delta = 1 if "-" in {added, deleted} else 0
            if content_delta > 0:
                changed.append(path)
        return tuple(changed)

    def promote(self, candidate: CandidateWorkspace, *, tag: str | None = None) -> str:
        commit = self.head(candidate.path)
        _git(self.repository, "update-ref", "refs/heads/champion", commit)
        if tag:
            _git(self.repository, "tag", "-f", tag, commit)
        return commit

    def mark_efficiency(self, candidate: CandidateWorkspace) -> str:
        commit = self.head(candidate.path)
        _git(self.repository, "update-ref", "refs/heads/efficiency", commit)
        return commit

    def snapshot_diff(
        self,
        candidate: CandidateWorkspace,
        destination: Path,
        *,
        base_commit: str | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if base_commit and base_commit != self.head(candidate.path):
            body = _git(
                candidate.path,
                "diff",
                "--stat",
                "--patch",
                base_commit,
                self.head(candidate.path),
            )
            header = _git(
                candidate.path,
                "log",
                "--format=fuller",
                "--reverse",
                f"{base_commit}..{self.head(candidate.path)}",
            )
            payload = header + "\n--- cumulative candidate diff ---\n" + body
        else:
            payload = _git(
                candidate.path,
                "show",
                "--format=fuller",
                "--stat",
                "--patch",
                "HEAD",
            )
        destination.write_text(payload + "\n", encoding="utf-8")
