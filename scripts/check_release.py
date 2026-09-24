#!/usr/bin/env python3
"""Check a source directory for accidental private or generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", ".cache", ".runtime", ".venv", "__pycache__", ".pytest_cache",
    ".ruff_cache", "outputs", "models", "datasets", "checkpoints", "build", "dist",
}
EXCLUDED_SUFFIXES = {
    ".jsonl", ".parquet", ".safetensors", ".pt", ".pth", ".ckpt", ".pyc",
    ".h5", ".hdf5", ".pruned_init", ".mp4", ".webm", ".pem", ".key", ".p12", ".pfx",
}
SECRET = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}|\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{16,}"
    r"|\bAKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
OLD_NAMESPACE = re.compile(r"\bP" + r"AR_[A-Z0-9_]+")
PRIVATE_PATH = re.compile(r"/(?:home|Users|Knowin)/[A-Za-z0-9_.-]+/|/mnt/data/[A-Za-z0-9_.-]+/")
EMBEDDED_AUTH = re.compile(r"https?://[^\s/]+:[^\s/]+@")
APPROVED_IMAGES = {
    "assets/framework.png": "6aa138c9fbb926fff444c04229e6bc0a0ca79cc807e1a197f94c30d2916a625d",
    "assets/main-results.png": "811a2feb7d95f79e76a0bbc57b72045de3666ce8247a1bd5b84165db997b8be0",
    "assets/libero-pro.png": "b8e537c7553629ff273db32bfefc6f701636c96109801fdd37893601c821c880",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".lock", ".json", ".txt", ".yaml", ".yml",
    ".sh", ".patch", ".example", ".csv", ".bddl", ".in",
}
TEXT_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".gitmodules", ".rayignore", ".python-version", ".dockerignore"}


def inspect(root: Path, *, tracked_only: bool = False) -> list[str]:
    findings: list[str] = []
    manifest = root / "configs/external_assets.json"
    assets = {entry["path"] for entry in json.loads(manifest.read_text())["assets"]}
    if tracked_only:
        try:
            names = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL
            ).decode().split("\0")
        except subprocess.CalledProcessError:
            return ["tracked scan requires a Git checkout"]
        paths = sorted(root / name for name in names if name)
        if not paths:
            return ["tracked scan has no files; stage the intended release first"]
    else:
        paths = sorted(root.rglob("*"))
    for path in paths:
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink():
            findings.append(f"symlink: {name}")
            continue
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            if path.is_file() or path.is_dir():
                findings.append(f"generated/private directory: {name}")
            continue
        if tracked_only and not path.exists():
            findings.append(f"missing tracked file: {name}")
            continue
        if not path.is_file():
            continue
        if name in assets or path.suffix in EXCLUDED_SUFFIXES:
            findings.append(f"external/generated asset: {name}")
        if path.name == "local.env" or path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            findings.append(f"local credential file: {name}")
        if path.resolve() == Path(__file__).resolve():
            continue
        if name in APPROVED_IMAGES:
            if hashlib.sha256(path.read_bytes()).hexdigest() != APPROVED_IMAGES[name]:
                findings.append(f"unreviewed image content: {name}")
            continue
        if path.name not in TEXT_NAMES and path.suffix not in TEXT_SUFFIXES:
            findings.append(f"unreviewed file type: {name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append(f"binary file requires explicit release review: {name}")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if SECRET.search(line):
                findings.append(f"credential-shaped content: {name}:{number}")
            if EMBEDDED_AUTH.search(line):
                findings.append(f"embedded URL credentials: {name}:{number}")
            if PRIVATE_PATH.search(line):
                findings.append(f"machine path: {name}:{number}")
            if relative.parts[0] != "third_party" and OLD_NAMESPACE.search(line):
                findings.append(f"obsolete configuration namespace: {name}:{number}")
    for required in ("LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/rats/LICENSE"):
        if not (root / required).is_file():
            findings.append(f"missing license material: {required}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tracked", action="store_true", help="Check working-tree content of Git-tracked files only")
    args = parser.parse_args()
    findings = inspect(args.root.resolve(), tracked_only=args.tracked)
    if findings:
        print("\n".join(findings))
        return 1
    print("Source release checks passed for the selected files (heuristic checks only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
