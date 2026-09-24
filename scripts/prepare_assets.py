#!/usr/bin/env python3
"""Provision explicitly listed external data from an authorized local directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path, *, root: Path = ROOT) -> int:
    source = source.resolve()
    root = root.resolve()
    entries = json.loads((root / "configs/external_assets.json").read_text())["assets"]
    pending: list[tuple[Path, Path]] = []
    for entry in entries:
        relative = Path(entry["path"])
        origin = (source / relative).resolve()
        destination = (root / relative).resolve()
        if relative.is_absolute() or not origin.is_relative_to(source) or not destination.is_relative_to(root):
            raise ValueError("Asset paths must remain inside their configured directories")
        if not origin.is_file():
            continue
        expected = entry["sha256"]
        if hashlib.sha256(origin.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset integrity mismatch: {relative}")
        if destination.exists():
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Refusing to overwrite a different asset: {relative}")
        else:
            pending.append((origin, destination))
    if not pending and not any((source / entry["path"]).is_file() for entry in entries):
        raise ValueError("No declared assets found in the supplied directory")
    # Validate the entire supplied set before writing any destination.
    for origin, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare(args.source)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot provision assets: {exc}\n")
    print(f"Prepared {count} external assets in ignored runtime locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
