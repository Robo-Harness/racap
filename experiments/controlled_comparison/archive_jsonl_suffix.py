#!/usr/bin/env python3
"""Archive telemetry rows at or after a known interrupted-run timestamp.

This utility is intentionally narrow: it operates only on evaluator telemetry
JSONL files, requires every nonempty line to parse, writes through a temporary
file, and records before/after hashes plus the archived rows.  Episode records,
summaries, videos, and trajectories are never selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


PATTERNS = (
    "llm_calls.worker*.jsonl",
    "native_states.worker*.jsonl",
    "sim_episodes.worker*.jsonl",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--time-threshold", type=float, required=True)
    args = parser.parse_args()

    if args.source.resolve() == args.archive.resolve():
        raise SystemExit("source and archive must differ")
    args.archive.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "archive": str(args.archive.resolve()),
        "time_threshold": args.time_threshold,
        "files": [],
    }

    paths = sorted({path for pattern in PATTERNS for path in args.source.glob(pattern)})
    for path in paths:
        before = path.read_bytes()
        kept: list[bytes] = []
        archived: list[bytes] = []
        episode_keys: set[str] = set()
        for number, raw_line in enumerate(before.splitlines(keepends=True), 1):
            if not raw_line.strip():
                kept.append(raw_line)
                continue
            try:
                row = json.loads(raw_line)
                timestamp = float(row["time"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"invalid telemetry row {path}:{number}: {exc}") from exc
            if timestamp >= args.time_threshold:
                archived.append(raw_line)
                episode_keys.add(str(row.get("episode_key") or row.get("key") or ""))
            else:
                kept.append(raw_line)
        if not archived:
            continue

        archived_bytes = b"".join(archived)
        archive_path = args.archive / path.name
        if archive_path.exists():
            raise SystemExit(f"refusing to overwrite archive file: {archive_path}")
        archive_path.write_bytes(archived_bytes)

        kept_bytes = b"".join(kept)
        mode = path.stat().st_mode
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(kept_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        manifest["files"].append(
            {
                "path": str(path.resolve()),
                "archive_path": str(archive_path.resolve()),
                "before_sha256": _sha256(before),
                "after_sha256": _sha256(kept_bytes),
                "archived_sha256": _sha256(archived_bytes),
                "archived_rows": len(archived),
                "episode_keys": sorted(episode_keys),
            }
        )

    manifest["archived_rows"] = sum(row["archived_rows"] for row in manifest["files"])
    manifest_path = args.archive / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
