#!/usr/bin/env python3
"""Capture a credential-free environment and artifact provenance snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAM3_CHECKPOINT = REPOSITORY_ROOT / ".cache" / "models" / "sam3" / "sam3.pt"
CONTACT_GRASPNET_CHECKPOINT = (
    REPOSITORY_ROOT
    / ".cache"
    / "models"
    / "contact_graspnet_pytorch"
    / "checkpoints"
    / "contact_graspnet"
    / "checkpoints"
    / "model.pt"
)


def _protocol_exact_response_cache(protocol: Path) -> bool:
    """Read the registered cache policy without requiring PyYAML.

    The provenance command is also useful as a stand-alone diagnostic, where
    ``configs/env.sh`` may not have been sourced.  In that case, defaulting an
    absent ``RACAP_LLM_CACHE`` to enabled can contradict the frozen protocol.
    Keep the environment variable as an explicit override, but otherwise
    report the registered value.
    """

    if "RACAP_LLM_CACHE" in os.environ:
        return os.environ["RACAP_LLM_CACHE"] != "0"
    try:
        for raw_line in protocol.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("exact_response_cache:"):
                continue
            value = line.split(":", 1)[1].strip().lower()
            if value in {"true", "yes", "1"}:
                return True
            if value in {"false", "no", "0"}:
                return False
    except OSError:
        pass
    return False


def _command(argv: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            argv, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=30
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}"


def _git(path: Path) -> dict[str, object]:
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    ).stdout
    return {
        "path": str(path.resolve()),
        "commit": _command(["git", "rev-parse", "HEAD"], path),
        "status_porcelain": _command(["git", "status", "--short"], path),
        "remote": _command(["git", "remote", "get-url", "origin"], path),
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(path: Path) -> dict[str, object]:
    if path.is_file():
        return {
            "root": str(path.resolve()),
            "sha256": _sha256(path),
            "files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            ],
        }
    rows = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        rows.append(
            {
                "path": str(item.relative_to(path)),
                "bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(path.resolve()), "sha256": digest, "files": rows}


def _port(host: str, port: int) -> dict[str, object]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return {
                "host": host,
                "port": port,
                "reachable": True,
                "latency_ms": round(1000 * (time.perf_counter() - started), 3),
            }
    except OSError as exc:
        return {
            "host": host,
            "port": port,
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _versions(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rats-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "rats",
    )
    parser.add_argument(
        "--champion-root",
        type=Path,
        default=REPOSITORY_ROOT / "policies" / "phase2",
    )
    parser.add_argument("--rats-library", type=Path, default=None)
    args = parser.parse_args()

    protocol = REPOSITORY_ROOT / "experiments" / "controlled_comparison" / "protocol.yaml"
    task_manifest = protocol.with_name("task_manifest.json")
    payload: dict[str, object] = {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "repositories": {
            "racap": _git(REPOSITORY_ROOT),
            "rats": _git(args.rats_root),
        },
        "artifacts": {
            "racap_phase2": _tree_manifest(args.champion_root),
            "comparison_harness": _tree_manifest(protocol.parent),
            "service_model_assets": {
                "sam3": _tree_manifest(SAM3_CHECKPOINT),
                "contact_graspnet": _tree_manifest(CONTACT_GRASPNET_CHECKPOINT),
            },
            "protocol_sha256": _sha256(protocol),
            "task_manifest_sha256": _sha256(task_manifest),
        },
        "packages": _versions(
            [
                "gymnasium",
                "libero",
                "mujoco",
                "numpy",
                "open3d",
                "pandas",
                "robosuite",
                "scipy",
                "torch",
            ]
        ),
        "services": [
            _port("127.0.0.1", 8214),
            _port("127.0.0.1", 8215),
            _port("127.0.0.1", 8216),
        ],
        "model_configuration": {
            "provider": "vapi",
            "base_url": os.environ.get("RACAP_VAPI_BASE", ""),
            "requested_model": os.environ.get("RACAP_MODEL", "gpt-5.5"),
            "credential_present": bool(os.environ.get("RACAP_VAPI_KEY")),
            "credential_value_recorded": False,
            "maximum_concurrency": int(os.environ.get("RACAP_VLM_MAX_CONCURRENCY", "10")),
            "exact_response_cache": _protocol_exact_response_cache(protocol),
            "provider_prefix_token_cache": "allowed_and_logged_per_request",
        },
        "gpu": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ).splitlines(),
    }
    if args.rats_library:
        payload["artifacts"]["rats_library"] = _tree_manifest(args.rats_library)  # type: ignore[index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "services": payload["services"]}))


if __name__ == "__main__":
    main()
