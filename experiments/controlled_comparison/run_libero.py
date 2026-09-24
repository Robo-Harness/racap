#!/usr/bin/env python3
"""Run one frozen method at a time on the registered LIBERO comparison grid.

This is infrastructure, not a policy: it maps the common task manifest onto
each upstream method's CLI, enforces the registered worker cap, preserves raw method
artifacts, and pauses rather than scoring provider/quota failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from experiments.controlled_comparison.supervise_after_development import (
    _verify_frozen_artifact_seal,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_RATS_ROOT = ROOT / "third_party" / "rats"
DEFAULT_PYTHON = Path(os.environ.get("RACAP_EXPERIMENT_PYTHON", sys.executable))
DEFAULT_MAIN_COHORTS = [
    "libero90_id_replay",
    "libero_pro_zero_shot",
    "libero_base_diagnostic",
    "libero_long",
]
QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient quota",
    "insufficient balance",
    "balance is insufficient",
    "need pre-deduct",
    "no available channel",
    "no available route",
    "quota_exhausted",
)
# Dedicated ports prevent measured runs from silently using unrelated
# long-lived services on the upstream defaults (8114--8116).
API_SERVER_PORTS = (8214, 8215, 8216)
RATS_EVAL_CONFIG = HERE / "configs" / "rats_libero_eval.yaml"
CAPX_EVAL_CONFIG = HERE / "configs" / "capx_libero_eval.yaml"
SAM3_CHECKPOINT = ROOT / ".cache" / "models" / "sam3" / "sam3.pt"
CONTACT_GRASPNET_ROOT = (
    ROOT / ".cache" / "models" / "contact_graspnet_pytorch"
)
CONTROLLED_LIBERO_CONFIG_DIR = (
    ROOT
    / "outputs"
    / "controlled_comparison"
    / "provenance"
    / "libero_runtime_config"
)
# Frozen evaluation is one continuous simulator episode per registered
# (suite, task, seed).  Development may use reset-based attempts, but exposing
# the same evaluation initial state repeatedly would give RATS privileged
# retry opportunities that CaP-X and RACaP do not receive.
EVAL_CONTINUOUS_TURNS = 10
EVAL_RATS_ATTEMPTS = 1
EVAL_RATS_SELF_CHECK_REPAIRS = 0
RESUME_MIGRATIONS_DIR = HERE / "resume_migrations"
IMPORT_ATTESTATION_TIMEOUT_SECONDS = float(
    os.environ.get("RACAP_IMPORT_ATTESTATION_TIMEOUT_SECONDS", "300")
)


def _acquire_method_run_lock(method_root: Path):
    """Hold an exclusive method lock before any resume mutation occurs.

    The descriptor intentionally remains live in ``main`` until the runner
    process exits.  Kernel ``flock`` ownership is released on every exit path,
    including signals, while the small metadata file makes an accidental
    concurrent launch diagnosable without trusting process-name matching.
    """

    path = method_root / ".run.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"method evaluation is already running for {method_root}: {owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "method_root": str(method_root.resolve()),
                "acquired_at_unix": time.time(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_payload_sha256(payload: object) -> str:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _manifest_diff_paths(before: object, after: object, prefix: str = "") -> list[str]:
    """Return deterministic leaf paths changed between two JSON payloads."""

    if isinstance(before, dict) and isinstance(after, dict):
        changed: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.append(path)
            else:
                changed.extend(_manifest_diff_paths(before[key], after[key], path))
        return changed
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [prefix]
        changed = []
        for index, (left, right) in enumerate(zip(before, after)):
            changed.extend(
                _manifest_diff_paths(left, right, f"{prefix}[{index}]")
            )
        return changed
    return [] if before == after else [prefix]


def _completed_generated_episode_keys(
    method_root: Path, jobs: list[dict[str, Any]]
) -> list[str]:
    keys = []
    for job in jobs:
        if (_generated_task_root(method_root, job) / "COMPLETE.json").is_file():
            keys.append(
                f"{job['suite']}/{int(job['task_id'])}/seed{int(job['seed'])}"
            )
    return sorted(keys)


def _episode_key_set_sha256(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode("utf-8")).hexdigest()


def _apply_registered_manifest_migration(
    *,
    method: str,
    method_root: Path,
    manifest_path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    jobs: list[dict[str, Any]],
    registry_dir: Path = RESUME_MIGRATIONS_DIR,
) -> dict[str, Any] | None:
    """Apply an exact, pre-registered infrastructure-only resume migration.

    Arbitrary source changes remain fail-closed.  A migration must bind both
    manifest hashes, the complete set of changed fields, and the exact set of
    already-scored episodes.  It may additionally restrict completed public
    seeds.  The old and new manifests plus an applied record are retained next
    to the measured method output.
    """

    before_sha = _json_payload_sha256(before)
    after_sha = _json_payload_sha256(after)
    changed_paths = _manifest_diff_paths(before, after)
    completed_keys = _completed_generated_episode_keys(method_root, jobs)
    completed_sha = _episode_key_set_sha256(completed_keys)

    for registration_path in sorted(registry_dir.glob("*.json")):
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if (
            registration.get("method") != method
            or registration.get("from_manifest_sha256") != before_sha
            or registration.get("to_manifest_sha256") != after_sha
        ):
            continue
        if sorted(registration.get("allowed_manifest_diff_paths", [])) != changed_paths:
            raise RuntimeError(
                "registered resume migration diff does not match: "
                f"expected={registration.get('allowed_manifest_diff_paths')}, "
                f"observed={changed_paths}"
            )
        if int(registration.get("expected_completed_episode_count", -1)) != len(
            completed_keys
        ):
            raise RuntimeError("registered resume migration episode count does not match")
        if registration.get("expected_completed_episode_keys_sha256") != completed_sha:
            raise RuntimeError("registered resume migration episode set does not match")
        allowed_seeds = {
            int(seed) for seed in registration.get("compatible_completed_public_seeds", [])
        }
        completed_seeds = {
            int(key.rsplit("seed", 1)[1]) for key in completed_keys
        }
        if not completed_seeds.issubset(allowed_seeds):
            raise RuntimeError(
                "registered resume migration does not cover completed public seeds"
            )

        migration_id = str(registration["migration_id"])
        archive = method_root / "resume_migrations" / migration_id
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archive / "run_manifest_before.json")
        _write_json(archive / "run_manifest_after.json", after)
        applied = {
            "schema_version": 1,
            "migration_id": migration_id,
            "registration_path": str(registration_path.resolve()),
            "registration_sha256": _sha256(registration_path),
            "applied_at_unix": time.time(),
            "from_manifest_sha256": before_sha,
            "to_manifest_sha256": after_sha,
            "changed_manifest_paths": changed_paths,
            "completed_episode_count": len(completed_keys),
            "completed_episode_keys_sha256": completed_sha,
            "completed_public_seeds": sorted(completed_seeds),
        }
        _write_json(archive / "APPLIED.json", applied)
        _write_json(manifest_path, after)
        return applied
    return None


def _public_model_route(endpoint: str) -> str:
    """Return a credential-free endpoint identity for resume compatibility.

    Model identity telemetry catches provider substitutions after a request,
    while this value prevents a resumed grid from silently switching gateway
    or API route before the next request.  Userinfo, query parameters, and
    fragments are deliberately excluded from the persisted manifest.
    """

    parsed = urlsplit(str(endpoint).strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("RACAP_VAPI_BASE must be an absolute HTTP(S) endpoint")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))


def _tree_sha256(path: Path) -> str:
    """Content hash for a directory, independent of timestamps and modes."""

    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _controlled_pythonpath() -> str:
    """Return absolute, checkout-stable import roots for every child process.

    A relative outer ``PYTHONPATH=.`` changes meaning after an episode launcher
    changes cwd.  That previously allowed LIBERO to resolve from an unrelated
    editable BCap-X checkout.  The controlled comparison needs the vendored
    CaP-X/RATS implementations while the registered experiment environment
    supplies LIBERO and robosuite through its own installed path hooks.
    """

    return os.pathsep.join(
        str(path.resolve())
        for path in (
            ROOT / "third_party" / "rats" / "capx-baseline",
            ROOT / "third_party" / "rats",
            ROOT,
        )
    )


def _controlled_libero_env() -> dict[str, str]:
    """Keep LIBERO data lookup independent of the login user's home config."""

    return {"LIBERO_CONFIG_PATH": str(CONTROLLED_LIBERO_CONFIG_DIR.resolve())}


def _controlled_rats_libero_env() -> dict[str, str]:
    """Restrict RATS' optional API registry to the evaluated simulator stack.

    RATS supports several optional robot stacks from one package.  Importing
    all of them in a fresh process can leave the LIBERO registry only partly
    populated when an earlier optional stack triggers a circular import.  The
    experiment is explicitly a LIBERO run, so binding the registry to
    ``libero`` is both the intended upstream mechanism and a fail-closed way
    to make process startup independent of optional Robosuite installations.
    """

    return {"CAPX_ENV_STACK": "libero"}


def _materialize_controlled_libero_config(libero_package: Path) -> dict[str, Any]:
    """Bind all benchmark resources to the attested LIBERO package tree.

    LIBERO normally reads ``~/.libero/config.yaml``.  That mutable global file
    can point at an unrelated checkout even when Python imports the intended
    package.  The controlled comparison instead writes a private config and
    passes its directory to every child process through ``LIBERO_CONFIG_PATH``.
    """

    benchmark_root = libero_package.resolve()
    roots = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    missing = [str(path) for path in roots.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing controlled LIBERO resources: " + ", ".join(missing))

    config_dir = CONTROLLED_LIBERO_CONFIG_DIR.resolve()
    datasets = config_dir / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    payload = {
        "assets": str(roots["assets"]),
        "bddl_files": str(roots["bddl_files"]),
        "benchmark_root": str(roots["benchmark_root"]),
        # Demonstration datasets are not consumed by these online rollouts,
        # but LIBERO expects this key to name an existing directory.
        "datasets": str(datasets),
        "init_states": str(roots["init_states"]),
    }
    config_path = config_dir / "config.yaml"
    rendered = yaml.safe_dump(payload, sort_keys=True)
    if not config_path.exists() or config_path.read_text(encoding="utf-8") != rendered:
        config_path.write_text(rendered, encoding="utf-8")
    return {
        "config_dir": str(config_dir),
        "config_file": str(config_path),
        "config_sha256": _sha256(config_path),
        "resource_paths": {name: str(path) for name, path in roots.items()},
        "resource_tree_sha256": {
            name: _tree_sha256(path)
            for name, path in roots.items()
            if name != "benchmark_root"
        },
    }


def _python_source_tree_sha256(path: Path) -> str:
    """Hash importable Python source without volatile bytecode/cache files."""

    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.py")):
        if "__pycache__" in item.parts or not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_import_roots(module_files: dict[str, str]) -> dict[str, Any]:
    """Fail before reset if policy/simulator modules come from mixed trees."""

    paths = {name: Path(value).resolve() for name, value in module_files.items()}
    expected_capx = (ROOT / "third_party" / "rats" / "capx-baseline").resolve()
    expected_rats = (ROOT / "third_party" / "rats").resolve()
    errors: list[str] = []
    if not _is_within(paths["capx"], expected_capx):
        errors.append(f"capx:{paths['capx']} not within {expected_capx}")
    if not _is_within(paths["rats"], expected_rats):
        errors.append(f"rats:{paths['rats']} not within {expected_rats}")

    libero_third_party = next(
        (parent.parent for parent in paths["libero"].parents if parent.name == "LIBERO-PRO"),
        None,
    )
    robosuite_third_party = next(
        (
            parent.parent
            for parent in paths["robosuite"].parents
            if parent.name == "libero_dependencies"
        ),
        None,
    )
    if libero_third_party is None:
        errors.append(f"libero source is outside a LIBERO-PRO tree: {paths['libero']}")
    if robosuite_third_party is None:
        errors.append(
            f"robosuite source is outside a libero_dependencies tree: {paths['robosuite']}"
        )
    if (
        libero_third_party is not None
        and robosuite_third_party is not None
        and libero_third_party.resolve() != robosuite_third_party.resolve()
    ):
        errors.append(
            "LIBERO and robosuite resolve from different third_party roots: "
            f"{libero_third_party} != {robosuite_third_party}"
        )
    if errors:
        raise RuntimeError("incompatible experiment import roots: " + "; ".join(errors))
    libero_runtime = _materialize_controlled_libero_config(paths["libero"].parent)
    return {
        "pythonpath": _controlled_pythonpath(),
        "module_files": {name: str(path) for name, path in paths.items()},
        "module_sha256": {name: _sha256(path) for name, path in paths.items()},
        "source_tree_sha256": {
            "capx": _python_source_tree_sha256(expected_capx / "capx"),
            "rats": _python_source_tree_sha256(expected_rats / "rats"),
            "libero": _python_source_tree_sha256(paths["libero"].parent),
            "robosuite": _python_source_tree_sha256(paths["robosuite"].parent),
        },
        "simulator_third_party_root": str(libero_third_party.resolve()),
        "libero_runtime": libero_runtime,
    }


def _experiment_import_roots(
    python: Path, *, verify_rats_registry: bool = False
) -> dict[str, Any]:
    """Resolve and attest child-process source roots with the actual interpreter.

    Importing ``rats.integrations`` eagerly initializes the full LIBERO motion
    stack (including JAX / IK machinery) and can take several minutes on a cold
    host.  That registry is execution-critical for RATS, but it is not imported
    by CaP-X or RACaP.  All methods therefore receive the same source-root
    attestation, while only RATS runs pay for and require the dynamic registry
    check they actually depend on.
    """

    spec_marker = "RACAP_LIBERO_SPEC="
    marker = "RACAP_IMPORT_ROOTS="
    spec_probe = (
        "import importlib.util, pathlib; "
        "spec=importlib.util.find_spec('libero'); "
        "assert spec is not None and spec.origin is not None; "
        f"print('{spec_marker}' + str(pathlib.Path(spec.origin).resolve()))"
    )
    registry_probe = (
        "import rats.integrations; "
        "assert 'FrankaLiberoApiReducedSkillLibrary' in "
        "rats.integrations.list_apis(), "
        "'RATS LIBERO reduced-skill API was not registered'; "
        if verify_rats_registry
        else ""
    )
    probe = (
        "import importlib.util, json, pathlib; "
        "names=('capx','rats','libero','robosuite'); "
        "specs={name: importlib.util.find_spec(name) for name in names}; "
        "assert all(spec is not None and spec.origin is not None "
        "for spec in specs.values()); "
        + registry_probe
        + f"print('{marker}' + json.dumps({{name: "
        "str(pathlib.Path(spec.origin).resolve()) "
        "for name, spec in specs.items()}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = _controlled_pythonpath()
    env.update(_controlled_rats_libero_env())
    # Locate the package without executing its __init__.  This lets us create
    # the isolated config before LIBERO's import-time setup can consult it.
    spec_result = subprocess.run(
        [str(python), "-c", spec_probe],
        cwd=ROOT / "third_party" / "rats" / "capx-baseline",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=IMPORT_ATTESTATION_TIMEOUT_SECONDS,
    )
    spec_payload = next(
        (
            line[len(spec_marker):]
            for line in spec_result.stdout.splitlines()
            if line.startswith(spec_marker)
        ),
        None,
    )
    if spec_result.returncode != 0 or spec_payload is None:
        raise RuntimeError(
            "failed to locate LIBERO before import: "
            f"returncode={spec_result.returncode}; output={spec_result.stdout[-2000:]}"
        )
    _materialize_controlled_libero_config(Path(spec_payload).parent)
    env.update(_controlled_libero_env())
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=ROOT / "third_party" / "rats" / "capx-baseline",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=IMPORT_ATTESTATION_TIMEOUT_SECONDS,
    )
    payload = next(
        (line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if result.returncode != 0 or payload is None:
        raise RuntimeError(
            "failed to attest experiment import roots: "
            f"returncode={result.returncode}; output={result.stdout[-2000:]}"
        )
    attestation = _validate_import_roots(json.loads(payload))
    attestation["rats_registry_verified"] = bool(verify_rats_registry)
    return attestation


def _frozen_rats_state(library: Path, memory: Path | None) -> dict[str, Any]:
    return {
        "library": str(library.resolve()),
        "library_sha256": _sha256(library),
        "memory": str(memory.resolve()) if memory is not None else None,
        "memory_tree_sha256": _tree_sha256(memory) if memory is not None else None,
    }


def _require_frozen_rats_seal(library: Path, memory: Path | None = None) -> dict[str, object]:
    """Fail closed unless the registered RATS inputs belong to one valid seal."""

    root = library.resolve().parent
    if memory is not None and memory.resolve() != (root / "failure_memory").resolve():
        raise SystemExit("RATS library and failure memory must come from the same frozen artifact")
    audit = _verify_frozen_artifact_seal(root)
    if audit["status"] != "pass":
        raise SystemExit(f"frozen RATS artifact seal failed: {audit['error']}")
    return audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _generated_task_root(method_root: Path, job: dict[str, Any]) -> Path:
    return (
        method_root
        / str(job["cohort"])
        / str(job["suite"])
        / f"task_{int(job['task_id']):02d}"
        / f"seed_{int(job['seed']):02d}"
    )


def _payload_summary(path: Path) -> dict[str, Any]:
    """Return a content-addressed summary before moving an interrupted payload."""

    if path.is_file():
        return {
            "kind": "file",
            "files": 1,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "kind": "directory",
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "tree_sha256": _tree_sha256(path),
    }


def _archive_incomplete_generated_resume_state(
    method: str,
    jobs: list[dict[str, Any]],
    method_root: Path,
) -> dict[str, Any] | None:
    """Transactionally isolate partial episode artifacts before a grid resume.

    Generated-method telemetry is append-only. Reusing a task directory after
    quota or infrastructure interruption would therefore mix two simulator
    resets, model traces, and videos under one registered episode key. Complete
    episodes remain immutable and are skipped; every non-empty incomplete task
    directory is moved outside ``measured/`` before its clean rerun. Method-level
    abort/job/service ledgers are archived in the same event so a resumed run
    can write fresh state without destroying the prior checkpoint.
    """

    incomplete_jobs = [
        job
        for job in jobs
        if not (_generated_task_root(method_root, job) / "COMPLETE.json").is_file()
    ]
    if not incomplete_jobs:
        return None

    partials: list[tuple[dict[str, Any], Path]] = []
    for job in incomplete_jobs:
        root = _generated_task_root(method_root, job)
        if root.is_dir() and any(root.iterdir()):
            partials.append((job, root))

    method_state_names = (
        "ABORTED.json",
        "FAILURES.json",
        "job_results.json",
        "frozen_artifact_audit.json",
        "shared_api_services",
    )
    method_state = [
        method_root / name
        for name in method_state_names
        if (method_root / name).exists()
    ]
    if not partials and not method_state:
        return None

    event_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + f"_{time.time_ns()}"
    )
    event_root = (
        method_root.parent.parent
        / "interrupted_attempts"
        / method
        / event_id
    )
    event_root.mkdir(parents=True, exist_ok=False)

    plans: list[tuple[Path, Path, dict[str, Any]]] = []
    for job, source in partials:
        relative = source.relative_to(method_root)
        destination = event_root / "episodes" / relative
        plans.append(
            (
                source,
                destination,
                {
                    "category": "incomplete_episode",
                    "job": job,
                    "source": str(source.resolve()),
                    "archive": str(destination.resolve()),
                    "payload": _payload_summary(source),
                },
            )
        )
    for source in method_state:
        destination = event_root / "method_state" / source.name
        plans.append(
            (
                source,
                destination,
                {
                    "category": "method_state",
                    "source": str(source.resolve()),
                    "archive": str(destination.resolve()),
                    "payload": _payload_summary(source),
                },
            )
        )

    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination, _ in plans:
            if destination.exists():
                raise RuntimeError(f"resume archive collision: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append((source, destination))
        manifest = {
            "schema_version": 1,
            "reason": "incomplete_generated_method_resume",
            "method": method,
            "event_id": event_id,
            "archived_at_unix": time.time(),
            "method_root": str(method_root.resolve()),
            "registered_jobs": len(jobs),
            "already_complete_jobs": len(jobs) - len(incomplete_jobs),
            "incomplete_jobs": len(incomplete_jobs),
            "archived_partial_episode_directories": len(partials),
            "payloads": [row for _, _, row in plans],
        }
        _write_json(event_root / "resume_archive.json", manifest)
    except BaseException:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(source)
        raise
    return {**manifest, "manifest": str((event_root / "resume_archive.json").resolve())}


def _nonempty_files(root: Path, pattern: str) -> list[Path]:
    """Return deterministic, non-empty evidence files below ``root``."""

    return sorted(
        path.resolve()
        for path in root.rglob(pattern)
        if path.is_file() and path.stat().st_size > 0
    )


def _generated_episode_evidence(
    task_root: Path,
    method: str,
    *,
    expected_episode_keys: set[str] | None = None,
    maximum_registered_resets: int | None = None,
    expected_init_state_indices: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Audit the evidence contract before an episode can be marked complete.

    A zero process return code only means that the upstream CLI exited. It is
    not sufficient evidence that the benchmark episode has an authoritative
    native label, resource telemetry, a model trace, a human-readable trace,
    and a recording. Requiring those artifacts here prevents a partially
    written run from becoming an immutable result on resume.
    """

    common: dict[str, list[Path]] = {
        "native_telemetry": _nonempty_files(task_root, "native_states.jsonl"),
        "simulator_telemetry": _nonempty_files(task_root, "sim_episodes.jsonl"),
        "videos": _nonempty_files(task_root, "*.mp4"),
    }
    if method == "capx":
        method_specific = {
            "model_telemetry": _nonempty_files(task_root, "llm_calls.jsonl"),
            "machine_summary": _nonempty_files(task_root, "summaries.txt"),
            "human_trace": _nonempty_files(task_root, "all_responses.json"),
            "upstream_done_flag": _nonempty_files(task_root, "aaa_done_flag.txt"),
        }
    elif method in {"rats_base", "rats_90"}:
        method_specific = {
            "model_telemetry": _nonempty_files(task_root / "agent_io", "*.json"),
            "machine_summary": _nonempty_files(
                task_root / "artifacts", "final_summary.json"
            ),
            "human_trace": (
                _nonempty_files(task_root / "artifacts", "iteration_*.json")
                + _nonempty_files(task_root / "artifacts", "report.html")
            ),
        }
    else:
        raise ValueError(
            f"evidence audit is only defined for generated methods: {method}"
        )

    categories = {**common, **method_specific}
    missing = sorted(name for name, paths in categories.items() if not paths)
    reset_counts: dict[str, int] = {}
    observed_init_state_indices: dict[str, list[int | None]] = {}
    if expected_episode_keys is not None:
        reset_events = _jsonl_rows(common["simulator_telemetry"])
        reset_rows = _episode_key_counts(reset_events)
        reset_counts = {
            key: int(reset_rows.get(key, 0)) for key in sorted(expected_episode_keys)
        }
        observed_init_state_indices = {
            key: [
                int(row["init_state_index"])
                if row.get("init_state_index") is not None
                else None
                for row in reset_events
                if str(row.get("episode_key") or row.get("key") or "").strip() == key
            ]
            for key in sorted(expected_episode_keys)
        }
        for key, count in reset_counts.items():
            if count <= 0:
                missing.append(f"registered_reset:{key}")
            elif maximum_registered_resets is not None and count > maximum_registered_resets:
                missing.append(
                    f"excess_registered_resets:{key}:{count}>{maximum_registered_resets}"
                )
            if expected_init_state_indices is not None:
                expected_index = int(expected_init_state_indices[key])
                observed = observed_init_state_indices[key]
                if observed != [expected_index]:
                    missing.append(
                        f"wrong_init_state_index:{key}:{observed}!={[expected_index]}"
                    )
    missing = sorted(set(missing))
    return {
        "schema_version": 1,
        "method": method,
        "complete": not missing,
        "missing": missing,
        "registered_reset_counts": reset_counts,
        "registered_init_state_indices": observed_init_state_indices,
        "expected_init_state_indices": expected_init_state_indices,
        "maximum_registered_resets": maximum_registered_resets,
        "files": {
            name: [str(path) for path in paths]
            for name, paths in categories.items()
        },
    }


def _jsonl_episode_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("episode_key") or row.get("key") or "").strip()
            if key:
                keys.add(key)
    return keys


def _jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _episode_key_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("episode_key") or row.get("key") or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _racap_group_evidence(
    run_dir: Path,
    *,
    suite: str,
    task_ids: set[int],
    seeds: set[int],
) -> dict[str, Any]:
    """Verify every registered RACaP episode and its keyed telemetry."""

    expected = {
        f"{suite}/{task_id}/seed{seed}"
        for task_id in task_ids
        for seed in seeds
    }
    summary_path = run_dir / "summary.json"
    records: list[dict[str, Any]] = []
    parse_error = ""
    if summary_path.is_file() and summary_path.stat().st_size > 0:
        try:
            records = list(json.loads(summary_path.read_text(encoding="utf-8")).get("records") or [])
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    else:
        parse_error = "missing_or_empty_summary"

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_key[str(row.get("key") or "")].append(row)
    duplicate_keys = sorted(key for key, rows in by_key.items() if key and len(rows) != 1)
    missing_records = sorted(expected - set(by_key))
    unexpected_records = sorted(set(by_key) - expected - {""})

    artifact_errors: dict[str, list[str]] = {}
    for key in sorted(expected & set(by_key)):
        row = by_key[key][0]
        issues: list[str] = []
        artifacts = row.get("artifacts") or {}
        for name in ("video", "trace", "trajectory"):
            value = str(artifacts.get(name) or "").strip()
            path = Path(value) if value else None
            if path is None or not path.is_file() or path.stat().st_size <= 0:
                issues.append(f"missing_{name}")
        if int(artifacts.get("video_frames") or 0) <= 0:
            issues.append("no_video_frames")
        if row.get("video_error"):
            issues.append("video_error")
        if "native_success" not in row:
            issues.append("missing_native_label")
        if "simulator_steps" not in row:
            issues.append("missing_simulator_steps")
        expected_seed = int(key.rsplit("seed", 1)[-1])
        if row.get("init_state_index") != expected_seed:
            issues.append(
                f"wrong_init_state_index:{row.get('init_state_index')}!={expected_seed}"
            )
        if issues:
            artifact_errors[key] = issues

    llm_files = _nonempty_files(run_dir, "llm_calls*.jsonl")
    native_files = _nonempty_files(run_dir, "native_states*.jsonl")
    reset_files = _nonempty_files(run_dir, "sim_episodes*.jsonl")
    llm_keys = _jsonl_episode_keys(llm_files)
    native_keys = _jsonl_episode_keys(native_files)
    reset_events = _jsonl_rows(reset_files)
    reset_counts = _episode_key_counts(reset_events)
    reset_keys = set(reset_counts)
    missing_llm_telemetry = sorted(expected - llm_keys)
    missing_native_telemetry = sorted(expected - native_keys)
    missing_reset_telemetry = sorted(expected - reset_keys)
    unexpected_reset_telemetry = sorted(reset_keys - expected)
    reset_errors: dict[str, list[str]] = {}
    for key in sorted(expected):
        expected_seed = int(key.rsplit("seed", 1)[-1])
        rows = [
            row
            for row in reset_events
            if str(row.get("episode_key") or row.get("key") or "").strip() == key
        ]
        issues: list[str] = []
        if len(rows) != 1:
            issues.append(f"reset_count:{len(rows)}!=1")
        observed_indices = [row.get("init_state_index") for row in rows]
        if observed_indices != [expected_seed]:
            issues.append(
                f"wrong_init_state_index:{observed_indices}!={[expected_seed]}"
            )
        observed_public_seeds = [row.get("public_seed") for row in rows]
        if observed_public_seeds != [expected_seed]:
            issues.append(
                f"wrong_public_seed:{observed_public_seeds}!={[expected_seed]}"
            )
        if issues:
            reset_errors[key] = issues
    complete = not any(
        (
            parse_error,
            duplicate_keys,
            missing_records,
            unexpected_records,
            artifact_errors,
            missing_llm_telemetry,
            missing_native_telemetry,
            missing_reset_telemetry,
            unexpected_reset_telemetry,
            reset_errors,
        )
    )
    return {
        "schema_version": 1,
        "method_family": "racap",
        "complete": complete,
        "expected_episode_count": len(expected),
        "record_count": len(records),
        "summary": str(summary_path.resolve()),
        "summary_parse_error": parse_error,
        "duplicate_keys": duplicate_keys,
        "missing_records": missing_records,
        "unexpected_records": unexpected_records,
        "artifact_errors": artifact_errors,
        "llm_telemetry_files": [str(path) for path in llm_files],
        "native_telemetry_files": [str(path) for path in native_files],
        "reset_telemetry_files": [str(path) for path in reset_files],
        "missing_llm_telemetry": missing_llm_telemetry,
        "missing_native_telemetry": missing_native_telemetry,
        "missing_reset_telemetry": missing_reset_telemetry,
        "unexpected_reset_telemetry": unexpected_reset_telemetry,
        "reset_errors": reset_errors,
    }


def _port_ready(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _service_client_env() -> dict[str, str]:
    """Bind clients to comparison-owned perception and control services."""

    sam3, graspnet, pyroki = API_SERVER_PORTS
    return {
        "SAM3_SERVICE_URL": f"http://127.0.0.1:{sam3}",
        "GRASPNET_SERVICE_URL": f"http://127.0.0.1:{graspnet}",
        "PYROKI_SERVICE_URL": f"http://127.0.0.1:{pyroki}",
    }


def _service_process_env() -> dict[str, str]:
    """Add local model assets needed by parent-owned service processes."""

    return {
        **_service_client_env(),
        "CONTACT_GRASPNET_ROOT": str(CONTACT_GRASPNET_ROOT.resolve()),
        "PYTHONPATH": _controlled_pythonpath(),
        **_controlled_libero_env(),
        **_controlled_rats_libero_env(),
    }


def _service_asset_manifest() -> dict[str, dict[str, Any]]:
    """Hash the exact large-model files used by every measured method."""

    assets = {
        "sam3": SAM3_CHECKPOINT,
        "contact_graspnet": (
            CONTACT_GRASPNET_ROOT
            / "checkpoints"
            / "contact_graspnet"
            / "checkpoints"
            / "model.pt"
        ),
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing comparison model assets: " + ", ".join(missing))
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in assets.items()
    }


def _remap_api_server_ports(config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a copied upstream config onto the dedicated service ports."""

    mapping = {
        "sam3": API_SERVER_PORTS[0],
        "contact_graspnet": API_SERVER_PORTS[1],
        "pyroki": API_SERVER_PORTS[2],
    }
    for server in config.get("api_servers") or []:
        target = str(server.get("_target_") or "").lower()
        matches = [name for name in mapping if name in target]
        if len(matches) == 1:
            server["port"] = mapping[matches[0]]
            if matches[0] == "sam3":
                server["checkpoint_path"] = str(SAM3_CHECKPOINT.resolve())
    return config


def _stop_service_bundle(process: subprocess.Popen[Any]) -> int | None:
    """Stop a method-owned server bundle and all of its child services."""

    # ``launch_servers`` spawns one process per service.  Terminating only the
    # launcher leaves those children orphaned and keeps the comparison ports
    # occupied for the next method.  The launcher is created with
    # ``start_new_session=True``, so its PID is also the private process-group
    # id and can be stopped without touching unrelated services.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        returncode = (
            process.wait(timeout=30)
            if process.poll() is None
            else process.returncode
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        returncode = process.wait(timeout=10)

    deadline = time.time() + 30
    while any(_port_ready(port) for port in API_SERVER_PORTS):
        if time.time() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(1)
            if any(_port_ready(port) for port in API_SERVER_PORTS):
                raise RuntimeError(
                    "method-owned API services remained alive after process-group shutdown"
                )
            break
        time.sleep(0.2)
    return returncode


@contextlib.contextmanager
def _sigterm_as_keyboard_interrupt():
    """Let context-manager cleanup run when an experiment receives SIGTERM.

    The comparison runners own subprocess groups containing the shared vision,
    grasp, and IK services.  Python's default SIGTERM action exits immediately
    and therefore skips ``finally`` blocks, leaving those groups orphaned and
    their registered ports occupied.  Translate SIGTERM into the same unwind
    path as Ctrl-C while this process owns such a bundle.  ``signal.signal`` is
    only legal in the main thread; worker-thread callers retain normal signal
    semantics and still benefit from their caller's lifecycle management.
    """

    previous: Any | None = None

    def _raise_interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    except ValueError:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def _shared_api_server_bundle(
    args: argparse.Namespace,
    method_root: Path,
    *,
    owner: str = "run_libero_method_parent",
):
    """Keep one API-service bundle alive for both episode workers.

    The upstream episode launcher normally starts services on fixed ports and
    stops whichever services that episode happened to create.  With two
    independent episode processes this creates a lifecycle race: the first
    episode to finish can remove a service still used by its peer.  A single
    parent-owned bundle avoids that race without duplicating the large GPU
    models.  Episode launchers see the occupied ports and become clients only.
    """

    occupied = [port for port in API_SERVER_PORTS if _port_ready(port)]
    if occupied:
        raise RuntimeError(
            "refusing to share unowned API services before measured run; "
            f"ports already occupied: {occupied}"
        )
    # ``launch_servers`` changes cwd to the vendored RATS tree. Resolve every
    # path handed to that child so a caller's relative --output-root cannot
    # turn a valid config into a cwd-relative missing file.
    service_root = (method_root / "shared_api_services").resolve()
    service_root.mkdir(parents=True, exist_ok=True)
    assets = _service_asset_manifest()
    resolved_config = service_root / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(
            _remap_api_server_ports(yaml.safe_load(RATS_EVAL_CONFIG.read_text())),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log_path = service_root / "launcher.log"
    argv = [
        str(args.python),
        "-m",
        "rats.serving.launch_servers",
        "--config-path",
        str(resolved_config),
        "--gpus",
        "0",
        "--workers",
        "1",
        "--log-dir",
        str(service_root / "logs"),
        "--timeout",
        "300",
    ]
    started = time.time()
    handle = log_path.open("a", encoding="utf-8")
    handle.write("\n# launch: " + " ".join(argv) + "\n")
    handle.flush()
    # Service implementations are a comparison-owned dependency, independent
    # of whichever external RATS checkout supplies an embodiment executor.
    # ``python -m`` puts cwd ahead of PYTHONPATH; using ``args.rats_root`` here
    # therefore let an explicitly registered old executor silently replace the
    # validated SAM3/PyRoki service CLIs.  Pin cwd to the controlled source that
    # also heads ``_service_process_env()['PYTHONPATH']``.
    service_source = DEFAULT_RATS_ROOT.resolve()
    process = subprocess.Popen(
        argv,
        cwd=service_source,
        env={**os.environ.copy(), **_service_process_env()},
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    with _sigterm_as_keyboard_interrupt():
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "shared API service launcher exited before readiness: "
                        f"returncode={process.returncode}; log={log_path}"
                    )
                if all(_port_ready(port) for port in API_SERVER_PORTS):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"shared API services did not become ready within 600s: {log_path}"
                )
            _write_json(
                service_root / "lifecycle.json",
                {
                    "schema_version": 1,
                    "owner": owner,
                    "pid": process.pid,
                    "ports": list(API_SERVER_PORTS),
                    "model_assets": assets,
                    "service_source": str(service_source),
                    "service_python_source_sha256": _python_source_tree_sha256(
                        service_source / "rats" / "serving"
                    ),
                    "started_at_unix": started,
                    "ready_at_unix": time.time(),
                    "argv": argv,
                    "log": str(log_path.resolve()),
                    "stopped_at_unix": None,
                    "returncode": None,
                },
            )
            yield
        finally:
            returncode = _stop_service_bundle(process)
            handle.close()
            lifecycle_path = service_root / "lifecycle.json"
            lifecycle = (
                json.loads(lifecycle_path.read_text(encoding="utf-8"))
                if lifecycle_path.is_file()
                else {
                    "schema_version": 1,
                    "owner": owner,
                    "pid": process.pid,
                    "ports": list(API_SERVER_PORTS),
                    "service_source": str(service_source),
                    "started_at_unix": started,
                    "argv": argv,
                    "log": str(log_path.resolve()),
                }
            )
            lifecycle.update(
                {"stopped_at_unix": time.time(), "returncode": returncode}
            )
            _write_json(lifecycle_path, lifecycle)


def _rats_seed_binding(job: dict[str, Any]) -> dict[str, str]:
    """Bind a 0-based public seed to RATS's 1-based internal trial seed."""

    public_seed = int(job["seed"])
    episode_key = f"{job['suite']}/{int(job['task_id'])}/seed{public_seed}"
    return {
        "RATS_EPISODE_KEY": episode_key,
        "RATS_REGISTERED_EPISODE_KEY": episode_key,
        # LifelongLoop resets iteration 1 with ``1 + offset``.  Passing the
        # public seed as offset therefore selects LIBERO init-state index
        # ``(1 + public_seed - 1) == public_seed``.
        "RATS_SEED_OFFSET": str(public_seed),
    }


def _capx_seed_binding(job: dict[str, Any]) -> dict[str, str]:
    """Keep CaP-X's 1-based simulator seed separate from the public key."""

    public_seed = int(job["seed"])
    episode_key = f"{job['suite']}/{int(job['task_id'])}/seed{public_seed}"
    return {
        "CAPX_EPISODE_KEY": episode_key,
        "CAPX_REGISTERED_EPISODE_KEY": episode_key,
        # CaP-X executes trial 1. Its LIBERO adapter loads ``seed - 1``, so an
        # offset equal to the public seed selects that exact 0-based state.
        "CAPX_SEED_OFFSET": str(public_seed),
    }


def _rats_eval_budget_args(*, turns: int = EVAL_CONTINUOUS_TURNS) -> list[str]:
    """Return the no-reset RATS budget used by every frozen evaluation.

    Runtime self-check executes the candidate in the simulator and then resets
    before official delivery.  It is useful during skill development, but it
    would constitute an extra hidden evaluation episode here, so it is
    explicitly disabled.  Source correction remains available through the
    ordinary continuous multi-turn decider.
    """

    if turns <= 0:
        raise ValueError("evaluation turns must be positive")
    return [
        "--turns-per-attempt",
        str(turns),
        "--attempts-per-iteration",
        str(EVAL_RATS_ATTEMPTS),
        "--policy-self-check-repairs",
        str(EVAL_RATS_SELF_CHECK_REPAIRS),
        "--multi-turn-decision",
    ]


def _is_quota_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


def _run_logged(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    stop_event: threading.Event | None = None,
    hard_wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# argv: " + " ".join(argv) + "\n")
        handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        quota_failure = False
        scan_offset = 0
        stopped_by_peer = False
        hard_wall_timeout = False
        hard_deadline = (
            time.monotonic() + float(hard_wall_time_seconds)
            if hard_wall_time_seconds is not None
            else None
        )
        while process.poll() is None:
            time.sleep(0.5)
            try:
                with log_path.open("rb") as reader:
                    reader.seek(scan_offset)
                    chunk = reader.read()
                    scan_offset = reader.tell()
                if chunk and _is_quota_failure(chunk.decode("utf-8", errors="replace")):
                    quota_failure = True
                    if stop_event is not None:
                        stop_event.set()
            except OSError:
                pass
            if hard_deadline is not None and time.monotonic() >= hard_deadline:
                hard_wall_timeout = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            if stop_event is not None and stop_event.is_set():
                stopped_by_peer = not quota_failure
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
        returncode = process.wait()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    quota_failure = quota_failure or _is_quota_failure(text)
    if quota_failure and stop_event is not None:
        stop_event.set()
    return {
        "returncode": 124 if hard_wall_timeout else (75 if quota_failure else returncode),
        "seconds": round(time.time() - started, 3),
        "quota_failure": quota_failure,
        "stopped_by_peer_quota": stopped_by_peer,
        "hard_wall_timeout": hard_wall_timeout,
        "hard_wall_time_seconds": hard_wall_time_seconds,
        "log": str(log_path),
    }


def _materialize_rats_hard_wall_timeout_artifacts(
    task_root: Path,
    job: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Create an auditable terminal trace after the outer RATS watchdog fires.

    The 1150-second watchdog is a registered per-episode resource limit.  A
    watchdog termination is therefore a scored episode timeout, not a broken
    method launch and not a reason to terminate peer episodes.  The upstream
    process may be killed before its final JSON epilogue, so the harness writes
    the two terminal files required by the evidence contract.  The native
    telemetry remains the sole authority for success and predicate completion.
    """

    artifacts = task_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    iteration_path = artifacts / "iteration_001.json"
    summary_path = artifacts / "final_summary.json"
    need_iteration = not iteration_path.is_file() or iteration_path.stat().st_size == 0
    need_summary = not summary_path.is_file() or summary_path.stat().st_size == 0
    episode_key = f"{job['suite']}/{job['task_id']}/seed{job['seed']}"
    native_path = task_root / "native_states.jsonl"
    states = [
        row
        for row in (_jsonl_rows([native_path]) if native_path.is_file() else [])
        if str(row.get("episode_key") or row.get("key") or "") == episode_key
    ]
    states.sort(key=lambda row: float(row.get("time") or 0.0))
    final = states[-1] if states else {}
    native_success = bool(final.get("native_success", False))
    elapsed = float(result.get("seconds") or result.get("hard_wall_time_seconds") or 0.0)
    termination = {
        "kind": "registered_hard_wall_timeout",
        "harness_generated": True,
        "hard_wall_time_seconds": result.get("hard_wall_time_seconds"),
        "launcher_returncode": int(result.get("returncode", 124)),
        "native_state_events": len(states),
        "native_success_at_termination": native_success,
    }
    iteration = {
        "schema_version": 1,
        "iteration": 1,
        "harness_generated": True,
        "task_proposal": {
            "activity_name": f"{job['suite']}_task{job['task_id']}",
            "scene_model": job["suite"],
            "activity_definition_id": int(job["task_id"]),
            "goal_conditions": job["instruction"],
            "language": job["instruction"],
        },
        "success": native_success,
        "elapsed_seconds": elapsed,
        "feedback_action": "registered_hard_wall_timeout",
        "termination": termination,
    }
    summary = {
        "schema_version": 1,
        "harness_generated": True,
        "total_iterations": 1,
        "successful_iterations": int(native_success),
        "failed_iterations": int(not native_success),
        "success_rate": float(native_success),
        "termination": termination,
        "iterations": [iteration],
    }
    # Preserve a complete upstream epilogue if the child managed to flush one
    # immediately before the outer watchdog reaped its process group.  These
    # harness records are only a recovery path for genuinely missing terminal
    # evidence, never a replacement for richer upstream evidence.
    if need_iteration:
        _write_json(iteration_path, iteration)
    if need_summary:
        _write_json(summary_path, summary)
    _materialize_timeout_public_visual_trace(task_root)


def _materialize_timeout_public_visual_trace(task_root: Path) -> Path | None:
    """Encode retained public observations when a watchdog preempts RATS.

    RATS normally writes an MP4 in its final epilogue.  A hard watchdog can
    preempt that epilogue even though the model-facing RGB observations were
    already durably written under ``agent_io``.  Those observations are the
    public visual evidence available to the policy, so encoding them in
    timestamp/name order preserves an inspectable episode trace without using
    simulator-private state or weakening the evidence gate.
    """

    existing = _nonempty_files(task_root, "*.mp4")
    if existing:
        return existing[0]

    primary = sorted((task_root / "agent_io").glob("*_img_0.png"))
    candidates = primary or sorted((task_root / "agent_io").glob("*_img_*.png"))
    if not candidates:
        candidates = sorted(
            path
            for pattern in ("*.png", "*.jpg", "*.jpeg")
            for path in (task_root / "artifacts").rglob(pattern)
            if "plot" not in path.name.lower()
            and "success_rate" not in path.name.lower()
            and "efficiency" not in path.name.lower()
            and "growth" not in path.name.lower()
        )
    if not candidates:
        return None

    # Remove only adjacent duplicates: repeated observations still matter
    # after an intervening action, while identical verifier copies do not.
    frames: list[Path] = []
    previous_hash: str | None = None
    for path in candidates:
        digest = _sha256(path)
        if digest == previous_hash:
            continue
        frames.append(path.resolve())
        previous_hash = digest
    if not frames:
        return None

    artifacts = task_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    concat_path = artifacts / "registered_timeout_visual_frames.ffconcat"
    output_path = artifacts / "registered_timeout_public_trace.mp4"

    def ffconcat_quote(path: Path) -> str:
        return str(path).replace("'", "'\\''")

    lines = ["ffconcat version 1.0"]
    for path in frames:
        lines.extend((f"file '{ffconcat_quote(path)}'", "duration 0.5"))
    # The concat demuxer needs the final frame repeated to honor its duration.
    lines.append(f"file '{ffconcat_quote(frames[-1])}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r",
        "10",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    provenance = {
        "schema_version": 1,
        "kind": "registered_timeout_public_visual_trace",
        "source": "retained_model_facing_rgb_observations",
        "private_simulator_state_used": False,
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4000:],
        "frames": [
            {"path": str(path), "sha256": _sha256(path)} for path in frames
        ],
        "output": str(output_path.resolve()),
    }
    if output_path.is_file() and output_path.stat().st_size > 0:
        provenance["output_sha256"] = _sha256(output_path)
        provenance["output_bytes"] = output_path.stat().st_size
    _write_json(artifacts / "registered_timeout_visual_trace.json", provenance)
    return (
        output_path.resolve()
        if output_path.is_file() and output_path.stat().st_size > 0
        else None
    )


def _jobs(manifest: dict[str, Any], cohorts: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "cohort": row["cohort"],
            "suite": row["suite"],
            "task_id": int(row["task_id"]),
            "seed": int(row["seed"]),
            "instruction": str(row["instruction"]),
            "instruction_source": str(
                row.get("instruction_source", "benchmark_task_language")
            ),
        }
        for row in manifest["rows"]
        if not cohorts or row["cohort"] in cohorts
    ]


def _target_family(suite: str) -> str | None:
    for family in ("spatial", "goal", "object"):
        if f"_{family}_" in suite or suite.endswith(f"_{family}"):
            return family
    return None


def _experience_card(
    args: argparse.Namespace,
    *,
    suite: str,
) -> Path | None:
    if args.experience_card_dir is None:
        return None
    family = _target_family(suite)
    if family is None:
        return None
    card = args.experience_card_dir / f"{args.method}_{family}.md"
    if not card.is_file():
        raise SystemExit(f"missing registered experience card: {card}")
    return card


def _capx_job(
    job: dict[str, Any],
    args: argparse.Namespace,
    method_root: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {
            "job": job,
            "returncode": 75,
            "quota_failure": False,
            "stopped_by_peer_quota": True,
        }
    task_root = _generated_task_root(method_root, job)
    complete = task_root / "COMPLETE.json"
    if complete.exists():
        return {"job": job, "skipped": True, "returncode": 0, "quota_failure": False}
    task_root.mkdir(parents=True, exist_ok=True)
    config = _remap_api_server_ports(
        copy.deepcopy(yaml.safe_load(CAPX_EVAL_CONFIG.read_text()))
    )
    config["env"]["cfg"]["low_level"]["suite_name"] = job["suite"]
    config["env"]["cfg"]["low_level"]["task_id"] = job["task_id"]
    config["output_dir"] = str(task_root / "artifacts")
    config["trials"] = 1
    config["multi_turn_limit"] = EVAL_CONTINUOUS_TURNS
    config_path = task_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    card = _experience_card(args, suite=job["suite"])
    env.update(
        {
            "PYTHONPATH": _controlled_pythonpath(),
            **_controlled_libero_env(),
            "OPENAI_API_KEY": env.get("RACAP_VAPI_KEY", ""),
            **_service_client_env(),
            **_capx_seed_binding(job),
            "CONTROLLED_TASK_INSTRUCTION": job["instruction"],
            "CONTROLLED_TASK_INSTRUCTION_SOURCE": job["instruction_source"],
            "CAPX_LLM_TELEMETRY_PATH": str(task_root / "llm_calls.jsonl"),
            "CAPX_SIM_EPISODE_TELEMETRY_PATH": str(task_root / "sim_episodes.jsonl"),
            "CAPX_NATIVE_TELEMETRY_PATH": str(task_root / "native_states.jsonl"),
            "CAPX_RUNTIME_VLM_URL": env.get("RACAP_VAPI_BASE", "").rstrip("/")
            + "/chat/completions",
            "CAPX_RUNTIME_VLM_MODEL": args.model,
            "CAPX_RUNTIME_VLM_KEY": env.get("RACAP_VAPI_KEY", ""),
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if card is not None:
        env["CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"] = str(card.resolve())
    base = env.get("RACAP_VAPI_BASE", "").rstrip("/")
    argv = [
        str(args.python),
        "-m",
        "capx.envs.launch",
        "--config-path",
        str(config_path),
        "--server-url",
        f"{base}/chat/completions",
        "--model",
        args.model,
        "--temperature",
        "0",
        "--max-tokens",
        "8192",
        "--reasoning-effort",
        "medium",
        "--visual-differencing-model",
        args.model,
        "--visual-differencing-model-server-url",
        f"{base}/chat/completions",
        "--total-trials",
        "1",
        "--num-workers",
        "1",
        "--record-video",
        "True",
        "--output-dir",
        str(task_root / "artifacts"),
    ]
    result = _run_logged(
        argv,
        cwd=args.rats_root / "capx-baseline",
        env=env,
        log_path=task_root / "run.log",
        stop_event=stop_event,
    )
    if result["returncode"] == 0:
        evidence = _generated_episode_evidence(
            task_root,
            "capx",
            expected_episode_keys={
                f"{job['suite']}/{job['task_id']}/seed{job['seed']}"
            },
            maximum_registered_resets=1,
            expected_init_state_indices={
                f"{job['suite']}/{job['task_id']}/seed{job['seed']}": int(job["seed"])
            },
        )
        _write_json(task_root / "EVIDENCE.json", evidence)
        if evidence["complete"]:
            _write_json(complete, {"job": job, "evidence": evidence, **result})
        else:
            result["returncode"] = 2
            result["missing_evidence"] = evidence["missing"]
    return {"job": job, **result}


def _rats_job(
    job: dict[str, Any],
    args: argparse.Namespace,
    method_root: Path,
    library: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {
            "job": job,
            "returncode": 75,
            "quota_failure": False,
            "stopped_by_peer_quota": True,
        }
    task_root = _generated_task_root(method_root, job)
    complete = task_root / "COMPLETE.json"
    if complete.exists():
        return {"job": job, "skipped": True, "returncode": 0, "quota_failure": False}
    task_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    card = _experience_card(args, suite=job["suite"])
    env.update(
        {
            "PYTHONPATH": _controlled_pythonpath(),
            **_controlled_libero_env(),
            **_controlled_rats_libero_env(),
            "RATS_VAPI_URL": env.get("RACAP_VAPI_BASE", "").rstrip("/")
            + "/chat/completions",
            "RATS_VAPI_KEY": env.get("RACAP_VAPI_KEY", ""),
            "RATS_LLM_MODEL": args.model,
            "RATS_RUNTIME_VLM_MODEL": args.model,
            "RATS_LLM_FALLBACK": "0",
            "RATS_EPISODE_WALL_TIME_SECONDS": "1000",
            **_service_client_env(),
            "RATS_VERIFY_STEP_MODE": "strict",
            "RATS_VERIFIER_STRICT_BENCHMARK": "1",
            "RATS_AGENT_IO_DIR": str(task_root / "agent_io"),
            "RATS_SIM_EPISODE_TELEMETRY_PATH": str(task_root / "sim_episodes.jsonl"),
            "RATS_NATIVE_TELEMETRY_PATH": str(task_root / "native_states.jsonl"),
            **_rats_seed_binding(job),
            "CONTROLLED_TASK_INSTRUCTION": job["instruction"],
            "CONTROLLED_TASK_INSTRUCTION_SOURCE": job["instruction_source"],
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if card is not None:
        env["CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"] = str(card.resolve())
    argv = [
        str(args.python),
        "scripts/run_rats.py",
        "--config",
        str(RATS_EVAL_CONFIG),
        "--model",
        args.model,
        "--env-type",
        "libero",
        "--libero-suite",
        job["suite"],
        "--libero-task",
        str(job["task_id"]),
        "--iterations",
        "1",
        "--fixed-task",
        "--skill-library",
        str(library),
        *_rats_eval_budget_args(),
        "--output-dir",
        str(task_root / "artifacts"),
        "--log-agent-io",
    ]
    if args.method == "rats_base":
        argv.extend(["--no-skill-reuse", "--no-failure-memory"])
    elif args.rats_memory is not None:
        argv.extend(["--failure-memory-path", str(args.rats_memory)])
    result = _run_logged(
        argv,
        cwd=args.rats_root,
        env=env,
        log_path=task_root / "run.log",
        stop_event=stop_event,
        hard_wall_time_seconds=1150,
    )
    if result["hard_wall_timeout"]:
        # The registered watchdog is an episode-local resource outcome.  Keep
        # the upstream launcher code for provenance, then normalize the
        # harness return code so peer episodes continue and this episode is
        # scored from its native predicate telemetry.
        _materialize_rats_hard_wall_timeout_artifacts(task_root, job, result)
        result["launcher_returncode"] = int(result["returncode"])
        result["returncode"] = 0
        result["registered_timeout_failure"] = True
    if result["returncode"] == 0:
        evidence = _generated_episode_evidence(
            task_root,
            args.method,
            expected_episode_keys={
                f"{job['suite']}/{job['task_id']}/seed{job['seed']}"
            },
            maximum_registered_resets=1,
            expected_init_state_indices={
                f"{job['suite']}/{job['task_id']}/seed{job['seed']}": int(job["seed"])
            },
        )
        _write_json(task_root / "EVIDENCE.json", evidence)
        if evidence["complete"]:
            _write_json(complete, {"job": job, "evidence": evidence, **result})
        else:
            result["returncode"] = 2
            result["missing_evidence"] = evidence["missing"]
    return {"job": job, **result}


def _run_generated_method(
    method: str,
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    method_root: Path,
) -> int:
    stop_event = threading.Event()
    resume_archive = _archive_incomplete_generated_resume_state(
        method,
        jobs,
        method_root,
    )
    if resume_archive is not None:
        print(
            f"[{method}] archived interrupted resume state: "
            f"{resume_archive['manifest']}",
            flush=True,
        )
    frozen_before: dict[str, Any] | None = None
    if method == "capx":
        def runner(job: dict[str, Any]) -> dict[str, Any]:
            return _capx_job(job, args, method_root, stop_event)
    else:
        library = (
            args.rats_root / "skill_library" / "libero_nonpriv_skills.json"
            if method == "rats_base"
            else args.rats_library
        )
        if library is None or not library.is_file():
            raise SystemExit(f"missing frozen RATS library for {method}: {library}")
        if method == "rats_90":
            if args.rats_memory is None or not args.rats_memory.is_dir():
                raise SystemExit("rats_90 requires the frozen failure-memory directory")
            frozen_before = _frozen_rats_state(library, args.rats_memory)
        def runner(job: dict[str, Any]) -> dict[str, Any]:
            return _rats_job(job, args, method_root, library, stop_event)

    results: list[dict[str, Any]] = []
    with _shared_api_server_bundle(args, method_root):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(runner, job): job for job in jobs}
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if int(result.get("returncode", 1)) != 0:
                        # A nonzero launcher result is infrastructure failure,
                        # not a native task failure (which upstream records
                        # with rc=0). Stop the peer and prevent queued jobs
                        # from consuming the remainder of a broken grid.
                        stop_event.set()
                    print(
                        f"[{method}] {result['job']['suite']}/{result['job']['task_id']} "
                        f"seed={result['job']['seed']} "
                        f"rc={result['returncode']} quota={result.get('quota_failure', False)}",
                        flush=True,
                    )
            except BaseException:
                # Episode launchers create private process groups.  Signal the
                # worker loops before ThreadPoolExecutor.__exit__ waits for
                # them; _run_logged will terminate each group and the outer
                # service context will then clean up the shared servers.
                stop_event.set()
                for pending in futures:
                    pending.cancel()
                raise
    _write_json(method_root / "job_results.json", results)
    if frozen_before is not None:
        frozen_after = _frozen_rats_state(library, args.rats_memory)
        frozen_audit = {
            "schema_version": 1,
            "before": frozen_before,
            "after": frozen_after,
            "unchanged": frozen_before == frozen_after,
        }
        _write_json(method_root / "frozen_artifact_audit.json", frozen_audit)
        if not frozen_audit["unchanged"]:
            _write_json(
                method_root / "FAILURES.json",
                [{"reason": "frozen_rats_artifact_mutated", **frozen_audit}],
            )
            return 1
    quota = [row for row in results if row.get("quota_failure")]
    failures = [row for row in results if int(row.get("returncode", 1)) != 0]
    if quota:
        _write_json(method_root / "ABORTED.json", {"reason": "quota", "jobs": quota})
        return 75
    if failures:
        _write_json(method_root / "FAILURES.json", failures)
        return 1
    _write_json(method_root / "COMPLETE.json", {"method": method, "jobs": len(jobs)})
    return 0


def _run_racap_method(
    method: str,
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    method_root: Path,
) -> int:
    groups: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(
        lambda: {"tasks": set(), "seeds": set()}
    )
    for job in jobs:
        group = groups[(job["cohort"], job["suite"])]
        group["tasks"].add(job["task_id"])
        group["seeds"].add(job["seed"])
    solution = (
        ROOT / "policies" / ("phase2" if method == "racap_phase2" else "phase1")
    )
    results = []
    for (cohort, suite), group in sorted(groups.items()):
        tag = f"{cohort}__{suite}"
        out = method_root / cohort
        argv = [
            str(args.python),
            str(ROOT / "scripts" / "eval_full_agent.py"),
            "--suite",
            suite,
            "--task-ids",
            *[str(value) for value in sorted(group["tasks"])],
            "--seeds",
            *[str(value) for value in sorted(group["seeds"])],
            "--workers",
            str(args.workers),
            "--turns",
            "45",
            "--max-pickplace-calls",
            "8",
            "--max-push-calls",
            "3",
            "--max-insert-calls",
            "2",
            "--max-state-calls",
            "5",
            "--max-stack-calls",
            "2",
            "--model",
            args.model,
            "--max-steps",
            "8000",
            "--episode-timeout-seconds",
            "1000",
            "--record-rollouts",
            "--tag",
            tag,
            "--output-dir",
            str(out),
        ]
        if solution:
            argv.extend(["--solution-root", str(solution)])
        if (out / tag / "run_manifest.json").exists():
            argv.append("--resume")
        env = os.environ.copy()
        card = _experience_card(args, suite=suite)
        env.update(
            {
                "PYTHONPATH": _controlled_pythonpath(),
                **_controlled_libero_env(),
                "RACAP_LLM_CACHE": "0",
                "RACAP_MODEL": args.model,
                "RACAP_VLM_MAX_CONCURRENCY": str(args.workers),
                "RACAP_SAM3_URL": _service_client_env()["SAM3_SERVICE_URL"],
                "RACAP_GRASPNET_URL": _service_client_env()["GRASPNET_SERVICE_URL"],
                "RACAP_PYROKI_URL": _service_client_env()["PYROKI_SERVICE_URL"],
                "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        if card is not None:
            env["CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"] = str(card.resolve())
        result = _run_logged(
            argv,
            cwd=ROOT,
            env=env,
            log_path=out / f"{tag}.log",
        )
        result.update({"cohort": cohort, "suite": suite})
        results.append(result)
        if result["quota_failure"] or result["returncode"] == 75:
            _write_json(method_root / "ABORTED.json", result)
            _write_json(method_root / "job_results.json", results)
            return 75
        if result["returncode"] != 0:
            _write_json(method_root / "FAILURES.json", results)
            return 1
        evidence = _racap_group_evidence(
            out / tag,
            suite=suite,
            task_ids=group["tasks"],
            seeds=group["seeds"],
        )
        _write_json(out / tag / "EVIDENCE.json", evidence)
        result["evidence"] = evidence
        if not evidence["complete"]:
            result["returncode"] = 2
            _write_json(method_root / "FAILURES.json", results)
            _write_json(method_root / "job_results.json", results)
            return 1
    _write_json(method_root / "job_results.json", results)
    _write_json(method_root / "COMPLETE.json", {"method": method, "groups": len(groups)})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"],
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=DEFAULT_MAIN_COHORTS,
        help=(
            "Registered cohorts to execute. Defaults to the three main-report "
            "cohorts; one-shot and long-horizon runs use dedicated drivers."
        ),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument("--rats-library", type=Path, default=None)
    parser.add_argument("--rats-memory", type=Path, default=None)
    parser.add_argument("--experience-card-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "controlled_comparison" / "measured"
    )
    args = parser.parse_args()
    if args.workers != 10:
        raise SystemExit("the amended protocol requires exactly ten workers")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    if args.method == "rats_90":
        if args.rats_library is None or not args.rats_library.is_file():
            raise SystemExit("rats_90 requires the frozen skill library")
        if args.rats_memory is None or not args.rats_memory.is_dir():
            raise SystemExit("rats_90 requires the frozen failure-memory directory")
        _require_frozen_rats_seal(args.rats_library, args.rats_memory)

    import_roots = _experiment_import_roots(
        args.python,
        verify_rats_registry=args.method in {"rats_base", "rats_90"},
    )

    manifest_path = HERE / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_cohorts = {str(row["cohort"]) for row in manifest["rows"]}
    unknown_cohorts = sorted(set(args.cohorts) - known_cohorts)
    if unknown_cohorts:
        raise SystemExit(f"unknown manifest cohorts: {unknown_cohorts}")
    jobs = _jobs(manifest, set(args.cohorts))
    if not jobs:
        raise SystemExit("no jobs selected")
    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    # Keep the descriptor referenced until ``main`` returns.  The process is
    # the unit of orchestration, so process exit releases the kernel lock on
    # success, exception, quota pause, and SIGTERM alike.
    _method_run_lock = _acquire_method_run_lock(method_root)
    protocol_payload = yaml.safe_load((HERE / "protocol.yaml").read_text())
    run_manifest = {
        "schema_version": 1,
        "protocol_sha256": _sha256(HERE / "protocol.yaml"),
        "task_manifest_sha256": _sha256(manifest_path),
        "method": args.method,
        "cohorts": sorted(set(args.cohorts)),
        "workers": args.workers,
        "latency_reporting": (protocol_payload.get("runtime") or {}).get(
            "latency_reporting"
        ),
        "model": args.model,
        "hosted_model_route": _public_model_route(os.environ["RACAP_VAPI_BASE"]),
        "python_import_roots": import_roots,
        "simulator_horizon": 8000,
        "episode_wall_time_seconds": 1000,
        "registered_initial_states_per_episode": 1,
        "post_action_environment_resets": 0,
        "generated_continuous_turns": EVAL_CONTINUOUS_TURNS,
        "rats_policy_self_check_repairs": EVAL_RATS_SELF_CHECK_REPAIRS,
        "shared_service_urls": _service_client_env(),
        "jobs": jobs,
        "rats_library": str(args.rats_library.resolve()) if args.rats_library else None,
        "rats_library_sha256": _sha256(args.rats_library) if args.rats_library else None,
        "rats_memory": str(args.rats_memory.resolve()) if args.rats_memory else None,
        "rats_memory_tree_sha256": (
            _tree_sha256(args.rats_memory) if args.rats_memory else None
        ),
        "experience_cards": (
            {
                family: {
                    "path": str(
                        (args.experience_card_dir / f"{args.method}_{family}.md").resolve()
                    ),
                    "sha256": _sha256(
                        args.experience_card_dir / f"{args.method}_{family}.md"
                    ),
                }
                for family in ("spatial", "goal", "object")
            }
            if args.experience_card_dir is not None
            else None
        ),
    }
    manifest_out = method_root / "run_manifest.json"
    if manifest_out.exists():
        previous_manifest = json.loads(manifest_out.read_text())
        if previous_manifest != run_manifest:
            migration = _apply_registered_manifest_migration(
                method=args.method,
                method_root=method_root,
                manifest_path=manifest_out,
                before=previous_manifest,
                after=run_manifest,
                jobs=jobs,
            )
            if migration is None:
                raise SystemExit(
                    "refusing incompatible resume: run_manifest.json differs and "
                    "no exact registered migration applies"
                )
    else:
        _write_json(manifest_out, run_manifest)
    if args.method in {"capx", "rats_base", "rats_90"}:
        return _run_generated_method(args.method, jobs, args, method_root)
    with _shared_api_server_bundle(args, method_root):
        return _run_racap_method(args.method, jobs, args, method_root)


if __name__ == "__main__":
    raise SystemExit(main())
