#!/usr/bin/env python3
"""Run the seven-task Robosuite transfer diagnostic one method at a time."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _generated_episode_evidence,
        _jsonl_episode_keys,
        _jsonl_rows,
        _nonempty_files,
        _public_model_route,
        _python_source_tree_sha256,
        _remap_api_server_ports,
        _require_frozen_rats_seal,
        _run_logged,
        _service_client_env,
        _service_process_env,
        _shared_api_server_bundle,
        _sha256,
        _write_json,
    )
except ImportError:  # direct ``python path/to/script.py`` execution
    from run_libero import (  # type: ignore[no-redef]
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _generated_episode_evidence,
        _jsonl_episode_keys,
        _jsonl_rows,
        _nonempty_files,
        _public_model_route,
        _python_source_tree_sha256,
        _remap_api_server_ports,
        _require_frozen_rats_seal,
        _run_logged,
        _service_client_env,
        _service_process_env,
        _shared_api_server_bundle,
        _sha256,
        _write_json,
    )

TASKS = (
    "cube_lifting",
    "cube_restack",
    "cube_stack",
    "nut_assembly",
    "spill_wipe",
    "two_arm_handover",
    "two_arm_lift",
)

ROBOSUITE_DEVELOPMENT_SEEDS = (100, 101, 102)
ROBOSUITE_DEVELOPMENT_WORKERS = 3
ROBOSUITE_EVALUATION_WORKERS = 5
SINGLE_RESET_SHIM_ROOT = (
    ROOT
    / "experiments"
    / "controlled_comparison"
    / "runtime_shims"
    / "single_reset"
)
TRANSPORT_RETRY_SHIM_ROOT = (
    ROOT
    / "experiments"
    / "controlled_comparison"
    / "runtime_shims"
    / "transport_retry"
)


def _executor_source_audit(rats_root: Path, *, capture_phase: str) -> dict[str, Any]:
    """Hash the exact dirty-but-pinned executor used by both generated methods."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=rats_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=rats_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("cannot attest the registered RATS executor source")
    return {
        "schema_version": 1,
        "capture_phase": capture_phase,
        "rats_root": str(rats_root.resolve()),
        "git_commit": commit.stdout.strip(),
        "git_status_porcelain": status.stdout,
        "git_status_sha256": _sha256_text(status.stdout),
        "capx_python_source_sha256": _python_source_tree_sha256(
            rats_root / "capx-baseline" / "capx"
        ),
        # Robosuite is attested separately by its git tree.  Excluding
        # ``rats/third_party`` avoids rehashing that large checkout before
        # every development sweep while still covering all mutable RATS code.
        "rats_first_party_python_source_sha256": _rats_first_party_source_sha256(
            rats_root / "rats"
        ),
    }


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rats_first_party_source_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.py")):
        relative = item.relative_to(path)
        if "__pycache__" in relative.parts or "third_party" in relative.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_worker_protocol(
    *,
    method: str,
    workers: int,
    seeds: tuple[int, ...],
    services_already_running: bool,
) -> str:
    """Fail closed on concurrency drift between development and evaluation.

    The generated executor's ``workers`` parameter is a process-level ceiling.
    With five evaluation seeds, evaluation uses five workers explicitly.
    Domain evolution instead uses exactly three
    development seeds and three workers while sharing the parent service
    bundle.  These are different registered phases and must not be collapsed
    into one unconditional worker assertion.
    """

    development = (
        method == "rats_rs_evolved"
        and services_already_running
        and seeds == ROBOSUITE_DEVELOPMENT_SEEDS
    )
    if development:
        if workers != ROBOSUITE_DEVELOPMENT_WORKERS:
            raise SystemExit(
                "registered RATS-RS development requires exactly three workers"
            )
        return "matched_domain_development"
    if services_already_running:
        raise SystemExit(
            "parent-owned services are registered only for the RATS-RS "
            "three-seed development sweep"
        )
    if workers != ROBOSUITE_EVALUATION_WORKERS:
        raise SystemExit(
            "the sealed Robosuite evaluation requires exactly five workers"
        )
    return "sealed_evaluation"


def _archive_incomplete_task(method_root: Path, task_root: Path, task: str) -> Path | None:
    """Move a partially executed task outside the scored method tree.

    The upstream launcher cannot resume an arbitrary subset of concurrent
    trials without risking duplicate simulator resets.  Completed task blocks
    remain untouched; an interrupted block of at most five trials is retained
    as explicitly unscored evidence and rerun cleanly after recharge.
    """

    if not task_root.exists() or not any(task_root.iterdir()):
        return None
    if (task_root / "COMPLETE.json").is_file():
        return None
    archive_root = method_root.parent / f"{method_root.name}_invalidated_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    candidate = archive_root / f"{task}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"{task}_{stamp}_{suffix:02d}"
        suffix += 1
    shutil.move(str(task_root), str(candidate))
    _write_json(
        candidate / "NOT_SCORED.json",
        {
            "schema_version": 1,
            "reason": "interrupted task block rerun to prevent duplicate resets",
            "source_method_root": str(method_root.resolve()),
            "task": task,
        },
    )
    return candidate


def _write_frozen_library_audit(
    method_root: Path, library: Path | None, before_sha256: str | None
) -> None:
    if library is None or before_sha256 is None:
        return
    after_sha256 = _sha256(library)
    _write_json(
        method_root / "frozen_artifact_audit.json",
        {
            "artifact": str(library.resolve()),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "unchanged": before_sha256 == after_sha256,
        },
    )
    if before_sha256 != after_sha256:
        raise RuntimeError("Robosuite execution mutated the sealed RATS library")


def _validate_robosuite_source(
    python: Path,
    source: Path,
    *,
    rats_root: Path = DEFAULT_RATS_ROOT,
    execution_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Prove that the executor imports the registered Robosuite checkout.

    When ``execution_env`` is supplied this probes the *exact* environment
    passed to ``capx.envs.launch``.  A lightweight preflight that constructs a
    different ``PYTHONPATH`` can pass while the real launcher resolves another
    editable Robosuite checkout, which would invalidate the comparison.
    """

    source = source.resolve()
    required = (
        source / "robosuite" / "__init__.py",
        source
        / "robosuite"
        / "controllers"
        / "composite"
        / "composite_controller_factory.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Robosuite transfer source is not initialized; missing: "
            + ", ".join(missing)
        )
    rats_root = rats_root.resolve()
    capx_root = rats_root / "capx-baseline"
    pythonpath = os.pathsep.join(
        [str(source), str(capx_root.resolve()), str(rats_root)]
    )
    probe = (
        "import json, pathlib, robosuite; "
        "from robosuite.controllers.composite.composite_controller_factory "
        "import load_composite_controller_config; "
        "print('RACAP_ROBOSUITE_SOURCE=' + json.dumps({"
        "'robosuite': str(pathlib.Path(robosuite.__file__).resolve()), "
        "'composite_loader': str(pathlib.Path(load_composite_controller_config.__code__.co_filename).resolve())}))"
    )
    env = execution_env.copy() if execution_env is not None else os.environ.copy()
    if execution_env is None:
        env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=capx_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )
    marker = "RACAP_ROBOSUITE_SOURCE="
    payload = next(
        (line[len(marker) :] for line in result.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if result.returncode != 0 or payload is None:
        raise RuntimeError(
            "failed to import the registered Robosuite executor source: "
            f"returncode={result.returncode}; output={result.stdout[-2000:]}"
        )
    observed = json.loads(payload)
    for name, value in observed.items():
        if source not in Path(value).resolve().parents:
            raise RuntimeError(f"{name} silently resolved outside {source}: {value}")
    return {str(key): str(value) for key, value in observed.items()}


def _robosuite_git_identity(source: Path) -> dict[str, Any]:
    """Content-stable identity for the registered source checkout.

    A generic recursive hash includes interpreter caches and transient files.
    Git's commit/tree identity plus a tracked-files cleanliness check captures
    the code that can affect execution while remaining stable across imports.
    """

    def query(*argv: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source), *argv],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"failed Robosuite git provenance query {' '.join(argv)}: "
                f"{result.stderr[-1000:]}"
            )
        return result.stdout.strip()

    tracked_status = query("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "commit": query("rev-parse", "HEAD"),
        "tree": query("rev-parse", "HEAD^{tree}"),
        "tracked_status_porcelain": tracked_status,
        "tracked_clean": not tracked_status,
    }


def _robosuite_process_env(
    *, robosuite_root: Path, capx_root: Path, rats_root: Path
) -> dict[str, str]:
    """Build the one authoritative environment for probe and execution.

    ``_service_process_env`` intentionally supplies the LIBERO comparison
    ``PYTHONPATH`` for the shared perception services.  Robosuite is a separate
    diagnostic and must put its registered checkout first.  Assigning this
    field after merging the shared-service variables prevents the latter from
    silently replacing it.
    """

    env = os.environ.copy()
    env.update(_service_process_env())
    python_path = [
        str(SINGLE_RESET_SHIM_ROOT.resolve()),
        str(TRANSPORT_RETRY_SHIM_ROOT.resolve()),
        str(robosuite_root.resolve()),
        str(capx_root.resolve()),
        str(rats_root.resolve()),
    ]
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited:
        python_path.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["RACAP_SINGLE_RESET_PROTOCOL"] = "1"
    env["RACAP_REGISTERED_TRANSPORT_RETRY"] = "1"
    env["RACAP_REGISTERED_VAPI_BASE"] = env.get("RACAP_VAPI_BASE", "").rstrip("/")
    env["RACAP_TRANSPORT_MAX_ATTEMPTS"] = "3"
    # These variables also cover the repository-local patched executor.  The
    # sitecustomize shim above enforces the same rule for older pinned source
    # snapshots whose constants predate environment-variable support.
    env["CAPX_MAX_TRIAL_RETRIES"] = "1"
    env["RATS_MAX_TRIAL_RETRIES"] = "1"
    return env


def _validate_registered_transport_retry(
    python: Path, execution_env: dict[str, str]
) -> dict[str, Any]:
    """Prove the exact child interpreter installs the VAPI-only retry hook."""

    probe = r"""
import json
import usercustomize
print(json.dumps({
    "usercustomize": usercustomize.__file__,
    "hook_installed": bool(usercustomize.RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED),
    "maximum_attempts": int(usercustomize.RACAP_REGISTERED_TRANSPORT_MAX_ATTEMPTS),
}))
"""
    result = subprocess.run(
        [str(python), "-c", probe],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=execution_env,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            "failed to audit registered VAPI transport retry:\n"
            + result.stderr[-2000:]
        )
    try:
        audit = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            "transport retry audit returned invalid output: "
            + result.stdout[-2000:]
        ) from exc
    expected = (TRANSPORT_RETRY_SHIM_ROOT / "usercustomize.py").resolve()
    if Path(str(audit.get("usercustomize", ""))).resolve() != expected:
        raise RuntimeError("transport retry hook resolved from an unregistered source")
    if audit.get("hook_installed") is not True:
        raise RuntimeError("registered VAPI transport retry hook is not installed")
    if int(audit.get("maximum_attempts") or 0) != 3:
        raise RuntimeError("registered VAPI transport retry budget drifted")
    audit.update(
        {
            "shim": str(expected),
            "shim_sha256": _sha256(expected),
            "retryable_failures": ["requests.Timeout", "requests.ConnectionError"],
            "request_invariant": True,
            "environment_resets": 0,
        }
    )
    return audit


def _validate_single_reset_runner(
    python: Path, execution_env: dict[str, str]
) -> dict[str, Any]:
    """Prove the exact child environment installs the single-reset hook.

    Importing the full runner in a separate audit process eagerly imports the
    robotics stack and can itself exhaust worker memory.  The hook is tested
    independently against a synthetic upstream module; scored evidence then
    verifies exactly one observed reset for every registered episode.
    """

    probe = r"""
import json
import os
import sitecustomize
print(json.dumps({
    "sitecustomize": sitecustomize.__file__,
    "hook_installed": bool(sitecustomize.RACAP_SINGLE_RESET_HOOK_INSTALLED),
    "target_modules": sorted(sitecustomize.TARGET_MODULES),
    "protocol_environment": os.environ.get("RACAP_SINGLE_RESET_PROTOCOL"),
}))
"""
    result = subprocess.run(
        [str(python), "-c", probe],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=execution_env,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            "failed to audit generated-code single-reset runner:\n"
            + result.stderr[-2000:]
        )
    try:
        audit = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            "single-reset runner audit returned invalid output: "
            + result.stdout[-2000:]
        ) from exc
    expected_sitecustomize = (SINGLE_RESET_SHIM_ROOT / "sitecustomize.py").resolve()
    if Path(str(audit.get("sitecustomize", ""))).resolve() != expected_sitecustomize:
        raise RuntimeError("single-reset hook resolved from an unregistered source")
    if audit.get("hook_installed") is not True:
        raise RuntimeError("single-reset import hook is not installed")
    if set(audit.get("target_modules", [])) != {
        "capx.envs.runner",
        "rats.envs.runner",
    }:
        raise RuntimeError("single-reset hook does not cover both generated-code runners")
    if audit.get("protocol_environment") != "1":
        raise RuntimeError("single-reset child environment was not propagated")
    audit["shim"] = str((SINGLE_RESET_SHIM_ROOT / "sitecustomize.py").resolve())
    audit["shim_sha256"] = _sha256(SINGLE_RESET_SHIM_ROOT / "sitecustomize.py")
    return audit


def _configure_robosuite_eval(
    template: dict[str, Any],
    *,
    method: str,
    output_dir: Path,
    model: str,
    rats_library: Path | None,
    workers: int,
    trials: int = 5,
) -> dict[str, Any]:
    """Resolve both rows from the same no-RATS upstream template.

    The RATS-90 row must differ from CaP-X only by explicit retrieval of the
    sealed comparison artifact. Starting from the upstream ``rats_iter050``
    YAML would happen to work after path replacement, but it would also inherit
    undocumented settings from a different 50-round library and make the
    controlled contrast needlessly ambiguous.
    """

    config = _remap_api_server_ports(copy.deepcopy(template))
    config["trials"] = trials
    config["num_workers"] = min(workers, trials)
    config["output_dir"] = str(output_dir)
    # A single fail-closed parent-owned bundle is shared across all seven
    # tasks.  Leaving these entries in the child config cold-starts three large
    # services per task and the upstream launcher continues after readiness
    # warnings, allowing trials to begin with missing perception/control APIs.
    config["api_servers"] = []
    if method == "rats_rs_evolved":
        # The historical nut-assembly diagnostic template is marked
        # ``privileged: true`` even though its generated programs were audited
        # not to consume object-pose fields.  The new evolved comparison has a
        # stricter contract: both RATS-RS and RACaP-RS construct every task in
        # non-privileged mode, so candidate code can only use public RGB-D and
        # robot state.
        config["env"]["cfg"]["privileged"] = False
    if method in {"rats_90", "rats_rs_evolved"}:
        if rats_library is None or not rats_library.is_file():
            raise FileNotFoundError(f"{method} requires a sealed external skill library")
        env_cfg = config["env"]["cfg"]
        env_cfg.update(
            {
                "external_skill_library_path": str(rats_library.resolve()),
                "external_skill_library_mode": "planner",
                "external_skill_library_max_skills": 0,
                "external_skill_library_include_code": True,
                "external_skill_library_include_primitives": False,
                "external_skill_planner_model": model,
                "external_skill_planner_max_selected": 6,
                "external_skill_planner_include_code": False,
                "external_skill_policy_include_code": True,
            }
        )
        config["save_skill_planner_prompts"] = True
    return config


def _robosuite_task_evidence(
    task_root: Path,
    task: str,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Require one complete, paired artifact bundle for every requested seed."""

    expected = {f"robosuite/{task}/seed{seed}" for seed in seeds}
    base = _generated_episode_evidence(
        task_root,
        "capx",
        expected_episode_keys=expected,
        maximum_registered_resets=1,
    )
    llm_files = _nonempty_files(task_root, "llm_calls.jsonl")
    native_files = _nonempty_files(task_root, "native_states.jsonl")
    reset_files = _nonempty_files(task_root, "sim_episodes.jsonl")
    llm_keys = _jsonl_episode_keys(llm_files)
    native_keys = _jsonl_episode_keys(native_files)
    reset_keys = _jsonl_episode_keys(reset_files)
    reset_rows = _jsonl_rows(reset_files)
    transport_files = _nonempty_files(task_root, "transport_attempts.jsonl")
    transport_rows = _jsonl_rows(transport_files)
    trace_files = _nonempty_files(task_root, "all_responses.json")
    trial_summaries = [
        path
        for path in _nonempty_files(task_root, "summary.txt")
        if path.parent.name.startswith("trial_")
    ]
    videos = _nonempty_files(task_root, "*.mp4")
    missing = list(base["missing"])
    cardinality = {
        "human_traces": len(trace_files),
        "trial_summaries": len(trial_summaries),
        "videos": len(videos),
    }
    for name, count in cardinality.items():
        if count < len(seeds):
            missing.append(f"{name}_expected_{len(seeds)}_found_{count}")
    missing_llm = sorted(expected - llm_keys)
    missing_native = sorted(expected - native_keys)
    missing_resets = sorted(expected - reset_keys)
    unexpected_resets = sorted(reset_keys - expected)
    wrong_reset_seeds: dict[str, list[int | None]] = {}
    for seed in seeds:
        key = f"robosuite/{task}/seed{seed}"
        observed = [
            int(row["seed"]) if row.get("seed") is not None else None
            for row in reset_rows
            if str(row.get("episode_key") or row.get("key") or "").strip() == key
        ]
        if observed != [seed]:
            wrong_reset_seeds[key] = observed
    if missing_llm:
        missing.append("paired_llm_telemetry")
    if missing_native:
        missing.append("paired_native_telemetry")
    if missing_resets:
        missing.append("paired_reset_telemetry")
    if unexpected_resets:
        missing.append("unexpected_reset_telemetry")
    if wrong_reset_seeds:
        missing.append("wrong_or_duplicate_reset_seed")
    exhausted_transport = [
        row for row in transport_rows if row.get("outcome") == "exhausted"
    ]
    if exhausted_transport:
        missing.append("registered_vapi_transport_retry_exhausted")
    return {
        **base,
        "schema_version": 1,
        "task": task,
        "complete": not missing,
        "missing": sorted(set(missing)),
        "expected_episode_keys": sorted(expected),
        "missing_llm_telemetry": missing_llm,
        "missing_native_telemetry": missing_native,
        "missing_reset_telemetry": missing_resets,
        "unexpected_reset_telemetry": unexpected_resets,
        "wrong_reset_seeds": wrong_reset_seeds,
        "transport_retry": {
            "records": len(transport_rows),
            "transient_failures": sum(
                row.get("outcome") == "transport_error" for row in transport_rows
            ),
            "recoveries": sum(
                row.get("outcome") == "recovered" for row in transport_rows
            ),
            "exhaustions": len(exhausted_transport),
            "environment_resets": sum(
                bool(row.get("environment_reset")) for row in transport_rows
            ),
        },
        "artifact_cardinality": cardinality,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=["capx", "rats_90", "rats_rs_evolved"]
    )
    parser.add_argument(
        "--method-label",
        default=None,
        help="Artifact directory label; execution semantics still come from --method.",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=Path(
            os.environ.get(
                "RACAP_ROBOSUITE_ROOT",
                DEFAULT_RATS_ROOT / "rats" / "third_party" / "robosuite",
            )
        ),
        help="Initialized upstream Robosuite checkout used by both methods.",
    )
    parser.add_argument("--rats-library", type=Path, default=None)
    parser.add_argument(
        "--services-already-running",
        action="store_true",
        help="Use the parent evolution process's fail-closed API-service bundle.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "robosuite",
    )
    args = parser.parse_args()
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    if args.method in {"rats_90", "rats_rs_evolved"} and (
        args.rats_library is None or not args.rats_library.is_file()
    ):
        raise SystemExit(f"{args.method} requires --rats-library pointing to a frozen artifact")
    if args.method in {"rats_90", "rats_rs_evolved"}:
        _require_frozen_rats_seal(args.rats_library)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or tuple(sorted(seeds)) != seeds:
        raise SystemExit("--seeds must be a non-empty, sorted, duplicate-free list")
    if any(right != left + 1 for left, right in zip(seeds, seeds[1:])):
        raise SystemExit("the CaP-X trial runner requires contiguous --seeds")
    protocol_phase = _validate_worker_protocol(
        method=args.method,
        workers=args.workers,
        seeds=seeds,
        services_already_running=args.services_already_running,
    )
    tasks = tuple(args.tasks)
    label = args.method_label or args.method
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise SystemExit("--method-label may contain only letters, digits, dot, dash, underscore")

    capx_root = args.rats_root / "capx-baseline"
    robosuite_root = args.robosuite_root.resolve()
    execution_env = _robosuite_process_env(
        robosuite_root=robosuite_root,
        capx_root=capx_root,
        rats_root=args.rats_root,
    )
    robosuite_imports = _validate_robosuite_source(
        args.python,
        robosuite_root,
        rats_root=args.rats_root,
        execution_env=execution_env,
    )
    single_reset_runner_audit = _validate_single_reset_runner(
        args.python, execution_env
    )
    transport_retry_audit = _validate_registered_transport_retry(
        args.python, execution_env
    )
    robosuite_git = _robosuite_git_identity(robosuite_root)
    if not robosuite_git["tracked_clean"]:
        raise SystemExit("registered Robosuite source has tracked modifications")
    config_root = capx_root / "env_configs" / "robosuite_7task_10trial_compare"
    method_root = args.output_root / label
    method_root.mkdir(parents=True, exist_ok=True)
    executor_audit = _executor_source_audit(
        args.rats_root, capture_phase="before_first_registered_reset"
    )
    executor_audit_path = method_root / "executor_source_audit.json"
    if executor_audit_path.exists():
        recorded_executor_audit = json.loads(
            executor_audit_path.read_text(encoding="utf-8")
        )
        comparable_fields = (
            "rats_root",
            "git_commit",
            "git_status_sha256",
            "capx_python_source_sha256",
            "rats_first_party_python_source_sha256",
        )
        if any(
            recorded_executor_audit.get(field) != executor_audit.get(field)
            for field in comparable_fields
        ):
            raise SystemExit("refusing changed generated-method executor source")
    else:
        _write_json(executor_audit_path, executor_audit)
    transport_audit_path = method_root / "transport_protocol_audit.json"
    if transport_audit_path.exists() and json.loads(
        transport_audit_path.read_text(encoding="utf-8")
    ) != transport_retry_audit:
        raise SystemExit("refusing incompatible registered transport retry resume")
    if not transport_audit_path.exists():
        _write_json(transport_audit_path, transport_retry_audit)
    frozen_library_sha256 = _sha256(args.rats_library) if args.rats_library else None
    manifest = {
        "schema_version": 1,
        "method": label,
        "execution_semantics": args.method,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "workers": args.workers,
        "effective_trial_workers": min(args.workers, len(seeds)),
        "protocol_phase": protocol_phase,
        "model": args.model,
        "hosted_model_route": _public_model_route(os.environ["RACAP_VAPI_BASE"]),
        "registered_initial_states_per_episode": 1,
        "post_action_environment_resets": 0,
        "timeout_retry_resets": 0,
        "single_reset_runner_audit": single_reset_runner_audit,
        "privileged_runtime": False if args.method == "rats_rs_evolved" else "template_defined",
        "generated_method_service_urls": _service_client_env(),
        "service_lifecycle": (
            "one_evolution_parent_owned_fail_closed_bundle"
            if args.services_already_running
            else "one_method_parent_owned_fail_closed_bundle_for_all_tasks"
        ),
        "rats_library": str(args.rats_library.resolve()) if args.rats_library else None,
        "rats_library_sha256": _sha256(args.rats_library) if args.rats_library else None,
        "common_config_template": "*_no_rats.yaml",
        "rats_90_configuration_delta": (
            "sealed external skill retrieval only"
            if args.method in {"rats_90", "rats_rs_evolved"}
            else None
        ),
        "robosuite_source": str(robosuite_root),
        "robosuite_source_git_tree": robosuite_git["tree"],
        "robosuite_git": robosuite_git,
        "robosuite_imports": robosuite_imports,
    }
    manifest_path = method_root / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("refusing incompatible Robosuite resume")
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)

    aborted = method_root / "ABORTED.json"
    if aborted.is_file():
        archive_root = method_root.parent / f"{method_root.name}_invalidated_attempts"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        destination = archive_root / f"ABORTED_{stamp}.json"
        suffix = 1
        while destination.exists():
            destination = archive_root / f"ABORTED_{stamp}_{suffix:02d}.json"
            suffix += 1
        shutil.move(str(aborted), str(destination))

    results = []
    service_context = (
        contextlib.nullcontext()
        if args.services_already_running
        else _shared_api_server_bundle(
            args,
            method_root,
            owner="run_robosuite_method_parent",
        )
    )
    with service_context:
        for task in tasks:
            task_root = method_root / task
            complete = task_root / "COMPLETE.json"
            if complete.exists():
                results.append({"task": task, "returncode": 0, "skipped": True})
                continue
            _archive_incomplete_task(method_root, task_root, task)
            task_root.mkdir(parents=True, exist_ok=True)
            source_config = config_root / f"{task}_no_rats.yaml"
            config = _configure_robosuite_eval(
                yaml.safe_load(source_config.read_text(encoding="utf-8")),
                method=args.method,
                output_dir=task_root / "artifacts",
                model=args.model,
                rats_library=args.rats_library,
                workers=args.workers,
                trials=len(seeds),
            )
            config_path = task_root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            env = execution_env.copy()
            base = env.get("RACAP_VAPI_BASE", "").rstrip("/")
            env.update(
                {
                    "OPENAI_API_KEY": env.get("RACAP_VAPI_KEY", ""),
                    "CAPX_EPISODE_KEY": f"robosuite/{task}",
                    "CAPX_LLM_TELEMETRY_PATH": str(task_root / "llm_calls.jsonl"),
                    "CAPX_SIM_EPISODE_TELEMETRY_PATH": str(
                        task_root / "sim_episodes.jsonl"
                    ),
                    "CAPX_NATIVE_TELEMETRY_PATH": str(
                        task_root / "native_states.jsonl"
                    ),
                    "RACAP_TRANSPORT_RETRY_LOG": str(
                        task_root / "transport_attempts.jsonl"
                    ),
                    "CAPX_RUNTIME_VLM_URL": f"{base}/chat/completions",
                    "CAPX_RUNTIME_VLM_MODEL": args.model,
                    "CAPX_RUNTIME_VLM_KEY": env.get("RACAP_VAPI_KEY", ""),
                    "CAPX_SEED_OFFSET": str(seeds[0] - 1),
                    "RACAP_VLM_MAX_CONCURRENCY": str(args.workers),
                    "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
                    "CUDA_VISIBLE_DEVICES": "0",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
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
                str(len(seeds)),
                "--num-workers",
                str(min(args.workers, len(seeds))),
                "--record-video",
                "True",
                "--output-dir",
                str(task_root / "artifacts"),
            ]
            result = _run_logged(
                argv,
                cwd=capx_root,
                env=env,
                log_path=task_root / "run.log",
            )
            result["task"] = task
            results.append(result)
            _write_json(method_root / "job_results.json", results)
            if result["quota_failure"] or result["returncode"] == 75:
                _write_frozen_library_audit(
                    method_root, args.rats_library, frozen_library_sha256
                )
                _write_json(method_root / "ABORTED.json", result)
                return 75
            if result["returncode"] != 0:
                _write_frozen_library_audit(
                    method_root, args.rats_library, frozen_library_sha256
                )
                _write_json(method_root / "FAILURES.json", results)
                return 1
            evidence = _robosuite_task_evidence(task_root, task, seeds=seeds)
            _write_json(task_root / "EVIDENCE.json", evidence)
            result["evidence"] = evidence
            _write_json(method_root / "job_results.json", results)
            if not evidence["complete"]:
                result["returncode"] = 2
                _write_frozen_library_audit(
                    method_root, args.rats_library, frozen_library_sha256
                )
                _write_json(method_root / "FAILURES.json", results)
                _write_json(method_root / "job_results.json", results)
                return 1
            _write_json(complete, result)
    _write_frozen_library_audit(method_root, args.rats_library, frozen_library_sha256)
    _write_json(
        method_root / "COMPLETE.json",
        {
            "method": label,
            "execution_semantics": args.method,
            "episodes": len(tasks) * len(seeds),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
