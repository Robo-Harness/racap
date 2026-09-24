#!/usr/bin/env python3
"""Run the frozen RACaP Phase 2 on Robosuite without policy adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Evaluations must use the live end-effector-link pose in the robot base frame.
ROBOSUITE_CARTESIAN_CONTRACT_ID = "live_eef_link_robot0_base"

try:
    from .run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _public_model_route,
        _python_source_tree_sha256,
        _service_client_env,
        _shared_api_server_bundle,
        _write_json,
    )
    from .run_robosuite import (
        TASKS,
        _rats_first_party_source_sha256,
        _robosuite_git_identity,
        _robosuite_process_env,
        _validate_robosuite_source,
    )
except ImportError:
    from run_libero import (  # type: ignore[no-redef]
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _public_model_route,
        _python_source_tree_sha256,
        _service_client_env,
        _shared_api_server_bundle,
        _write_json,
    )
    from run_robosuite import (  # type: ignore[no-redef]
        TASKS,
        _rats_first_party_source_sha256,
        _robosuite_git_identity,
        _robosuite_process_env,
        _validate_robosuite_source,
    )


def _frozen_candidate_sha256(root: Path) -> str:
    """Hash candidate source/artifacts while ignoring interpreter caches."""
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=ROOT / "policies" / "phase2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "robosuite",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--method-label", default="racap_phase2_zero_shot")
    parser.add_argument(
        "--protocol-note",
        default="frozen LIBERO-90 champion; no Robosuite rollout feedback",
    )
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--rats-root",
        type=Path,
        default=DEFAULT_RATS_ROOT,
        help="Registered RATS/CaP-X source used by the Robosuite adapter.",
    )
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=(DEFAULT_RATS_ROOT / "rats" / "third_party" / "robosuite"),
    )
    args = parser.parse_args()
    # Shared service lifecycle helper accepts the same namespace contract as
    # the generated-method runner.
    args.rats_root = args.rats_root.resolve()

    solution_root = args.solution_root.resolve()
    if not args.method_label.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--method-label must contain only letters, numbers, _ or -")
    method_root = (args.output_root / args.method_label).resolve()
    evaluation_root = method_root / "evaluation"
    if method_root.exists() and any(method_root.iterdir()) and not args.resume:
        raise SystemExit(f"refusing non-empty method directory: {method_root}")
    method_root.mkdir(parents=True, exist_ok=True)
    rats_root = args.rats_root
    capx_root = rats_root / "capx-baseline"
    robosuite_root = args.robosuite_root.resolve()
    execution_env = _robosuite_process_env(
        robosuite_root=robosuite_root,
        capx_root=capx_root,
        rats_root=rats_root,
    )
    execution_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(robosuite_root),
            str(ROOT),
            str(capx_root),
            str(rats_root),
            execution_env.get("PYTHONPATH", ""),
        ]
    )
    execution_env.update(_service_client_env())
    execution_env["RACAP_MODEL"] = str(args.model)
    imports = _validate_robosuite_source(
        args.python, robosuite_root, rats_root=rats_root, execution_env=execution_env
    )
    robosuite_git = _robosuite_git_identity(robosuite_root)
    if not robosuite_git["tracked_clean"]:
        raise SystemExit("registered Robosuite source has tracked modifications")
    frozen_before = _frozen_candidate_sha256(solution_root)
    manifest = {
        "schema_version": 1,
        "method": str(args.method_label),
        "tasks": list(args.tasks or TASKS),
        "seeds": list(dict.fromkeys(int(seed) for seed in args.seeds)),
        "episodes": len(args.tasks or TASKS) * len(set(args.seeds)),
        "workers": min(max(1, int(args.workers)), 5),
        "model": str(args.model),
        "protocol_note": str(args.protocol_note),
        "hosted_model_route": _public_model_route(os.environ["RACAP_VAPI_BASE"]),
        "solution_root": str(solution_root),
        "solution_tree_sha256": frozen_before,
        "policy_artifact_updates": False,
        "privileged_runtime": False,
        "robosuite_cartesian_contract": {
            "id": ROBOSUITE_CARTESIAN_CONTRACT_ID,
            "backend_source": str((ROOT / "racap" / "backends" / "robosuite.py").resolve()),
            "backend_source_sha256": _file_sha256(
                ROOT / "racap" / "backends" / "robosuite.py"
            ),
            "calibration_pre_contact_pose_error_cm": 29.7,
            "calibration_post_contact_pose_error_cm": 0.6,
            "requested_delta_z_cm": 5.0,
            "observed_delta_z_cm": 4.65,
        },
        "robosuite_source": str(robosuite_root),
        "robosuite_git": robosuite_git,
        "rats_source": str(rats_root),
        # Upstream Robosuite wrappers resolve controller JSONs relative to the
        # RATS checkout.  This is an executor resource path, not candidate
        # state; pin it explicitly so launches do not depend on the caller's
        # current directory.
        "executor_working_directory": str(rats_root),
        "rats_first_party_python_source_sha256": _rats_first_party_source_sha256(
            rats_root / "rats"
        ),
        "capx_python_source_sha256": _python_source_tree_sha256(capx_root / "capx"),
        "robosuite_imports": imports,
        "generated_method_service_urls": _service_client_env(),
        "service_lifecycle": "one_parent_owned_fail_closed_bundle_for_all_tasks",
    }
    manifest_path = method_root / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("refusing incompatible RACaP Robosuite resume")
    else:
        _write_json(manifest_path, manifest)

    command = [
        str(args.python),
        str(ROOT / "scripts" / "eval_robosuite_agent.py"),
        "--solution-root",
        str(solution_root),
        "--output-dir",
        str(evaluation_root),
        "--workers",
        str(min(max(1, int(args.workers)), 5)),
        "--model",
        str(args.model),
        "--method-label",
        str(args.method_label),
        "--protocol-note",
        str(args.protocol_note),
        "--record-rollouts",
    ]
    if args.tasks:
        command.extend(["--tasks", *args.tasks])
    command.extend(["--seeds", *(str(seed) for seed in args.seeds)])
    if args.resume:
        command.append("--resume")

    returncode = 1
    with _shared_api_server_bundle(
        args,
        method_root,
        owner="run_robosuite_racap_method_parent",
    ):
        log_path = method_root / "run.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("# argv: " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.run(
                command,
                cwd=rats_root,
                env=execution_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = int(process.returncode)

    frozen_after = _frozen_candidate_sha256(solution_root)
    _write_json(
        method_root / "frozen_artifact_audit.json",
        {
            "before_sha256": frozen_before,
            "after_sha256": frozen_after,
            "unchanged": frozen_before == frozen_after,
        },
    )
    if frozen_before != frozen_after:
        raise RuntimeError("frozen RACaP candidate changed during Robosuite evaluation")
    if returncode == 75:
        _write_json(method_root / "PAUSED.json", {"reason": "quota_or_provider"})
        return 75
    if returncode != 0:
        _write_json(method_root / "FAILURES.json", {"returncode": returncode})
        return returncode
    if not (evaluation_root / "COMPLETE.json").is_file():
        raise RuntimeError("RACaP evaluator returned zero without COMPLETE.json")
    complete = json.loads((evaluation_root / "COMPLETE.json").read_text(encoding="utf-8"))
    _write_json(method_root / "COMPLETE.json", {"method": manifest["method"], **complete})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
