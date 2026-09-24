#!/usr/bin/env python3
"""Evaluator-only audit of the Robosuite Cartesian / TCP contract.

This is not a policy and its output is never delivered to a candidate.  It
compares the public Cartesian observation against live MuJoCo link/site poses
before and after one public contact grasp and one relative move.  The audit is
intended to catch embodiment-adapter errors such as stale pose caches or an
IK target and observation referring to different tool-center points.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from racap.backends.robosuite import RobosuitePrimitiveRuntime


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _named_body_pose(runtime: RobosuitePrimitiveRuntime, name: str) -> dict[str, Any] | None:
    model = runtime._sim.model
    data = runtime._sim.data
    try:
        body_id = int(model.body_name2id(name))
    except Exception:
        return None
    return {
        "name": name,
        "id": body_id,
        "world_position": np.asarray(data.xpos[body_id], dtype=float),
        "world_quat_wxyz": np.asarray(data.xquat[body_id], dtype=float),
    }


def _named_site_pose(runtime: RobosuitePrimitiveRuntime, name: str) -> dict[str, Any] | None:
    model = runtime._sim.model
    data = runtime._sim.data
    try:
        site_id = int(model.site_name2id(name))
    except Exception:
        return None
    return {
        "name": name,
        "id": site_id,
        "world_position": np.asarray(data.site_xpos[site_id], dtype=float),
        "world_rotation": np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3),
    }


def _raw_object_pose(runtime: RobosuitePrimitiveRuntime) -> Any:
    raw = runtime._env.get_observation()
    return raw.get("cube_poses")


def _snapshot(runtime: RobosuitePrimitiveRuntime, label: str) -> dict[str, Any]:
    raw = runtime._env.robosuite_env._get_observations()
    cached = np.asarray(runtime._env.gripper_link_wxyz_xyz, dtype=float)
    live_id = int(runtime._env.gripper_link_idx)
    live = np.concatenate(
        [
            np.asarray(runtime._sim.data.xquat[live_id], dtype=float),
            np.asarray(runtime._sim.data.xpos[live_id], dtype=float),
        ]
    )
    raw_eef = {
        key: value
        for key, value in raw.items()
        if "eef" in str(key).lower() and np.asarray(value).size <= 16
    }
    return {
        "label": label,
        "time": time.time(),
        "simulator_steps": int(getattr(runtime._env, "_sim_step_count", 0)),
        "public_ee_pose": runtime.ee_pose(),
        "cached_gripper_link_wxyz_xyz": cached,
        "live_gripper_link_wxyz_xyz": live,
        "cache_live_max_abs_error": float(np.max(np.abs(cached - live))),
        "raw_eef_observations": raw_eef,
        "bodies": [
            value
            for value in (
                _named_body_pose(runtime, "fixed_mount0_base"),
                _named_body_pose(runtime, "panda_hand"),
                _named_body_pose(runtime, "gripper0_right_eef"),
            )
            if value is not None
        ],
        "sites": [
            value
            for value in (
                _named_site_pose(runtime, "gripper0_grip_site"),
                _named_site_pose(runtime, "gripper0_right_grip_site"),
                _named_site_pose(runtime, "gripper0_right_eef"),
            )
            if value is not None
        ],
        "evaluator_only_cube_pose": _raw_object_pose(runtime),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--relative-dz", type=float, default=0.05)
    args = parser.parse_args()

    os.environ["CAPX_ENV_STACK"] = "robosuite"
    runtime = RobosuitePrimitiveRuntime("cube_lifting", seed=args.seed, max_steps=4000)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "scope": "evaluator_only_mechanism_neutral_adapter_calibration",
        "candidate_visible": False,
        "task": "cube_lifting",
        "seed": args.seed,
        "relative_dz_m": args.relative_dz,
        "snapshots": [],
    }
    try:
        audit["snapshots"].append(_snapshot(runtime, "initial"))
        target = runtime.localize("red cube", detector="vlm_bbox_sam3")
        audit["public_grounding"] = {
            "label": target.label,
            "pose": target.pose,
            "confidence": target.confidence,
            "metadata": target.metadata,
        }
        audit["snapshots"].append(_snapshot(runtime, "after_grounding"))

        runtime.open_gripper()
        audit["snapshots"].append(_snapshot(runtime, "after_open"))
        audit["contact_grasp_report"] = runtime.contact_grasp(
            target, strategy="top_down", lift=0.0
        )
        audit["snapshots"].append(_snapshot(runtime, "after_contact_grasp"))

        before_delta = np.asarray(runtime.ee_pose()["position"], dtype=float)
        requested = before_delta + np.asarray([0.0, 0.0, args.relative_dz], dtype=float)
        audit["relative_move"] = {
            "public_position_before": before_delta,
            "requested_public_position": requested,
        }
        runtime.delta_move(dz=args.relative_dz)
        audit["snapshots"].append(_snapshot(runtime, "after_delta_immediate"))

        # One ordinary controller hold step updates the environment's cached
        # link pose. Comparing the two snapshots diagnoses stale observation
        # state without changing the requested motion or exposing it to policy.
        runtime._env._step_once()
        audit["snapshots"].append(_snapshot(runtime, "after_delta_one_hold_step"))
        audit["evaluator_only_native_success"] = runtime.oracle.task_completed()
        audit["evaluator_only_reward"] = runtime.oracle.reward()
    finally:
        runtime.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(audit), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
