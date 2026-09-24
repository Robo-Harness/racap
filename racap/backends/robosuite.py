"""Non-privileged Robosuite backend for RACaP transfer experiments.

This module is deliberately an *embodiment adapter*, not a Robosuite policy.
It maps the public RGB-D, Cartesian, IK, joint-control, and gripper interfaces
already used by the controlled CaP-X/RATS executor onto RACaP's runtime
contract.  It contains no success-conditioned action sequence and never
returns Robosuite reward or ``task_completed`` to policy code.

The frozen LIBERO champion can therefore be evaluated without modification.
It will use only the capabilities it already knows.  An evolved controller may
also discover the geometry-neutral multi-arm primitives exposed here; the
strategy for combining them remains candidate-owned code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from racap.backends.libero import LiberoPrimitiveRuntime, TOP_DOWN_QUAT


@dataclass(frozen=True)
class RobosuiteTaskSpec:
    """Public task registration needed to construct one benchmark episode."""

    env_name: str
    instruction: str
    bimanual: bool = False
    handover: bool = False


TASK_SPECS: dict[str, RobosuiteTaskSpec] = {
    "cube_lifting": RobosuiteTaskSpec(
        "franka_robosuite_cube_lift_low_level",
        "Pick up the red cube and lift it.",
    ),
    "cube_restack": RobosuiteTaskSpec(
        "franka_robosuite_cubes_restack_low_level",
        "Place the red cube on top of the green cube and then open the gripper.",
    ),
    "cube_stack": RobosuiteTaskSpec(
        "franka_robosuite_cubes_low_level",
        "Place the red cube on top of the green cube and then open the gripper.",
    ),
    "nut_assembly": RobosuiteTaskSpec(
        "franka_robosuite_nut_assembly_low_level_visual",
        "Grasp and insert the brown square nut onto the brown square block. "
        "The extruded handle and hollow centre are visible parts of the same rigid nut.",
    ),
    "spill_wipe": RobosuiteTaskSpec(
        "franka_robosuite_spill_wipe_low_level",
        "Wipe up the brown spill. A sponge is already attached to the end effector.",
    ),
    "two_arm_handover": RobosuiteTaskSpec(
        "two_arm_handover_robosuite",
        "Arm 0 should pick up the hammer and hand it to Arm 1. Arm 1 must grasp "
        "the hammer handle; keep the hammer above the central gap during transfer.",
        bimanual=True,
        handover=True,
    ),
    "two_arm_lift": RobosuiteTaskSpec(
        "two_arm_lift_robosuite",
        "Coordinate both arms to lift the pot: Arm 0 grasps green handle 0 and "
        "Arm 1 grasps blue handle 1, then both arms lift to the same height.",
        bimanual=True,
    ),
}


class RobosuiteOracle:
    """Evaluator-only native state; never passed through :class:`PublicRuntime`."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def task_completed(self) -> bool:
        return bool(self._env.task_completed())

    def reward(self) -> float:
        return float(self._env.compute_reward())


class RobosuitePrimitiveRuntime(LiberoPrimitiveRuntime):
    """RACaP runtime backed by the registered seven-task Robosuite executor.

    The large perception and single-arm transport implementation is inherited
    from ``LiberoPrimitiveRuntime`` because it already consumes only calibrated
    RGB-D and public robot state.  This subclass replaces construction, camera
    aliases, evaluator telemetry, and Cartesian-to-joint actuation.  In
    particular it does *not* expose the simulator handle, object poses, reward,
    or completion predicate to the controller.
    """

    suite_name = "robosuite_transfer"

    def __init__(
        self,
        task: str,
        *,
        seed: int = 0,
        max_steps: int = 4000,
    ) -> None:
        if task not in TASK_SPECS:
            raise KeyError(f"unknown registered Robosuite task {task!r}")
        # RATs registers simulator families at import time.  Restricting the
        # stack avoids importing LIBERO into the Robosuite interpreter.
        # The shared API launcher is also used by LIBERO and may leave this
        # variable set to ``libero`` in the worker environment. Registration
        # is selected at import time, so ``setdefault`` would silently register
        # no Robosuite environments. Each episode worker is an isolated process
        # dedicated to this backend, therefore select its stack explicitly.
        os.environ["CAPX_ENV_STACK"] = "robosuite"
        from rats.envs.base import get_env
        import rats.envs.simulators  # noqa: F401  (registration side effect)
        from rats.envs.base import list_envs
        from rats.integrations.franka.control_reduced_skill_library import (
            FrankaControlApiReducedSkillLibrary,
        )

        if TASK_SPECS[task].env_name not in list_envs():
            raise RuntimeError(
                "Robosuite environment registration failed for "
                f"{TASK_SPECS[task].env_name!r}; registered={sorted(list_envs())!r}. "
                "Check the Robosuite executor installation and import log."
            )

        self.task = task
        self.task_id = tuple(TASK_SPECS).index(task)
        self.spec = TASK_SPECS[task]
        self.seed = int(seed)
        self.public_seed_indexed = False
        # ``rats.envs.base.get_env`` is cached for interactive notebooks.  An
        # evaluator process may execute several isolated episodes, and reusing
        # an already closed MuJoCo object would silently couple their lifecycle.
        get_env.cache_clear()
        self._env = get_env(self.spec.env_name, privileged=False)
        # The low-level constructor has its own default; the registered
        # comparison cap is applied explicitly and recorded in the manifest.
        self._env.max_steps = int(max_steps)
        self._env.reset(seed=self.seed)
        self._record_experiment_reset(public_seed=self.seed, simulator_seed=self.seed)

        self._api = FrankaControlApiReducedSkillLibrary(
            self._env,
            bimanual=self.spec.bimanual,
            is_handover=self.spec.handover,
            use_sam3=True,
        )
        self._fn = self._api.functions()
        self._fn["goto_pose"] = self._goto_pose_arm0
        self._fn["goto_home_joint_position"] = self.go_home
        self.oracle = RobosuiteOracle(self._env)
        self.instruction = self.spec.instruction
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()
        self._points_cache: dict[str, np.ndarray] = {}
        self._camera_home: dict[str, np.ndarray] = {}
        self._view_offset: np.ndarray | None = None
        self._grasp_origin: dict[str, np.ndarray] = {}
        self._home_joints = self._read_home_joints()

    # ---------------------------------------------------------- construction

    @property
    def _sim(self):
        return self._env.robosuite_env.sim

    def _read_home_joints(self) -> dict[int, np.ndarray]:
        obs = self._env.get_observation()
        homes: dict[int, np.ndarray] = {}
        if "robot_joint_pos" in obs:
            homes[0] = np.asarray(obs["robot_joint_pos"], dtype=float).reshape(-1)[:7]
        elif "robot0_joint_pos" in obs:
            homes[0] = np.asarray(obs["robot0_joint_pos"], dtype=float).reshape(-1)[:7]
        if "robot1_joint_pos" in obs:
            homes[1] = np.asarray(obs["robot1_joint_pos"], dtype=float).reshape(-1)[:7]
        return homes

    def _record_experiment_reset(
        self, *, public_seed: int, simulator_seed: int
    ) -> None:
        target = os.environ.get("RACAP_SIM_EPISODE_TELEMETRY_PATH", "").strip()
        if not target:
            return
        record = {
            "event": "environment_reset",
            "time": time.time(),
            "pid": os.getpid(),
            "episode_key": os.environ.get("RACAP_EPISODE_KEY", ""),
            "task_prompt": self.spec.instruction,
            "public_seed": int(public_seed),
            "seed": int(simulator_seed),
            "init_state_index": int(simulator_seed),
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task": self.task,
            "environment_class": type(self._env).__name__,
        }
        self._append_private_jsonl(target, record)

    @staticmethod
    def _append_private_jsonl(target: str, record: dict[str, Any]) -> None:
        path = Path(target).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_evaluator_checkpoint(self, event: str) -> None:
        """Write native state one-way for evaluation, returning nothing."""
        target = os.environ.get("RACAP_NATIVE_TELEMETRY_PATH", "").strip()
        if not target:
            return
        self._append_private_jsonl(
            target,
            {
                "event": str(event),
                "time": time.time(),
                "pid": os.getpid(),
                "episode_key": os.environ.get("RACAP_EPISODE_KEY", ""),
                "native_success": self.oracle.task_completed(),
                "reward": self.oracle.reward(),
                "native_predicates": [],
                "simulator_steps": int(getattr(self._env, "_sim_step_count", 0)),
                "init_state_index": self.seed,
            },
        )

    def reset(self, seed: int | None = None) -> None:
        public_seed = self.seed if seed is None else int(seed)
        self.reset_cameras()
        self._env.reset(seed=public_seed)
        self.seed = public_seed
        self._record_experiment_reset(public_seed=public_seed, simulator_seed=public_seed)
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()
        self._points_cache.clear()
        self._grasp_origin.clear()
        self._home_joints = self._read_home_joints()

    # ------------------------------------------------------------ perception

    def observe(self) -> dict[str, Any]:
        raw = self._fn["get_observation"]()
        # Robosuite's raw observation dictionary may include benchmark object
        # poses (for example ``nut_poses`` and ``*_pos`` keys) even when the
        # high-level executor uses only vision.  Build an allow-listed copy so
        # candidate code cannot discover those fields through ``observe``.
        obs: dict[str, Any] = {}
        for name, value in raw.items():
            if isinstance(value, dict) and isinstance(value.get("images"), dict):
                obs[name] = {
                    key: value[key]
                    for key in ("images", "intrinsics", "pose", "pose_mat")
                    if key in value
                }
            elif name in {
                "robot_cartesian_pos",
                "robot0_cartesian_pos",
                "robot1_cartesian_pos",
                "robot_joint_pos",
                "robot0_joint_pos",
                "robot1_joint_pos",
            }:
                obs[name] = value
        # The mature RACaP stack calls its scene camera ``agentview``.  The
        # single-arm Robosuite tasks call the same calibrated stream
        # ``robot0_robotview``.  Aliasing the dictionary changes no pixels or
        # calibration and avoids benchmark-specific branches in Policy APIs.
        if "agentview" not in obs and "robot0_robotview" in obs:
            obs["agentview"] = obs["robot0_robotview"]
        if "robot0_robotview" not in obs and "agentview" in obs:
            obs["robot0_robotview"] = obs["agentview"]
        if "robot_cartesian_pos" not in obs and "robot0_cartesian_pos" in obs:
            obs["robot_cartesian_pos"] = obs["robot0_cartesian_pos"]
        # The upstream Robosuite wrappers append a 10.7 cm local TCP offset to
        # ``gripper*_right_eef`` when constructing robot_cartesian_pos, while
        # their IK API applies the same offset when solving for that EEF link.
        # Consequently a pose commanded through solve_ik and the pose observed
        # after it refer to different points.  The wrapper also caches the EEF
        # body pose only in ``_step_once``; joint-space blocking motion can
        # therefore leave robot_cartesian_pos stale until a later gripper step.
        # Rebuild only the public robot pose from the live kinematic link, in
        # the same robot-0 base frame and orientation convention used by IK.
        # This is embodiment calibration, not task state: no object pose,
        # reward, predicate, or success signal is read.
        if self.spec.bimanual:
            for arm in (0, 1):
                key = f"robot{arm}_cartesian_pos"
                if key in obs:
                    obs[key] = self._live_ik_cartesian_state(arm, obs[key])
        elif "robot_cartesian_pos" in obs:
            obs["robot_cartesian_pos"] = self._live_ik_cartesian_state(
                0, obs["robot_cartesian_pos"]
            )
        for value in obs.values():
            if not isinstance(value, dict):
                continue
            images = value.get("images")
            if isinstance(images, dict) and "depth" in images:
                depth = np.asarray(images["depth"])
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    images["depth"] = depth[..., 0]
        return obs

    def _live_ik_cartesian_state(self, arm: int, fallback: Any) -> np.ndarray:
        """Return the live EEF-link pose in the coordinate frame accepted by IK.

        Synthetic/unit-test runtimes do not own a MuJoCo simulator, so they
        retain the supplied public state.  Production Robosuite runtimes read
        only robot kinematics and preserve the wrapper's public gripper scalar.
        """

        state = np.asarray(fallback, dtype=float).reshape(-1).copy()
        if state.size < 7:
            return state
        try:
            import viser.transforms as vtf

            env = self._env
            if self.spec.bimanual:
                link_index = int(getattr(env, f"gripper_link_idx_{int(arm)}"))
                # Both public arm poses are expressed in robot 0's base frame;
                # this is also the frame used by the shared bimanual API.
                base = np.asarray(env.base_link_wxyz_xyz_0, dtype=float)
            else:
                if int(arm) != 0:
                    return state
                link_index = int(env.gripper_link_idx)
                base = np.asarray(env.base_link_wxyz_xyz, dtype=float)
            link = np.concatenate(
                [
                    np.asarray(self._sim.data.xquat[link_index], dtype=float),
                    np.asarray(self._sim.data.xpos[link_index], dtype=float),
                ]
            )
            pose = (
                vtf.SE3(wxyz_xyz=base).inverse()
                @ vtf.SE3(wxyz_xyz=link)
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                    translation=np.zeros(3, dtype=float),
                )
            )
            state[:3] = np.asarray(pose.translation(), dtype=float)
            state[3:7] = np.asarray(pose.rotation().wxyz, dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            # Test doubles and future non-MuJoCo adapters keep their already
            # public observation rather than gaining a hidden-state fallback.
            return state
        return state

    def enable_video_capture(
        self, *, subsample_rate: int = 8, wrist_camera: bool = False
    ) -> None:
        self._env._subsample_rate = max(1, int(subsample_rate))
        try:
            self._env.enable_video_capture(
                True, clear=True, wrist_camera=bool(wrist_camera)
            )
        except TypeError:
            # The registered bimanual environments expose the same scene
            # recorder without a wrist-camera option.
            self._env.enable_video_capture(True, clear=True)

    def disable_video_capture(self) -> None:
        try:
            self._env.enable_video_capture(False, clear=False, wrist_camera=False)
        except TypeError:
            self._env.enable_video_capture(False, clear=False)

    def capture_video_frame(self) -> None:
        self._env._record_frame()

    def video_capture_enabled(self) -> bool:
        return bool(getattr(self._env, "_record_frames", False))

    def video_frame_count(self) -> int:
        return len(getattr(self._env, "_frame_buffer", []))

    def video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return list(self._env.get_video_frames(clear=clear))

    def wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(getattr(self._env, "_wrist_frame_buffer", []))
        if clear and hasattr(self._env, "_wrist_frame_buffer"):
            self._env._wrist_frame_buffer.clear()
        return frames

    def _camera(
        self, obs: dict[str, Any], wrist: bool, view: str | None = None
    ) -> tuple[np.ndarray, ...]:
        requested = view or ("robot0_eye_in_hand" if wrist else "agentview")
        if requested == "agentview" and requested not in obs:
            requested = "robot0_robotview"
        cam = obs.get(requested)
        if cam is not None:
            return (
                np.asarray(cam["images"]["rgb"]),
                np.asarray(cam["images"]["depth"]),
                np.asarray(cam["intrinsics"]),
                np.asarray(cam["pose_mat"]),
            )
        return self._render_view(requested)

    # --------------------------------------------------------------- motion

    def _goto_pose_arm0(
        self, position: Any, quaternion_wxyz: Any, _gripper: float = 0.0
    ) -> None:
        if self.spec.bimanual:
            joints = self._fn["solve_ik_arm0"](
                np.asarray(position, dtype=float), np.asarray(quaternion_wxyz, dtype=float)
            )
            self._fn["move_to_joints_arm0"](joints)
        else:
            joints = self._fn["solve_ik"](
                np.asarray(position, dtype=float), np.asarray(quaternion_wxyz, dtype=float)
            )
            self._fn["move_to_joints"](joints)

    def open_gripper(self) -> None:
        self._fn["open_gripper_arm0" if self.spec.bimanual else "open_gripper"]()

    def close_gripper(self) -> None:
        self._fn["close_gripper_arm0" if self.spec.bimanual else "close_gripper"]()

    def ee_pose(self) -> dict[str, Any]:
        return self.ee_pose_arm(0)

    def ee_pose_arm(self, arm: int) -> dict[str, Any]:
        arm = int(arm)
        if arm not in ({0, 1} if self.spec.bimanual else {0}):
            raise ValueError(f"arm {arm} is unavailable for task {self.task}")
        obs = self.observe()
        key = "robot_cartesian_pos" if not self.spec.bimanual else f"robot{arm}_cartesian_pos"
        state = np.asarray(obs[key], dtype=float).reshape(-1)
        return {
            "arm": arm,
            "position": [float(value) for value in state[:3]],
            "quat": [float(value) for value in state[3:7]],
            "gripper_opening": float(state[7]) if state.size > 7 else float("nan"),
        }

    def move_to_arm(
        self, arm: int, position: Any, quat: Any = None, *, yaw: float | None = None
    ) -> None:
        from racap.backends.libero import _yaw_quat

        arm = int(arm)
        if arm not in ({0, 1} if self.spec.bimanual else {0}):
            raise ValueError(f"arm {arm} is unavailable for task {self.task}")
        if yaw is not None:
            quat = _yaw_quat(float(yaw))
        target_quat = self._last_grasp_quat if quat is None else np.asarray(quat, dtype=float)
        solve = self._fn[f"solve_ik_arm{arm}"] if self.spec.bimanual else self._fn["solve_ik"]
        move = (
            self._fn[f"move_to_joints_arm{arm}"]
            if self.spec.bimanual
            else self._fn["move_to_joints"]
        )
        move(solve(np.asarray(position, dtype=float), target_quat))

    def move_bimanual(
        self,
        position0: Any,
        position1: Any,
        *,
        quat0: Any = None,
        quat1: Any = None,
    ) -> None:
        if not self.spec.bimanual:
            raise RuntimeError(f"task {self.task} has only one arm")
        q0 = self._last_grasp_quat if quat0 is None else np.asarray(quat0, dtype=float)
        q1 = self._last_grasp_quat if quat1 is None else np.asarray(quat1, dtype=float)
        j0 = self._fn["solve_ik_arm0"](np.asarray(position0, dtype=float), q0)
        j1 = self._fn["solve_ik_arm1"](np.asarray(position1, dtype=float), q1)
        self._fn["move_to_joints_both"](j0, j1)

    def open_gripper_arm(self, arm: int) -> None:
        arm = int(arm)
        if not self.spec.bimanual:
            if arm != 0:
                raise ValueError("single-arm task has no arm 1")
            return self.open_gripper()
        self._fn[f"open_gripper_arm{arm}"]()

    def close_gripper_arm(self, arm: int) -> None:
        arm = int(arm)
        if not self.spec.bimanual:
            if arm != 0:
                raise ValueError("single-arm task has no arm 1")
            return self.close_gripper()
        self._fn[f"close_gripper_arm{arm}"]()

    def go_home(self) -> None:
        if not self._home_joints:
            return
        if self.spec.bimanual and 0 in self._home_joints and 1 in self._home_joints:
            self._fn["move_to_joints_both"](self._home_joints[0], self._home_joints[1])
        elif 0 in self._home_joints:
            key = "move_to_joints_arm0" if self.spec.bimanual else "move_to_joints"
            self._fn[key](self._home_joints[0])
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()
