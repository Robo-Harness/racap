from __future__ import annotations

"""MolmoSpaces-only frame helpers.

`robot_base_pose` is a required 7D pose in `[x, y, z, w, x, y, z]` order.
Helpers in this module are scoped only to the current MolmoSpaces IK boundary.
"""

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_EPS = 1e-8


def _normalize_quaternion_wxyz(quaternion_wxyz: object) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion_wxyz shape (4,), got {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion contains non-finite values")
    norm = np.linalg.norm(quat)
    if norm <= _EPS:
        raise ValueError("Quaternion has near-zero norm")
    return quat / norm


def validate_robot_base_pose(robot_base_pose: object) -> np.ndarray:
    """Return normalized float64 shape-(7,) base pose in [x, y, z, w, x, y, z] order."""
    pose = np.asarray(robot_base_pose, dtype=np.float64)
    if pose.ndim != 1 or pose.shape != (7,):
        raise ValueError(f"Expected robot_base_pose shape (7,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("robot_base_pose contains non-finite values")
    quat = _normalize_quaternion_wxyz(pose[3:])
    return np.concatenate([pose[:3], quat])


def _pose7d_to_matrix_wxyz(pose_7d: np.ndarray) -> np.ndarray:
    quat_wxyz = _normalize_quaternion_wxyz(pose_7d[3:])
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciRotation.from_quat(quat_xyzw).as_matrix()
    transform[:3, 3] = np.asarray(pose_7d[:3], dtype=np.float64)
    return transform


def grasp_site_quat_to_panda_hand_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Pre-rotate a grasp_site target quaternion into the corresponding
    panda_hand target quaternion (what pyroki IK actually solves for).

    The mismatch comes from three rotations baked into the model files:

    - panda_description URDF: panda_hand_joint has rpy="0 0 -π/4" — so
      panda_hand is Rz(-45°) from panda_link8.
    - franka_droid MJCF: the Robotiq is mounted inside a
      <frame quat="0.7071 0 0 0.7071"> wrapping <attach> — Rz(+90°) from
      attachment_site (= panda_link8) to the Robotiq base.
    - franka_droid MJCF Robotiq: grasp_site has xyaxes="0 -1 0 1 0 0"
      — Rz(-90°) from Robotiq base to grasp_site.

    Net effect: R_grasp_site = R_panda_hand @ Rz(+45°). So to make the
    achieved grasp_site land at a user-commanded orientation R_user, the
    IK must target R_user @ Rz(-45°). This helper performs that
    composition in wxyz convention.

    Note: only the rotational component changes; the gripper's local +Z
    axis is invariant under any Rz, so any position offset expressed
    along local +Z (e.g. _TCP_OFFSET = [0, 0, -0.155]) gives an
    identical world-frame translation whether applied with R_user or
    R_user @ Rz(-45°).
    """
    q_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    r = SciRotation.from_quat(q_xyzw) * SciRotation.from_euler("z", -np.pi / 4)
    q_xyzw_ik = r.as_quat()
    return np.array(
        [q_xyzw_ik[3], q_xyzw_ik[0], q_xyzw_ik[1], q_xyzw_ik[2]], dtype=np.float64
    )


def world_pose_to_robot_base_frame(
    position_world: np.ndarray,
    quaternion_world_wxyz: np.ndarray,
    robot_base_pose: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world-frame EE target pose into robot-base-frame pose."""
    position_world = np.asarray(position_world, dtype=np.float64)
    if position_world.shape != (3,):
        raise ValueError(f"Expected position_world shape (3,), got {position_world.shape}")
    if not np.all(np.isfinite(position_world)):
        raise ValueError("position_world contains non-finite values")

    quaternion_world_wxyz = _normalize_quaternion_wxyz(quaternion_world_wxyz)
    robot_base_pose = validate_robot_base_pose(robot_base_pose)

    world_from_base = _pose7d_to_matrix_wxyz(robot_base_pose)
    base_from_world = np.linalg.inv(world_from_base)

    world_from_target = _pose7d_to_matrix_wxyz(
        np.concatenate([position_world, quaternion_world_wxyz]),
    )
    base_from_target = base_from_world @ world_from_target

    position_base = np.asarray(base_from_target[:3, 3], dtype=np.float64)
    quat_xyzw = SciRotation.from_matrix(base_from_target[:3, :3]).as_quat()
    quaternion_base_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    quaternion_base_wxyz = _normalize_quaternion_wxyz(quaternion_base_wxyz)
    return position_base, quaternion_base_wxyz
