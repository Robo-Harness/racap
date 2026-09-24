"""CaP-Gym R1Pro primitives cataloged from rats/integrations/r1pro/control.py.

This file documents EVERY perception and control primitive available to
agent-generated code running in the CaP-Gym BEHAVIOR-1K environment with
the R1ProControlApi.

Sourced directly from R1ProControlApi.functions() in
rats/integrations/r1pro/control.py (27 primitives).
"""

# Each primitive is documented with: name, signature, description, category

R1PRO_CONTROL_PRIMITIVES = [
    # ---- PERCEPTION / VISION ----
    {
        "name": "segment_sam3_text_prompt",
        "signature": "segment_sam3_text_prompt(rgb: ndarray, text_prompt: str) -> list[dict]",
        "description": "Run SAM3 segmentation on an RGB image conditioned on a text prompt. Returns list of masks with confidence scores.",
        "category": "perception",
    },
    {
        "name": "segment_sam3_point_prompt",
        "signature": "segment_sam3_point_prompt(rgb: ndarray, point_coords: tuple) -> list[dict]",
        "description": "Run SAM3 segmentation with point prompt conditioning. Returns list of segmentation masks.",
        "category": "perception",
    },
    {
        "name": "point_prompt_molmo",
        "signature": "point_prompt_molmo(image: ndarray, text_prompt: str) -> dict[str, tuple]",
        "description": "Use Molmo VLM to point to object locations in image given text description. Returns dict of pixel coordinates.",
        "category": "perception",
    },
    {
        "name": "get_object_pose",
        "signature": "get_object_pose(object_name: str, return_bbox_extent: bool = False) -> tuple[ndarray, ndarray, ndarray | None]",
        "description": "Get 3D pose of named object via perception pipeline. Returns (position_xyz, quaternion_wxyz, optional bbox_extent).",
        "category": "perception",
    },
    {
        "name": "get_sam3_mask",
        "signature": "get_sam3_mask(object_name: str) -> int",
        "description": "Get SAM3 segmentation mask for named object. Returns mask pixel count (sum).",
        "category": "perception",
    },
    {
        "name": "get_env_observation",
        "signature": "get_env_observation() -> tuple[ndarray, ndarray]",
        "description": "Get current RGB and depth observation from environment camera. Returns (rgb: (H,W,3), depth: (H,W)).",
        "category": "perception",
    },
    {
        "name": "save_current_observation",
        "signature": "save_current_observation(name: str) -> None",
        "description": "Save current observation frame to disk with given name.",
        "category": "perception",
    },

    # ---- NAVIGATION ----
    {
        "name": "navigate_to_pose",
        "signature": "navigate_to_pose(pose_2d: ndarray) -> None",
        "description": "Navigate robot base to 2D pose (x, y, yaw).",
        "category": "navigation",
    },
    {
        "name": "get_navigation_pose",
        "signature": "get_navigation_pose(P_table: ndarray, P_object: ndarray) -> ndarray",
        "description": "Compute navigation pose for reaching a table-top object. Returns 2D pose.",
        "category": "navigation",
    },
    {
        "name": "find_object_base_rotate",
        "signature": "find_object_base_rotate(object_name: str) -> None",
        "description": "Rotate base to search for and find named object in the scene.",
        "category": "navigation",
    },
    {
        "name": "find_object_torso_rotate",
        "signature": "find_object_torso_rotate(object_name: str) -> None",
        "description": "Rotate torso to search for named object (used if base rotation search fails).",
        "category": "navigation",
    },

    # ---- MOTION CONTROL ----
    {
        "name": "move_hand",
        "signature": "move_hand(target_pose: tuple, arm: int = 0) -> None",
        "description": "Move end-effector hand to target pose. target_pose = (position, quaternion_xyzw). arm: 0=left, 1=right.",
        "category": "motion",
    },
    {
        "name": "move_to_joint_positions",
        "signature": "move_to_joint_positions(target_joint_positions: ndarray, max_steps: int = 20, settle_steps: int = 10) -> None",
        "description": "Move robot to target joint configuration. 28 joints total: base(6) + torso(4) + arms(14) + grippers(4).",
        "category": "motion",
    },
    {
        "name": "solve_ik",
        "signature": "solve_ik(position: ndarray, quaternion_wxyz: ndarray, arm: int = 0, offset_translation: list = [0.02, 0.0, -0.05]) -> ndarray",
        "description": "Solve inverse kinematics for target end-effector pose. Returns 28-dim joint angles.",
        "category": "motion",
    },
    {
        "name": "get_current_eef_pose",
        "signature": "get_current_eef_pose(arm: int = 0) -> tuple[ndarray, ndarray]",
        "description": "Get current end-effector pose. Returns (position_xyz, quaternion_xyzw).",
        "category": "motion",
    },
    {
        "name": "get_current_joint_positions",
        "signature": "get_current_joint_positions() -> ndarray",
        "description": "Get current joint positions as numpy array.",
        "category": "motion",
    },
    {
        "name": "get_robot_position",
        "signature": "get_robot_position() -> tuple[ndarray, ndarray, ndarray]",
        "description": "Get robot base position and orientation. Returns (position, quaternion, yaw).",
        "category": "motion",
    },
    {
        "name": "get_robot_relative_eef_pose",
        "signature": "get_robot_relative_eef_pose(arm: int = 0) -> tuple",
        "description": "Get end-effector pose relative to robot base frame.",
        "category": "motion",
    },
    {
        "name": "lift_arm",
        "signature": "lift_arm(arm: int = 0) -> None",
        "description": "Lift specified arm up in the environment.",
        "category": "motion",
    },
    {
        "name": "reset_torso",
        "signature": "reset_torso() -> None",
        "description": "Reset torso to initial/home position.",
        "category": "motion",
    },

    # ---- GRASPING ----
    {
        "name": "sample_grasp_pose",
        "signature": "sample_grasp_pose(object_name: str) -> tuple[list, list]",
        "description": "Plan grasp for named object using GraspNet. Returns (pregrasp_poses, grasp_poses).",
        "category": "grasping",
    },
    {
        "name": "grasp_object",
        "signature": "grasp_object(pregrasp_pose: ndarray, grasp_pose: ndarray, object_name: str, arm: int = 0) -> None",
        "description": "Execute grasp sequence: approach pre-grasp pose, move to grasp pose, close gripper.",
        "category": "grasping",
    },

    # ---- GRIPPER ----
    {
        "name": "open_gripper",
        "signature": "open_gripper(arm: int = 0) -> None",
        "description": "Open gripper fully. arm: 0=left, 1=right.",
        "category": "gripper",
    },
    {
        "name": "close_gripper",
        "signature": "close_gripper(arm: int = 0) -> None",
        "description": "Close gripper fully. arm: 0=left, 1=right.",
        "category": "gripper",
    },
    {
        "name": "check_object_in_hand",
        "signature": "check_object_in_hand(arm: int = 0) -> bool",
        "description": "Check if an object is currently grasped in the specified gripper. Returns bool.",
        "category": "gripper",
    },

    # ---- UTILITY ----
    {
        "name": "write_video",
        "signature": "write_video(name: str) -> None",
        "description": "Write video of execution trajectory to disk.",
        "category": "utility",
    },
]


def get_all_primitive_names() -> list[str]:
    """Return sorted list of all primitive function names."""
    return sorted(p["name"] for p in R1PRO_CONTROL_PRIMITIVES)


def get_primitives_by_category(category: str) -> list[dict]:
    """Return primitives filtered by category."""
    return [p for p in R1PRO_CONTROL_PRIMITIVES if p["category"] == category]


def get_primitive_docs() -> str:
    """Return formatted documentation string of all primitives for prompts."""
    lines = []
    for p in R1PRO_CONTROL_PRIMITIVES:
        lines.append(f"{p['signature']}")
        lines.append(f"  {p['description']}")
        lines.append("")
    return "\n".join(lines)


def build_initial_skills() -> list[dict]:
    """Build initial skill entries for the skill library from CaP-Gym primitives."""
    skills = []
    for p in R1PRO_CONTROL_PRIMITIVES:
        skills.append({
            "skill_id": f"primitive_{p['name']}",
            "name": p["name"],
            "description": p["description"],
            "code": f"# Built-in CaP-Gym primitive: {p['signature']}",
            "api_primitives_used": [p["name"]],
            "preconditions": [],
            "effects": [],
            "dependent_skills": [],
            "success_rate": 1.0,
            "source_task": "capgym_builtin",
            "is_primitive": True,
        })
    return skills
