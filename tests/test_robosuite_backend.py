from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evolution.harness.public_runtime import PublicRuntime
from evolution.harness.robosuite import (
    DEVELOPMENT_SEEDS,
    DEVELOPMENT_TASKS,
    EVALUATION_SEEDS,
    HELDOUT_TASKS,
    RobosuiteEvaluationRunner,
    development_curriculum,
)
from racap.backends.robosuite import (
    TASK_SPECS,
    RobosuitePrimitiveRuntime,
    RobosuiteTaskSpec,
)
from scripts.eval_robosuite_agent import _completed, _outcome_abort_reason
from evolution.harness.guards import _executable_tokens, _FORBIDDEN
from racap.agent.artifacts import compact_trajectory_row


class _FakeEnv:
    def __init__(self, observation):
        self.observation = observation

    def get_observation(self):
        return self.observation


def _camera():
    return {
        "images": {
            "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "depth": np.ones((8, 8, 1), dtype=np.float32),
        },
        "intrinsics": np.eye(3),
        "pose_mat": np.eye(4),
    }


def _runtime(*, bimanual: bool) -> RobosuitePrimitiveRuntime:
    runtime = object.__new__(RobosuitePrimitiveRuntime)
    runtime.task = "synthetic"
    runtime.spec = RobosuiteTaskSpec(
        "synthetic_env", "Synthetic public instruction", bimanual=bimanual
    )
    observation = {
        ("agentview" if bimanual else "robot0_robotview"): _camera(),
        (
            "robot0_cartesian_pos" if bimanual else "robot_cartesian_pos"
        ): np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4]),
        # Representative evaluator/object truth that exists in some raw
        # Robosuite observations and must never cross the public boundary.
        "nut_poses": {"square_nut": np.arange(7)},
        "cube_pos": np.array([0.4, 0.0, 0.1]),
    }
    if bimanual:
        observation["robot1_cartesian_pos"] = np.array(
            [0.8, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.6]
        )
    runtime._env = _FakeEnv(observation)
    calls = []

    def solve(position, quat):
        calls.append(("solve", tuple(position), tuple(quat)))
        return np.arange(7, dtype=float)

    runtime._fn = {
        "get_observation": runtime._env.get_observation,
        "solve_ik": solve,
        "move_to_joints": lambda joints: calls.append(("move", tuple(joints))),
        "solve_ik_arm0": solve,
        "solve_ik_arm1": solve,
        "move_to_joints_arm0": lambda joints: calls.append(("move0", tuple(joints))),
        "move_to_joints_arm1": lambda joints: calls.append(("move1", tuple(joints))),
        "move_to_joints_both": lambda j0, j1: calls.append(
            ("both", tuple(j0), tuple(j1))
        ),
        "open_gripper": lambda: calls.append(("open",)),
        "close_gripper": lambda: calls.append(("close",)),
        "open_gripper_arm0": lambda: calls.append(("open0",)),
        "close_gripper_arm0": lambda: calls.append(("close0",)),
        "open_gripper_arm1": lambda: calls.append(("open1",)),
        "close_gripper_arm1": lambda: calls.append(("close1",)),
    }
    runtime._last_grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])
    runtime.oracle = SimpleNamespace(task_completed=lambda: True)
    runtime._test_calls = calls
    return runtime


def test_registered_tasks_are_complete_and_do_not_encode_action_sequences():
    assert set(TASK_SPECS) == {
        "cube_lifting",
        "cube_restack",
        "cube_stack",
        "nut_assembly",
        "spill_wipe",
        "two_arm_handover",
        "two_arm_lift",
    }
    for spec in TASK_SPECS.values():
        assert spec.env_name
        assert spec.instruction
        # A backend registration may describe the public goal, but must not
        # carry executable waypoints or task-conditioned control parameters.
        assert not hasattr(spec, "waypoints")


def test_single_arm_observation_is_schema_aliased_without_changing_arrays():
    runtime = _runtime(bimanual=False)
    original = runtime._env.observation["robot0_robotview"]
    observed = runtime.observe()
    # The public camera dictionary is allow-list copied to strip any future
    # simulator-only fields, while both public aliases still share one camera
    # record and retain the original pixel buffer.
    assert observed["agentview"] is observed["robot0_robotview"]
    assert np.shares_memory(
        observed["agentview"]["images"]["rgb"], original["images"]["rgb"]
    )
    assert observed["agentview"]["images"]["depth"].shape == (8, 8)
    assert "nut_poses" not in observed
    assert "cube_pos" not in observed


def test_live_cartesian_state_matches_ik_eef_link_without_duplicate_tcp_offset():
    runtime = _runtime(bimanual=False)
    model = SimpleNamespace()
    data = SimpleNamespace(
        xquat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        xpos=np.asarray([[0.03, -0.016, 0.854]], dtype=float),
    )
    runtime._env.gripper_link_idx = 0
    runtime._env.base_link_wxyz_xyz = np.asarray(
        [1.0, 0.0, 0.0, 0.0, -0.56, 0.0, 0.922], dtype=float
    )
    runtime._env.robosuite_env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))

    observed = runtime.observe()["robot_cartesian_pos"]

    # World EEF [0.03, -0.016, 0.854] relative to base
    # [-0.56, 0, 0.922]. No second +/-10.7 cm TCP translation is added.
    assert observed[:3] == pytest.approx([0.59, -0.016, -0.068])
    assert observed[7] == pytest.approx(0.4)


def test_public_runtime_exposes_geometry_neutral_bimanual_control_only():
    backend = _runtime(bimanual=True)
    runtime = PublicRuntime(backend)
    assert runtime.ee_pose_arm(1)["position"] == [0.8, 0.2, 0.3]
    runtime.move_bimanual([0.2, 0.0, 0.4], [0.7, 0.0, 0.4])
    runtime.open_gripper_arm(1)
    assert any(call[0] == "both" for call in backend._test_calls)
    assert ("open1",) in backend._test_calls
    with pytest.raises(AttributeError):
        _ = runtime.oracle


def test_single_arm_rejects_arm_one_commands():
    runtime = _runtime(bimanual=False)
    with pytest.raises(ValueError, match="no arm 1|unavailable"):
        runtime.move_to_arm(1, [0.1, 0.2, 0.3])


def test_robosuite_evolution_split_is_product_and_sealed():
    curriculum = development_curriculum()
    stage = curriculum.get("rs_cross_embodiment")
    assert len(stage.development) == len(DEVELOPMENT_TASKS) * len(DEVELOPMENT_SEEDS)
    assert {
        (item.context["task_name"], item.seed) for item in stage.development
    } == {
        (task, seed) for task in DEVELOPMENT_TASKS for seed in DEVELOPMENT_SEEDS
    }
    sealed = {item.key for item in curriculum.sealed}
    assert all(
        f"robosuite/{task}/seed{seed}" in sealed
        for task in HELDOUT_TASKS
        for seed in DEVELOPMENT_SEEDS
    )
    assert all(
        f"robosuite/{task}/seed{seed}" in sealed
        for task in TASK_SPECS
        for seed in EVALUATION_SEEDS
    )
    assert not ({item.key for item in stage.active} & sealed)


def test_infrastructure_failures_never_enter_resume_or_accuracy_set(tmp_path):
    episode = tmp_path / "raw" / "cube_lifting" / "seed0"
    episode.mkdir(parents=True)
    (episode / "record.json").write_text(
        '{"key":"robosuite/cube_lifting/seed0",'
        '"native_success":false,"infrastructure_error":true,'
        '"scorable":false,"abort_reason":""}',
        encoding="utf-8",
    )
    assert _completed(tmp_path) == {}


def test_persisted_trajectory_omits_pixels_but_keeps_control_evidence():
    row = {
        "steps": [
            {
                "action": "pick_lift",
                "report": {
                    "video_frame_range": [2, 8],
                    "grasp_depth": None,
                    "grounding": {
                        "result": {
                            "agentview": {
                                "images": {
                                    "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
                                    "depth": np.ones((64, 64), dtype=np.float32),
                                },
                                "pose": [0.1, 0.2, 0.3],
                            }
                        }
                    },
                },
            }
        ]
    }

    compact = compact_trajectory_row(row)
    report = compact["steps"][0]["report"]
    images = report["grounding"]["result"]["agentview"]["images"]
    assert report["video_frame_range"] == [2, 8]
    assert report["grasp_depth"] is None
    assert report["grounding"]["result"]["agentview"]["pose"] == [0.1, 0.2, 0.3]
    assert images["raw_sensor_payload_omitted"] is True
    assert images["channels"]["rgb"]["shape"] == [64, 64, 3]
    assert images["channels"]["depth"]["shape"] == [64, 64]


def test_caught_quota_failure_is_still_an_evaluator_abort(tmp_path):
    outcome = SimpleNamespace(
        stopped="task planner did not answer usably: insufficient_user_quota"
    )
    assert _outcome_abort_reason(outcome, {"quota_failures": 1}) == "quota_exhausted"

    episode = tmp_path / "raw" / "spill_wipe" / "seed0"
    episode.mkdir(parents=True)
    (episode / "record.json").write_text(
        '{"key":"robosuite/spill_wipe/seed0","native_success":false,'
        '"scorable":true,"abort_reason":"","stopped":"planner stopped",'
        '"evaluator_llm_calls":{"quota_failures":1}}',
        encoding="utf-8",
    )
    assert _completed(tmp_path) == {}


def test_robosuite_runner_preserves_virtualenv_launcher_symlink(tmp_path):
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to("/usr/bin/python3")
    runner = RobosuiteEvaluationRunner(
        tmp_path,
        tmp_path / "experiment",
        python=launcher,
        robosuite_root=tmp_path / "robosuite",
        rats_root=tmp_path / "rats",
        capx_root=tmp_path / "capx",
    )
    assert runner.python == launcher.absolute()
    assert runner.python != launcher.resolve()


def test_leakage_guard_ignores_oracle_prose_but_rejects_executable_access():
    prose = '''"""This is not a task-completion oracle."""\n# never call runtime.oracle\nvalue = "task_completed"\n'''
    executable = _executable_tokens(prose)
    assert not any(pattern.search(executable) for pattern in _FORBIDDEN.values())
    forbidden = _executable_tokens("value = runtime.oracle.task_completed()\n")
    assert _FORBIDDEN["native oracle"].search(forbidden)
    assert _FORBIDDEN["native predicate"].search(forbidden)
