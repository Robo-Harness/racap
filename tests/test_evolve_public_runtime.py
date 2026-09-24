import pytest

from evolution.harness.public_runtime import PublicRuntime


class Backend:
    oracle = object()

    def observe(self):
        return {"rgb": "frame"}

    def _camera(self, observation, wrist=False):
        return observation["rgb"], "depth", "intrinsics", "extrinsics"

    def move_to(self, position, **kwargs):
        self.position = tuple(position)
        return position

    def ee_pose(self):
        return {"position": list(getattr(self, "position", (0.0, 0.0, 0.0)))}

    def record_evaluator_checkpoint(self, event):
        self.checkpoints = [*getattr(self, "checkpoints", []), event]

    def localize(self, label, **options):
        target = type("Target", (), {})()
        target.label = label
        target.pose = [0.1, 0.2, 0.3]
        target.options = options
        self.localize_calls = getattr(self, "localize_calls", 0) + 1
        return target

    def grasp(self, target, *, strategy):
        assert hasattr(target, "pose")
        return {"label": target.label, "strategy": strategy}

    def contact_grasp(self, target, *, strategy, lift=0.0):
        assert hasattr(target, "pose")
        return {"label": target.label, "strategy": strategy, "lift": lift}

    def place(self, target, *, strategy):
        assert hasattr(target, "pose")
        return {"label": target.label, "strategy": strategy}


def test_proxy_hides_oracle_and_meters_public_calls():
    runtime = PublicRuntime(Backend())
    assert runtime.observe() == {"rgb": "frame"}
    runtime.move_to((0, 0, 0))
    with pytest.raises(AttributeError):
        _ = runtime.oracle
    assert runtime.call_stats() == {
        "public_calls": 2,
        "perception_calls": 1,
        "observe": 1,
        "motion_calls": 1,
        "move_to": 1,
    }


def test_proxy_exposes_only_the_rgbd_camera_decoder_needed_by_visual_react():
    runtime = PublicRuntime(Backend())
    assert runtime._camera(runtime.observe(), wrist=False) == (
        "frame",
        "depth",
        "intrinsics",
        "extrinsics",
    )
    assert runtime.call_stats()["_camera"] == 1
    with pytest.raises(AttributeError):
        _ = runtime.task_completed


def test_public_runtime_exposes_geometry_neutral_paths_and_manifest():
    runtime = PublicRuntime(Backend())
    reached = runtime.linear_contact((0, 0, 0), (0.1, 0, 0), steps=3)
    assert [item["reached"]["position"] for item in reached] == [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ]
    arc = runtime.arc_contact((0, 0), 0.1, 0.0, 1.57079632679, 0.2, steps=3)
    assert len(arc) == 3
    manifest = runtime.capability_manifest()
    assert manifest["native_oracle_available"] is False
    assert "linear_contact" in manifest["composed_primitives"]
    assert "target_or_label" in manifest["signatures"]["contact_grasp"]


def test_grounded_motion_accepts_cached_label_without_duplicate_grounding():
    backend = Backend()
    runtime = PublicRuntime(backend)

    target = runtime.localize("red cube", detector="sam3")
    assert runtime.contact_grasp("red cube", strategy="top_down", lift=0.02) == {
        "label": "red cube",
        "strategy": "top_down",
        "lift": 0.02,
    }
    assert backend.localize_calls == 1
    assert target is runtime._coerce_target("red cube")
    assert runtime.call_stats()["localize"] == 1
    assert runtime.call_stats()["contact_grasp"] == 1


def test_grounded_motion_localizes_an_uncached_label_and_accepts_targets():
    backend = Backend()
    runtime = PublicRuntime(backend)

    assert runtime.grasp("blue block", strategy="pca_axis")["label"] == "blue block"
    target = runtime.localize("green bowl")
    assert runtime.place(target, strategy="center")["label"] == "green bowl"
    assert backend.localize_calls == 2


def test_public_runtime_forwards_write_only_after_tool_checkpoints():
    backend = Backend()
    runtime = PublicRuntime(backend)

    assert runtime.record_evaluator_checkpoint("after_tool:pickplace") is None
    assert backend.checkpoints == ["after_tool:pickplace"]
    assert runtime.call_stats() == {}
    with pytest.raises(ValueError):
        runtime.record_evaluator_checkpoint("episode_end")
