import numpy as np

import racap.agent.react as react
from racap.agent.react import (
    Episode,
    _qualified_visual_alias,
    _semantic_first_call,
    _normalise_insert_args,
    _requires_insert,
    _supports_insert,
    _retry_after_check,
    _spatially_qualified,
)
from racap.agent.tools import (
    Session,
    Step,
    _correction_nudge,
    _footprint_in_polygon,
    _visibly_recoverable_miss,
    _verify_destination_alias,
    describe_tools,
)
from racap.contracts import GroundedTarget, SkillResult
from racap.policy_api.pickplace_api import (
    _cavity_region,
    _destination_aspect,
    _long_axis,
)


def test_check_converts_right_down_error_to_right_up_correction() -> None:
    # Object right+down of goal -> command left+up.
    assert _correction_nudge(3.0, 4.0, 0.05) == [-0.023, 0.03]
    # Object left+up of goal -> command right+down (negative public up).
    assert _correction_nudge(-3.0, -4.0, 0.05) == [0.023, -0.03]


def test_insert_routes_bounded_shelf_interior_but_not_exposed_top() -> None:
    assert _supports_insert("cabinet shelf") is True
    assert _supports_insert("under the cabinet shelf") is True
    assert _supports_insert("the wine rack") is True
    assert _supports_insert("bottom drawer of the cabinet") is True
    assert _supports_insert("inside the wooden box") is True
    assert _supports_insert("on the wooden tray") is False
    assert _supports_insert("top of the shelf") is False
    assert _supports_insert("to the right of the shelf") is False
    args = _normalise_insert_args(
        {"pick": "book", "place": "black storage box"},
        Episode(
            instruction="put the book on the cabinet shelf",
            pick="book",
            place="cabinet shelf",
        ),
        allow_place_alias=True,
    )
    assert args["place"] == "black storage box"
    assert args["frontload"] is True


def test_only_unambiguous_containment_hard_overrides_agent_transport() -> None:
    assert _requires_insert("the wine rack", "put the wine bottle on the wine rack") is False
    assert _requires_insert("cabinet shelf", "put the book on the cabinet shelf") is True
    assert _requires_insert("cabinet shelf", "put the book inside the cabinet shelf") is True
    assert _requires_insert("bottom drawer", "put the bowl in the bottom drawer") is False


def test_spatial_pick_qualifier_cannot_be_dropped_for_visual_alias() -> None:
    assert _spatially_qualified("the butter at the front") is True
    assert _spatially_qualified("middle black bowl") is True
    assert _spatially_qualified("orange carton") is False
    assert _qualified_visual_alias("snack bar", "butter at the front") == "snack bar at the front"
    assert _qualified_visual_alias("black bowl", "middle black bowl") == "middle black bowl"


def test_not_grounded_retry_helper_preserves_visual_destination_alias() -> None:
    episode = Episode(
        instruction="put the wine bottle on the wine rack",
        pick="wine bottle",
        place="the wine rack",
    )

    retry = _retry_after_check(
        episode,
        report={},
        failure_mode="not_grounded",
        reflected={"place": "wooden wine rack"},
    )

    assert retry["pick"] == "wine bottle"
    assert retry["place"] == "wooden wine rack"


def test_destination_alias_is_rejected_without_grounding_failure() -> None:
    episode = Episode(
        instruction="put the bowl on the plate",
        pick="bowl",
        place="the plate",
    )

    retry = _retry_after_check(
        episode,
        report={},
        failure_mode="empty_grasp",
        reflected={"place": "nearby tray"},
    )

    assert retry["place"] == "the plate"


def test_empty_grasp_retry_preserves_valid_visual_agent_grasp() -> None:
    episode = Episode(
        instruction="put the bowl in the tray",
        pick="bowl",
        place="tray",
    )

    retry = _retry_after_check(
        episode,
        report={},
        failure_mode="empty_grasp",
        reflected={"grasp": ["graspnet"]},
    )

    assert retry["grasp"] == ["graspnet"]


def test_exposed_plate_placement_is_centred_and_contact_conditioned() -> None:
    episode = Episode(
        instruction="put the red mug on the left plate",
        pick="red mug",
        place="left plate",
    )

    first = _semantic_first_call({}, episode)
    retry = _retry_after_check(
        episode,
        report={"destination_kind": "surface"},
        failure_mode="placement_miss",
        reflected={"nudge": [0.01, -0.01]},
    )

    for call in (first, retry):
        assert call["centre"] is True
        assert call["release_on"] == "floor"
        assert call["place_margin"] == 0.003


def test_default_budget_keeps_transport_and_precision_calls_independent() -> None:
    episode = Episode(instruction="move it", pick="object", place="target")

    assert episode.max_pickplace_calls == 8
    assert episode.max_push_calls == 3
    assert episode.max_insert_calls == 2
    assert episode.turns == 27


def test_transport_tool_description_excludes_parent_level_actions() -> None:
    prompt = describe_tools(list(react.TRANSPORT_TOOL_NAMES))

    assert "- pickplace:" in prompt
    assert "- insert:" in prompt
    assert "- check:" in prompt
    assert "- articulate:" not in prompt
    assert "- actuate_control:" not in prompt
    assert "- stack:" not in prompt


class _SuccessfulReleaseSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill

    def frame(self):
        return "scene"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            return Step(
                action=action,
                args=args,
                image="released",
                success=True,
                report={"release_xy": [0.1, 0.2], "failure_mode": ""},
                observation="released; placement awaits check",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="checked",
                success=True,
                report={"reliable": True},
                observation="looks placed",
            )
        return Step(action=action, args=args, image="checked", success=True)


def test_successful_check_skips_post_transport_reflection(monkeypatch) -> None:
    calls = []

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        calls.append(system)
        assert not system.startswith("You are diagnosing")
        return "{}", {
            "thought": "move once",
            "action": "pickplace",
            "args": {"pick": "bowl", "place": "plate"},
        }

    monkeypatch.setattr(react, "Session", _SuccessfulReleaseSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the bowl on the plate",
            pick="bowl",
            place="plate",
            turns=5,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "done",
    ]
    assert len(calls) == 1
    assert outcome.reflections == []


class _InfrastructureFailureSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill

    def frame(self):
        return "scene"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        assert action == "pickplace"
        return Step(
            action=action,
            args=args,
            image="scene",
            success=False,
            report={
                "failure_mode": "perception_unavailable",
                "infrastructure_failure": {
                    "kind": "provider_unavailable",
                    "operation": "detect_many",
                    "model": "test-model",
                    "detail": "timeout",
                },
            },
            observation="visual evidence unavailable",
        )


def test_transport_react_stops_immediately_on_perception_infrastructure_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(react, "Session", _InfrastructureFailureSession)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put bowl on plate",
            pick="bowl",
            place="plate",
            turns=21,
            initial_skill="pickplace",
        ),
    )

    assert len(outcome.steps) == 1
    assert "evaluation is inconclusive" in outcome.stopped


def test_parent_selected_transport_skips_redundant_first_model_call(
    monkeypatch,
) -> None:
    def unexpected_ask(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the parent-selected first transport needs no VLM call")

    monkeypatch.setattr(react, "Session", _SuccessfulReleaseSession)
    monkeypatch.setattr(react, "_ask_json_reply", unexpected_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the bowl on the plate",
            pick="bowl",
            place="plate",
            turns=5,
            initial_skill="pickplace",
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "done",
    ]
    assert outcome.success is True


class _AlwaysEmptyGraspSession:
    calls: list[dict] = []

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).calls = []

    def frame(self):
        return "scene"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        type(self).calls.append(dict(args))
        return Step(
            action=action,
            args=args,
            image="after-contact",
            success=False,
            report={"failure_mode": "empty_grasp"},
            observation="contacted the object but lifted nothing",
        )


def test_visual_agent_can_escalate_empty_grasps_to_learned_grasp(monkeypatch) -> None:
    diagnoses = iter(
        [
            {
                "what_happened": "the object stayed down",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": ["pca_axis@low"]},
                "reason": "change the grasp family",
            },
            {
                "what_happened": "the object still stayed down",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": ["graspnet"]},
                "reason": "analytic families were empty",
            },
            {
                "what_happened": "the learned grasp was also empty",
                "looks_placed": False,
                "next": "stop",
                "retry_args": {},
                "reason": "no useful untried control remains",
            },
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", next(diagnoses)
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "object", "place": "target"},
        }

    monkeypatch.setattr(react, "Session", _AlwaysEmptyGraspSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put object on target",
            pick="object",
            place="target",
            turns=8,
            max_pickplace_calls=8,
        ),
    )

    assert len(_AlwaysEmptyGraspSession.calls) == 3
    assert _AlwaysEmptyGraspSession.calls[-1]["grasp"] == ["graspnet"]
    assert "visual state authority stopped" in outcome.stopped


def test_insert_args_reject_broad_pickplace_controls() -> None:
    episode = Episode(instruction="put the book in the slot", pick="book", place="the slot")

    args = _normalise_insert_args(
        {
            "pick": "book",
            "place": "the slot",
            "grasp": ["rim", "pca_axis@top", "graspnet"],
            "uncertainty_m": 0.1,
            "insertion_depth": 0.03,
            "nudge": [0.04, 0.04],
            "release_on": "floor",
        },
        episode,
    )

    assert args == {
        "pick": "book",
        "place": "the slot",
        "grasp": ["pca_axis@top", "graspnet"],
        "uncertainty_m": 0.025,
        "insertion_depth": 0.03,
    }


def test_bounded_opening_routing_preserves_learned_grasp_request(
    monkeypatch,
) -> None:
    class Session:
        calls = []

        def __init__(self, runtime, skill):
            del runtime, skill
            type(self).calls = []

        def frame(self):
            return "frame"

        def holding(self):
            return "the gripper is empty"

        def run(self, action, args, thought=""):
            del thought
            type(self).calls.append((action, dict(args)))
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "failure_mode": "empty_grasp",
                    "preflight_only": False,
                },
                observation="the insertion grasp was empty",
            )

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the grasp was empty",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": ["graspnet"]},
                "reason": "use a learned 6-DoF grasp",
            }
        return "{}", {
            "thought": "preflight",
            "action": "insert",
            "args": {"pick": "book", "place": "cabinet shelf"},
        }

    monkeypatch.setattr(react, "Session", Session)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the cabinet shelf",
            pick="book",
            place="cabinet shelf",
            turns=4,
        ),
    )

    assert [action for action, _ in Session.calls] == ["insert", "insert"]
    assert Session.calls[1][1]["grasp"] == ["graspnet"]
    assert _normalise_insert_args(
        {"pick": "book", "place": "cabinet shelf", "grasp": ["graspnet"]},
        Episode(
            instruction="put the book in the cabinet shelf",
            pick="book",
            place="cabinet shelf",
        ),
    )["grasp"] == ["graspnet"]


class _AliasInspector:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.target = None

    def inspect(self, label, *, options, target):
        assert options == [label, "a different destination"]
        self.target = target
        return {"ok": True, "answer": self.answer}


def test_destination_alias_uses_pre_action_crop_for_semantic_check() -> None:
    runtime = _AliasInspector("the wine rack")
    previous = Step(
        action="pickplace",
        args={"pick": "wine bottle", "place": "wooden stand"},
        report={
            "destination_pose": [0.5, -0.3, 0.2],
            "destination_kind": "object",
            "destination_bbox": [10, 20, 100, 140],
        },
    )

    matches, evidence = _verify_destination_alias(runtime, previous, "the wine rack")

    assert matches is True
    assert evidence == {
        "used": True,
        "alias": "wooden stand",
        "matches": True,
        "answer": "the wine rack",
    }
    assert runtime.target.metadata["bbox"] == [10, 20, 100, 140]


def test_destination_alias_mismatch_is_rejected() -> None:
    runtime = _AliasInspector("a different destination")
    previous = Step(
        action="pickplace",
        args={"pick": "black book", "place": "black box"},
        report={
            "destination_pose": [0.5, 0.3, 0.1],
            "destination_kind": "object",
            "destination_bbox": [20, 30, 80, 120],
        },
    )

    matches, evidence = _verify_destination_alias(runtime, previous, "the cabinet shelf")

    assert matches is False
    assert evidence["matches"] is False


class _InventoryRuntime:
    def observe(self):
        return {}

    def _camera(self, observation, wrist=False):
        del observation, wrist
        return (
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.ones((16, 16), dtype=float),
            None,
            None,
        )

    def verify_grasp(self, label):
        del label
        return False

    def list_objects(self):
        return ["wooden stand", "wine bottle"]

    def localize(self, label, detector=None):
        del detector
        return GroundedTarget(
            label=label,
            pose=(0.5, -0.3, 0.2),
            metadata={"bbox": [1, 2, 10, 12]},
        )

    def inspect(self, label, *, options, target):
        del options
        return {
            "ok": True,
            "answer": label if target.label == "wooden stand" else "a different destination",
        }


def test_destination_grounding_failure_tries_equivalent_then_visual_alias(
    monkeypatch,
) -> None:
    decisions = iter(
        [
            {
                "thought": "try the instruction wording",
                "action": "pickplace",
                "args": {"pick": "wine bottle", "place": "the wine holder"},
            },
            {
                "thought": "inventory identifies the structure",
                "action": "pickplace",
                "args": {"pick": "wine bottle", "place": "wooden stand"},
            },
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            value = {
                "what_happened": "nothing moved",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"place": "wooden stand"},
                "reason": "destination was not grounded",
            }
        else:
            value = next(decisions)
        return "{}", value

    def never_grounded(runtime, pick_label, place_label, **options):
        del runtime, options
        return SkillResult(
            success=False,
            status="retryable_failure",
            pick_label=pick_label,
            place_label=place_label,
            attempts=1,
            strategy="test",
            failure_mode="not_grounded",
            report={
                "grounding_failure": "destination",
                "failed_label": place_label,
            },
        )

    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        _InventoryRuntime(),
        never_grounded,
        Episode(
            instruction="put the wine bottle on the wine holder",
            pick="wine bottle",
            place="the wine holder",
            turns=4,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "pickplace",
        "look",
        "pickplace",
    ]
    assert outcome.steps[1]["args"]["place"] == "wine holder"
    assert outcome.steps[3]["args"]["place"] == "wooden stand"


class _OvereagerAliasRuntime(_InventoryRuntime):
    def inspect(self, label, *, options, target):
        del options, target
        return {"ok": True, "answer": label}


def test_inventory_never_accepts_moved_object_as_destination() -> None:
    session = Session(_OvereagerAliasRuntime(), object())
    step = Step(
        action="look",
        args={"verify_destination": "the wine rack", "exclude": "wine bottle"},
    )

    session._do_look(step)

    assert "wine bottle" not in step.report["verified_destination_aliases"]
    bottle = next(row for row in step.report["alias_evidence"] if row["candidate"] == "wine bottle")
    assert bottle == {
        "candidate": "wine bottle",
        "answer": "excluded_moved_object",
        "matches": False,
    }


class _AliasContinuitySession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        self.alias_calls = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "look":
            return Step(
                action=action,
                args=args,
                image="frame",
                report={"verified_destination_aliases": ["wooden stand"]},
                observation="verified wooden stand",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "reliable": True,
                    "destination_kind": "container",
                    "suggested_nudge": [0.01, 0.0],
                },
                observation="slightly off centre",
            )
        if args["place"] in {"the wine holder", "wine holder"}:
            return Step(
                action=action,
                args=args,
                image="frame",
                report={
                    "failure_mode": "not_grounded",
                    "grounding_failure": "destination",
                },
                observation="destination not grounded",
            )
        self.alias_calls += 1
        return Step(
            action=action,
            args=args,
            image="frame",
            success=True,
            report={"failure_mode": ""},
            observation="placed",
        )


def test_verified_destination_alias_persists_for_measured_retry(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            value = {
                "what_happened": "not yet verified",
                "looks_placed": False,
                "next": "check",
                "retry_args": {},
                "reason": "measure it",
            }
        else:
            value = {
                "thought": "start with instruction wording",
                "action": "pickplace",
                "args": {"pick": "wine bottle", "place": "the wine holder"},
            }
        return "{}", value

    monkeypatch.setattr(react, "Session", _AliasContinuitySession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the wine bottle on the wine holder",
            pick="wine bottle",
            place="the wine holder",
            turns=6,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "pickplace",
        "look",
        "pickplace",
        "check",
        "pickplace",
    ]
    assert outcome.steps[3]["args"]["place"] == "wooden stand"
    assert outcome.steps[5]["args"]["place"] == "wooden stand"


class _UnreliableCheckSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={"reliable": False, "moved_points": 0},
                observation="localization is unreliable",
            )
        return Step(action=action, args=args, image="frame", observation="looked")


def test_check_uses_canonical_labels_and_stops_repeated_unreliable_loop(
    monkeypatch,
) -> None:
    decisions = iter(
        [
            {
                "thought": "verify the alias",
                "action": "check",
                "args": {"pick": "black binder", "place": "black box"},
            },
            {
                "thought": "verify the alias again",
                "action": "check",
                "args": {"pick": "black binder", "place": "black box"},
            },
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        value = next(decisions)
        return "{}", value

    monkeypatch.setattr(react, "Session", _UnreliableCheckSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book on top of the cabinet shelf",
            pick="black book",
            place="top of the cabinet shelf",
        ),
    )

    checks = [step for step in outcome.steps if step["action"] == "check"]
    assert len(checks) == 2
    assert all(
        step["args"] == {"pick": "black book", "place": "top of the cabinet shelf"}
        for step in checks
    )
    assert "two checks" in outcome.stopped


class _ReleasedUnreliableLookSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            return Step(
                action=action,
                args=args,
                image="released",
                success=False,
                report={
                    "failure_mode": "verification_inconclusive",
                    "release_xy": [0.1, 0.2],
                },
                observation="released but could not verify",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="released",
                success=False,
                report={"reliable": False, "moved_points": 0},
                observation="localization is unreliable",
            )
        return Step(action=action, args=args, image="released", observation="looked")


def test_post_release_unreliable_look_is_followed_by_one_recheck(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        return "{}", {
            "thought": "start with the manipulation",
            "action": "pickplace",
            "args": {"pick": "book", "place": "caddy"},
        }

    monkeypatch.setattr(react, "Session", _ReleasedUnreliableLookSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the caddy",
            pick="book",
            place="caddy",
            turns=23,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "look",
        "check",
    ]
    assert "two checks" in outcome.stopped


class _PreserveStateSession(_ReleasedUnreliableLookSession):
    def run(self, action, args, thought=""):
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="released",
                success=False,
                report={
                    "reliable": False,
                    "preserve_state": True,
                    "moved_points": 400,
                },
                observation="post-release identity is ambiguous; preserve",
            )
        return super().run(action, args, thought)


def test_ambiguous_post_release_check_stops_without_regrasp_or_look(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the object may be inside",
                "looks_placed": False,
                "next": "check",
                "retry_args": {},
                "reason": "measure it",
            }
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "book", "place": "caddy"},
        }

    monkeypatch.setattr(react, "Session", _PreserveStateSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the caddy",
            pick="book",
            place="caddy",
            turns=23,
        ),
    )

    assert [step["action"] for step in outcome.steps] == ["pickplace", "check"]
    assert "preserved" in outcome.stopped


class _VisualInsertRetrySession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        self.inserts = 0
        self.checks = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "insert":
            self.inserts += 1
            return Step(
                action=action,
                args=args,
                image=f"insert-{self.inserts}",
                success=False,
                report={
                    "failure_mode": "verification_inconclusive",
                    "release_xy": [0.1, -0.1],
                    "preflight_only": False,
                },
                observation="guided insertion was not confirmed",
            )
        if action == "check":
            self.checks += 1
            if self.checks == 1:
                return Step(
                    action=action,
                    args=args,
                    image="object-on-table",
                    success=False,
                    report={
                        "reliable": False,
                        "preserve_state": True,
                        "destination_kind": "region",
                    },
                    observation="metric localization is ambiguous",
                )
            return Step(
                action=action,
                args=args,
                image="inside",
                success=True,
                report={"reliable": True},
                observation="inside",
            )
        return Step(action=action, args=args, image="inside", success=True)


def test_visual_retry_overrides_unreliable_check_and_preserves_insert(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the object visibly fell back to the table",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": "pca_axis@top"},
                "reason": "retry the bounded-opening transport",
            }
        return "{}", {
            "thought": "insert into the drawer",
            "action": "insert",
            "args": {"pick": "box", "place": "top drawer"},
        }

    monkeypatch.setattr(react, "Session", _VisualInsertRetrySession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the box in the top drawer",
            pick="box",
            place="top drawer",
            turns=7,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "insert",
        "check",
        "insert",
        "check",
        "done",
    ]
    assert outcome.steps[2]["args"]["grasp"] == "pca_axis@top"
    assert outcome.success is True


class _RelationalIdentityReviewSession(_VisualInsertRetrySession):
    def run(self, action, args, thought=""):
        if action == "check":
            self.checks += 1
            return Step(
                action=action,
                args=args,
                image=("wrong-lookalike" if self.checks == 1 else "right-instance"),
                success=True,
                report={
                    "reliable": True,
                    "transport_identity_established": self.checks > 1,
                },
                observation="an object is geometrically inside",
            )
        return super().run(action, args, thought)


def test_relational_pick_gets_visual_identity_review_after_positive_check(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "a neighbouring look-alike was moved",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {},
                "reason": "the requested front instance is still at source",
            }
        return "{}", {
            "thought": "insert the selected instance",
            "action": "insert",
            "args": {"pick": "box at the front", "place": "top drawer"},
        }

    monkeypatch.setattr(react, "Session", _RelationalIdentityReviewSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the box at the front in the top drawer",
            pick="box at the front",
            place="top drawer",
            turns=7,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "insert",
        "check",
        "insert",
        "check",
        "done",
    ]
    assert len(outcome.reflections) == 1
    assert outcome.success is True


def test_visible_support_miss_is_recoverable_only_with_independent_evidence() -> None:
    report = {
        "destination_kind": "top_region",
        "destination_region_confidence": 1.0,
        "identity_clear": True,
        "moved_points": 803,
        "landed": False,
        "footprint_containment": {
            "usable": True,
            "contained": False,
            "point_inside_ratio": 0.0,
            "hull_inside_ratio": 0.0,
        },
    }

    assert _visibly_recoverable_miss(report) is True
    assert _visibly_recoverable_miss({**report, "identity_clear": False}) is False
    assert _visibly_recoverable_miss({**report, "landed": True}) is False


class _VisibleSupportMissSession(_ReleasedUnreliableLookSession):
    def __init__(self, runtime, skill) -> None:
        super().__init__(runtime, skill)
        self.picks = 0
        self.checks = 0

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            self.picks += 1
            return Step(
                action=action,
                args=args,
                image=f"release-{self.picks}",
                success=False,
                report={
                    "failure_mode": "verification_inconclusive",
                    "release_xy": [0.1, 0.2],
                },
                observation="released but could not verify",
            )
        if action == "check":
            self.checks += 1
            if self.checks == 1:
                return Step(
                    action=action,
                    args=args,
                    image="back-on-source",
                    success=False,
                    report={
                        "reliable": False,
                        # Exercise the control-layer fallback as well as the
                        # corrected producer flag: old traces can still carry
                        # preserve_state=True with this evidence.
                        "preserve_state": True,
                        "destination_kind": "top_region",
                        "destination_region_confidence": 1.0,
                        "identity_clear": True,
                        "moved_points": 803,
                        "landed": False,
                        "footprint_containment": {
                            "usable": True,
                            "contained": False,
                            "point_inside_ratio": 0.0,
                            "hull_inside_ratio": 0.0,
                        },
                    },
                    observation="identified bowl is wholly outside and below",
                )
            return Step(
                action=action,
                args=args,
                image="placed",
                success=True,
                report={"reliable": True},
                observation="placed",
            )
        return Step(
            action=action,
            args=args,
            image="placed",
            success=True,
            observation="done",
        )


def test_visible_support_miss_retries_instead_of_preserving(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the bowl fell back to the table",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": "rim", "nudge": [0.08, 0.08]},
                "reason": "re-ground and transport it again",
            }
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "black bowl", "place": "top of cabinet"},
        }

    monkeypatch.setattr(react, "Session", _VisibleSupportMissSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the black bowl on top of the cabinet",
            pick="black bowl",
            place="top of cabinet",
            turns=7,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "pickplace",
        "check",
        "done",
    ]
    assert "nudge" not in outcome.steps[2]["args"]
    assert outcome.success is True


class _VisibleContainerMissSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        self.picks = 0
        self.checks = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            self.picks += 1
            return Step(
                action=action,
                args=args,
                image=f"placed-{self.picks}",
                report={
                    "failure_mode": "",
                    "release_xy": [0.6, 0.15],
                },
                observation="released",
            )
        if action == "check":
            self.checks += 1
            if self.checks == 1:
                return Step(
                    action=action,
                    args=args,
                    image="edge-miss",
                    success=False,
                    report={
                        "reliable": False,
                        "destination_kind": "container",
                        "identity_clear": True,
                        "moved_points": 11698,
                        "destination_region_confidence": 1.0,
                        "footprint_containment": {
                            "usable": True,
                            "contained": False,
                            "point_inside_ratio": 0.0,
                            "hull_inside_ratio": 0.0,
                        },
                    },
                    observation="dense bowl footprint is outside the drawer",
                )
            return Step(
                action=action,
                args=args,
                image="inside",
                success=True,
                report={"reliable": True},
                observation="placed",
            )
        return Step(
            action=action,
            args=args,
            image="inside",
            success=True,
            observation="done",
        )


def test_dense_visually_confirmed_container_miss_retries_without_look(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the bowl is visibly caught on the drawer edge",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {
                    "destination": "drawer centre",
                    "release_on": "inside",
                },
                "reason": "the footprint remains outside",
            }
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "akita black bowl", "place": "the top drawer"},
        }

    monkeypatch.setattr(react, "Session", _VisibleContainerMissSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    # This test isolates the legacy broad-transport recovery branch; bounded
    # drawer phrases normally route through insert before reaching it.
    monkeypatch.setattr(react, "_supports_insert", lambda place: False)
    monkeypatch.setattr(react, "_requires_insert", lambda place, instruction="": False)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the black bowl in the top drawer",
            pick="akita black bowl",
            place="the top drawer",
            turns=7,
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "pickplace",
        "check",
        "done",
    ]
    retry = outcome.steps[2]["args"]
    assert retry["destination"] == "inside"
    assert retry["release_on"] == "floor"
    assert retry["place_margin"] == 0.01
    assert "nudge" not in retry


class _GoHomeRuntime:
    def __init__(self) -> None:
        self.home = False
        self.frame_seen_home = False

    def verify_grasp(self, label):
        del label
        return False

    def go_home(self):
        self.home = True

    def observe(self):
        return {}

    def _camera(self, observation, wrist=False):
        del observation, wrist
        self.frame_seen_home = self.home
        return (
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.ones((16, 16), dtype=float),
            None,
            None,
        )


def test_session_goes_home_before_capturing_post_skill_frame() -> None:
    runtime = _GoHomeRuntime()

    def skill(runtime, pick, place, **kwargs):
        del runtime, kwargs
        return SkillResult(
            success=True,
            status="success",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="test",
            report={"release_xy": [0.1, 0.2], "release_z": 0.1},
        )

    step = Session(runtime, skill).run("pickplace", {"pick": "bowl", "place": "drawer"})

    assert step.report["go_home_hook"] == "completed"
    assert runtime.frame_seen_home is True


class _StickyReleaseRuntime(_GoHomeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.holding = True
        self.open_calls = 0

    def verify_grasp(self, label):
        del label
        return self.holding

    def open_gripper(self):
        self.open_calls += 1
        self.holding = False


def test_release_report_forces_open_then_home_despite_sticky_holding_band() -> None:
    runtime = _StickyReleaseRuntime()
    step = Step(action="pickplace", args={}, report={"release_xy": [0.1, 0.2]})
    session = Session(runtime, object())

    session._go_home_after_motion(step)

    assert runtime.open_calls == 1
    assert runtime.home is True
    assert step.report["release_cleanup"] == "reopened"
    assert step.report["go_home_hook"] == "completed"


def test_static_destination_lock_keeps_first_clean_generic_grounding() -> None:
    session = Session(object(), object())
    first = {
        "destination_pose": [0.5, -0.2, 0.1],
        "destination_confidence": 0.8,
        "destination_extent": [0.3, 0.2, 0.05],
        "destination_kind": "object",
        "destination_bbox": [10, 20, 100, 120],
    }
    session._remember_destination("plate", "plate", first)
    session._remember_destination("plate", "plate", {**first, "destination_pose": [1.7, 0.9, -0.4]})

    locked = session._destination_locks["plate"]
    assert np.allclose(locked.pose, [0.5, -0.2, 0.1])
    assert locked.metadata["bbox"] == [10, 20, 100, 120]


class _CheckEvidenceRuntime:
    def __init__(
        self,
        *,
        rebind: bool = False,
        cavity: bool = False,
        identity_match: bool = True,
    ) -> None:
        self.rebind = rebind
        self.cavity = cavity
        self.identity_match = identity_match
        self.inspect_calls = 0

    def localize_many(self, labels):
        label = labels[0]
        pose = (0.40, 0.40, 0.10) if self.rebind else (0.0, 0.0, 0.10)
        return {
            label: GroundedTarget(
                label=label,
                pose=pose,
                metadata={
                    "n_points": 1500 if self.rebind else 1200,
                    "extent": [0.04, 0.04, 0.04],
                    "top_z": pose[2],
                },
            )
        }

    def localize(self, label, detector=None):
        del detector
        if self.rebind and "now resting at" in label:
            return GroundedTarget(
                label=label,
                pose=(0.0, 0.0, 0.10),
                metadata={"n_points": 600, "extent": [0.04, 0.04, 0.04], "top_z": 0.12},
            )
        raise AssertionError(f"unexpected re-grounding of {label!r}")

    def object_points(self, label):
        del label
        xs = np.linspace(-0.018, 0.018, 15)
        ys = np.linspace(-0.018, 0.018, 15)
        return np.asarray([[x, y, 0.10] for x in xs for y in ys])

    def image_axes(self):
        return {"right": [1.0, 0.0], "down": [0.0, 1.0]}

    def inspect(self, prompt, *, options, target):
        del prompt, target
        self.inspect_calls += 1
        return {
            "ok": True,
            "answer": options[0] if self.identity_match else options[1],
        }


def test_check_uses_latest_real_release_not_later_no_motion_failure() -> None:
    session = Session(_CheckEvidenceRuntime(), object())
    session.steps.extend(
        [
            Step(
                action="pickplace",
                args={"pick": "book", "place": "plate"},
                report={
                    "destination_pose": [0.0, 0.0, 0.10],
                    "destination_rim_z": 0.10,
                    "destination_extent": [0.2, 0.2, 0.01],
                    "release_xy": [0.0, 0.0],
                },
            ),
            Step(
                action="pickplace",
                args={"pick": "book", "place": "plate"},
                report={"failure_mode": "not_grounded", "grounding_failure": "pick"},
            ),
        ]
    )
    check = Step(action="check", args={"pick": "book", "place": "plate"})

    session._do_check(check)

    assert check.report["release_error_cm"] == 0.0
    assert check.report["offset_cm"] == 0.0


def test_check_rebinds_pre_motion_qualifier_by_release_relation() -> None:
    session = Session(_CheckEvidenceRuntime(rebind=True), object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "middle bowl", "place": "plate"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
            },
        )
    )
    check = Step(action="check", args={"pick": "middle bowl", "place": "plate"})

    session._do_check(check)

    assert check.success is True
    assert check.report["localization_source"] == "post_release_relation"
    assert check.report["release_error_cm"] == 0.0


def test_relation_rebind_requires_independent_crop_identity() -> None:
    session = Session(_CheckEvidenceRuntime(rebind=True, identity_match=False), object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "requested product", "place": "tray"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
            },
        )
    )
    check = Step(action="check", args={"pick": "requested product", "place": "tray"})

    session._do_check(check)

    assert check.success is False
    assert check.report["localization_source"] == "post_release_relation"
    assert check.report["identity_clear"] is False
    assert check.report["failure_mode"] == "wrong_object_grasp"
    assert check.report["preserve_state"] is False


def test_relation_rebind_uses_verified_pregrasp_identity_continuity() -> None:
    runtime = _CheckEvidenceRuntime(rebind=True, identity_match=False)
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "requested product", "place": "tray"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": True,
                    "answer": "requested product",
                },
            },
        )
    )
    check = Step(action="check", args={"pick": "requested product", "place": "tray"})

    session._do_check(check)

    assert check.success is True
    assert check.report["identity_clear"] is True
    # The post-release crop is occluded and need not relitigate an identity
    # already established before the continuous grasp/carry/release action.
    assert runtime.inspect_calls == 0


def test_explicit_pregrasp_identity_mismatch_cannot_be_rebound_as_success() -> None:
    runtime = _CheckEvidenceRuntime(rebind=True, identity_match=True)
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "requested product", "place": "tray"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": False,
                    "answer": "a different object",
                },
            },
        )
    )
    check = Step(action="check", args={"pick": "requested product", "place": "tray"})

    session._do_check(check)

    assert check.success is False
    assert check.report["failure_mode"] == "wrong_object_grasp"


def test_advisory_pregrasp_mismatch_can_be_overruled_by_release_crop() -> None:
    runtime = _CheckEvidenceRuntime(rebind=True, identity_match=True)
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "butter at the front", "place": "tray"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": False,
                    "answer": "a different object",
                    "veto_enabled": False,
                },
            },
        )
    )
    check = Step(action="check", args={"pick": "butter at the front", "place": "tray"})

    session._do_check(check)

    assert check.success is True
    assert check.report["identity_clear"] is True


def test_transient_instance_keeps_causal_transport_identity_when_crop_is_occluded() -> None:
    runtime = _CheckEvidenceRuntime(rebind=True, identity_match=False)
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="pickplace",
            args={"pick": "butter at the front", "place": "tray"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
                "transport_identity_established": True,
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": False,
                    "answer": "a different object",
                    "veto_enabled": False,
                },
            },
        )
    )
    check = Step(action="check", args={"pick": "butter at the front", "place": "tray"})

    session._do_check(check)

    assert check.success is True
    assert check.report["transport_identity_established"] is True
    assert check.report["identity_clear"] is True


def test_empty_grasp_invalidates_stale_unique_pick_geometry() -> None:
    def empty_skill(runtime, pick, place, **kwargs):
        del runtime, kwargs
        return SkillResult(
            success=False,
            status="retryable",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="test",
            failure_mode="empty_grasp",
            report={
                "pick_pose": [0.6, 0.1, 0.05],
                "pick_extent": [0.1, 0.1, 0.1],
                "pick_n_points": 500,
            },
        )

    session = Session(object(), empty_skill)
    step = Step(action="pickplace", args={"pick": "moka pot", "place": "stove"})

    session._do_pickplace(step)

    assert "moka pot" not in session._pick_locks
    assert step.report["pick_lock_after_failure"] == "invalidated_for_reground"


def test_empty_grasp_keeps_relational_instance_lock() -> None:
    def empty_skill(runtime, pick, place, **kwargs):
        del runtime, kwargs
        return SkillResult(
            success=False,
            status="retryable",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="test",
            failure_mode="empty_grasp",
            report={
                "pick_pose": [0.6, 0.1, 0.05],
                "pick_extent": [0.1, 0.1, 0.1],
                "pick_n_points": 500,
            },
        )

    session = Session(object(), empty_skill)
    step = Step(
        action="pickplace",
        args={"pick": "butter at the front", "place": "drawer"},
    )

    session._do_pickplace(step)

    assert "butter at the front" in session._pick_locks


def test_articulate_reuses_matching_persistent_opening_polygon() -> None:
    calls = []

    def articulate_skill(runtime, target, **kwargs):
        del runtime
        calls.append(kwargs)
        return SkillResult(
            success=False,
            status="retryable",
            pick_label=target,
            place_label=target,
            attempts=1,
            strategy="articulate",
            failure_mode="not_grounded",
            report={},
        )

    session = Session(object(), object(), articulate_skill=articulate_skill)
    polygon = [[0.5, -0.2], [0.7, -0.2], [0.7, -0.1], [0.5, -0.1]]
    session._destination_locks["top drawer of the cabinet"] = GroundedTarget(
        label="the top drawer of the cabinet",
        pose=(0.6, -0.15, 0.1),
        metadata={"region_polygon_xy": polygon},
    )
    step = Step(
        action="articulate",
        args={"target": "cabinet", "part": "top drawer", "goal": "closed"},
    )

    session._do_articulate(step)

    assert calls[0]["opening_polygon_hint"] == polygon


def test_prismatic_retry_remembers_original_handle_height() -> None:
    calls = []

    def articulate_skill(runtime, target, **kwargs):
        del runtime
        calls.append(kwargs)
        return SkillResult(
            success=False,
            status="retryable",
            pick_label=target,
            place_label=target,
            attempts=1,
            strategy="articulate",
            failure_mode="insufficient_motion",
            report={
                "mechanism": "prismatic",
                "handle_before": [0.4, 0.1, 0.072],
                "motion_axis": [0.2, 0.98],
            },
        )

    session = Session(object(), object(), articulate_skill=articulate_skill)
    args = {
        "target": "cabinet",
        "part": "bottom drawer",
        "goal": "open",
    }
    session._do_articulate(Step(action="articulate", args=args))
    session._do_articulate(Step(action="articulate", args=args))

    assert calls[0]["handle_z_hint"] is None
    assert calls[1]["handle_z_hint"] == 0.072
    assert calls[0]["axis_hint"] is None
    assert calls[1]["axis_hint"] == [0.2, 0.98]


class _StackBoundaryRuntime(_CheckEvidenceRuntime):
    def localize_many(self, labels):
        label = labels[0]
        return {
            label: GroundedTarget(
                label=label,
                pose=(0.028, 0.0, 0.12),
                metadata={
                    "n_points": 1200,
                    "extent": [0.04, 0.04, 0.04],
                    "top_z": 0.14,
                },
            )
        }


def test_stack_check_keeps_conservative_margin_under_measurement_uncertainty() -> None:
    session = Session(_StackBoundaryRuntime(), object())
    session.steps.append(
        Step(
            action="stack",
            args={"pick": "bowl", "support": "plate"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.15, 0.15, 0.01],
                "release_xy": [0.0, 0.0],
            },
        )
    )
    check = Step(action="check", args={"pick": "bowl", "place": "plate"})

    session._do_check(check)

    assert check.success is False
    assert check.report["offset_cm"] == 2.8
    assert check.report["tolerance_cm"] == 2.0
    assert check.report["measurement_uncertainty_cm"] == 0.0


def test_stack_check_rejects_relation_mask_at_support_plane_height() -> None:
    session = Session(_CheckEvidenceRuntime(rebind=True), object())
    session.steps.append(
        Step(
            action="stack",
            args={"pick": "white bowl", "support": "plate"},
            report={
                "destination_pose": [0.0, 0.0, 0.10],
                "destination_rim_z": 0.10,
                "destination_extent": [0.2, 0.2, 0.01],
                "release_xy": [0.0, 0.0],
                "pick_extent": [0.08, 0.08, 0.036],
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": True,
                    "answer": "white bowl",
                },
            },
        )
    )
    check = Step(action="check", args={"pick": "white bowl", "place": "plate"})

    session._do_check(check)

    assert check.success is False
    assert check.report["centred"] is True
    assert check.report["landed"] is False
    assert check.report["required_support_rise_cm"] == 0.7


def test_guided_insert_uses_containment_not_visible_centroid_rim_height() -> None:
    session = Session(_CheckEvidenceRuntime(cavity=True), object())
    polygon = [[-0.08, -0.08], [0.08, -0.08], [0.08, 0.08], [-0.08, 0.08]]
    session.steps.append(
        Step(
            action="insert",
            args={"pick": "book", "place": "right compartment of caddy"},
            report={
                "destination_pose": [0.0, 0.0, 0.08],
                "destination_rim_z": 0.08,
                "destination_floor_z": 0.0,
                "destination_extent": [0.16, 0.16, 0.08],
                "destination_region_polygon_xy": polygon,
                "destination_region_confidence": 1.0,
                "destination_region_source": "rgbd_cavity",
                "release_xy": [0.0, 0.0],
                "achieved_insertion_depth_cm": 5.0,
                "preflight_only": False,
            },
        )
    )
    check = Step(action="check", args={"pick": "book", "place": "right compartment of caddy"})

    session._do_check(check)

    assert check.success is True
    assert check.report["guided_insert_evidence"] is True
    assert check.report["footprint_containment"]["contained"] is True


class _ReleasedThenEmptySession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        self.picks = 0
        self.checks = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            self.picks += 1
            if self.picks == 1:
                return Step(
                    action=action,
                    args=args,
                    image="frame",
                    report={
                        "failure_mode": "verification_inconclusive",
                        "release_xy": [0.2, -0.1],
                    },
                    observation="released but verification was inconclusive",
                )
            return Step(
                action=action,
                args=args,
                image="frame",
                report={"failure_mode": "empty_grasp"},
                observation="empty_grasp",
            )
        if action == "check":
            self.checks += 1
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "reliable": True,
                    "landed": False,
                    "centred": False,
                    "identity_clear": True,
                    "offset_cm": 8.0,
                    "suggested_nudge": [0.04, 0.0],
                },
                observation="missed",
            )
        return Step(action=action, args=args, image="frame")


def test_empty_grasp_after_release_routes_to_check_not_another_pick(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if not system.startswith("You are diagnosing"):
            return "{}", {
                "thought": "start",
                "action": "pickplace",
                "args": {"pick": "object", "place": "target"},
            }
        diagnosis = {
            "what_happened": "the object may already have moved",
            "looks_placed": False,
            "next": "retry",
            "retry_args": {"grasp": "rim"},
            "reason": "try another grasp",
        }
        return "{}", diagnosis

    monkeypatch.setattr(react, "Session", _ReleasedThenEmptySession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(instruction="put object on target", pick="object", place="target", turns=4),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "pickplace",
        "check",
    ]


def test_visual_agent_done_overrides_advisory_failed_placement_check(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if not system.startswith("You are diagnosing"):
            return "{}", {
                "thought": "move it",
                "action": "pickplace",
                "args": {"pick": "object", "place": "target"},
            }
        return "{}", {
            "what_happened": "the object is visibly resting inside target",
            "looks_placed": True,
            "next": "done",
            "retry_args": {},
            "reason": "the visual state satisfies the instruction",
        }

    monkeypatch.setattr(react, "Session", _ReleasedThenEmptySession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(instruction="put object on target", pick="object", place="target", turns=4),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "done",
    ]
    assert outcome.success is True
    assert "agent judged it complete" in outcome.stopped


class _PushSelectionSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        self.checks = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "pickplace":
            return Step(
                action=action,
                args=args,
                image="frame",
                report={"failure_mode": "verification_inconclusive", "release_xy": [0.2, -0.1]},
                observation="released",
            )
        if action == "push":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=True,
                report={"failure_mode": ""},
                observation="pushed",
            )
        if action == "check":
            self.checks += 1
            good = self.checks > 1
            return Step(
                action=action,
                args=args,
                image="frame",
                success=good,
                report={
                    "reliable": True,
                    "landed": True,
                    "centred": good,
                    "destination_kind": "surface",
                    "identity_clear": True,
                    "offset_cm": 0.5 if good else 2.0,
                    "suggested_nudge": [0.0, 0.0] if good else [-0.018, 0.004],
                },
                observation="placed" if good else "two centimetres off",
            )
        return Step(action=action, args=args, image="frame", success=True)


class _InsertPreflightSession:
    def __init__(self, runtime, skill) -> None:
        del runtime, skill

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "insert":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "failure_mode": "no_feasible_insertion",
                    "preflight_only": True,
                },
                observation="top-down footprint cannot fit; scene unchanged",
            )
        if action == "pickplace":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "failure_mode": "verification_inconclusive",
                    "release_xy": [0.2, -0.1],
                },
                observation="released",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=True,
                report={"reliable": True, "centred": True, "landed": True},
                observation="placed",
            )
        return Step(action=action, args=args, image="frame", success=True)


def test_narrow_opening_never_downgrades_to_top_down_pickplace(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if system.startswith("You are diagnosing"):
            return "{}", {
                "what_happened": "the broad transport released the object",
                "looks_placed": False,
                "next": "check",
                "retry_args": {},
                "reason": "measure it",
            }
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "book", "place": "the slot"},
        }

    monkeypatch.setattr(react, "Session", _InsertPreflightSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the slot",
            pick="book",
            place="the slot",
            turns=5,
            max_insert_calls=2,
        ),
    )

    actions = [step["action"] for step in outcome.steps]
    assert actions == ["insert"]
    assert "pickplace" not in actions
    assert "configuration space is empty" in outcome.stopped


class _ExhaustedInsertSession:
    calls: list[str] = []

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).calls = []

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del args, thought
        type(self).calls.append(action)
        return Step(
            action=action,
            args={},
            image="frame",
            success=False,
            report={"failure_mode": "empty_grasp"},
            observation="the grasp was empty; scene unchanged",
        )


def test_exhausted_insert_budget_stops_without_passive_look_loop(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        return "{}", {
            # The policy may propose a broad transport after insert failures;
            # bounded-opening routing must reject it without entering a look
            # loop once the insertion budget is exhausted.
            "thought": "retry transport",
            "action": "pickplace",
            "args": {"pick": "book", "place": "cabinet shelf"},
        }

    monkeypatch.setattr(react, "Session", _ExhaustedInsertSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the cabinet shelf",
            pick="book",
            place="cabinet shelf",
            turns=20,
            max_insert_calls=2,
        ),
    )

    assert _ExhaustedInsertSession.calls == ["insert", "insert"]
    assert "look" not in _ExhaustedInsertSession.calls
    assert "budget exhausted without a release" in outcome.stopped


class _RepeatedLookSession:
    calls = 0

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).calls = 0

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del args, thought
        assert action == "look"
        type(self).calls += 1
        return Step(
            action="look",
            args={},
            image="frame",
            success=True,
            report={"inventory": ["book", "shelf"]},
            observation="the detector names: book, shelf",
        )


def test_two_identical_passive_looks_stop_no_information_loop(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        return "{}", {"thought": "inspect", "action": "look", "args": {}}

    monkeypatch.setattr(react, "Session", _RepeatedLookSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="inspect before deciding",
            pick="book",
            place="plate",
            turns=20,
        ),
    )

    assert _RepeatedLookSession.calls == 2
    assert "same inventory" in outcome.stopped


class _NoOpeningThenSurfaceSession:
    calls: list[tuple[str, dict]] = []

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).calls = []

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        type(self).calls.append((action, dict(args)))
        if action == "insert":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "failure_mode": "no_opening_geometry",
                    "preflight_only": True,
                },
                observation="no bounded opening polygon; scene unchanged",
            )
        if action == "look":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=True,
                report={"verified_destination_aliases": ["wooden stand"]},
                observation="verified alias: wooden stand",
            )
        if action == "pickplace":
            return Step(
                action=action,
                args=args,
                image="released",
                success=True,
                report={"release_xy": [0.2, -0.1], "failure_mode": ""},
                observation="released on support",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="checked",
                success=True,
                report={"reliable": True, "centred": True, "landed": True},
                observation="placed",
            )
        return Step(action=action, args=args, image="frame", success=True)


def test_two_no_opening_preflights_allow_one_surface_fallback(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        return "{}", {
            "thought": "start",
            "action": "insert",
            "args": {"pick": "bottle", "place": "inside rack"},
        }

    monkeypatch.setattr(react, "Session", _NoOpeningThenSurfaceSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put bottle inside rack",
            pick="bottle",
            place="inside rack",
            turns=8,
            max_insert_calls=4,
        ),
    )

    assert [action for action, _ in _NoOpeningThenSurfaceSession.calls] == [
        "insert",
        "look",
        "insert",
        "look",
        "pickplace",
        "check",
        "done",
    ]
    assert _NoOpeningThenSurfaceSession.calls[2][1]["place"] == "wooden stand"
    assert _NoOpeningThenSurfaceSession.calls[4][1]["place"] == "wooden stand"


class _SourceAliasRecoverySession:
    calls: list[tuple[str, dict]] = []

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).calls = []

    def frame(self):
        return "frame"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        type(self).calls.append((action, dict(args)))
        if action == "insert":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "failure_mode": "not_grounded",
                    "grounding_failure": "pick",
                    "preflight_only": True,
                },
                observation="source was not grounded; scene unchanged",
            )
        if action == "look":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=True,
                report={"verified_source_aliases": ["snack bar"]},
                observation="verified source alias: snack bar",
            )
        return Step(action=action, args=args, image="frame", success=False)


def test_pick_grounding_failure_inspects_source_once_and_keeps_selector(
    monkeypatch,
) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del system, prompt, images, model, max_tokens
        return "{}", {
            "thought": "start",
            "action": "pickplace",
            "args": {"pick": "butter at the front", "place": "slot"},
        }

    monkeypatch.setattr(react, "Session", _SourceAliasRecoverySession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put butter at the front in slot",
            pick="butter at the front",
            place="slot",
            turns=10,
            max_insert_calls=4,
        ),
    )

    assert [action for action, _ in _SourceAliasRecoverySession.calls] == [
        "insert",
        "look",
        "insert",
        "look",
    ]
    assert _SourceAliasRecoverySession.calls[1][1] == {"verify_source": "butter at the front"}
    assert _SourceAliasRecoverySession.calls[2][1]["pick"] == ("snack bar at the front")
    assert "no new admissible referent" in outcome.stopped


def test_reliable_small_lateral_miss_uses_measured_push(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if not system.startswith("You are diagnosing"):
            return "{}", {
                "thought": "start",
                "action": "pickplace",
                "args": {"pick": "bowl", "place": "plate"},
            }
        return "{}", {
            "what_happened": "inspect the result",
            "looks_placed": False,
            "next": "check",
            "retry_args": {},
            "reason": "measure it",
        }

    monkeypatch.setattr(react, "Session", _PushSelectionSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(instruction="put bowl on plate", pick="bowl", place="plate", turns=5),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "push",
        "check",
        "done",
    ]
    assert outcome.steps[2]["args"] == {"pick": "bowl", "place": "plate", "nudge": [-0.018, 0.004]}


def test_visual_retry_on_reliable_small_miss_still_uses_push(monkeypatch) -> None:
    """Agent authority chooses retry; geometry chooses its safest tool."""

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if not system.startswith("You are diagnosing"):
            return "{}", {
                "thought": "start",
                "action": "pickplace",
                "args": {"pick": "bowl", "place": "plate"},
            }
        return "{}", {
            "what_happened": "the bowl is supported just off centre",
            "looks_placed": False,
            "next": "retry",
            "retry_args": {"nudge": [-0.018, 0.004]},
            "reason": "apply the measured correction",
        }

    monkeypatch.setattr(react, "Session", _PushSelectionSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(instruction="put bowl on plate", pick="bowl", place="plate", turns=5),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "push",
        "check",
        "done",
    ]
    assert outcome.steps[2]["args"] == {"pick": "bowl", "place": "plate", "nudge": [-0.018, 0.004]}


class _NoMotionPushSession(_PushSelectionSession):
    def run(self, action, args, thought=""):
        if action == "push":
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={"failure_mode": "no_motion"},
                observation="contact did not move the object",
            )
        if action == "check":
            self.checks += 1
            return Step(
                action=action,
                args=args,
                image="frame",
                success=False,
                report={
                    "reliable": True,
                    "landed": True,
                    "centred": False,
                    "destination_kind": "surface",
                    "identity_clear": True,
                    "offset_cm": 2.0,
                    "suggested_nudge": [-0.018, 0.004],
                },
                observation="still two centimetres off",
            )
        return super().run(action, args, thought)


def test_no_motion_push_is_not_immediately_repeated(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        if not system.startswith("You are diagnosing"):
            return "{}", {
                "thought": "start",
                "action": "pickplace",
                "args": {"pick": "bowl", "place": "plate"},
            }
        return "{}", {
            "what_happened": "inspect the result",
            "looks_placed": False,
            "next": "check",
            "retry_args": {},
            "reason": "measure it",
        }

    monkeypatch.setattr(react, "Session", _NoMotionPushSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(instruction="put bowl on plate", pick="bowl", place="plate", turns=5),
    )

    assert [step["action"] for step in outcome.steps] == [
        "pickplace",
        "check",
        "push",
        "check",
        "pickplace",
    ]


class _ReleasedThenEmptyInsertSession:
    insert_calls = 0

    def __init__(self, runtime, skill) -> None:
        del runtime, skill
        type(self).insert_calls = 0

    def frame(self):
        return "before"

    def holding(self):
        return "the gripper is empty"

    def run(self, action, args, thought=""):
        del thought
        if action == "insert":
            type(self).insert_calls += 1
            if type(self).insert_calls == 1:
                return Step(
                    action=action,
                    args=args,
                    image="book-in-compartment",
                    success=False,
                    report={
                        "release_xy": [0.2, 0.1],
                        "failure_mode": "placement_miss",
                    },
                    observation="released; footprint check remains conservative",
                )
            return Step(
                action=action,
                args=args,
                image="book-in-compartment",
                success=False,
                report={"failure_mode": "empty_grasp"},
                observation="the follow-up grasp was empty; nothing moved",
            )
        if action == "check":
            return Step(
                action=action,
                args=args,
                image="book-in-compartment",
                success=False,
                report={
                    "reliable": True,
                    "failure_mode": "placement_miss",
                    "transport_identity_established": True,
                },
                observation="the footprint verifier withheld containment",
            )
        return Step(action=action, args=args, image="book-in-compartment", success=True)


def test_visual_done_after_no_motion_regrasp_preserves_earlier_release(
    monkeypatch,
) -> None:
    reflections = iter(
        [
            {
                "what_happened": "the book is in the named compartment",
                "looks_placed": False,
                "next": "retry",
                "retry_args": {"grasp": ["pca_axis@low"]},
                "reason": "one verification retry",
            },
            {
                "what_happened": "the unchanged book is visibly resting inside",
                "looks_placed": True,
                "next": "done",
                "retry_args": {},
                "reason": "another grasp would disturb the placed book",
            },
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens=900):
        del prompt, images, model, max_tokens
        assert system.startswith("You are diagnosing")
        value = next(reflections)
        return "{}", value

    monkeypatch.setattr(react, "Session", _ReleasedThenEmptyInsertSession)
    monkeypatch.setattr(react, "_ask_json_reply", fake_ask)
    outcome = react.run_episode(
        object(),
        object(),
        Episode(
            instruction="put the book in the right compartment",
            pick="book",
            place="right compartment",
            turns=8,
            max_insert_calls=4,
            initial_skill="insert",
        ),
    )

    assert [step["action"] for step in outcome.steps] == [
        "insert",
        "check",
        "insert",
        "done",
    ]
    assert outcome.success is True


def test_reliable_planar_relation_miss_can_be_micro_pushed() -> None:
    report = {
        "reliable": True,
        "landed": True,
        "centred": False,
        "destination_kind": "relation",
        "identity_clear": True,
        "offset_cm": 2.0,
        "suggested_nudge": [-0.018, 0.004],
    }

    assert react._check_supports_push(report) is True


def test_polygon_check_rejects_a_rim_straddling_footprint() -> None:
    polygon = np.asarray([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    inside = np.column_stack(
        [
            np.linspace(-0.03, 0.03, 80),
            np.linspace(-0.02, 0.02, 80),
            np.full(80, 0.08),
        ]
    )
    straddling = inside.copy()
    straddling[:35, 0] += 0.10

    assert _footprint_in_polygon(inside, polygon)["contained"] is True
    assert _footprint_in_polygon(straddling, polygon)["contained"] is False


class _DenseCropRuntime:
    def __init__(self) -> None:
        rows, cols = 90, 80
        ys, xs = np.mgrid[:rows, :cols]
        self.pixels = np.stack([xs, ys], axis=-1)
        self.points = np.stack(
            [
                (xs - 40) * 0.002,
                (ys - 45) * 0.002,
                np.full((rows, cols), 0.14),
            ],
            axis=-1,
        ).astype(float)
        # A bounded cavity floor and a thin strip of table at the crop edge.
        self.points[15:75, 18:64, 2] = 0.06
        self.points[86:, :, 2] = -0.03

    def probe_bbox(self, bbox):
        return {"bbox": list(bbox), "points": self.points, "pixels": self.pixels}


def test_dense_cavity_grounding_keeps_opening_polygon_and_safe_point() -> None:
    target = GroundedTarget(
        label="right compartment",
        pose=(0.0, 0.0, 0.1),
        metadata={"bbox": [0, 0, 80, 90], "top_z": 0.14},
    )
    trace = []

    region = _cavity_region(_DenseCropRuntime(), target, trace)

    assert region is not None
    assert region.metadata["region_source"] == "rgbd_cavity"
    assert region.metadata["region_area_pixels"] > 2000
    assert abs(region.pose[0]) < 0.01
    assert abs(region.pose[1]) < 0.03
    assert abs(region.pose[2] - 0.06) < 0.005
    assert len(region.metadata["region_polygon_xy"]) >= 4


def test_dense_cavity_polygon_rejects_single_deprojection_edge_outlier() -> None:
    runtime = _DenseCropRuntime()
    # A valid floor pixel at a depth discontinuity has a wildly wrong XY
    # deprojection.  It must not become a convex-hull corner.
    runtime.points[15, 18, :2] = [-1.5, 0.9]
    target = GroundedTarget(
        label="right compartment",
        pose=(0.0, 0.0, 0.1),
        metadata={"bbox": [0, 0, 80, 90], "top_z": 0.14},
    )

    region = _cavity_region(runtime, target, [])

    assert region is not None
    extent = np.asarray(region.metadata["extent"][:2])
    assert np.all(extent < 0.20)
    assert abs(region.pose[0]) < 0.02
    assert len(region.metadata["region_polygon_xy"]) >= 4


def test_dense_cavity_polygon_rejects_locally_consistent_depth_spur() -> None:
    runtime = _DenseCropRuntime()
    # Model a discontinuity wider than the local repair window.  Every nearby
    # cavity pixel agrees with the bad deprojection, but the resulting thin
    # world-space spur is inconsistent with the connected component's bulk.
    runtime.points[15:21, 18:24, :2] = [-0.55, 0.42]
    target = GroundedTarget(
        label="right compartment",
        pose=(0.0, 0.0, 0.1),
        metadata={"bbox": [0, 0, 80, 90], "top_z": 0.14},
    )

    region = _cavity_region(runtime, target, [])

    assert region is not None
    extent = np.asarray(region.metadata["extent"][:2])
    assert np.all(extent < 0.20)
    assert abs(region.pose[0]) < 0.02
    assert abs(region.pose[1]) < 0.03


def test_synthetic_cavity_polygon_retains_orientation() -> None:
    target = GroundedTarget(
        label="compartment",
        pose=(0.0, 0.0, 0.05),
        kind="region",
        metadata={
            "synthetic": True,
            "region_polygon_xy": [
                [-0.08, -0.02],
                [0.08, -0.02],
                [0.08, 0.02],
                [-0.08, 0.02],
            ],
        },
    )

    axis = _long_axis(object(), target)
    assert axis is not None
    assert abs(np.sin(axis)) < 0.05
    assert _destination_aspect(object(), target) > 3.0


def test_session_reuses_first_dense_destination_on_retry() -> None:
    calls = []

    def skill(runtime, pick, place, **kwargs):
        del runtime
        calls.append((pick, place, kwargs))
        report = {
            "destination_pose": [0.31, 0.02, 0.06],
            "destination_kind": "region",
            "destination_extent": [0.08, 0.13, 0.08],
            "destination_synthetic": True,
            "destination_rim_z": 0.14,
            "destination_floor_z": 0.06,
            "destination_region_source": "rgbd_cavity",
            "destination_region_confidence": 0.91,
            "destination_region_area_pixels": 800,
            "destination_region_polygon_xy": [
                [0.27, -0.04],
                [0.35, -0.04],
                [0.35, 0.08],
                [0.27, 0.08],
            ],
        }
        return SkillResult(
            success=False,
            status="retryable_failure",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="phase1_policy_api",
            failure_mode="verification_inconclusive",
            report=report,
        )

    session = Session(object(), skill, push_skill=object())
    session._do_pickplace(Step(action="pickplace", args={"pick": "book", "place": "compartment"}))
    session._do_pickplace(
        Step(
            action="pickplace",
            args={"pick": "book", "place": "compartment", "destination": "inside"},
        )
    )

    assert "_destination_target" not in calls[0][2]
    locked = calls[1][2]["_destination_target"]
    assert isinstance(locked, GroundedTarget)
    assert locked.pose == (0.31, 0.02, 0.06)


def test_session_keeps_spatial_pick_instance_across_empty_grasp_retry() -> None:
    calls = []

    def skill(runtime, pick, place, **kwargs):
        del runtime
        calls.append((pick, place, kwargs))
        return SkillResult(
            success=False,
            status="retryable_failure",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="phase1_policy_api",
            failure_mode="empty_grasp",
            report={
                "pick_pose": [0.70, 0.06, 0.01],
                "pick_extent": [0.084, 0.049, 0.027],
                "pick_confidence": 0.72,
                "pick_top_z": 0.027,
                "pick_n_points": 900,
                "pick_principal_axis": [0.0, 1.0],
            },
        )

    session = Session(object(), skill, push_skill=object())
    args = {
        "pick": "the butter at the front",
        "place": "the top drawer of the cabinet",
    }
    session._do_pickplace(Step(action="pickplace", args=args))
    session._do_pickplace(Step(action="pickplace", args=args))

    assert "_pick_target" not in calls[0][2]
    locked = calls[1][2]["_pick_target"]
    assert isinstance(locked, GroundedTarget)
    assert locked.pose == (0.70, 0.06, 0.01)
    assert locked.metadata["points_label"] == "the butter at the front"


def test_insert_reuses_spatial_pick_instance_across_empty_grasp_retry() -> None:
    calls = []

    def insert_skill(runtime, pick, place, **kwargs):
        del runtime
        calls.append((pick, place, kwargs))
        return SkillResult(
            success=False,
            status="retryable_failure",
            pick_label=pick,
            place_label=place,
            attempts=1,
            strategy="insert",
            failure_mode="empty_grasp",
            report={
                "preflight_only": False,
                "pick_pose": [0.70, 0.06, 0.01],
                "pick_extent": [0.084, 0.049, 0.027],
                "pick_confidence": 0.72,
                "pick_top_z": 0.027,
                "pick_n_points": 900,
                "pick_principal_axis": [0.0, 1.0],
                "pregrasp_identity_evidence": {
                    "used": True,
                    "matches": True,
                },
            },
        )

    session = Session(object(), object(), push_skill=object(), insert_skill=insert_skill)
    args = {
        "pick": "the butter at the front",
        "place": "the top drawer of the cabinet",
    }
    session._do_insert(Step(action="insert", args=args))
    session._do_insert(Step(action="insert", args=args))

    assert "_pick_target" not in calls[0][2]
    locked = calls[1][2]["_pick_target"]
    assert isinstance(locked, GroundedTarget)
    assert locked.pose == (0.70, 0.06, 0.01)
    assert locked.metadata["points_label"] == "the butter at the front"


class _StackCheckRuntime:
    def localize_many(self, labels):
        return {
            labels[0]: GroundedTarget(
                label=labels[0],
                pose=(0.6198, 0.2032, 0.0545),
                metadata={"n_points": 1963, "extent": [0.095, 0.091, 0.06]},
            )
        }

    def localize(self, label, **kwargs):
        raise AssertionError(f"stack check re-grounded ambiguous support {label!r}")

    def image_axes(self):
        return {"right": [1.0, 0.0], "down": [0.0, 1.0]}


def test_stack_check_reuses_the_pre_action_support_pose() -> None:
    session = Session(_StackCheckRuntime(), object())
    session.steps.append(
        Step(
            action="stack",
            args={"pick": "the bowl at the front", "support": "the bowl in the middle"},
            report={
                "destination_pose": [0.611, 0.2, 0.025],
                "destination_extent": [0.1053, 0.1029, 0.0589],
                "destination_rim_z": 0.0377,
            },
        )
    )
    check = Step(
        action="check",
        args={"pick": "the bowl at the front", "place": "the bowl in the middle"},
    )

    session._do_check(check)

    assert check.success is True
    assert check.report["offset_cm"] < 1.0


def test_reflection_prompt_treats_small_support_contact_as_insufficient() -> None:
    assert "small support such as a plate" in react.REFLECT
    assert "outside tolerance" in react.REFLECT
    assert "Return `retry` unless" in react.REFLECT
    assert "overlap at the support edge cannot" in react.REFLECT
