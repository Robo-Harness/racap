from evolution.harness.critic import _causal_packet, _compact, _compact_human_trace


def test_compact_removes_camera_tensors_but_keeps_runtime_exception() -> None:
    value = {
        "observe": {
            "images": {"rgb": [[[1, 2, 3]]], "depth": [[0.4]]},
            "completed": False,
            "error": "AttributeError: 'str' object has no attribute 'pose'",
        }
    }

    compact = _compact(value)

    assert "images" not in compact["observe"]
    assert "AttributeError" in compact["observe"]["error"]


def test_compact_human_trace_omits_duplicate_full_reports() -> None:
    trace = (
        "### Turn 1\n**Observation:** contact failed\n"
        "<details><summary>Full tool report</summary>\n"
        "```json\n{\"images\": [1, 2, 3]}\n```\n</details>\n"
        "### Turn 2\n**Observation:** retry\n"
    )

    compact = _compact_human_trace(trace)

    assert "contact failed" in compact
    assert "retry" in compact
    assert '"images"' not in compact
    assert "Full tool report omitted" in compact


def test_compact_human_trace_has_a_hard_request_bound() -> None:
    compact = _compact_human_trace("a" * 200_000, max_chars=10_000)

    assert len(compact) < 10_100
    assert "trace characters omitted" in compact


def test_causal_packet_preserves_candidate_named_public_report() -> None:
    packet = _causal_packet(
        {
            "instruction": "lift the cube",
            "steps": [
                {
                    "candidate_single_object_lift": {
                        "target_pose_before": [0.56, -0.01, -0.08],
                        "eef_after_close": {
                            "completed": True,
                            "result": {"position": [0.56, -0.01, 0.186]},
                        },
                        "verify_grasp_after_close": {
                            "completed": True,
                            "result": False,
                        },
                    }
                }
            ],
        }
    )

    step = packet["tool_steps"][0]
    assert step["action"] == "candidate_single_object_lift"
    report = step["report"]["candidate_single_object_lift"]
    assert report["target_pose_before"] == [0.56, -0.01, -0.08]
    assert report["eef_after_close"]["result"]["position"] == [0.56, -0.01, 0.186]
    assert report["verify_grasp_after_close"]["result"] is False
