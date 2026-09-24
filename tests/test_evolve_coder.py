import json

from evolution.harness.coder import (
    _decode_whole_file_escapes,
    _recent_experiment_history,
    parse_proposal,
)


def test_parse_coding_proposal_with_unbounded_unified_diff():
    response = r"""{
      "title": "Expose recovery",
      "hypothesis": "Structured recovery helps a family of empty grasps",
      "predicted_effect": "More native successes",
      "risk": "Longer episodes",
      "patch": "diff --git a/solution/a.py b/solution/a.py\n--- a/solution/a.py\n+++ b/solution/a.py\n@@ -1 +1 @@\n-x=1\n+x=2"
    }"""
    proposal = parse_proposal(response)
    assert proposal.patch.startswith("diff --git")
    assert "x=2" in proposal.patch


def test_parse_coding_proposal_accepts_complete_files_without_diff_syntax():
    response = r'''{
      "title": "Implement contact path",
      "hypothesis": "A runnable actuator connects the discovered mechanism",
      "predicted_effect": "Control candidates reach runtime",
      "risk": "Contact direction may be wrong",
      "files": {"solution/contact.py": "VALUE = 2\n", "tests/obsolete.py": null}
    }'''
    proposal = parse_proposal(response)
    assert proposal.patch == ""
    assert proposal.files == {
        "solution/contact.py": "VALUE = 2\n",
        "tests/obsolete.py": None,
    }


def test_codex_context_patch_in_patch_field_routes_to_safe_files_adapter():
    payload = """*** Begin Patch
*** Update File: solution/controller.py
@@
-VALUE = 1
+VALUE = 2
*** End Patch"""
    proposal = parse_proposal(
        json.dumps(
            {
                "title": "repair",
                "hypothesis": "repair",
                "predicted_effect": "compile",
                "risk": "none",
                "patch": payload,
            }
        )
    )

    assert proposal.patch == ""
    assert list(proposal.files.values()) == [payload]
    assert next(iter(proposal.files)).startswith("solution/.racap-context-patch")


def test_complete_files_take_precedence_over_redundant_prose_patch():
    response = r'''{
      "title": "Implement lift",
      "hypothesis": "A bounded lift closes the missing mechanism",
      "predicted_effect": "Lift tasks reach execution",
      "risk": "Grasp may detach",
      "patch": "Here is the implementation described above.",
      "files": {"solution/lift.py": "def lift():\n    return True\n"}
    }'''
    proposal = parse_proposal(response)
    assert proposal.patch == ""
    assert "solution/lift.py" in proposal.files


def test_missing_descriptive_metadata_does_not_discard_executable_code():
    proposal = parse_proposal(
        json.dumps(
            {
                "target_failure_cluster": "lift_without_transport",
                "mechanism": "Add an explicit vertical lift primitive.",
                "files": {
                    "solution/controller.py": (
                        "def run_episode(runtime, episode):\n    return {}\n"
                    )
                },
            }
        )
    )

    assert proposal.title == "Executable candidate for lift_without_transport"
    assert proposal.hypothesis == "Add an explicit vertical lift primitive."
    assert "paired" in proposal.predicted_effect
    assert "strict native-success improvement" in proposal.risk


def test_double_escaped_complete_file_is_recovered_without_damaging_literals():
    encoded = (
        'import re\\n\\nPATTERN = re.compile(r"a\\\\nb")\\n'
        'MESSAGE = "你好"\\n'
    )
    decoded = _decode_whole_file_escapes(encoded)

    assert decoded == (
        'import re\n\nPATTERN = re.compile(r"a\\nb")\n'
        'MESSAGE = "你好"\n'
    )
    proposal = parse_proposal(
        json.dumps(
            {
                "title": "decode",
                "hypothesis": "decode",
                "predicted_effect": "compile",
                "risk": "none",
                "files": {"solution/controller.py": encoded},
            },
            ensure_ascii=False,
        )
    )
    assert proposal.files["solution/controller.py"] == decoded


def test_recent_history_excludes_requests_without_a_candidate():
    history = [
        {"iteration": 1, "status": "capability_promoted"},
        {"iteration": 2, "status": "proposal_failed", "error": "no route"},
        {"iteration": 3, "status": "retained_not_promoted"},
    ]
    assert [row["iteration"] for row in _recent_experiment_history(history)] == [1, 3]


def test_recent_history_surfaces_nested_runtime_exceptions(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "key": "suite/7/seed0",
                "steps": [
                    {
                        "report": {
                            "recovery": {
                                "contact_grasp": {
                                    "completed": False,
                                    "error": "AttributeError: str has no pose",
                                },
                                "images": {"rgb": [[[1, 2, 3]]]},
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    history = [
        {
            "iteration": 4,
            "status": "retained_not_promoted",
            "candidate_metrics": {"records_path": str(records)},
        }
    ]

    recent = _recent_experiment_history(history)

    assert recent[0]["runtime_diagnostics"] == [
        {
            "episode": "suite/7/seed0",
            "error": (
                "steps[0].report.recovery.contact_grasp.error: "
                "AttributeError: str has no pose"
            ),
        }
    ]
