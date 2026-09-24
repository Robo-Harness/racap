import subprocess

from evolution.harness.workspace import WorkspaceManager


def test_git_candidates_retain_lineage_and_promote_by_ref(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)
    manager.apply_patch(
        candidate,
        """diff --git a/solution/controller.py b/solution/controller.py
--- a/solution/controller.py
+++ b/solution/controller.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
""",
    )
    commit = manager.commit_candidate(candidate, "improve")
    assert commit != parent
    assert manager.promote(candidate) == commit
    champion = subprocess.check_output(
        ["git", "-C", str(manager.repository), "rev-parse", "champion"], text=True
    ).strip()
    assert champion == commit


def test_unified_diff_without_final_newline_is_transport_normalized(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    # This is the exact hosted-model serialization shape that Git otherwise
    # rejects as "corrupt patch at line N": valid hunk, no terminal newline.
    patch = """diff --git a/solution/controller.py b/solution/controller.py
--- a/solution/controller.py
+++ b/solution/controller.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2"""
    assert not patch.endswith("\n")

    assert manager.apply_patch(candidate, patch) == ()
    assert (candidate.path / "solution" / "controller.py").read_text() == "VALUE = 2\n"


def test_generated_patch_can_reach_guards_when_only_one_hunk_has_stale_context(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "memory").mkdir()
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    (seed / "memory" / "experience.md").write_text("actual memory\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    rejected = manager.apply_patch(
        candidate,
        """diff --git a/memory/experience.md b/memory/experience.md
--- a/memory/experience.md
+++ b/memory/experience.md
@@ -1 +1 @@
-stale model context
+new memory
diff --git a/solution/controller.py b/solution/controller.py
--- a/solution/controller.py
+++ b/solution/controller.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
""",
    )

    assert rejected
    assert (candidate.path / "solution" / "controller.py").read_text() == "VALUE = 2\n"
    assert not list(candidate.path.rglob("*.rej"))
    manager.commit_candidate(candidate, "partial but runnable")


def test_temporary_proposal_file_is_not_counted_as_a_source_change(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)
    (candidate.path / ".racap-proposal.patch").write_text("temporary\n")

    assert manager._proposal_source_status(candidate.path) == ()


def test_malformed_file_boundary_does_not_hide_other_runnable_sections(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "memory").mkdir()
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    (seed / "memory" / "experience.md").write_text("old\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    # The first hunk deliberately claims one output line but contains two.
    # The whole diff is malformed at the next file boundary; per-file recount
    # can still recover both unambiguous edits.
    manager.apply_patch(
        candidate,
        """diff --git a/memory/experience.md b/memory/experience.md
--- a/memory/experience.md
+++ b/memory/experience.md
@@ -1 +1 @@
-old
+new
+extra
diff --git a/solution/controller.py b/solution/controller.py
--- a/solution/controller.py
+++ b/solution/controller.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
""",
    )

    assert (candidate.path / "solution" / "controller.py").read_text() == "VALUE = 2\n"
    assert (candidate.path / "memory" / "experience.md").read_text() == "new\nextra\n"


def test_complete_file_proposal_avoids_diff_formatting_failure(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    manager.apply_proposal(
        candidate,
        files={
            "solution/controller.py": "VALUE = 3\n",
            "tests/test_new.py": "def test_value():\n    assert 3 == 3\n",
        },
    )
    commit = manager.commit_candidate(candidate, "complete files")

    assert commit != parent
    assert (candidate.path / "solution" / "controller.py").read_text() == "VALUE = 3\n"


def test_empty_generated_files_are_not_a_substantive_program_change(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    # This is the exact failure shape produced by a malformed hosted-model
    # diff: Git can create the declared files while applying none of their
    # intended hunks.  The old compile/test guard accepted the two empty files.
    manager.apply_proposal(
        candidate,
        files={
            "solution/generated.py": "",
            "tests/test_generated.py": "",
        },
    )
    manager.commit_candidate(candidate, "empty malformed candidate")

    assert manager.changed_files(candidate, base_commit=parent) == (
        "solution/generated.py",
        "tests/test_generated.py",
    )
    assert manager.substantive_program_changes(candidate, base_commit=parent) == ()


def test_runtime_memory_edit_is_a_substantive_program_change(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "memory").mkdir()
    (seed / "solution" / "controller.py").write_text("VALUE = 1\n")
    (seed / "memory" / "experience.md").write_text("old\n")
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    manager.apply_proposal(
        candidate,
        files={"memory/experience.md": "old\nnew evidence-backed rule\n"},
    )
    manager.commit_candidate(candidate, "update runtime memory")

    assert manager.substantive_program_changes(candidate, base_commit=parent) == (
        "memory/experience.md",
    )


def test_apply_patch_text_misplaced_in_files_is_recovered(tmp_path):
    seed = tmp_path / "seed"
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "controller.py").write_text(
        "VALUE = 1\n\ndef value():\n    return VALUE\n"
    )
    manager = WorkspaceManager(tmp_path / "experiment")
    parent = manager.initialize(seed)
    candidate = manager.create_candidate("i001", parent)

    manager.apply_proposal(
        candidate,
        files={
            "solution/controller.py": """*** Begin Patch
*** Update File: solution/controller.py
@@
-VALUE = 1
+VALUE = 4
@@
 def value():
-    return VALUE
+    return VALUE + 1
*** Add File: tests/test_generated.py
+def test_generated():
+    assert True
*** End Patch"""
        },
    )

    assert (candidate.path / "solution" / "controller.py").read_text() == (
        "VALUE = 4\n\ndef value():\n    return VALUE + 1\n"
    )
    assert (candidate.path / "tests" / "test_generated.py").is_file()
    manager.commit_candidate(candidate, "recover misplaced apply-patch")
