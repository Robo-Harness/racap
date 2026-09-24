"""Policy Quality Checker: two-tier gatekeeper.

Tier 1 (hard gate): syntax errors, invalid API calls, infinite loops.
  - Blocks execution, returns to Policy Writer.
  - Does NOT count as an execution attempt.

Tier 2 (advisory): trial-and-error patterns, style issues.
  - Flags concerns but code proceeds to Executor.

Server availability is probed at init time so that primitives whose
backing servers are offline are automatically blocked.
"""

from __future__ import annotations

import ast
import os
import re
import socket
from typing import Any

from pathlib import Path

from rats.agents.base_agent import query_llm_json
from skill_library.initial_primitives import get_all_primitive_names


def _load_server_deps() -> dict[str, tuple[str, int]]:
    """Load primitive->server mappings from rats/config/default.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent / "rats" / "config" / "default.yaml"
    deps: dict[str, tuple[str, int]] = {}
    if config_path.exists():
        import yaml
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        for _name, srv in (cfg.get("servers") or {}).items():
            host = srv.get("host", "127.0.0.1")
            port = int(srv.get("port", 0))
            for prim in srv.get("primitives", []):
                deps[prim] = (host, port)
    return deps


# Maps primitive names to the (host, port) of the server they require.
_PRIMITIVE_SERVER_DEPS = _load_server_deps()


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _detect_unavailable_primitives() -> dict[str, str]:
    """Probe servers and return primitives whose servers are down."""
    blocked: dict[str, str] = {}
    checked_ports: dict[tuple[str, int], bool] = {}
    for prim, (host, port) in _PRIMITIVE_SERVER_DEPS.items():
        # In controlled comparisons point prompting can be routed through the
        # same hosted VLM as the other agents.  In that mode there is no Molmo
        # TCP service to probe, and treating port 8122 as authoritative would
        # incorrectly reject a working primitive.
        if prim == "point_prompt_molmo" and os.getenv("RATS_RUNTIME_VLM_MODEL"):
            continue
        key = (host, port)
        if key not in checked_ports:
            checked_ports[key] = _is_port_open(host, port)
        if not checked_ports[key]:
            blocked[prim] = f"server {host}:{port} is not reachable; do not use {prim}"
    return blocked


class PolicyQualityChecker:
    # Patterns that ALWAYS cause Tier 1 rejection (regardless of servers)
    FORBIDDEN_PATTERNS = {
        "while True": "unbounded retry loop",
        "exec(": "dynamic execution forbidden",
        "eval(": "dynamic evaluation forbidden",
        "env.": "use imported primitive functions instead of env.<method>",
        "low_level_env.": "use imported primitive functions instead of low_level_env.<method>",
    }
    POST_GRASP_GATE_KEYS = {
        "verified",
        "ready",
        "release_ok",
        "drop_ready",
        "holding_likely",
        "object_visible_in_wrist",
        "held_ok",
        "wrist_ok",
    }
    GRASP_CALLS = {
        "close_gripper",
        "grasp_object_topdown",
        "grasp_object_with_verification",
        "grasp_with_wrist_closeloop",
        # MolmoSpaces / reduced API helpers that commit to a pick/grasp action.
        "execute_pick",
        "execute_staged_pick",
        "pick_object",
        "pick_up_object",
    }

    def __init__(self) -> None:
        # Probe which servers are actually running at init time
        self._server_blocked = _detect_unavailable_primitives()
        if self._server_blocked:
            names = ", ".join(sorted(self._server_blocked))
            print(f"[quality_checker] Blocked primitives (server offline): {names}")

    def refresh_server_status(self) -> None:
        """Re-probe servers (call if a server was started after init)."""
        self._server_blocked = _detect_unavailable_primitives()

    def check(
        self,
        code: str,
        available_functions: list[str] | None = None,
        goal: str = "",
        learned_skill_names: list[str] | None = None,
        full_policy_check: bool = True,
    ) -> dict[str, Any]:
        """Run both Tier 1 and Tier 2 checks.

        ``full_policy_check`` (default True): treat ``code`` as a complete
        executable policy and enforce structural rules that only make
        sense for full policies:
            - must set ``RESULT`` or have a ``return`` statement
            - pick-and-place progression check (grasp must have a later
              placement/release call in the same code blob)
        Multiturn-reset's per-step writer output is a BARE STEP BODY
        (the orchestrator appends ``RESULT`` and the placement/release
        will appear in a later step's body), so callers in that path
        pass ``full_policy_check=False``. All other Tier-1 rules
        (forbidden patterns, server-blocked primitives, banned
        primitives, unbounded while loops, API validation) still fire.

        Returns:
            Dict with:
              - approved: bool (False if Tier 1 fails)
              - tier1_issues: list of blocking issues
              - tier2_issues: list of advisory issues
              - feedback: str summary
        """
        tier1_issues = self._check_tier1(
            code, available_functions, learned_skill_names, goal=goal,
            full_policy_check=full_policy_check,
        )
        tier2_issues = []

        if not tier1_issues:
            tier2_issues = self._check_tier2(code, goal)

        approved = len(tier1_issues) == 0
        feedback_parts = []
        if tier1_issues:
            feedback_parts.append("BLOCKED: " + "; ".join(tier1_issues))
        if tier2_issues:
            feedback_parts.append("ADVISORY: " + "; ".join(tier2_issues))

        return {
            "approved": approved,
            "tier1_issues": tier1_issues,
            "tier2_issues": tier2_issues,
            "feedback": " | ".join(feedback_parts) if feedback_parts else "approved",
        }

    def _check_tier1(
        self,
        code: str,
        available_functions: list[str] | None = None,
        learned_skill_names: list[str] | None = None,
        goal: str = "",
        full_policy_check: bool = True,
    ) -> list[str]:
        """Hard gate: syntax, API validation, loop detection."""
        issues = []

        if not code.strip():
            issues.append("policy draft is empty")
            return issues

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"syntax error: {e.msg} at line {e.lineno}")
            return issues

        # Always-forbidden patterns
        for pattern, reason in self.FORBIDDEN_PATTERNS.items():
            if pattern in code:
                issues.append(f"rejected: {reason} (found '{pattern}')")

        # AST check: block unbounded while loops (while <var>, while not <var>, etc.)
        # A while loop is considered "bounded" only if it contains a break statement.
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    # Check if there's a break anywhere inside this while body
                    has_break = any(
                        isinstance(child, ast.Break)
                        for child in ast.walk(node)
                    )
                    if not has_break:
                        line = getattr(node, "lineno", "?")
                        issues.append(
                            f"rejected: unbounded while loop at line {line} "
                            f"(use 'for _ in range(max_retries)' instead)"
                        )
        except SyntaxError:
            pass  # already caught above

        # Server-dependent primitives whose servers are offline.
        #
        # FIX (mirrors lifelong_loop.py:1870+ api_docs fix): a server-blocked
        # primitive may still be registered through a privileged-API path
        # (e.g. FrankaLiberoPrivilegedApi.sample_grasp_pose works without
        # the graspnet server). Rejecting code that calls it forces an
        # otherwise-fine policy into a retry loop. Skip the rejection when
        # the function is in available_functions — the registered variant
        # handles the call without the server.
        registered = set(available_functions or [])
        for prim, reason in self._server_blocked.items():
            if prim in registered:
                continue
            if prim in code:
                issues.append(f"rejected: {reason}")

        # Full-policy-only structural checks. Skipped when the caller
        # passes a per-step fragment (multiturn-reset writer output is
        # one step body; the orchestrator composes RESULT and the
        # placement/release call will be in a later step's body).
        if full_policy_check:
            if "RESULT" not in code and "return" not in code:
                issues.append("policy must set RESULT variable or have return statement")
            if self._is_pick_and_place_goal(goal):
                issues.extend(self._check_pick_place_progression(code))

        # NOTE — perception_verify_consumed used to live here as a
        # Tier 1 hard gate (commit 215ee6a8) but produced too many
        # false-positive rejections on legitimate post-action
        # verification patterns (``final_check = verify_object_
        # identity(...)`` at end of policy to record the outcome with
        # no downstream code to gate). Demoted to Tier 2 advisory; see
        # ``_check_tier2`` below. The verifier + diagnoser still catch
        # the original v7 anti-pattern (wrong-object grasp on
        # unverified perception) post-execution.

        # Validate API calls if we have the function list
        if available_functions:
            all_valid = set(available_functions) | set(get_all_primitive_names())
            all_valid.update([
                "print", "len", "range", "int", "float", "str", "list", "dict",
                "tuple", "set", "bool", "abs", "min", "max", "sum", "sorted",
                "enumerate", "zip", "map", "filter", "isinstance", "type",
                "hasattr", "getattr", "setattr", "np", "numpy", "time",
                "json", "math", "RESULT", "all", "any", "round", "format",
                "repr", "id", "hex", "bin", "oct", "ord", "chr", "input",
                "open", "super", "property", "staticmethod", "classmethod",
                "Exception", "RuntimeError", "ValueError", "TypeError",
                "KeyError", "IndexError", "AttributeError", "StopIteration",
                "next", "iter", "reversed", "slice", "bytes", "bytearray",
                "frozenset", "object", "memoryview", "complex", "divmod",
                "pow", "hash", "callable", "vars", "dir", "globals", "locals",
                "breakpoint", "NotImplementedError", "AssertionError",
                "OSError", "IOError", "FileNotFoundError",
            ])
            # Runtime logging markers injected by rats.envs.tasks.base for
            # ALL envs (molmospaces + libero, see commit 45eaf5a6 where
            # injection became unconditional). Always whitelist them so
            # quality_checker doesn't reject marker-instrumented code as
            # "unknown function call" — observed in libero_main_30iter
            # iter 1 where every other policy_writer attempt was blocked
            # with 11x "step_context" issues, forcing a quality-rewrite
            # loop that effectively doubled per-attempt LLM cost AND
            # silently stripped markers from the executed code (so the
            # per-step verifier remained blind even after 45eaf5a6).
            all_valid.update({"step_context", "begin_step", "end_step"})
            # Add learned skill function names as valid callables
            if learned_skill_names:
                all_valid.update(learned_skill_names)
            # Remove server-blocked primitives from valid set — but keep
            # ones that are registered through a privileged-API path. Same
            # logic as the rejection block above (sample_grasp_pose etc.).
            effectively_blocked = set(self._server_blocked.keys()) - set(available_functions or [])
            all_valid -= effectively_blocked
            try:
                tree = ast.parse(code)
                # Collect locally-defined function and class names
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        all_valid.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        all_valid.add(node.name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        fn_name = node.func.id
                        if fn_name not in all_valid and not fn_name.startswith("_"):
                            # "did you mean" hint — observed the same hallucinated
                            # names (e.g. place_object_into_container, topdown_grasp_*)
                            # getting blocked in ≥3 consecutive attempts because the
                            # block message gave no alternative and the LLM just
                            # retried with the same typo.
                            import difflib
                            hint = difflib.get_close_matches(
                                fn_name, all_valid, n=2, cutoff=0.6,
                            )
                            if hint:
                                issues.append(
                                    f"unknown function call: {fn_name} "
                                    f"(did you mean: {', '.join(hint)}?)"
                                )
                            else:
                                issues.append(f"unknown function call: {fn_name}")
            except Exception:
                pass

        return issues

    # Perception-verify primitives whose `verified` field MUST be consulted
    # before downstream grasp / motion code uses the localized mask.
    _PERCEPTION_VERIFY_CALLS = {
        "vlm_verify",
        "verify_object_identity",
    }

    @classmethod
    def _check_perception_verify_consumed(cls, code: str) -> list[str]:
        """Flag PRE-grasp perception verifications whose verdict is ignored.

        Two anti-patterns this catches, both observed in v7 iter 3:

        1. ``pv = verify_object_identity(...)`` (or ``v = vlm_verify(...)``)
           where ``pv`` is bound in a function scope but is never compared
           against the ``"verified"`` field anywhere in that function.
           The LLM stored the verdict and then ignored it.

        2. A string literal anywhere in the code containing the substring
           ``"unverified"`` as a method/tag name (e.g.
           ``"method": "text_fallback_unverified"``). The LLM is literally
           telling itself "I'm using an unverified result anyway".

        Both patterns are Tier-1 blocking: the prompt's hard rule says
        the policy MUST bail when verification fails, so any code that
        knowingly proceeds with unverified perception gets rejected.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # syntax error is reported by the earlier ast.parse

        issues: list[str] = []

        # NOTE on the dropped textual ``unverified`` gate. An earlier
        # version of this function (commit 215ee6a8) flagged any token
        # containing the substring "unverified" as an anti-pattern path.
        # That regex turned out to false-positive on the prompt's own
        # recommended bail string — the policy_writer prompt's hard rule
        # says to write ``RESULT[step] = {success: False, reason:
        # 'perception_unverified'}; return`` when vlm_verify fails. The
        # LLM dutifully emits ``"perception_unverified"`` as a string
        # value, and the gate blocked the entire policy. v6 iter 1 had
        # all 4 attempts rejected for following the prompt's example.
        # The AST-level Anti-pattern 1 below (verify_* call result bound
        # but never read) is the actual robust signal; the textual check
        # was a heuristic add-on and is removed entirely.

        # Anti-pattern 1: assigned verify result that's never consulted.
        # Walk each function scope independently; the bind + consume must
        # live in the same scope. Track which (scope_id, name) we've
        # already flagged so the module-level walk doesn't re-flag a
        # binding that's already going to be flagged inside the function
        # scope it lives in.
        scopes: list[ast.AST] = [tree]
        scopes.extend(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        flagged: set[tuple[int, str]] = set()
        for scope in scopes:
            # Names bound to perception-verify calls in this scope.
            verify_names: dict[str, int] = {}
            for node in ast.walk(scope):
                # Don't cross into nested function scopes for binding
                # detection (the nested scope is its own iteration).
                if (
                    node is not scope
                    and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ):
                    continue
                if isinstance(node, ast.Assign):
                    call = node.value
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id in cls._PERCEPTION_VERIFY_CALLS
                    ):
                        for target in node.targets:
                            for name in cls._assigned_names(target):
                                verify_names.setdefault(name, getattr(node, "lineno", 0))

            if not verify_names:
                continue

            # Look for any consumption of name["verified"] / name.get("verified")
            # / name.verified in this scope.
            consumed: set[str] = set()
            for node in ast.walk(scope):
                if (
                    node is not scope
                    and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ):
                    continue
                # name["verified"] -> Subscript with Constant("verified") slice
                if isinstance(node, ast.Subscript):
                    value = node.value
                    slice_node = node.slice
                    key = None
                    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                        key = slice_node.value
                    elif (  # py3.8 compat
                        hasattr(ast, "Index")
                        and isinstance(slice_node, getattr(ast, "Index", ()))
                        and isinstance(slice_node.value, ast.Constant)
                    ):
                        key = slice_node.value.value
                    if (
                        key == "verified"
                        and isinstance(value, ast.Name)
                        and value.id in verify_names
                    ):
                        consumed.add(value.id)
                # name.get("verified") -> Call(func=Attribute(value=Name, attr="get"), args=[Constant("verified")])
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in verify_names
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "verified"
                ):
                    consumed.add(node.func.value.id)
                # name.verified attribute access
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "verified"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in verify_names
                ):
                    consumed.add(node.value.id)

            for name, lineno in verify_names.items():
                if name in consumed:
                    continue
                key = (lineno, name)
                if key in flagged:
                    continue
                flagged.add(key)
                issues.append(
                    f"rejected: ``{name}`` at line {lineno} is bound to a "
                    "perception-verify call (vlm_verify / verify_object_identity) "
                    f"but ``{name}['verified']`` is never consulted to gate "
                    "downstream behavior. When verified=False the policy MUST "
                    "bail (RESULT[step] = {success: False, reason: "
                    "'perception_unverified'}; return) — do NOT pass an "
                    "unverified mask to grasp / motion code."
                )

        return issues

    @staticmethod
    def _is_pick_and_place_goal(goal: str) -> bool:
        normalized = f" {str(goal or '').lower().replace('_', ' ')} "
        # Do not apply pick/place guardrails to articulation/navigation tasks
        # whose goals often include "open/close/on/in" language but should not
        # force a later release/place step after a pull/grasp.
        articulation_or_navigation = any(
            token in normalized
            for token in (
                " open ",
                " close ",
                " drawer ",
                " dresser ",
                " cabinet ",
                " door ",
                " microwave ",
                " oven ",
                " laptop ",
                " articulated ",
                " pull ",
                " push ",
                " navigate ",
                " go to ",
                " move to ",
            )
        )
        explicit_pick_place = any(
            token in normalized
            for token in (
                " put ",
                " place ",
                " pick and place ",
                " onto ",
                " into ",
                " inside ",
            )
        )
        if explicit_pick_place:
            return True
        if articulation_or_navigation:
            return False
        if any(
            token in normalized
            for token in (" turn on ", " switch on ", " power on ")
        ):
            return False
        return any(token in normalized for token in (" on the ", " in the "))

    @staticmethod
    def _call_name(call: ast.Call) -> str:
        fn = call.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
        return ""

    @classmethod
    def _is_gate_call_name(cls, name: str) -> bool:
        if not name:
            return False
        return (
            name == "verify_step"
            or name.startswith("verify_")
            or name.startswith("gate_")
            or name.startswith("confirm_held")
            or "_gate_" in name
        )

    @classmethod
    def _is_release_or_place_call_name(cls, name: str) -> bool:
        if not name:
            return False
        if name == "open_gripper":
            return True
        if cls._is_gate_call_name(name) or name.startswith("check_"):
            return False
        return any(part in name for part in ("place", "release", "drop"))

    @staticmethod
    def _runtime_nodes(tree: ast.AST, node_type: type[ast.AST]) -> list[ast.AST]:
        nodes: list[ast.AST] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def generic_visit(self, node: ast.AST) -> None:
                if isinstance(node, node_type):
                    nodes.append(node)
                super().generic_visit(node)

        Visitor().visit(tree)
        return nodes

    @staticmethod
    def _assigned_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for elt in target.elts:
                names.update(PolicyQualityChecker._assigned_names(elt))
            return names
        return set()

    @staticmethod
    def _subscript_ref(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript):
            return None
        if not isinstance(node.value, ast.Name):
            return None
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, (str, int)):
            return f"{node.value.id}[{sl.value!r}]"
        return None

    @classmethod
    def _assigned_gate_refs(cls, target: ast.AST) -> set[str]:
        ref = cls._subscript_ref(target)
        if ref:
            return {ref}
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            refs: set[str] = set()
            for elt in target.elts:
                refs.update(cls._assigned_gate_refs(elt))
            return refs
        return set()

    @classmethod
    def _collect_gate_vars(cls, tree: ast.AST) -> set[str]:
        gate_vars: set[str] = set()
        for node in cls._runtime_nodes(tree, ast.Assign):
            assert isinstance(node, ast.Assign)
            if cls._expr_has_post_grasp_gate(node.value, gate_vars):
                for target in node.targets:
                    gate_vars.update(cls._assigned_gate_refs(target))
                if isinstance(node.value, ast.Dict):
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if not isinstance(key_node, ast.Constant):
                            continue
                        if not isinstance(key_node.value, (str, int)):
                            continue
                        if cls._expr_has_post_grasp_gate(value_node, gate_vars):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    gate_vars.add(f"{target.id}[{key_node.value!r}]")
        return gate_vars

    @classmethod
    def _expr_has_post_grasp_gate(cls, expr: ast.AST, gate_vars: set[str]) -> bool:
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id in gate_vars:
                return True
            ref = cls._subscript_ref(node)
            if ref and ref in gate_vars:
                return True
            if isinstance(node, ast.Call) and cls._is_gate_call_name(cls._call_name(node)):
                return True
            if isinstance(node, ast.Attribute) and node.attr in cls.POST_GRASP_GATE_KEYS:
                return True
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in cls.POST_GRASP_GATE_KEYS:
                    return True
        return False

    @classmethod
    def _node_contains_release_or_place(cls, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Call) and cls._is_release_or_place_call_name(
                cls._call_name(child)
            ):
                return True
        return False

    @classmethod
    def _check_pick_place_progression(cls, code: str) -> list[str]:
        """Reject policies that can terminate a pick-and-place after grasp only."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        issues: list[str] = []
        runtime_calls = [
            node for node in cls._runtime_nodes(tree, ast.Call)
            if isinstance(node, ast.Call)
        ]
        grasp_lines = [
            getattr(call, "lineno", 0)
            for call in runtime_calls
            if cls._call_name(call) in cls.GRASP_CALLS
        ]
        release_or_place_lines = [
            getattr(call, "lineno", 0)
            for call in runtime_calls
            if cls._is_release_or_place_call_name(cls._call_name(call))
        ]

        if grasp_lines and not any(
            line > min(grasp_lines) for line in release_or_place_lines
        ):
            issues.append(
                "rejected: pick-and-place policy grasps/closes but has no "
                "later placement/release call; do not stop with the object "
                "held in the gripper"
            )

        gate_vars = cls._collect_gate_vars(tree)
        for node in cls._runtime_nodes(tree, ast.If):
            assert isinstance(node, ast.If)
            if not cls._expr_has_post_grasp_gate(node.test, gate_vars):
                continue
            if any(cls._node_contains_release_or_place(child) for child in node.body + node.orelse):
                issues.append(
                    "rejected: placement/release is hard-gated on a post-grasp "
                    "VLM or wrist holding check; record that check in RESULT, "
                    "but after a successful grasp continue to target placement "
                    "and verify after release"
                )
                break

        return issues

    def _check_tier2(self, code: str, goal: str = "") -> list[str]:
        """Advisory: semantic issues (non-blocking)."""
        issues = []

        if "random" in code.lower() and "import random" in code:
            issues.append("trial-and-error pattern: random sampling detected")

        lines = code.strip().splitlines()
        if len(lines) > 200:
            issues.append(f"code is very long ({len(lines)} lines), consider simplification")

        if "except:" in code and "except Exception" not in code:
            issues.append("bare except clause may hide important errors")

        if "time.sleep" in code and "time.sleep(0" not in code:
            issues.append("sleep in robot code may indicate polling pattern")

        # Perception-verify-consumed (was a Tier 1 gate; demoted because
        # legitimate post-action verifications were being rejected). Now
        # advisory — surfaced in the quality report so the LLM sees it
        # on retry, but the policy is not blocked.
        issues.extend(self._check_perception_verify_consumed(code))

        return issues
