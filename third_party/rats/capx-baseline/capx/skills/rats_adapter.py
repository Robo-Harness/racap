"""Formatting helpers for RATS skill-library JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


EXTERNAL_SKILLS_PROMPT_HEADER = "Reusable skills from previous successful LIBERO runs:"


def load_skill_entries(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    skill_path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not skill_path.exists():
        raise FileNotFoundError(f"External skill library not found: {skill_path}")

    data = json.loads(skill_path.read_text())
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    if isinstance(data, dict) and isinstance(data.get("skills"), dict):
        entries: list[dict[str, Any]] = []
        for name, info in data["skills"].items():
            if not isinstance(info, dict):
                continue
            entries.append(
                {
                    "name": info.get("name", name),
                    "description": info.get("docstring", ""),
                    "code": info.get("code", ""),
                    "source_task": ", ".join(info.get("source_tasks", [])),
                    "success_rate": info.get("success_rate"),
                    "is_primitive": False,
                }
            )
        return entries

    raise ValueError(
        "Unsupported external skill library format. Expected a RATS list or "
        "a CaP-X {'skills': ...} object."
    )


def _skill_identifier(skill: dict[str, Any], idx: int) -> str:
    return str(skill.get("skill_id") or skill.get("name") or f"skill_{idx}")


def _filter_skill_entries(
    entries: list[dict[str, Any]],
    *,
    max_skills: int = 12,
    include_primitives: bool = False,
    skip_deprecated: bool = True,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if not include_primitives and bool(entry.get("is_primitive", False)):
            continue
        # Skip deprecated learned skills by default. RATS marks a skill
        # deprecated when its empirical success rate is too low to be a
        # safe wrapper, so injecting them as callables risks the LLM
        # picking up a known-bad helper. ``skip_deprecated=False`` if a
        # caller wants the full bag for forensics.
        if skip_deprecated and str(entry.get("tier", "")).lower() == "deprecated":
            continue
        if not entry.get("name") and not entry.get("description") and not entry.get("skill_id"):
            continue
        filtered.append(entry)

    if not filtered and not include_primitives:
        filtered = [
            entry
            for entry in entries
            if entry.get("name") or entry.get("description") or entry.get("skill_id")
        ]

    if max_skills > 0:
        filtered = filtered[:max_skills]
    return filtered


def format_skill_entries_for_prompt(
    entries: list[dict[str, Any]],
    *,
    include_code: bool = True,
    selected_by_planner: bool = True,
    injected_into_scope: bool = False,
) -> str:
    """Format selected skill entries as policy-writer prompt context.

    ``injected_into_scope=True`` rewrites the intro to claim that the
    skills are *already callable* in the execution namespace (matches
    what ``inject_external_skills_into_namespace`` actually does at
    exec time), and removes the "copy or adapt" framing entirely so the
    LLM gets one consistent message: call these by name.
    """
    if not entries:
        return ""

    if injected_into_scope:
        intro = (
            "LEARNED SKILLS — these are reusable helper functions extracted "
            "from previous successful runs on this same environment. They are "
            "PRE-LOADED and CALLABLE in your execution scope, on the same "
            "footing as the API primitives above. The difference is provenance: "
            "the API above is the robot's built-in primitive surface (perception, "
            "motion, gripper); the SKILLS below are higher-level patterns the "
            "agent has learned (top-down grasps with retries, in-basket placement "
            "with retreat, etc.) and that wrap multiple primitive calls into a "
            "single verified routine.\n"
            "Each skill below is rendered as a signature + docstring only. The "
            "signature gives you the input types and return shape; the docstring "
            "spells out tensor/array shapes for each argument and the keys of "
            "the returned dict. The body is NOT shown — call the skill by name "
            "with the documented arguments, do NOT inline its implementation."
        )
        usage_directive = None
    elif selected_by_planner:
        intro = (
            "The following skills were selected by a skill planner as the most "
            "relevant reusable examples for the current task. Use them when they "
            "help solve the task, especially for common manipulation patterns, "
            "perception helpers, or motion-planning structure."
        )
        usage_directive = (
            "These skill functions are not automatically imported into the execution "
            "environment. If you want to call a skill, first copy or adapt its "
            "function definition or logic into your generated Python code, then call "
            "that local helper. Prefer the APIs listed above as the executable "
            "interface to the robot and environment."
        )
    else:
        intro = (
            "The following skills are optional reusable examples from previous "
            "successful LIBERO runs. Use them when they are relevant to the current "
            "task, especially for common manipulation patterns, perception helpers, "
            "or motion-planning structure."
        )
        usage_directive = (
            "These skill functions are not automatically imported into the execution "
            "environment. If you want to call a skill, first copy or adapt its "
            "function definition or logic into your generated Python code, then call "
            "that local helper. Prefer the APIs listed above as the executable "
            "interface to the robot and environment."
        )

    lines = [
        EXTERNAL_SKILLS_PROMPT_HEADER,
        intro,
    ]
    if usage_directive:
        lines.append(usage_directive)
    lines.append(
        "Do not write an explanation about the skills; still output only the "
        "executable Python code requested by the task prompt."
    )

    for idx, skill in enumerate(entries, start=1):
        skill_id = _skill_identifier(skill, idx)
        name = str(skill.get("name") or skill_id)
        lines.append(f"\nSkill {idx}: {name}")
        lines.append(f"Skill ID: {skill_id}")

        description = str(skill.get("description") or "").strip()
        if description:
            lines.append(f"Description: {description}")

        api_primitives = skill.get("api_primitives_used") or []
        if api_primitives:
            if isinstance(api_primitives, list):
                api_text = ", ".join(str(api) for api in api_primitives)
            else:
                api_text = str(api_primitives)
            lines.append(f"Uses APIs: {api_text}")

        success_rate = skill.get("success_rate")
        if success_rate is not None:
            lines.append(f"Success rate: {success_rate}")

        source_task = str(skill.get("source_task") or "").strip()
        if source_task:
            lines.append(f"Source task: {source_task}")

        code = str(skill.get("code") or "").strip()
        if include_code and code:
            lines.append("Reference code:")
            lines.append("```python")
            lines.append(code)
            lines.append("```")
        elif injected_into_scope and code:
            # include_code is OFF but the skill IS callable in exec scope —
            # render just the def signature + leading docstring so the LLM
            # sees typings + I/O shapes without the implementation. The
            # body itself is unnecessary noise once the function is in
            # scope; the LLM just needs to know what to pass and what
            # comes back.
            sig_doc = _signature_and_docstring(code)
            if sig_doc:
                lines.append("Signature & I/O:")
                lines.append("```python")
                lines.append(sig_doc)
                lines.append("```")

    return "\n".join(lines)


def _signature_and_docstring(code: str) -> str:
    """Extract `def fn(...) -> T:` + the function's docstring (no body).

    Used when ``injected_into_scope=True and include_code=False`` to show
    the LLM only what it needs to USE the skill (typings + shape hints
    from the docstring), without the body. The function is already
    callable via ``inject_external_skills_into_namespace``.

    Returns an empty string if the code doesn't parse or doesn't have a
    top-level ``def``.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef)),
        None,
    )
    if fn is None:
        return ""

    try:
        args_src = ast.unparse(fn.args)
    except Exception:
        return ""
    returns_src = ""
    if fn.returns is not None:
        try:
            returns_src = " -> " + ast.unparse(fn.returns)
        except Exception:
            returns_src = ""

    docstring = ast.get_docstring(fn) or ""

    parts = [f"def {fn.name}({args_src}){returns_src}:"]
    if docstring:
        first, _, rest = docstring.partition("\n")
        parts.append('    """' + first)
        for ln in rest.splitlines():
            parts.append("    " + ln if ln else "")
        parts.append('    """')
    else:
        parts.append("    ...")
    return "\n".join(parts)


def format_external_skills_for_prompt(
    path: str | os.PathLike[str],
    *,
    max_skills: int = 12,
    include_code: bool = True,
    include_primitives: bool = False,
    injected_into_scope: bool = False,
) -> str:
    """Format RATS/CaP-X skill JSON as policy-writer prompt context.

    With ``injected_into_scope=False`` (default) these are reference-only
    examples the policy writer is asked to copy/adapt. With
    ``injected_into_scope=True``, the intro is rewritten to claim the
    skills are pre-loaded and callable — paired with
    ``inject_external_skills_into_namespace`` at exec time so the claim
    is actually true.
    """
    entries = _filter_skill_entries(
        load_skill_entries(path),
        max_skills=max_skills,
        include_primitives=include_primitives,
    )
    return format_skill_entries_for_prompt(
        entries,
        include_code=include_code,
        selected_by_planner=False,
        injected_into_scope=injected_into_scope,
    )


def inject_external_skills_into_namespace(
    path: str | os.PathLike[str],
    namespace: dict[str, Any],
    *,
    max_skills: int = 12,
    include_primitives: bool = False,
) -> dict[str, Any]:
    """exec() each skill's ``code`` body into ``namespace``.

    Mirrors RATS proper's ``skill_preamble`` injection: every skill's
    top-level ``def`` becomes a callable in the user-code execution
    namespace, on the same footing as the API primitives. The policy
    writer can then call learned skills directly by name without inline
    copy/adapt.

    Filtering shares ``_filter_skill_entries`` with the prompt-side
    formatter so the listing the LLM sees and the bindings the executor
    provides are GUARANTEED to be the same skill set — a mismatch (skill
    in prompt but not injected, or vice versa) would cause NameErrors at
    call time or silently-orphan helpers.

    Skills that fail to ``exec`` (SyntaxError, decorator that references
    an undefined name at define time) are logged and dropped; the trial
    continues with whatever loaded successfully.

    Returns a dict ``{skill_name: status}`` for audit / debugging where
    status is ``"injected"`` or an error string.
    """
    results: dict[str, Any] = {}
    try:
        all_entries = load_skill_entries(path)
    except FileNotFoundError as exc:
        print(f"[ExternalSkills] Library not found at {path}: {exc}")
        return results

    filtered = _filter_skill_entries(
        all_entries,
        max_skills=max_skills,
        include_primitives=include_primitives,
    )

    for entry in filtered:
        name = str(entry.get("name") or entry.get("skill_id") or "<unnamed>")
        code = str(entry.get("code") or "").strip()
        if not code:
            results[name] = "skipped: no code"
            continue
        try:
            exec(code, namespace, namespace)  # noqa: S102
            results[name] = "injected"
        except Exception as exc:
            results[name] = f"{type(exc).__name__}: {exc}"
            print(f"[ExternalSkills] Failed to inject '{name}': {exc}")
    return results


def format_skill_catalog_for_planner(
    entries: list[dict[str, Any]],
    *,
    include_code: bool = False,
) -> str:
    """Format skill entries as a compact catalog for a selection planner."""
    if not entries:
        return "No skills are available."

    lines = [
        "Available reusable skills catalog:",
        "Select by Skill ID. Prefer a small set of directly relevant skills.",
    ]
    for idx, skill in enumerate(entries, start=1):
        skill_id = _skill_identifier(skill, idx)
        name = str(skill.get("name") or skill_id)
        lines.append(f"\nSkill {idx}")
        lines.append(f"Skill ID: {skill_id}")
        lines.append(f"Name: {name}")

        description = str(skill.get("description") or "").strip()
        if description:
            lines.append(f"Description: {description}")

        api_primitives = skill.get("api_primitives_used") or []
        if api_primitives:
            if isinstance(api_primitives, list):
                api_text = ", ".join(str(api) for api in api_primitives)
            else:
                api_text = str(api_primitives)
            lines.append(f"Uses APIs: {api_text}")

        source_task = str(skill.get("source_task") or "").strip()
        if source_task:
            lines.append(f"Source task: {source_task}")

        success_rate = skill.get("success_rate")
        if success_rate is not None:
            lines.append(f"Success rate: {success_rate}")

        if bool(skill.get("is_primitive", False)):
            lines.append("Primitive: true")

        code = str(skill.get("code") or "").strip()
        if include_code and code:
            lines.append("Reference code:")
            lines.append("```python")
            lines.append(code)
            lines.append("```")

    return "\n".join(lines)


def load_filtered_skill_entries(
    path: str | os.PathLike[str],
    *,
    max_skills: int = 12,
    include_primitives: bool = False,
) -> list[dict[str, Any]]:
    return _filter_skill_entries(
        load_skill_entries(path),
        max_skills=max_skills,
        include_primitives=include_primitives,
    )


def select_skill_entries(
    entries: list[dict[str, Any]],
    selected_ids: list[str],
) -> list[dict[str, Any]]:
    """Return entries matching planner-selected IDs, names, or catalog indices."""
    lookup: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(entries, start=1):
        skill_id = _skill_identifier(entry, idx)
        keys = {
            skill_id,
            str(entry.get("name") or ""),
            str(entry.get("skill_id") or ""),
            str(idx),
            f"skill_{idx}",
        }
        for key in keys:
            normalized = key.strip().lower()
            if normalized:
                lookup[normalized] = entry

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for selected_id in selected_ids:
        entry = lookup.get(str(selected_id).strip().lower())
        if entry is None:
            continue
        entry_key = id(entry)
        if entry_key in seen:
            continue
        selected.append(entry)
        seen.add(entry_key)
    return selected


def strip_external_skills_from_prompt(prompt: str) -> str:
    """Remove appended external-skill context from a prompt string."""
    marker_idx = prompt.find(EXTERNAL_SKILLS_PROMPT_HEADER)
    if marker_idx == -1:
        return prompt
    return prompt[:marker_idx].rstrip()
