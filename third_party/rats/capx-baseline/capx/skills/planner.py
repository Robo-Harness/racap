"""Planner for selecting a small subset of external reusable skills."""

from __future__ import annotations

import json
import re
from typing import Any

from capx.llm.client import ModelQueryArgs, query_model
from capx.skills.rats_adapter import (
    format_skill_catalog_for_planner,
    format_skill_entries_for_prompt,
    load_filtered_skill_entries,
    select_skill_entries,
)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("Skill planner response did not contain a JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Skill planner JSON must be an object")
    return data


def parse_skill_planner_response(text: str, *, max_selected: int) -> dict[str, Any]:
    """Parse planner output and normalize selected skill identifiers."""
    data = _extract_json_object(text)
    selected = (
        data.get("selected_skill_ids")
        or data.get("selected_skills")
        or data.get("skills")
        or []
    )
    if not isinstance(selected, list):
        selected = []

    selected_ids: list[str] = []
    for item in selected:
        if isinstance(item, dict):
            item = item.get("skill_id") or item.get("id") or item.get("name")
        if item is None:
            continue
        selected_ids.append(str(item))

    if max_selected > 0:
        selected_ids = selected_ids[:max_selected]

    return {
        "selected_skill_ids": selected_ids,
        "rationale": str(data.get("rationale") or data.get("reason") or ""),
    }


def build_skill_planner_prompt(
    *,
    task_context: str,
    skill_catalog: str,
    max_selected: int,
) -> list[dict[str, Any]]:
    """Build the prompt used to select relevant skills."""
    selection_limit = (
        f"Choose at most {max_selected} skills"
        if max_selected > 0
        else "Choose any number of skills"
    )
    user_text = f"""Task and environment context:
{task_context}

{skill_catalog}

{selection_limit} that are likely to help the code-writing model solve the current task.
Select only skills that are directly useful. It is acceptable to select zero skills.
Return strict JSON only, with this schema:
{{
  "selected_skill_ids": ["skill_id_1", "skill_id_2"],
  "rationale": "brief reason for the selection"
}}
Do not include Markdown, code, or any text outside the JSON object.
"""
    return [
        {
            "role": "system",
            "content": (
                "You are a robotics skill-selection planner. Your job is to "
                "choose a small subset of reusable skills for a separate "
                "code-writing policy. You do not write robot-control code."
            ),
        },
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def run_external_skill_planner(
    *,
    args: Any,
    task_context: str,
    skill_library_path: str,
    max_library_skills: int,
    include_primitives: bool,
    planner_include_code: bool,
    policy_include_code: bool,
    max_selected: int,
    planner_model: str | None = None,
) -> dict[str, Any]:
    """Select relevant external skills and format them for the policy writer."""
    entries = load_filtered_skill_entries(
        skill_library_path,
        max_skills=max_library_skills,
        include_primitives=include_primitives,
    )
    catalog = format_skill_catalog_for_planner(
        entries,
        include_code=planner_include_code,
    )
    planner_prompt = build_skill_planner_prompt(
        task_context=task_context,
        skill_catalog=catalog,
        max_selected=max_selected,
    )

    result: dict[str, Any] = {
        "planner_prompt": planner_prompt,
        "planner_response": None,
        "selected_skill_ids": [],
        "selected_skills_prompt": "",
        "selected_skill_count": 0,
        "rationale": "",
        "error": None,
    }

    if not entries:
        result["error"] = "No external skill entries were available."
        return result

    query_args = ModelQueryArgs(
        model=planner_model or args.model,
        server_url=args.server_url,
        api_key=getattr(args, "api_key", None),
        temperature=0.0,
        max_tokens=min(getattr(args, "max_tokens", 4096), 2048),
        reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        debug=getattr(args, "debug", False),
    )

    try:
        response = query_model(query_args, planner_prompt)
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        parsed = parse_skill_planner_response(content, max_selected=max_selected)
        selected_entries = select_skill_entries(entries, parsed["selected_skill_ids"])
        selected_prompt = format_skill_entries_for_prompt(
            selected_entries,
            include_code=policy_include_code,
        )
        result.update(
            {
                "planner_response": content,
                "selected_skill_ids": parsed["selected_skill_ids"],
                "selected_skills_prompt": selected_prompt,
                "selected_skill_count": len(selected_entries),
                "rationale": parsed["rationale"],
            }
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result
