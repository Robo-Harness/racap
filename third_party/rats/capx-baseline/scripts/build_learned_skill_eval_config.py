#!/usr/bin/env python3
"""Build a CaP-X eval config with learned skills appended to the prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CaP-X LIBERO eval YAML that exposes learned helper "
            "functions from a saved skill-library JSON in the model prompt."
        )
    )
    parser.add_argument(
        "--skill-json",
        type=Path,
        default=Path(
            "outputs/capx_gpt55_30tasks_5attempts_skill_collect/"
            "capx_gpt55_30tasks_5attempts.json"
        ),
        help="Path to the CaP-X learned skill JSON.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("env_configs/libero/franka_libero_cap_agent0_single_turn_skill_collect.yaml"),
        help="Base CaP-X YAML config to modify.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path(
            "outputs/capx_eval_gpt55_object_swap_learned_skills/"
            "base_config_with_learned_skills.yaml"
        ),
        help="Where to write the generated eval YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/capx_eval_gpt55_object_swap_learned_skills",
        help="output_dir value to write into the generated YAML.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Include skills that appeared in at least this many successful trials.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="trials value to write into the generated YAML.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="num_workers value to write into the generated YAML.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Enable video recording in the generated YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Base config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Base config must be a YAML mapping: {path}")
    return data


def load_skills(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Skill JSON not found: {path}")
    data = json.loads(path.read_text())
    skills = data.get("skills", {})
    if not isinstance(skills, dict):
        raise ValueError(f"Skill JSON must contain a dict field named 'skills': {path}")
    return skills


def build_skill_prompt(skills: dict[str, dict[str, Any]], *, min_occurrences: int) -> tuple[str, int]:
    selected = [
        (name, skill)
        for name, skill in sorted(skills.items())
        if int(skill.get("occurrences", 0) or 0) >= min_occurrences
    ]

    lines = [
        "",
        "Reusable helper functions learned from previous successful CaP-X trials:",
        (
            "These functions are not pre-imported. If you use one, copy its "
            "full definition into your generated code, including any helper dependencies."
        ),
        "",
    ]

    for name, skill in selected:
        lines.append(f"### {name}  (occurrences={skill.get('occurrences', 0)})")
        source_tasks = skill.get("source_tasks") or []
        if source_tasks:
            lines.append("Source tasks: " + ", ".join(str(task) for task in source_tasks))
        docstring = str(skill.get("docstring") or "").strip()
        if docstring:
            lines.append(docstring)
        lines.append("```python")
        lines.append(str(skill.get("code") or "").rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines), len(selected)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.base_config)
    skills = load_skills(args.skill_json)
    skill_prompt, count = build_skill_prompt(skills, min_occurrences=args.min_occurrences)

    env_cfg = cfg.setdefault("env", {}).setdefault("cfg", {})
    prompt = str(env_cfg.get("prompt") or "")
    env_cfg["prompt"] = prompt.rstrip() + "\n\n" + skill_prompt

    cfg["evolve_skill_library"] = False
    cfg["skill_library_path"] = str(args.skill_json)
    cfg["output_dir"] = args.output_dir
    cfg["trials"] = args.trials
    cfg["num_workers"] = args.num_workers
    cfg["record_video"] = bool(args.record_video)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Wrote {args.output_config}")
    print(f"Included {count} learned skills with occurrences >= {args.min_occurrences}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
