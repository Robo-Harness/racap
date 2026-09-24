from pathlib import Path

import yaml


SRC = Path(
    "env_configs/libero/"
    "franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video_rats_skills.yaml"
)
DST = Path(
    "env_configs/libero/"
    "franka_libero_cap_agent0_object_swap_5trials_noensemble_singleturn_video_rats_skills.yaml"
)
OUT = (
    "./outputs/"
    "capx_eval_agent0_object_swap_5trials_noensemble_singleturn_video_gemini31_rats_skill_planner"
)


def main() -> None:
    cfg = yaml.safe_load(SRC.read_text())
    cfg["env"]["cfg"]["multi_turn_prompt"] = None
    cfg["output_dir"] = OUT
    cfg["skill_library_path"] = f"{OUT}/skills.json"

    DST.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
