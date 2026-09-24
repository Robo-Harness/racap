# Real Franka Panda QuickStart

Make sure you have [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) cloned and have tested launching the real Franka Panda using the robots_realtime repo instructions. For example test with `configs/franka/franka_robotiq_viser_teleop.yaml` (uses a robotiq gripper interfacing with an RS485).

The default task configured in [`env_configs/real/real.yaml`](../env_configs/real/real.yaml) is **"pick up the red cube and lift it"**. Feel free to modify the `task_only_prompt` and `prompt` fields in that file to define your own task, e.g. another cool task that we have tried in real (and works!) is: **"stack these objects as high as possible"**

## Requirements

Beyond a Franka Panda robot arm, this workflow requires a **stereo camera that produces calibrated metric-scale depth maps** (e.g. a ZED stereo camera). The depth data is used by SAM3 + Contact-GraspNet to generate grasp poses in 3D. A monocular RGB-only camera may be insufficient.

## Camera Extrinsics Setup

The `robots_realtime` client config points to a camera extrinsics file that describes the camera's pose in the robot world frame. You must create one for your own setup.

**1. Create your extrinsics file** in `robots_realtime/configs/camera_extrinsics/`. Use the AutoLab ZED setup as a reference:

[`configs/camera_extrinsics/autolab_franka_zed_top.yaml`](https://github.com/uynitsuj/robots_realtime/blob/main/configs/camera_extrinsics/autolab_franka_zed_top.yaml)

```yaml
# Camera extrinsics for your Franka setup.
#
# All values are expressed in the robot world frame (base link origin).
#
# position: [x, y, z] in meters
# rpy_radians: [roll, pitch, yaw] in radians (applied in that order)
#
# For your own setup, create a new file in this directory, then point
# the ZedCamera config at it via the `extrinsics_file` field.

position: [1.007, 0.0, 0.29]
rpy_radians: [1.0472, 3.14159, -1.5708]
```

**2. Point the client config at your file.** In `robots_realtime/configs/franka/franka_robotiq_client.yaml` (or whichever client config you use), update the `extrinsics_file` field under the ZedCamera sensor node:

```yaml
extrinsics_file: "configs/camera_extrinsics/your_setup.yaml"
```

**3. Calibrate your extrinsics.** The `position` and `rpy_radians` values must match your physical camera mounting. Poor calibration will cause grasp poses to be offset from the real object locations.

**4. Then run the CaP-X real experiment configs.**
```bash
uv sync --active --extra contactgraspnet
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/real.yaml
```

Open up the interactive web UI at the provided port (defaulted to http://localhost:8200).

Once you see:
```
Waiting for observation from real environment...
Waiting for observation from real environment...
```
In a separate terminal with the [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) repo set as the current directory, launch:
```bash
uv run rr-session configs/franka/franka_robotiq_client.yaml
```

## Running with the LIBERO iter050 skill library (v2 planner mode)

To run the same real-world setup but with the iter050 RATS-learned skill library available to the policy writer through the **v2 planner-mode** skill-retrieval path (the same mechanism used in the LIBERO Table-2 reproduction config `franka_libero_cap_agent0_..._rats_skills_iter050.yaml`), use the sibling config [`env_configs/real/real_libero.yaml`](../env_configs/real/real_libero.yaml):

```bash
uv sync --active --extra contactgraspnet
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/real_libero.yaml
```

What this changes compared to `real.yaml`:

- **Skill library** — loads [`skill_libraries/libero_iter050.json`](../skill_libraries/libero_iter050.json) (the preserved iter050 RATS training snapshot, 59 skills).
- **v2 planner mode** — a planner LLM (`google/gemini-3.1-pro-preview`) selects up to 6 skills per turn based on the task; only those selected skills are appended to the policy-writer prompt with their code.
- **Same API** — still `FrankaRealControlApi`, so skill code is shown as inspiration; LIBERO-only calls that don't have a real-world equivalent will fail at runtime and trigger a `REGENERATE` on the multi-turn loop.
- **Output dir** — `./outputs/franka_real_libero/` (so it doesn't overwrite `real.yaml`'s output).
- **`save_skill_planner_prompts: true`** — the per-turn planner prompt is saved so you can inspect which skills it picked.

Everything else (web UI port, vision servers, `robots_realtime` client) is identical to `real.yaml` — the same `rr-session` command from above is what you launch in the second terminal.

## Running with the MolmoSpaces formula-play skill library (v2 planner mode)

Same v2 planner-mode path as the iter050 demo above, but the injected library is the **MolmoSpaces formula-play skills** (40 skills: 13 primitives + 27 learned skills, sourced from `skill_library/molmospaces_formula_play.json`). Use the sibling config [`env_configs/real/real_molmospace.yaml`](../env_configs/real/real_molmospace.yaml):

```bash
uv sync --active --extra contactgraspnet
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/real_molmospace.yaml
```

What this changes compared to `real_libero.yaml`:

- **Skill library** — loads [`skill_libraries/molmospace_formula_play.json`](../skill_libraries/molmospace_formula_play.json) (40 skills learned on the FrankaMolmoSpacesNonPriv env).
- **API match** — the MolmoSpaces perception API (`point_prompt_molmo`, `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, `plan_grasp`, `solve_ik`, `move_to_joints`, `open_gripper`, `close_gripper`) is the same surface exposed by `FrankaRealControlApi` (see `capx-baseline/capx/integrations/franka/control_reduced.py`), so most learned skills crib cleanly. MolmoSpaces-only calls (`inspect_at_wrist`, `mask_to_world_points`) will fail at runtime and trigger `REGENERATE`.
- **Output dir** — `./outputs/franka_real_molmospace/` (so it doesn't overwrite `real_libero`'s output).

Everything else (web UI port, vision servers, planner model, `robots_realtime` client) is identical to `real_libero.yaml`.
