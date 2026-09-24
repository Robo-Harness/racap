# Cross-Environment Evaluation

Cross-environment eval checks whether skills learned in one environment still
help policy generation in another. The learned skill code is used as planner
context, not imported as a module. See [setup.md](setup.md) for the
`capx-baseline/` and Robosuite environments.

## Robosuite

Robosuite transfer uses the CaP-X baseline runner with the LIBERO iter050 skill
library preserved at `capx-baseline/skill_libraries/libero_iter050.json`.

Release configs live in:

```text
capx-baseline/env_configs/robosuite_7task_10trial_compare/
```

Each `*_rats_iter050.yaml` config loads the frozen LIBERO skill library in
planner mode. Run one pair like this:

```bash
cd capx-baseline
source .venv-robosuite/bin/activate

python -m capx.envs.launch \
  --config-path env_configs/robosuite_7task_10trial_compare/cube_lifting_no_rats.yaml \
  --model google/gemini-3.1-pro-preview \
  --output-dir outputs/robosuite_compare/cube_lifting_no_rats

python -m capx.envs.launch \
  --config-path env_configs/robosuite_7task_10trial_compare/cube_lifting_rats_iter050.yaml \
  --model google/gemini-3.1-pro-preview \
  --output-dir outputs/robosuite_compare/cube_lifting_rats_iter050
```

To run the full seven-task comparison, repeat that pair for:

- `cube_lifting`
- `cube_restack`
- `cube_stack`
- `nut_assembly`
- `spill_wipe`
- `two_arm_handover`
- `two_arm_lift`

## Real World

Real-world transfer lives in `capx-baseline/env_configs/real/`. First read
[../capx-baseline/docs/real-franka.md](../capx-baseline/docs/real-franka.md) and
bring up the Franka Panda with `robots_realtime`. The setup needs calibrated
metric-depth camera extrinsics plus the usual SAM3, Contact-GraspNet, and
PyRoKi servers.

Run the three release configs:

```bash
cd capx-baseline
source .venv-libero/bin/activate

python -m capx.envs.launch --config-path env_configs/real/real.yaml
python -m capx.envs.launch --config-path env_configs/real/real_libero.yaml
python -m capx.envs.launch --config-path env_configs/real/real_molmospace.yaml
```

`real_libero.yaml` loads `skill_libraries/libero_iter050.json`.
`real_molmospace.yaml` loads `skill_libraries/molmospace_formula_play.json`.
