# Evaluation

RATS evaluation compares learned-skill reuse against CaP-X baselines in
LIBERO-PRO and MolmoSpaces. See [setup.md](setup.md) for environment setup.

## LIBERO-PRO RATS

Full pipeline (planner + verifier + retry) over the 6 PRO suites × 10 tasks × 5
trials, skill reuse frozen. Vision servers (SAM3 `8114`, Contact-GraspNet
`8115`, pyroki `8116`) auto-launch from the config.
`RATS_VERIFIER_STRICT_BENCHMARK=1` counts only the native `task_completed` as
success:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CAPX_ENV_STACK=libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RATS_VERIFIER_STRICT_BENCHMARK=1 \
  python scripts/run_rats_libero_pro_batch.py \
    --seed-skill-library outputs/play_libero/snapshots/iter050/skills.json \
    --extra-rats-flags "--model google/gemini-3.5-flash" \
    --output-dir outputs/rats_libero_pro_iter050seed \
    --gpus 0,1,2,3,4,6,7 --workers 18 --skip-completed
```

Omit `--seed-skill-library` for the no-library (noseed) condition. `--config`
defaults to `env_configs/libero/rats_libero_pro_nonpriv.yaml`.

## LIBERO-PRO CaP-X baseline

The CaP-X inference baseline (one-shot + multi-turn writer). Run from
`capx-baseline/` so `import rats` resolves to the baseline copy.
`CAPX_DISABLE_VERTEX=1` routes Gemini through the Developer API (`GEMINI_API_KEY`),
with OpenRouter fallback. Same vision servers as the RATS eval above.

```bash
cd capx-baseline

CAPX_DISABLE_VERTEX=1 GEMINI_API_KEY="$GEMINI_API_KEY" OPENROUTER_API_KEY="$OPENROUTER_API_KEY" MUJOCO_GL=egl \
  python -m capx.envs.scripts.run_libero_batch \
    --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video.yaml \
    --args.models google/gemini-3.1-pro-preview \
    --args.total-trials 5 --args.num-workers 5 \
    --args.suites libero_object_swap \
    --args.output-dir outputs/capx_libero_object_swap
```

For the skill-injected condition, swap the base config to the sibling
`..._rats_skills_iter050.yaml` (`external_skill_library_mode: planner`). Run one
batch per suite, then summarize with
`python scripts/summarize_libero_success.py <output-dir>`.

## MolmoSpaces RATS

Start the MolmoSpaces bridge:

```bash
conda activate mlspaces

MUJOCO_GL=egl python scripts/mlspaces_server.py \
  --port 9162 \
  --task-type open \
  --scene-dataset ithor \
  --data-split test \
  --benchmark-dir rats/benchmarks/molmospaces/capx_rats_eval_core_40 \
  --use-recorded-cameras \
  --output-dir outputs/rats_molmospaces_eval_core_40/mlspaces_server \
  --lazy-init
```

Run RATS over the 40-episode eval subset. Each invocation makes one pass over
all 40 episodes; repeat it for 10 trials (`run0`…`run9`) so the score covers the
full 40 × 10 = 400 rollouts. The config loads the frozen learned skill library
(`skill_library/molmospaces_formula_play.json` — the 13
primitives + 27 skills distilled during formula-play) and uses the perception
API `FrankaMolmoSpacesApiReducedSkillLibrary` (SAM3 + Contact-GraspNet + pyroki
+ molmo — no separate grasp server):

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CAPX_ENV_STACK=molmospaces
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

for trial in $(seq 0 9); do
  python scripts/run_rats.py \
    --config env_configs/molmospaces/rats_molmospaces_eval_core_40.yaml \
    --env-type molmospaces \
    --explore \
    --iterations 40 \
    --model google/gemini-3.5-flash \
    --log-agent-io \
    --output-dir outputs/rats_molmospaces_eval_core_40/run${trial}
done
```

## MolmoSpaces CaP-X baseline

This is the no-RATS counterpart to the RATS eval above: the **same** 40-episode
eval subset and the **same** perception API (`FrankaMolmoSpacesApiReducedSkillLibrary`
— SAM3 + Contact-GraspNet + pyroki + molmo), but with **no learned skill library**
loaded. `--total-trials 10` gives the comparable 40 × 10 = 400 rollouts.

Start the MolmoSpaces bridge for the 40-episode eval subset:

```bash
conda activate mlspaces

MUJOCO_GL=egl python scripts/mlspaces_server.py \
  --port 9161 \
  --task-type open \
  --scene-dataset ithor \
  --data-split test \
  --benchmark-dir rats/benchmarks/molmospaces/capx_rats_eval_core_40 \
  --output-dir outputs/capx_molmospaces_eval_core_40/mlspaces_server \
  --lazy-init
```

Run CaP-X:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CAPX_ENV_STACK=molmospaces
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m rats.envs.scripts.run_molmospaces_batch \
  --base-config-path env_configs/molmospaces/capx_molmospaces_eval_core_40.yaml \
  --total-trials 10 \
  --output-dir outputs/capx_molmospaces_eval_core_40 \
  --models google/gemini-3.5-flash
```

## Outputs

Common artifacts:

- `final_summary.json`: run-level success/failure summary
- `iteration_*.json`: per-iteration planner/writer/verifier records
- `skills.json`: learned or seeded skill library snapshot
- `failure_memory/`: failure episodes and distilled lessons
- `agent_io/`: optional per-agent LLM call logs when `--log-agent-io` is set
- `video_*.mp4` and `iter*_attempt*.mp4`: rollout videos
