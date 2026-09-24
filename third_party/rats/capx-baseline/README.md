# CaP-X Baseline for RATS

This directory is a local CaP-X baseline snapshot used for LIBERO object-swap evaluations. It is intended to live inside the RATS repository so coworkers can run the same baseline without needing a separate CaP-X checkout.

The snapshot includes CaP-X source code, configs, scripts, tests, and lightweight assets. It intentionally does not include experiment outputs, virtual environments, logs, nested git metadata, or heavyweight third-party dependency checkouts.

## Layout

- `capx/`: CaP-X Python package.
- `env_configs/libero/`: LIBERO configs used for object-swap baseline runs.
- `scripts/summarize_libero_success.py`: utility for summarizing `taskcompleted_1` results.
- `scripts/setup_third_party.sh`: helper to fetch excluded third-party repos.
- `outputs/`: generated at runtime and git-ignored.

## One-Time Setup

Run commands from this directory:

```bash
cd capx-baseline
```

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Fetch the third-party repos excluded from the RATS git snapshot:

```bash
bash scripts/setup_third_party.sh
```

Create the LIBERO environment:

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

SAM3 requires HuggingFace access. Log in before the first run:

```bash
huggingface-cli login
```

## Model Proxy

The configs assume an OpenAI-compatible chat endpoint at:

```text
http://127.0.0.1:8110/chat/completions
```

For OpenRouter:

```bash
echo "sk-or-v1-your-key" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py \
  --key-file .openrouterkey \
  --port 8110
```

`.openrouterkey` is git-ignored.

For Gemini proxy usage, set:

```bash
export CAPX_USE_GEMINI_PROXY=1
```

## Run Object-Swap Baselines

Activate the LIBERO environment first:

```bash
source .venv-libero/bin/activate
```

GPT-5.5 single-turn, no RATS skills:

```bash
python -m capx.envs.scripts.run_libero_batch \
  --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_singleturn_video.yaml \
  --args.suites libero_object_swap \
  --args.models openai/gpt-5.5 \
  --args.total-trials 5 \
  --args.num-workers 1 \
  --args.record-video True \
  --args.output-dir outputs/capx_eval_agent0_object_swap_5trials_noensemble_singleturn_video
```

GPT-5.5 multi-turn, no RATS skills:

```bash
python -m capx.envs.scripts.run_libero_batch \
  --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video.yaml \
  --args.suites libero_object_swap \
  --args.models openai/gpt-5.5 \
  --args.total-trials 5 \
  --args.num-workers 1 \
  --args.record-video True \
  --args.use-img-differencing True \
  --args.output-dir outputs/capx_eval_agent0_object_swap_5trials_noensemble_multiturn_video
```

Gemini 3.1 multi-turn, no RATS skills:

```bash
CAPX_USE_GEMINI_PROXY=1 python -m capx.envs.scripts.run_libero_batch \
  --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video.yaml \
  --args.suites libero_object_swap \
  --args.models google/gemini-3.1-pro-preview \
  --args.total-trials 5 \
  --args.num-workers 1 \
  --args.record-video True \
  --args.use-img-differencing True \
  --args.output-dir outputs/capx_eval_agent0_object_swap_5trials_noensemble_multiturn_video_gemini31
```

Gemini 3.1 single-turn with RATS skill planner:

```bash
CAPX_USE_GEMINI_PROXY=1 python -m capx.envs.scripts.run_libero_batch \
  --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_singleturn_video_rats_skills.yaml \
  --args.suites libero_object_swap \
  --args.models google/gemini-3.1-pro-preview \
  --args.total-trials 5 \
  --args.num-workers 1 \
  --args.record-video True \
  --args.output-dir outputs/capx_eval_agent0_object_swap_5trials_noensemble_singleturn_video_gemini31_rats_skill_planner
```

Gemini 3.1 multi-turn with RATS skill planner:

```bash
CAPX_USE_GEMINI_PROXY=1 python -m capx.envs.scripts.run_libero_batch \
  --args.base-config-path env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video_rats_skills.yaml \
  --args.suites libero_object_swap \
  --args.models google/gemini-3.1-pro-preview \
  --args.total-trials 5 \
  --args.num-workers 1 \
  --args.record-video True \
  --args.use-img-differencing True \
  --args.output-dir outputs/capx_eval_agent0_object_swap_5trials_noensemble_multiturn_video_gemini31_rats_skill_planner
```

## RATS Skills File

The RATS-skill configs expect:

```text
../skill_library/libero_main_30iter.json
```

This path is relative to `capx-baseline/`. If the skills file is elsewhere, edit `external_skill_library_path` in:

- `env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_singleturn_video_rats_skills.yaml`
- `env_configs/libero/franka_libero_cap_agent0_object_swap_5trials_noensemble_multiturn_video_rats_skills.yaml`

## Summarize Results

```bash
python scripts/summarize_libero_success.py \
  outputs/capx_eval_agent0_object_swap_5trials_noensemble_singleturn_video
```

The summary script reads trial directory names and reports per-task plus total `taskcompleted_1` rates.

## Notes

- Do not commit `outputs/`, videos, logs, `.venv-libero/`, API keys, or downloaded third-party checkouts.
- The third-party repos are deliberately fetched by `scripts/setup_third_party.sh` rather than committed into RATS.
- Perception servers for PyRoKi, Contact-GraspNet, and SAM3 are declared in the YAML configs and launched by CaP-X.
