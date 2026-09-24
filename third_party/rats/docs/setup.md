# Setup

RATS has two execution surfaces:

- the root `rats` checkout, which runs RATS and MolmoSpaces play/eval
- `capx-baseline/`, which runs CaP-X baselines, Robosuite transfer, and real-world transfer

The key environment families are:

- `libero` for normal LIBERO-PRO runs
- `libero-privileged` for smoke tests and privileged LIBERO runs
- `molmospaces` for MolmoSpaces play and evaluation

## Prerequisites

Use a Linux x86_64 machine with an NVIDIA GPU, CUDA-capable PyTorch wheels,
`git`, `uv`, and `conda` available on `PATH`.

Headless MuJoCo / robosuite runs on Ubuntu need the EGL/OpenGL runtime libraries
(libglvnd). Any CUDA/GPU box almost always already has them — check with
`ldconfig -p | grep -E 'libEGL|libGL'` first, and install only if they're
missing (requires sudo):

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libgl1
```

Put the checkout, Python environments, and package caches on a low-latency
executable filesystem with enough free space, ideally a local SSD. Do not place
the repo, `.venv`, conda envs, `UV_CACHE_DIR`, or `PIP_CACHE_DIR` on a `noexec`
mount such as `/dev/shm`; compiled packages such as NumPy and Torch need to map
shared objects during build and import. Avoid slow shared filesystems for
`.venv` and `UV_CACHE_DIR` when possible: they may appear to have enough space
but can stall while `uv` unpacks large CUDA wheels. The checkout itself also
needs quota because editable builds write metadata into the repo and submodule
worktrees. The root LIBERO env pulls large CUDA packages, so plan for at least
100 GB across the repo, submodules, `.venv`, `TMPDIR`, and `uv` cache.
MolmoSpaces assets may require additional tens of GB.

If your home or root disk is small, point caches at a large local executable
disk before running any install command:

```bash
export RATS_CACHE_ROOT=/path/to/large-local-executable-disk/rats-cache
mkdir -p "$RATS_CACHE_ROOT/uv" "$RATS_CACHE_ROOT/pip" "$RATS_CACHE_ROOT/molmospaces"
export UV_CACHE_DIR="$RATS_CACHE_ROOT/uv"
export PIP_CACHE_DIR="$RATS_CACHE_ROOT/pip"
export MLSPACES_CACHE_DIR="$RATS_CACHE_ROOT/molmospaces"
export TMPDIR="$RATS_CACHE_ROOT/tmp"
mkdir -p "$TMPDIR"
```

If `df -h` shows free space but `git submodule update` or `uv sync` still fails
with `Disk quota exceeded`, the filesystem likely has a user or project quota.
Move the checkout and caches to a different disk or increase that quota.

## Repository

```bash
git clone --branch main --depth 1 https://github.com/Playful-RATs/RATs rats
cd rats
```

Initialize only the submodules needed for the runtime you are setting up. For
LIBERO-PRO RATS, start with the root runtime submodules:

```bash
git submodule update --init --depth 1 \
  rats/third_party/LIBERO-PRO \
  rats/third_party/libero_dependencies/robosuite \
  rats/third_party/robosuite \
  rats/third_party/contact_graspnet_pytorch \
  rats/third_party/curobo \
  rats/third_party/sam3
```

Do not run `git submodule update --init --recursive` for the release setup
unless you intentionally want every research submodule. Large optional
submodules such as `rats/third_party/b1k` and `rats/third_party/verl` are not
needed for the release commands and can exceed user quotas on shared
filesystems.

## Common runtime

Set at least one model provider:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Make the root packages importable when running from the repo root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Shared runtime ports used by the release configs:

- PyRoKi: `8116`
- SAM3: `8114`
- Contact-GraspNet: `8115`
- Molmo VLM: `8122`

Check them before a run:

```bash
ss -tln | rg ':8114|:8115|:8116|:8122'
```

## LIBERO-PRO runtime

Use the root RATS env for LIBERO-PRO and MolmoSpaces. The release commands
assume the root env is named `.venv`:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv sync --frozen --active --extra libero --extra contactgraspnet
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If you already have the env, just reactivate it and keep the same exports.

LIBERO creates `~/.libero/config.yaml` on first import and prompts for a
dataset path if that file does not exist. In non-interactive shells, CI, or
batch launchers, pre-create the file so imports do not block on stdin:

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $PWD/rats/third_party/LIBERO-PRO/libero/libero
bddl_files: $PWD/rats/third_party/LIBERO-PRO/libero/libero/./bddl_files
init_states: $PWD/rats/third_party/LIBERO-PRO/libero/libero/./init_files
datasets: $PWD/rats/third_party/LIBERO-PRO/libero/datasets
assets: $PWD/rats/third_party/LIBERO-PRO/libero/libero/./assets
EOF
```

The `datasets` path above may not exist yet on a fresh checkout. That warning
is expected unless you have downloaded LIBERO datasets separately.

`CAPX_ENV_STACK` selects the LIBERO startup family:

```bash
export CAPX_ENV_STACK=libero
```

Use `libero-privileged` only for privileged smoke runs:

```bash
export CAPX_ENV_STACK=libero-privileged
```

## MolmoSpaces runtime

> **Note:** MolmoSpaces (the `allenai/molmospaces` submodule and its assets) is gated by AI2 — you need access to that GitHub repo and the corresponding Hugging Face org. The LIBERO-PRO and cross-environment (Robosuite / real-Franka) workflows do not require it.

If you want `scripts/run_play_molmospaces.sh` to start the local Molmo VLM on
port `8122`, install the root env with the `molmo` extra. That extra provides
`vllm`; without it, `point_prompt_molmo` and VLM-only verification remain
unavailable unless you start an external OpenAI-compatible Molmo server on
`8122`.

```bash
source .venv/bin/activate
uv sync --frozen --active --extra libero --extra contactgraspnet --extra molmo
```

For a MolmoSpaces-only root env, the minimal variant is:

```bash
source .venv/bin/activate
uv sync --frozen --active --extra contactgraspnet --extra molmo
```

Install the MolmoSpaces bridge and assets:

```bash
git submodule update --init --depth 1 rats/third_party/molmospaces
bash scripts/setup_molmospaces.sh

cd rats/third_party/molmospaces
conda create -n mlspaces python=3.11 -y
conda activate mlspaces
pip install -e .[mujoco]
cd ../../..

# Optional: set this before bootstrapping if the default repo-local cache is too small.
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$PWD/rats-cache/molmospaces}"
export MLSPACES_ASSETS_DIR="$PWD/rats/third_party/molmospaces/assets"
bash scripts/bootstrap_molmospaces_assets.sh
```

`bootstrap_molmospaces_assets.sh` normalizes relative cache paths to absolute
paths before creating asset symlinks. It also prefetches the scenes, Objaverse
objects, and Objaverse grasps referenced by the default MolmoSpaces play
benchmark so the play server does not need to lazily download them at startup.

If you later move the shared cache, keep both variables set when bootstrapping
and running MolmoSpaces:

```bash
export MLSPACES_CACHE_DIR=/path/to/your/cache
export MLSPACES_ASSETS_DIR="$PWD/rats/third_party/molmospaces/assets"
```

## CaP-X baseline runtime

Use `capx-baseline/` for CaP-X-only runs, Robosuite transfer, and real-world transfer:

```bash
git submodule update --init --depth 1 \
  rats/third_party/LIBERO-PRO \
  rats/third_party/libero_dependencies/robosuite \
  rats/third_party/contact_graspnet_pytorch \
  rats/third_party/curobo \
  rats/third_party/sam3

cd capx-baseline
bash scripts/setup_third_party.sh
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --frozen --active --extra libero --extra contactgraspnet
```

For Robosuite transfer, use a separate Robosuite-enabled env:

```bash
git submodule update --init --depth 1 rats/third_party/robosuite

uv venv .venv-robosuite --python 3.10
source .venv-robosuite/bin/activate
uv sync --frozen --active --extra robosuite --extra contactgraspnet
```
