#!/usr/bin/env bash
# Source this file before running RACaP: source configs/env.sh

_racap_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RACAP_ROOT="$(cd "$_racap_config_dir/.." && pwd)"
export RATS_REPO="${RATS_REPO:-$RACAP_ROOT/third_party/rats}"
export VIRTUAL_ENV="${RACAP_VIRTUAL_ENV:-$RATS_REPO/.venv}"

export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$RACAP_ROOT:$RATS_REPO${PYTHONPATH:+:$PYTHONPATH}"

export RACAP_CACHE_DIR="${RACAP_CACHE_DIR:-$RACAP_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RACAP_CACHE_DIR/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RACAP_CACHE_DIR/pip}"
export HF_HOME="${HF_HOME:-$RACAP_CACHE_DIR/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$RACAP_CACHE_DIR/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RACAP_CACHE_DIR/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RACAP_CACHE_DIR/xdg}"
export RACAP_LLM_CACHE_DIR="${RACAP_LLM_CACHE_DIR:-$RACAP_CACHE_DIR/llm}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RACAP_ROOT/.runtime/libero}"

export CAPX_ENV_STACK="${CAPX_ENV_STACK:-libero}"
export RATS_VERIFIER_STRICT_BENCHMARK=1
export RATS_VERIFY_STEP_MODE=strict

if [[ -f "$RACAP_ROOT/configs/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$RACAP_ROOT/configs/local.env"
fi

export RACAP_SAM3_URL="${RACAP_SAM3_URL:-http://127.0.0.1:8114}"
export RACAP_GRASPNET_URL="${RACAP_GRASPNET_URL:-http://127.0.0.1:8115}"
export RACAP_PYROKI_URL="${RACAP_PYROKI_URL:-http://127.0.0.1:8116}"
export RACAP_MODEL="${RACAP_MODEL:-gpt-5.5}"
export RACAP_GROUNDER_MODEL="${RACAP_GROUNDER_MODEL:-$RACAP_MODEL}"
export RACAP_VERIFIER_MODEL="${RACAP_VERIFIER_MODEL:-$RACAP_MODEL}"
export RACAP_CODER_MODEL="${RACAP_CODER_MODEL:-gpt-5.6-sol}"
export RACAP_CRITIC_MODEL="${RACAP_CRITIC_MODEL:-gpt-5.5}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
else
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi

mkdir -p "$RACAP_CACHE_DIR" "$RACAP_LLM_CACHE_DIR" "$LIBERO_CONFIG_PATH" \
  "$RACAP_ROOT/outputs"

unset _racap_config_dir
