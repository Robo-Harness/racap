#!/usr/bin/env bash
# Launch the focused formula play ablation: mlspaces server + RATS lifelong loop.
# Usage: scripts/run_play_molmospaces.sh [output_dir]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"

OUTPUT_DIR="${1:-outputs/play_molmospaces}"
MLSPACES_PORT=9270
BENCHMARK_DIR="rats/benchmarks/molmospaces/capx_rats_play_disjoint_from_eval40"
CONFIG="env_configs/molmospaces/play_molmospaces.yaml"
ITERATIONS=50
SNAPSHOT_INTERVAL=10

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MLSPACES_CONDA_ENV="${MLSPACES_CONDA_ENV:-mlspaces}"
MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$REPO_ROOT/rats-cache/molmospaces}"
case "$MLSPACES_CACHE_DIR" in
  /*) ;;
  *) MLSPACES_CACHE_DIR="$REPO_ROOT/$MLSPACES_CACHE_DIR" ;;
esac
MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$REPO_ROOT/rats/third_party/molmospaces/assets}"
RATS_LLM_FALLBACK="${RATS_LLM_FALLBACK:-1}"
CAPX_GEMINI_RETRY_MAX="${CAPX_GEMINI_RETRY_MAX:-0}"
RATS_GEMINI_PROXY_ATTEMPTS="${RATS_GEMINI_PROXY_ATTEMPTS:-1}"
RATS_OPENROUTER_RETRY_MAX="${RATS_OPENROUTER_RETRY_MAX:-0}"
EGL_DEVICE_ID="${EGL_DEVICE_ID:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$EGL_DEVICE_ID}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$EGL_DEVICE_ID}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
GRASPNET_HOST="${GRASPNET_HOST:-127.0.0.1}"
GRASPNET_PORT="${GRASPNET_PORT:-8115}"
GRASPNET_DEVICE="${GRASPNET_DEVICE:-cuda}"
GRASPNET_CUDA_VISIBLE_DEVICES="${GRASPNET_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
GRASPNET_SERVICE_URL="${GRASPNET_SERVICE_URL:-http://${GRASPNET_HOST}:${GRASPNET_PORT}}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_PORT:-8114}"
SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
SAM3_CUDA_VISIBLE_DEVICES="${SAM3_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
MOLMO_HOST="${MOLMO_HOST:-127.0.0.1}"
MOLMO_PORT="${MOLMO_PORT:-8122}"
MOLMO_MODEL="${MOLMO_MODEL:-allenai/Molmo2-8B}"
MOLMO_CUDA_VISIBLE_DEVICES="${MOLMO_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
MOLMO_VLLM_BIN="${MOLMO_VLLM_BIN:-}"
REQUIRE_MOLMO_VLM="${REQUIRE_MOLMO_VLM:-0}"

# --- Helpers ---
port_open() {
  "$PYTHON_BIN" -c 'import socket,sys
s=socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1"
}

wait_for_port() {
  local port="$1" label="$2" timeout="${3:-180}" start
  start="$(date +%s)"
  while ! port_open "$port"; do
    if (( "$(date +%s)" - start >= timeout )); then
      echo "ERROR: Timed out waiting for ${label} on port ${port}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "  ${label} ready on port ${port}"
}

port_pids() {
  local port="$1"
  command -v fuser >/dev/null 2>&1 || return 0
  fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+$/ {print}'
}

process_cmdline() {
  local pid="$1"
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
}

process_env_value() {
  local pid="$1" key="$2"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n "s/^${key}=//p" | head -n 1
}

mlspaces_server_pid() {
  local pid cmd
  for pid in $(port_pids "$MLSPACES_PORT"); do
    cmd="$(process_cmdline "$pid")"
    if [[ "$cmd" == *"scripts/mlspaces_server.py"* ]]; then
      echo "$pid"
      return 0
    fi
  done
}

mlspaces_env_matches() {
  local pid="$1" key expected actual
  for key in CUDA_VISIBLE_DEVICES EGL_DEVICE_ID MUJOCO_EGL_DEVICE_ID MLSPACES_CACHE_DIR MLSPACES_ASSETS_DIR; do
    expected="${!key}"
    actual="$(process_env_value "$pid" "$key")"
    if [[ "$actual" != "$expected" ]]; then
      echo "  $key: running=${actual:-<unset>} requested=$expected"
      return 1
    fi
  done
  return 0
}

check_molmospaces_assets() {
  local material_db="$MLSPACES_ASSETS_DIR/objects/thor/material-database.json"
  if [[ -e "$material_db" ]]; then
    return 0
  fi

  echo "ERROR: MolmoSpaces assets are not linked correctly." >&2
  echo "Missing: $material_db" >&2
  if [[ -L "$MLSPACES_ASSETS_DIR/objects/thor" ]]; then
    echo "objects/thor -> $(readlink "$MLSPACES_ASSETS_DIR/objects/thor")" >&2
  fi
  echo "Expected cache: $MLSPACES_CACHE_DIR" >&2
  echo "Repair with:" >&2
  echo "  MLSPACES_CACHE_DIR=\"$MLSPACES_CACHE_DIR\" conda run --no-capture-output -n \"$MLSPACES_CONDA_ENV\" bash scripts/bootstrap_molmospaces_assets.sh" >&2
  exit 1
}

cleanup() {
  local status=$?
  if [[ -n "${MLSPACES_PID:-}" ]]; then
    kill "$MLSPACES_PID" 2>/dev/null || true
  fi
  if [[ -n "${GRASPNET_PID:-}" ]]; then
    kill "$GRASPNET_PID" 2>/dev/null || true
  fi
  if [[ -n "${SAM3_PID:-}" ]]; then
    kill "$SAM3_PID" 2>/dev/null || true
  fi
  if [[ -n "${MOLMO_PID:-}" ]]; then
    kill "$MOLMO_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

# --- Preflight checks ---
echo "=== Play ablation: formula (focused interactions) ==="
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Iterations: $ITERATIONS (snapshot every $SNAPSHOT_INTERVAL)"
echo "GPU:        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES EGL_DEVICE_ID=$EGL_DEVICE_ID MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "Assets:     MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR"
echo "Cache:      MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR"
echo "SAM3:       ${SAM3_HOST}:${SAM3_PORT} device=$SAM3_DEVICE server_cuda=$SAM3_CUDA_VISIBLE_DEVICES"
echo "GraspNet:   $GRASPNET_SERVICE_URL device=$GRASPNET_DEVICE server_cuda=$GRASPNET_CUDA_VISIBLE_DEVICES"
echo "Molmo VLM:  ${MOLMO_HOST}:${MOLMO_PORT} model=$MOLMO_MODEL server_cuda=$MOLMO_CUDA_VISIBLE_DEVICES"
echo "LLM retry:  fallback=$RATS_LLM_FALLBACK genai_retries=$CAPX_GEMINI_RETRY_MAX proxy_attempts=$RATS_GEMINI_PROXY_ATTEMPTS openrouter_retries=$RATS_OPENROUTER_RETRY_MAX"
echo ""

# Ensure SAM3 is available for text / point segmentation.
check_molmospaces_assets
mkdir -p "$OUTPUT_DIR"
if port_open "$SAM3_PORT"; then
  echo "[OK] SAM3 already running on port $SAM3_PORT"
  SAM3_PID=""
else
  echo "Starting SAM3 on ${SAM3_HOST}:${SAM3_PORT} ..."
  env CUDA_VISIBLE_DEVICES="$SAM3_CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" -m rats.serving.launch_sam3_server \
      --host "$SAM3_HOST" \
      --port "$SAM3_PORT" \
      --device "$SAM3_DEVICE" \
      > "$OUTPUT_DIR/sam3_server.log" 2>&1 &
  SAM3_PID=$!
  wait_for_port "$SAM3_PORT" "SAM3" 240
fi

# Ensure GraspNet is available for plan_grasp / plan_grasp_from_point_clouds.
if port_open "$GRASPNET_PORT"; then
  echo "[OK] GraspNet already running on port $GRASPNET_PORT"
  GRASPNET_PID=""
else
  echo "Starting GraspNet on ${GRASPNET_HOST}:${GRASPNET_PORT} ..."
  env CUDA_VISIBLE_DEVICES="$GRASPNET_CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" -m capx.serving.launch_contact_graspnet_server \
      --host "$GRASPNET_HOST" \
      --port "$GRASPNET_PORT" \
      --device "$GRASPNET_DEVICE" \
      > "$OUTPUT_DIR/graspnet_server.log" 2>&1 &
  GRASPNET_PID=$!
  wait_for_port "$GRASPNET_PORT" "GraspNet" 240
fi

# Start/check Molmo VLM for point_prompt_molmo and optional VLM verification.
if port_open "$MOLMO_PORT"; then
  echo "[OK] Molmo VLM already running on port $MOLMO_PORT"
  MOLMO_PID=""
else
  if [[ -z "$MOLMO_VLLM_BIN" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/vllm" ]]; then
      MOLMO_VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
      MOLMO_VLLM_BIN="$(command -v vllm)"
    fi
  fi

  if [[ -n "$MOLMO_VLLM_BIN" ]]; then
    echo "Starting Molmo VLM on ${MOLMO_HOST}:${MOLMO_PORT} ..."
    mkdir -p "$OUTPUT_DIR/molmo_server"
    env CUDA_VISIBLE_DEVICES="$MOLMO_CUDA_VISIBLE_DEVICES" \
      "$MOLMO_VLLM_BIN" serve "$MOLMO_MODEL" \
        --host "$MOLMO_HOST" \
        --port "$MOLMO_PORT" \
        --trust-remote-code \
        --gpu-memory-utilization "${MOLMO_GPU_MEMORY_UTILIZATION:-0.6}" \
        --dtype "${MOLMO_DTYPE:-bfloat16}" \
        --max-model-len "${MOLMO_MAX_MODEL_LEN:-8192}" \
        --max-num-batched-tokens "${MOLMO_MAX_NUM_BATCHED_TOKENS:-8192}" \
        > "$OUTPUT_DIR/molmo_server/molmo.log" 2>&1 &
    MOLMO_PID=$!
    wait_for_port "$MOLMO_PORT" "Molmo VLM" 600
  else
    MOLMO_PID=""
    echo "WARNING: Molmo VLM not running on port $MOLMO_PORT and vLLM was not found; point_prompt_molmo and VLM verifier will be unavailable." >&2
    if [[ "$REQUIRE_MOLMO_VLM" == "1" ]]; then
      echo "ERROR: REQUIRE_MOLMO_VLM=1 but Molmo VLM could not be started." >&2
      exit 1
    fi
  fi
fi

# Check API keys
if [[ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
  echo "WARNING: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set." >&2
fi

# --- Start MolmoSpaces bridge server ---
mkdir -p "$OUTPUT_DIR/mlspaces_server"

START_MLSPACES=1
if port_open "$MLSPACES_PORT"; then
  EXISTING_MLSPACES_PID="$(mlspaces_server_pid || true)"
  if [[ -n "$EXISTING_MLSPACES_PID" ]]; then
    if mlspaces_env_matches "$EXISTING_MLSPACES_PID"; then
      echo "[OK] MolmoSpaces server already running on port $MLSPACES_PORT with matching env; reusing."
      MLSPACES_PID=""
      START_MLSPACES=0
    else
      echo "MolmoSpaces server on port $MLSPACES_PORT has stale env; restarting pid $EXISTING_MLSPACES_PID."
      kill "$EXISTING_MLSPACES_PID" 2>/dev/null || true
      for _ in {1..30}; do
        port_open "$MLSPACES_PORT" || break
        sleep 1
      done
      if port_open "$MLSPACES_PORT"; then
        echo "ERROR: Timed out waiting for stale MolmoSpaces server to stop." >&2
        exit 1
      fi
    fi
  else
    echo "ERROR: Port $MLSPACES_PORT is already in use by a non-MolmoSpaces process." >&2
    exit 1
  fi
fi

if (( START_MLSPACES )); then
  echo "Starting MolmoSpaces server on port $MLSPACES_PORT ..."
  CLEAN_PATH="$("$PYTHON_BIN" -c 'import os; print(":".join(p for p in os.environ["PATH"].split(":") if ".venv" not in p))')"
  env -u VIRTUAL_ENV -u CONDA_PROMPT_MODIFIER \
    PATH="$CLEAN_PATH" \
    MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    EGL_DEVICE_ID="$EGL_DEVICE_ID" \
    MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    XLA_FLAGS="$XLA_FLAGS" \
    MLSPACES_CACHE_DIR="$MLSPACES_CACHE_DIR" \
    MLSPACES_ASSETS_DIR="$MLSPACES_ASSETS_DIR" \
    PYTHONPATH="$PWD/rats/third_party/molmospaces:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "$MLSPACES_CONDA_ENV" python scripts/mlspaces_server.py \
      --port "$MLSPACES_PORT" \
      --task-type pick_and_place \
      --benchmark-dir "$BENCHMARK_DIR" \
      --use-recorded-cameras \
      --output-dir "$OUTPUT_DIR/mlspaces_server" \
      --lazy-init \
      > "$OUTPUT_DIR/mlspaces_server.log" 2>&1 &
  MLSPACES_PID=$!
  wait_for_port "$MLSPACES_PORT" "MolmoSpaces" 180
fi

# --- Launch RATS ---
echo ""
echo "Starting RATS lifelong loop ..."
env \
  RATS_SUBAGENT_DISABLED=1 \
  RATS_LLM_MODEL="google/gemini-3.1-pro-preview" \
  RATS_LLM_FALLBACK="$RATS_LLM_FALLBACK" \
  RATS_OPENROUTER_URL="http://localhost:8111/chat/completions" \
  CAPX_GEMINI_RETRY_MAX="$CAPX_GEMINI_RETRY_MAX" \
  RATS_GEMINI_PROXY_ATTEMPTS="$RATS_GEMINI_PROXY_ATTEMPTS" \
  RATS_OPENROUTER_RETRY_MAX="$RATS_OPENROUTER_RETRY_MAX" \
  RATS_POLICY_WRITER_MODEL="${RATS_POLICY_WRITER_MODEL:-google/gemini-3.1-pro-preview}" \
  GRASPNET_SERVICE_URL="$GRASPNET_SERVICE_URL" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  EGL_DEVICE_ID="$EGL_DEVICE_ID" \
  MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  XLA_FLAGS="$XLA_FLAGS" \
  MLSPACES_CACHE_DIR="$MLSPACES_CACHE_DIR" \
  MLSPACES_ASSETS_DIR="$MLSPACES_ASSETS_DIR" \
  "$PYTHON_BIN" scripts/run_rats.py \
    --config "$CONFIG" \
    --explore --play-mode \
    --iterations "$ITERATIONS" \
    --log-agent-io \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --output-dir "$OUTPUT_DIR" \
    --proposer-include-eval-task-context \
    --skip-completed
