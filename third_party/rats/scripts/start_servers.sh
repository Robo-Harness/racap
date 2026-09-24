#!/usr/bin/env bash
# Start vision servers with log files (matching partner's setup)
# Usage: bash scripts/start_servers.sh
#
# Servers:
#   SAM3       → port 8114, GPU 4, log: logs/sam3_8114.log
#   GraspNet   → port 8115, GPU 4, log: logs/graspnet_8115.log
#   pyroki     → port 8116, CPU,   log: logs/pyroki_8116.log
#   Molmo      → port 8122, GPU 5, log: outputs/molmo_server/molmo.log (separate venv)

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs

echo "Starting vision servers..."

# SAM3
if ss -tln | grep -q ':8114 '; then
    echo "[SAM3] Already running on 8114, skipping"
else
    echo "[SAM3] Starting on GPU 4, port 8114..."
    CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python -c \
        "from rats.serving.launch_sam3_server import main; main(device='cuda', port=8114)" \
        > logs/sam3_8114.log 2>&1 &
    echo "[SAM3] PID=$!, log=logs/sam3_8114.log"
fi

# GraspNet
if ss -tln | grep -q ':8115 '; then
    echo "[GraspNet] Already running on 8115, skipping"
else
    echo "[GraspNet] Starting on GPU 4, port 8115..."
    CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python -c \
        "from rats.serving.launch_contact_graspnet_server import main; main(port=8115)" \
        > logs/graspnet_8115.log 2>&1 &
    echo "[GraspNet] PID=$!, log=logs/graspnet_8115.log"
fi

# pyroki (CPU only)
if ss -tln | grep -q ':8116 '; then
    echo "[pyroki] Already running on 8116, skipping"
else
    echo "[pyroki] Starting on CPU, port 8116..."
    nohup .venv/bin/python -c \
        "from rats.serving.launch_pyroki_server import main; main(port=8116, robot='panda_description', target_link='panda_hand')" \
        > logs/pyroki_8116.log 2>&1 &
    echo "[pyroki] PID=$!, log=logs/pyroki_8116.log"
fi

# Molmo (separate venv)
if ss -tln | grep -q ':8122 '; then
    echo "[Molmo] Already running on 8122, skipping"
else
    echo "[Molmo] Starting on GPU 5, port 8122..."
    mkdir -p outputs/molmo_server
    CUDA_VISIBLE_DEVICES=5 nohup .venv/bin/vllm serve allenai/Molmo2-8B \
        --host 127.0.0.1 --port 8122 \
        --trust-remote-code \
        --gpu-memory-utilization 0.6 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 8192 \
        > outputs/molmo_server/molmo.log 2>&1 &
    echo "[Molmo] PID=$!, log=outputs/molmo_server/molmo.log"
fi

echo ""
echo "Waiting for servers to start..."
sleep 5

echo ""
echo "=== Server Status ==="
for port in 8114 8115 8116 8122; do
    if ss -tln | grep -q ":${port} "; then
        echo "  ✓ Port $port is listening"
    else
        echo "  ✗ Port $port NOT listening (check logs/)"
    fi
done

echo ""
echo "Log files:"
echo "  tail -f logs/sam3_8114.log"
echo "  tail -f logs/graspnet_8115.log"
echo "  tail -f logs/pyroki_8116.log"
echo "  tail -f outputs/molmo_server/molmo.log"
