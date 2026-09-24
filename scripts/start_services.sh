#!/usr/bin/env bash
# Start the three local perception/motion services required by RACaP.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
racap_root="$(cd "$script_dir/.." && pwd)"
source "$racap_root/configs/env.sh"

python_bin="$VIRTUAL_ENV/bin/python"
log_dir="$racap_root/.runtime/logs"
gpu="${RACAP_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
mkdir -p "$log_dir"

port_open() { python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

if ! port_open 8114; then
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -m \
    rats.serving.launch_sam3_server --device cuda --host 127.0.0.1 --port 8114 \
    >"$log_dir/sam3.log" 2>&1 &
fi
if ! port_open 8115; then
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -m \
    rats.serving.launch_contact_graspnet_server --host 127.0.0.1 --port 8115 \
    >"$log_dir/graspnet.log" 2>&1 &
fi
if ! port_open 8116; then
  nohup "$python_bin" -m rats.serving.launch_pyroki_server \
    --host 127.0.0.1 --port 8116 --robot panda_description --target-link panda_hand \
    >"$log_dir/pyroki.log" 2>&1 &
fi

echo "Services launched; logs are under $log_dir."
echo "Model calls use the credentials and route in configs/local.env."
