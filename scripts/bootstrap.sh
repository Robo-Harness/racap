#!/usr/bin/env bash
# Bootstrap the pinned simulator stack for the in-tree RATs compatibility layer.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
racap_root="$(cd "$script_dir/.." && pwd)"
rats_repo="$racap_root/third_party/rats"
vendor_root="$rats_repo/rats/third_party"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "Python 3.10-3.12 is required" >&2; exit 1; }

mkdir -p "$vendor_root/libero_dependencies" "$racap_root/.runtime/libero" \
  "$racap_root/.cache"

clone_pinned() {
  local url="$1"
  local commit="$2"
  local target="$3"
  if [[ ! -d "$target/.git" ]]; then
    git clone --filter=blob:none "$url" "$target"
  fi
  git -C "$target" fetch origin "$commit"
  git -C "$target" checkout --detach "$commit"
}

# RATs itself is vendored and versioned inside RACaP. Only large upstream
# simulator/perception repositories are downloaded here at pinned commits.
clone_pinned https://github.com/uynitsuj/LIBERO-PRO.git \
  47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a "$vendor_root/LIBERO-PRO"
clone_pinned https://github.com/Max-Fu/robosuite.git \
  a498b087d4bc5a3981e3d27030d09bc537a537f3 \
  "$vendor_root/libero_dependencies/robosuite"
clone_pinned https://github.com/uynitsuj/contact_graspnet_pytorch.git \
  8cd98632047e418dc938fc80add258a1ca15f9a9 \
  "$vendor_root/contact_graspnet_pytorch"
clone_pinned https://github.com/Max-Fu/sam3.git \
  6fe87d64a5beb9084923d7a9e002741178635b09 "$vendor_root/sam3"

if ! command -v uv >/dev/null 2>&1; then
  tools_env="$racap_root/.runtime/bootstrap-tools"
  python3 -m venv "$tools_env"
  "$tools_env/bin/python" -m pip install "uv>=0.8,<0.9"
  export PATH="$tools_env/bin:$PATH"
fi

UV_CACHE_DIR="$racap_root/.cache/uv" uv sync \
  --directory "$rats_repo" --frozen --extra libero --extra contactgraspnet
uv pip install --python "$rats_repo/.venv/bin/python" --no-deps -e "$racap_root"

libero_pkg="$rats_repo/rats/third_party/LIBERO-PRO/libero/libero"
cat > "$racap_root/.runtime/libero/config.yaml" <<EOF
benchmark_root: $libero_pkg
bddl_files: $libero_pkg/bddl_files
init_states: $libero_pkg/init_files
datasets: $rats_repo/rats/third_party/LIBERO-PRO/libero/datasets
assets: $libero_pkg/assets
EOF

echo "RACaP bootstrap complete."
echo "Next: cp configs/local.env.example configs/local.env"
echo "Then: source configs/env.sh && bash scripts/doctor.sh"
