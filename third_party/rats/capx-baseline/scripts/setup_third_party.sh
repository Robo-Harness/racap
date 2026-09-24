#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/capx/third_party"

clone_if_missing() {
  local path="$1"
  local url="$2"
  local branch="${3:-}"

  if [[ -e "$path" ]]; then
    echo "exists: $path"
    return
  fi

  mkdir -p "$(dirname "$path")"
  if [[ -n "$branch" ]]; then
    git clone --depth 1 --branch "$branch" "$url" "$path"
  else
    git clone --depth 1 "$url" "$path"
  fi
}

clone_if_missing "$THIRD_PARTY_DIR/LIBERO-PRO" "https://github.com/uynitsuj/LIBERO-PRO.git"
clone_if_missing "$THIRD_PARTY_DIR/verl" "https://github.com/Max-Fu/verl.git"
clone_if_missing "$THIRD_PARTY_DIR/robosuite" "https://github.com/uynitsuj/robosuite"
clone_if_missing "$THIRD_PARTY_DIR/contact_graspnet_pytorch" "https://github.com/uynitsuj/contact_graspnet_pytorch"
clone_if_missing "$THIRD_PARTY_DIR/libero_dependencies/robosuite" "https://github.com/Max-Fu/robosuite" "maxf/egl_context"
clone_if_missing "$THIRD_PARTY_DIR/sam3" "https://github.com/Max-Fu/sam3.git" "main"
clone_if_missing "$THIRD_PARTY_DIR/curobo" "https://github.com/NVlabs/curobo.git"
clone_if_missing "$THIRD_PARTY_DIR/b1k" "https://github.com/qingh097/b1k.git"

echo "Third-party checkout complete."
