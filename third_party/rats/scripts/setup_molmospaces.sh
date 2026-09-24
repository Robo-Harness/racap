#!/usr/bin/env bash
# Initialize the molmospaces submodule and apply local patches.
#
# Runs `git submodule update --init --recursive` for rats/third_party/molmospaces
# and then applies every patch under scripts/patches/molmospaces/ in lexicographic order.
# The patch step is idempotent: if a patch is already applied the script skips
# it instead of failing, so the script is safe to re-run after a fresh submodule
# update or after a `git submodule update --remote`.
#
# Usage:
#   bash scripts/setup_molmospaces.sh
#
# When upstream (allenai/molmospaces) accepts a patch, bump the submodule commit
# and delete the corresponding file from scripts/patches/molmospaces/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/rats/third_party/molmospaces"
PATCH_DIR="$REPO_ROOT/scripts/patches/molmospaces"

echo "[setup-molmospaces] initialising submodule..."
git -C "$REPO_ROOT" submodule update --init --recursive rats/third_party/molmospaces

if [[ ! -d "$PATCH_DIR" ]]; then
    echo "[setup-molmospaces] no patches/ dir at $PATCH_DIR — nothing to apply."
    exit 0
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
    echo "[setup-molmospaces] no *.patch files in $PATCH_DIR — nothing to apply."
    exit 0
fi

for patch in "${patches[@]}"; do
    name="$(basename "$patch")"
    if git -C "$SUBMODULE_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "[setup-molmospaces] $name already applied — skipping."
        continue
    fi
    if ! git -C "$SUBMODULE_DIR" apply --check "$patch" >/dev/null 2>&1; then
        echo "[setup-molmospaces] ERROR: $name does not apply cleanly."
        echo "  The submodule may have diverged from the base this patch was"
        echo "  generated against. Regenerate the patch or drop it if upstream"
        echo "  has already merged the change."
        git -C "$SUBMODULE_DIR" apply --check "$patch" || true
        exit 1
    fi
    git -C "$SUBMODULE_DIR" apply "$patch"
    echo "[setup-molmospaces] applied $name."
done

echo "[setup-molmospaces] done."
