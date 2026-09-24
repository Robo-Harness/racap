#!/usr/bin/env bash
# Evaluate one frozen RACaP phase and record videos plus readable traces.
# Usage: bash scripts/evaluate.sh phase2 [tag] [extra eval_full_agent.py arguments]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
racap_root="$(cd "$script_dir/.." && pwd)"
source "$racap_root/configs/env.sh"

phase="${1:-phase2}"
shift || true
case "$phase" in
  phase1|1) solution_root="$racap_root/policies/phase1"; phase="phase1" ;;
  phase2|2) solution_root="$racap_root/policies/phase2"; phase="phase2" ;;
  *) echo "usage: $0 {phase1|phase2} [tag] [extra arguments]" >&2; exit 2 ;;
esac
tag="${1:-racap_${phase}_$(date +%Y%m%d_%H%M%S)}"
shift || true
cd "$racap_root"
exec "$VIRTUAL_ENV/bin/python" "$racap_root/scripts/eval_full_agent.py" \
  --suite libero_90 --workers "${RACAP_WORKERS:-5}" --turns 45 \
  --max-pickplace-calls 8 --max-push-calls 3 --max-insert-calls 4 \
  --max-state-calls 5 --max-stack-calls 2 --max-steps 8000 \
  --record-rollouts --solution-root "$solution_root" --tag "$tag" "$@"
