#!/usr/bin/env bash
# Check code, simulator data, strict evaluation flags, and runtime services.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
racap_root="$(cd "$script_dir/.." && pwd)"
source "$racap_root/configs/env.sh"
fail=0

check_imports() {
  "$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib
for name in ("racap", "evolution", "rats", "libero", "robosuite", "mujoco", "numpy", "PIL"):
    module = importlib.import_module(name)
    print(f"ok import {name} {getattr(module, '__version__', '')}".rstrip())
from racap.envs.paths import libero_root
print(f"ok LIBERO data {libero_root()}")
PY
}

if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
  echo "FAIL missing environment: $VIRTUAL_ENV (run scripts/bootstrap.sh)"
  fail=1
elif ! check_imports; then
  fail=1
fi

if [[ "${RATS_VERIFIER_STRICT_BENCHMARK:-}" != 1 ]] || \
   [[ "${RATS_VERIFY_STEP_MODE:-}" != strict ]]; then
  echo "FAIL strict native-predicate evaluation flags are not enabled"
  fail=1
else
  echo "ok strict native-predicate evaluation enabled"
fi

for endpoint in \
  "SAM3:${RACAP_SAM3_URL}/openapi.json" \
  "GraspNet:${RACAP_GRASPNET_URL}/openapi.json" \
  "PyRoKi:${RACAP_PYROKI_URL}/openapi.json"; do
  name="${endpoint%%:*}"
  url="${endpoint#*:}"
  code="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then
    echo "ok $name $url"
  else
    echo "FAIL $name unavailable at $url (HTTP ${code:-none})"
    fail=1
  fi
done

for model in "$RACAP_MODEL" "$RACAP_GROUNDER_MODEL" "$RACAP_VERIFIER_MODEL"; do
  if [[ "$model" == openrouter/* ]]; then
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
      echo "FAIL OPENROUTER_API_KEY is required for the configured model route"
      fail=1
    fi
  elif [[ -z "${RACAP_VAPI_KEY:-}" || -z "${RACAP_VAPI_BASE:-}" ]]; then
    echo "FAIL configure both RACAP_VAPI_KEY and RACAP_VAPI_BASE in configs/local.env"
    fail=1
  fi
done

exit "$fail"
