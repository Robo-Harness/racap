"""Environment verifiers — env-agnostic name, two impls.

After an env is created or rebound, we want a uniform way to ask "is this
env actually suitable for the proposed task?". Two impls live here:

- ``MolmoEnvironmentVerifier``: post-rebind Molmo-grounding probe used by
  the molmospaces path. Captures an agentview frame, asks a local Molmo
  detector to point at the task target, and reports suitability.
  Implementation lives in ``molmospaces_environment_verifier.py``; we
  re-export it here under the prefix-free name so call sites and the
  agent-flow doc can refer to a single concept. The old class name
  ``MolmoSpacesEnvironmentVerifier`` is kept as an alias for backwards
  compatibility with existing tests/imports.

- ``LiberoEnvironmentVerifier``: thin adapter over
  ``EnvironmentCreator._verify_libero_scene_runtime`` that wraps libero's
  4-check sim probe (render OK, declared instances exist, no severe init
  penetrations, goal predicates evaluate without raising). Logic stays in
  EnvironmentCreator (it owns the BDDL parsing + MuJoCo helpers it needs);
  this adapter just gives libero the same shape as the molmo verifier.

Unified return shape (both ``verify(...)`` methods produce):

    {
        "suitable": bool,    # canonical "is the env OK"
        "ok": bool,          # alias for suitable
        "reason": str,       # one-line summary
        "errors": list[str],
        "warnings": list[str],
        "details": dict,
        ... impl-specific extras (provider/queries/inventory_eligibility/...)
    }

The shared ``EnvironmentVerifier`` base class is a marker — both impls
override ``verify()``. We don't try to enforce a single call signature
(the two impls legitimately consume different inputs: bddl_text for
libero, scene_context+iteration for molmo) so each subclass keeps its
own kwargs.
"""

from __future__ import annotations

from typing import Any

# Re-export the molmo impl under the prefix-free name. The original
# class kept in molmospaces_environment_verifier.py is the implementation
# of record — this module just exposes a cleaner public name.
from rats.agents.molmospaces_environment_verifier import (
    MolmoSpacesEnvironmentVerifier as _MolmoImpl,
)


class EnvironmentVerifier:
    """Marker base class. Override ``verify``."""

    def verify(self, env: Any, task_proposal: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


# Public prefix-free name for the molmo impl.
class MolmoEnvironmentVerifier(_MolmoImpl, EnvironmentVerifier):
    """Molmo-grounding env verifier (post-rebind, molmospaces path).

    Inherits the full implementation from ``MolmoSpacesEnvironmentVerifier``
    in ``agents/molmospaces_environment_verifier.py``. ``verify`` is wrapped
    only to add the shared unified-shape aliases (``ok``, ``errors``,
    ``warnings``, ``details``) without disturbing any of the molmo-specific
    fields the existing tests pin on (``reason``, ``inventory_eligibility``,
    ``queries``, ``points``, ...).
    """

    def verify(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raw = super().verify(*args, **kwargs)
        if not isinstance(raw, dict):
            return raw
        # Project into the shared shape without rewriting molmo's keys.
        suitable = bool(raw.get("suitable", True))
        reason = str(raw.get("reason") or "")
        raw.setdefault("ok", suitable)
        if "errors" not in raw:
            raw["errors"] = [] if suitable else ([reason] if reason else ["unsuitable"])
        raw.setdefault("warnings", [])
        raw.setdefault("details", {})
        return raw


# Backwards-compat alias so legacy imports keep working.
MolmoSpacesEnvironmentVerifier = MolmoEnvironmentVerifier


def _libero_result_to_unified(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the libero scene-runtime result into the shared shape."""
    errors = list(raw.get("errors") or [])
    warnings = list(raw.get("warnings") or [])
    ok = bool(raw.get("ok"))
    if errors:
        reason = errors[0]
    elif warnings:
        reason = "warnings: " + warnings[0]
    else:
        reason = "ok"
    return {
        "suitable": ok,
        "ok": ok,
        "reason": reason,
        "errors": errors,
        "warnings": warnings,
        "details": raw.get("details") or {},
    }


class LiberoEnvironmentVerifier(EnvironmentVerifier):
    """Adapter wrapping EnvironmentCreator's libero scene-runtime checks.

    The actual check logic stays on the creator because it depends on
    BDDL-parsing helpers (``_parse_bddl_scene_spec``) and MuJoCo body /
    contact helpers (``_find_body_id``, ``_contact_names``, ...) that
    other env_creator paths also use. This adapter exists so callers see
    the same surface as the molmo verifier (constructor + ``verify``
    returning a unified dict).
    """

    def __init__(self, env_creator: Any) -> None:
        # Hold a reference to the EnvironmentCreator so we can delegate
        # to its inline check method. Avoids duplicating ~250 lines of
        # MuJoCo helpers that the creator already has.
        self._env_creator = env_creator

    def verify(
        self,
        env: Any,
        task_proposal: dict[str, Any],
        *,
        bddl_text: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Probe the realized libero env. Requires the BDDL it was built from."""
        raw = self._env_creator._verify_libero_scene_runtime(
            env, bddl_text, task_proposal,
        )
        return _libero_result_to_unified(raw)
