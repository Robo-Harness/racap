"""Enforce one physical reset per registered generated-code episode.

The pinned upstream CaP-X / RATS checkout retries a timed-out trial by calling
``env.reset()`` again inside the same registered trial.  That is useful for
interactive demos but invalid for the controlled protocol: a timeout is an
episode outcome, not permission to sample the initial state again.  This
process-local shim changes only the upstream timeout retry count.  It does not
alter model calls, generated programs, observations, actions, or scoring.

``sitecustomize`` is used so multiprocessing ``spawn`` workers receive the
same rule without modifying the hash-pinned upstream repository.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any


TARGET_MODULES = frozenset({"capx.envs.runner", "rats.envs.runner"})
RACAP_SINGLE_RESET_HOOK_INSTALLED = False


class _SingleResetLoader(importlib.abc.Loader):
    """Delegate normal loading, then change only the timeout retry constant."""

    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        creator = getattr(self._wrapped, "create_module", None)
        return None if creator is None else creator(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        module.MAX_TRIAL_RETRIES = 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _SingleResetFinder(importlib.abc.MetaPathFinder):
    """Intercept only the two upstream trial-runner modules."""

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in TARGET_MODULES:
            return None
        # PathFinder bypasses sys.meta_path, avoiding recursive invocation of
        # this finder while retaining the upstream module's exact origin.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _SingleResetLoader(spec.loader)
        return spec


if os.environ.get("RACAP_SINGLE_RESET_PROTOCOL") == "1":
    for _name in TARGET_MODULES:
        _loaded = sys.modules.get(_name)
        if _loaded is not None:
            _loaded.MAX_TRIAL_RETRIES = 1
    if not any(isinstance(finder, _SingleResetFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _SingleResetFinder())
    RACAP_SINGLE_RESET_HOOK_INSTALLED = True
