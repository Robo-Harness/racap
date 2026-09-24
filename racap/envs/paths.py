"""Portable discovery of the LIBERO benchmark data directories."""

from __future__ import annotations

import os
from pathlib import Path


def libero_root() -> Path:
    """Return the directory containing ``bddl_files`` and ``init_files``.

    An explicit ``RACAP_LIBERO_ROOT`` wins.  Otherwise discover the installed
    LIBERO package, which also works for RATs' editable LIBERO-PRO checkout.
    Discovery is deliberately lazy so importing RACaP's pure geometry modules
    does not require the simulator to be installed.
    """
    configured = os.environ.get("RACAP_LIBERO_ROOT") or os.environ.get("LIBERO_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        try:
            import libero
        except ImportError as exc:
            raise RuntimeError(
                "LIBERO is not installed. Run scripts/bootstrap.sh or set "
                "RACAP_LIBERO_ROOT to the LIBERO package data directory."
            ) from exc
        root = Path(libero.__file__).resolve().parent

    if not (root / "bddl_files").is_dir() or not (root / "init_files").is_dir():
        raise RuntimeError(
            f"{root} is not a LIBERO data root: expected bddl_files/ and init_files/"
        )
    return root


def bddl_root() -> Path:
    return libero_root() / "bddl_files"


def init_root() -> Path:
    return libero_root() / "init_files"


__all__ = ["libero_root", "bddl_root", "init_root"]
