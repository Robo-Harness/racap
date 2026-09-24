"""CaP-X Interactive Web UI Backend.

This module provides a FastAPI-based web server with WebSocket support
for real-time interactive robot code execution demos.
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazily import the FastAPI app factory.

    Importing ``rats.web.models`` should not require server-only dependencies
    such as ``tyro``.  Keep the public ``rats.web.create_app`` symbol while
    avoiding that eager import.
    """
    from rats.web.server import create_app as _create_app

    return _create_app(*args, **kwargs)
