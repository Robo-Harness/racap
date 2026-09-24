"""Backends that satisfy :class:`racap.contracts.PrimitiveRuntime`.

``libero`` is imported lazily because it pulls in MuJoCo, robosuite and the
RATs package, which is far too heavy for callers that only want the mock.
"""

__all__ = ["LiberoPrimitiveRuntime", "LiberoOracle"]


def __getattr__(name: str):
    if name in __all__:
        from racap.backends import libero

        return getattr(libero, name)
    raise AttributeError(name)
