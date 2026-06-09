"""Strategy registry – discover and instantiate strategies by name.

Usage::

    from firm.strategies.registry import register, get

    @register("momentum")
    class MomentumStrategy(BaseStrategy): ...

    cls = get("momentum")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firm.strategies.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(name: str):
    """Class decorator that registers a strategy under *name*."""

    def wrapper(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get(name: str) -> type[BaseStrategy]:
    """Look up a registered strategy class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY)
