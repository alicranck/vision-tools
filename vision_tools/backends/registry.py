"""
BackendRegistry — Central registry of model backends.

Task nodes resolve their backend at construction time using this registry.

Example::

    @BackendRegistry.register(task="detection", model="yolo_detector")
    class YoloDetectorBackend:
        ...

    # Later, in a task node:
    backend = BackendRegistry.get(task="detection", model="yolo_detector")
"""
from __future__ import annotations

import logging
from typing import Any, Type

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry mapping (task, model) pairs to backend implementations.

    Backends register themselves via the ``@register`` decorator.
    Task nodes look up their backend via ``get(task, model)``.
    """

    _registry: dict[tuple[str, str], Any] = {}

    @classmethod
    def register(cls, task: str, model: str):
        """Decorator to register a backend for a (task, model) pair.

        Args:
            task: Task category (e.g. "detection", "embedding").
            model: Model family (e.g. "yolo_detector", "siglip2", "clip").

        Returns:
            Decorator that registers the class and returns it unchanged.

        Example::

            @BackendRegistry.register(task="detection", model="yolo_detector")
            class YoloDetectorBackend:
                ...
        """
        def wrapper(backend_cls):
            cls.register_class(task, model, backend_cls)
            return backend_cls

        return wrapper

    @classmethod
    def register_class(cls, task: str, model: str, backend_cls) -> None:
        key = (task, model)
        if key in cls._registry:
            logger.warning(
                f"BackendRegistry: overwriting {key} "
                f"({cls._registry[key].__name__} → {backend_cls.__name__})"
            )
        cls._registry[key] = backend_cls
        logger.debug(f"BackendRegistry: registered {key}")

    @classmethod
    def get(cls, task: str, model: str) -> Any:
        """Look up and instantiate a backend for the given (task, model) pair.

        Args:
            task: Task category (e.g. "detection").
            model: Model family (e.g. "yolo_detector").

        Returns:
            An instance of the registered backend class.

        Raises:
            KeyError: If no backend is registered for the pair.
        """
        key = (task, model)
        if key not in cls._registry:
            # Lazy bootstrap: supports test suites or callers that clear registries.
            try:
                import vision_tools.nodes as nodes_pkg  # noqa: F401
                if hasattr(nodes_pkg, "register_all"):
                    nodes_pkg.register_all()
            except Exception:
                pass

        if key not in cls._registry:
            available = sorted(
                f"{t}:{m}" for t, m in cls._registry.keys()
            )
            raise KeyError(
                f"No backend registered for task={task!r}, model={model!r}. "
                f"Available: {available}"
            )
        backend_cls = cls._registry[key]
        return backend_cls()

    @classmethod
    def list_backends(cls) -> list[dict[str, str]]:
        """Return metadata for all registered backends.

        Returns:
            List of dicts with 'task', 'model', and 'class_name' keys.
        """
        return [
            {
                "task": task,
                "model": model,
                "class_name": backend_cls.__name__,
            }
            for (task, model), backend_cls in cls._registry.items()
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Mainly for testing."""
        cls._registry.clear()

    @classmethod
    def has(cls, task: str, model: str) -> bool:
        """Check if a backend is registered for the given pair."""
        return (task, model) in cls._registry
