"""
NodeRegistry — Central registry of available node types.

Nodes register themselves via the ``@NodeRegistry.register`` decorator.
LLM agents discover available nodes via ``NodeRegistry.list_nodes()``.

Example::

    @NodeRegistry.register("object_detector", category="detection")
    class ObjectDetector(ModelNode):
        ...

    # Later, construct from config:
    node = NodeRegistry.create(node_config, model_cache=cache)
"""
from __future__ import annotations

import logging
from typing import Any, Type

from vision_tools.core.node import Node

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Central registry of available node types.

    All registrations are stored at the class level so they persist
    across the application lifetime.
    """

    _registry: dict[str, Type[Node]] = {}
    _categories: dict[str, str] = {}  # node_type → category

    # --- Registration ---

    @classmethod
    def register(
        cls,
        name: str,
        category: str = "general",
    ):
        """Decorator to register a node type.

        Args:
            name: Unique identifier for this node type (e.g. "object_detector").
            category: Grouping category (e.g. "detection", "embedding").

        Returns:
            Decorator that registers the class and returns it unchanged.

        Example::

            @NodeRegistry.register("object_detector", category="detection")
            class ObjectDetector(ModelNode):
                ...
        """
        def wrapper(node_cls: Type[Node]) -> Type[Node]:
            if name in cls._registry:
                logger.warning(
                    f"NodeRegistry: overwriting existing registration "
                    f"for '{name}' ({cls._registry[name].__name__} → "
                    f"{node_cls.__name__})"
                )
            cls._registry[name] = node_cls
            cls._categories[name] = category
            logger.debug(f"NodeRegistry: registered '{name}' ({category})")
            return node_cls

        return wrapper

    # --- Discovery ---

    @classmethod
    def list_nodes(cls) -> list[dict[str, Any]]:
        """Return metadata for all registered nodes.

        Designed for LLM agent discovery — returns enough information
        to construct a valid ``NodeConfig``.
        """
        return [
            {
                "type": name,
                "category": cls._categories.get(name, "general"),
                **node_cls.get_metadata(),
            }
            for name, node_cls in cls._registry.items()
        ]

    @classmethod
    def get(cls, name: str) -> Type[Node]:
        """Look up a registered node class by name.

        Raises:
            KeyError: If the node type is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown node type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, node_config, **services: Any) -> Node:
        """Instantiate a node from a NodeConfig, injecting runtime services.

        Args:
            node_config: A ``NodeConfig`` instance with node_id, node_type, config.
            **services: Runtime services to inject (model_cache, model_resolver, etc.)

        Returns:
            Instantiated ``Node``.
        """
        node_cls = cls.get(node_config.node_type)
        return node_cls(
            node_id=node_config.node_id,
            config=node_config.config,
            **services,
        )

    # --- Utilities ---

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Mainly for testing."""
        cls._registry.clear()
        cls._categories.clear()

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return sorted list of registered node type names."""
        return sorted(cls._registry.keys())
