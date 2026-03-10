from __future__ import annotations

import logging
from typing import Any, Type

from vision_tools.core.node import Node
from vision_tools.core.sources import list_sources

logger = logging.getLogger(__name__)


class NodeRegistry:
    _registry: dict[str, Type[Node]] = {}
    _categories: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, category: str = "general"):
        def wrapper(node_cls: Type[Node]) -> Type[Node]:
            cls.register_class(name, node_cls, category)
            return node_cls

        return wrapper

    @classmethod
    def register_class(cls, name: str, node_cls: Type[Node], category: str = "general") -> None:
        if name in cls._registry:
            logger.warning(
                "NodeRegistry: overwriting existing registration for '%s' (%s -> %s)",
                name,
                cls._registry[name].__name__,
                node_cls.__name__,
            )
        cls._registry[name] = node_cls
        cls._categories[name] = category

    @classmethod
    def list_nodes(cls) -> list[dict[str, Any]]:
        return [
            {
                "type": name,
                "category": cls._categories.get(name, "general"),
                **node_cls.get_metadata(),
            }
            for name, node_cls in cls._registry.items()
        ]

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        return list_sources()

    @classmethod
    def get(cls, name: str) -> Type[Node]:
        if name not in cls._registry:
            try:
                import vision_tools.nodes as nodes_pkg  # noqa: F401
                if hasattr(nodes_pkg, "register_all"):
                    nodes_pkg.register_all()
            except Exception:
                pass

        if name not in cls._registry:
            raise KeyError(
                f"Unknown node type '{name}'. Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, node_config, **services: Any) -> Node:
        node_cls = cls.get(node_config.node_type)
        return node_cls(
            node_id=node_config.node_id,
            config=node_config.config,
            **services,
        )

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._categories.clear()

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(cls._registry.keys())
