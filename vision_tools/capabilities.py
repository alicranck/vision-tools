from __future__ import annotations

from typing import Any

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.registry import NodeRegistry
from vision_tools.runtime.model_catalog import ModelCatalog


def list_capabilities() -> dict[str, Any]:
    """Expose discoverable node/type/model capabilities for app-side orchestration."""
    try:
        import vision_tools.nodes as nodes_pkg  # noqa: F401
        if hasattr(nodes_pkg, "register_all"):
            nodes_pkg.register_all()
    except Exception:
        pass
    return {
        "nodes": NodeRegistry.list_nodes(),
        "sources": NodeRegistry.list_sources(),
        "backends": BackendRegistry.list_backends(),
        "model_catalog": ModelCatalog.list_options(),
    }
