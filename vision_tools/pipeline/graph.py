"""
DAG — Directed Acyclic Graph for pipeline topology.

Builds an adjacency list from ``NodeConfig.depends_on``,
provides topological sort (into execution layers), and cycle detection.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from vision_tools.core.config import PipelineConfig

logger = logging.getLogger(__name__)


class CycleError(Exception):
    """Raised when a cycle is detected in the pipeline graph."""
    pass


class DAG:
    """Directed acyclic graph for pipeline node ordering.

    Nodes in the same layer have no mutual dependencies and can
    execute in parallel.

    Args:
        config: PipelineConfig with nodes and their depends_on edges.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.nodes = {n.node_id: n for n in config.nodes}
        self._adjacency: dict[str, list[str]] = {}
        self._in_degree: dict[str, int] = {}
        self._layers: list[list[str]] | None = None
        self._build()

    def _build(self) -> None:
        """Build adjacency list and in-degree map."""
        for nid in self.nodes:
            self._adjacency.setdefault(nid, [])
            self._in_degree.setdefault(nid, 0)

        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(
                        f"Node '{nid}' depends on '{dep}' which is not in the pipeline."
                    )
                self._adjacency[dep].append(nid)
                self._in_degree[nid] += 1

    def topological_sort(self) -> list[list[str]]:
        """Sort nodes into execution layers using Kahn's algorithm.

        Returns:
            List of layers, where each layer is a list of node IDs
            that can execute in parallel.

        Raises:
            CycleError: If the graph contains a cycle.
        """
        if self._layers is not None:
            return self._layers

        in_degree = dict(self._in_degree)
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        layers: list[list[str]] = []
        processed = 0

        while queue:
            layer = list(queue)
            queue.clear()
            layers.append(layer)
            processed += len(layer)

            for nid in layer:
                for successor in self._adjacency[nid]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        if processed != len(self.nodes):
            raise CycleError(
                f"Cycle detected in pipeline graph. "
                f"Processed {processed}/{len(self.nodes)} nodes."
            )

        self._layers = layers
        logger.debug(f"DAG layers: {layers}")
        return layers

    def execution_order(self) -> list[str]:
        """Return node IDs in flat execution order (layer by layer)."""
        return [nid for layer in self.topological_sort() for nid in layer]

    def dependencies(self, node_id: str) -> list[str]:
        """Return direct dependencies of a node."""
        node = self.nodes.get(node_id)
        return list(node.depends_on) if node else []

    def dependents(self, node_id: str) -> list[str]:
        """Return nodes that depend on the given node."""
        return list(self._adjacency.get(node_id, []))

    def __repr__(self) -> str:
        return f"DAG(nodes={list(self.nodes.keys())}, layers={self.topological_sort()})"
