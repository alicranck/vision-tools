from __future__ import annotations

from collections import deque

from vision_tools.core.config import PipelineConfig
from vision_tools.pipeline.refs import parse_port_ref


class CycleError(Exception):
    pass


class DAG:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.nodes = {node.node_id: node for node in config.nodes}
        self._adjacency: dict[str, list[str]] = {}
        self._in_degree: dict[str, int] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._layers: list[list[str]] | None = None
        self._build()

    def _build(self) -> None:
        for node_id in self.nodes:
            self._adjacency[node_id] = []
            self._in_degree[node_id] = 0
            self._dependencies[node_id] = []

        for node_id, node in self.nodes.items():
            deps: list[str] = []
            for binding in node.inputs.values():
                producer_id, _ = parse_port_ref(binding)
                if producer_id == "input":
                    continue
                if producer_id not in self.nodes:
                    raise ValueError(
                        f"Node '{node_id}' references '{producer_id}' which is not in the pipeline."
                    )
                if producer_id not in deps:
                    deps.append(producer_id)

            self._dependencies[node_id] = deps
            for dep in deps:
                self._adjacency[dep].append(node_id)
                self._in_degree[node_id] += 1

    def topological_sort(self) -> list[list[str]]:
        if self._layers is not None:
            return self._layers

        in_degree = dict(self._in_degree)
        queue = deque(node_id for node_id, degree in in_degree.items() if degree == 0)
        layers: list[list[str]] = []
        processed = 0

        while queue:
            layer = list(queue)
            queue.clear()
            layers.append(layer)
            processed += len(layer)

            for node_id in layer:
                for dependent in self._adjacency[node_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if processed != len(self.nodes):
            raise CycleError(
                f"Cycle detected in pipeline graph. Processed {processed}/{len(self.nodes)} nodes."
            )

        self._layers = layers
        return layers

    def execution_order(self) -> list[str]:
        return [node_id for layer in self.topological_sort() for node_id in layer]

    def dependencies(self, node_id: str) -> list[str]:
        return list(self._dependencies.get(node_id, []))

    def dependents(self, node_id: str) -> list[str]:
        return list(self._adjacency.get(node_id, []))

    def terminal_nodes(self) -> list[str]:
        return [node_id for node_id, dependents in self._adjacency.items() if not dependents]

    def __repr__(self) -> str:
        return f"DAG(nodes={list(self.nodes.keys())}, layers={self.topological_sort()})"
