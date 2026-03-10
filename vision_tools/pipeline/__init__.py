from __future__ import annotations

from typing import Any

from vision_tools.core.config import ExecutionConfig, PipelineConfig
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.pipeline.executor import PipelineExecutor
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.validator import SchemaValidator

try:
    import vision_tools.nodes  # noqa: F401
except ImportError:
    pass


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._execution = config.execution or ExecutionConfig()
        self.dag = DAG(config)
        self._nodes = {
            node_config.node_id: NodeRegistry.create(node_config)
            for node_config in config.nodes
        }
        self._warnings = SchemaValidator.validate(self.dag, self._nodes)
        self._executor = PipelineExecutor(
            dag=self.dag,
            nodes=self._nodes,
        )

    def load(self) -> None:
        for node_id in self.dag.execution_order():
            node = self._nodes[node_id]
            if hasattr(node, "load"):
                node.load()

        if self._execution.warmup_rounds > 0:
            self.warmup(self._execution.warmup_rounds)

        if self._execution.verify_on_init and not self.verify():
            failed = [
                node_id
                for node_id, node in self._nodes.items()
                if hasattr(node, "state") and node.state != NodeState.READY
            ]
            raise RuntimeError(f"Pipeline verification failed for nodes: {failed}")

    def warmup(self, rounds: int = 1) -> None:
        for node in self._nodes.values():
            if hasattr(node, "warmup"):
                node.warmup(rounds)

    def verify(self) -> bool:
        all_ok = True
        for node in self._nodes.values():
            if hasattr(node, "verify") and not node.verify():
                all_ok = False
        return all_ok

    def run(self, frame: Any, context: NodeContext | None = None) -> dict[str, Any]:
        return self._executor.run(frame, context)

    def run_batch(
        self,
        frames: list[Any],
        contexts: list[NodeContext] | None = None,
    ) -> list[dict[str, Any]]:
        return self._executor.run_batch(frames, contexts)

    def shutdown(self) -> None:
        for node in self._nodes.values():
            if hasattr(node, "unload"):
                node.unload()

    def get_node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in pipeline.")
        return self._nodes[node_id]

    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self._nodes)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.config.name}', "
            f"nodes={list(self._nodes.keys())}, "
            f"layers={self.dag.topological_sort()})"
        )
