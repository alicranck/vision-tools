"""
Pipeline — High-level facade for building and running vision pipelines.

Usage::

    from vision_tools.pipeline import Pipeline
    from vision_tools.core.config import PipelineConfig

    config = PipelineConfig(
        name="my_pipeline",
        nodes=[
            NodeConfig(node_id="det", node_type="object_detector",
                       config={"model": "yolo", "vocabulary": ["person"]}),
            NodeConfig(node_id="emb", node_type="embedder",
                       config={"model": "siglip2"}, depends_on=["det"]),
        ],
    )
    pipeline = Pipeline(config)
    pipeline.load()
    results = pipeline.run(frame)
    pipeline.shutdown()
"""
from __future__ import annotations

import logging
from typing import Any

from vision_tools.core.config import PipelineConfig, ExecutionConfig
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.validator import SchemaValidator
from vision_tools.pipeline.executor import PipelineExecutor

logger = logging.getLogger(__name__)


class Pipeline:
    """High-level pipeline facade.

    Wraps DAG construction, schema validation, node lifecycle,
    and execution into a clean public API.

    Args:
        config: PipelineConfig with node definitions and execution settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._execution = config.execution or ExecutionConfig()

        # Build DAG
        self.dag = DAG(config)

        # Create node instances from registry
        self._nodes: dict[str, Node] = {}
        for node_config in config.nodes:
            self._nodes[node_config.node_id] = NodeRegistry.create(node_config)

        # Validate I/O schemas
        self._warnings = SchemaValidator.validate(self.dag, self._nodes)

        # Create executor
        self._executor = PipelineExecutor(
            dag=self.dag,
            nodes=self._nodes,
            max_workers=self._execution.max_workers,
        )

    def load(self) -> None:
        """Load all nodes in execution order."""
        for node_id in self.dag.execution_order():
            node = self._nodes[node_id]
            if hasattr(node, "load"):
                node.load()
                logger.info(f"Pipeline: loaded '{node_id}'")

        if self._execution.warmup_rounds > 0:
            self.warmup(self._execution.warmup_rounds)

        if self._execution.verify_on_init:
            self.verify()

    def warmup(self, rounds: int = 4) -> None:
        """Warm up all nodes."""
        for node_id, node in self._nodes.items():
            if hasattr(node, "warmup"):
                node.warmup(rounds)

    def verify(self) -> bool:
        """Verify all nodes produce valid output."""
        all_ok = True
        for node_id, node in self._nodes.items():
            if hasattr(node, "verify"):
                if not node.verify():
                    logger.error(f"Pipeline: verification failed for '{node_id}'")
                    all_ok = False
        return all_ok

    def run(self, frame: Any, context: NodeContext | None = None) -> dict[str, Any]:
        """Execute pipeline on a single frame.

        Args:
            frame: Input data (numpy array).
            context: Optional runtime context.

        Returns:
            Dict mapping node_id → output data.
        """
        return self._executor.run(frame, context)

    def run_batch(
        self,
        frames: list[Any],
        contexts: list[NodeContext] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute pipeline on a batch of frames.

        Args:
            frames: List of input frames.
            contexts: Optional list of contexts.

        Returns:
            List of result dicts.
        """
        return self._executor.run_batch(frames, contexts)

    def shutdown(self) -> None:
        """Unload all nodes and clean up resources."""
        for node_id, node in self._nodes.items():
            if hasattr(node, "unload"):
                node.unload()
        logger.info("Pipeline: shutdown complete.")

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            The Node instance.

        Raises:
            KeyError: If node not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in pipeline.")
        return self._nodes[node_id]

    @property
    def nodes(self) -> dict[str, Node]:
        """All nodes in the pipeline."""
        return dict(self._nodes)

    @property
    def warnings(self) -> list[str]:
        """Schema validation warnings from construction."""
        return list(self._warnings)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.config.name}', "
            f"nodes={list(self._nodes.keys())}, "
            f"layers={self.dag.topological_sort()})"
        )
