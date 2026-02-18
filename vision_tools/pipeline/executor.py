"""
PipelineExecutor — Layer-by-layer execution with optional parallelism.

Executes nodes in topological order (layer by layer). Nodes within
the same layer have no mutual dependencies and can run concurrently.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from vision_tools.core.node import Node, NodeContext
from vision_tools.pipeline.graph import DAG

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes pipeline nodes layer-by-layer.

    Args:
        dag: The pipeline DAG (provides execution layers).
        nodes: Mapping of node_id → Node instance.
        max_workers: Max parallel workers within a layer (0 = sequential).
    """

    def __init__(
        self,
        dag: DAG,
        nodes: dict[str, Node],
        max_workers: int = 0,
    ) -> None:
        self.dag = dag
        self.nodes = nodes
        self.max_workers = max_workers

    def run(self, frame: Any, context: NodeContext | None = None) -> dict[str, Any]:
        """Execute the pipeline on a single frame.

        Args:
            frame: Input data (typically a numpy array).
            context: Optional runtime context.

        Returns:
            Dict mapping node_id → output data for each node.
        """
        context = context or NodeContext()
        results: dict[str, Any] = {}

        for layer in self.dag.topological_sort():
            if self.max_workers > 0 and len(layer) > 1:
                # Parallel execution within layer
                self._run_layer_parallel(layer, frame, context, results)
            else:
                # Sequential execution
                self._run_layer_sequential(layer, frame, context, results)

        return results

    def run_batch(
        self,
        frames: list[Any],
        contexts: list[NodeContext] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the pipeline on a batch of frames.

        Args:
            frames: List of input frames.
            contexts: Optional list of contexts (one per frame).

        Returns:
            List of result dicts, one per frame.
        """
        if contexts is None:
            contexts = [NodeContext(frame_idx=i) for i in range(len(frames))]

        return [self.run(frame, ctx) for frame, ctx in zip(frames, contexts)]

    def _run_layer_sequential(
        self,
        layer: list[str],
        frame: Any,
        context: NodeContext,
        results: dict[str, Any],
    ) -> None:
        """Execute nodes in a layer sequentially."""
        for node_id in layer:
            self._run_single_node(node_id, frame, context, results)

    def _run_layer_parallel(
        self,
        layer: list[str],
        frame: Any,
        context: NodeContext,
        results: dict[str, Any],
    ) -> None:
        """Execute nodes in a layer concurrently."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_single_node, nid, frame, context, results): nid
                for nid in layer
            }
            for future in as_completed(futures):
                nid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Node '{nid}' failed: {e}")
                    raise

    def _run_single_node(
        self,
        node_id: str,
        frame: Any,
        context: NodeContext,
        results: dict[str, Any],
    ) -> None:
        """Execute a single node, injecting upstream results into context."""
        node = self.nodes[node_id]

        # Inject upstream results into context
        deps = self.dag.dependencies(node_id)
        if deps:
            upstream = {dep: results[dep] for dep in deps if dep in results}
            context = NodeContext(
                frame_idx=context.frame_idx,
                timestamp=context.timestamp,
                frame_shape=context.frame_shape,
                upstream_results=upstream,
            )

        output = node.process(frame, context)
        results[node_id] = output
