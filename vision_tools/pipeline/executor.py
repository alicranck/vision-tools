from __future__ import annotations

from typing import Any

from vision_tools.core.graph_types import FrameInfo, Image
from vision_tools.core.node import Node, NodeContext
from vision_tools.core.type_refs import TypeRegistry
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.refs import parse_port_ref
from vision_tools.pipeline.validator import INPUT_PORTS


class PipelineExecutor:
    def __init__(self, dag: DAG, nodes: dict[str, Node], max_workers: int = 1) -> None:
        self.dag = dag
        self.nodes = nodes
        self.max_workers = max_workers

    def _build_context(self, frame: Any, context: NodeContext | None = None) -> NodeContext:
        context = context or NodeContext()

        frame_shape = context.frame_shape
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            inferred_shape = tuple(int(v) for v in frame.shape[:3])
            if len(inferred_shape) == 2:
                inferred_shape = (inferred_shape[0], inferred_shape[1], 1)
            frame_shape = inferred_shape

        return context.model_copy(update={"frame_shape": frame_shape})

    def _build_image(self, frame: Any, context: NodeContext) -> Image:
        if isinstance(frame, Image):
            return frame

        height = int(context.frame_shape[0]) if len(context.frame_shape) > 0 else 0
        width = int(context.frame_shape[1]) if len(context.frame_shape) > 1 else 0
        channels = int(context.frame_shape[2]) if len(context.frame_shape) > 2 else 1

        if (height <= 0 or width <= 0) and hasattr(frame, "shape") and len(frame.shape) >= 2:
            height = int(frame.shape[0])
            width = int(frame.shape[1])
            channels = int(frame.shape[2]) if len(frame.shape) > 2 else 1

        return TypeRegistry.validate(
            INPUT_PORTS["image"],
            Image(data=frame, width=width, height=height, channels=channels),
        )

    def _build_frame_info(self, context: NodeContext) -> FrameInfo:
        return TypeRegistry.validate(
            INPUT_PORTS["frame_info"],
            FrameInfo(
                frame_idx=context.frame_idx,
                timestamp=context.timestamp,
                camera_id=context.camera_id,
                frame_shape=context.frame_shape,
                scene_change_score=context.scene_change_score,
            ),
        )

    def run(self, frame: Any, context: NodeContext | None = None) -> dict[str, Any]:
        base_context = self._build_context(frame, context)
        port_values: dict[str, Any] = {
            "input.image": self._build_image(frame, base_context),
            "input.frame_info": self._build_frame_info(base_context),
        }

        for node_id in self.dag.execution_order():
            node = self.nodes[node_id]
            node_config = self.dag.nodes[node_id]
            inputs: dict[str, Any] = {}
            missing_input = False

            for input_name in node.get_input_ports():
                producer_id, producer_port = parse_port_ref(node_config.inputs[input_name])
                ref = f"{producer_id}.{producer_port}"
                if ref not in port_values:
                    missing_input = True
                    break
                inputs[input_name] = port_values[ref]

            if missing_input:
                continue

            frame_context = base_context.model_copy(update={"port_values": dict(port_values)})
            validated_inputs = node.validate_inputs(inputs)
            outputs = node.process(validated_inputs, frame_context)
            validated_outputs = node.validate_outputs(outputs)

            for port_name, value in validated_outputs.items():
                port_values[f"{node_id}.{port_name}"] = value

        if self.dag.config.outputs:
            return {
                alias: port_values[binding]
                for alias, binding in self.dag.config.outputs.items()
                if binding in port_values
            }

        terminal_outputs: dict[str, Any] = {}
        for node_id in self.dag.terminal_nodes():
            for port_name in self.nodes[node_id].get_output_ports():
                ref = f"{node_id}.{port_name}"
                if ref in port_values:
                    terminal_outputs[ref] = port_values[ref]
        return terminal_outputs

    def run_batch(
        self,
        frames: list[Any],
        contexts: list[NodeContext] | None = None,
    ) -> list[dict[str, Any]]:
        if contexts is None:
            contexts = [NodeContext(frame_idx=index) for index in range(len(frames))]
        return [self.run(frame, ctx) for frame, ctx in zip(frames, contexts)]
