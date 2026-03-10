from __future__ import annotations

from collections import deque

from pydantic import BaseModel, Field

from vision_tools.core.graph_types import History
from vision_tools.core.node import NodeContext
from vision_tools.core.sources import SOURCE_NODE_ID
from vision_tools.core.type_refs import GenericTypeRef, PortTypeRef, parse_type_ref
from vision_tools.nodes.logic.logic_node import LogicNode


class BufferNodeConfig(BaseModel):
    item_type: str
    window_size_seconds: float = Field(..., gt=0.0)
    emit_every_frames: int = Field(1, ge=1)
    emit_on_empty: bool = True


class BufferNode(LogicNode):
    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = BufferNodeConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated
        self._item_type = parse_type_ref(validated.item_type)
        self._history: deque[tuple[float, object]] = deque()
        self._frames_seen = 0

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        return {"items": self._item_type}

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        return {"history": GenericTypeRef("History", self._item_type)}

    def validate_config(self, dag, nodes) -> None:
        binding = dag.nodes[self.node_id].inputs.get("items")
        if not binding:
            return
        producer_id, producer_port = binding.split(".", 1)
        if producer_id == SOURCE_NODE_ID:
            raise ValueError("BufferNode cannot buffer input ports in phase 1.")
        producer = nodes[producer_id]
        source_type = producer.get_output_ports()[producer_port]
        if source_type != self._item_type:
            raise ValueError(
                f"item_type={self._item_type} does not match upstream type {source_type}"
            )

    def execute(self, inputs, context: NodeContext):
        self._frames_seen += 1
        self._history.append((context.timestamp, inputs["items"]))

        cutoff = context.timestamp - self._config.window_size_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if self._frames_seen % self._config.emit_every_frames != 0:
            return {}

        if not self._history and not self._config.emit_on_empty:
            return {}

        items = [item for _, item in self._history]
        window_start = self._history[0][0] if self._history else None
        window_end = self._history[-1][0] if self._history else None
        return {
            "history": History(
                items=items,
                window_start=window_start,
                window_end=window_end,
                emitted_at_frame=context.frame_idx,
            )
        }

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return BufferNodeConfig
