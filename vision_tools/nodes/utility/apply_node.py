from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vision_tools.core.graph_types import Sequence
from vision_tools.core.node import NodeContext, NodeState
from vision_tools.core.type_refs import PortTypeRef, parse_type_ref
from vision_tools.nodes.logic.logic_node import LogicNode


class ApplyConfig(BaseModel):
    node_type: str = Field(..., description="Node type to apply per image, e.g. 'captioner', 'embedder'")
    output_port: str = Field(..., description="Output port name on the wrapped node, e.g. 'caption', 'embedding'")
    output_type: str = Field(..., description="Graph type name for each result item, e.g. 'Caption', 'Embedding'")
    node_config: dict[str, Any] = Field(default_factory=dict, description="Config forwarded to the wrapped node")


class ApplyNode(LogicNode):
    """Apply a model node to each image in a Sequence[Image], producing a Sequence of results."""

    DynamicPorts = True

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = ApplyConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._apply_config = validated
        self._output_ref = parse_type_ref(f"Sequence[{validated.output_type}]")
        self._sub_node = None
        self._state = NodeState.UNLOADED

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        return {"images": parse_type_ref("Sequence[Image]")}

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        return {"results": self._output_ref}

    def load(self) -> None:
        if self._state == NodeState.READY:
            return
        from vision_tools.core.registry import NodeRegistry

        node_cls = NodeRegistry.get(self._apply_config.node_type)
        self._sub_node = node_cls(
            node_id=f"{self.node_id}_sub",
            config=self._apply_config.node_config,
        )
        self._sub_node.load()
        self._state = NodeState.READY

    def unload(self) -> None:
        if self._sub_node is not None:
            self._sub_node.unload()
            self._sub_node = None
        self._state = NodeState.UNLOADED

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        if self._sub_node is None:
            raise RuntimeError(f"{self.node_id}: not loaded. Call load() first.")
        sequence = inputs["images"]
        results = []
        for image in sequence.items:
            output = self._sub_node.process({"image": image}, context)
            results.append(output[self._apply_config.output_port])
        return {"results": Sequence(items=results)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return ApplyConfig
