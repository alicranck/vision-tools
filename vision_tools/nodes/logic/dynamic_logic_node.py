from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vision_tools.core.node import NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.type_refs import PortTypeRef, TypeRegistry, parse_type_ref
from vision_tools.nodes.logic.logic_node import LogicNode


class DynamicLogicConfig(BaseModel):
    input_ports: dict[str, str] = Field(default_factory=dict)
    output_ports: dict[str, str] = Field(default_factory=dict)
    code: str
    params: dict[str, Any] = Field(default_factory=dict)


@NodeRegistry.register("dynamic_logic", category="logic")
class DynamicLogicNode(LogicNode):
    DynamicPorts = True

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = DynamicLogicConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._input_ports = {
            name: parse_type_ref(type_ref)
            for name, type_ref in validated.input_ports.items()
        }
        self._output_ports = {
            name: parse_type_ref(type_ref)
            for name, type_ref in validated.output_ports.items()
        }
        self.params = dict(validated.params)
        self._state_store: dict[str, Any] = {}
        self._namespace: dict[str, Any] = {}
        exec(validated.code, self._namespace)
        if "execute" not in self._namespace:
            raise ValueError("Provided code must define an 'execute(inputs, context, config, state)' function.")
        self._user_execute = self._namespace["execute"]

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        return dict(self._input_ports)

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        return dict(self._output_ports)

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        result = self._user_execute(inputs, context, self.params, self._state_store)
        if not isinstance(result, dict):
            raise TypeError(f"{self.node_id}: dynamic logic must return a dict of outputs.")
        return result

    def validate_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        validated = super().validate_outputs(outputs)
        expected = set(self.get_output_ports().keys())
        if set(validated.keys()) != expected:
            raise ValueError(
                f"{self.node_id}: expected outputs {sorted(expected)}, got {sorted(validated.keys())}"
            )
        return validated

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return DynamicLogicConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"accepted_type_names": TypeRegistry.list_type_names()}
