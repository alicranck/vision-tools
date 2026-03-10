from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from vision_tools.core.type_refs import (
    PortTypeRef,
    TypeRegistry,
    parse_type_ref,
    serialize_type_ref,
)

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    UNLOADED = "unloaded"
    READY = "ready"
    FAILED = "failed"
    TRAINING = "training"


class NodeContext(BaseModel):
    frame_idx: int = 0
    timestamp: float = 0.0
    frame_shape: tuple[int, int, int] = (0, 0, 3)
    scene_change_score: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_id: str = "default"
    port_values: dict[str, Any] = Field(default_factory=dict)


class Node(ABC):
    InputSchema: ClassVar[Optional[Type[BaseModel]]] = None
    OutputSchema: ClassVar[Optional[Type[BaseModel]]] = None

    InputPorts: ClassVar[dict[str, str | PortTypeRef]] = {}
    OutputPorts: ClassVar[dict[str, str | PortTypeRef]] = {}
    DynamicPorts: ClassVar[bool] = False

    def __init__(self, node_id: str, config: dict[str, Any] | None = None) -> None:
        self.node_id = node_id
        self.config = config or {}
        self._state = NodeState.READY

    @abstractmethod
    def process(self, inputs: Any, context: NodeContext) -> dict[str, Any]:
        ...

    @property
    def state(self) -> NodeState:
        return self._state

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        return {
            name: parse_type_ref(type_ref)
            for name, type_ref in self.InputPorts.items()
        }

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        return {
            name: parse_type_ref(type_ref)
            for name, type_ref in self.OutputPorts.items()
        }

    def validate_inputs(self, inputs: dict[str, Any]) -> dict[str, BaseModel]:
        expected_ports = self.get_input_ports()
        if set(inputs.keys()) != set(expected_ports.keys()):
            raise ValueError(
                f"{self.node_id}: expected inputs {sorted(expected_ports.keys())}, "
                f"got {sorted(inputs.keys())}"
            )

        return {
            name: TypeRegistry.validate(type_ref, inputs[name])
            for name, type_ref in expected_ports.items()
        }

    def validate_outputs(self, outputs: dict[str, Any]) -> dict[str, BaseModel]:
        expected_ports = self.get_output_ports()
        unknown = set(outputs.keys()) - set(expected_ports.keys())
        if unknown:
            raise ValueError(
                f"{self.node_id}: produced undeclared output ports {sorted(unknown)}"
            )

        return {
            name: TypeRegistry.validate(expected_ports[name], value)
            for name, value in outputs.items()
        }

    @classmethod
    def get_config_schema(cls) -> Optional[Type[BaseModel]]:
        return None

    @classmethod
    def get_config_options(cls) -> Optional[dict[str, Any]]:
        return None

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {
            "description": cls.__doc__ or "",
            "input_ports": {
                name: serialize_type_ref(type_ref)
                for name, type_ref in cls.InputPorts.items()
            },
            "output_ports": {
                name: serialize_type_ref(type_ref)
                for name, type_ref in cls.OutputPorts.items()
            },
            "dynamic_ports": cls.DynamicPorts,
            "config_schema": (
                cls.get_config_schema().model_json_schema()
                if cls.get_config_schema()
                else None
            ),
            "config_options": cls.get_config_options(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(node_id={self.node_id!r}, "
            f"state={self._state.value})>"
        )
