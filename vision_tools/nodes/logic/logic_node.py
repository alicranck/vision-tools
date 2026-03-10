from __future__ import annotations

from abc import abstractmethod
from typing import Any

from vision_tools.core.node import Node, NodeContext, NodeState


class LogicNode(Node):
    def __init__(self, node_id: str, config: dict | None = None) -> None:
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, inputs: Any, context: NodeContext) -> dict[str, Any]:
        if not isinstance(inputs, dict):
            if len(self.get_input_ports()) != 1:
                raise TypeError(f"{self.node_id}: expected a dict of named inputs.")
            inputs = {next(iter(self.get_input_ports())): inputs}

        outputs = self.execute(inputs, context)
        return self.validate_outputs(outputs)

    @abstractmethod
    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        ...
