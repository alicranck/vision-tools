from typing import Any
from vision_tools.nodes.logic_node import LogicNode
from vision_tools.core.node import NodeContext

class DynamicLogicNode(LogicNode):
    """Executes arbitrary python code passed via config"""
    
    def __init__(self, node_id: str, config: dict | None = None) -> None:
        super().__init__(node_id, config)
        self.code_str = self.config.get("code", "def execute(data, context):\n    return []")
        
        # Compile the code block into a namespace
        self._namespace = {}
        exec(self.code_str, self._namespace)
        
        if "execute" not in self._namespace:
            raise ValueError("Provided code must define an 'execute(data, context)' function.")
            
        self._user_execute = self._namespace["execute"]

    def execute(self, data: Any, context: NodeContext) -> Any:
        # Run the dynamically loaded function
        return self._user_execute(data, context)
