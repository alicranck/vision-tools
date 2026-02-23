from typing import Any
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.state_manager import StateManager
from vision_tools.utils.schemas import FrameResult

class AggregatorNode(Node):
    def __init__(self, node_id: str, config: dict | None = None) -> None:
        super().__init__(node_id, config or {})
        self.window_size_seconds = self.config.get("window_size_seconds", 5.0)
        self.stride_frames = self.config.get("stride_frames", 30) # Emit every N frames
        
        # Estimate history size based on 30fps
        hist_size = int(self.window_size_seconds * 30)
        self.state_manager = StateManager(history_size=hist_size)
        self.frame_count = 0
        self._state = NodeState.READY

    def process(self, data: FrameResult, context: NodeContext) -> Any:
        self.state_manager.ingest([data])
        self.frame_count += 1
        
        if self.frame_count % self.stride_frames == 0:
            # Emit the time window result
            return {"history": list(self.state_manager._history)}
        
        return None
