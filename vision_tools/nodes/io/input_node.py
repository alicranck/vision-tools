from __future__ import annotations

from vision_tools.core.node import Node, NodeContext


class InputNode(Node):
    OutputPorts = {"image": "Image", "frame_info": "FrameInfo"}

    def process(self, inputs, context: NodeContext):
        raise RuntimeError("InputNode is seeded by the executor and should not be executed directly.")
