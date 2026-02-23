import pytest
from vision_tools.nodes.aggregator_node import AggregatorNode
from vision_tools.utils.schemas import FrameResult, FrameMetadata

def test_aggregator_buffers_frames():
    node = AggregatorNode(node_id="agg1", config={"window_size_seconds": 2.0, "stride_frames": 2})
    
    # Send 1 frame, shouldn't emit yet
    f1 = FrameResult(metadata=FrameMetadata(frame_idx=0, timestamp=0.0, source_id="test"), results={"boxes": []})
    out1 = node.process(f1, None)
    assert out1 is None
    
    # Send 2nd frame, should emit (stride=2)
    f2 = FrameResult(metadata=FrameMetadata(frame_idx=1, timestamp=1.0, source_id="test"), results={"boxes": []})
    out2 = node.process(f2, None)
    
    assert out2 is not None
    assert len(out2["history"]) == 2
