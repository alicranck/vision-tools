"""
Stage 5: LogicNode + RemoteNode Tests
=======================================

Run: pytest tests/test_logic_node.py tests/test_remote_node.py -v
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pydantic import BaseModel, ValidationError

from vision_tools.core.node import NodeContext, NodeState
from vision_tools.core.schemas import DetectionResult, BoundingBox
from vision_tools.nodes.logic_node import LogicNode


# ===================================================================
# Concrete LogicNode implementations for testing
# ===================================================================

class ConfidenceFilter(LogicNode):
    """Filter detections by confidence threshold."""
    InputSchema = DetectionResult
    OutputSchema = DetectionResult

    def execute(self, data, context):
        threshold = self.config.get("threshold", 0.5)
        if isinstance(data, dict):
            boxes = [b for b in data.get("boxes", []) if b["confidence"] >= threshold]
            return {"boxes": boxes, "class_names": data.get("class_names", {})}
        return data


class CounterNode(LogicNode):
    """Counts items — no schema validation."""
    InputSchema = None
    OutputSchema = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0

    def execute(self, data, context):
        self.call_count += 1
        return {"count": self.call_count, "input_type": type(data).__name__}


class BadOutputNode(LogicNode):
    """Produces output that doesn't match OutputSchema."""

    class StrictOut(BaseModel):
        required_field: str

    InputSchema = None
    OutputSchema = StrictOut

    def execute(self, data, context):
        return {"wrong_field": "bad"}


# ===================================================================
# Tests
# ===================================================================

class TestLogicNodeBasics:
    """Test LogicNode fundamentals."""

    def test_always_ready(self):
        node = CounterNode(node_id="counter")
        assert node.state == NodeState.READY

    def test_execute_called(self):
        node = CounterNode(node_id="counter")
        node.process("hello", NodeContext())
        assert node.call_count == 1

    def test_multiple_calls(self):
        node = CounterNode(node_id="counter")
        for _ in range(5):
            node.process("data", NodeContext())
        assert node.call_count == 5


class TestLogicNodeSchemaValidation:
    """Test input/output schema validation."""

    def test_valid_input_passes(self):
        node = ConfidenceFilter(
            node_id="filter",
            config={"threshold": 0.5}
        )
        input_data = DetectionResult(
            boxes=[
                BoundingBox(xyxy=[0, 0, 100, 100], class_id=0,
                           confidence=0.9, class_name="person"),
                BoundingBox(xyxy=[0, 0, 50, 50], class_id=1,
                           confidence=0.3, class_name="car"),
            ]
        ).model_dump()

        result = node.process(input_data, NodeContext())
        assert len(result["boxes"]) == 1
        assert result["boxes"][0]["class_name"] == "person"

    def test_output_schema_enforcement(self):
        node = BadOutputNode(node_id="bad")
        with pytest.raises(ValidationError):
            node.process("input", NodeContext())

    def test_no_schema_passthrough(self):
        node = CounterNode(node_id="counter")
        result = node.process({"anything": "goes"}, NodeContext())
        assert result["count"] == 1
        assert result["input_type"] == "dict"


class TestLogicNodeInPipeline:
    """Test LogicNode as part of a pipeline flow."""

    def test_confidence_filter(self):
        node = ConfidenceFilter(
            node_id="filter",
            config={"threshold": 0.8}
        )
        input_data = {
            "boxes": [
                {"xyxy": [0, 0, 100, 100], "class_id": 0,
                 "confidence": 0.95, "class_name": "person"},
                {"xyxy": [50, 50, 200, 200], "class_id": 1,
                 "confidence": 0.4, "class_name": "car"},
                {"xyxy": [10, 10, 90, 90], "class_id": 0,
                 "confidence": 0.85, "class_name": "person"},
            ],
            "class_names": {0: "person", 1: "car"}
        }
        result = node.process(input_data, NodeContext())
        assert len(result["boxes"]) == 2  # Only 0.95 and 0.85

    def test_config_driven(self):
        node = ConfidenceFilter(node_id="f", config={"threshold": 0.0})
        data = {
            "boxes": [
                {"xyxy": [0, 0, 1, 1], "class_id": 0,
                 "confidence": 0.01, "class_name": "x"}
            ],
            "class_names": {}
        }
        result = node.process(data, NodeContext())
        assert len(result["boxes"]) == 1  # Threshold 0.0, everything passes
