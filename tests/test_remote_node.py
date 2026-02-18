"""
Stage 5: RemoteNode Tests
===========================

Tests use mocked HTTP requests — no real server needed.

Run: pytest tests/test_remote_node.py -v
"""
import base64
import json

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from vision_tools.core.node import NodeContext, NodeState
from vision_tools.nodes.remote_node import RemoteNode


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def remote_node():
    return RemoteNode(
        node_id="remote_det",
        config={"endpoint": "http://gpu-server:8000/detect", "timeout": 5.0}
    )


@pytest.fixture
def mock_metadata():
    return {
        "description": "Remote object detector",
        "output_schema": {"type": "object"},
    }


@pytest.fixture
def mock_process_result():
    return {
        "boxes": [{"xyxy": [0, 0, 100, 100], "class_id": 0,
                   "confidence": 0.9, "class_name": "person"}]
    }


# ===================================================================
# Tests
# ===================================================================

class TestRemoteNodeLifecycle:
    """Test RemoteNode connection lifecycle."""

    def test_initial_state(self, remote_node):
        assert remote_node.state == NodeState.UNLOADED

    @patch("vision_tools.nodes.remote_node.requests")
    def test_load_success(self, mock_requests, remote_node, mock_metadata):
        mock_response = MagicMock()
        mock_response.json.return_value = mock_metadata
        mock_response.raise_for_status.return_value = None
        mock_requests.get.return_value = mock_response

        remote_node.load()
        assert remote_node.state == NodeState.READY
        assert remote_node.remote_metadata == mock_metadata

    @patch("vision_tools.nodes.remote_node.requests")
    def test_load_failure(self, mock_requests, remote_node):
        mock_requests.get.side_effect = ConnectionError("Connection refused")

        with pytest.raises(ConnectionError):
            remote_node.load()
        assert remote_node.state == NodeState.FAILED

    def test_load_no_endpoint(self):
        node = RemoteNode("empty", config={})
        with pytest.raises(ValueError, match="no endpoint"):
            node.load()

    def test_unload(self, remote_node):
        remote_node._state = NodeState.READY
        remote_node.unload()
        assert remote_node.state == NodeState.UNLOADED


class TestRemoteNodeProcessing:
    """Test RemoteNode data processing."""

    @patch("vision_tools.nodes.remote_node.requests")
    def test_process_numpy(self, mock_requests, remote_node, mock_metadata, mock_process_result):
        # Setup: load first
        meta_resp = MagicMock()
        meta_resp.json.return_value = mock_metadata
        meta_resp.raise_for_status.return_value = None

        proc_resp = MagicMock()
        proc_resp.json.return_value = mock_process_result
        proc_resp.raise_for_status.return_value = None

        mock_requests.get.return_value = meta_resp
        mock_requests.post.return_value = proc_resp

        remote_node.load()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = remote_node.process(frame, NodeContext())

        assert result == mock_process_result
        # Verify POST was called
        mock_requests.post.assert_called_once()

    @patch("vision_tools.nodes.remote_node.requests")
    def test_process_dict(self, mock_requests, remote_node, mock_metadata, mock_process_result):
        meta_resp = MagicMock()
        meta_resp.json.return_value = mock_metadata
        meta_resp.raise_for_status.return_value = None

        proc_resp = MagicMock()
        proc_resp.json.return_value = mock_process_result
        proc_resp.raise_for_status.return_value = None

        mock_requests.get.return_value = meta_resp
        mock_requests.post.return_value = proc_resp

        remote_node.load()
        result = remote_node.process({"key": "value"}, NodeContext())
        assert result == mock_process_result

    def test_process_not_loaded(self, remote_node):
        with pytest.raises(RuntimeError, match="not connected"):
            remote_node.process(np.zeros((10, 10, 3)), NodeContext())


class TestRemoteNodeSerialization:
    """Test data serialization for transport."""

    def test_serialize_numpy(self):
        frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        payload = RemoteNode._serialize_input(frame, NodeContext())

        assert payload["data"]["type"] == "ndarray"
        assert payload["data"]["shape"] == [2, 2]
        assert payload["data"]["dtype"] == "uint8"
        # Verify base64 roundtrip
        decoded = np.frombuffer(
            base64.b64decode(payload["data"]["bytes"]),
            dtype=np.uint8
        ).reshape(2, 2)
        np.testing.assert_array_equal(decoded, frame)

    def test_serialize_dict(self):
        data = {"boxes": [{"x": 1}]}
        payload = RemoteNode._serialize_input(data, NodeContext())
        assert payload["data"] == data

    def test_context_included(self):
        ctx = NodeContext(frame_idx=42)
        payload = RemoteNode._serialize_input("data", ctx)
        assert payload["context"]["frame_idx"] == 42


class TestRemoteNodeConfig:
    """Test RemoteNode configuration."""

    def test_config_schema(self):
        schema = RemoteNode.get_config_schema().model_json_schema()
        assert "endpoint" in schema["properties"]
        assert "timeout" in schema["properties"]

    def test_registry_registered(self):
        # Ensure module is imported to trigger registration
        from vision_tools.nodes.remote_node import RemoteNode
        from vision_tools.core.registry import NodeRegistry
        assert NodeRegistry.get("remote") is RemoteNode

    def test_registry_recovers_after_clear(self):
        """Built-in bootstrap should restore remote node after registry clear."""
        from vision_tools.nodes.remote_node import RemoteNode
        from vision_tools.core.registry import NodeRegistry

        NodeRegistry.clear()
        assert NodeRegistry.get("remote") is RemoteNode
