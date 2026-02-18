"""
RemoteNode — REST proxy node for distributed pipeline execution.

Wraps a remote vision-tools endpoint as a local node.
The remote endpoint exposes the standard Node interface via REST:
- ``POST /process`` — runs inference and returns results.
- ``GET /metadata`` — returns node metadata (schemas, etc.).

Usage::

    NodeConfig(node_id="remote_det", node_type="remote",
               config={"endpoint": "http://gpu-server:8000/detect"})
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import requests
from pydantic import BaseModel, Field

from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry

logger = logging.getLogger(__name__)


class RemoteNodeConfig(BaseModel):
    """Config schema for RemoteNode."""
    endpoint: str = Field(..., description="Base URL of the remote node endpoint")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")


@NodeRegistry.register("remote", category="infrastructure")
class RemoteNode(Node):
    """Proxy node that delegates processing to a remote endpoint.

    The remote server must expose:
    - ``POST /process`` accepting JSON with ``data`` and ``context`` fields.
    - ``GET /metadata`` returning node metadata.

    Data serialization:
    - numpy arrays are base64-encoded for transport.
    - Results are returned as dicts (matching OutputSchema if available).
    """

    OutputSchema = None  # Loaded dynamically from remote
    InputSchema = None

    def __init__(self, node_id: str = "remote", config: dict | None = None) -> None:
        super().__init__(node_id, config or {})
        self._endpoint = self.config.get("endpoint", "")
        self._timeout = self.config.get("timeout", 30.0)
        self._verify_ssl = self.config.get("verify_ssl", True)
        self._state = NodeState.UNLOADED
        self._remote_metadata: dict | None = None

    def load(self) -> None:
        """Connect to remote endpoint and fetch metadata."""
        if not self._endpoint:
            raise ValueError(f"{self.node_id}: no endpoint configured.")

        try:
            self._remote_metadata = self._fetch_metadata()
            self._state = NodeState.READY
            logger.info(
                f"{self.node_id}: connected to {self._endpoint}"
            )
        except Exception as e:
            self._state = NodeState.FAILED
            logger.error(f"{self.node_id}: failed to connect: {e}")
            raise

    def unload(self) -> None:
        """Disconnect from remote endpoint."""
        self._remote_metadata = None
        self._state = NodeState.UNLOADED

    def process(self, data: Any, context: NodeContext) -> Any:
        """Send data to remote endpoint and return results.

        Args:
            data: Input data (numpy array or dict).
            context: Runtime context.

        Returns:
            Dict with remote processing results.
        """
        if self._state != NodeState.READY:
            raise RuntimeError(
                f"{self.node_id}: not connected. Call load() first."
            )

        payload = self._serialize_input(data, context)

        try:
            response = requests.post(
                f"{self._endpoint}/process",
                json=payload,
                timeout=self._timeout,
                verify=self._verify_ssl,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"{self.node_id}: remote processing failed: {e}")
            raise

    def _fetch_metadata(self) -> dict:
        """Fetch metadata from remote endpoint."""

        response = requests.get(
            f"{self._endpoint}/metadata",
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _serialize_input(data: Any, context: NodeContext) -> dict:
        """Serialize input for JSON transport."""
        import base64

        payload: dict[str, Any] = {"context": context.model_dump()}

        if isinstance(data, np.ndarray):
            payload["data"] = {
                "type": "ndarray",
                "dtype": str(data.dtype),
                "shape": list(data.shape),
                "bytes": base64.b64encode(data.tobytes()).decode("ascii"),
            }
        elif isinstance(data, dict):
            payload["data"] = data
        else:
            payload["data"] = str(data)

        return payload

    @property
    def remote_metadata(self) -> dict | None:
        """Metadata fetched from the remote endpoint."""
        return self._remote_metadata

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return RemoteNodeConfig
