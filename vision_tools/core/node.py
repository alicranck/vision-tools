"""
Node — The fundamental abstraction in vision-tools v2.

Every participant in a VisionPipeline is a Node. Nodes declare typed
I/O contracts via Pydantic schemas. The pipeline validates schema
compatibility at construction time.

Node types:
    - ModelNode   — wraps an ML model (detection, embedding, etc.)
    - LogicNode   — custom code / LLM-generated logic
    - RemoteNode  — REST proxy to a node on another machine
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------

class NodeState(str, Enum):
    """Lifecycle state of a pipeline node."""
    UNLOADED = "unloaded"    # Not yet loaded (ModelNode default)
    READY = "ready"          # Ready to process
    FAILED = "failed"        # Failed to load or verify
    TRAINING = "training"    # Currently being fine-tuned


# ---------------------------------------------------------------------------
# Node context (runtime data passed through the pipeline)
# ---------------------------------------------------------------------------

class NodeContext(BaseModel):
    """Runtime context passed to every node during processing.

    Carries frame-level metadata and upstream results so nodes can
    access outputs from their dependency nodes.
    """
    frame_idx: int = 0
    timestamp: float = 0.0
    frame_shape: tuple[int, ...] = (0, 0, 3)
    scene_change_score: float = Field(0.0, ge=0.0, le=1.0)
    camera_id: str = "default"
    upstream_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="node_id → output from dependency nodes",
    )


# ---------------------------------------------------------------------------
# Node ABC
# ---------------------------------------------------------------------------

class Node(ABC):
    """Base interface for all pipeline participants.

    Subclasses must:
    1. Set ``InputSchema`` / ``OutputSchema`` class attributes as Pydantic models.
    2. Implement ``process(data, context) -> Any``.

    Everything else (lifecycle, model loading, caching) lives in
    specialized subclasses like ``ModelNode``, not here.
    """

    # Typed I/O contracts — override in subclasses
    InputSchema: ClassVar[Optional[Type[BaseModel]]] = None
    OutputSchema: ClassVar[Optional[Type[BaseModel]]] = None

    def __init__(self, node_id: str, config: dict[str, Any] | None = None) -> None:
        self.node_id = node_id
        self.config = config or {}
        self._state = NodeState.READY  # non-model nodes are immediately ready

    # --- Core interface ---

    @abstractmethod
    def process(self, data: Any, context: NodeContext) -> Any:
        """Process input data and return output.

        Args:
            data: Input data matching ``InputSchema`` (or raw frame/dict).
            context: Runtime context with frame metadata and upstream results.

        Returns:
            Output matching ``OutputSchema``.
        """
        ...

    # --- State ---

    @property
    def state(self) -> NodeState:
        """Current lifecycle state."""
        return self._state

    # --- Introspection (for LLM agents and registry) ---

    @classmethod
    def get_config_schema(cls) -> Optional[Type[BaseModel]]:
        """Return a Pydantic model describing valid config for this node type.

        Used by LLM agents to know what parameters are available.
        Override in subclasses that accept config.
        """
        return None

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        """Return introspection metadata for registry discovery.

        Returns a dict with:
            description, input_schema, output_schema, config_schema
        """
        return {
            "description": cls.__doc__ or "",
            "input_schema": (
                cls.InputSchema.model_json_schema()
                if cls.InputSchema else None
            ),
            "output_schema": (
                cls.OutputSchema.model_json_schema()
                if cls.OutputSchema else None
            ),
            "config_schema": (
                cls.get_config_schema().model_json_schema()
                if cls.get_config_schema() else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(node_id={self.node_id!r}, "
            f"state={self._state.value})>"
        )
