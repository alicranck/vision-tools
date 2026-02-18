"""
LogicNode — Node for custom user/LLM-generated processing logic.

LogicNode provides a typed container where the ``execute()`` method
holds the actual processing logic. Input is validated against InputSchema,
then ``execute()`` runs, then output is validated against OutputSchema.

This is the primary extension point for LLM agents to inject custom
processing into pipelines without writing a full Backend.

Usage::

    class FilterSmallBoxes(LogicNode):
        InputSchema = DetectionResult
        OutputSchema = DetectionResult

        def execute(self, data, context):
            boxes = [b for b in data["boxes"] if b["confidence"] > 0.5]
            return {"boxes": boxes, "class_names": data.get("class_names", {})}
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry

logger = logging.getLogger(__name__)


class LogicNode(Node):
    """Base class for custom processing logic nodes.

    Subclasses must implement:
    - ``execute(data, context) -> Any`` — the core processing logic.

    Optionally set:
    - ``InputSchema`` — Pydantic model to validate input.
    - ``OutputSchema`` — Pydantic model to validate output.

    The ``process()`` method handles validation automatically:
    1. Validate input against ``InputSchema`` (if set).
    2. Call ``execute()``.
    3. Validate output against ``OutputSchema`` (if set).
    """

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        super().__init__(node_id, config or {})
        self._state = NodeState.READY  # Logic nodes are always ready

    def process(self, data: Any, context: NodeContext) -> Any:
        """Validate → execute → validate.

        Args:
            data: Input data (dict or raw).
            context: Runtime context.

        Returns:
            Processed output, validated against OutputSchema if set.
        """
        # Input validation
        if self.InputSchema is not None and isinstance(data, dict):
            self.InputSchema.model_validate(data)

        # Execute user logic
        result = self.execute(data, context)

        # Output validation
        if self.OutputSchema is not None and isinstance(result, dict):
            validated = self.OutputSchema.model_validate(result)
            return validated.model_dump()

        return result

    @abstractmethod
    def execute(self, data: Any, context: NodeContext) -> Any:
        """Override with custom processing logic.

        Args:
            data: Input data.
            context: Runtime context with upstream results.

        Returns:
            Processed data.
        """
        ...
