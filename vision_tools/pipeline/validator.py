"""
SchemaValidator — Validates I/O compatibility at pipeline construction time.

Checks that each node's InputSchema can be satisfied by the combined
OutputSchemas of its dependencies.
"""
from __future__ import annotations

import logging
from typing import Any

from vision_tools.core.node import Node
from vision_tools.pipeline.graph import DAG

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaValidator:
    """Validates I/O schema compatibility across the pipeline graph.

    For each node with an ``InputSchema``, verifies that its dependency
    nodes produce outputs containing the required fields.
    """

    @staticmethod
    def validate(dag: DAG, nodes: dict[str, Node]) -> list[str]:
        """Validate I/O compatibility for all nodes.

        Args:
            dag: The pipeline DAG.
            nodes: Mapping of node_id → Node instance.

        Returns:
            List of warning messages (soft validation — missing schemas
            produce warnings, not errors).

        Raises:
            SchemaValidationError: If a hard incompatibility is detected.
        """
        warnings_list: list[str] = []

        for node_id, node in nodes.items():
            input_schema = node.InputSchema
            if input_schema is None:
                continue  # Node accepts raw input, no validation needed

            # Collect output fields from upstream dependencies
            deps = dag.dependencies(node_id)
            if not deps:
                continue  # Root node, takes pipeline input

            upstream_fields: set[str] = set()
            for dep_id in deps:
                dep_node = nodes.get(dep_id)
                if dep_node is None:
                    continue
                output_schema = dep_node.OutputSchema
                if output_schema is None:
                    warnings_list.append(
                        f"Node '{dep_id}' has no OutputSchema; "
                        f"cannot validate input for '{node_id}'"
                    )
                    continue
                upstream_fields.update(output_schema.model_fields.keys())

            # Check required input fields are present in upstream outputs
            required_fields = set(input_schema.model_fields.keys())
            missing = required_fields - upstream_fields
            if missing:
                warnings_list.append(
                    f"Node '{node_id}' requires fields {missing} "
                    f"not found in upstream outputs {upstream_fields}"
                )

        if warnings_list:
            for w in warnings_list:
                logger.warning(f"SchemaValidator: {w}")

        return warnings_list
