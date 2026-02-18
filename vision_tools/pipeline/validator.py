"""
SchemaValidator — Validates I/O compatibility at pipeline construction time.

Checks that each node's InputSchema can be satisfied by the combined
OutputSchemas of its dependencies.
"""
from __future__ import annotations

import logging
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

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
        errors: list[str] = []

        for node_id, node in nodes.items():
            input_schema = node.InputSchema
            if input_schema is None:
                continue  # Node accepts raw input, no validation needed

            # Collect output fields from upstream dependencies
            deps = dag.dependencies(node_id)
            if not deps:
                continue  # Root node, takes pipeline input

            upstream_fields: dict[str, list[Any]] = {}
            for dep_id in deps:
                dep_node = nodes.get(dep_id)
                if dep_node is None:
                    continue
                output_schema = dep_node.OutputSchema
                if output_schema is None:
                    errors.append(
                        f"Node '{dep_id}' has no OutputSchema; "
                        f"cannot validate input for '{node_id}'"
                    )
                    continue
                for field_name, field_info in output_schema.model_fields.items():
                    upstream_fields.setdefault(field_name, []).append(field_info.annotation)

            # Check required input fields are present and type-compatible
            for field_name, field_info in input_schema.model_fields.items():
                if not field_info.is_required():
                    continue

                if field_name not in upstream_fields:
                    errors.append(
                        f"Node '{node_id}' requires field '{field_name}' "
                        f"not found in upstream outputs {set(upstream_fields.keys())}"
                    )
                    continue

                expected = field_info.annotation
                provided = upstream_fields[field_name]
                if not any(
                    SchemaValidator._is_type_compatible(src, expected)
                    for src in provided
                ):
                    errors.append(
                        f"Node '{node_id}' field '{field_name}' type mismatch: "
                        f"expected {expected!r}, upstream provides {provided!r}"
                    )

        if errors:
            for err in errors:
                logger.error(f"SchemaValidator: {err}")
            raise SchemaValidationError(
                "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return []

    @staticmethod
    def _is_type_compatible(source: Any, target: Any) -> bool:
        """Best-effort type compatibility check for schema field annotations."""
        if target is Any or source is Any:
            return True
        if source == target:
            return True

        source_origin = get_origin(source)
        target_origin = get_origin(target)
        source_args = get_args(source)
        target_args = get_args(target)
        union_origins = (Union, UnionType)

        # Union handling (including Optional[T]).
        if target_origin in union_origins:
            return any(
                SchemaValidator._is_type_compatible(source, t_arg)
                for t_arg in target_args
            )
        if source_origin in union_origins:
            return all(
                SchemaValidator._is_type_compatible(s_arg, target)
                for s_arg in source_args
            )

        if source_origin is None and target_origin is None:
            try:
                return issubclass(source, target)
            except Exception:
                return False

        if source is NoneType or target is NoneType:
            return source is target

        if source_origin in (tuple, list, dict, set) and target_origin in (tuple, list, dict, set):
            if source_origin != target_origin:
                return False
            if not target_args:
                return True
            if len(source_args) != len(target_args):
                return False
            return all(
                SchemaValidator._is_type_compatible(s_arg, t_arg)
                for s_arg, t_arg in zip(source_args, target_args)
            )

        # Matching generic origins with arguments (e.g., list[int] -> list[float]).
        if source_origin is not None and target_origin is not None:
            if source_origin != target_origin:
                return False
            if not target_args:
                return True
            if len(source_args) != len(target_args):
                return False
            return all(
                SchemaValidator._is_type_compatible(s_arg, t_arg)
                for s_arg, t_arg in zip(source_args, target_args)
            )

        return False
