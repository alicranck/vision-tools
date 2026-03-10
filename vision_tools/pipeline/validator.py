from __future__ import annotations

from vision_tools.core.node import Node
from vision_tools.core.sources import SOURCE_NODE_ID, get_source_ports
from vision_tools.core.type_refs import TypeRegistry, serialize_type_ref
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.refs import parse_port_ref


class SchemaValidationError(Exception):
    pass


SOURCE_PORTS = get_source_ports()


class SchemaValidator:
    @staticmethod
    def validate(dag: DAG, nodes: dict[str, Node]) -> list[str]:
        errors: list[str] = []

        for node_id, node_config in dag.nodes.items():
            if node_id == "input":
                errors.append("The node id 'input' is reserved.")
                continue

            if node_id not in nodes:
                errors.append(f"Node '{node_id}' was not instantiated.")
                continue

            node = nodes[node_id]
            expected_inputs = node.get_input_ports()
            bound_inputs = node_config.inputs

            if set(bound_inputs.keys()) != set(expected_inputs.keys()):
                errors.append(
                    f"Node '{node_id}' input bindings {sorted(bound_inputs.keys())} "
                    f"do not match declared inputs {sorted(expected_inputs.keys())}"
                )

            for input_name, target_type in expected_inputs.items():
                binding = bound_inputs.get(input_name)
                if binding is None:
                    continue

                try:
                    producer_id, producer_port = parse_port_ref(binding)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue

                producer_ports = SOURCE_PORTS if producer_id == SOURCE_NODE_ID else None
                if producer_id != SOURCE_NODE_ID:
                    producer = nodes.get(producer_id)
                    if producer is None:
                        errors.append(
                            f"Node '{node_id}' references unknown producer '{producer_id}'."
                        )
                        continue
                    producer_ports = producer.get_output_ports()

                assert producer_ports is not None
                if producer_port not in producer_ports:
                    errors.append(
                        f"Node '{node_id}' references unknown port '{binding}'."
                    )
                    continue

                source_type = producer_ports[producer_port]
                if not TypeRegistry.is_assignable(source_type, target_type):
                    errors.append(
                        f"Node '{node_id}' input '{input_name}' expects "
                        f"{serialize_type_ref(target_type)}, got "
                        f"{serialize_type_ref(source_type)} from '{binding}'."
                    )

            if hasattr(node, "validate_config"):
                try:
                    node.validate_config(dag, nodes)
                except Exception as exc:
                    errors.append(f"Node '{node_id}' config invalid: {exc}")

        for output_name, binding in dag.config.outputs.items():
            try:
                producer_id, producer_port = parse_port_ref(binding)
            except ValueError as exc:
                errors.append(f"Output '{output_name}' invalid: {exc}")
                continue

            if producer_id == SOURCE_NODE_ID:
                if producer_port not in SOURCE_PORTS:
                    errors.append(
                        f"Output '{output_name}' references unknown input port '{binding}'."
                    )
                continue

            if producer_id not in nodes:
                errors.append(
                    f"Output '{output_name}' references unknown node '{producer_id}'."
                )
                continue

            if producer_port not in nodes[producer_id].get_output_ports():
                errors.append(
                    f"Output '{output_name}' references unknown port '{binding}'."
                )

        try:
            dag.topological_sort()
        except Exception as exc:
            errors.append(str(exc))

        if errors:
            raise SchemaValidationError(
                "Schema validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            )

        return []
