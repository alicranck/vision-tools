from __future__ import annotations


def parse_port_ref(ref: str) -> tuple[str, str]:
    if "." not in ref:
        raise ValueError(
            f"Invalid port ref '{ref}'. Expected '<node_id>.<port_name>'."
        )

    node_id, port_name = ref.split(".", 1)
    node_id = node_id.strip()
    port_name = port_name.strip()

    if not node_id or not port_name:
        raise ValueError(
            f"Invalid port ref '{ref}'. Expected '<node_id>.<port_name>'."
        )

    return node_id, port_name
