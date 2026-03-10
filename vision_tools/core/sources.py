from __future__ import annotations

from typing import Any

from vision_tools.core.type_refs import SimpleTypeRef, serialize_type_ref

SOURCE_NODE_ID = "input"
SOURCE_DESCRIPTION = "Built-in frame source seeded by the executor for each frame."
SOURCE_PORTS = {
    "image": SimpleTypeRef("Image"),
    "frame_info": SimpleTypeRef("FrameInfo"),
}


def get_source_ports() -> dict[str, SimpleTypeRef]:
    return dict(SOURCE_PORTS)


def list_sources() -> list[dict[str, Any]]:
    return [
        {
            "type": SOURCE_NODE_ID,
            "category": "io",
            "summary": "Built-in frame source",
            "description": SOURCE_DESCRIPTION,
            "output_ports": {
                name: serialize_type_ref(type_ref)
                for name, type_ref in SOURCE_PORTS.items()
            },
            "executable": False,
        }
    ]
