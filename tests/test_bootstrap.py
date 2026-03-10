import logging

import vision_tools.nodes as nodes_pkg
from vision_tools.backends.registry import BackendRegistry
from vision_tools.capabilities import list_capabilities
from vision_tools.core.registry import NodeRegistry


def setup_function():
    NodeRegistry.clear()
    BackendRegistry.clear()


def test_register_all_keeps_builtin_nodes_when_optional_backend_import_fails(monkeypatch, caplog):
    real_import_module = nodes_pkg.import_module

    def flaky_import(name):
        if name == "vision_tools.backends.captioning.smolvlm":
            raise ImportError("missing optional dependency")
        return real_import_module(name)

    monkeypatch.setattr(nodes_pkg, "import_module", flaky_import)

    with caplog.at_level(logging.INFO):
        nodes_pkg.register_all()

    names = {entry["type"] for entry in NodeRegistry.list_nodes()}
    assert "object_detector" in names
    assert "dynamic_logic" in names
    assert "buffer" in names
    assert "smolvlm" not in {
        f"{entry['task']}:{entry['model']}" for entry in BackendRegistry.list_backends()
    }
    assert "Skipping optional backend registration for captioning:smolvlm" in caplog.text


def test_capabilities_expose_sources():
    nodes_pkg.register_all()
    capabilities = list_capabilities()
    assert "sources" in capabilities
    assert capabilities["sources"][0]["type"] == "input"
    assert capabilities["sources"][0]["output_ports"]["frame_info"] == "FrameInfo"
